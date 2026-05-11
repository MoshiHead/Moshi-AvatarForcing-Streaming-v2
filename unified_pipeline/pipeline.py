"""
unified_pipeline/pipeline.py
=============================
Orchestrates:
  1. MoshiRunner       — runs Moshi on user audio, yields tokens + raw PCM per step
  2. BridgeRunner      — accumulates tokens → audio_emb (T+1, 10752)
  3. AvatarForcingRunner — generates talking-head video from image + audio_emb
  4. ffmpeg merge      — combines silent video + Moshi response audio → final .mp4

Latency benefit:
  The bridge converts Moshi's DISCRETE TOKENS directly — before Mimi decodes
  them to waveform. So AvatarForcing video generation begins as soon as
  enough tokens are buffered (teacher_len/2 Moshi steps ≈ 3.2 s at 12.5 Hz).
  Moshi audio is collected in parallel and merged afterwards via ffmpeg.
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torchaudio

from .moshi_runner import MoshiRunner
from .bridge_runner import BridgeRunner
from .avatarforcing_runner import AvatarForcingRunner


class UnifiedMoshiAvatarPipeline:
    """
    End-to-end pipeline:
        user_audio  +  reference_image  →  talking_head_video.mp4

    Parameters
    ----------
    bridge_checkpoint  : path to trained moshi-wav2vec-bridge .pt checkpoint
    generator_ckpt     : path to AvatarForcing generator checkpoint
    af_config          : OmegaConf config object (loaded from avatarforcing.yaml)
    bridge_config      : path to bridge config.yaml (auto-detected if None)
    moshi_hf_repo      : HuggingFace repo for Moshi weights
    moshi_weight       : optional local override for Moshi LM weights
    mimi_weight        : optional local override for Mimi codec weights
    moshi_tokenizer    : optional local SentencePiece tokenizer path
    teacher_len        : AvatarForcing teacher_len (must be even, default 80)
    num_output_frames  : number of video frames to generate (default 21)
    device             : 'cuda' or 'cpu'
    default_prompt     : default text conditioning for AvatarForcing
    """

    DEFAULT_PROMPT = (
        "A person talking naturally, realistic facial expressions, "
        "high quality video, detailed face."
    )

    def __init__(
        self,
        bridge_checkpoint:  str,
        generator_ckpt:     str,
        af_config,
        bridge_config:      Optional[str] = None,
        moshi_hf_repo:      str = "kyutai/moshiko-pytorch-q8",
        moshi_weight:       Optional[str] = None,
        mimi_weight:        Optional[str] = None,
        moshi_tokenizer:    Optional[str] = None,
        teacher_len:        int = 80,
        num_output_frames:  int = 21,
        device:             str = "cuda",
        default_prompt:     Optional[str] = None,
        dtype:              torch.dtype = torch.bfloat16,
        use_ema:            bool = False,
    ):
        self._device_str   = device
        self._device       = torch.device(device)
        self.teacher_len   = teacher_len
        self.num_output_frames = num_output_frames
        self.default_prompt = default_prompt or self.DEFAULT_PROMPT

        # ── Moshi ─────────────────────────────────────────────────────────────
        self.moshi = MoshiRunner(
            hf_repo      = moshi_hf_repo,
            moshi_weight = moshi_weight,
            mimi_weight  = mimi_weight,
            tokenizer    = moshi_tokenizer,
            device       = device,
            dtype        = dtype,
        )
        self.moshi.load()

        # ── Bridge ────────────────────────────────────────────────────────────
        self.bridge = BridgeRunner(
            checkpoint_path = bridge_checkpoint,
            config_path     = bridge_config,
            teacher_len     = teacher_len,
            device          = device,
        )

        # ── AvatarForcing ─────────────────────────────────────────────────────
        self.af = AvatarForcingRunner(
            af_config          = af_config,
            checkpoint_path    = generator_ckpt,
            device             = self._device,
            num_output_frames  = num_output_frames,
            use_ema            = use_ema,
            dtype              = dtype,
        )

        print("[UnifiedPipeline] ✅ All models loaded.")

    def run(
        self,
        user_audio_path: str,
        image_path:      str,
        output_path:     str = "output.mp4",
        prompt:          Optional[str] = None,
        fps:             int = 25,
    ) -> str:
        """
        Full end-to-end inference.

        Parameters
        ----------
        user_audio_path : path to user input audio (.wav, .mp3, etc.)
        image_path      : path to reference face image
        output_path     : where to save the final .mp4
        prompt          : text prompt (uses default if None)
        fps             : output video fps (must match AvatarForcing fps=25)

        Returns
        -------
        output_path : path to the written .mp4 file
        """
        prompt = prompt or self.default_prompt
        print(f"\n[UnifiedPipeline] ▶ user_audio={user_audio_path}")
        print(f"[UnifiedPipeline]   image={image_path}")
        print(f"[UnifiedPipeline]   output={output_path}")

        # ── 1. Load user audio at Moshi's rate (24 kHz) ──────────────────────
        in_pcm, native_sr = torchaudio.load(user_audio_path)
        if in_pcm.shape[0] > 1:
            in_pcm = in_pcm.mean(0, keepdim=True)       # → mono
        if native_sr != self.moshi.sample_rate:
            in_pcm = torchaudio.functional.resample(
                in_pcm, native_sr, self.moshi.sample_rate
            )
        in_pcm_3d = in_pcm.unsqueeze(0)                  # (1, 1, N)
        duration  = in_pcm.shape[-1] / self.moshi.sample_rate
        print(f"[UnifiedPipeline]   audio duration: {duration:.2f}s")

        # ── 2. Run Moshi step-by-step, accumulate PCM + tokens ───────────────
        self.bridge.reset()
        all_pcm_chunks:  list[torch.Tensor] = []
        all_tokens:      list[torch.Tensor] = []

        print("[UnifiedPipeline] Running Moshi …")
        for step in self.moshi.run(in_pcm_3d):
            all_pcm_chunks.append(step.pcm_chunk)         # (1, frame_size)
            all_tokens.append(step.audio_tokens.squeeze(0).cpu())  # (8,)

        n_steps = len(all_pcm_chunks)
        print(f"[UnifiedPipeline] Moshi done: {n_steps} steps")

        if not all_tokens:
            raise RuntimeError(f"No Moshi tokens generated — audio too short.")

        # ── 3. Generate Audio Embeddings for the entire sequence ──────────────
        tokens_batch = torch.stack(all_tokens, dim=0).unsqueeze(0).to(self.bridge.device) # (1, N, 8)
        audio_emb_for_video = self.bridge._run_bridge(tokens_batch, target_len=None)
        print(f"[UnifiedPipeline] audio_emb → AvatarForcing: {audio_emb_for_video.shape}")

        # ── 4. Generate silent talking-head video ─────────────────────────────
        import time
        print("[UnifiedPipeline] Running AvatarForcing diffusion …")
        t0 = time.time()
        frames_np = self.af.generate_to_numpy(
            image_path = image_path,
            audio_emb  = audio_emb_for_video,
            prompt     = prompt,
        )
        t1 = time.time()
        gen_time = t1 - t0
        num_frames = frames_np.shape[0]
        gen_fps = num_frames / gen_time if gen_time > 0 else 0
        video_dur = num_frames / fps
        
        print(f"[UnifiedPipeline] Video generated: {num_frames} frames "
              f"({video_dur:.2f}s at {fps} fps) in {gen_time:.2f}s "
              f"→ Generation Speed: {gen_fps:.2f} fps. "
              f"Resolution: {frames_np.shape[2]}x{frames_np.shape[1]}")

        # ── 5. Concatenate Moshi raw PCM ──────────────────────────────────────
        response_pcm = torch.cat(all_pcm_chunks, dim=1)   # (1, total_samples)

        # ── 6. Merge video + audio via ffmpeg ─────────────────────────────────
        print("[UnifiedPipeline] Merging video + audio …")
        output_path = _merge_video_audio(
            frames      = frames_np,
            pcm         = response_pcm,
            sample_rate = self.moshi.sample_rate,
            output_path = str(output_path),
            fps         = fps,
        )
        print(f"[UnifiedPipeline] ✅ Done → {output_path}")
        return output_path


# ─── ffmpeg merge ──────────────────────────────────────────────────────────────

def _merge_video_audio(
    frames:      np.ndarray,    # (N, H, W, 3) uint8
    pcm:         torch.Tensor,  # (1, total_samples) float32
    sample_rate: int,
    output_path: str,
    fps:         int = 25,
) -> str:
    """
    Write frames as silent .mp4, write PCM as .wav, then use ffmpeg to
    mux them together. Returns output_path.
    """
    import cv2

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp = Path(tmp_dir)
        silent_mp4 = str(tmp / "silent.mp4")
        audio_wav  = str(tmp / "audio.wav")

        # ── Write silent video ────────────────────────────────────────────────
        H, W = frames.shape[1], frames.shape[2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        vw = cv2.VideoWriter(silent_mp4, fourcc, fps, (W, H))
        for frame in frames:
            vw.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        vw.release()

        # ── Write Moshi response audio ────────────────────────────────────────
        torchaudio.save(audio_wav, pcm.float().cpu(), sample_rate)

        # ── ffmpeg mux (copy video stream, encode audio to AAC) ───────────────
        cmd = [
            "ffmpeg", "-y",
            "-i", silent_mp4,
            "-i", audio_wav,
            "-c:v", "libx264",          # re-encode to H.264 for compatibility
            "-c:a", "aac",
            "-b:a", "192k",
            "-shortest",                # trim to shorter of video/audio
            "-movflags", "+faststart",  # web-friendly mp4
            output_path,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"[ffmpeg] stderr:\n{result.stderr}")
            raise RuntimeError(f"ffmpeg failed (code {result.returncode})")

    return output_path
