"""
streaming_server/pipeline/streaming_avatarforcing.py
=====================================================
Streaming AvatarForcing adapter using the self-forcing block inference pattern.

THIS IS NOT DITTO. Architecture differences:
  - AvatarForcing uses self-forcing: iterative denoising per BLOCK (4 frames)
  - KV-cache (kv_cache_clean + crossattn_cache) persists across blocks within a session
  - Audio conditioning: audio_emb sliced per window from the growing buffer
  - One streaming iteration → 4 decoded video frames (160ms at 25 FPS)

Design:
  - Session init: encode reference image → initial_latent, prefill KV-cache
  - Per block: run one self-forcing block with whatever audio_emb is available
  - Emit 4 RGB frames per block
  - KV-cache updated in-place by the AF pipeline (no manual management needed)
  - If audio_emb buffer is too short, pad with zeros (silence) — avoids startup delay

Audio alignment:
  - AvatarForcing audio_emb is at 25 Hz (bridge output after 2× upsample)
  - AF frame rate is 25 FPS → 1 audio_emb frame per video frame
  - For block of 4 frames: need audio_emb[af_frame_idx : af_frame_idx + 4]
  - Plus the initial_latent (reference frame = frame 0)
  - So total audio_emb tensor indexing: [0 : af_frame_idx + 4 + 1]

Context noise: default_config.yaml has context_noise=0 → use 0 for scheduler.add_noise.
"""

from __future__ import annotations

import asyncio
import logging
import math
import sys
from pathlib import Path
from typing import Iterator, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from einops import rearrange
from PIL import Image
from torchvision.transforms import InterpolationMode

# ── Resolve paths ─────────────────────────────────────────────────────────────
_ROOT = Path(__file__).resolve().parent.parent.parent
_AF_ROOT = _ROOT / "AvatarForcing-inference"
for _p in [str(_ROOT), str(_AF_ROOT)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from utils.inject import slice_conditional_dict  # noqa: E402

logger = logging.getLogger(__name__)


# ── Image preprocessing (identical to AvatarForcing inference.py) ─────────────

class _ResizeKeepRatioArea16:
    def __init__(self, area_hw=(480, 832), div=16):
        self.A = area_hw[0] * area_hw[1]
        self.d = div

    def __call__(self, img: Image.Image) -> Image.Image:
        w, h = img.size
        s  = min(1.0, math.sqrt(self.A / (h * w)))
        nh = max(self.d, int(h * s) // self.d * self.d)
        nw = max(self.d, int(w * s) // self.d * self.d)
        return TF.resize(img, (nh, nw),
                         interpolation=InterpolationMode.BILINEAR, antialias=True)


_IMAGE_TRANSFORM = transforms.Compose([
    _ResizeKeepRatioArea16((480, 832), 16),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
])


# ── Streaming AvatarForcing ────────────────────────────────────────────────────

class StreamingAvatarForcing:
    """
    Wraps AvatarForcingInferencePipeline for block-by-block streaming generation.

    Each call to generate_block() produces num_frame_per_block=4 video frames
    using one iteration of self-forcing denoising.

    The KV-cache inside the AF pipeline is automatically maintained across
    generate_block() calls — this is how temporal consistency is preserved.

    Parameters
    ----------
    pipeline   : loaded AvatarForcingInferencePipeline
    device     : torch.device
    dtype      : torch.bfloat16 (default)
    cuda_stream: optional dedicated CUDA stream
    """

    # How many AF time-steps of audio_emb are needed to cover the
    # initial reference frame + one block of generated frames.
    # audio_emb[:1] = zero prefix (reference frame)
    # audio_emb[1:1+num_frame_per_block] = current block's audio
    AUDIO_EMB_MIN_FRAMES = 1  # just the zero-prefix is enough to start

    def __init__(
        self,
        pipeline,           # AvatarForcingInferencePipeline
        device: torch.device,
        dtype:  torch.dtype = torch.bfloat16,
        cuda_stream: Optional[torch.cuda.Stream] = None,
        prompt: str = (
            "A person talking naturally, realistic facial expressions, "
            "high quality video, detailed face."
        ),
    ):
        self._pipeline     = pipeline
        self._device       = device
        self._dtype        = dtype
        self._stream       = cuda_stream
        self._prompt       = prompt

        self._num_frame_per_block: int = pipeline.num_frame_per_block  # 4
        self._frame_seq_length:    int = pipeline.frame_seq_length      # 1560
        self._context_noise:       int = getattr(pipeline.args, "context_noise", 0)

        # Per-session state
        self._initial_latent: Optional[torch.Tensor] = None  # (1, 1, C, H, W)
        self._output_latents: Optional[torch.Tensor] = None  # growing output buffer
        self._current_start:  int  = 0    # current block start frame index
        self._h: Optional[int] = None
        self._w: Optional[int] = None
        self._session_active: bool = False

        # Precomputed text conditioning (set on session start)
        self._cond_dict = None

    # ── Session lifecycle ────────────────────────────────────────────────────

    @torch.no_grad()
    def start_session(self, image_path: str, prompt: Optional[str] = None) -> None:
        """
        Initialise a new streaming session.
          1. Encode reference image → initial_latent
          2. Reset AvatarForcing KV-cache
          3. Prefill KV-cache with the reference frame
          4. Set internal state to ready

        Must be called once before any generate_block() calls.
        """
        self._prompt = prompt or self._prompt
        pipe = self._pipeline
        dev  = self._device
        dt   = self._dtype

        logger.info(f"[StreamingAF] Starting session: {image_path}")

        # ── Load + encode reference image ───────────────────────────────
        img = Image.open(image_path).convert("RGB")
        img_t = _IMAGE_TRANSFORM(img).unsqueeze(0).unsqueeze(2)  # (1, 3, 1, H, W)
        img_t = img_t.to(device=dev, dtype=dt)

        with torch.cuda.amp.autocast(dtype=dt):
            initial_latent = pipe.vae.encode_to_latent(img_t)   # (1, 1, C, H, W)

        self._initial_latent = initial_latent.to(device=dev, dtype=dt)
        self._h = initial_latent.shape[-2]
        self._w = initial_latent.shape[-1]
        c = initial_latent.shape[2]

        # ── Reset KV-caches ─────────────────────────────────────────────
        pipe._reset_or_init_caches(batch_size=1, dtype=dt, device=dev)

        # ── Prefill KV-cache with reference frame ────────────────────────
        # We need a dummy conditional_dict with at least the image conditioning (y).
        # Use zeros for audio_emb at this stage (reference frame has no audio).
        zero_audio = torch.zeros((1, 1, 10752), device=dev, dtype=dt)

        img_lat = initial_latent.permute(0, 2, 1, 3, 4)  # (1, C, 1, H, W)
        # Build a minimal y tensor for the reference frame
        msk_prefix = torch.zeros_like(img_lat[:, :1, :1])  # (1, 1, 1, H_lat, W_lat)
        # For prefill, we only condition on the reference frame (not mask)
        y_prefix = torch.cat([img_lat, msk_prefix], dim=1)  # (1, C+1, 1, H, W)

        cond_prefix = pipe._build_conditionals(
            text_prompts=[self._prompt],
            noise=torch.zeros((1, 1, c, self._h, self._w), device=dev, dtype=dt),
            audio_embeddings=zero_audio,
            y=y_prefix,
        )

        zero_ts = torch.zeros([1, 1], device=dev, dtype=torch.int64)
        with torch.no_grad(), torch.cuda.amp.autocast(dtype=dt):
            pipe.generator(
                noisy_image_or_video=initial_latent,
                conditional_dict=slice_conditional_dict(cond_prefix, 0, 1),
                timestep=zero_ts,
                kv_cache=pipe.kv_cache_clean,
                crossattn_cache=pipe.crossattn_cache,
                current_start=0,
            )

        # The output latents buffer starts with the encoded reference frame
        self._output_latents = initial_latent.clone()   # (1, 1, C, H, W)
        self._current_start  = 1   # next block starts at frame index 1
        self._session_active = True
        logger.info(
            f"[StreamingAF] Session ready. "
            f"Latent shape: {initial_latent.shape}, "
            f"num_frame_per_block={self._num_frame_per_block}"
        )

    # ── Block generation ──────────────────────────────────────────────────────

    @torch.no_grad()
    def generate_block(
        self,
        audio_emb_buffer: List[torch.Tensor],
    ) -> Optional[List[np.ndarray]]:
        """
        Generate one block of num_frame_per_block=4 video frames.

        Uses self-forcing: iterative denoising over denoise_steps,
        with the KV-cache from previous blocks providing temporal context.

        Parameters
        ----------
        audio_emb_buffer : list of (T_chunk, 10752) tensors — growing buffer
                           We concatenate and slice what we need.
                           Padded with zeros if shorter than needed.

        Returns
        -------
        list of num_frame_per_block RGB numpy arrays (H, W, 3) uint8
        None if session not started
        """
        if not self._session_active:
            logger.warning("[StreamingAF] generate_block called before start_session()")
            return None

        pipe = self._pipeline
        dev  = self._device
        dt   = self._dtype
        nfpb = self._num_frame_per_block   # 4

        # ── Build audio_emb for this block ───────────────────────────────
        # We need audio_emb covering all frames up to current_start + nfpb
        # audio_emb index 0 = zero prefix (reference frame)
        # audio_emb index 1..current_start+nfpb-1 = actual audio conditioning

        total_frames_needed = self._current_start + nfpb

        if audio_emb_buffer:
            full_emb = torch.cat(audio_emb_buffer, dim=0)  # (T_available, 10752)
        else:
            full_emb = torch.zeros((0, 10752), dtype=torch.float32)

        # Zero-prefix: 1 frame for the reference image
        zero_prefix = torch.zeros((1, 10752), dtype=torch.float32)
        full_emb_with_prefix = torch.cat([zero_prefix, full_emb], dim=0)

        # Pad if we don't have enough yet
        have_frames = full_emb_with_prefix.shape[0]
        if have_frames < total_frames_needed:
            n_pad = total_frames_needed - have_frames
            pad   = torch.zeros((n_pad, 10752), dtype=torch.float32)
            full_emb_with_prefix = torch.cat([full_emb_with_prefix, pad], dim=0)

        # AvatarForcing expects shape (B, T, 10752)
        audio_emb_batch = full_emb_with_prefix[:total_frames_needed].unsqueeze(0)
        audio_emb_batch = audio_emb_batch.to(device=dev, dtype=dt)

        # ── Build conditioning dict for the FULL sequence so far ─────────
        frame_s = self._current_start
        frame_e = frame_s + nfpb
        c = self._initial_latent.shape[2]

        # Build y: image conditioning tensor covering the full output so far
        # AvatarForcing uses: image_cat = img_lat repeated, msk[:, :, 1:] = 1
        img_lat = self._initial_latent.permute(0, 2, 1, 3, 4)  # (1, C, 1, H, W)
        total_t = frame_e + 20  # extra padding as in original inference.py
        image_cat = img_lat.repeat(1, 1, total_t, 1, 1)        # (1, C, T+20, H, W)
        msk       = torch.zeros_like(image_cat[:, :1])          # (1, 1, T+20, H, W)
        msk[:, :, 1:] = 1
        y = torch.cat([image_cat, msk], dim=1)                  # (1, C+1, T+20, H, W)
        y = y.to(device=dev, dtype=dt)

        # Sample noise for this block
        sampled_noise = torch.randn(
            (1, nfpb, 16, self._h, self._w),
            device=dev, dtype=dt
        )

        # Build full conditional dict
        cond = pipe._build_conditionals(
            text_prompts=[self._prompt],
            noise=sampled_noise,
            audio_embeddings=audio_emb_batch,
            y=y,
        )

        # ── Self-forcing: iterative denoising for this block ─────────────
        cond_window = slice_conditional_dict(cond, frame_s, frame_e)
        noisy_input = sampled_noise  # (1, nfpb, 16, H, W)
        num_steps   = len(pipe.denoise_steps)

        with torch.cuda.amp.autocast(dtype=dt):
            for si, cur_step in enumerate(pipe.denoise_steps):
                timestep = torch.ones(
                    (1, nfpb), device=dev, dtype=torch.int64
                ) * cur_step

                _, denoised_pred = pipe.generator(
                    noisy_image_or_video=noisy_input,
                    conditional_dict=cond_window,
                    timestep=timestep,
                    kv_cache=pipe.kv_cache_clean,
                    crossattn_cache=pipe.crossattn_cache,
                    current_start=frame_s * self._frame_seq_length,
                )

                if si < num_steps - 1:
                    next_step   = pipe.denoise_steps[si + 1]
                    noisy_input = pipe.scheduler.add_noise(
                        denoised_pred.flatten(0, 1),
                        torch.randn_like(denoised_pred.flatten(0, 1)),
                        next_step * torch.ones(
                            [nfpb], device=dev, dtype=torch.long
                        ),
                    ).unflatten(0, denoised_pred.shape[:2])

            # ── Update KV-cache with context noise ───────────────────────
            context_ts = torch.ones_like(timestep) * self._context_noise
            ctx_latents = denoised_pred.clone()
            if self._context_noise > 0:
                ctx_latents = pipe.scheduler.add_noise(
                    ctx_latents.flatten(0, 1),
                    torch.randn_like(ctx_latents.flatten(0, 1)),
                    context_ts.flatten(),
                ).unflatten(0, ctx_latents.shape[:2])

            pipe.generator(
                noisy_image_or_video=ctx_latents,
                conditional_dict=cond_window,
                timestep=context_ts,
                kv_cache=pipe.kv_cache_clean,
                crossattn_cache=pipe.crossattn_cache,
                current_start=frame_s * self._frame_seq_length,
                updating_cache=True,
            )

            # ── Decode latents → pixels ───────────────────────────────────
            # Concatenate with initial latent so VAE has full context
            out_latents = denoised_pred  # (1, nfpb, 16, H, W)
            decoded = pipe.vae.decode_to_pixel(out_latents, use_cache=False)
            decoded = (decoded * 0.5 + 0.5).clamp(0, 1)   # (1, nfpb, 3, H, W)

            # Clear VAE cache
            pipe.vae.model.clear_cache()

        # ── Convert to numpy ─────────────────────────────────────────────
        # decoded: (1, nfpb, 3, H, W) float in [0,1]
        frames_np = (
            rearrange(decoded, "b t c h w -> b t h w c")
            .cpu().float().numpy()
        )
        frames_np = (frames_np[0] * 255.0).clip(0, 255).astype(np.uint8)
        # frames_np: (nfpb, H, W, 3) uint8

        # ── Advance frame counter ─────────────────────────────────────────
        self._current_start += nfpb

        logger.debug(
            f"[StreamingAF] Block done. Frames {frame_s}→{frame_e}, "
            f"next_start={self._current_start}"
        )

        return [frames_np[i] for i in range(nfpb)]

    # ── Session cleanup ───────────────────────────────────────────────────────

    def end_session(self) -> None:
        """Release session state."""
        self._session_active  = False
        self._initial_latent  = None
        self._output_latents  = None
        self._current_start   = 0
        self._cond_dict       = None
        logger.info("[StreamingAF] Session ended.")

    @property
    def frames_generated(self) -> int:
        return max(0, self._current_start - 1)  # subtract 1 for reference frame

    @property
    def num_frame_per_block(self) -> int:
        return self._num_frame_per_block
