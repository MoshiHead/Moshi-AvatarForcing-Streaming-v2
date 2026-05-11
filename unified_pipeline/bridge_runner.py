"""
unified_pipeline/bridge_runner.py
===================================
Accumulates Moshi audio tokens frame-by-frame, then runs the
moshi-wav2vec-bridge to produce the exact audio_emb tensor that
AvatarForcing expects in its conditional dict.

Output replicates AvatarForcing utils/dataset.py lines 426-437 exactly:

    hs = audio_encoder(input_values, seq_len=teacher_len, output_hidden_states=True)
    audio_emb = hs.last_hidden_state
    for h in hs.hidden_states:
        audio_emb = torch.cat([audio_emb, h], dim=-1)   # → (1, T, 10752)
    audio_emb = audio_emb.squeeze(0)                     # (T, 10752)
    audio_emb = torch.cat([zeros_like(audio_emb[:1]), audio_emb], dim=0)  # (T+1, 10752)
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import torch
import yaml

# Add bridge root to path
BRIDGE_ROOT = Path(__file__).resolve().parent.parent / "moshi-wav2vec-bridge"
if str(BRIDGE_ROOT) not in sys.path:
    sys.path.insert(0, str(BRIDGE_ROOT))

from model import MimiWav2Vec2Bridge  # noqa: E402


class BridgeRunner:
    """
    Converts Moshi response audio tokens → AvatarForcing audio_emb.

    Key parameters (from AvatarForcing avatarforcing.yaml):
        fps         = 25 Hz   → output frame rate of bridge
        teacher_len = 80      → how many output frames per segment
                               → need teacher_len / 2 = 40 Mimi input frames (at 12.5 Hz)

    Token accumulation:
        Moshi yields one (1, 8) token per step at 12.5 Hz.
        We accumulate `mimi_frames_needed` tokens then run the bridge.
        bridge output: (1, teacher_len, 10752) after concat.
        With zero-prefix: (teacher_len + 1, 10752) — shape expected by AvatarForcing.
    """

    # AvatarForcing avatarforcing.yaml defaults
    FPS         = 25       # Hz — AvatarForcing fps
    MIMI_RATE   = 12.5     # Hz — Moshi token rate

    def __init__(
        self,
        checkpoint_path: str,
        config_path: Optional[str] = None,
        teacher_len: int = 80,
        device: str = "cuda",
    ):
        """
        Parameters
        ----------
        checkpoint_path : path to trained bridge .pt checkpoint
        config_path     : path to bridge config.yaml (default: auto-detect)
        teacher_len     : number of output frames (matches AvatarForcing teacher_len)
                          Must be even — mimi_frames = teacher_len / 2
        device          : 'cuda' or 'cpu'
        """
        self.device       = torch.device(device)
        self.teacher_len  = teacher_len

        # Number of Mimi tokens needed to produce teacher_len output frames:
        # bridge upsamples ×2, so: mimi_frames × 2 = teacher_len
        self.mimi_frames_needed = teacher_len // 2
        assert teacher_len % 2 == 0, (
            f"teacher_len must be even (bridge upsamples ×2). Got {teacher_len}."
        )

        # Load config
        if config_path is None:
            config_path = str(BRIDGE_ROOT / "config.yaml")
        with open(config_path) as f:
            self.cfg = yaml.safe_load(f)

        # Load bridge model
        self.model = MimiWav2Vec2Bridge(self.cfg).to(self.device)
        self._load_checkpoint(checkpoint_path)
        self.model.eval()

        # Token accumulation buffer: list of (1, 8) tensors
        self._token_buffer: list[torch.Tensor] = []

        num_codebooks = self.cfg["model"]["num_codebooks"]
        output_dim    = self.cfg["model"]["output_dim"]
        print(
            f"[BridgeRunner] Ready. teacher_len={teacher_len}, "
            f"mimi_frames_needed={self.mimi_frames_needed}, "
            f"concat_dim={14 * output_dim} (14×{output_dim})"
        )

    def _load_checkpoint(self, path: str):
        try:
            ckpt = torch.load(path, map_location=self.device, weights_only=True)
        except TypeError:
            ckpt = torch.load(path, map_location=self.device)
        sd = ckpt.get("bridge", ckpt)
        missing, unexpected = self.model.load_state_dict(sd, strict=False)
        if missing:
            print(f"[BridgeRunner] WARNING missing keys: {missing}")
        if unexpected:
            print(f"[BridgeRunner] WARNING unexpected keys: {unexpected}")

    def reset(self):
        """Clear accumulated token buffer (call at the start of each utterance)."""
        self._token_buffer = []

    def push_tokens(self, audio_tokens: torch.Tensor) -> Optional[torch.Tensor]:
        """
        Accumulate one Moshi step's worth of audio tokens.
        When enough tokens are collected (mimi_frames_needed), run the bridge
        and return the audio_emb tensor ready for AvatarForcing.

        Parameters
        ----------
        audio_tokens : (1, 8) or (8,) int64 — one Moshi LM step output

        Returns
        -------
        audio_emb : (teacher_len + 1, 10752) float32  — when buffer is full
        None                                           — still accumulating
        """
        # Normalise to (8,) CPU int64
        if audio_tokens.dim() == 2:
            audio_tokens = audio_tokens.squeeze(0)   # (8,)
        audio_tokens = audio_tokens.cpu()

        self._token_buffer.append(audio_tokens)

        if len(self._token_buffer) < self.mimi_frames_needed:
            return None

        # We have exactly mimi_frames_needed tokens — run bridge
        # Stack into (1, T_mimi, 8)
        tokens_batch = torch.stack(self._token_buffer, dim=0)   # (T_mimi, 8)
        tokens_batch = tokens_batch.unsqueeze(0).to(self.device) # (1, T_mimi, 8)

        # Clear buffer for next segment
        self._token_buffer = []

        return self._run_bridge(tokens_batch, target_len=self.teacher_len + 1)

    def flush(self) -> Optional[torch.Tensor]:
        """
        If there are remaining tokens (< mimi_frames_needed), pad with zeros and run.
        Call at the end of the audio stream.

        Returns
        -------
        audio_emb or None (if buffer is empty)
        """
        if not self._token_buffer:
            return None

        n = len(self._token_buffer)
        pad_needed = self.mimi_frames_needed - n

        # Pad with zero tokens (silence)
        zero_token = torch.zeros(8, dtype=torch.long)
        for _ in range(pad_needed):
            self._token_buffer.append(zero_token)

        tokens_batch = torch.stack(self._token_buffer, dim=0).unsqueeze(0).to(self.device)
        self._token_buffer = []
        return self._run_bridge(tokens_batch, target_len=self.teacher_len + 1)

    @torch.no_grad()
    def _run_bridge(self, tokens: torch.Tensor, target_len: Optional[int] = None) -> torch.Tensor:
        """
        Core bridge forward + AvatarForcing concat replication.

        Parameters
        ----------
        tokens : (1, T_mimi, 8) int64
        target_len : Optional int, if set, pads or trims to this length

        Returns
        -------
        audio_emb : float32 tensor
                    Matches sample['audio_emb'] from AvatarForcing dataset.py.
        """
        # Bridge forward — Wav2Vec2LikeOutput
        hs, _ = self.model(tokens, output_hidden_states=True)

        # ── Replicate AvatarForcing dataset.py lines 430-436 exactly ──────────
        # audio_emb = hs.last_hidden_state
        # for h in hs.hidden_states:
        #     audio_emb = torch.cat([audio_emb, h], dim=-1)
        # audio_emb = audio_emb.squeeze(0)
        # audio_emb = torch.cat([zeros_like(audio_emb[:1]), audio_emb], dim=0)
        audio_emb = hs.last_hidden_state                              # (1, T_out, 768)
        for h in hs.hidden_states:
            audio_emb = torch.cat([audio_emb, h], dim=-1)            # → (1, T_out, 10752)

        audio_emb = audio_emb.squeeze(0).float()                     # (T_out, 10752)

        # Zero-prefix frame (identical to dataset.py)
        zero_prefix = torch.zeros_like(audio_emb[:1])                # (1, 10752)
        audio_emb = torch.cat([zero_prefix, audio_emb], dim=0)       # (T_out+1, 10752)

        # Trim or pad to exactly target_len
        if target_len is None:
            return audio_emb.cpu()

        if audio_emb.shape[0] > target_len:
            audio_emb = audio_emb[:target_len]
        elif audio_emb.shape[0] < target_len:
            pad = torch.zeros(
                target_len - audio_emb.shape[0], audio_emb.shape[1],
                dtype=audio_emb.dtype
            )
            audio_emb = torch.cat([audio_emb, pad], dim=0)

        return audio_emb.cpu()    # (target_len, 10752)
