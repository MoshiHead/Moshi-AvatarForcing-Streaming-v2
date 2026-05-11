"""
unified_pipeline/moshi_runner.py
=================================
Thin wrapper around the Moshi InferenceState that:
  - Accepts a user audio waveform (24 kHz, mono)
  - Runs Moshi step-by-step
  - Yields per-step results as a dataclass

Yields
------
MoshiStep.audio_tokens : (1, 8)  int64  — 8 response codebook indices for this step
MoshiStep.text_token   : int            — decoded text piece id
MoshiStep.pcm_chunk    : (1, frame_size) float32 — raw audio from Mimi decoder

This keeps the full Moshi logic completely unchanged; we only
intercept the tokens that run_inference.py already exposes.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Optional

import torch


# ─── Moshi InferenceState lives in moshi-inference/moshi/run_inference.py ────
# Add the moshi-inference root so we can import it as a package.
MOSHI_ROOT = Path(__file__).resolve().parent.parent / "moshi-inference"
if str(MOSHI_ROOT) not in sys.path:
    sys.path.insert(0, str(MOSHI_ROOT))

from moshi.models import loaders                        # noqa: E402
from moshi.run_inference import InferenceState          # noqa: E402


@dataclass
class MoshiStep:
    """One decoded step from the Moshi LM."""
    audio_tokens: torch.Tensor   # (1, 8)  int64 — dep_q=8 response audio tokens
    text_token:   int            # SentencePiece id
    pcm_chunk:    torch.Tensor   # (1, frame_size) float32 at mimi.sample_rate


@dataclass
class MoshiRunner:
    """
    Loads Moshi model weights and wraps InferenceState.

    Parameters
    ----------
    hf_repo       : HuggingFace repo id, e.g. 'kyutai/moshiko-pytorch-q8'
    moshi_weight  : optional local path to override the Moshi LM checkpoint
    mimi_weight   : optional local path to override the Mimi codec checkpoint
    tokenizer     : optional local path to the SentencePiece tokenizer
    device        : 'cuda' or 'cpu'
    dtype         : torch.bfloat16 (default) or torch.float16
    """

    hf_repo:      str  = "kyutai/moshiko-pytorch-q8"
    moshi_weight: Optional[str] = None
    mimi_weight:  Optional[str] = None
    tokenizer:    Optional[str] = None
    device:       str  = "cuda"
    dtype:        torch.dtype = torch.bfloat16

    # filled by __post_init__
    _checkpoint_info: object = field(default=None, repr=False, init=False)
    _mimi:            object = field(default=None, repr=False, init=False)
    _text_tokenizer:  object = field(default=None, repr=False, init=False)
    _lm:              object = field(default=None, repr=False, init=False)
    _frame_size:      int    = field(default=0,    repr=False, init=False)
    _sample_rate:     int    = field(default=24000, repr=False, init=False)

    def __post_init__(self):
        pass  # lazy load via .load()

    def load(self):
        """Download / load model weights. Call once before inference."""
        print("[MoshiRunner] Loading checkpoint info …")
        ci = loaders.CheckpointInfo.from_hf_repo(
            self.hf_repo,
            self.moshi_weight,
            self.mimi_weight,
            self.tokenizer,
        )
        print("[MoshiRunner] Loading Mimi codec …")
        mimi = ci.get_mimi(device=self.device)
        print("[MoshiRunner] Loading Moshi LM …")
        lm   = ci.get_moshi(device=self.device, dtype=self.dtype)

        self._checkpoint_info = ci
        self._mimi = mimi
        self._text_tokenizer = ci.get_text_tokenizer()
        self._lm   = lm
        self._frame_size  = int(mimi.sample_rate / mimi.frame_rate)
        self._sample_rate = mimi.sample_rate
        print(f"[MoshiRunner] Ready. frame_size={self._frame_size}, "
              f"sample_rate={self._sample_rate}")

    @property
    def frame_size(self) -> int:
        return self._frame_size

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    def run(self, in_pcm: torch.Tensor) -> Iterator[MoshiStep]:
        """
        Run Moshi on a full utterance and yield one MoshiStep per LM step.

        Parameters
        ----------
        in_pcm : (1, 1, N) float32 tensor at 24 kHz (mono)
                 OR (N,) / (1, N) — will be reshaped automatically

        Yields
        ------
        MoshiStep  (one per Moshi frame = 1/12.5 s = 80 ms)
        """
        assert self._mimi is not None, "Call .load() first."
        mimi = self._mimi
        lm   = self._lm
        ci   = self._checkpoint_info
        dev  = torch.device(self.device)

        # normalise shape to (1, 1, N)
        if in_pcm.dim() == 1:
            in_pcm = in_pcm.unsqueeze(0).unsqueeze(0)
        elif in_pcm.dim() == 2:
            in_pcm = in_pcm.unsqueeze(0)
        in_pcm = in_pcm.to(dev)

        # Build InferenceState (this calls mimi.streaming_forever + lm_gen.streaming_forever)
        state = InferenceState(
            ci, mimi, self._text_tokenizer, lm,
            batch_size=1,
            cfg_coef=1.0,
            device=dev,
            **ci.lm_gen_config,
        )

        # Split audio into chunks
        chunks = [
            c for c in in_pcm.split(self._frame_size, dim=2)
            if c.shape[-1] == self._frame_size
        ]

        first_frame = True
        with torch.no_grad():
            for chunk in chunks:
                # Encode audio chunk → Mimi codes
                codes = mimi.encode(chunk)       # (1, n_codebooks, 1)

                if first_frame:
                    tokens = state.lm_gen.step(codes)
                    if max(state.lm_gen.lm_model.delays) > 0:
                        assert tokens is None
                    first_frame = False

                tokens = state.lm_gen.step(codes)  # (1, dep_q+1, 1) or None
                if tokens is None:
                    continue

                # tokens[:, 0, :] = text; tokens[:, 1:, :] = 8 audio codebooks
                text_tok   = tokens[:, 0, 0]        # (1,) int64
                audio_toks = tokens[:, 1:, :]        # (1, 8, 1) int64

                # Decode to audio (exactly what run_inference.py does)
                pcm_chunk = mimi.decode(audio_toks).cpu()  # (1, 1, frame_size)
                pcm_chunk = pcm_chunk.squeeze(1)            # (1, frame_size)

                yield MoshiStep(
                    audio_tokens = audio_toks[:, :, 0].cpu(),   # (1, 8) int64
                    text_token   = int(text_tok[0].item()),
                    pcm_chunk    = pcm_chunk,
                )
