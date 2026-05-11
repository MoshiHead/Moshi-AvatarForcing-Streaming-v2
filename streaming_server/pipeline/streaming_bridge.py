"""
streaming_server/pipeline/streaming_bridge.py
===============================================
Streaming-capable bridge adapter using KV-cache for incremental inference.

The MimiWav2Vec2Bridge already supports use_cache=True in its transformer
layers (CausalSelfAttention with past_kvs). This module wraps it into a
stateful streaming object that:

  1. Processes token chunks incrementally (BRIDGE_CHUNK tokens at a time)
  2. Maintains KV-cache (past_kvs) across calls within a session
  3. Returns audio_emb for ONLY the newly processed tokens (not the full history)
  4. Runs in a dedicated CUDA stream to overlap with Moshi inference

Output per call:
  audio_emb_chunk : (2 * BRIDGE_CHUNK, 10752) float32
  (bridge upsamples ×2: 12.5 Hz Moshi → 25 Hz AvatarForcing)

The caller (bridge_loop.py) accumulates these chunks into a rolling buffer
that AvatarForcing consumes block-by-block.
"""

from __future__ import annotations

import asyncio
import logging
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

import torch

# ── Resolve paths ─────────────────────────────────────────────────────────────
_ROOT = Path(__file__).resolve().parent.parent.parent
_BRIDGE_ROOT = _ROOT / "moshi-wav2vec-bridge"
for _p in [str(_ROOT), str(_BRIDGE_ROOT)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from model import MimiWav2Vec2Bridge  # noqa: E402

logger = logging.getLogger(__name__)

# ── Configuration ──────────────────────────────────────────────────────────────
BRIDGE_CHUNK            = 2      # Number of Moshi tokens to accumulate before flushing
BRIDGE_FLUSH_TIMEOUT_MS = 50.0   # Flush even if chunk not full (avoids pause stalls)


class StreamingBridge:
    """
    Incremental bridge inference with KV-cache.

    Each call to step() processes one or more new tokens and returns the
    corresponding audio_emb slice. The transformer KV-cache is maintained
    across all calls within a session — no full-sequence recomputation.

    Parameters
    ----------
    model          : loaded MimiWav2Vec2Bridge (shared across sessions is OK
                     only if this is session-private — use per-session instances)
    device         : torch.device
    cuda_stream    : optional dedicated CUDA stream for overlap with Moshi
    chunk_size     : how many Moshi tokens to batch per bridge call
    flush_timeout  : max ms to wait before flushing a partial chunk
    """

    def __init__(
        self,
        model:         MimiWav2Vec2Bridge,
        device:        torch.device,
        cuda_stream:   Optional[torch.cuda.Stream] = None,
        chunk_size:    int   = BRIDGE_CHUNK,
        flush_timeout: float = BRIDGE_FLUSH_TIMEOUT_MS,
    ):
        self._model        = model
        self._device       = device
        self._stream       = cuda_stream
        self.chunk_size    = chunk_size
        self.flush_timeout = flush_timeout

        # Per-session state (reset on reset_session())
        self._past_kvs:    Optional[list]          = None
        self._token_buf:   List[torch.Tensor]      = []   # list of (1, 8)
        self._buf_seq_start: int                   = 0
        self._buf_first_ts: Optional[float]        = None  # time.monotonic()

    # ── Session lifecycle ────────────────────────────────────────────────────

    def reset_session(self) -> None:
        """Clear all KV-cache and buffers for a new session."""
        self._past_kvs     = None
        self._token_buf    = []
        self._buf_seq_start = 0
        self._buf_first_ts  = None
        logger.info("[StreamingBridge] Session reset.")

    # ── Incremental token push ────────────────────────────────────────────────

    def push_token(
        self,
        audio_tokens: torch.Tensor,  # (1, 8) int64
        seq: int,
    ) -> Optional[Tuple[int, int, torch.Tensor]]:
        """
        Accept one Moshi token. Returns audio_emb if flush threshold is met.

        Returns
        -------
        (seq_start, seq_end, audio_emb_chunk)  if flush occurred
        None                                    if still buffering
        """
        tok = audio_tokens.cpu()
        if tok.dim() == 2:
            tok = tok.squeeze(0)   # → (8,)

        if not self._token_buf:
            self._buf_seq_start = seq
            self._buf_first_ts  = time.monotonic()

        self._token_buf.append(tok)

        # Check flush conditions
        if len(self._token_buf) >= self.chunk_size:
            return self._flush()

        # Timeout flush (prevents stalls during silence/pauses)
        elapsed_ms = (time.monotonic() - self._buf_first_ts) * 1000.0
        if elapsed_ms >= self.flush_timeout:
            logger.debug(
                f"[StreamingBridge] Timeout flush after {elapsed_ms:.1f}ms, "
                f"tokens={len(self._token_buf)}"
            )
            return self._flush()

        return None

    def force_flush(self) -> Optional[Tuple[int, int, torch.Tensor]]:
        """Flush any remaining buffered tokens (call at end of session)."""
        if not self._token_buf:
            return None
        return self._flush(pad=True)

    def check_timeout_flush(self) -> Optional[Tuple[int, int, torch.Tensor]]:
        """
        Called by the bridge loop to check for timeout flush even when
        no new token arrived. Safe to call repeatedly.
        """
        if not self._token_buf or self._buf_first_ts is None:
            return None
        elapsed_ms = (time.monotonic() - self._buf_first_ts) * 1000.0
        if elapsed_ms >= self.flush_timeout:
            return self._flush()
        return None

    # ── Internal flush ────────────────────────────────────────────────────────

    def _flush(self, pad: bool = False) -> Tuple[int, int, torch.Tensor]:
        """Run bridge on buffered tokens. Returns (seq_start, seq_end, audio_emb)."""
        buf = list(self._token_buf)

        if pad and len(buf) < self.chunk_size:
            n_pad = self.chunk_size - len(buf)
            zero  = torch.zeros(8, dtype=torch.long)
            buf.extend([zero] * n_pad)

        # Stack: (1, T_chunk, 8)
        tokens_batch = torch.stack(buf, dim=0).unsqueeze(0).to(self._device)

        seq_start = self._buf_seq_start
        seq_end   = seq_start + len(self._token_buf)

        # Clear buffer
        self._token_buf    = []
        self._buf_seq_start = seq_end
        self._buf_first_ts  = None

        # ── Run bridge with KV-cache ─────────────────────────────────────
        ctx = torch.cuda.stream(self._stream) if self._stream else _null_ctx()
        with ctx, torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.bfloat16):
            hs, present_kvs = self._model(
                tokens_batch,
                output_hidden_states=True,
                use_cache=True,
                past_kvs=self._past_kvs,
            )

        # Update KV-cache for next call
        self._past_kvs = present_kvs

        # Replicate AvatarForcing dataset.py concatenation
        audio_emb = hs.last_hidden_state   # (1, 2*T_chunk, 768)
        for h in hs.hidden_states:
            audio_emb = torch.cat([audio_emb, h], dim=-1)   # → (1, 2*T, 10752)

        audio_emb = audio_emb.squeeze(0).float().cpu()      # (2*T_chunk, 10752)

        logger.debug(
            f"[StreamingBridge] Flushed seqs [{seq_start},{seq_end}), "
            f"emb shape={audio_emb.shape}"
        )
        return seq_start, seq_end, audio_emb


# ── Null context manager for when no CUDA stream is provided ─────────────────

class _null_ctx:
    def __enter__(self): return self
    def __exit__(self, *_): pass
