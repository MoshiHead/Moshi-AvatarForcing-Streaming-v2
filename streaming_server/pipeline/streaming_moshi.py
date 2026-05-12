"""
streaming_server/pipeline/streaming_moshi.py
=============================================
Async streaming adapter for Moshi.

Converts the existing file-based MoshiRunner into a live session that:
  - Accepts audio chunks from an asyncio.Queue (browser mic PCM)
  - Maintains a persistent InferenceState across the full session
  - Yields (seq, pcm_chunk, audio_tokens, text_token) per Moshi LM step
  - Yields asyncio event-loop control after every step (asyncio.sleep(0))

Audio input: raw int16 PCM at 24kHz (FRAME_SIZE=1920 samples per chunk).
Audio output: float32 PCM at 24kHz (1920 samples) + 8 acoustic token indices.

Critical design decisions:
  - InferenceState is created ONCE per session and kept alive for the entire session.
  - close_session() resets mimi/lm streaming state UNCONDITIONALLY — multiple
    StreamingMoshi instances share the same underlying mimi/lm via _runner, so
    a new instance (self._state=None) must still reset shared state from a prior
    instance. Never bail early in close_session().
  - No torch.autocast around Moshi calls — Moshi pre-allocates its KV cache at
    a fixed dtype on load. Autocast changes k/v tensor dtypes mid-inference,
    causing: scatter(): Expected self.dtype to be equal to src.dtype
"""

from __future__ import annotations

import asyncio
import logging
import sys
from pathlib import Path
from typing import AsyncIterator, Optional, Tuple

import torch

# ── Resolve paths ─────────────────────────────────────────────────────────────
_ROOT = Path(__file__).resolve().parent.parent.parent
_MOSHI_ROOT = _ROOT / "moshi-inference"
for _p in [str(_ROOT), str(_MOSHI_ROOT)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from unified_pipeline.moshi_runner import MoshiRunner  # noqa: E402
from moshi.run_inference import InferenceState          # noqa: E402

logger = logging.getLogger(__name__)

# Sentinel to signal end of audio stream
_STOP = object()


class StreamingMoshi:
    """
    Streaming adapter wrapping MoshiRunner.

    Usage per session:
        sm = StreamingMoshi(runner)
        sm.reset_session()          # called once at WS connect
        async for result in sm.run(audio_queue, stop_event, seq_fn):
            seq, pcm, audio_toks, text_tok = result
        sm.close_session()          # called at WS disconnect
    """

    def __init__(self, runner: MoshiRunner):
        self._runner = runner
        self._state: Optional[InferenceState] = None
        self._first_frame_done: bool = False

    # ── Session lifecycle ────────────────────────────────────────────────────

    def close_session(self) -> None:
        """
        Reset mimi and lm_gen streaming contexts UNCONDITIONALLY.

        CRITICAL: Multiple StreamingMoshi instances share the SAME underlying
        mimi/lm objects via _runner. A freshly created instance has
        self._state = None, but the shared mimi/lm may still hold streaming
        state from a previous instance. We NEVER bail early — always reset.

        Failing to do this on every disconnect causes:
            AssertionError: is already streaming!
        on the next connection attempt.
        """
        mimi = self._runner._mimi
        lm   = self._runner._lm

        for name, obj in [("mimi", mimi), ("lm", lm)]:
            # Preferred path: Moshi's own reset_streaming() API
            if hasattr(obj, "reset_streaming"):
                try:
                    obj.reset_streaming()
                    logger.debug(f"[StreamingMoshi] reset_streaming() on {name}.")
                except Exception as e:
                    logger.warning(f"[StreamingMoshi] reset_streaming failed on {name}: {e}")
            else:
                # Fallback: walk every sub-module and clear _streaming_state
                cleared = 0
                for module in obj.modules():
                    if hasattr(module, "_streaming_state"):
                        module._streaming_state = None
                        cleared += 1
                if cleared:
                    logger.debug(f"[StreamingMoshi] Cleared _streaming_state on "
                                 f"{cleared} {name} sub-modules (fallback).")

        self._state = None
        self._first_frame_done = False
        logger.info("[StreamingMoshi] Streaming contexts reset (close_session).")

    def reset_session(self) -> None:
        """
        Create a fresh InferenceState for a new WebSocket session.
        Always calls close_session() first to guarantee clean state.
        """
        # ── Always reset streaming state first ───────────────────────────────
        # Even if self._state is None (new instance), the shared mimi/lm may
        # hold state from a previous session's StreamingMoshi instance.
        self.close_session()

        ci   = self._runner._checkpoint_info
        mimi = self._runner._mimi
        tt   = self._runner._text_tokenizer
        lm   = self._runner._lm
        dev  = torch.device(self._runner.device)

        self._state = InferenceState(
            ci, mimi, tt, lm,
            batch_size=1,
            cfg_coef=1.0,
            device=dev,
            **ci.lm_gen_config,
        )
        self._first_frame_done = False
        logger.info("[StreamingMoshi] Fresh InferenceState created for new session.")

    # ── Main async generator ─────────────────────────────────────────────────

    async def run(
        self,
        audio_queue: asyncio.Queue,
        stop_event: asyncio.Event,
        seq_counter_fn,          # callable() → int, monotonic seq IDs
    ) -> AsyncIterator[Tuple[int, torch.Tensor, torch.Tensor, int]]:
        """
        Main streaming loop. Yields one result per Moshi LM step.

        Reads raw PCM chunks from audio_queue. Each chunk must be
        (1, FRAME_SIZE) float32 at 24kHz.

        Yields
        ------
        (seq, pcm_chunk, audio_tokens, text_token)
          seq          : int — monotonic sequence id
          pcm_chunk    : (1, FRAME_SIZE) float32 — Moshi response audio
          audio_tokens : (1, 8) int64 — acoustic codebook indices
          text_token   : int — SentencePiece token id
        """
        assert self._state is not None, "Call reset_session() first."

        mimi   = self._runner._mimi
        state  = self._state
        dev    = torch.device(self._runner.device)

        with torch.no_grad():
            while not stop_event.is_set():
                # ── Get next audio chunk ─────────────────────────────────
                try:
                    pcm_in = await asyncio.wait_for(
                        audio_queue.get(), timeout=1.0
                    )
                except asyncio.TimeoutError:
                    continue

                if pcm_in is _STOP:
                    break

                # Ensure shape (1, 1, FRAME_SIZE) for mimi.encode
                if pcm_in.dim() == 1:
                    pcm_in = pcm_in.unsqueeze(0).unsqueeze(0)
                elif pcm_in.dim() == 2:
                    pcm_in = pcm_in.unsqueeze(0)
                pcm_in = pcm_in.to(dev)

                # ── Mimi encode + LM step ────────────────────────────────
                # NO autocast: Moshi pre-allocates KV cache at a fixed dtype
                # on load. Autocast converts k/v tensors to bfloat16 mid-
                # inference, causing scatter() dtype mismatch in the cache.
                codes = mimi.encode(pcm_in)   # (1, n_codebooks, 1)

                # First-frame: double step to handle causal delay
                if not self._first_frame_done:
                    _ = state.lm_gen.step(codes)
                    self._first_frame_done = True

                tokens = state.lm_gen.step(codes)  # (1, dep_q+1, 1) | None

                # ── Yield event-loop control (prevents starvation) ───────
                await asyncio.sleep(0)

                if tokens is None:
                    continue

                # ── Extract outputs ──────────────────────────────────────
                text_tok   = int(tokens[:, 0, 0].item())
                audio_toks = tokens[:, 1:, :]               # (1, 8, 1)

                # Decode response audio — also no autocast
                pcm_out = mimi.decode(audio_toks)            # (1, 1, FRAME_SIZE)

                pcm_out    = pcm_out.squeeze(1).cpu().float()  # (1, FRAME_SIZE)
                audio_toks = audio_toks[:, :, 0].cpu()         # (1, 8)

                seq = seq_counter_fn()
                yield seq, pcm_out, audio_toks, text_tok

                # Yield again to ensure downstream tasks run
                await asyncio.sleep(0)

        logger.info("[StreamingMoshi] run() loop exited.")

    @staticmethod
    def stop_sentinel() -> object:
        """Put this into the audio_queue to signal end of stream."""
        return _STOP
