"""
streaming_server/tasks/moshi_loop.py
======================================
Task: Moshi streaming inference loop.

Pulls audio PCM from session.audio_queue, runs Moshi LM step-by-step,
and for each step:
  1. Sends PCM response audio IMMEDIATELY via priority path (bypass send_queue)
  2. Pushes TaggedToken to session.token_queue for the bridge

Audio is HIGHEST PRIORITY — it bypasses the normal send queue and is sent
directly under ws_write_lock even if the video pipeline is backed up.
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

import torch
from fastapi import WebSocket

from ..utils.protocol import pack_audio, pack_text
from ..utils.sync_types import TaggedToken

if TYPE_CHECKING:
    from ..session import SessionState
    from ..pipeline.streaming_moshi import StreamingMoshi

logger = logging.getLogger(__name__)


async def moshi_loop_task(
    websocket: WebSocket,
    session: "SessionState",
    streaming_moshi: "StreamingMoshi",
) -> None:
    """
    Core Moshi inference loop.

    Runs until session.error_event is set or the audio_queue receives
    the stop sentinel.
    """
    logger.info(f"[MoshiLoop/{session.session_id[:8]}] Started.")

    try:
        # Wait for image to be ready before generating video-synced audio
        # (audio itself can start immediately, but token_queue feeds AF)
        async for seq, pcm_chunk, audio_toks, text_tok in streaming_moshi.run(
            audio_queue    = session.audio_queue,
            stop_event     = session.error_event,
            seq_counter_fn = session.next_seq,
        ):
            # ── 1. PRIORITY: send audio immediately ─────────────────────
            try:
                audio_bytes = pack_audio(seq, pcm_chunk)
                async with session.ws_write_lock:
                    await websocket.send_bytes(audio_bytes)
            except Exception as e:
                logger.warning(f"[MoshiLoop] Audio send failed: {e}")
                session.error_event.set()
                break

            # ── 2. Push token for bridge ─────────────────────────────────
            tagged = TaggedToken(
                seq          = seq,
                audio_tokens = audio_toks,   # (1, 8)
                text_token   = text_tok,
                pcm_chunk    = pcm_chunk,
            )

            # Drop oldest if token_queue is full (live freshness)
            if session.token_queue.full():
                try:
                    session.token_queue.get_nowait()
                    logger.debug("[MoshiLoop] Token queue full — dropped oldest.")
                except asyncio.QueueEmpty:
                    pass

            try:
                session.token_queue.put_nowait(tagged)
            except asyncio.QueueFull:
                pass

            # ── 3. Optional: send text token ──────────────────────────────
            # Uncomment to enable text streaming
            # if text_tok > 3:  # skip special tokens
            #     try:
            #         session.send_queue.put_nowait(pack_text(text_tok))
            #     except asyncio.QueueFull:
            #         pass

            await asyncio.sleep(0)  # yield after every step

    except asyncio.CancelledError:
        pass
    except Exception as e:
        logger.error(f"[MoshiLoop] Fatal error: {e}", exc_info=True)
        session.error_event.set()
    finally:
        logger.info(f"[MoshiLoop/{session.session_id[:8]}] Stopped.")
