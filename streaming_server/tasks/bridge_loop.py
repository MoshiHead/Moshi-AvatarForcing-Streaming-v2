"""
streaming_server/tasks/bridge_loop.py
=======================================
Task: Bridge streaming inference loop.

Pulls TaggedTokens from session.token_queue, feeds them to StreamingBridge,
and appends resulting audio_emb chunks to session.audio_emb_buffer.

Also handles timeout flushing (BRIDGE_FLUSH_TIMEOUT_MS) to avoid stalls
during speech pauses.

The audio_emb_buffer is a shared rolling list protected by session.emb_lock.
The AvatarForcing loop reads from this buffer independently.
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

from ..utils.sync_types import TaggedEmb

if TYPE_CHECKING:
    from ..session import SessionState
    from ..pipeline.streaming_bridge import StreamingBridge

logger = logging.getLogger(__name__)

POLL_INTERVAL_S = 0.010   # 10ms polling interval for timeout-flush check


async def bridge_loop_task(
    session: "SessionState",
    streaming_bridge: "StreamingBridge",
) -> None:
    """
    Bridge inference loop.
    Exits when session.error_event is set.
    """
    logger.info(f"[BridgeLoop/{session.session_id[:8]}] Started.")

    try:
        while not session.error_event.is_set():
            result = None

            # ── Try to get a token (with short timeout for flush polling) ─
            try:
                tagged_token = await asyncio.wait_for(
                    session.token_queue.get(), timeout=POLL_INTERVAL_S
                )
                # Push token → may return flush result
                result = streaming_bridge.push_token(
                    audio_tokens = tagged_token.audio_tokens,
                    seq          = tagged_token.seq,
                )
            except asyncio.TimeoutError:
                # No token arrived — check for timeout flush
                result = streaming_bridge.check_timeout_flush()

            # ── Handle flush result ──────────────────────────────────────
            if result is not None:
                seq_start, seq_end, audio_emb = result
                # audio_emb: (T_out, 10752) float32

                # Append to shared rolling buffer under lock
                async with session.emb_lock:
                    session.audio_emb_buffer.append(audio_emb)

                # Also push a TaggedEmb to emb_queue so AF loop knows
                # new data arrived (used as a trigger signal)
                tagged_emb = TaggedEmb(
                    seq_start = seq_start,
                    seq_end   = seq_end,
                    audio_emb = audio_emb,
                )
                if session.emb_queue.full():
                    try:
                        session.emb_queue.get_nowait()
                    except asyncio.QueueEmpty:
                        pass
                try:
                    session.emb_queue.put_nowait(tagged_emb)
                except asyncio.QueueFull:
                    pass

                logger.debug(
                    f"[BridgeLoop] Emb [{seq_start},{seq_end}) → "
                    f"shape={audio_emb.shape}, "
                    f"buf_len={len(session.audio_emb_buffer)}"
                )

            await asyncio.sleep(0)

    except asyncio.CancelledError:
        # Flush remaining tokens at end
        result = streaming_bridge.force_flush()
        if result is not None:
            _, _, audio_emb = result
            async with session.emb_lock:
                session.audio_emb_buffer.append(audio_emb)
    except Exception as e:
        logger.error(f"[BridgeLoop] Fatal error: {e}", exc_info=True)
        session.error_event.set()
    finally:
        logger.info(f"[BridgeLoop/{session.session_id[:8]}] Stopped.")
