"""
streaming_server/tasks/audio_receive.py
=========================================
Task: WebSocket audio receive loop.

Reads binary messages from the WebSocket and dispatches them:
  - 0x01 messages → parse PCM → put into session.audio_queue
  - Other messages → log and ignore

If audio_queue is full (browser sending faster than Moshi can process),
we drop the oldest chunk to maintain live freshness.
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

from fastapi import WebSocket

from ..utils.protocol import MSG_AUDIO, FRAME_SIZE, unpack_client_audio

if TYPE_CHECKING:
    from ..session import SessionState

logger = logging.getLogger(__name__)


async def audio_receive_task(
    websocket: WebSocket,
    session: "SessionState",
) -> None:
    """
    Continuously receive binary WebSocket messages and route audio to audio_queue.
    Exits when error_event is set or WebSocket closes.
    """
    logger.info(f"[AudioReceive/{session.session_id[:8]}] Started.")

    try:
        while not session.error_event.is_set():
            try:
                data = await asyncio.wait_for(
                    websocket.receive_bytes(), timeout=2.0
                )
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.warning(f"[AudioReceive] WebSocket receive error: {e}")
                session.error_event.set()
                break

            if not data:
                continue

            msg_type = data[0]

            if msg_type == MSG_AUDIO:
                try:
                    _, pcm = unpack_client_audio(data)
                except Exception as e:
                    logger.debug(f"[AudioReceive] Bad audio packet: {e}")
                    continue

                # Ensure correct frame size
                n = pcm.shape[-1]
                if n != FRAME_SIZE:
                    # Pad or trim to FRAME_SIZE
                    import torch
                    if n < FRAME_SIZE:
                        pad = torch.zeros(1, FRAME_SIZE - n, dtype=pcm.dtype)
                        pcm = torch.cat([pcm, pad], dim=-1)
                    else:
                        pcm = pcm[:, :FRAME_SIZE]

                # Latest-chunk-wins: drop oldest if queue full
                if session.audio_queue.full():
                    try:
                        session.audio_queue.get_nowait()
                        logger.debug("[AudioReceive] Audio queue full — dropped oldest chunk.")
                    except asyncio.QueueEmpty:
                        pass

                try:
                    session.audio_queue.put_nowait(pcm)
                except asyncio.QueueFull:
                    pass  # Already drained above; shouldn't happen

            else:
                logger.debug(f"[AudioReceive] Unknown message type: 0x{msg_type:02X}")

    except asyncio.CancelledError:
        pass
    except Exception as e:
        logger.error(f"[AudioReceive] Fatal error: {e}", exc_info=True)
        session.error_event.set()
    finally:
        logger.info(f"[AudioReceive/{session.session_id[:8]}] Stopped.")
