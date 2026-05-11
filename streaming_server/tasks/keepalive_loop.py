"""
streaming_server/tasks/keepalive_loop.py
==========================================
Task: Periodic keepalive ping loop.

Sends a 0xFE byte every KEEPALIVE_INTERVAL_S seconds to prevent
proxy/load-balancer connection timeouts during silence gaps.

Also handles the WebSocket send_queue: drains queued video messages
and sends them to the client at up to TARGET_FPS.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import TYPE_CHECKING

from fastapi import WebSocket

from ..utils.protocol import pack_keepalive

if TYPE_CHECKING:
    from ..session import SessionState

logger = logging.getLogger(__name__)

KEEPALIVE_INTERVAL_S  = 15.0          # send 0xFE every 15s
TARGET_FPS            = 25.0          # target video frame send rate
FRAME_INTERVAL_S      = 1.0 / TARGET_FPS   # ~40ms between frames


async def keepalive_loop_task(
    websocket: WebSocket,
    session: "SessionState",
) -> None:
    """
    Combined keepalive + video send loop.

    Drains session.send_queue (JPEG frames + other non-audio messages)
    at TARGET_FPS, and periodically sends keepalive pings.
    """
    logger.info(f"[Keepalive/{session.session_id[:8]}] Started.")
    last_keepalive = time.monotonic()

    try:
        while not session.error_event.is_set():
            now = time.monotonic()

            # ── Keepalive ping ───────────────────────────────────────────
            if now - last_keepalive >= KEEPALIVE_INTERVAL_S:
                try:
                    async with session.ws_write_lock:
                        await websocket.send_bytes(pack_keepalive())
                    last_keepalive = now
                    logger.debug(f"[Keepalive] Sent 0xFE ping.")
                except Exception as e:
                    logger.warning(f"[Keepalive] Ping send failed: {e}")
                    session.error_event.set()
                    break

            # ── Drain send_queue (video frames + text) ───────────────────
            drained = 0
            while not session.send_queue.empty() and drained < 3:
                try:
                    msg = session.send_queue.get_nowait()
                except asyncio.QueueEmpty:
                    break

                try:
                    async with session.ws_write_lock:
                        await websocket.send_bytes(msg)
                    drained += 1
                except Exception as e:
                    logger.warning(f"[Keepalive] Send failed: {e}")
                    session.error_event.set()
                    break

            # ── Sleep at ~TARGET_FPS ─────────────────────────────────────
            await asyncio.sleep(FRAME_INTERVAL_S)

    except asyncio.CancelledError:
        pass
    except Exception as e:
        logger.error(f"[Keepalive] Fatal error: {e}", exc_info=True)
        session.error_event.set()
    finally:
        logger.info(f"[Keepalive/{session.session_id[:8]}] Stopped.")
