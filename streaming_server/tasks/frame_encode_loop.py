"""
streaming_server/tasks/frame_encode_loop.py
=============================================
Task: RGB frame JPEG encoding loop.

Pulls TaggedFrame (raw numpy RGB) from session.frame_queue,
encodes to JPEG using TurboJPEG (or cv2 fallback), and
pushes packed binary message to session.send_queue.

TurboJPEG is ~10x faster than PIL — critical for 25 FPS throughput.
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

from ..utils.jpeg_encoder import JpegEncoder
from ..utils.protocol import pack_video

if TYPE_CHECKING:
    from ..session import SessionState

logger = logging.getLogger(__name__)


async def frame_encode_loop_task(
    session: "SessionState",
    encoder: JpegEncoder,
) -> None:
    """
    Frame encoding loop.
    Runs encode() in a thread to avoid blocking the event loop.
    """
    logger.info(
        f"[FrameEncode/{session.session_id[:8]}] Started. "
        f"Backend={encoder.backend}"
    )

    try:
        while not session.error_event.is_set():
            # ── Pull a frame ───────────────────────────────────────────
            try:
                tagged = await asyncio.wait_for(
                    session.frame_queue.get(), timeout=1.0
                )
            except asyncio.TimeoutError:
                continue

            # ── Encode in thread (blocking I/O) ────────────────────────
            try:
                jpeg_bytes: bytes = await asyncio.to_thread(
                    encoder.encode, tagged.frame_np
                )
            except Exception as e:
                logger.warning(f"[FrameEncode] JPEG encode failed: {e}")
                continue

            # ── Pack and enqueue for WebSocket sender ──────────────────
            msg = pack_video(tagged.seq, jpeg_bytes)

            if session.send_queue.full():
                try:
                    session.send_queue.get_nowait()
                    logger.debug("[FrameEncode] Send queue full — dropped oldest video msg.")
                except asyncio.QueueEmpty:
                    pass

            try:
                session.send_queue.put_nowait(msg)
            except asyncio.QueueFull:
                pass

            await asyncio.sleep(0)

    except asyncio.CancelledError:
        pass
    except Exception as e:
        logger.error(f"[FrameEncode] Fatal error: {e}", exc_info=True)
        session.error_event.set()
    finally:
        logger.info(f"[FrameEncode/{session.session_id[:8]}] Stopped.")
