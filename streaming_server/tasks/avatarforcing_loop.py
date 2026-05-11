"""
streaming_server/tasks/avatarforcing_loop.py
=============================================
Task: AvatarForcing streaming block generation loop.

Monitors the shared audio_emb_buffer and, whenever new embeddings arrive,
runs one self-forcing block to generate num_frame_per_block=4 video frames.

Key design:
  - Waits for at least 1 new emb chunk (from emb_queue signal) before generating
  - Reads the FULL audio_emb_buffer (not just the new chunk) when calling generate_block()
    so AvatarForcing has all available conditioning context
  - Runs generate_block() in asyncio.to_thread to avoid blocking the event loop
    (AF inference is GPU-heavy and takes ~100-400ms per block)
  - Backpressure: if frame_queue is full, drops oldest frames

Audio-to-video alignment:
  - Bridge emb chunks at 25 Hz (2 tokens → 4 emb frames)
  - AF generates 4 video frames per block at 25 FPS
  - Alignment is approximate: AF uses whatever emb is in the buffer
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, List

import numpy as np

from ..utils.sync_types import TaggedFrame

if TYPE_CHECKING:
    from ..session import SessionState
    from ..pipeline.streaming_avatarforcing import StreamingAvatarForcing

logger = logging.getLogger(__name__)

# Minimum audio_emb frames before triggering first AF generation
MIN_EMB_FRAMES_TO_START = 2   # very low — 2 bridge chunks = 4 audio emb frames


async def avatarforcing_loop_task(
    session: "SessionState",
    streaming_af: "StreamingAvatarForcing",
) -> None:
    """
    AvatarForcing block generation loop.

    Waits for new audio_emb to arrive, then generates one block of 4 frames.
    Runs generate_block() in a thread to avoid blocking the event loop.
    """
    logger.info(f"[AFLoop/{session.session_id[:8]}] Started. "
                f"Waiting for image_ready event …")

    try:
        # Wait until the reference image is uploaded + AF session started
        await asyncio.wait_for(session.image_ready.wait(), timeout=120.0)
    except asyncio.TimeoutError:
        logger.error("[AFLoop] Timed out waiting for image upload.")
        session.error_event.set()
        return

    logger.info(f"[AFLoop/{session.session_id[:8]}] Image ready — generating.")

    emb_chunks_received = 0

    try:
        while not session.error_event.is_set():
            # ── Wait for a new audio_emb signal ─────────────────────────
            try:
                _ = await asyncio.wait_for(
                    session.emb_queue.get(), timeout=1.0
                )
                emb_chunks_received += 1
            except asyncio.TimeoutError:
                continue

            # ── Check minimum buffer ────────────────────────────────────
            async with session.emb_lock:
                n_chunks = len(session.audio_emb_buffer)

            if n_chunks < MIN_EMB_FRAMES_TO_START:
                continue

            # ── Copy buffer snapshot (thread-safe read) ──────────────────
            async with session.emb_lock:
                buf_snapshot = list(session.audio_emb_buffer)

            # ── Run AF block in thread (GPU-heavy) ───────────────────────
            try:
                frames: List[np.ndarray] = await asyncio.to_thread(
                    streaming_af.generate_block,
                    buf_snapshot,
                )
            except Exception as e:
                logger.error(f"[AFLoop] generate_block failed: {e}", exc_info=True)
                continue

            if frames is None:
                continue

            # ── Push frames to frame_queue ────────────────────────────────
            block_seq = session.current_seq  # approximate seq for this block
            for i, frame_np in enumerate(frames):
                tagged = TaggedFrame(seq=block_seq + i, frame_np=frame_np)

                # Backpressure: drop oldest if full
                if session.frame_queue.full():
                    try:
                        session.frame_queue.get_nowait()
                        logger.debug("[AFLoop] Frame queue full — dropped oldest frame.")
                    except asyncio.QueueEmpty:
                        pass

                try:
                    session.frame_queue.put_nowait(tagged)
                except asyncio.QueueFull:
                    pass

            logger.debug(
                f"[AFLoop] Generated {len(frames)} frames, "
                f"total={streaming_af.frames_generated}, "
                f"frame_q={session.frame_queue.qsize()}"
            )

            await asyncio.sleep(0)

    except asyncio.CancelledError:
        pass
    except Exception as e:
        logger.error(f"[AFLoop] Fatal error: {e}", exc_info=True)
        session.error_event.set()
    finally:
        logger.info(f"[AFLoop/{session.session_id[:8]}] Stopped.")
