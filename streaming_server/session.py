"""
streaming_server/session.py
=============================
Per-WebSocket-session isolated state container.

Every connected client gets its own SessionState with:
  - Isolated asyncio queues for each pipeline stage
  - Isolated bridge KV-cache (via StreamingBridge instance)
  - Isolated AvatarForcing state (via StreamingAvatarForcing instance)
  - Monotonic sequence counter
  - Background task registry + cleanup
  - Error signalling event

Session lifecycle:
  1. POST /session/start → create SessionState
  2. POST /session/{id}/image → call af_runner.start_session(image_path)
  3. WS /ws/{id} connect → start all background tasks
  4. WS disconnect → set error_event, cancel all tasks, cleanup
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# ── Queue size limits ──────────────────────────────────────────────────────────
AUDIO_QUEUE_SIZE  =  8    # raw PCM chunks from browser (~640ms buffer)
TOKEN_QUEUE_SIZE  = 16    # Moshi acoustic tokens
EMB_QUEUE_SIZE    = 32    # bridge audio_emb chunks
FRAME_QUEUE_SIZE  = 30    # AvatarForcing raw RGB frames (≈1.2s at 25fps)
SEND_QUEUE_SIZE   = 60    # encoded JPEG + audio packets ready to send


@dataclass
class SessionState:
    """
    All state for a single client WebSocket session.

    Instantiate via SessionState.create().
    """

    session_id:     str
    image_path:     Optional[str]

    # ── asyncio queues (pipeline stages) ──────────────────────────────────────
    audio_queue:    asyncio.Queue   # (1, FRAME_SIZE) float32 PCM from browser
    token_queue:    asyncio.Queue   # TaggedToken
    emb_queue:      asyncio.Queue   # TaggedEmb
    frame_queue:    asyncio.Queue   # TaggedFrame  (raw RGB)
    send_queue:     asyncio.Queue   # bytes  (packed protocol messages for WS send)

    # ── synchronisation primitives ────────────────────────────────────────────
    ws_write_lock:  asyncio.Lock    # serialises direct priority audio writes
    error_event:    asyncio.Event   # set on WS disconnect / fatal error
    image_ready:    asyncio.Event   # set once image uploaded + AF session started

    # ── sequence counter ──────────────────────────────────────────────────────
    _seq:           int = field(default=0, repr=False)

    # ── background tasks ──────────────────────────────────────────────────────
    tasks:          List[asyncio.Task] = field(default_factory=list, repr=False)

    # ── shared rolling audio_emb buffer (for AvatarForcing) ──────────────────
    # Protected by emb_lock; AvatarForcing loop reads this, bridge loop appends
    audio_emb_buffer: list = field(default_factory=list, repr=False)
    emb_lock:         asyncio.Lock = field(default=None, repr=False)

    # ── pipeline adapters (set by server at session init) ─────────────────────
    streaming_bridge: object = field(default=None, repr=False)
    streaming_af:     object = field(default=None, repr=False)

    @classmethod
    def create(cls, session_id: Optional[str] = None) -> "SessionState":
        """Factory: creates a fresh session with all queues and locks."""
        sid = session_id or str(uuid.uuid4())
        return cls(
            session_id    = sid,
            image_path    = None,
            audio_queue   = asyncio.Queue(maxsize=AUDIO_QUEUE_SIZE),
            token_queue   = asyncio.Queue(maxsize=TOKEN_QUEUE_SIZE),
            emb_queue     = asyncio.Queue(maxsize=EMB_QUEUE_SIZE),
            frame_queue   = asyncio.Queue(maxsize=FRAME_QUEUE_SIZE),
            send_queue    = asyncio.Queue(maxsize=SEND_QUEUE_SIZE),
            ws_write_lock = asyncio.Lock(),
            error_event   = asyncio.Event(),
            image_ready   = asyncio.Event(),
            emb_lock      = asyncio.Lock(),
        )

    def next_seq(self) -> int:
        """Return the next monotonic sequence ID (NOT thread-safe, asyncio-safe)."""
        self._seq += 1
        return self._seq

    @property
    def current_seq(self) -> int:
        return self._seq

    async def cleanup(self) -> None:
        """Cancel all background tasks and clear queues."""
        self.error_event.set()
        for task in self.tasks:
            if not task.done():
                task.cancel()

        if self.tasks:
            await asyncio.gather(*self.tasks, return_exceptions=True)
        self.tasks.clear()

        # Drain queues to unblock any awaiting coroutines
        for q in [self.audio_queue, self.token_queue, self.emb_queue,
                  self.frame_queue, self.send_queue]:
            while not q.empty():
                try:
                    q.get_nowait()
                except asyncio.QueueEmpty:
                    break

        # Reset AF adapter if active
        if self.streaming_af is not None:
            try:
                self.streaming_af.end_session()
            except Exception:
                pass

        if self.streaming_bridge is not None:
            try:
                self.streaming_bridge.reset_session()
            except Exception:
                pass

        logger.info(f"[Session {self.session_id}] Cleaned up.")


# ── Global session registry ────────────────────────────────────────────────────

class SessionRegistry:
    """Thread-safe (asyncio-safe) session registry."""

    def __init__(self):
        self._sessions: Dict[str, SessionState] = {}

    def create(self) -> SessionState:
        state = SessionState.create()
        self._sessions[state.session_id] = state
        logger.info(f"[Registry] Created session {state.session_id}")
        return state

    def get(self, session_id: str) -> Optional[SessionState]:
        return self._sessions.get(session_id)

    async def remove(self, session_id: str) -> None:
        state = self._sessions.pop(session_id, None)
        if state:
            await state.cleanup()
            logger.info(f"[Registry] Removed session {session_id}")

    def __len__(self) -> int:
        return len(self._sessions)

    def active_count(self) -> int:
        return len(self._sessions)
