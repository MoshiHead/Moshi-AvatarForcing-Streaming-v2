"""
streaming_server/server.py
============================
Main FastAPI + WebSocket server for the real-time streaming AI avatar system.

Endpoints:
  GET  /                          → serve static/index.html (browser UI)
  GET  /static/{path}             → serve static assets
  GET  /health                    → server health + session count
  POST /session/start             → create session, return {session_id}
  POST /session/{id}/image        → upload face image, init AF session
  GET  /session/{id}/status       → session status
  WS   /ws/{session_id}           → main streaming WebSocket

Startup:
  - Load all models (Moshi, Bridge, AvatarForcing) ONCE
  - Create shared CUDA streams for pipeline parallelism
  - Run on port 7865 for RunPod compatibility

WebSocket session lifecycle:
  1. Client connects → start 6 concurrent background tasks
  2. Client streams audio → Moshi → Bridge → AvatarForcing → client gets A+V
  3. Client disconnects → tasks cancelled, session cleaned up
"""

from __future__ import annotations

import asyncio
import logging
import os
import shutil
import sys
import tempfile
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Dict, Optional

import torch
import uvicorn
from fastapi import FastAPI, File, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from omegaconf import OmegaConf

# ── Resolve project paths ──────────────────────────────────────────────────────
_ROOT       = Path(__file__).resolve().parent.parent
_MOSHI_ROOT = _ROOT / "moshi-inference"
_BRIDGE_ROOT= _ROOT / "moshi-wav2vec-bridge"
_AF_ROOT    = _ROOT / "AvatarForcing-inference"

for _p in [str(_ROOT), str(_MOSHI_ROOT), str(_BRIDGE_ROOT), str(_AF_ROOT)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from unified_pipeline.moshi_runner    import MoshiRunner
from model                             import MimiWav2Vec2Bridge
from pipeline.avatar_forcing_inference import AvatarForcingInferencePipeline
from utils.inject                      import _apply_lora
from collections                       import OrderedDict

from .session import SessionRegistry, SessionState
from .pipeline.streaming_moshi       import StreamingMoshi
from .pipeline.streaming_bridge      import StreamingBridge
from .pipeline.streaming_avatarforcing import StreamingAvatarForcing
from .utils.jpeg_encoder             import JpegEncoder
from .tasks.audio_receive            import audio_receive_task
from .tasks.moshi_loop               import moshi_loop_task
from .tasks.bridge_loop              import bridge_loop_task
from .tasks.avatarforcing_loop       import avatarforcing_loop_task
from .tasks.frame_encode_loop        import frame_encode_loop_task
from .tasks.keepalive_loop           import keepalive_loop_task

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level  = logging.INFO,
    format = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    datefmt= "%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Configuration (loaded from environment or defaults) ───────────────────────
CFG = {
    "BRIDGE_CKPT"   : os.environ.get("BRIDGE_CKPT",    "/workspace/Moshi-AvatarForcing-bridge-v4/checkpoints/bridge_best.pt"),
    "BRIDGE_CONFIG" : os.environ.get("BRIDGE_CONFIG",  str(_BRIDGE_ROOT / "config.yaml")),
    "AF_CKPT"       : os.environ.get("AF_CKPT",        "/workspace/Moshi-AvatarForcing-bridge-v4/checkpoints/model.pt"),
    "AF_CONFIG"     : os.environ.get("AF_CONFIG",      str(_AF_ROOT / "configs/avatarforcing.yaml")),
    "MOSHI_HF_REPO" : os.environ.get("MOSHI_HF_REPO",  "kyutai/moshiko-pytorch-q8"),
    "DEVICE"        : os.environ.get("DEVICE",          "cuda"),
    "PROMPT"        : os.environ.get("AF_PROMPT", (
        "A person talking naturally, realistic facial expressions, "
        "high quality video, detailed face."
    )),
    "PORT"          : int(os.environ.get("PORT", 7865)),
    "HOST"          : os.environ.get("HOST", "0.0.0.0"),
    "TEACHER_LEN"   : int(os.environ.get("TEACHER_LEN", 80)),
    "USE_EMA"       : os.environ.get("USE_EMA", "false").lower() == "true",
}

# ── Global singletons (loaded once at startup) ─────────────────────────────────
_moshi_runner:    Optional[MoshiRunner]                  = None
_bridge_model:    Optional[MimiWav2Vec2Bridge]           = None
_af_pipeline:     Optional[AvatarForcingInferencePipeline] = None
_jpeg_encoder:    Optional[JpegEncoder]                  = None
_bridge_stream:   Optional[torch.cuda.Stream]            = None
_af_stream:       Optional[torch.cuda.Stream]            = None
_session_registry = SessionRegistry()
_upload_dir       = Path(tempfile.mkdtemp(prefix="af_uploads_"))


# ── Model loading ─────────────────────────────────────────────────────────────

def _load_all_models():
    global _moshi_runner, _bridge_model, _af_pipeline, _jpeg_encoder
    global _bridge_stream, _af_stream

    device = torch.device(CFG["DEVICE"])
    dtype  = torch.bfloat16

    logger.info("=" * 60)
    logger.info("  Real-Time Streaming AI Avatar Server")
    logger.info("  Loading all models …")
    logger.info("=" * 60)

    # ── CUDA streams ──────────────────────────────────────────────────────────
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        _bridge_stream = torch.cuda.Stream(device=device)
        _af_stream     = torch.cuda.Stream(device=device)
        logger.info("[Server] CUDA streams created (bridge + af).")

    # ── Moshi ─────────────────────────────────────────────────────────────────
    logger.info("[Server] Loading Moshi …")
    _moshi_runner = MoshiRunner(
        hf_repo = CFG["MOSHI_HF_REPO"],
        device  = CFG["DEVICE"],
        dtype   = dtype,
    )
    _moshi_runner.load()
    logger.info("[Server] ✅ Moshi loaded.")

    # ── Bridge ────────────────────────────────────────────────────────────────
    logger.info("[Server] Loading Bridge …")
    import yaml
    with open(CFG["BRIDGE_CONFIG"]) as f:
        bridge_cfg = yaml.safe_load(f)
    _bridge_model = MimiWav2Vec2Bridge(bridge_cfg).to(device)

    # Load checkpoint
    try:
        ckpt = torch.load(CFG["BRIDGE_CKPT"], map_location=device, weights_only=True)
    except TypeError:
        ckpt = torch.load(CFG["BRIDGE_CKPT"], map_location=device)
    sd = ckpt.get("bridge", ckpt)
    missing, unexpected = _bridge_model.load_state_dict(sd, strict=False)
    if missing:     logger.warning(f"[Bridge] Missing keys: {missing}")
    if unexpected:  logger.warning(f"[Bridge] Unexpected keys: {unexpected}")
    _bridge_model.eval()
    logger.info("[Server] ✅ Bridge loaded.")

    # ── AvatarForcing ─────────────────────────────────────────────────────────
    logger.info("[Server] Loading AvatarForcing …")
    default_cfg_path = Path(CFG["AF_CONFIG"]).parent / "default_config.yaml"
    af_config = OmegaConf.load(CFG["AF_CONFIG"])
    if default_cfg_path.exists():
        default_cfg = OmegaConf.load(str(default_cfg_path))
        af_config   = OmegaConf.merge(default_cfg, af_config)
    OmegaConf.update(af_config, "data.teacher_len", CFG["TEACHER_LEN"], merge=True)

    _af_pipeline = AvatarForcingInferencePipeline(af_config, device=device)

    state_dict = torch.load(CFG["AF_CKPT"], map_location="cpu", weights_only=False)
    if CFG["USE_EMA"]:
        sd = state_dict["generator_ema"]
        clean = OrderedDict()
        for k, v in sd.items():
            clean[k.replace("_fsdp_wrapped_module.", "")] = v
        sd = clean
    else:
        sd = state_dict.get("generator", state_dict)

    if hasattr(af_config, "models") and hasattr(af_config.models, "lora"):
        _af_pipeline.generator.model = _apply_lora(
            _af_pipeline.generator.model, af_config.models.lora
        )

    _af_pipeline.generator.load_state_dict(sd, strict=False)
    _af_pipeline = _af_pipeline.to(device=device, dtype=dtype)
    logger.info("[Server] ✅ AvatarForcing loaded.")

    # ── JPEG Encoder ──────────────────────────────────────────────────────────
    _jpeg_encoder = JpegEncoder(quality=85)
    logger.info(f"[Server] ✅ JPEG encoder: {_jpeg_encoder.backend}")

    logger.info("=" * 60)
    logger.info("  All models loaded. Server ready.")
    logger.info(f"  Running on port {CFG['PORT']}")
    logger.info("=" * 60)


# ── FastAPI lifespan ──────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: load models in a thread (blocking GPU calls)
    await asyncio.to_thread(_load_all_models)
    yield
    # Shutdown: cleanup sessions
    logger.info("[Server] Shutting down …")
    for sid in list(_session_registry._sessions.keys()):
        await _session_registry.remove(sid)


# ── FastAPI app ───────────────────────────────────────────────────────────────

app = FastAPI(
    title       = "Real-Time Streaming AI Avatar Server",
    description = "Moshi + Bridge + AvatarForcing continuous streaming pipeline",
    version     = "1.0.0",
    lifespan    = lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins     = ["*"],
    allow_credentials = True,
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)

# Serve static frontend files
_STATIC_DIR = _ROOT / "static"
if _STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")


# ── HTTP Endpoints ────────────────────────────────────────────────────────────

@app.get("/", response_class=FileResponse)
async def index():
    """Serve the browser UI."""
    idx = _STATIC_DIR / "index.html"
    if idx.exists():
        return FileResponse(str(idx))
    return JSONResponse({"message": "AI Avatar Streaming Server", "status": "running"})


@app.get("/health")
async def health():
    """Liveness check."""
    return {
        "status"          : "ok",
        "models_loaded"   : _moshi_runner is not None,
        "active_sessions" : _session_registry.active_count(),
        "device"          : CFG["DEVICE"],
        "port"            : CFG["PORT"],
    }


@app.post("/session/start")
async def create_session():
    """Create a new streaming session. Returns {session_id}."""
    if _moshi_runner is None:
        raise HTTPException(503, "Models not loaded yet.")
    state = _session_registry.create()
    return {"session_id": state.session_id, "status": "created"}


@app.post("/session/{session_id}/image")
async def upload_image(session_id: str, file: UploadFile = File(...)):
    """
    Upload reference face image and initialise the AvatarForcing session.
    Must be called before opening the WebSocket.
    """
    state = _session_registry.get(session_id)
    if state is None:
        raise HTTPException(404, f"Session {session_id} not found.")

    # Save uploaded image
    suffix = Path(file.filename).suffix or ".jpg"
    img_path = _upload_dir / f"{session_id}{suffix}"
    with open(img_path, "wb") as f:
        shutil.copyfileobj(file.file, f)
    state.image_path = str(img_path)

    # Create per-session AvatarForcing adapter and start session
    device = torch.device(CFG["DEVICE"])
    af_adapter = StreamingAvatarForcing(
        pipeline    = _af_pipeline,
        device      = device,
        dtype       = torch.bfloat16,
        cuda_stream = _af_stream,
        prompt      = CFG["PROMPT"],
    )

    try:
        await asyncio.to_thread(af_adapter.start_session, str(img_path))
    except Exception as e:
        logger.error(f"[Server] AF session start failed: {e}", exc_info=True)
        raise HTTPException(500, f"AvatarForcing init failed: {e}")

    state.streaming_af = af_adapter
    state.image_ready.set()

    logger.info(f"[Server] Session {session_id[:8]} image uploaded + AF ready.")
    return {"status": "ready", "session_id": session_id}


@app.get("/session/{session_id}/status")
async def session_status(session_id: str):
    """Get session status."""
    state = _session_registry.get(session_id)
    if state is None:
        raise HTTPException(404, f"Session {session_id} not found.")
    return {
        "session_id"     : session_id,
        "image_ready"    : state.image_ready.is_set(),
        "error"          : state.error_event.is_set(),
        "seq"            : state.current_seq,
        "audio_queue"    : state.audio_queue.qsize(),
        "token_queue"    : state.token_queue.qsize(),
        "emb_queue"      : state.emb_queue.qsize(),
        "frame_queue"    : state.frame_queue.qsize(),
        "send_queue"     : state.send_queue.qsize(),
        "emb_buf_chunks" : len(state.audio_emb_buffer),
        "frames_generated": (
            state.streaming_af.frames_generated
            if state.streaming_af else 0
        ),
    }


# ── WebSocket Endpoint ────────────────────────────────────────────────────────

@app.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    """
    Main real-time streaming WebSocket.

    Starts 6 concurrent background tasks:
      1. audio_receive  — browser mic → audio_queue
      2. moshi_loop     — audio_queue → Moshi → token_queue + priority audio send
      3. bridge_loop    — token_queue → Bridge(KV) → audio_emb_buffer
      4. af_loop        — audio_emb_buffer → AvatarForcing → frame_queue
      5. frame_encode   — frame_queue → JPEG → send_queue
      6. keepalive      — send_queue → WS + periodic 0xFE pings
    """
    state = _session_registry.get(session_id)
    if state is None:
        await websocket.close(code=4004, reason="Session not found")
        return

    await websocket.accept()
    logger.info(f"[WS] Client connected: session={session_id[:8]}")

    # ── Create per-session pipeline adapters ─────────────────────────────────
    device = torch.device(CFG["DEVICE"])

    # Moshi adapter (per-session InferenceState)
    sm = StreamingMoshi(_moshi_runner)
    sm.reset_session()

    # Bridge adapter (per-session KV-cache)
    sb = StreamingBridge(
        model       = _bridge_model,
        device      = device,
        cuda_stream = _bridge_stream,
    )
    sb.reset_session()
    state.streaming_bridge = sb

    try:
        # ── Launch all background tasks ─────────────────────────────────────
        tasks = [
            asyncio.create_task(
                audio_receive_task(websocket, state),
                name=f"audio_recv_{session_id[:8]}"
            ),
            asyncio.create_task(
                moshi_loop_task(websocket, state, sm),
                name=f"moshi_{session_id[:8]}"
            ),
            asyncio.create_task(
                bridge_loop_task(state, sb),
                name=f"bridge_{session_id[:8]}"
            ),
            asyncio.create_task(
                # ALWAYS create the AF task — it waits internally for image_ready
                # and reads state.streaming_af dynamically after the event fires.
                # Do NOT use _noop() here — it returns immediately, causing
                # asyncio.wait(FIRST_COMPLETED) to kill all other tasks instantly.
                avatarforcing_loop_task(state),
                name=f"af_{session_id[:8]}"
            ),
            asyncio.create_task(
                frame_encode_loop_task(state, _jpeg_encoder),
                name=f"encode_{session_id[:8]}"
            ),
            asyncio.create_task(
                keepalive_loop_task(websocket, state),
                name=f"keepalive_{session_id[:8]}"
            ),
        ]
        state.tasks = tasks

        # ── Wait for any task to finish (error or disconnect) ───────────────
        done, pending = await asyncio.wait(
            tasks, return_when=asyncio.FIRST_COMPLETED
        )

        for t in done:
            if t.exception():
                logger.error(f"[WS] Task {t.get_name()} raised: {t.exception()}")

    except WebSocketDisconnect:
        logger.info(f"[WS] Client disconnected: {session_id[:8]}")
    except Exception as e:
        logger.error(f"[WS] Session error: {e}", exc_info=True)
    finally:
        state.error_event.set()
        # Reset Moshi streaming contexts so reconnect doesn't fail
        try:
            sm.close_session()
        except Exception:
            pass
        # Cancel remaining tasks
        for t in state.tasks:
            if not t.done():
                t.cancel()
        if state.tasks:
            await asyncio.gather(*state.tasks, return_exceptions=True)

        logger.info(f"[WS] Session {session_id[:8]} WebSocket closed.")


async def _wait_for_stop(stop_event: asyncio.Event):
    """Persistent waiter — only exits when the session stops.
    Use this instead of _noop() to avoid firing FIRST_COMPLETED immediately."""
    await stop_event.wait()


# ── Entry point ───────────────────────────────────────────────────────────────

def run_server():
    """Start the uvicorn server. Call this from notebook or CLI."""
    uvicorn.run(
        "streaming_server.server:app",
        host        = CFG["HOST"],
        port        = CFG["PORT"],
        log_level   = "info",
        ws_ping_interval   = 20,
        ws_ping_timeout    = 60,
        timeout_keep_alive = 30,
    )


if __name__ == "__main__":
    run_server()
