#!/usr/bin/env python3
"""
start_server.py
================
Standalone launcher for the streaming server.

Fixes the Jupyter "event loop already running" issue by running
uvicorn in a clean subprocess (not in the notebook's event loop).

Usage from notebook:
    !python start_server.py

Usage from terminal:
    python start_server.py
"""

import os
import sys
from pathlib import Path

# ── Auto-detect workspace ──────────────────────────────────────────────────────
# Try common RunPod workspace names
_CANDIDATES = [
    "/workspace/Moshi-AvatarForcing-Streaming-v2",
    "/workspace/Moshi-AvatarForcing-bridge-v4",
    "/workspace/Moshi-AvatarForcing-bridge-v4-try-update-streaming-v2",
    str(Path(__file__).resolve().parent),  # directory containing this script
]
WORKSPACE = None
for c in _CANDIDATES:
    if Path(c).exists() and Path(c, "streaming_server").exists():
        WORKSPACE = c
        break
if WORKSPACE is None:
    WORKSPACE = str(Path(__file__).resolve().parent)

print(f"[start_server] Workspace: {WORKSPACE}")
os.chdir(WORKSPACE)

# ── Add all paths ─────────────────────────────────────────────────────────────
for p in [
    WORKSPACE,
    f"{WORKSPACE}/moshi-inference",
    f"{WORKSPACE}/moshi-wav2vec-bridge",
    f"{WORKSPACE}/AvatarForcing-inference",
]:
    if p not in sys.path:
        sys.path.insert(0, p)

# ── Set defaults (overridden by env vars) ─────────────────────────────────────
os.environ.setdefault("BRIDGE_CKPT",    f"{WORKSPACE}/checkpoints/bridge_best.pt")
os.environ.setdefault("BRIDGE_CONFIG",  f"{WORKSPACE}/moshi-wav2vec-bridge/config.yaml")
os.environ.setdefault("AF_CKPT",        f"{WORKSPACE}/checkpoints/model.pt")
os.environ.setdefault("AF_CONFIG",      f"{WORKSPACE}/AvatarForcing-inference/configs/avatarforcing.yaml")
os.environ.setdefault("MOSHI_HF_REPO",  "kyutai/moshiko-pytorch-q8")
os.environ.setdefault("DEVICE",         "cuda")
os.environ.setdefault("PORT",           "7865")
os.environ.setdefault("HOST",           "0.0.0.0")
os.environ.setdefault("TEACHER_LEN",    "80")
os.environ.setdefault("USE_EMA",        "false")
os.environ.setdefault("AF_PROMPT", (
    "A person talking naturally, realistic facial expressions, "
    "high quality video, detailed face."
))

# ── Launch ────────────────────────────────────────────────────────────────────
print("=" * 60)
print("  Real-Time AI Avatar Streaming Server")
print(f"  Port: {os.environ['PORT']}")
print("  Loading models (may take 1–3 min) …")
print("=" * 60)

import uvicorn

uvicorn.run(
    "streaming_server.server:app",
    host               = os.environ["HOST"],
    port               = int(os.environ["PORT"]),
    log_level          = "info",
    ws_ping_interval   = 20,
    ws_ping_timeout    = 60,
    timeout_keep_alive = 30,
)
