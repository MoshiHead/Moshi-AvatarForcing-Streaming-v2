"""
streaming_server/utils/protocol.py
=====================================
Binary WebSocket protocol constants and pack/unpack helpers.

Server → Client wire format:
  [0x01] [seq_id: 4 bytes big-endian uint32] [pcm_bytes: int16 LE]   — audio
  [0x02] [seq_id: 4 bytes big-endian uint32] [jpeg_bytes]            — video frame
  [0x03] [text_token: 4 bytes big-endian int32]                       — text token (optional)
  [0xFE]                                                              — keepalive ping

Client → Server wire format:
  [0x01] [pcm_bytes: int16 LE, 24kHz mono, FRAME_SIZE samples]       — audio chunk
"""

from __future__ import annotations

import struct
from typing import Tuple

import numpy as np
import torch

# ── Protocol byte constants ──────────────────────────────────────────────────
MSG_AUDIO   = 0x01   # PCM audio (server→client and client→server)
MSG_VIDEO   = 0x02   # JPEG frame (server→client)
MSG_TEXT    = 0x03   # text token (server→client, optional)
MSG_KEEP    = 0xFE   # keepalive ping (server→client)

# ── Audio constants ──────────────────────────────────────────────────────────
SAMPLE_RATE  = 24_000   # Hz — Moshi / Mimi sample rate
FRAME_RATE   = 12.5     # Hz — Moshi LM frame rate
FRAME_SIZE   = int(SAMPLE_RATE / FRAME_RATE)   # = 1920 samples per Moshi step


# ── Pack helpers (server → client) ──────────────────────────────────────────

def pack_audio(seq: int, pcm: torch.Tensor) -> bytes:
    """
    Pack a PCM chunk for transmission.

    Parameters
    ----------
    seq : uint32 sequence id
    pcm : (1, N) or (N,) float32 tensor in [-1, 1]

    Returns
    -------
    bytes: [0x01][seq:4B][pcm as int16 LE]
    """
    pcm_np = pcm.squeeze().float().numpy()
    # Clip and convert to int16
    pcm_int16 = (pcm_np.clip(-1.0, 1.0) * 32767).astype(np.int16)
    header = struct.pack(">BI", MSG_AUDIO, seq & 0xFFFFFFFF)
    return header + pcm_int16.tobytes()


def pack_video(seq: int, jpeg_bytes: bytes) -> bytes:
    """
    Pack a JPEG frame for transmission.

    Parameters
    ----------
    seq        : uint32 sequence id
    jpeg_bytes : JPEG-encoded frame bytes

    Returns
    -------
    bytes: [0x02][seq:4B][jpeg_bytes]
    """
    header = struct.pack(">BI", MSG_VIDEO, seq & 0xFFFFFFFF)
    return header + jpeg_bytes


def pack_text(text_token: int) -> bytes:
    """Pack a text token: [0x03][token:4B]"""
    return struct.pack(">Bi", MSG_TEXT, text_token)


def pack_keepalive() -> bytes:
    """Pack a keepalive: [0xFE]"""
    return bytes([MSG_KEEP])


# ── Unpack helpers (client → server) ─────────────────────────────────────────

def unpack_client_audio(data: bytes) -> Tuple[int, torch.Tensor]:
    """
    Unpack a client audio message.

    Parameters
    ----------
    data : raw bytes from WebSocket (must start with 0x01)

    Returns
    -------
    (msg_type, pcm_tensor)
    msg_type    : the message type byte (should be MSG_AUDIO = 0x01)
    pcm_tensor  : (1, N) float32 — normalised PCM in [-1, 1]
    """
    if len(data) < 1:
        raise ValueError("Empty message")
    msg_type = data[0]
    if msg_type != MSG_AUDIO:
        raise ValueError(f"Expected MSG_AUDIO (0x01), got 0x{msg_type:02X}")

    pcm_int16 = np.frombuffer(data[1:], dtype=np.int16)
    pcm_float = pcm_int16.astype(np.float32) / 32768.0
    return msg_type, torch.from_numpy(pcm_float).unsqueeze(0)  # (1, N)
