"""
streaming_server/utils/sync_types.py
======================================
Tagged data containers that carry monotonic sequence IDs
through the entire pipeline: Moshi → Bridge → AvatarForcing → Browser.

Sequence ID alignment enables browser-side lip-sync:
  audio packet  seq=N  ←→  video frame seq=N
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch


@dataclass
class TaggedToken:
    """
    One Moshi LM step's acoustic tokens, tagged with a sequence ID.

    Attributes
    ----------
    seq          : monotonic integer, assigned by the Moshi loop
    audio_tokens : (1, 8) int64  — 8 response codebook indices
    text_token   : int           — SentencePiece text token id
    pcm_chunk    : (1, frame_size) float32 — decoded Mimi PCM (sent directly to browser)
    """
    seq:          int
    audio_tokens: torch.Tensor          # (1, 8)
    text_token:   int
    pcm_chunk:    torch.Tensor          # (1, frame_size) float32


@dataclass
class TaggedEmb:
    """
    Bridge output: one chunk of audio_emb, tagged with the starting sequence ID.

    Attributes
    ----------
    seq_start    : sequence ID of the first token in this chunk
    seq_end      : sequence ID of the last token in this chunk (exclusive)
    audio_emb    : (T_out, 10752) float32 — bridge output, ready for AvatarForcing
                   T_out = 2 * num_tokens_processed (bridge upsamples ×2)
    """
    seq_start:  int
    seq_end:    int
    audio_emb:  torch.Tensor            # (T_out, 10752)


@dataclass
class TaggedFrame:
    """
    One AvatarForcing video frame, tagged with the sequence ID of the
    audio tokens that conditioned its generation.

    Attributes
    ----------
    seq          : sequence ID linking back to the originating audio tokens
    frame_np     : (H, W, 3) uint8 numpy array — RGB frame
    """
    seq:        int
    frame_np:   np.ndarray              # (H, W, 3) uint8 RGB


@dataclass
class SessionCounters:
    """Shared monotonic counters for a session."""
    _seq: int = field(default=0, init=False)

    def next_seq(self) -> int:
        self._seq += 1
        return self._seq

    @property
    def current_seq(self) -> int:
        return self._seq
