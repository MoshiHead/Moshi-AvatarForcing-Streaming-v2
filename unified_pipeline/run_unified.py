"""
unified_pipeline/run_unified.py
=================================
CLI entry point for the unified Moshi + Bridge + AvatarForcing pipeline.

Usage
-----
python -m unified_pipeline.run_unified \\
    --user-audio  input.wav \\
    --image       face.jpg \\
    --output      output.mp4 \\
    --bridge-ckpt checkpoints/bridge_best.pt \\
    --af-ckpt     checkpoints/ode_audio_init.pt \\
    --af-config   AvatarForcing-inference/configs/avatarforcing.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch


def parse_args():
    parser = argparse.ArgumentParser(
        description="Unified Moshi + Bridge + AvatarForcing Talking-Head Pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── Inputs ────────────────────────────────────────────────────────────────
    parser.add_argument("--user-audio", required=True,
                        help="Path to user input audio (.wav / .mp3)")
    parser.add_argument("--image",      required=True,
                        help="Path to reference face image (.jpg / .png)")
    parser.add_argument("--output",     default="output.mp4",
                        help="Output .mp4 path")

    # ── Checkpoints ───────────────────────────────────────────────────────────
    parser.add_argument("--bridge-ckpt",   required=True,
                        help="moshi-wav2vec-bridge checkpoint (.pt)")
    parser.add_argument("--bridge-config", default=None,
                        help="bridge config.yaml (auto-detected if not set)")
    parser.add_argument("--af-ckpt",       required=True,
                        help="AvatarForcing generator checkpoint (.pt)")
    parser.add_argument("--af-config",
                        default="AvatarForcing-inference/configs/avatarforcing.yaml",
                        help="AvatarForcing config yaml")

    # ── Moshi model ───────────────────────────────────────────────────────────
    parser.add_argument("--moshi-repo",     default="kyutai/moshiko-pytorch-q8")
    parser.add_argument("--moshi-weight",   default=None)
    parser.add_argument("--mimi-weight",    default=None)
    parser.add_argument("--moshi-tokenizer",default=None)

    # ── Generation ────────────────────────────────────────────────────────────
    parser.add_argument("--teacher-len",       type=int, default=80,
                        help="AvatarForcing teacher_len (must be even)")
    parser.add_argument("--num-output-frames", type=int, default=21)
    parser.add_argument("--fps",               type=int, default=25)
    parser.add_argument("--prompt",            default=None)

    # ── Runtime ───────────────────────────────────────────────────────────────
    parser.add_argument("--device",  default="cuda")
    parser.add_argument("--use-ema", action="store_true",
                        help="Load EMA weights from AvatarForcing checkpoint")
    parser.add_argument("--half",    action="store_true",
                        help="Use float16 instead of bfloat16")
    parser.add_argument("--seed",    type=int, default=0)

    return parser.parse_args()


def main():
    args = parse_args()
    dtype = torch.float16 if args.half else torch.bfloat16

    import random
    import numpy as np
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    # ── Load AvatarForcing config with OmegaConf (same as inference.py) ──────
    from omegaconf import OmegaConf
    af_cfg_path     = args.af_config
    default_cfg_path = str(Path(af_cfg_path).parent / "default_config.yaml")

    af_config = OmegaConf.load(af_cfg_path)
    if Path(default_cfg_path).exists():
        default_cfg = OmegaConf.load(default_cfg_path)
        af_config   = OmegaConf.merge(default_cfg, af_config)

    # Override teacher_len from CLI so bridge and pipeline are in sync
    OmegaConf.update(af_config, "data.teacher_len", args.teacher_len, merge=True)

    # ── Build unified pipeline (loads all models) ─────────────────────────────
    # Add project root so unified_pipeline package resolves
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from unified_pipeline.pipeline import UnifiedMoshiAvatarPipeline

    pipeline = UnifiedMoshiAvatarPipeline(
        bridge_checkpoint  = args.bridge_ckpt,
        generator_ckpt     = args.af_ckpt,
        af_config          = af_config,
        bridge_config      = args.bridge_config,
        moshi_hf_repo      = args.moshi_repo,
        moshi_weight       = args.moshi_weight,
        mimi_weight        = args.mimi_weight,
        moshi_tokenizer    = args.moshi_tokenizer,
        teacher_len        = args.teacher_len,
        num_output_frames  = args.num_output_frames,
        device             = args.device,
        default_prompt     = args.prompt,
        dtype              = dtype,
        use_ema            = args.use_ema,
    )

    # ── Run ───────────────────────────────────────────────────────────────────
    with torch.no_grad():
        out = pipeline.run(
            user_audio_path = args.user_audio,
            image_path      = args.image,
            output_path     = args.output,
            prompt          = args.prompt,
            fps             = args.fps,
        )
    print(f"\n✅ Saved to: {out}")


if __name__ == "__main__":
    main()
