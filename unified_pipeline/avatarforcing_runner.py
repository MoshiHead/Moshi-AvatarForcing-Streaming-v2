"""
unified_pipeline/avatarforcing_runner.py
==========================================
Wraps AvatarForcing's pipeline and accepts bridge audio_emb directly,
bypassing the wav2vec encoder entirely.

Mirrors AvatarForcing inference.py EXACTLY:
  - Same image preprocessing (ResizeKeepRatioArea16 + Normalize)
  - Same VAE encode + latent building
  - Same noise sampling
  - Same inference_avatar_forcing() call signature
  - Same video decoding

Only difference: audio_emb comes from BridgeRunner, not wav2vec.
"""

from __future__ import annotations

import math
import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from PIL import Image
from einops import rearrange
from torchvision.transforms import InterpolationMode
from collections import OrderedDict

# Add AvatarForcing root so all its internal imports resolve
AF_ROOT = Path(__file__).resolve().parent.parent / "AvatarForcing-inference"
if str(AF_ROOT) not in sys.path:
    sys.path.insert(0, str(AF_ROOT))

from pipeline.avatar_forcing_inference import AvatarForcingInferencePipeline  # noqa: E402
from utils.inject import _apply_lora                                           # noqa: E402


# ─── Image transform — copied exactly from AvatarForcing inference.py ─────────

class ResizeKeepRatioArea16:
    """Resize image so pixel area ≈ target_area, keeping dims divisible by 16."""
    def __init__(self, area_hw=(480, 832), div=16):
        self.A = area_hw[0] * area_hw[1]
        self.d = div

    def __call__(self, img: Image.Image) -> Image.Image:
        w, h = img.size
        s  = min(1.0, math.sqrt(self.A / (h * w)))
        nh = max(self.d, int(h * s) // self.d * self.d)
        nw = max(self.d, int(w * s) // self.d * self.d)
        return TF.resize(img, (nh, nw),
                         interpolation=InterpolationMode.BILINEAR, antialias=True)


_IMAGE_TRANSFORM = transforms.Compose([
    ResizeKeepRatioArea16((480, 832), 16),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
])


class AvatarForcingRunner:
    """
    Wraps AvatarForcingInferencePipeline.

    Calls inference_avatar_forcing() with bridge audio_emb instead of wav2vec.
    Every other argument is built the same way as the original inference.py.

    Parameters
    ----------
    af_config    : OmegaConf / Namespace config object (from avatarforcing.yaml)
    checkpoint_path : path to the AvatarForcing generator checkpoint
    device       : torch.device
    num_output_frames : video frames to generate (default 21)
    use_ema      : whether to load EMA weights
    """

    def __init__(
        self,
        af_config,
        checkpoint_path: str,
        device: torch.device,
        num_output_frames: int = 21,
        use_ema:           bool = False,
        dtype:             torch.dtype = torch.bfloat16,
    ):
        self.device            = device
        self.num_output_frames = num_output_frames
        self.dtype             = dtype

        print("[AvatarForcingRunner] Building AvatarForcing pipeline …")
        self._pipeline = AvatarForcingInferencePipeline(af_config, device=device)

        print(f"[AvatarForcingRunner] Loading checkpoint: {checkpoint_path}")
        state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

        if use_ema:
            sd = state_dict["generator_ema"]
            # Strip FSDP prefix if present (mirrors inference.py)
            clean = OrderedDict()
            for k, v in sd.items():
                clean[k.replace("_fsdp_wrapped_module.", "")] = v
            sd = clean
        else:
            sd = state_dict.get("generator", state_dict)

        # Apply LoRA if the config has lora settings
        if hasattr(af_config, "models") and hasattr(af_config.models, "lora"):
            self._pipeline.generator.model = _apply_lora(
                self._pipeline.generator.model, af_config.models.lora
            )

        self._pipeline.generator.load_state_dict(sd, strict=False)
        self._pipeline = self._pipeline.to(device=device, dtype=dtype)
        print("[AvatarForcingRunner] Ready.")

    @torch.no_grad()
    def generate(
        self,
        image_path: str,
        audio_emb:  torch.Tensor,
        prompt:     str,
        neg_prompt: Optional[str] = None,
        num_samples: int = 1,
    ) -> torch.Tensor:
        """
        Generate a talking-head video.

        Parameters
        ----------
        image_path : path to reference face image
        audio_emb  : (T+1, 10752) float32 — from BridgeRunner, matches batch['audio_emb']
        prompt     : text conditioning string
        neg_prompt : negative prompt (unused in current AvatarForcing but kept for API compat)
        num_samples: how many samples to draw (default 1)

        Returns
        -------
        video : (B, T, 3, H, W) float tensor in [0, 1]  — B=num_samples
        """
        device = self.device
        pipeline = self._pipeline

        # ── 0. Dynamically determine num_output_frames ──
        T_audio = audio_emb.shape[0] if audio_emb.dim() == 2 else audio_emb.shape[1]
        raw_num_output_frames = max(1, (T_audio - 1) // 4 + 1)
        
        # Enforce that (num_output_frames - 1) is a multiple of pipeline's block size
        num_frame_per_block = getattr(pipeline, "num_frame_per_block", 1)
        noise_frames = raw_num_output_frames - 1
        noise_frames = (noise_frames // num_frame_per_block) * num_frame_per_block
        current_num_output_frames = noise_frames + 1

        print(f"[AvatarForcingRunner] Dynamically setting num_output_frames to {current_num_output_frames} "
              f"based on audio_emb length {T_audio} (rounded down for block size {num_frame_per_block}).")

        # ── 1. Load and preprocess image (mirrors inference.py lines 109-113, 154-165) ──
        img = Image.open(image_path).convert("RGB")
        img_tensor = _IMAGE_TRANSFORM(img)                    # (3, H, W) in [-1,1]
        img_tensor = img_tensor.unsqueeze(0)                  # (1, 3, H, W)

        image = img_tensor.unsqueeze(2).to(device=device, dtype=self.dtype)
        # image: (1, 3, 1, H, W)

        # ── 2. VAE encode reference frame (mirrors inference.py line 158-165) ─────────
        initial_latent = pipeline.vae.encode_to_latent(image).to(
            device=device, dtype=self.dtype
        )
        initial_latent = initial_latent.repeat(num_samples, 1, 1, 1, 1)
        # initial_latent: (B, 1, C_lat, H_lat, W_lat)

        img_lat = initial_latent.permute(0, 2, 1, 3, 4)
        # Build conditioning tensor y (mirrors inference.py lines 162-165)
        msk = torch.zeros_like(
            img_lat.repeat(1, 1, current_num_output_frames + 20, 1, 1)[:, :1]
        )
        image_cat = img_lat.repeat(1, 1, current_num_output_frames + 20, 1, 1)
        msk[:, :, 1:] = 1
        y = torch.cat([image_cat, msk], dim=1)
        # y: (B, C_lat+1, T+20, H_lat, W_lat)

        # ── 3. Sample noise (mirrors inference.py lines 167-172) ──────────────────────
        h, w = initial_latent.shape[-2], initial_latent.shape[-1]
        sampled_noise = torch.randn(
            (num_samples, current_num_output_frames - 1, 16, h, w),
            device=device, dtype=self.dtype,
        )

        # ── 4. Prepare audio_emb — add batch dim, move to device (mirrors line 178) ───
        if audio_emb.dim() == 2:
            audio_emb_batch = audio_emb.unsqueeze(0).expand(num_samples, -1, -1)
        else:
            audio_emb_batch = audio_emb                        # already (B, T+1, 10752)
        audio_emb_batch = audio_emb_batch.to(device=device, dtype=self.dtype)

        # ── 5. Run inference (mirrors inference.py lines 175-182 EXACTLY) ────────────
        video = pipeline.inference_avatar_forcing(
            noise            = sampled_noise,
            text_prompts     = [prompt] * num_samples,
            audio_embeddings = audio_emb_batch,
            y                = y,
            return_latents   = False,
            initial_latent   = initial_latent,
        )
        # video: (B, T_total, 3, H, W) in [0, 1]

        # Clear VAE cache (mirrors inference.py line 187)
        pipeline.vae.model.clear_cache()

        return video

    @torch.no_grad()
    def generate_to_numpy(
        self,
        image_path: str,
        audio_emb:  torch.Tensor,
        prompt:     str,
        neg_prompt: Optional[str] = None,
    ) -> np.ndarray:
        """
        Convenience wrapper: returns (N_frames, H, W, 3) uint8 numpy array.
        Mirrors the rearrange + *255 in inference.py line 184.
        """
        video = self.generate(image_path, audio_emb, prompt, neg_prompt)
        # video: (B=1, T, 3, H, W) in [0, 1]
        video_np = 255.0 * rearrange(video, "b t c h w -> b t h w c").cpu().numpy()
        video_np = video_np[0].clip(0, 255).astype(np.uint8)   # (T, H, W, 3)
        return video_np
