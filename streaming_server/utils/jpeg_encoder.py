"""
streaming_server/utils/jpeg_encoder.py
=========================================
TurboJPEG wrapper for high-speed JPEG encoding of RGB frames.

Falls back to cv2.imencode if python-turbojpeg is not installed.
TurboJPEG is ~10× faster than PIL for JPEG encoding — critical for 25 FPS.
"""

from __future__ import annotations

import logging
import numpy as np

logger = logging.getLogger(__name__)


class JpegEncoder:
    """
    Encodes numpy RGB frames to JPEG bytes at high speed.

    Priority:
      1. TurboJPEG (libjpeg-turbo)  — fastest
      2. cv2.imencode               — fallback
    """

    QUALITY = 85  # good balance of quality vs size

    def __init__(self, quality: int = QUALITY):
        self.quality = quality
        self._backend: str = "none"
        self._tj = None

        # ── Try TurboJPEG first ────────────────────────────────────────────
        try:
            from turbojpeg import TurboJPEG, TJPF_RGB
            self._tj = TurboJPEG()
            self._TJPF_RGB = TJPF_RGB
            self._backend = "turbojpeg"
            logger.info("[JpegEncoder] Using TurboJPEG backend.")
            return
        except ImportError:
            pass
        except Exception as e:
            logger.warning(f"[JpegEncoder] TurboJPEG init failed: {e}")

        # ── Fall back to cv2 ───────────────────────────────────────────────
        try:
            import cv2  # noqa: F401
            self._backend = "cv2"
            logger.info("[JpegEncoder] TurboJPEG not available, using cv2 backend.")
            return
        except ImportError:
            pass

        raise RuntimeError(
            "[JpegEncoder] No JPEG encoder available. "
            "Install python-turbojpeg or opencv-python-headless."
        )

    def encode(self, frame_rgb: np.ndarray) -> bytes:
        """
        Encode a (H, W, 3) uint8 RGB frame to JPEG bytes.

        Parameters
        ----------
        frame_rgb : (H, W, 3) uint8  — RGB, NOT BGR

        Returns
        -------
        bytes : JPEG-encoded image
        """
        if self._backend == "turbojpeg":
            return self._tj.encode(frame_rgb, quality=self.quality,
                                   pixel_format=self._TJPF_RGB)
        elif self._backend == "cv2":
            import cv2
            # cv2 uses BGR
            frame_bgr = frame_rgb[:, :, ::-1]
            _, buf = cv2.imencode(".jpg", frame_bgr,
                                  [cv2.IMWRITE_JPEG_QUALITY, self.quality])
            return buf.tobytes()
        else:
            raise RuntimeError("No encoder initialised.")

    @property
    def backend(self) -> str:
        return self._backend
