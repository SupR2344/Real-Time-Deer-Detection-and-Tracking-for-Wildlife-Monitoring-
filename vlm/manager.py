# -*- coding: utf-8 -*-
"""VLM (BLIP) manager.

Loads the BLIP captioning model lazily and provides a ``caption`` method that
returns a caption for a given image crop. The manager respects a per-track
cooldown so that captions are generated at most once per ``VLM_COOLDOWN`` seconds.

The BLIP model name is taken from the ``VLM_MODEL`` environment variable (default
``Salesforce/blip-image-captioning-base``). CUDA is used when available, with a
CPU fallback.
"""

import time
import logging
from typing import Optional

import cv2
import numpy as np
import torch
from PIL import Image

from config.config import get

logger = logging.getLogger(__name__)


class VLMManager:
    """Lazy-loading BLIP captioning manager with per-track cooldown.

    The manager is *optional* – it will only be used when ``VLM_ENABLED`` is true.
    ``caption`` returns ``None`` if the cooldown for the given ``track_id`` has not
    elapsed, or if no HIGH_RISK event has occurred.
    """

    def __init__(self) -> None:
        self.enabled: bool = get("VLM_ENABLED", True)
        self.model_name: str = get("VLM_MODEL", "Salesforce/blip-image-captioning-base")
        self.cooldown: int = get("VLM_COOLDOWN", 30)  # seconds
        self._last_caption_time: dict = {}
        self._processor = None
        self._model = None
        self._device: Optional[torch.device] = None
        self._loaded: bool = False
        self._load_error: Optional[str] = None
        self.status: str = "STANDBY"

    def _load(self) -> bool:
        """Lazy-load the BLIP model. Returns True on success, False on failure."""
        if self._loaded:
            return self._model is not None
        if self._load_error:
            return False

        if not self.enabled:
            self.status = "DISABLED"
            return False

        self.status = "LOADING"
        logger.info(f"Loading BLIP model: {self.model_name}")
        try:
            from transformers import BlipProcessor, BlipForConditionalGeneration

            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            logger.info(f"BLIP using device: {self._device}")

            self._processor = BlipProcessor.from_pretrained(self.model_name)
            self._model = BlipForConditionalGeneration.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if self._device.type == "cuda" else torch.float32,
            )
            self._model.to(self._device)
            self._model.eval()
            self._loaded = True
            self.status = "ACTIVE"
            logger.info("BLIP model loaded successfully")
            return True
        except Exception as exc:
            self._load_error = str(exc)
            self._loaded = True  # mark as attempted so we don't retry every frame
            self.status = "ERROR"
            logger.error(f"BLIP model load failed: {exc}")
            return False

    def _can_generate(self, track_id: int) -> bool:
        now = time.time()
        last = self._last_caption_time.get(track_id, 0.0)
        return (now - last) >= self.cooldown

    def caption(self, image_crop: np.ndarray, track_id: int) -> Optional[str]:
        """Return a BLIP caption for ``image_crop`` if cooldown permits.

        Parameters
        ----------
        image_crop: np.ndarray – BGR crop from the original frame.
        track_id: int – ID of the tracked object.

        Returns None if VLM disabled, cooldown not elapsed, model failed to load,
        or the crop is too small.
        """
        if not self.enabled:
            return None
        if not self._can_generate(track_id):
            return None

        # Guard: crop must be non-empty and large enough to be meaningful
        if image_crop is None or image_crop.size == 0:
            return None
        h, w = image_crop.shape[:2]
        if h < 10 or w < 10:
            return None

        if not self._load():
            return None

        if self._processor is None or self._model is None:
            return None

        try:
            # Convert BGR to RGB then to PIL
            rgb = cv2.cvtColor(image_crop, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb)

            inputs = self._processor(images=pil_image, return_tensors="pt")
            inputs = {k: v.to(self._device) for k, v in inputs.items()}

            with torch.no_grad():
                out_ids = self._model.generate(**inputs, max_new_tokens=64)

            caption = self._processor.decode(out_ids[0], skip_special_tokens=True)
            caption = caption.strip()

            # Record timestamp
            self._last_caption_time[track_id] = time.time()
            logger.info(f"BLIP caption for track {track_id}: {caption}")
            return caption if caption else None

        except Exception as exc:
            logger.error(f"BLIP inference error for track {track_id}: {exc}")
            return None
