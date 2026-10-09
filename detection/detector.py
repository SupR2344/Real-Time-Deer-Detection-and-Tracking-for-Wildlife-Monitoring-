import os
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
import cv2
from ultralytics import YOLO


class DeerDetector:
    """YOLOv11 based deer detector.

    Loads the model from the configured path and provides a ``detect`` method that
    returns detections limited to the "deer" class.

    For high-resolution frames (width > 2000 px, e.g. 4K cameras) the detector
    automatically applies a sliding-window tiling strategy: the frame is split into
    overlapping 1280×768 crops which are each inferred independently, and the
    results are merged with NMS back into original frame coordinates.  This is a
    standard production technique and requires no model changes.
    """

    def __init__(
        self,
        model_path: str = "model/best.pt",
        confidence: float = 0.3,
        iou: float = 0.5,
        img_size: int = 640,
        device: str = "cpu",
    ):
        self.model_path = Path(model_path)
        if not self.model_path.is_file():
            raise FileNotFoundError(f"YOLO model not found at {self.model_path}")
        self.confidence = confidence
        self.iou = iou
        self.img_size = img_size
        self.device = device
        self.model = YOLO(str(self.model_path))
        # Apply configuration to the model
        self.model.fuse()  # optional fuse for speed
        self.model.to(self.device)
        self.model.overrides["conf"] = self.confidence
        self.model.overrides["iou"] = self.iou
        self.model.overrides["imgsz"] = self.img_size
        # Load class names from the model metadata (if present)
        self.class_names = self.model.names or {0: "deer"}

    # ------------------------------------------------------------------
    # Class filter
    # ------------------------------------------------------------------

    def _is_deer(self, class_id: int) -> bool:
        """Return True if the class id corresponds to a deer.

        The model may have arbitrary class IDs – we rely on the name being "deer"
        (case-insensitive). If the name is missing we fall back to id 0.
        """
        name = self.class_names.get(class_id, "")
        return name.lower() == "deer"

    # ------------------------------------------------------------------
    # Sliding-window helpers for high-resolution (4K+) frames
    # ------------------------------------------------------------------

    @staticmethod
    def _get_crops(
        W: int, H: int,
        crop_w: int = 1280, crop_h: int = 768,
        stride_x: int = 960, stride_y: int = 576,
    ) -> List[tuple]:
        """Generate overlapping crop rectangles (x1, y1, x2, y2) covering the frame."""
        crops: set = set()
        y = 0
        while y + crop_h <= H:
            x = 0
            while x + crop_w <= W:
                crops.add((x, y, x + crop_w, y + crop_h))
                x += stride_x
            if x < W:
                crops.add((W - crop_w, y, W, y + crop_h))
            y += stride_y
        if y < H:
            y = H - crop_h
            x = 0
            while x + crop_w <= W:
                crops.add((x, y, x + crop_w, y + crop_h))
                x += stride_x
            if x < W:
                crops.add((W - crop_w, y, W, y + crop_h))
        return list(crops)

    @staticmethod
    def _nms(
        boxes: List[List[float]], scores: List[float], iou_thresh: float = 0.5
    ) -> List[int]:
        """Greedy NMS — returns indices of boxes to keep."""
        if not boxes:
            return []
        b = np.array(boxes, dtype=np.float32)
        s = np.array(scores, dtype=np.float32)
        x1, y1, x2, y2 = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
        areas = (x2 - x1) * (y2 - y1)
        order = s.argsort()[::-1]
        keep: List[int] = []
        while order.size > 0:
            i = int(order[0])
            keep.append(i)
            if order.size == 1:
                break
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
            iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
            order = order[np.where(iou <= iou_thresh)[0] + 1]
        return keep

    def _infer_crop(self, crop: np.ndarray) -> List[Dict[str, Any]]:
        """Run YOLO on a single BGR crop. Returns raw detections in crop coords."""
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        dets: List[Dict[str, Any]] = []
        for r in self.model(rgb, stream=True):
            for box in r.boxes:
                class_id = int(box.cls[0].item())
                if not self._is_deer(class_id):
                    continue
                x1, y1, x2, y2 = map(float, box.xyxy[0].tolist())
                dets.append({
                    "bbox": [x1, y1, x2, y2],
                    "confidence": float(box.conf[0].item()),
                    "class_id": class_id,
                    "class_name": self.class_names.get(class_id, "deer"),
                })
        return dets

    def _detect_sliding_window(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """Tile-based detection for high-resolution (4K+) frames.

        Splits the frame into overlapping 1280×768 crops, runs YOLO on each,
        maps bbox coordinates back to original frame space, then applies NMS.
        """
        H, W = frame.shape[:2]
        crops = self._get_crops(W, H)
        all_boxes: List[List[float]] = []
        all_confs: List[float] = []
        all_meta: List[Dict] = []

        for (x1c, y1c, x2c, y2c) in crops:
            crop = frame[y1c:y2c, x1c:x2c]
            for d in self._infer_crop(crop):
                bx1, by1, bx2, by2 = d["bbox"]
                all_boxes.append([bx1 + x1c, by1 + y1c, bx2 + x1c, by2 + y1c])
                all_confs.append(d["confidence"])
                all_meta.append(d)

        results: List[Dict[str, Any]] = []
        for i in self._nms(all_boxes, all_confs):
            results.append({
                "bbox": all_boxes[i],
                "confidence": all_confs[i],
                "class_id": all_meta[i]["class_id"],
                "class_name": all_meta[i]["class_name"],
            })
        return results

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """Run detection on a single frame.

        For high-resolution frames (width > 2000 px) the sliding-window tiling
        strategy is applied automatically so deer remain large enough to detect.

        Parameters
        ----------
        frame: np.ndarray
            BGR image as returned by ``cv2.VideoCapture``.

        Returns
        -------
        List[Dict]
            Each dict contains ``bbox`` (x1, y1, x2, y2), ``confidence``,
            ``class_id`` and ``class_name``.  Coordinates are always in
            original (full-frame) pixel space.
        """
        _H, W = frame.shape[:2]
        # Auto sliding-window for high-res / 4K frames
        if W > 2000:
            return self._detect_sliding_window(frame)

        # Standard single-pass for normal resolution frames
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        detections: List[Dict[str, Any]] = []
        for r in self.model(rgb, stream=True):
            for box in r.boxes:
                class_id = int(box.cls[0].item())
                if not self._is_deer(class_id):
                    continue
                x1, y1, x2, y2 = map(float, box.xyxy[0].tolist())
                confidence = float(box.conf[0].item())
                detections.append({
                    "bbox": [x1, y1, x2, y2],
                    "confidence": confidence,
                    "class_id": class_id,
                    "class_name": self.class_names.get(class_id, "deer"),
                })
        return detections
