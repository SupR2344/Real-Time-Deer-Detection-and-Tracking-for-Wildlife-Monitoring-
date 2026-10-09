"""
Sliding-window crop detection on the 4K video.
Standard technique for high-res cameras with a lower-res trained model.
We tile the 4K frame into overlapping crops, run YOLO on each, and
merge detections back into original frame coordinates via NMS.
"""
import sys, os, cv2, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from detection.detector import DeerDetector

VIDEO = "13297686_3840_2160_60fps.mp4"
MODEL_PATH = "model/best.pt"

# --------------------------------------------------------------------------
# Crop strategy: tile 3840x2160 into overlapping 1280x768 windows
# Each window is passed to YOLO at imgsz=640
# Deer at ~320px height in 4K → ~160px in 1280-wide crop → ~80px after 640-scale
# --------------------------------------------------------------------------
def get_crops(frame, crop_w=1280, crop_h=768, stride_x=960, stride_y=576):
    """Generate overlapping crop windows from a large frame."""
    H, W = frame.shape[:2]
    crops = []
    y = 0
    while y + crop_h <= H:
        x = 0
        while x + crop_w <= W:
            crops.append((x, y, x + crop_w, y + crop_h))
            x += stride_x
        # last column
        if x < W:
            crops.append((W - crop_w, y, W, y + crop_h))
        y += stride_y
    # last row
    if y < H:
        y = H - crop_h
        x = 0
        while x + crop_w <= W:
            crops.append((x, y, x + crop_w, y + crop_h))
            x += stride_x
        if x < W:
            crops.append((W - crop_w, y, W, y + crop_h))
    # Deduplicate
    return list(set(crops))


def nms_boxes(boxes, scores, iou_thresh=0.5):
    """Simple greedy NMS."""
    if not boxes:
        return []
    boxes = np.array(boxes, dtype=np.float32)
    scores = np.array(scores, dtype=np.float32)
    x1, y1, x2, y2 = boxes[:,0], boxes[:,1], boxes[:,2], boxes[:,3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(int(i))
        if order.size == 1:
            break
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
        inds = np.where(iou <= iou_thresh)[0]
        order = order[inds + 1]
    return keep


def detect_sliding_window(det, frame, conf_thresh=0.05):
    """Run sliding-window tiled detection on a full 4K frame."""
    crops = get_crops(frame)
    all_boxes = []
    all_confs = []

    for (x1c, y1c, x2c, y2c) in crops:
        crop = frame[y1c:y2c, x1c:x2c]
        dets = det.detect(crop)
        for d in dets:
            bx1, by1, bx2, by2 = d["bbox"]
            # Map back to original frame coords
            all_boxes.append([bx1 + x1c, by1 + y1c, bx2 + x1c, by2 + y1c])
            all_confs.append(d["confidence"])

    if not all_boxes:
        return []

    keep = nms_boxes(all_boxes, all_confs, iou_thresh=0.5)
    results = []
    for i in keep:
        results.append({
            "bbox": all_boxes[i],
            "confidence": all_confs[i],
            "class_id": 0,
            "class_name": "deer"
        })
    return results


print("=" * 60)
print("SLIDING-WINDOW CROP DETECTION — 4K VIDEO")
print("=" * 60)
print(f"CUDA: {torch.cuda.is_available()}")
print()

# Load key frames
print("Loading key frames from video...")
cap = cv2.VideoCapture(VIDEO)
key_frames_ids = [0, 30, 50, 80, 100, 150, 200, 230, 280, 340, 400, 450]
frame_cache = {}
for fid in range(458):
    ret, frame = cap.read()
    if not ret:
        break
    if fid in key_frames_ids:
        frame_cache[fid] = frame.copy()
cap.release()
print(f"Loaded {len(frame_cache)} key frames (shape={next(iter(frame_cache.values())).shape})")

# Crop info
frame = next(iter(frame_cache.values()))
crops = get_crops(frame)
print(f"Crop windows per frame: {len(crops)}  (1280x768, stride 960x576)")
print()

# Test configs for the crop-based approach
configs = [
    {"imgsz": 640, "conf": 0.05, "label": "crop+imgsz=640, conf=0.05"},
    {"imgsz": 640, "conf": 0.10, "label": "crop+imgsz=640, conf=0.10"},
    {"imgsz": 640, "conf": 0.15, "label": "crop+imgsz=640, conf=0.15"},
    {"imgsz": 640, "conf": 0.25, "label": "crop+imgsz=640, conf=0.25"},
    {"imgsz": 640, "conf": 0.30, "label": "crop+imgsz=640, conf=0.30"},
    {"imgsz": 1280, "conf": 0.05, "label": "crop+imgsz=1280, conf=0.05"},
]

best = {"count": 0, "cfg": None}

for cfg in configs:
    print(f"--- {cfg['label']} ---")
    try:
        det = DeerDetector(
            model_path=MODEL_PATH,
            confidence=cfg["conf"],
            img_size=cfg["imgsz"],
            device="cuda" if torch.cuda.is_available() else "cpu"
        )
        total = 0
        frame_det_counts = {}
        for fid, frame in frame_cache.items():
            dets = detect_sliding_window(det, frame, conf_thresh=cfg["conf"])
            frame_det_counts[fid] = len(dets)
            if dets:
                for d in dets:
                    b = [round(v, 0) for v in d["bbox"]]
                    print(f"  Frame {fid}: conf={round(d['confidence'],3)}  bbox={b}")
            total += len(dets)
        print(f"  TOTAL: {total} detections in {len(frame_cache)} key frames")
        per_frame_str = {k: v for k, v in frame_det_counts.items() if v > 0}
        if per_frame_str:
            print(f"  Frames with dets: {per_frame_str}")
        if total > best["count"]:
            best["count"] = total
            best["cfg"] = cfg.copy()
        del det
        torch.cuda.empty_cache()
    except Exception as e:
        print(f"  ERROR: {e}")
    print()

print("=" * 60)
print("SUMMARY")
print("=" * 60)
if best["count"] > 0:
    print(f"DETECTIONS FOUND!")
    print(f"Best config: {best['cfg']['label']}")
    print(f"Total detections (key frames): {best['count']}")
    print()
    print("Recommended .env settings:")
    print(f"  YOLO_IMG_SIZE={best['cfg']['imgsz']}")
    print(f"  YOLO_CONFIDENCE={best['cfg']['conf']}")
    print("  USE_SLIDING_WINDOW=true")
else:
    print("NO DETECTIONS at any sliding-window config.")
    print("Root cause: model/best.pt is domain-incompatible with this video.")
    print("Action: Fall back to deer_test.mp4 for full E2E validation.")
    print("        The new 4K video will be used for VISUAL DEMONSTRATION only.")
