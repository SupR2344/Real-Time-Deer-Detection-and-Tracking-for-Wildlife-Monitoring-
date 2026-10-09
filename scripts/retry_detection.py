"""
Retry detection on the 4K video at multiple imgsz values.
Goal: find the optimal imgsz/conf that actually produces detections.
"""
import sys, os, cv2, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from detection.detector import DeerDetector

VIDEO = "13297686_3840_2160_60fps.mp4"

print("=" * 60)
print("GPU Info")
print("=" * 60)
print(f"CUDA: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"Device: {torch.cuda.get_device_name(0)}")
    vram = torch.cuda.get_device_properties(0).total_memory / 1e6
    print(f"VRAM total: {vram:.0f} MB")
print()

# Read key frames (where deer are clearly visible)
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
print()

# Test multiple configurations
configs = [
    {"imgsz": 3840, "conf": 0.05, "label": "Native 3840, conf=0.05"},
    {"imgsz": 1920, "conf": 0.05, "label": "1920, conf=0.05"},
    {"imgsz": 1920, "conf": 0.10, "label": "1920, conf=0.10"},
    {"imgsz": 1280, "conf": 0.05, "label": "1280, conf=0.05"},
    {"imgsz": 1280, "conf": 0.10, "label": "1280, conf=0.10"},
    {"imgsz": 960,  "conf": 0.05, "label": "960, conf=0.05"},
    {"imgsz": 640,  "conf": 0.05, "label": "640, conf=0.05"},
]

best_config = None
best_count = 0

for cfg in configs:
    print(f"--- Testing: {cfg['label']} ---")
    try:
        det = DeerDetector(
            model_path="model/best.pt",
            confidence=cfg["conf"],
            img_size=cfg["imgsz"],
            device="cuda" if torch.cuda.is_available() else "cpu"
        )
        total_dets = 0
        frame_results = {}
        for fid, frame in frame_cache.items():
            dets = det.detect(frame)
            frame_results[fid] = dets
            if dets:
                confs = [round(d["confidence"], 3) for d in dets]
                bboxes = [[round(v, 0) for v in d["bbox"]] for d in dets]
                print(f"  Frame {fid}: {len(dets)} det(s)  confs={confs}")
                for b in bboxes[:3]:
                    print(f"    bbox={b}")
            total_dets += len(dets)
        print(f"  TOTAL DETECTIONS: {total_dets} across {len(frame_cache)} key frames")
        if total_dets > best_count:
            best_count = total_dets
            best_config = cfg.copy()
            best_config["frame_results"] = {k: len(v) for k, v in frame_results.items()}
        # If we found good detections, keep going to find the best
        print()
        # Release GPU memory
        del det
        torch.cuda.empty_cache()
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print(f"  OOM at imgsz={cfg['imgsz']} — skipping")
            torch.cuda.empty_cache()
        else:
            print(f"  ERROR: {e}")
        print()

print("=" * 60)
print("SUMMARY")
print("=" * 60)
if best_config:
    print(f"Best config: imgsz={best_config['imgsz']}, conf={best_config['conf']}")
    print(f"Total detections: {best_count}")
    print(f"Per-frame: {best_config.get('frame_results', {})}")
else:
    print("NO DETECTIONS at any config. Model may not be compatible with this video.")
print()
print("Recommendation for .env:")
if best_config:
    print(f"YOLO_IMG_SIZE={best_config['imgsz']}")
    print(f"YOLO_CONFIDENCE={best_config['conf']}")
else:
    print("FALLBACK: Use deer_test.mp4 for E2E validation.")
