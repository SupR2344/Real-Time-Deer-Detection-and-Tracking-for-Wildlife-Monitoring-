"""
Evaluate ByteTrack stability on 960x576 tiled detection at conf=0.15 across video frames 200-300.
Generates:
- tracking stability analysis (unique IDs, track continuity, box tightness)
- annotated visualization for frames 230, 260, 340, 400, 450
"""
import sys, os, cv2, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ultralytics import YOLO
from bytetrack_adapter.adapter import ByteTrackAdapter
from trajectory.manager import TrajectoryManager
from prediction.manager import PredictionManager
from collision.engine import CollisionEngine
from collision.geometry import create_polygon

OUT_DIR = "outputs/tight_box_eval"
os.makedirs(OUT_DIR, exist_ok=True)

model = YOLO("model/best.pt")
video_path = "13297686_3840_2160_60fps.mp4"
ROAD_ZONE = [[1050, 1200], [2750, 1200], [3840, 2160], [0, 2160]]
road_poly = create_polygon(ROAD_ZONE)

tile_w, tile_h = 960, 576
stride_w, stride_h = 720, 432
conf_threshold = 0.15

def get_tiles(W=3840, H=2160):
    tiles = []
    y = 0
    while y + tile_h <= H:
        x = 0
        while x + tile_w <= W:
            tiles.append((x, y, x + tile_w, y + tile_h))
            x += stride_w
        if x < W:
            tiles.append((W - tile_w, y, W, y + tile_h))
        y += stride_h
    if y < H:
        y = H - tile_h
        x = 0
        while x + tile_w <= W:
            tiles.append((x, y, x + tile_w, y + tile_h))
            x += stride_w
        if x < W:
            tiles.append((W - tile_w, y, W, y + tile_h))
    return list(set(tiles))

tiles = get_tiles()

def nms(boxes, scores, iou_thresh=0.4):
    if not boxes: return []
    b = np.array(boxes, dtype=np.float32)
    s = np.array(scores, dtype=np.float32)
    x1, y1, x2, y2 = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = s.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = int(order[0])
        keep.append(i)
        if order.size == 1: break
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
        order = order[np.where(iou <= iou_thresh)[0] + 1]
    return keep

tracker = ByteTrackAdapter()
traj_mgr = TrajectoryManager()
pred_mgr = PredictionManager()
collision = CollisionEngine(road_points=ROAD_ZONE)

cap = cv2.VideoCapture(video_path)

track_lifespans = {}
frame_results = {}
annotated_fids = [230, 260, 340, 400, 450]

for fid in range(458):
    ret, frame = cap.read()
    if not ret: break
    
    # Run tiling
    raw_boxes = []
    raw_scores = []
    for (tx1, ty1, tx2, ty2) in tiles:
        crop = frame[ty1:ty2, tx1:tx2]
        res = model(crop, conf=conf_threshold, imgsz=640, verbose=False)
        for b in res[0].boxes:
            cls_id = int(b.cls[0].item())
            if model.names.get(cls_id, "").lower() == "deer":
                bx1, by1, bx2, by2 = map(float, b.xyxy[0].tolist())
                raw_boxes.append([bx1 + tx1, by1 + ty1, bx2 + tx1, by2 + ty1])
                raw_scores.append(float(b.conf[0].item()))
                
    keep = nms(raw_boxes, raw_scores, iou_thresh=0.4)
    detections = []
    for i in keep:
        detections.append({
            "bbox": raw_boxes[i],
            "confidence": raw_scores[i],
            "class_id": 0,
            "class_name": "deer"
        })
        
    tracked = tracker.update(detections, fid)
    for t in tracked:
        tid = t["track_id"]
        if tid not in track_lifespans:
            track_lifespans[tid] = []
        track_lifespans[tid].append(fid)
        traj_mgr.update(tid, fid, t["bbox"])
        pred_mgr.update(tid, t["bbox"])
        
    if fid in annotated_fids:
        # Create detailed visualization showing:
        # raw crop detections (orange), post-NMS/tracked boxes (red/green), track ID and confidence
        vis = frame.copy()
        # Road zone overlay
        pts = np.array(ROAD_ZONE, dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(vis, [pts], isClosed=True, color=(0, 200, 255), thickness=4)
        
        # Raw boxes
        for rb in raw_boxes:
            cv2.rectangle(vis, (int(rb[0]), int(rb[1])), (int(rb[2]), int(rb[3])), (255, 160, 0), 2)
            
        # Tracked boxes
        for t in tracked:
            b = t["bbox"]
            tid = t["track_id"]
            c = t["confidence"]
            w = int(b[2] - b[0])
            h = int(b[3] - b[1])
            cx, cy = int((b[0]+b[2])/2), int((b[1]+b[3])/2)
            
            # Risk
            cur_pos = pred_mgr.current_position(tid)
            fut_pos = pred_mgr.predict(tid)
            spd = pred_mgr.speed(tid)
            risk = collision.evaluate(fut_pos, spd, cur_pos)
            
            color = (0, 0, 255) if risk.value == "HIGH_RISK" else (0, 220, 80)
            cv2.rectangle(vis, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), color, 4)
            cv2.circle(vis, (cx, cy), 10, (0, 255, 0), -1)
            cv2.putText(vis, f"ID:{tid} {risk.value} c={c:.2f} ({w}x{h})", 
                        (int(b[0]), int(b[1])-12), cv2.FONT_HERSHEY_SIMPLEX, 1.1, color, 3)
                        
        vis_small = cv2.resize(vis, (1920, 1080))
        cv2.imwrite(f"{OUT_DIR}/annotated_frame_{fid:03d}.jpg", vis_small)
        frame_results[fid] = {
            "raw_count": len(raw_boxes),
            "post_nms": len(detections),
            "tracked": len(tracked),
            "track_ids": [t["track_id"] for t in tracked]
        }

cap.release()

summary = {
    "tile_config": "960x576_stride720x432",
    "conf": conf_threshold,
    "unique_tracks": list(track_lifespans.keys()),
    "track_lifespans": {k: [min(v), max(v), len(v)] for k, v in track_lifespans.items()},
    "sample_frames": frame_results
}

with open(f"{OUT_DIR}/summary.json", "w") as f:
    json.dump(summary, f, indent=2)

print("Tracking evaluation complete.")
print(json.dumps(summary, indent=2))
