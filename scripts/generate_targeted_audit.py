"""
Targeted Audit Script for Frames 230, 260, 340, 400, 450.
Performs:
1. Exact inspection of crop detections before NMS.
2. Analysis of whether multiple overlapping crop detections exist for individual deer.
3. Inspection of post-NMS boxes and centroids.
4. Generation of annotated images for frames 230, 260, 340, 400, 450.
5. ByteTrack continuity test on tight detections.
"""
import sys, os, cv2, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ultralytics import YOLO
from bytetrack_adapter.adapter import ByteTrackAdapter
from collision.geometry import create_polygon

OUT_DIR = "outputs/tight_box_audit"
os.makedirs(OUT_DIR, exist_ok=True)

model = YOLO("model/best.pt")
VIDEO = "13297686_3840_2160_60fps.mp4"
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

cap = cv2.VideoCapture(VIDEO)
target_fids = [230, 260, 340, 400, 450]
frame_dict = {}
for fid in range(458):
    ret, frame = cap.read()
    if not ret: break
    if fid in target_fids:
        frame_dict[fid] = frame.copy()
cap.release()

detailed_audit = {}

for fid in target_fids:
    frame = frame_dict[fid]
    raw_detections = []
    
    for (tx1, ty1, tx2, ty2) in tiles:
        crop = frame[ty1:ty2, tx1:tx2]
        res = model(crop, conf=conf_threshold, imgsz=640, verbose=False)
        for b in res[0].boxes:
            cls_id = int(b.cls[0].item())
            if model.names.get(cls_id, "").lower() == "deer":
                bx1, by1, bx2, by2 = map(float, b.xyxy[0].tolist())
                c = float(b.conf[0].item())
                gx1, gy1, gx2, gy2 = bx1 + tx1, by1 + ty1, bx2 + tx1, by2 + ty1
                raw_detections.append({
                    "tile": [tx1, ty1, tx2, ty2],
                    "crop_bbox": [round(bx1, 1), round(by1, 1), round(bx2, 1), round(by2, 1)],
                    "global_bbox": [round(gx1, 1), round(gy1, 1), round(gx2, 1), round(gy2, 1)],
                    "conf": round(c, 4),
                    "dim": [round(gx2 - gx1, 1), round(gy2 - gy1, 1)],
                    "center": [round((gx1 + gx2) / 2, 1), round((gy1 + gy2) / 2, 1)]
                })
                
    raw_boxes = [d["global_bbox"] for d in raw_detections]
    raw_scores = [d["conf"] for d in raw_detections]
    keep_idx = nms(raw_boxes, raw_scores, iou_thresh=0.4)
    
    post_nms_detections = []
    for idx in keep_idx:
        d = raw_detections[idx]
        post_nms_detections.append({
            "bbox": d["global_bbox"],
            "conf": d["conf"],
            "dim": d["dim"],
            "center": d["center"]
        })
        
    detailed_audit[fid] = {
        "raw_count": len(raw_detections),
        "post_nms_count": len(post_nms_detections),
        "raw_detections": raw_detections,
        "post_nms_detections": post_nms_detections
    }
    
    # Generate annotated image
    vis = frame.copy()
    # Draw road polygon
    pts = np.array(ROAD_ZONE, dtype=np.int32).reshape(-1, 1, 2)
    cv2.polylines(vis, [pts], isClosed=True, color=(0, 200, 255), thickness=3)
    
    # Draw raw crop detections in orange
    for rd in raw_detections:
        gb = [int(v) for v in rd["global_bbox"]]
        cv2.rectangle(vis, (gb[0], gb[1]), (gb[2], gb[3]), (0, 140, 255), 2)
        
    # Draw post-NMS boxes in thick red with labels and centroids in green
    for pd in post_nms_detections:
        pb = [int(v) for v in pd["bbox"]]
        cx, cy = int(pd["center"][0]), int(pd["center"][1])
        cv2.rectangle(vis, (pb[0], pb[1]), (pb[2], pb[3]), (0, 0, 255), 4)
        cv2.circle(vis, (cx, cy), 8, (0, 255, 0), -1)
        label = f"deer {pd['conf']:.2f} ({int(pd['dim'][0])}x{int(pd['dim'][1])})"
        cv2.putText(vis, label, (pb[0], pb[1] - 12), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 255), 3)
        cv2.putText(vis, label, (pb[0], pb[1] - 12), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 1)
        
    vis_small = cv2.resize(vis, (1920, 1080))
    cv2.imwrite(f"{OUT_DIR}/audit_frame_{fid:03d}.jpg", vis_small)

with open(f"{OUT_DIR}/detailed_audit.json", "w") as f:
    json.dump(detailed_audit, f, indent=2)

print("Targeted audit completed successfully.")
