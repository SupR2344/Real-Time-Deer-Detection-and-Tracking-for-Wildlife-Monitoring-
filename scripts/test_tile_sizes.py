"""
Comprehensive Crop Size & Tiling Diagnostic for 4K Footage.
Tests:
- Tile sizes: 960x576, 768x512, 640x480, and the previous 1280x768
- Strides: with ~30-50% overlap
- Confidences: 0.05, 0.10, 0.15, 0.25
- Detailed inspection of pre-NMS vs post-NMS boxes
- Generation of annotated images for frames 230, 260, 340, 400, 450
"""
import sys, os, cv2, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from ultralytics import YOLO

OUT_DIR = "outputs/tile_investigation"
os.makedirs(OUT_DIR, exist_ok=True)

model = YOLO("model/best.pt")
video_path = "13297686_3840_2160_60fps.mp4"

frames_to_test = [230, 260, 340, 400, 450]

cap = cv2.VideoCapture(video_path)
frame_data = {}
for fid in range(458):
    ret, f = cap.read()
    if not ret: break
    if fid in frames_to_test:
        frame_data[fid] = f.copy()
cap.release()

print(f"Loaded {len(frame_data)} test frames.")

def get_tiles(W, H, tile_w, tile_h, stride_w, stride_h):
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

def nms(boxes, scores, iou_thresh=0.4):
    if not boxes:
        return []
    b = np.array(boxes, dtype=np.float32)
    s = np.array(scores, dtype=np.float32)
    x1, y1, x2, y2 = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = s.argsort()[::-1]
    keep = []
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

configs = [
    {"name": "640x480_stride480x360", "w": 640, "h": 480, "sw": 480, "sh": 360},
    {"name": "768x512_stride576x384", "w": 768, "h": 512, "sw": 576, "sh": 384},
    {"name": "960x576_stride720x432", "w": 960, "h": 576, "sw": 720, "sh": 432},
    {"name": "1280x768_stride960x576", "w": 1280, "h": 768, "sw": 960, "sh": 576},
]

conf_levels = [0.05, 0.10, 0.15, 0.25]

results_summary = []

for cfg in configs:
    tile_w, tile_h = cfg["w"], cfg["h"]
    stride_w, stride_h = cfg["sw"], cfg["sh"]
    tiles = get_tiles(3840, 2160, tile_w, tile_h, stride_w, stride_h)
    print(f"\n========================================================")
    print(f"Testing Config: {cfg['name']} (Total tiles per frame: {len(tiles)})")
    print(f"========================================================")
    
    for conf in conf_levels:
        total_boxes_before_nms = 0
        total_boxes_after_nms = 0
        box_dimensions = []
        is_macro_box_count = 0
        
        for fid in frames_to_test:
            frame = frame_data[fid]
            raw_boxes = []
            raw_scores = []
            
            for (tx1, ty1, tx2, ty2) in tiles:
                crop = frame[ty1:ty2, tx1:tx2]
                res = model(crop, conf=conf, imgsz=640, verbose=False)
                for b in res[0].boxes:
                    cls_id = int(b.cls[0].item())
                    c = float(b.conf[0].item())
                    # Check class is deer
                    if model.names.get(cls_id, "").lower() == "deer":
                        bx1, by1, bx2, by2 = map(float, b.xyxy[0].tolist())
                        # global coords
                        gx1, gy1, gx2, gy2 = bx1 + tx1, by1 + ty1, bx2 + tx1, by2 + ty1
                        raw_boxes.append([gx1, gy1, gx2, gy2])
                        raw_scores.append(c)
            
            total_boxes_before_nms += len(raw_boxes)
            keep_indices = nms(raw_boxes, raw_scores, iou_thresh=0.4)
            kept_boxes = [raw_boxes[i] for i in keep_indices]
            kept_scores = [raw_scores[i] for i in keep_indices]
            total_boxes_after_nms += len(kept_boxes)
            
            for b in kept_boxes:
                bw = b[2] - b[0]
                bh = b[3] - b[1]
                box_dimensions.append((bw, bh))
                # If box covers more than 80% of tile width and height, it's a macro tile box
                if bw > 0.75 * tile_w and bh > 0.75 * tile_h:
                    is_macro_box_count += 1
            
            # Save visual annotation for frame 230 on key configs
            if fid == 230 and conf in [0.05, 0.15]:
                vis = frame.copy()
                # Draw raw pre-nms in thin blue
                for b in raw_boxes:
                    cv2.rectangle(vis, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (255, 150, 0), 2)
                # Draw kept post-nms in thick red
                for i, b in enumerate(kept_boxes):
                    cv2.rectangle(vis, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (0, 0, 255), 5)
                    cx, cy = int((b[0]+b[2])/2), int((b[1]+b[3])/2)
                    cv2.circle(vis, (cx, cy), 12, (0, 255, 0), -1)
                    cv2.putText(vis, f"conf={kept_scores[i]:.2f} {int(b[2]-b[0])}x{int(b[3]-b[1])}", 
                                (int(b[0]), int(b[1])-15), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
                
                vis_small = cv2.resize(vis, (1920, 1080))
                out_name = f"{OUT_DIR}/vis_f230_{cfg['name']}_conf{int(conf*100):02d}.jpg"
                cv2.imwrite(out_name, vis_small)
        
        avg_w = np.mean([d[0] for d in box_dimensions]) if box_dimensions else 0
        avg_h = np.mean([d[1] for d in box_dimensions]) if box_dimensions else 0
        results_summary.append({
            "config": cfg["name"],
            "conf": conf,
            "raw_total": total_boxes_before_nms,
            "nms_total": total_boxes_after_nms,
            "avg_w": round(avg_w, 1),
            "avg_h": round(avg_h, 1),
            "macro_boxes": is_macro_box_count
        })
        print(f"  conf={conf:.2f} -> Raw: {total_boxes_before_nms} | Post-NMS: {total_boxes_after_nms} | Avg Box: {avg_w:.1f}x{avg_h:.1f} | Macro Boxes: {is_macro_box_count}")

with open(f"{OUT_DIR}/results_summary.json", "w") as f:
    json.dump(results_summary, f, indent=2)

print("\nFinished diagnostic tests.")
