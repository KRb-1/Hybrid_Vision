from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


THIS_DIR = Path(__file__).resolve().parent

def calculate_iou(boxA, boxB):
    """Calculate the Intersection over Union (IoU) of two bounding boxes."""
    # Determine the coordinates of the intersection
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # Calculate the area of intersection
    interArea = max(0, xB - xA) * max(0, yB - yA)

    # Calculate the area of both bounding boxes
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    # Calculate the IoU
    iou = interArea / float(boxAArea + boxBArea - interArea + 1e-6)
    return iou

def nms_fusion(boxes, scores, iou_threshold=0.5):
    """
    Perform Non-Maximum Suppression (NMS) on a set of bounding boxes.
    boxes: list, each element is [x1, y1, x2, y2]
    scores: list, corresponding confidence scores
    Return: list of indices of kept boxes
    """
    if len(boxes) == 0:
        return []

    boxes = np.array(boxes)
    scores = np.array(scores)
    
    # 1. Sort by confidence score in descending order
    order = scores.argsort()[::-1]
    keep = []

    while order.size > 0:
        i = order[0] # Pick the box with the highest confidence
        keep.append(i)
        
        if order.size == 1:
            break
            
        # 2. Calculate IoU of the highest-score box with the remaining boxes
        ious = np.array([calculate_iou(boxes[i], boxes[j]) for j in order[1:]])
        
        # 3. Keep boxes with IoU less than the threshold for the next iteration, 
        # and discard redundant boxes with IoU greater than the threshold
        inds = np.where(ious <= iou_threshold)[0]
        order = order[inds + 1]

    return keep

def load_predictions(path: Path) -> dict[str, list[list[float]]]:
    preds = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 7:
                continue
            filename, x1, y1, x2, y2, conf, cls_id = parts
            preds.setdefault(filename, []).append([float(x1), float(y1), float(x2), float(y2), float(conf)])
    return preds


def parse_args():
    parser = argparse.ArgumentParser(description="Fuse RGB and Event prediction txt files with NMS.")
    parser.add_argument("--rgb-preds", type=Path, default=THIS_DIR / "predictions_rgb.txt")
    parser.add_argument("--event-preds", type=Path, default=THIS_DIR / "predictions_event.txt")
    parser.add_argument("--out", type=Path, default=THIS_DIR / "predictions_fused.txt")
    parser.add_argument("--iou", type=float, default=0.45)
    return parser.parse_args()


def main():
    args = parse_args()
    # 1. Read single-model prediction results
    rgb_preds = load_predictions(args.rgb_preds)
    event_preds = load_predictions(args.event_preds)

    # 2. Perform post-fusion
    fused_results = []
    
    # Get the set of all test images
    all_filenames = set(rgb_preds.keys()).union(set(event_preds.keys()))
    
    print("Starting dual-modality prediction fusion...")
    for filename in all_filenames:
        boxes_for_this_img = []
        scores_for_this_img = []
        
        # Add RGB model boxes to the candidate pool
        for pred in rgb_preds.get(filename, []):
            boxes_for_this_img.append(pred[:4])
            scores_for_this_img.append(pred[4])
            
        # Add Event model boxes to the candidate pool
        for pred in event_preds.get(filename, []):
            boxes_for_this_img.append(pred[:4])
            scores_for_this_img.append(pred[4])
            
        # Perform NMS fusion to remove duplicates (IoU > 0.45 treated as the same target)
        keep_indices = nms_fusion(boxes_for_this_img, scores_for_this_img, iou_threshold=args.iou)
        
        # Record final fused results
        for idx in keep_indices:
            b = boxes_for_this_img[idx]
            conf = scores_for_this_img[idx]
            fused_results.append(f"{filename} {b[0]} {b[1]} {b[2]} {b[3]} {conf} 0") # Assume category is always drone (0)

    # 3. Save fused results
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        for line in fused_results:
            f.write(line + '\n')
            
    print(f"Fusion complete! Final predictions saved to: {args.out}")
    print(f"Total raw RGB prediction boxes: {sum(len(v) for v in rgb_preds.values())}")
    print(f"Total raw Event prediction boxes: {sum(len(v) for v in event_preds.values())}")
    print(f"Total boxes after fusion: {len(fused_results)}")

if __name__ == "__main__":
    main()
