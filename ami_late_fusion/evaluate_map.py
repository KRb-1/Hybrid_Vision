from __future__ import annotations

import argparse
import os
from pathlib import Path

import cv2
import numpy as np


THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent

def calculate_iou(boxA, boxB):
    """Calculate Intersection over Union (IoU)"""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    return interArea / float(boxAArea + boxBArea - interArea + 1e-6)

def load_ground_truth(label_dir, image_dir):
    """Load original YOLO labels and convert to absolute coordinates [x1, y1, x2, y2]"""
    gt_dict = {}
    total_gt_boxes = 0
    
    label_files = [f for f in os.listdir(label_dir) if f.endswith('.txt')]
    for label_file in label_files:
        img_name = label_file.replace('.txt', '.jpg')
        img_path = os.path.join(image_dir, img_name)
        
        if not os.path.exists(img_path):
            continue
            
        # Read original image to get dimensions for denormalization
        img = cv2.imread(img_path)
        h, w = img.shape[:2]
        
        boxes = []
        with open(os.path.join(label_dir, label_file), 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 5:
                    cls_id, x_c, y_c, bw, bh = map(float, parts)
                    # Convert to top-left and bottom-right absolute coordinates
                    x1 = (x_c - bw / 2) * w
                    y1 = (y_c - bh / 2) * h
                    x2 = (x_c + bw / 2) * w
                    y2 = (y_c + bh / 2) * h
                    boxes.append([x1, y1, x2, y2])
                    total_gt_boxes += 1
                    
        gt_dict[img_name] = boxes
        
    return gt_dict, total_gt_boxes

def evaluate_map(predictions_file, label_dir, image_dir, iou_thresh=0.5):
    """Calculate mAP@0.5 for a single class"""
    print("Loading Ground Truth labels...")
    gt_dict, total_gt = load_ground_truth(label_dir, image_dir)
    print(f"Loading finished: {len(gt_dict)} images, {total_gt} ground truth boxes found.")
    
    # Load predictions and sort by confidence in descending order
    preds = []
    with open(predictions_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            filename, x1, y1, x2, y2, conf, cls_id = parts
            preds.append({
                'filename': filename,
                'box': [float(x1), float(y1), float(x2), float(y2)],
                'conf': float(conf)
            })
    preds.sort(key=lambda x: x['conf'], reverse=True)
    
    # Track which GT box has been matched (to avoid double counting TP)
    matched_gt = {filename: [False] * len(boxes) for filename, boxes in gt_dict.items()}
    
    TP = np.zeros(len(preds))
    FP = np.zeros(len(preds))
    
    for i, pred in enumerate(preds):
        filename = pred['filename']
        pred_box = pred['box']
        
        if filename not in gt_dict or len(gt_dict[filename]) == 0:
            FP[i] = 1 # Image has no target, but prediction was made, mark as False Positive
            continue
            
        gt_boxes = gt_dict[filename]
        best_iou = 0
        best_gt_idx = -1
        
        # Find the GT box with the highest IoU
        for j, gt_box in enumerate(gt_boxes):
            iou = calculate_iou(pred_box, gt_box)
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = j
                
        if best_iou >= iou_thresh:
            if not matched_gt[filename][best_gt_idx]:
                TP[i] = 1 # Successfully matched and not occupied, mark as True Positive
                matched_gt[filename][best_gt_idx] = True # Mark as occupied
            else:
                FP[i] = 1 # Target claimed by a higher confidence prediction, mark current as FP
        else:
            FP[i] = 1 # IoU too low, did not hit any target
            
    # Calculate cumulative TP and FP for PR curve
    acc_FP = np.cumsum(FP)
    acc_TP = np.cumsum(TP)
    
    recalls = acc_TP / total_gt if total_gt > 0 else acc_TP * 0
    precisions = acc_TP / np.maximum(acc_TP + acc_FP, np.finfo(np.float64).eps)
    
    # Calculate AP using 11-point interpolation method (VOC standard)
    ap = 0.0
    for t in np.arange(0.0, 1.1, 0.1):
        if np.sum(recalls >= t) == 0:
            p = 0
        else:
            p = np.max(precisions[recalls >= t])
        ap += p / 11.0
        
    print("\n" + "="*30)
    print("      Evaluation Result (Late Fusion)")
    print("="*30)
    print(f"Total prediction boxes: {len(preds)}")
    print(f"True Positives (TP): {int(np.sum(TP))}")
    print(f"False Positives (FP) / False Alarms: {int(np.sum(FP))}")
    print(f"False Negatives (FN): {total_gt - int(np.sum(TP))}")
    print("-" * 30)
    print(f"mAP@0.5: {ap:.4f}  ({ap*100:.1f}%)")
    print("="*30)

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate late-fusion prediction txt with VOC-style AP@0.5.")
    parser.add_argument("--predictions", type=Path, default=THIS_DIR / "predictions_fused.txt")
    parser.add_argument("--label-dir", type=Path, default=PROJECT_ROOT / "FRED_ready" / "val" / "labels")
    parser.add_argument("--image-dir", type=Path, default=PROJECT_ROOT / "FRED_ready" / "val" / "images")
    parser.add_argument("--iou", type=float, default=0.5)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    evaluate_map(args.predictions, args.label_dir, args.image_dir, iou_thresh=args.iou)
