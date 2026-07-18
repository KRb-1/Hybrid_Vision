from __future__ import annotations

import argparse
import os
from pathlib import Path

import cv2


THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent

def visualize_predictions(image_dir, predictions_file, output_dir):
    """
    Read prediction file, draw bounding boxes on images, and save to output directory.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Group predictions by filename
    preds_dict = {}
    with open(predictions_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 7:
                filename, x1, y1, x2, y2, conf, cls_id = parts
                preds_dict.setdefault(filename, []).append(
                    [float(x1), float(y1), float(x2), float(y2), float(conf)]
                )

    print(f"Starting visualization, found {len(preds_dict)} images with predictions...")
    
    for filename, boxes in preds_dict.items():
        img_path = os.path.join(image_dir, filename)
        if not os.path.exists(img_path):
            print(f"Could not find image: {img_path}, skipping.")
            continue

        img = cv2.imread(img_path)
        
        for box in boxes:
            x1, y1, x2, y2, conf = [int(v) if i < 4 else v for i, v in enumerate(box)]
            
            # Draw box (B, G, R)
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Write confidence score
            label = f"Drone: {conf:.2f}"
            cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        out_path = os.path.join(output_dir, filename)
        cv2.imwrite(out_path, img)

    print(f"Visualization complete! Images saved to: {output_dir}")

def parse_args():
    parser = argparse.ArgumentParser(description="Draw late-fusion prediction boxes on RGB images.")
    parser.add_argument("--image-dir", type=Path, default=PROJECT_ROOT / "FRED_ready" / "val" / "images")
    parser.add_argument("--predictions", type=Path, default=THIS_DIR / "predictions_fused.txt")
    parser.add_argument("--out", type=Path, default=THIS_DIR / "output_visuals")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    visualize_predictions(str(args.image_dir), str(args.predictions), str(args.out))
