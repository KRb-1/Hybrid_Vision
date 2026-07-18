from __future__ import annotations

import argparse
import glob
import os
from pathlib import Path

from ultralytics import YOLO

THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent


def run_inference_and_save(model_path, data_dir, output_file, conf=0.25):
    """
    Run inference on images in a directory using a single model 
    and save results to a text file.
    """
    print(f"Loading model: {model_path}...")
    model = YOLO(model_path)
    
    # Path of images used for validation
    image_paths = sorted(glob.glob(os.path.join(data_dir, '*.jpg')))
    
    results_list = []
    
    print(f"Starting inference for {len(image_paths)} images...")
    for img_path in image_paths:
        img_name = os.path.basename(img_path)
        
        # Run inference (setting a higher conf threshold filters out obvious noise)
        results = model.predict(source=img_path, conf=conf, verbose=False)
        
        # Extract results
        for r in results:
            boxes = r.boxes
            for box in boxes:
                # Get bounding box coordinates [x1, y1, x2, y2]
                b = box.xyxy[0].cpu().numpy()
                # Get confidence
                conf = box.conf[0].cpu().numpy()
                # Get category ID
                cls_id = int(box.cls[0].cpu().numpy())
                
                # Format: filename x1 y1 x2 y2 confidence categoryID
                res_str = f"{img_name} {b[0]} {b[1]} {b[2]} {b[3]} {conf} {cls_id}"
                results_list.append(res_str)
                
    # Write results to file
    with open(output_file, 'w') as f:
        for line in results_list:
            f.write(line + '\n')
    print(f"Inference results saved to: {output_file}")


def parse_args():
    parser = argparse.ArgumentParser(description="Run RGB and Event model inference for late fusion.")
    parser.add_argument("--rgb-weights", type=Path, default=THIS_DIR / "rgb_official_best.pt")
    parser.add_argument("--event-weights", type=Path, default=THIS_DIR / "event_official_best.pt")
    parser.add_argument("--rgb-dir", type=Path, default=PROJECT_ROOT / "FRED_ready" / "val" / "images")
    parser.add_argument("--event-dir", type=Path, default=PROJECT_ROOT / "FRED_ready" / "val" / "events")
    parser.add_argument("--rgb-out", type=Path, default=THIS_DIR / "predictions_rgb.txt")
    parser.add_argument("--event-out", type=Path, default=THIS_DIR / "predictions_event.txt")
    parser.add_argument("--conf", type=float, default=0.25)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_inference_and_save(args.rgb_weights, args.rgb_dir, args.rgb_out, conf=args.conf)
    run_inference_and_save(args.event_weights, args.event_dir, args.event_out, conf=args.conf)
