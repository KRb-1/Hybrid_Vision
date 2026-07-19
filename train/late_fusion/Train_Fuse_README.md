1. enter this workspace and create environment:

```
cd ami_late_fusion
python3 -m venv ami26
(mac/linux)source ami26/bin/activate 
(windows cmd)ami26\Scripts\activate.bat
pip install -r requirements.txt
```

2. train rgb model:
```
yolo detect train data=fred_rgb.yaml model=yolov8m-p2.yaml pretrained=../yolov8m.pt epochs=30 imgsz=1280 batch=16 save_period=5 device=0
```

train event model:
```
yolo detect train data=fred_event.yaml model=yolov8m-p2.yaml pretrained=../yolov8m.pt epochs=30 imgsz=1280 batch=16 save_period=5 device=0
```

    hint: 
    If `yolov8m-p2.yaml` is not available, use `model=yolov8m.pt`.
    If GPU memory is not enough, reduce the batch size to 8.

3. run `inference_single_model.py`

    By default, the script reads `ami_late_fusion/rgb_official_best.pt`
    and `ami_late_fusion/event_official_best.pt`.
    The default data paths are `../FRED_ready/val/images` and `../FRED_ready/val/events`.
    Override the paths with arguments if your local layout is different:

```
python inference_single_model.py \
  --rgb-weights path/to/rgb_best.pt \
  --event-weights path/to/event_best.pt \
  --rgb-dir ../FRED_ready/val/images \
  --event-dir ../FRED_ready/val/events
```

4. run `fuse_predictions_nms.py`
    
    Both `fuse_predictions.py` and `fuse_predictions_nms.py` currently implement
    NMS-style late fusion.

5. run `evaluate_map.py`
    Evaluate the prediction file against the dataset labels.

6. run `visualize_boxes.py`
    Visualize predicted bounding boxes.
