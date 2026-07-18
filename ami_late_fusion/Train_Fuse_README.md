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
    如果报错，把model=yolov8m-p2.yaml pretrained=yolov8m.pt改成model=yolov8m.pt 
    如果显存不够，batch=8

3. run `inference_single_model.py`

    默认读取 `ami_late_fusion/rgb_official_best.pt` 和 `ami_late_fusion/event_official_best.pt`，
    数据默认读取 `../FRED_ready/val/images` 和 `../FRED_ready/val/events`。
    如果路径不同，用参数覆盖：

```
python inference_single_model.py \
  --rgb-weights path/to/rgb_best.pt \
  --event-weights path/to/event_best.pt \
  --rgb-dir ../FRED_ready/val/images \
  --event-dir ../FRED_ready/val/events
```

4. run `fuse_predictions_nms.py`
    
    `fuse_predictions.py` 和 `fuse_predictions_nms.py` 当前都是 NMS-style late fusion。

5. run `evaluate_map.py`
    从数据上评估预测的结果

6. run `visualize_boxes.py`
    可视化预测结果
