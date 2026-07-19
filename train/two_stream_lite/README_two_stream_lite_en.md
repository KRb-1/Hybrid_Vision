# Two-Stream-Lite Fusion

This method corresponds to `ami_two_fusion/`. Training can still use the 6-channel TIFF dataset, but final inference and visualization support reading RGB JPG and Event PNG files directly from the project root. The uploaded TIFF dataset is not required for demo-time prediction.

## Method

The first stage of the model is a custom `RGBEventStem`:

```text
RGB   -> Conv(3 -> c_out/2)
Event -> Conv(3 -> c_out/2)
Concat -> 1x1 Conv fusion
```

The rest of the backbone and head stay as close as possible to the YOLOv8m structure.

Key files:

- `ami_two_fusion/fred_two_stream/rgb_event_stem.py`
- `ami_two_fusion/fred_two_stream/ultralytics_patch.py`
- `ami_two_fusion/configs/models/yolov8m_rgb_event_stem.yaml`
- `ami_two_fusion/scripts/train_two_stream_lite.py`
- `ami_two_fusion/scripts/inference_two_stream_lite.py`
- `ami_two_fusion/scripts/visualize_two_stream_predictions.py`

## Training

First prepare the 6-channel TIFF data. You can reuse the early-fusion dataset or generate it again inside the two-stream directory:

```bash
cd ami_two_fusion

python3 scripts/build_fusion_tiff.py \
  --raw-root ../FRED_ready \
  --out-root data/fred_fusion_6ch \
  --workers 8
```

Main training command:

```bash
python3 scripts/train_two_stream_lite.py \
  --data data/fred_fusion_6ch/fred_fusion_6ch.yaml \
  --model configs/models/yolov8m_rgb_event_stem.yaml \
  --pretrained ../yolov8m.pt \
  --imgsz 1280 \
  --epochs 30 \
  --batch 16 \
  --workers 16 \
  --device 0 \
  --amp true \
  --optimizer AdamW \
  --lr0 0.0005 \
  --cos-lr true \
  --warmup-epochs 5 \
  --warmup-bias-lr 0.0 \
  --hsv-h 0.0 \
  --hsv-s 0.0 \
  --hsv-v 0.0 \
  --project runs/fred \
  --name two_stream_lite_from_best_lr5e4_nohsv
```

Validation:

```bash
python3 scripts/val_two_stream_lite.py \
  --weights runs/fred/two_stream_lite_from_best_lr5e4_nohsv/weights/best.pt \
  --data data/fred_fusion_6ch/fred_fusion_6ch.yaml \
  --imgsz 1280 \
  --batch 16 \
  --device 0
```

## JPG Inference And Visualization

Batch inference can read images directly from the project directory.

Recommended splits:

- Use `FRED_ready/val` when the metrics should match the training validation metrics.
- Use `FRED_ready/test` when showing held-out images that were not used in training or validation.

```bash
python3 scripts/inference_two_stream_lite.py \
  --weights "../ami_runs/two-stream lite fusion/weights/best.pt" \
  --rgb-dir ../FRED_ready/test/images \
  --event-dir ../FRED_ready/test/events \
  --imgsz 1280 \
  --conf 0.25 \
  --device 0 \
  --out predictions_two_stream_lite.txt
```

Draw boxes on RGB JPG images:

```bash
python3 scripts/visualize_two_stream_predictions.py \
  --image-dir ../FRED_ready/test/images \
  --predictions predictions_two_stream_lite.txt \
  --out reports/two_stream_lite_predictions
```

Quick CPU smoke test on the held-out test split, using the default threshold `conf=0.25`:

```bash
python3 scripts/inference_two_stream_lite.py \
  --weights "../ami_runs/two-stream lite fusion/weights/best.pt" \
  --rgb-dir ../FRED_ready/test/images \
  --event-dir ../FRED_ready/test/events \
  --imgsz 320 \
  --device cpu \
  --conf 0.25 \
  --limit 30 \
  --out /tmp/predictions_two_stream_lite_check.txt
```

## Existing Results

`ami_runs/two-stream lite fusion/results.csv`:

- Best mAP50 epoch in the current continuation run: `4`
- Precision: `0.82723`
- Recall: `0.78741`
- mAP50: `0.78591`
- mAP50-95: `0.36482`
- Final epoch 5 mAP50-95: `0.36722`

Training-history note:

- The selected model comes from a checkpoint-continuation chain in `runs_two/`.
- Stage 1: `detect/runs/fred/two_stream_lite_yolov8m_1280_stable_lr001/results.csv`, 4 observed epochs, `22294.80s`.
- Stage 2: `fred/two_stream_lite_from_best_lr5e4_nohsv/results.csv`, 5 observed epochs, `28129.50s`.
- Reported observed training cost for the selected two-stream-lite model: 9 epochs, `50424.30s` / `14.01h`.
- The continuation starts from the predecessor `stable_lr001` `best.pt`.

This is currently the best-performing fusion method among the local experiments.
