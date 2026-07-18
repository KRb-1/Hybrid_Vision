# Early Fusion

This method corresponds to `ami_early_fusion/`. During training, RGB and Event images are packed into a 6-channel TIFF input. Final inference and visualization can read the original RGB/Event images from the project root directly, so the uploaded TIFF training set is not required for demo-time prediction.

## Method

Input channel order:

```text
[R, G, B, E_R, E_G, E_B]
```

The first convolution of YOLOv8 is expanded from 3 input channels to 6 input channels:

- RGB channels inherit the pretrained weights.
- Event channels are initialized according to the selected strategy; the default is `scratch_event`.

## Build The 6-Channel Training Data

```bash
cd ami_early_fusion

python3 scripts/build_fusion_tiff.py \
  --raw-root ../FRED_ready \
  --out-root data/fred_fusion_6ch \
  --workers 12 \
  --compression deflate
```

Check the generated dataset:

```bash
python3 scripts/verify_fusion_dataset.py \
  --data data/fred_fusion_6ch/fred_fusion_6ch.yaml

python3 scripts/check_dataset_labels.py \
  --data data/fred_fusion_6ch/fred_fusion_6ch.yaml
```

## Training

Standard training:

```bash
python3 scripts/train_early_fusion.py \
  --data data/fred_fusion_6ch/fred_fusion_6ch.yaml \
  --weights ../yolov8m.pt \
  --imgsz 960 \
  --epochs 50 \
  --batch 8 \
  --device 0
```

Recommended NaN-safe training:

```bash
python3 scripts/train_early_fusion_safe.py \
  --data data/fred_fusion_6ch/fred_fusion_6ch.yaml \
  --weights ../yolov8m.pt \
  --imgsz 1280 \
  --epochs 27 \
  --batch 16 \
  --workers 12 \
  --device 0 \
  --amp true \
  --optimizer AdamW \
  --lr0 0.0001 \
  --warmup-epochs 2 \
  --warmup-bias-lr 0.0001 \
  --mosaic 0.0 \
  --mixup 0.0 \
  --hsv-h 0.0 \
  --hsv-s 0.0 \
  --hsv-v 0.05
```

## JPG Inference And Visualization

The final validation/demo pipeline does not need TIFF inputs. The script reads the RGB/Event images from the project directory and creates the model input temporarily.

Recommended splits:

- Use `FRED_ready/val` when the metrics should match the training validation metrics.
- Use `FRED_ready/test` when showing held-out images that were not used in training or validation.

```bash
python3 scripts/inference_early_fusion.py \
  --weights "../ami_runs/early fusion/weights/best.pt" \
  --rgb-dir ../FRED_ready/test/images \
  --event-dir ../FRED_ready/test/events \
  --imgsz 1280 \
  --conf 0.25 \
  --device 0 \
  --out predictions_early_fusion.txt
```

Draw boxes on RGB JPG images:

```bash
python3 scripts/visualize_early_fusion_predictions.py \
  --image-dir ../FRED_ready/test/images \
  --predictions predictions_early_fusion.txt \
  --out reports/early_fusion_predictions
```

Quick CPU smoke test on the held-out test split, using the default threshold `conf=0.25`:

```bash
python3 scripts/inference_early_fusion.py \
  --weights "../ami_runs/early fusion/weights/best.pt" \
  --rgb-dir ../FRED_ready/test/images \
  --event-dir ../FRED_ready/test/events \
  --imgsz 320 \
  --device cpu \
  --conf 0.25 \
  --limit 30 \
  --out /tmp/predictions_early_fusion_check.txt
```

## Existing Results

`ami_runs/early fusion/results.csv`:

- Best epoch in the current continuation run: `4`
- Precision: `0.81577`
- Recall: `0.78715`
- mAP50: `0.77701`
- mAP50-95: `0.36133`

Training-history note:

- The selected model comes from a checkpoint-continuation chain in `runs_early/fred/`.
- Stage 1: `early_fusion_m1280_safe_3ep/results.csv`, 2 observed epochs, `18772.00s`.
- Stage 2: `early_fusion_from_safe2_amp_lr1e4_2ep/results.csv`, 1 observed epoch, `5072.51s`.
- Stage 3: `early_fusion_from_amp1_b16_biaslr1e4_e27/results.csv`, 4 observed epochs, `19936.60s`.
- Reported observed training cost for the selected early-fusion model: 7 epochs, `43781.11s` / `12.16h`.
- `early_fusion_from_epoch4_continue_e5_v2` ran 3 more epochs, but it did not improve the selected best result.

Tested JPG pipeline:

- Input: `FRED_ready/test/images` + `FRED_ready/test/events`
- Threshold: default `conf=0.25`
- Prediction output filenames: `.jpg`
- Visualization output: bounding boxes drawn on RGB JPG images
