from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def draw_detections(image_path: Path, detections: list[dict], output_path: Path) -> None:
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    colors = {"early_fusion": "#9b59b6", "two_fusion": "#ff3b30", "late_fusion": "#00a86b", "fused": "#00a86b"}
    for detection in detections:
        box = detection["bbox_xyxy"]
        color = colors.get(detection["source"], "#ff3b30")
        draw.rectangle(box, outline=color, width=3)
        label = f'{detection["class_name"]} {detection["confidence"]:.2f}'
        left, top = box[0], max(0, box[1] - 14)
        draw.rectangle((left, top, left + max(70, len(label) * 7), top + 14), fill=color)
        draw.text((left + 2, top + 1), label, fill="white", font=font)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path, quality=92)
