import csv
import gc
import json
import time
from pathlib import Path

import psutil
import torch
from PIL import Image, ImageDraw, ImageFont
from transformers import AutoImageProcessor, AutoModelForObjectDetection


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "cup"
OUT = ROOT / "results" / "cup"
OUT.mkdir(parents=True, exist_ok=True)

MODELS = [
    "facebook/detr-resnet-50",
    "hustvl/yolos-tiny",
    "PekingU/rtdetr_r18vd_coco_o365",
]

# Экспертная разметка наличия чашки на выданных изображениях.
# Бокалы/стаканы в 0.jpg не считаются классом COCO "cup".
GROUND_TRUTH = {"0.jpg": False, "1.jpg": False, "3.jpg": True, "4.jpg": True, "5.jpg": False}
THRESHOLD = 0.30


def rss_mb() -> float:
    return psutil.Process().memory_info().rss / (1024 * 1024)


def draw_detections(image: Image.Image, detections: list[dict], caption: str) -> Image.Image:
    canvas = image.copy()
    draw = ImageDraw.Draw(canvas)
    for item in detections:
        x1, y1, x2, y2 = item["box"]
        draw.rectangle((x1, y1, x2, y2), outline="red", width=max(2, image.width // 300))
        draw.text((x1 + 3, max(2, y1 - 16)), f"cup {item['score']:.2f}", fill="red")
    draw.rectangle((0, 0, min(canvas.width, 760), 28), fill="white")
    draw.text((5, 5), caption, fill="black")
    return canvas


def main() -> None:
    all_rows = []
    summary = []

    for model_id in MODELS:
        slug = model_id.replace("/", "__")
        model_out = OUT / slug
        model_out.mkdir(exist_ok=True)

        mem_before = rss_mb()
        t0 = time.perf_counter()
        processor = AutoImageProcessor.from_pretrained(model_id)
        model = AutoModelForObjectDetection.from_pretrained(model_id)
        model.eval()
        load_s = time.perf_counter() - t0
        mem_loaded = rss_mb()

        model_rows = []
        for image_path in sorted(DATA.iterdir()):
            if image_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                continue
            image = Image.open(image_path).convert("RGB")
            inputs = processor(images=image, return_tensors="pt")
            started = time.perf_counter()
            with torch.inference_mode():
                outputs = model(**inputs)
            elapsed = time.perf_counter() - started
            target_sizes = torch.tensor([image.size[::-1]])
            result = processor.post_process_object_detection(
                outputs, threshold=THRESHOLD, target_sizes=target_sizes
            )[0]

            detections = []
            for score, label, box in zip(result["scores"], result["labels"], result["boxes"]):
                name = model.config.id2label[int(label)]
                if name.lower() == "cup":
                    detections.append(
                        {"score": float(score), "box": [round(float(v), 1) for v in box]}
                    )

            predicted = bool(detections)
            expected = GROUND_TRUTH[image_path.name]
            row = {
                "model": model_id,
                "image": image_path.name,
                "expected": expected,
                "predicted": predicted,
                "correct": predicted == expected,
                "cup_count": len(detections),
                "max_score": max((d["score"] for d in detections), default=0.0),
                "inference_s": elapsed,
                "detections": detections,
            }
            model_rows.append(row)
            all_rows.append(row)
            caption = f"{model_id}; cup={predicted}; t={elapsed:.2f}s"
            draw_detections(image, detections, caption).save(model_out / image_path.name)

        tp = sum(r["expected"] and r["predicted"] for r in model_rows)
        tn = sum((not r["expected"]) and (not r["predicted"]) for r in model_rows)
        fp = sum((not r["expected"]) and r["predicted"] for r in model_rows)
        fn = sum(r["expected"] and (not r["predicted"]) for r in model_rows)
        summary.append(
            {
                "model": model_id,
                "threshold": THRESHOLD,
                "tp": tp,
                "tn": tn,
                "fp": fp,
                "fn": fn,
                "accuracy": (tp + tn) / len(model_rows),
                "precision": tp / (tp + fp) if tp + fp else 0.0,
                "recall": tp / (tp + fn) if tp + fn else 0.0,
                "mean_inference_s": sum(r["inference_s"] for r in model_rows) / len(model_rows),
                "load_s": load_s,
                "rss_increment_mb": max(0.0, mem_loaded - mem_before),
            }
        )

        del model, processor
        gc.collect()

    with (OUT / "detections.json").open("w", encoding="utf-8") as f:
        json.dump(all_rows, f, ensure_ascii=False, indent=2)

    fields = list(summary[0].keys())
    with (OUT / "summary.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summary)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
