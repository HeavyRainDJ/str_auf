import csv
import json
import math
import time
from pathlib import Path

import numpy as np
import psutil
from ai_edge_litert.interpreter import Interpreter
from huggingface_hub import hf_hub_download
from PIL import Image, ImageDraw
from scipy.ndimage import maximum_filter


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "lines"
OUT = ROOT / "results" / "lines" / "mlsd"
OUT.mkdir(parents=True, exist_ok=True)

MODEL_ID = "litert-community/M-LSD-tiny-LiteRT"
ANGLE_TOLERANCE_DEG = 3.0
MIN_LENGTH_512 = 20.0


def angle_of(line):
    x1, y1, x2, y2 = line
    return math.degrees(math.atan2(y2 - y1, x2 - x1)) % 180.0


def angle_distance(a, b):
    d = abs(a - b) % 180.0
    return min(d, 180.0 - d)


def length_of(line):
    x1, y1, x2, y2 = line
    return math.hypot(x2 - x1, y2 - y1)


def parallel_pairs(lines):
    pairs = []
    for i in range(len(lines)):
        a = lines[i]
        aa = angle_of(a)
        ax, ay = (a[0] + a[2]) / 2, (a[1] + a[3]) / 2
        for j in range(i + 1, len(lines)):
            b = lines[j]
            if angle_distance(aa, angle_of(b)) > ANGLE_TOLERANCE_DEG:
                continue
            bx, by = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
            # Разные, а не совпадающие отрезки: расстояние между серединами
            # по нормали к направлению должно быть заметным.
            normal = math.radians(aa + 90.0)
            normal_distance = abs((bx - ax) * math.cos(normal) + (by - ay) * math.sin(normal))
            if normal_distance < 4.0:
                continue
            pairs.append((i, j, angle_distance(aa, angle_of(b)), normal_distance))
    return pairs


def decode(tp):
    center = 1 / (1 + np.exp(-tp[0]))
    disp = tp[1:5]
    peak = (center == maximum_filter(center, 3)) & (center > 0.10)
    ys, xs = np.where(peak)
    order = center[ys, xs].argsort()[::-1][:200]
    lines = []
    for y, x0 in zip(ys[order], xs[order]):
        dxs, dys, dxe, dye = disp[:, y, x0]
        line = [(x0 + dxs) * 2, (y + dys) * 2, (x0 + dxe) * 2, (y + dye) * 2]
        if length_of(line) > MIN_LENGTH_512:
            lines.append([float(v) for v in line])
    return lines


def main():
    model_path = hf_hub_download(MODEL_ID, "mlsd_fp16.tflite")
    rss0 = psutil.Process().memory_info().rss / 2**20
    t0 = time.perf_counter()
    interpreter = Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    load_s = time.perf_counter() - t0
    rss1 = psutil.Process().memory_info().rss / 2**20
    input_info = interpreter.get_input_details()[0]
    output_info = interpreter.get_output_details()[0]

    rows = []
    for image_path in sorted(DATA.iterdir()):
        if image_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        original = Image.open(image_path).convert("RGB")
        resized = original.resize((512, 512))
        a = np.asarray(resized, np.float32)
        a = np.concatenate([a, np.ones((512, 512, 1), np.float32)], axis=-1)
        x = ((a.transpose(2, 0, 1)[None] / 127.5) - 1.0).copy()
        if input_info["dtype"] == np.float16:
            x = x.astype(np.float16)

        started = time.perf_counter()
        interpreter.set_tensor(input_info["index"], x)
        interpreter.invoke()
        tp = interpreter.get_tensor(output_info["index"])[0]
        inference_s = time.perf_counter() - started

        lines = decode(tp)
        pairs = parallel_pairs(lines)
        sx, sy = original.width / 512.0, original.height / 512.0
        scaled = [[p[0] * sx, p[1] * sy, p[2] * sx, p[3] * sy] for p in lines]

        canvas = original.copy()
        draw = ImageDraw.Draw(canvas)
        width = max(2, original.width // 600)
        for line in scaled:
            draw.line(line, fill=(20, 180, 60), width=width)
        highlighted = set()
        for i, j, _, _ in sorted(pairs, key=lambda v: (v[2], -min(length_of(lines[v[0]]), length_of(lines[v[1]]))))[:12]:
            highlighted.update((i, j))
        for index in highlighted:
            draw.line(scaled[index], fill=(255, 20, 20), width=width * 2)
        draw.rectangle((0, 0, min(original.width, 900), 30), fill="white")
        draw.text((5, 6), f"M-LSD: segments={len(lines)}, parallel pairs={len(pairs)}, t={inference_s:.3f}s", fill="black")
        canvas.save(OUT / image_path.name)

        rows.append(
            {
                "image": image_path.name,
                "segments": len(lines),
                "parallel_pairs": len(pairs),
                "parallel_found": bool(pairs),
                "inference_s": inference_s,
            }
        )

    summary = {
        "model": MODEL_ID,
        "images": len(rows),
        "images_with_parallel_pair": sum(r["parallel_found"] for r in rows),
        "recall_on_positive_images": sum(r["parallel_found"] for r in rows) / len(rows),
        "mean_inference_s": sum(r["inference_s"] for r in rows) / len(rows),
        "load_s": load_s,
        "rss_increment_mb": max(0.0, rss1 - rss0),
        "angle_tolerance_deg": ANGLE_TOLERANCE_DEG,
    }
    with (OUT / "details.json").open("w", encoding="utf-8") as f:
        json.dump({"summary": summary, "rows": rows}, f, ensure_ascii=False, indent=2)
    with (OUT / "summary.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"summary": summary, "rows": rows}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
