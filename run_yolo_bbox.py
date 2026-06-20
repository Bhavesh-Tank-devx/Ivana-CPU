"""
YOLOv12 Bounding Box Annotator
- Reads real images from data/YOLO-test-images/
- Generates synthetic images for any missing benchmark dimensions (320, 512, 640, 800, 1024)
- Runs YOLOv12m and YOLOv12l on every image
- Saves annotated images (with bounding boxes + labels) to bbox_results/<model>/<image>
"""

import os
import sys
import json
import time
import statistics
import threading
from pathlib import Path
from datetime import datetime

import cv2
import numpy as np
import psutil
from PIL import Image, ImageDraw, ImageFont

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR    = Path(__file__).parent
INPUT_DIR   = BASE_DIR / "data" / "YOLO-test-images"
OUTPUT_DIR  = BASE_DIR / "bbox_results"
SYNTH_DIR   = INPUT_DIR / "_generated"   # generated images live here

MODELS         = ["yolo12m.pt", "yolo12l.pt"]
TARGET_DIMS    = [320, 512, 640, 800, 1024]   # benchmark dimensions we want covered
CONF_THRESHOLD = 0.25
# ──────────────────────────────────────────────────────────────────────────────

# ── Resource monitoring ───────────────────────────────────────────────────────

class ResourceSampler:
    """Polls CPU % and RSS every 100 ms in a background thread."""
    def __init__(self):
        self._proc = psutil.Process()
        self._stop = threading.Event()
        self.cpu_samples: list[float] = []
        self.rss_samples: list[float] = []

    def start(self):
        self._stop.clear()
        self.cpu_samples.clear()
        self.rss_samples.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        self._thread.join(timeout=2)

    def _run(self):
        while not self._stop.is_set():
            self.cpu_samples.append(self._proc.cpu_percent(interval=None))
            self.rss_samples.append(self._proc.memory_info().rss / 1024 / 1024)
            time.sleep(0.1)

    @property
    def stats(self) -> dict:
        cpu = self.cpu_samples or [0.0]
        rss = self.rss_samples or [0.0]
        return {
            "cpu_mean_pct":  round(statistics.mean(cpu), 1),
            "cpu_peak_pct":  round(max(cpu), 1),
            "ram_mean_mib":  round(statistics.mean(rss), 1),
            "ram_peak_mib":  round(max(rss), 1),
        }


def system_snapshot() -> dict:
    vm = psutil.virtual_memory()
    try:
        freq = psutil.cpu_freq()
        freq_mhz = round(freq.current) if freq else "N/A"
    except Exception:
        freq_mhz = "N/A"
    return {
        "timestamp":          datetime.now().isoformat(timespec="seconds"),
        "cpu_logical_cores":  psutil.cpu_count(logical=True),
        "cpu_physical_cores": psutil.cpu_count(logical=False),
        "cpu_freq_mhz":       freq_mhz,
        "ram_total_gib":      round(vm.total / 1024**3, 2),
        "ram_available_gib":  round(vm.available / 1024**3, 2),
    }


# COCO class colours (cycling palette)
PALETTE = [
    (255, 56, 56), (255, 157, 151), (255, 112, 31), (255, 178, 29),
    (207, 210, 49), (72, 249, 10), (146, 204, 23), (61, 219, 134),
    (26, 147, 52), (0, 212, 187), (44, 153, 168), (0, 194, 255),
    (52, 69, 147), (100, 115, 255), (0, 24, 236), (132, 56, 255),
    (82, 0, 133), (203, 56, 255), (255, 149, 200), (255, 55, 199),
]


def color_for_class(cls_id: int) -> tuple:
    return PALETTE[int(cls_id) % len(PALETTE)]


# ── Synthetic image generation ────────────────────────────────────────────────

def _existing_dims() -> dict[int, Path]:
    """Return {max_side: path} for every real image already in INPUT_DIR."""
    dims = {}
    for p in INPUT_DIR.iterdir():
        if p.is_dir():
            continue
        try:
            img = Image.open(p)
            key = max(img.size)
            dims[key] = p
        except Exception:
            pass
    return dims


def _pick_base_image(existing: dict[int, Path]) -> Path:
    """Return the real image with the largest resolution to use as resizing base."""
    return existing[max(existing)]


def generate_missing_images(target_dims: list[int]) -> list[Path]:
    """
    For each target dimension not covered by an existing image,
    resize the largest available real image to that square size and save it.
    Returns list of newly created paths.
    """
    SYNTH_DIR.mkdir(parents=True, exist_ok=True)
    existing = _existing_dims()

    covered = set()
    for key in existing:
        for t in target_dims:
            if abs(key - t) <= 20:   # within 20px counts as covered
                covered.add(t)

    new_paths = []
    if not existing:
        print("[WARN] No real images found to use as resize base.")
        return new_paths

    base_path = _pick_base_image(existing)
    base_img = Image.open(base_path).convert("RGB")

    for dim in sorted(target_dims):
        if dim in covered:
            continue
        out_path = SYNTH_DIR / f"generated_{dim}x{dim}.jpg"
        resized = base_img.resize((dim, dim), Image.LANCZOS)
        resized.save(out_path, quality=95)
        print(f"  Generated {dim}x{dim} from {base_path.name}  →  {out_path.name}")
        new_paths.append(out_path)

    return new_paths


# ── Annotation ────────────────────────────────────────────────────────────────

def draw_boxes(img_bgr: np.ndarray, result, model_names: list) -> np.ndarray:
    """Draw bounding boxes, class names, and confidence scores onto img_bgr."""
    annotated = img_bgr.copy()
    h, w = annotated.shape[:2]
    font_scale = max(0.4, min(w, h) / 800)
    thickness  = max(1, int(min(w, h) / 300))

    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return annotated

    for box in boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        cls_id  = int(box.cls[0])
        conf    = float(box.conf[0])
        label   = f"{model_names[cls_id]} {conf:.2f}"
        color   = color_for_class(cls_id)

        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness)

        # Label background
        (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX,
                                              font_scale, thickness)
        ty = max(y1 - 4, th + 4)
        cv2.rectangle(annotated,
                      (x1, ty - th - baseline - 4),
                      (x1 + tw + 4, ty),
                      color, -1)
        cv2.putText(annotated, label, (x1 + 2, ty - baseline - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness,
                    cv2.LINE_AA)

    return annotated


def add_info_banner(img_bgr: np.ndarray, model_name: str,
                    img_path: Path, n_det: int, elapsed_ms: float) -> np.ndarray:
    """Paste a black banner at the bottom with run metadata."""
    h, w = img_bgr.shape[:2]
    banner_h = max(28, h // 18)
    banner = np.zeros((banner_h, w, 3), dtype=np.uint8)
    text = (f"Model: {model_name}  |  Image: {img_path.name} ({w}x{h})"
            f"  |  Detections: {n_det}  |  Inference: {elapsed_ms:.1f} ms")
    fs = max(0.35, banner_h / 60)
    cv2.putText(banner, text, (6, banner_h - 8),
                cv2.FONT_HERSHEY_SIMPLEX, fs, (200, 200, 200), 1, cv2.LINE_AA)
    return np.vstack([img_bgr, banner])


# ── Main ──────────────────────────────────────────────────────────────────────

def collect_images() -> list[Path]:
    """Return all image paths (real + generated)."""
    imgs = [p for p in INPUT_DIR.iterdir()
            if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
            and not p.is_dir()]
    generated = list(SYNTH_DIR.glob("*.jpg")) if SYNTH_DIR.exists() else []
    return sorted(imgs + generated)


def run_model(model_name: str, image_paths: list[Path]) -> list[dict]:
    from ultralytics import YOLO

    print(f"\n{'='*62}")
    print(f"  Running {model_name}")
    print(f"{'='*62}")

    model     = YOLO(model_name)
    cls_names = model.names          # {0: 'person', 1: 'bicycle', ...}
    out_dir   = OUTPUT_DIR / model_name.rstrip("/").split("/")[0]
    out_dir.mkdir(parents=True, exist_ok=True)

    sampler = ResourceSampler()
    summary = []

    print(f"  {'Image':<45} {'Res':>10}  {'Det':>4}  {'ms':>8}  {'CPU%':>6}  {'RAM MiB':>9}")
    print("  " + "-"*90)

    for img_path in image_paths:
        try:
            img_bgr = cv2.imread(str(img_path))
            if img_bgr is None:
                pil = Image.open(img_path).convert("RGB")
                img_bgr = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)

            h, w = img_bgr.shape[:2]

            sampler.start()
            t0 = time.perf_counter()
            results = model(img_bgr, conf=CONF_THRESHOLD, verbose=False)
            elapsed_ms = (time.perf_counter() - t0) * 1000
            sampler.stop()
            res = sampler.stats

            result  = results[0]
            n_det   = len(result.boxes) if result.boxes else 0
            det_labels = []
            if result.boxes:
                for box in result.boxes:
                    cid  = int(box.cls[0])
                    conf = float(box.conf[0])
                    det_labels.append({"class": cls_names[cid], "conf": round(conf, 3)})

            annotated = draw_boxes(img_bgr, result, cls_names)
            annotated = add_info_banner(annotated, model_name, img_path, n_det, elapsed_ms)

            out_name = img_path.stem + "_bbox" + ".jpg"
            out_path = out_dir / out_name
            cv2.imwrite(str(out_path), annotated, [cv2.IMWRITE_JPEG_QUALITY, 95])

            print(f"  {img_path.name:<45} {w}x{h:>5}  {n_det:>4}  "
                  f"{elapsed_ms:>8.1f}  {res['cpu_peak_pct']:>6.1f}  {res['ram_peak_mib']:>9.1f}")

            summary.append({
                "image":         img_path.name,
                "resolution":    f"{w}x{h}",
                "model":         model_name,
                "detections":    n_det,
                "labels":        det_labels,
                "latency_ms":    round(elapsed_ms, 1),
                "cpu_mean_pct":  res["cpu_mean_pct"],
                "cpu_peak_pct":  res["cpu_peak_pct"],
                "ram_mean_mib":  res["ram_mean_mib"],
                "ram_peak_mib":  res["ram_peak_mib"],
                "output":        str(out_path),
            })

        except Exception as exc:
            print(f"  [ERROR] {img_path.name}: {exc}")
            summary.append({"image": img_path.name, "model": model_name, "error": str(exc)})

    return summary


def print_final_summary(all_results: list[dict], sysinfo: dict):
    valid = [r for r in all_results if "error" not in r]
    if not valid:
        return
    print(f"\n{'='*100}")
    print("RESOURCE SUMMARY")
    print(f"  CPU: {sysinfo['cpu_logical_cores']} logical / {sysinfo['cpu_physical_cores']} physical cores"
          f"  @{sysinfo['cpu_freq_mhz']} MHz")
    print(f"  RAM: {sysinfo['ram_total_gib']} GiB total, {sysinfo['ram_available_gib']} GiB free at start")
    print()
    print(f"  {'Model':<14} {'Image':<45} {'Res':>10}  {'Det':>4}  {'ms':>8}  "
          f"{'CPU mean%':>10}  {'CPU peak%':>10}  {'RAM mean MiB':>13}  {'RAM peak MiB':>13}")
    print("  " + "-"*130)
    for r in valid:
        print(f"  {r['model']:<14} {r['image']:<45} {r['resolution']:>10}  {r['detections']:>4}  "
              f"{r['latency_ms']:>8.1f}  {r['cpu_mean_pct']:>10.1f}  {r['cpu_peak_pct']:>10.1f}  "
              f"{r['ram_mean_mib']:>13.1f}  {r['ram_peak_mib']:>13.1f}")
    print(f"{'='*100}")


def main():
    print("=== YOLOv12 Bounding Box Annotator ===\n")

    sysinfo = system_snapshot()
    print(f"System: {sysinfo['cpu_logical_cores']} cores @ {sysinfo['cpu_freq_mhz']} MHz  |  "
          f"RAM {sysinfo['ram_total_gib']} GiB total / {sysinfo['ram_available_gib']} GiB free\n")

    # 1. Generate missing-dimension images
    print("Checking for missing benchmark dimensions…")
    existing = _existing_dims()
    print(f"  Found real images at dims: {sorted(existing.keys())}")
    new = generate_missing_images(TARGET_DIMS)
    if not new:
        print("  All target dimensions already covered.")

    # 2. Collect all images
    all_images = collect_images()
    print(f"\nTotal images to process: {len(all_images)}")
    for p in all_images:
        try:
            img = Image.open(p)
            print(f"  {p.name:<50} {img.size[0]}x{img.size[1]}")
        except Exception:
            print(f"  {p.name}  (unreadable)")

    # 3. Run each model
    all_results = []
    for model_name in MODELS:
        rows = run_model(model_name, all_images)
        all_results.extend(rows)

    # 4. Print and save summary
    print_final_summary(all_results, sysinfo)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_out = OUTPUT_DIR / f"detections_{ts}.json"
    payload  = {"system": sysinfo, "results": all_results}
    json_out.write_text(json.dumps(payload, indent=2))

    print(f"\nAnnotated images → {OUTPUT_DIR}/")
    print(f"Full JSON        → {json_out}")


if __name__ == "__main__":
    main()
