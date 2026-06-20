"""
YOLOv12 Optimized Inference
- MKL_DEBUG_CPU_TYPE=5: forces Intel MKL to use AVX2 path on AMD CPU (~3x speedup)
- Input resized to 384x384 before inference (smaller feature map = faster)
- Bounding box coordinates scaled back to original image dimensions
- Crops of each detected object saved from the ORIGINAL full-resolution image
- Full CPU/RAM/latency resource tracking per image
"""

import os
import time
import json
import threading
import statistics
from pathlib import Path
from datetime import datetime

# ── AMD MKL fix — must be set before torch is imported ────────────────────────
os.environ["MKL_DEBUG_CPU_TYPE"] = "5"
os.environ["OMP_NUM_THREADS"] = "4"

import cv2
import numpy as np
import psutil
from PIL import Image

# ── Config ────────────────────────────────────────────────────────────────────
BASE_DIR       = Path(__file__).parent
INPUT_DIR      = BASE_DIR / "data" / "YOLO-test-images"
SYNTH_DIR      = INPUT_DIR / "_generated"
OUTPUT_DIR     = BASE_DIR / "bbox_results_optimized"
CROPS_DIR      = OUTPUT_DIR / "crops"

MODELS         = ["yolo12m.pt", "yolo12l.pt"]
INFER_SIZE     = 384          # image is resized to this before inference
CONF_THRESHOLD = 0.25
# ──────────────────────────────────────────────────────────────────────────────

PALETTE = [
    (255, 56, 56), (255, 157, 151), (255, 112, 31), (255, 178, 29),
    (207, 210, 49), (72, 249, 10), (146, 204, 23), (61, 219, 134),
    (26, 147, 52), (0, 212, 187), (44, 153, 168), (0, 194, 255),
    (52, 69, 147), (100, 115, 255), (0, 24, 236), (132, 56, 255),
    (82, 0, 133), (203, 56, 255), (255, 149, 200), (255, 55, 199),
]


def color_for(cls_id: int) -> tuple:
    return PALETTE[int(cls_id) % len(PALETTE)]


# ── Resource sampler ──────────────────────────────────────────────────────────

class ResourceSampler:
    def __init__(self):
        self._proc = psutil.Process()
        self._stop = threading.Event()
        self.cpu_samples: list = []
        self.rss_samples: list = []

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
            "cpu_mean_pct": round(statistics.mean(cpu), 1),
            "cpu_peak_pct": round(max(cpu), 1),
            "ram_mean_mib": round(statistics.mean(rss), 1),
            "ram_peak_mib": round(max(rss), 1),
        }


def system_snapshot() -> dict:
    import torch
    vm = psutil.virtual_memory()
    try:
        freq_mhz = round(psutil.cpu_freq().current)
    except Exception:
        freq_mhz = "N/A"
    return {
        "timestamp":          datetime.now().isoformat(timespec="seconds"),
        "cpu_model":          open("/proc/cpuinfo").read().split("model name")[1].split("\n")[0].strip(": "),
        "cpu_logical_cores":  psutil.cpu_count(logical=True),
        "cpu_physical_cores": psutil.cpu_count(logical=False),
        "cpu_freq_mhz":       freq_mhz,
        "ram_total_gib":      round(vm.total / 1024**3, 2),
        "ram_available_gib":  round(vm.available / 1024**3, 2),
        "torch_threads":      torch.get_num_threads(),
        "mkl_available":      torch.backends.mkl.is_available(),
        "mkldnn_available":   torch.backends.mkldnn.is_available(),
        "mkl_amd_fix":        os.environ.get("MKL_DEBUG_CPU_TYPE") == "5",
        "infer_size":         INFER_SIZE,
    }


# ── Image helpers ─────────────────────────────────────────────────────────────

def load_bgr(path: Path) -> np.ndarray:
    img = cv2.imread(str(path))
    if img is None:
        pil = Image.open(path).convert("RGB")
        img = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
    return img


def resize_for_inference(img_bgr: np.ndarray, size: int) -> np.ndarray:
    """Resize to size×size (squash, no letterbox). Letterbox is done inside YOLO."""
    return cv2.resize(img_bgr, (size, size), interpolation=cv2.INTER_LINEAR)


def scale_boxes_to_original(boxes_384, orig_w: int, orig_h: int, infer_size: int):
    """
    boxes_384: list of [x1, y1, x2, y2] in infer_size coordinate space.
    Returns scaled coordinates clamped to original image dimensions.
    """
    sx = orig_w / infer_size
    sy = orig_h / infer_size
    scaled = []
    for x1, y1, x2, y2 in boxes_384:
        scaled.append((
            max(0, int(x1 * sx)),
            max(0, int(y1 * sy)),
            min(orig_w, int(x2 * sx)),
            min(orig_h, int(y2 * sy)),
        ))
    return scaled


# ── Drawing ───────────────────────────────────────────────────────────────────

def draw_boxes_on_original(orig_bgr: np.ndarray, scaled_boxes, labels, confs) -> np.ndarray:
    out = orig_bgr.copy()
    h, w = out.shape[:2]
    fs = max(0.5, min(w, h) / 1000)
    th = max(1, int(min(w, h) / 400))
    for i, (x1, y1, x2, y2) in enumerate(scaled_boxes):
        color = color_for(i)
        label = f"{labels[i]} {confs[i]:.2f}"
        cv2.rectangle(out, (x1, y1), (x2, y2), color, th)
        (tw, text_h), bl = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, fs, th)
        ty = max(y1 - 4, text_h + 4)
        cv2.rectangle(out, (x1, ty - text_h - bl - 4), (x1 + tw + 4, ty), color, -1)
        cv2.putText(out, label, (x1 + 2, ty - bl - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, fs, (255, 255, 255), th, cv2.LINE_AA)
    return out


def add_banner(img_bgr: np.ndarray, text: str) -> np.ndarray:
    h, w = img_bgr.shape[:2]
    bh = max(28, h // 18)
    banner = np.zeros((bh, w, 3), dtype=np.uint8)
    cv2.putText(banner, text, (6, bh - 8),
                cv2.FONT_HERSHEY_SIMPLEX, max(0.35, bh / 60),
                (200, 200, 200), 1, cv2.LINE_AA)
    return np.vstack([img_bgr, banner])


# ── Core inference ────────────────────────────────────────────────────────────

def collect_images() -> list:
    imgs = [p for p in INPUT_DIR.iterdir()
            if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
            and not p.is_dir()]
    generated = list(SYNTH_DIR.glob("*.jpg")) if SYNTH_DIR.exists() else []
    return sorted(imgs + generated)


def run_model(model_name: str, image_paths: list) -> list:
    import torch
    from ultralytics import YOLO

    torch.set_num_threads(4)

    print(f"\n{'='*70}")
    print(f"  Model: {model_name}  |  infer_size={INFER_SIZE}  |  MKL_AMD_FIX=ON")
    print(f"{'='*70}")

    model     = YOLO(model_name)
    cls_names = model.names
    out_dir   = OUTPUT_DIR / model_name.replace(".pt", "")
    out_dir.mkdir(parents=True, exist_ok=True)

    sampler = ResourceSampler()
    summary = []

    print(f"  {'Image':<40} {'Orig res':>11}  {'Det':>4}  "
          f"{'ms':>7}  {'CPU%pk':>7}  {'RAM pk MiB':>11}")
    print("  " + "-"*88)

    for img_path in image_paths:
        try:
            # 1. Load original at full resolution
            orig_bgr = load_bgr(img_path)
            orig_h, orig_w = orig_bgr.shape[:2]

            # 2. Resize to INFER_SIZE × INFER_SIZE for inference
            small_bgr = resize_for_inference(orig_bgr, INFER_SIZE)

            # 3. Run inference on the small image (timed)
            sampler.start()
            t0 = time.perf_counter()
            results = model(small_bgr, imgsz=INFER_SIZE, conf=CONF_THRESHOLD, verbose=False)
            elapsed_ms = (time.perf_counter() - t0) * 1000
            sampler.stop()
            res_stats = sampler.stats

            result = results[0]
            boxes_raw   = []   # in INFER_SIZE space
            cls_ids     = []
            confs_list  = []
            det_labels  = []

            if result.boxes and len(result.boxes):
                for box in result.boxes:
                    x1, y1, x2, y2 = map(float, box.xyxy[0].tolist())
                    cid  = int(box.cls[0])
                    conf = float(box.conf[0])
                    boxes_raw.append((x1, y1, x2, y2))
                    cls_ids.append(cid)
                    confs_list.append(conf)
                    det_labels.append({"class": cls_names[cid], "conf": round(conf, 3)})

            n_det = len(boxes_raw)

            # 4. Scale boxes from INFER_SIZE space → original image space
            scaled_boxes = scale_boxes_to_original(boxes_raw, orig_w, orig_h, INFER_SIZE)

            # 5. Draw boxes on the ORIGINAL full-resolution image
            labels_str = [cls_names[c] for c in cls_ids]
            annotated  = draw_boxes_on_original(orig_bgr, scaled_boxes, labels_str, confs_list)
            banner_txt = (f"Model:{model_name}  Input:{INFER_SIZE}x{INFER_SIZE}→orig {orig_w}x{orig_h}"
                          f"  Det:{n_det}  Infer:{elapsed_ms:.1f}ms")
            annotated  = add_banner(annotated, banner_txt)

            # Save annotated image
            out_path = out_dir / (img_path.stem + "_bbox.jpg")
            cv2.imwrite(str(out_path), annotated, [cv2.IMWRITE_JPEG_QUALITY, 95])

            # 6. Crop each detected object from the ORIGINAL image and save
            crop_records = []
            if scaled_boxes:
                crop_subdir = CROPS_DIR / model_name.replace(".pt","") / img_path.stem
                crop_subdir.mkdir(parents=True, exist_ok=True)
                for i, (x1, y1, x2, y2) in enumerate(scaled_boxes):
                    crop = orig_bgr[y1:y2, x1:x2]
                    if crop.size == 0:
                        continue
                    crop_name = f"{img_path.stem}_det{i:02d}_{labels_str[i]}_{confs_list[i]:.2f}.jpg"
                    crop_path = crop_subdir / crop_name
                    cv2.imwrite(str(crop_path), crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
                    crop_records.append({
                        "class":      labels_str[i],
                        "conf":       round(confs_list[i], 3),
                        "box_infer":  list(map(int, boxes_raw[i])),
                        "box_orig":   [x1, y1, x2, y2],
                        "crop_wh":    [x2 - x1, y2 - y1],
                        "crop_file":  str(crop_path),
                    })

            print(f"  {img_path.name:<40} {orig_w}x{orig_h:>5}  {n_det:>4}  "
                  f"{elapsed_ms:>7.1f}  {res_stats['cpu_peak_pct']:>7.1f}  "
                  f"{res_stats['ram_peak_mib']:>11.1f}")

            summary.append({
                "image":         img_path.name,
                "orig_res":      f"{orig_w}x{orig_h}",
                "infer_res":     f"{INFER_SIZE}x{INFER_SIZE}",
                "model":         model_name,
                "detections":    n_det,
                "crops":         crop_records,
                "latency_ms":    round(elapsed_ms, 1),
                "cpu_mean_pct":  res_stats["cpu_mean_pct"],
                "cpu_peak_pct":  res_stats["cpu_peak_pct"],
                "ram_mean_mib":  res_stats["ram_mean_mib"],
                "ram_peak_mib":  res_stats["ram_peak_mib"],
                "annotated":     str(out_path),
            })

        except Exception as exc:
            import traceback
            print(f"  [ERROR] {img_path.name}: {exc}")
            traceback.print_exc()
            summary.append({"image": img_path.name, "model": model_name, "error": str(exc)})

    return summary


def print_summary(all_results: list, sysinfo: dict):
    valid = [r for r in all_results if "error" not in r]
    if not valid:
        return

    print(f"\n{'='*110}")
    print("FULL RESOURCE SUMMARY")
    print(f"  {sysinfo['cpu_model']}")
    print(f"  {sysinfo['cpu_logical_cores']} logical / {sysinfo['cpu_physical_cores']} physical cores"
          f"  @{sysinfo['cpu_freq_mhz']} MHz")
    print(f"  RAM: {sysinfo['ram_total_gib']} GiB total / {sysinfo['ram_available_gib']} GiB free at start")
    print(f"  PyTorch threads: {sysinfo['torch_threads']}  |  MKL: {sysinfo['mkl_available']}"
          f"  |  MKL AMD fix: {sysinfo['mkl_amd_fix']}  |  Infer size: {sysinfo['infer_size']}x{sysinfo['infer_size']}")
    print()
    print(f"  {'Model':<14} {'Image':<40} {'Orig res':>11}  {'Det':>4}  "
          f"{'ms':>7}  {'CPU mean%':>10}  {'CPU peak%':>10}  {'RAM mean':>10}  {'RAM peak':>10}")
    print("  " + "-"*120)
    for r in valid:
        print(f"  {r['model']:<14} {r['image']:<40} {r['orig_res']:>11}  {r['detections']:>4}  "
              f"{r['latency_ms']:>7.1f}  {r['cpu_mean_pct']:>10.1f}  {r['cpu_peak_pct']:>10.1f}  "
              f"{r['ram_mean_mib']:>10.1f}  {r['ram_peak_mib']:>10.1f}")

    # Per-model averages
    print()
    for mname in MODELS:
        rows = [r for r in valid if r["model"] == mname]
        if not rows:
            continue
        lats = [r["latency_ms"] for r in rows]
        print(f"  {mname}  avg={statistics.mean(lats):.1f}ms  "
              f"median={statistics.median(lats):.1f}ms  "
              f"min={min(lats):.1f}ms  max={max(lats):.1f}ms")

    print(f"{'='*110}")


def main():
    print("=== YOLOv12 Optimized Inference (384×384 input + MKL AMD fix) ===\n")

    import torch
    torch.set_num_threads(4)

    sysinfo    = system_snapshot()
    all_images = collect_images()

    print(f"System: {sysinfo['cpu_model']}")
    print(f"        {sysinfo['cpu_logical_cores']} logical / {sysinfo['cpu_physical_cores']} physical"
          f" @ {sysinfo['cpu_freq_mhz']} MHz  |  RAM {sysinfo['ram_available_gib']} GiB free")
    print(f"        MKL AMD fix: {sysinfo['mkl_amd_fix']}  |  threads: {sysinfo['torch_threads']}")
    print(f"\nImages: {len(all_images)}  |  Infer size: {INFER_SIZE}×{INFER_SIZE}\n")

    all_results = []
    for model_name in MODELS:
        rows = run_model(model_name, all_images)
        all_results.extend(rows)

    print_summary(all_results, sysinfo)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = OUTPUT_DIR / f"results_optimized_{ts}.json"
    out_path.write_text(json.dumps({"system": sysinfo, "results": all_results}, indent=2))
    print(f"\nAnnotated images → {OUTPUT_DIR}/")
    print(f"Object crops     → {CROPS_DIR}/")
    print(f"JSON results     → {out_path}")


if __name__ == "__main__":
    main()
