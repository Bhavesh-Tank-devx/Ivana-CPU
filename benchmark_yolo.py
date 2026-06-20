"""
YOLOv12 Resource Benchmark
Measures CPU %, RAM, inference time, and throughput for YOLOv12m and YOLOv12l
across multiple image dimensions on CPU.
"""

import time
import os
import sys
import json
import statistics
import threading
import subprocess
from pathlib import Path
from datetime import datetime

import psutil
import numpy as np
from PIL import Image


# ── Config ────────────────────────────────────────────────────────────────────
MODELS = ["yolo12m.pt", "yolo12l.pt"]
DIMENSIONS = [320, 512, 640, 800, 1024]   # square sides tested
WARMUP_RUNS = 2
BENCH_RUNS = 5                             # timed runs per (model × dim)
RESULTS_DIR = Path("benchmark_results")
RESULTS_DIR.mkdir(exist_ok=True)
# ──────────────────────────────────────────────────────────────────────────────


def _build_image(size: int) -> np.ndarray:
    """Return a random uint8 BGR image of shape (size, size, 3)."""
    return np.random.randint(0, 255, (size, size, 3), dtype=np.uint8)


class ResourceSampler:
    """Background thread that polls CPU % and RSS every 100 ms."""

    def __init__(self):
        self._proc = psutil.Process()
        self._stop = threading.Event()
        self.cpu_samples: list[float] = []
        self.rss_samples: list[float] = []   # MiB
        self._thread = threading.Thread(target=self._run, daemon=True)

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
            mem = self._proc.memory_info().rss / 1024 / 1024
            self.rss_samples.append(mem)
            time.sleep(0.1)

    @property
    def stats(self) -> dict:
        cpu = self.cpu_samples or [0]
        rss = self.rss_samples or [0]
        return {
            "cpu_mean_pct":   round(statistics.mean(cpu), 1),
            "cpu_peak_pct":   round(max(cpu), 1),
            "ram_mean_mib":   round(statistics.mean(rss), 1),
            "ram_peak_mib":   round(max(rss), 1),
        }


def system_snapshot() -> dict:
    vm = psutil.virtual_memory()
    cpu_count = psutil.cpu_count(logical=True)
    cpu_phys  = psutil.cpu_count(logical=False)
    try:
        freq = psutil.cpu_freq()
        freq_mhz = round(freq.current) if freq else "N/A"
    except Exception:
        freq_mhz = "N/A"

    # GPU check
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.total,memory.free",
             "--format=csv,noheader,nounits"],
            timeout=5, stderr=subprocess.DEVNULL
        ).decode().strip()
        gpu = out.split("\n")
    except Exception:
        gpu = ["No NVIDIA GPU detected — running CPU-only"]

    return {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "cpu_logical_cores": cpu_count,
        "cpu_physical_cores": cpu_phys,
        "cpu_freq_mhz": freq_mhz,
        "ram_total_gib": round(vm.total / 1024**3, 2),
        "ram_available_gib": round(vm.available / 1024**3, 2),
        "gpu": gpu,
        "python": sys.version.split()[0],
    }


def bench_model(model_name: str, dims: list[int]) -> list[dict]:
    from ultralytics import YOLO  # imported here so error is localised

    print(f"\n{'='*60}")
    print(f"  Model: {model_name}")
    print(f"{'='*60}")

    model = YOLO(model_name)
    model.overrides["device"] = "cpu"
    model.overrides["verbose"] = False

    sampler = ResourceSampler()
    rows: list[dict] = []

    for dim in dims:
        img = _build_image(dim)
        label = f"{dim}x{dim}"

        # Warm-up (not measured)
        for _ in range(WARMUP_RUNS):
            model(img, imgsz=dim, verbose=False)

        # Timed runs
        latencies: list[float] = []
        sampler.start()
        for _ in range(BENCH_RUNS):
            t0 = time.perf_counter()
            model(img, imgsz=dim, verbose=False)
            latencies.append(time.perf_counter() - t0)
        sampler.stop()

        res = sampler.stats
        median_ms   = round(statistics.median(latencies) * 1000, 1)
        mean_ms     = round(statistics.mean(latencies) * 1000, 1)
        min_ms      = round(min(latencies) * 1000, 1)
        max_ms      = round(max(latencies) * 1000, 1)
        fps         = round(1000 / median_ms, 2)

        row = {
            "model":          model_name,
            "image_size":     label,
            "runs":           BENCH_RUNS,
            "latency_median_ms": median_ms,
            "latency_mean_ms":   mean_ms,
            "latency_min_ms":    min_ms,
            "latency_max_ms":    max_ms,
            "fps":            fps,
            **res,
        }
        rows.append(row)

        print(
            f"  {label:>9}  │  {median_ms:>7.1f} ms  │  {fps:>6.2f} FPS  │"
            f"  CPU {res['cpu_peak_pct']:>5.1f}%  │  RAM peak {res['ram_peak_mib']:>7.1f} MiB"
        )

    return rows


def print_summary(all_rows: list[dict], sysinfo: dict):
    print("\n" + "="*80)
    print("BENCHMARK SUMMARY")
    print("="*80)
    print(f"  Host:  {sysinfo['cpu_logical_cores']} logical / {sysinfo['cpu_physical_cores']} physical cores"
          f"  @{sysinfo['cpu_freq_mhz']} MHz")
    print(f"  RAM:   {sysinfo['ram_total_gib']} GiB total, {sysinfo['ram_available_gib']} GiB free at start")
    for g in sysinfo["gpu"]:
        print(f"  GPU:   {g}")
    print()
    print(f"  {'Model':<14} {'Size':>9}  {'Median(ms)':>10}  {'FPS':>7}  "
          f"{'CPU peak%':>10}  {'RAM peak MiB':>13}")
    print("  " + "-"*74)
    for r in all_rows:
        print(
            f"  {r['model']:<14} {r['image_size']:>9}  "
            f"{r['latency_median_ms']:>10.1f}  {r['fps']:>7.2f}  "
            f"{r['cpu_peak_pct']:>10.1f}  {r['ram_peak_mib']:>13.1f}"
        )
    print("="*80)


def main():
    sysinfo = system_snapshot()
    print("\nSystem info:")
    for k, v in sysinfo.items():
        print(f"  {k}: {v}")

    all_rows: list[dict] = []
    failed_models: list[str] = []

    for model_name in MODELS:
        try:
            rows = bench_model(model_name, DIMENSIONS)
            all_rows.extend(rows)
        except Exception as exc:
            print(f"\n[ERROR] {model_name} failed: {exc}")
            failed_models.append(model_name)

    if not all_rows:
        print("\nNo results to report — all models failed.")
        return

    print_summary(all_rows, sysinfo)

    # Save JSON
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = RESULTS_DIR / f"yolo_bench_{ts}.json"
    payload = {"system": sysinfo, "results": all_rows, "failed": failed_models}
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"\nFull results saved to: {out_path}")

    # Save CSV
    csv_path = RESULTS_DIR / f"yolo_bench_{ts}.csv"
    headers = list(all_rows[0].keys())
    lines = [",".join(headers)]
    for r in all_rows:
        lines.append(",".join(str(r[h]) for h in headers))
    csv_path.write_text("\n".join(lines))
    print(f"CSV saved to:          {csv_path}")


if __name__ == "__main__":
    main()
