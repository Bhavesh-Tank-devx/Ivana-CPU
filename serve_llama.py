#!/usr/bin/env python3.12
"""
Start a llama.cpp OpenAI-compatible server for MiniCPM-V 4.6 (Q4_K_M).
Downloads GGUF files to /tmp on first run (~1.3 GB total).
Builds the llama-server binary from source on first run if not cached.
Requires llama.cpp >= b9049 for MiniCPM-V 4.6 support (auto-rebuilds if stale).

Usage:
    HF_TOKEN=<token> python3.12 serve_llama.py

Server listens on http://0.0.0.0:8080
OpenAI-compatible endpoint: POST /v1/chat/completions
"""

import logging
import logging.handlers
import os
import shutil
import subprocess
from pathlib import Path

from huggingface_hub import hf_hub_download

# ── Logging setup ─────────────────────────────────────────────────────────────
LOG_FILE = Path(__file__).parent / "caption.log"

_handler = logging.handlers.RotatingFileHandler(
    LOG_FILE, maxBytes=10 * 1024 * 1024, backupCount=3, encoding="utf-8"
)
_handler.setFormatter(logging.Formatter(
    "%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
))

log = logging.getLogger("serve_llama")
log.setLevel(logging.DEBUG)
if not log.handlers:
    log.addHandler(_handler)

# ── Model config ───────────────────────────────────────────────────────────────
REPO_ID     = "ggml-org/MiniCPM-V-4.6-GGUF"
MODEL_FILE  = "MiniCPM-V-4.6-Q4_K_M.gguf"
MMPROJ_FILE = "mmproj-MiniCPM-V-4.6-Q8_0.gguf"
MODEL_ALIAS = "minicpmv-4.6"
CACHE_DIR   = Path("/tmp/llama_gguf")

# ── llama-server build config ──────────────────────────────────────────────────
# Bump BUILD_TAG to force a fresh clone + rebuild (e.g. when min version changes).
BUILD_TAG      = "minicpmv46"   # requires llama.cpp >= b9049
BUILD_TAG_FILE = CACHE_DIR / ".build_tag"
LLAMA_SRC      = Path("/tmp/llama_src")
SERVER_BIN     = CACHE_DIR / "llama-server"


def download_if_missing(filename: str) -> Path:
    dest = CACHE_DIR / filename
    if dest.exists():
        log.info("Cache hit: %s", dest)
        return dest
    log.info("Downloading %s from %s …", filename, REPO_ID)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = hf_hub_download(
        repo_id=REPO_ID,
        filename=filename,
        local_dir=str(CACHE_DIR),
    )
    log.info("Downloaded to %s", path)
    return Path(path)


def _run(cmd: list[str], cwd: Path | None = None) -> None:
    log.info("$ %s", " ".join(cmd))
    result = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    if result.stdout:
        log.debug("stdout: %s", result.stdout[-2000:])
    if result.returncode != 0:
        log.error("stderr: %s", result.stderr[-2000:])
        raise RuntimeError(f"Command failed (exit {result.returncode}): {' '.join(cmd)}")


def ensure_server_binary() -> Path:
    # Invalidate cached binary if it predates MiniCPM-V 4.6 support
    if SERVER_BIN.exists():
        tag_ok = BUILD_TAG_FILE.exists() and BUILD_TAG_FILE.read_text().strip() == BUILD_TAG
        if tag_ok:
            log.info("llama-server binary found (%s): %s", BUILD_TAG, SERVER_BIN)
            return SERVER_BIN
        log.info("Stale binary (tag mismatch) — rebuilding for MiniCPM-V 4.6 support (b9049+)")
        SERVER_BIN.unlink()

    log.info("Building llama-server from source (one-time, ~5 min) …")
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    n_jobs = str(os.cpu_count() or 4)

    # Fresh clone to guarantee we have >= b9049
    if LLAMA_SRC.exists():
        log.info("Removing stale source tree for fresh clone")
        shutil.rmtree(LLAMA_SRC)
    _run([
        "git", "clone", "--depth=1",
        "https://github.com/ggml-org/llama.cpp",
        str(LLAMA_SRC),
    ])

    build_dir = LLAMA_SRC / "build"
    _run([
        "cmake", "-B", str(build_dir),
        "-DCMAKE_BUILD_TYPE=Release",
        "-DBUILD_SHARED_LIBS=OFF",
        "-DGGML_NATIVE=ON",          # AVX2/FMA for AMD EPYC 7R13
        "-DLLAMA_BUILD_TESTS=OFF",
        "-DLLAMA_BUILD_EXAMPLES=OFF",
    ], cwd=LLAMA_SRC)

    _run([
        "cmake", "--build", str(build_dir),
        "--config", "Release",
        "--target", "llama-server",
        "-j", n_jobs,
    ], cwd=LLAMA_SRC)

    built = build_dir / "bin" / "llama-server"
    built.rename(SERVER_BIN)
    BUILD_TAG_FILE.write_text(BUILD_TAG)
    log.info("llama-server built and cached at %s", SERVER_BIN)
    return SERVER_BIN


log.info("=== serve_llama starting (MiniCPM-V 4.6 Q4_K_M) ===")
model_path  = download_if_missing(MODEL_FILE)
mmproj_path = download_if_missing(MMPROJ_FILE)
server_bin  = ensure_server_binary()

n_threads = str(os.cpu_count() or 4)

cmd = [
    str(server_bin),
    "--model",          str(model_path),
    "--mmproj",         str(mmproj_path),
    "--alias",          MODEL_ALIAS,
    "--jinja",                           # use the model's embedded chat template
    "--host",           "0.0.0.0",
    "--port",           "8080",
    # context: SigLIP2 at 384px → ~64 vision tokens + prompt + 200 output
    "--ctx-size",       "2048",
    "--batch-size",     "1024",
    "--ubatch-size",    "1024",
    # CPU threading — use all 4 vCPUs
    "--threads",        n_threads,
    "--threads-batch",  n_threads,
    # memory
    "--mmap",
    "--mlock",
    # no GPU
    "--n-gpu-layers",   "0",
    "--cont-batching",
    "--flash-attn", "off",   # CPU-only; explicit value required in newer llama.cpp
    "--reasoning", "off",    # disable thinking mode — instruct checkpoint, not thinking variant
]

log.info("Starting llama-server: %s", " ".join(cmd))
result = subprocess.run(cmd)
log.warning("llama-server exited with code %s", result.returncode)
