#!/usr/bin/env python3.12
"""
Start a llama.cpp OpenAI-compatible server for Qwen2-VL-2B-Instruct (Q8_0).
Downloads GGUF files to /tmp on first run (~2.36 GB total).

Usage:
    HF_TOKEN=<token> python3.12 serve_llama.py

Server listens on http://0.0.0.0:8080
OpenAI-compatible endpoint: POST /v1/chat/completions
"""

import logging
import logging.handlers
import subprocess
import sys
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

REPO_ID     = "ggml-org/Qwen2-VL-2B-Instruct-GGUF"
MODEL_FILE  = "Qwen2-VL-2B-Instruct-Q8_0.gguf"
MMPROJ_FILE = "mmproj-Qwen2-VL-2B-Instruct-Q8_0.gguf"
CACHE_DIR   = Path("/tmp/llama_gguf")


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


log.info("=== serve_llama starting ===")
model_path  = download_if_missing(MODEL_FILE)
mmproj_path = download_if_missing(MMPROJ_FILE)

cmd = [
    sys.executable, "-m", "llama_cpp.server",
    "--model",           str(model_path),
    "--clip_model_path", str(mmproj_path),
    "--host",            "0.0.0.0",
    "--port",            "8080",
    "--n_ctx",           "512",
    "--n_batch",         "512",
    "--n_threads",       "4",
    "--n_threads_batch", "4",
    "--n_gpu_layers",    "0",
    "--use_mmap",        "true",
    "--use_mlock",       "true",
    "--chat_format",     "qwen2.5-vl",
]

log.info("Starting llama.cpp server: %s", " ".join(cmd))
result = subprocess.run(cmd)
log.warning("llama.cpp server exited with code %s", result.returncode)
