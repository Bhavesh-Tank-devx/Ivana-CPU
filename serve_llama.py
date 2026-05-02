#!/usr/bin/env python3.12
"""
Start a llama.cpp OpenAI-compatible server for SmolVLM2-2.2B-Instruct (Q8_0).
Downloads GGUF files to /tmp on first run (~2.5 GB total).
Builds the llama-server binary from source on first run if not cached.

Usage:
    HF_TOKEN=<token> python3.12 serve_llama.py

Server listens on http://0.0.0.0:8080
OpenAI-compatible endpoint: POST /v1/chat/completions
"""

import logging
import logging.handlers
import os
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
REPO_ID     = "ggml-org/SmolVLM2-2.2B-Instruct-GGUF"
MODEL_FILE  = "SmolVLM2-2.2B-Instruct-Q8_0.gguf"
MMPROJ_FILE = "mmproj-SmolVLM2-2.2B-Instruct-Q8_0.gguf"
CACHE_DIR   = Path("/tmp/llama_gguf")

# ── llama-server build config ──────────────────────────────────────────────────
LLAMA_SRC  = Path("/tmp/llama_src")
SERVER_BIN = CACHE_DIR / "llama-server"


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
    if SERVER_BIN.exists():
        log.info("llama-server binary found: %s", SERVER_BIN)
        return SERVER_BIN

    log.info("Building llama-server from source (one-time, ~5 min) …")
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    n_jobs = str(os.cpu_count() or 4)

    if not LLAMA_SRC.exists():
        _run([
            "git", "clone", "--depth=1",
            "https://github.com/ggerganov/llama.cpp",
            str(LLAMA_SRC),
        ])
    else:
        log.info("Source already cloned at %s, skipping clone", LLAMA_SRC)

    build_dir = LLAMA_SRC / "build"
    _run([
        "cmake", "-B", str(build_dir),
        "-DCMAKE_BUILD_TYPE=Release",
        "-DBUILD_SHARED_LIBS=OFF",   # static: no external lib deps at runtime
        "-DGGML_NATIVE=ON",          # compile for this CPU (AVX2/FMA enabled)
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
    log.info("llama-server built and cached at %s", SERVER_BIN)
    return SERVER_BIN


log.info("=== serve_llama starting ===")
model_path  = download_if_missing(MODEL_FILE)
mmproj_path = download_if_missing(MMPROJ_FILE)
server_bin  = ensure_server_binary()

n_threads = str(os.cpu_count() or 4)

cmd = [
    str(server_bin),
    "--model",         str(model_path),
    "--mmproj",        str(mmproj_path),
    "--host",          "0.0.0.0",
    "--port",          "8080",
    # context & batching
    "--ctx-size",      "4096",
    "--batch-size",    "512",
    "--ubatch-size",   "512",
    # CPU threading
    "--threads",       n_threads,
    "--threads-batch", n_threads,
    # no GPU
    "--n-gpu-layers",  "0",
    # server throughput
    "--cont-batching",
    "--flash-attn", "on",
    "--chat-template", "smolvlm",
    "--no-jinja",
    # quantised KV cache — halves memory with minimal quality loss
    "--cache-type-k",  "q8_0",
    "--cache-type-v",  "q8_0",
]

log.info("Starting llama-server: %s", " ".join(cmd))
result = subprocess.run(cmd)
log.warning("llama-server exited with code %s", result.returncode)
