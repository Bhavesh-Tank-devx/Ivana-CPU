# Jewellery Image Captioning — llama.cpp + Qwen2-VL

A Streamlit app that accepts a jewellery image upload and generates a detailed text description using **Qwen2-VL-2B-Instruct** (Q8_0 GGUF) served via a llama.cpp OpenAI-compatible HTTP server. Runs entirely CPU-only.

## Architecture

```
User uploads image
       │
       ▼
caption_app.py  (Streamlit UI)
       │  POST /v1/chat/completions  (base64-encoded image)
       ▼
serve_llama.py  (llama.cpp server on :8080)
       │
       ▼
Qwen2-VL-2B-Instruct-Q8_0.gguf  (downloaded to /tmp on first run)
```

## Files

| File | Purpose |
|------|---------|
| `serve_llama.py` | Downloads GGUF model files on first run, then starts the llama.cpp OpenAI-compatible server on port 8080 |
| `caption_app.py` | Streamlit UI — auto-starts the server on load, accepts image uploads, streams the caption response |
| `caption.log` | Rotating log file written by both scripts (max 10 MB, 3 backups) |

## Prerequisites

- Python 3.12
- `llama-cpp-python[server]` with CPU build
- `huggingface_hub`, `streamlit`, `Pillow`, `requests`
- ~2.4 GB free disk space in `/tmp` for model files
- A Hugging Face token with access to `ggml-org/Qwen2-VL-2B-Instruct-GGUF`

## Setup

```bash
pip install "llama-cpp-python[server]" huggingface_hub streamlit Pillow requests
```

## Running

**Terminal 1 — start the model server:**

```bash
HF_TOKEN=<your_hf_token> python3.12 serve_llama.py
```

On first run this downloads ~2.36 GB to `/tmp/llama_gguf`. Subsequent runs use the cache.

**Terminal 2 — start the Streamlit app:**

```bash
python3.12 -m streamlit run caption_app.py
```

Open the URL printed by Streamlit (default: `http://localhost:8501`).

> `caption_app.py` will also auto-start `serve_llama.py` on first load if the server is not already running.

## Configuration

Both scripts default to **4 CPU threads** (`--n_threads 4`). Adjust to match your instance's vCPU count:

```python
# serve_llama.py
"--n_threads",       "4",
"--n_threads_batch", "4",
```

## Model

| Setting | Value |
|---------|-------|
| Model | Qwen2-VL-2B-Instruct |
| Quantization | Q8_0 |
| Source | `ggml-org/Qwen2-VL-2B-Instruct-GGUF` (Hugging Face) |
| Context length | 512 tokens |
| GPU layers | 0 (CPU-only) |
