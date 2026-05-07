# IVANA Image Captioning — Implementation Reference

## Project Overview

**IVANA × Devx** is a Streamlit web application that generates detailed text descriptions of jewellery images using a vision-language AI model served locally via llama.cpp. The system is CPU-only, self-contained, and requires no cloud API — all inference runs on the host machine.

**Goal:** Accept a jewellery image upload, return a natural-language product description in under 6 seconds (latency SLA not yet met on CPU; see [Known Issues](#known-issues)).

---

## Repository Layout

```
/home/ec2-user/ivana/
├── caption_app.py           # Streamlit UI (530 lines)
├── serve_llama.py           # GGUF model downloader + llama-server launcher (150 lines)
├── README.md                # Quick-start guide
├── CONTEXT_llama_caption.md # Detailed technical context & requirements
├── caption.log              # Rotating runtime log (gitignored)
└── .gitignore
```

---

## Architecture

```
User (Browser)
      │
      ▼  port 8501
Streamlit UI  ─── caption_app.py
      │
      │  POST /v1/chat/completions (OpenAI-compatible, streaming)
      ▼  port 8080
llama.cpp server  ─── serve_llama.py → llama-server binary
      │
      ▼
GGUF model files  ─── /tmp/llama_gguf/
```

Two processes cooperate:

| Process | File | Port | Role |
|---------|------|------|------|
| Streamlit UI | `caption_app.py` | 8501 | User-facing frontend, spawns server process |
| llama.cpp server | `serve_llama.py` | 8080 | Downloads model, builds binary, serves inference |

`caption_app.py` auto-spawns `serve_llama.py` as a subprocess on first load using `@st.cache_resource`, ensuring only one server instance runs.

---

## Active Model

| Property | Value |
|----------|-------|
| Name | LFM2-VL-450M |
| HuggingFace repo | `runanywhere/LFM2-VL-450M-GGUF` |
| Model file | `LFM2-VL-450M-Q8_0.gguf` (~500 MB) |
| Projection file | `mmproj-LFM2-VL-450M-Q8_0.gguf` (~500 MB) |
| Quantization | Q8_0 (8-bit) |
| Parameters | 450M |
| Context window | 2048 tokens |
| GPU layers | 0 (CPU-only) |
| Cache dir | `/tmp/llama_gguf/` |

### Model History

| Branch | Model | Params | Notes |
|--------|-------|--------|-------|
| `main` | Qwen2-VL-2B-Instruct | 2B | Original version |
| `smolvlm_model` (old) | SmolVLM2-2.2B-Instruct | 2.2B | Intermediate experiment |
| `smolvlm_model` (current, uncommitted) | LFM2-VL-450M | 450M | Lightest model, current experiment |

---

## Configuration Constants

### `caption_app.py`

```python
SERVER_URL     = "http://localhost:8080"
MAX_NEW_TOKENS = 150          # max output tokens
IMAGE_MAX_DIM  = 112          # resize longest side to 112px before encoding
CAPTION_PROMPT = "Describe this jewellery product in detail..."
```

### `serve_llama.py`

```python
REPO_ID      = "runanywhere/LFM2-VL-450M-GGUF"
MODEL_FILE   = "LFM2-VL-450M-Q8_0.gguf"
MMPROJ_FILE  = "mmproj-LFM2-VL-450M-Q8_0.gguf"
CACHE_DIR    = Path("/tmp/llama_gguf")
```

### llama-server launch flags

| Flag | Value | Purpose |
|------|-------|---------|
| `--ctx-size` | 2048 | Context window |
| `--batch-size` | 512 | Prefill batch size |
| `--ubatch-size` | 512 | Unified micro-batch size |
| `--threads` | `$(nproc)` | Inference threads |
| `--threads-batch` | `$(nproc)` | Batch threads |
| `--n-gpu-layers` | 0 | CPU-only (set to 99 for GPU) |
| `--cont-batching` | — | Continuous batching |
| `--flash-attn` | on | Flash attention kernel |
| `--cache-type-k` | q8_0 | Quantized KV cache (K) |
| `--cache-type-v` | q8_0 | Quantized KV cache (V) |
| `--alias` | `lfm2-vl-450m` | Model alias for API requests |

---

## Data Flows

### Startup Flow

```
caption_app.py loaded by Streamlit
  → _is_server_up()  GET /v1/models  (1s timeout)
      ↓ not ready
  → _launch_server()  subprocess.Popen(["python3.12", "serve_llama.py"])
        ↓
      serve_llama.py:
        download_if_missing(MODEL_FILE)    # skipped if /tmp/llama_gguf/LFM2-VL-450M-Q8_0.gguf exists
        download_if_missing(MMPROJ_FILE)   # skipped if cached
        ensure_server_binary()             # builds llama-server from source if not cached (~5 min first run)
        subprocess.run([llama-server, ...]) # blocking — runs forever
  → UI polls /v1/models until 200 OK
```

### Inference Flow

```
User uploads image
  → Image.open() + convert("RGB")
  → click "Generate Description"
      → _resize(img, 112)         # LANCZOS downscale if largest dim > 112px
      → _to_data_uri(img)         # JPEG encode (quality=95) + base64 → data URI
      → POST /v1/chat/completions
          model:      "lfm2-vl-450m"
          messages:   [{role: "user", content: [{type: "image_url", url: <data URI>}, {type: "text", text: CAPTION_PROMPT}]}]
          max_tokens: 150
          temperature: 0.0
          stream:     true
      → parse SSE chunks ("data: {...}")
      → accumulate tokens, record TTFT on first non-empty delta
      → return {description, preprocess_ms, ttft_ms, decode_tok_s, total_s, n_out_tokens, target_met}
  → UI renders 5 metric cards + PASS/FAIL badge + description box
```

---

## API Contract

### Outbound: Chat Completions (to llama-server)

**Request**
```
POST http://localhost:8080/v1/chat/completions
Content-Type: application/json

{
  "model": "lfm2-vl-450m",
  "messages": [
    {
      "role": "user",
      "content": [
        { "type": "image_url", "image_url": { "url": "data:image/jpeg;base64,..." } },
        { "type": "text",      "text": "Describe this jewellery product in detail..." }
      ]
    }
  ],
  "max_tokens": 150,
  "temperature": 0.0,
  "stream": true
}
```

**Response** (Server-Sent Events)
```
data: {"choices":[{"delta":{"content":"The"}}]}
data: {"choices":[{"delta":{"content":" ring"}}]}
...
data: [DONE]
```

### Health Check

```
GET http://localhost:8080/v1/models
→ 200 OK  (server ready)
→ connection error / timeout  (server not ready)
```

---

## UI Design

### Styling

- **Fonts:** Cormorant Garamond (serif, headings) + Jost (sans-serif, body) via Google Fonts
- **Color palette:**
  - Background: `#FAF7F2` (cream)
  - Primary dark: `#1A1410` (dark brown)
  - Accent: `#C9A96E` (gold)
- **Brand:** "Ivana × Devx" in navbar
- **Subtitle:** "Caption Demo · LFM2-VL-450M · llama.cpp · CPU Q8_0"

### Metric Cards (displayed after inference)

| Card | Metric | Source |
|------|--------|--------|
| Preprocess | resize + base64 encode time (ms) | `time.perf_counter` around `_resize` + `_to_data_uri` |
| TTFT | Time to first token (ms) | time of first non-empty SSE delta |
| Decode speed | tokens / second | `(n_tokens - 1) / decode_elapsed` |
| Total time | end-to-end wall time (s) | from start of `run_caption()` to stream end |
| Output tokens | count of generated tokens | SSE chunk count |

**Target:** Total time ≤ 6.0 s → PASS badge; otherwise FAIL.

---

## Build & Runtime Dependencies

### Python Packages

```
streamlit
Pillow
requests
huggingface_hub
```

The llama.cpp server binary is built from source (not pip-installed).

### Build Tools (needed once on first run)

```
git
cmake
g++ / clang
make or ninja
```

### Runtime

- Python 3.12 (explicitly required)
- `HF_TOKEN` environment variable (HuggingFace access token for model download)

---

## First-Run Sequence (cold start)

1. `python3.12 caption_app.py` (or `streamlit run caption_app.py`)
2. App auto-spawns `serve_llama.py`
3. `serve_llama.py` downloads two GGUF files (~1 GB total) to `/tmp/llama_gguf/`
4. If `/tmp/llama_gguf/llama-server` does not exist:
   - Clones `ggerganov/llama.cpp` (shallow, depth=1)
   - Builds with CMake (`Release`, `DGGML_NATIVE=ON`, `DBUILD_SHARED_LIBS=OFF`)
   - Takes ~5 minutes; binary placed at `/tmp/llama_gguf/llama-server`
5. `llama-server` starts on port 8080
6. UI polls `/v1/models` until ready
7. Subsequent runs skip steps 3–4 (cache hit)

---

## Logging

Both processes write to the same rotating log file.

| Property | Value |
|----------|-------|
| File | `caption.log` (gitignored) |
| Max size | 10 MB |
| Backups | 3 |
| Format | `%(asctime)s  %(levelname)-8s  %(message)s` |
| Date format | `%Y-%m-%d %H:%M:%S` |
| Level | DEBUG |

**Key log events:**

| Event | Level | Message pattern |
|-------|-------|-----------------|
| Server start | INFO | `=== serve_llama starting ===` |
| Model cache hit | INFO | `Cache hit: /tmp/llama_gguf/<file>` |
| Image upload | INFO | `Image uploaded: name=... size=... bytes` |
| Inference request | INFO | `REQUEST  original=WxH  resized=WxH  max_tokens=150` |
| Preprocess timing | DEBUG | `Preprocess (resize+encode): X ms` |
| Result summary | INFO | `RESULT   preprocess=Xms  ttft=Xms  decode=X tok/s  total=Xs  tokens=N  target=PASS/FAIL` |

---

## Git Branches

| Branch | State | Description |
|--------|-------|-------------|
| `main` | stable | Original Qwen2-VL-2B implementation |
| `smolvlm_model` | active, uncommitted changes | Current LFM2-VL-450M experiment |

**Uncommitted changes on `smolvlm_model`:**
- `caption_app.py`: model alias updated to `lfm2-vl-450m`
- `serve_llama.py`: repo changed to `runanywhere/LFM2-VL-450M-GGUF`

---

## Performance Observations

From `caption.log` (actual runs):

| Metric | Observed | Target |
|--------|----------|--------|
| Preprocess (resize + encode) | ~0.6–1 ms | — |
| TTFT (time to first token) | 7,000–30,000 ms | — |
| Decode speed | ~9–10 tok/s | — |
| Total inference time | 15–46 seconds | ≤ 6 s |

**TTFT is the dominant bottleneck.** Vision encoding (processing the image into tokens) is CPU-bound and takes 7–30 seconds. Token generation is fast once started (~9–10 tok/s).

### Image Resizing Strategy

Reducing the image before encoding reduces the number of vision tokens fed to the model, which directly cuts TTFT:

| Max dim | Vision tokens (approx) | TTFT impact |
|---------|------------------------|-------------|
| 224 px | ~256 | Slow (original) |
| 112 px | ~64 | 4× fewer tokens |

Current setting: `IMAGE_MAX_DIM = 112`.

---

## Known Issues

1. **Latency SLA not met.** All tested models (Qwen2-VL-2B, SmolVLM2-2.2B, LFM2-VL-450M) exceed the 6-second target on CPU. TTFT alone is 7–30 s. GPU acceleration (`--n-gpu-layers 99`) would likely solve this.

2. **First-run build time.** llama.cpp must be compiled from source if not cached, adding ~5 minutes to first startup.

3. **Model in `/tmp/`.** `/tmp/llama_gguf/` is not persistent across reboots on most Linux systems. Re-download (~1 GB) and potentially re-build will occur after a reboot.

4. **Single-user design.** `@st.cache_resource` ensures one server subprocess, but concurrent multi-user usage is not handled.

5. **Uncommitted working changes.** The `smolvlm_model` branch has model changes (`LFM2-VL-450M`) that have not been committed to git.

---

## Extending the Project

### Switch to GPU inference
In `serve_llama.py`, change:
```python
"--n-gpu-layers", "0"
# → 
"--n-gpu-layers", "99"
```
Also rebuild llama.cpp with CUDA support (`-DGGML_CUDA=ON`).

### Change model
Update `REPO_ID`, `MODEL_FILE`, `MMPROJ_FILE` in `serve_llama.py` and the `--alias` flag to match the new model name used in `caption_app.py`.

### Adjust output length
Change `MAX_NEW_TOKENS` in `caption_app.py`.

### Change the caption prompt
Edit `CAPTION_PROMPT` in `caption_app.py`.

### Increase image resolution
Increase `IMAGE_MAX_DIM` in `caption_app.py` (at the cost of higher TTFT).
