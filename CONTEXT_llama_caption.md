# Context: Jewellery Image Captioning via llama.cpp (CPU)

## Task
Run a Streamlit app that accepts a jewellery image upload and generates a detailed text description using `Qwen2-VL-2B-Instruct` (Q8_0) served via a llama.cpp OpenAI-compatible HTTP server. Everything runs CPU-only.

---

## Two files to create

### `serve_llama.py`
Downloads the GGUF model files on first run and starts the llama.cpp server on port 8080.

```python
#!/usr/bin/env python3.12
"""
Start a llama.cpp OpenAI-compatible server for Qwen2-VL-2B-Instruct (Q8_0).
Downloads GGUF files to /tmp on first run (~2.36 GB total).

Usage:
    HF_TOKEN=<token> python3.12 serve_llama.py

Server listens on http://0.0.0.0:8080
OpenAI-compatible endpoint: POST /v1/chat/completions
"""

import subprocess
import sys
from pathlib import Path

from huggingface_hub import hf_hub_download

REPO_ID     = "ggml-org/Qwen2-VL-2B-Instruct-GGUF"
MODEL_FILE  = "Qwen2-VL-2B-Instruct-Q8_0.gguf"
MMPROJ_FILE = "mmproj-Qwen2-VL-2B-Instruct-Q8_0.gguf"
CACHE_DIR   = Path("/tmp/llama_gguf")


def download_if_missing(filename: str) -> Path:
    dest = CACHE_DIR / filename
    if dest.exists():
        print(f"  [cache hit] {dest}")
        return dest
    print(f"  Downloading {filename} …")
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = hf_hub_download(
        repo_id=REPO_ID,
        filename=filename,
        local_dir=str(CACHE_DIR),
    )
    return Path(path)


print("Preparing model files …")
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
    "--n_threads",       "4",       # set to vCPU count of the instance
    "--n_threads_batch", "4",       # set to vCPU count of the instance
    "--n_gpu_layers",    "0",
    "--use_mmap",        "true",
    "--use_mlock",       "false",
    "--chat_format",     "qwen2.5-vl",
]

print("\nStarting llama.cpp server …")
print(" ".join(cmd))
subprocess.run(cmd)
```

---

### `caption_app.py`
Streamlit UI. Auto-starts `serve_llama.py` as a subprocess on first load, polls until ready, then routes image uploads through the llama.cpp server.

```python
"""
Qwen2-VL-2B Image Caption Demo — backed by llama.cpp server.
Upload an image → get a jewellery description + timing breakdown.

Run with:
    python3.12 -m streamlit run caption_app.py

The llama.cpp server is auto-started on first load (downloads ~2.36 GB to /tmp on
first run; subsequent runs use the cache).
"""

from __future__ import annotations

import base64
import io
import json
import subprocess
import sys
import time
from pathlib import Path

import requests
import streamlit as st
from PIL import Image

SERVER_URL      = "http://localhost:8080"
SERVE_SCRIPT    = Path(__file__).parent / "serve_llama.py"
MAX_NEW_TOKENS  = 150
IMAGE_MAX_DIM   = 224
CAPTION_PROMPT  = (
    "Describe this jewellery product in detail. "
    "Include: type of jewellery, metal colour, gemstones or diamonds present, "
    "setting style, design features, and overall aesthetic."
)

st.set_page_config(
    page_title="Ivana × Devx — Caption",
    page_icon="💎",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:ital,wght@0,300;0,400;1,300;1,400&family=Jost:wght@300;400;500&display=swap');

.stApp { background-color: #FAF7F2 !important; }
#MainMenu, footer { visibility: hidden; }
header[data-testid="stHeader"]  { display: none !important; }
[data-testid="stDecoration"]    { display: none !important; }
.block-container {
    padding-top: 0 !important;
    padding-left: 0 !important;
    padding-right: 0 !important;
    max-width: 100% !important;
}
.ivana-nav {
    background-color: #1A1410;
    padding: 1.25rem 3rem;
    display: flex;
    align-items: center;
    width: 100%;
    box-sizing: border-box;
}
.nav-brand {
    font-family: 'Cormorant Garamond', serif;
    font-size: 1.15rem;
    font-weight: 400;
    letter-spacing: 0.38em;
    color: #E8D5B0;
    text-transform: uppercase;
}
.nav-x { color: #C9A96E; font-size: 1.1rem; font-weight: 300;
          margin: 0 0.55rem; font-family: 'Cormorant Garamond', serif; }
.nav-sub {
    font-family: 'Jost', sans-serif;
    font-size: 0.58rem;
    font-weight: 300;
    letter-spacing: 0.28em;
    color: rgba(201,169,110,0.5);
    text-transform: uppercase;
    margin-left: auto;
}
.gold-rule {
    height: 1px;
    background: linear-gradient(90deg, #C9A96E 0%, rgba(201,169,110,0.04) 100%);
    opacity: 0.45;
    margin-bottom: 2.8rem;
}
.section-label {
    font-family: 'Jost', sans-serif;
    font-size: 0.58rem;
    font-weight: 500;
    letter-spacing: 0.22em;
    color: #C9A96E;
    text-transform: uppercase;
    margin-bottom: 0.5rem;
}
.page-heading {
    font-family: 'Cormorant Garamond', serif;
    font-size: 2.5rem;
    font-weight: 300;
    font-style: italic;
    color: #1A1410;
    line-height: 1.1;
    margin-bottom: 0.4rem;
}
.page-sub {
    font-family: 'Jost', sans-serif;
    font-size: 0.72rem;
    font-weight: 300;
    letter-spacing: 0.12em;
    color: #8C7B6B;
    margin-bottom: 2.5rem;
}
.metric-card {
    background: #FFFFFF;
    border: 1px solid #EDE8E0;
    border-radius: 6px;
    padding: 1.1rem 1.4rem;
    text-align: center;
}
.metric-value {
    font-family: 'Cormorant Garamond', serif;
    font-size: 2rem;
    font-weight: 400;
    color: #1A1410;
    line-height: 1;
}
.metric-unit {
    font-family: 'Jost', sans-serif;
    font-size: 0.68rem;
    font-weight: 300;
    color: #C9A96E;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    margin-top: 0.2rem;
}
.metric-label {
    font-family: 'Jost', sans-serif;
    font-size: 0.6rem;
    font-weight: 500;
    letter-spacing: 0.18em;
    color: #8C7B6B;
    text-transform: uppercase;
    margin-top: 0.45rem;
}
.badge-pass {
    display: inline-block;
    background: #E8F5E9;
    color: #2E7D32;
    font-family: 'Jost', sans-serif;
    font-size: 0.65rem;
    font-weight: 500;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    padding: 0.3rem 0.8rem;
    border-radius: 2px;
}
.badge-fail {
    display: inline-block;
    background: #FBE9E7;
    color: #BF360C;
    font-family: 'Jost', sans-serif;
    font-size: 0.65rem;
    font-weight: 500;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    padding: 0.3rem 0.8rem;
    border-radius: 2px;
}
.desc-box {
    background: #FFFFFF;
    border: 1px solid #EDE8E0;
    border-left: 3px solid #C9A96E;
    border-radius: 4px;
    padding: 1.6rem 2rem;
    font-family: 'Jost', sans-serif;
    font-size: 0.85rem;
    font-weight: 300;
    line-height: 1.75;
    color: #2A2018;
    white-space: pre-wrap;
}
.model-loaded {
    display: inline-flex; align-items: center; gap: 0.4rem;
    background: #E8F5E9; color: #2E7D32;
    font-family: 'Jost', sans-serif;
    font-size: 0.62rem; font-weight: 500;
    letter-spacing: 0.14em; text-transform: uppercase;
    padding: 0.25rem 0.7rem; border-radius: 2px;
}
.model-loading {
    display: inline-flex; align-items: center; gap: 0.4rem;
    background: #FFF8E1; color: #F57F17;
    font-family: 'Jost', sans-serif;
    font-size: 0.62rem; font-weight: 500;
    letter-spacing: 0.14em; text-transform: uppercase;
    padding: 0.25rem 0.7rem; border-radius: 2px;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="ivana-nav">
  <span class="nav-brand">Ivana</span>
  <span class="nav-x">×</span>
  <span class="nav-brand">Devx</span>
  <span class="nav-sub">Caption Demo · Qwen2-VL-2B · llama.cpp · CPU Q8_0</span>
</div>
<div class="gold-rule"></div>
""", unsafe_allow_html=True)

_, col_h, _ = st.columns([1, 10, 1])
with col_h:
    st.markdown("""
    <div class="page-heading">Image Caption Demo</div>
    <div class="page-sub">Upload a jewellery image — the model generates a detailed description and reports timing for every stage.</div>
    """, unsafe_allow_html=True)


def _is_server_up() -> bool:
    try:
        r = requests.get(f"{SERVER_URL}/v1/models", timeout=1)
        return r.ok
    except Exception:
        return False


@st.cache_resource(show_spinner=False)
def _launch_server() -> subprocess.Popen:
    return subprocess.Popen(
        [sys.executable, str(SERVE_SCRIPT)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def _resize(img: Image.Image) -> Image.Image:
    w, h = img.size
    if max(w, h) > IMAGE_MAX_DIM:
        scale = IMAGE_MAX_DIM / max(w, h)
        img = img.resize((int(w * scale), int(h * scale)), Image.Resampling.LANCZOS)
    return img


def _to_data_uri(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=95)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def run_caption(pil_img: Image.Image) -> dict:
    img = _resize(pil_img)

    t_pre = time.perf_counter()
    data_uri = _to_data_uri(img)
    preprocess_ms = (time.perf_counter() - t_pre) * 1000

    messages = [{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": data_uri}},
        {"type": "text", "text": CAPTION_PROMPT},
    ]}]

    payload = {
        "model":       "Qwen2-VL-2B-Instruct-Q8_0",
        "messages":    messages,
        "max_tokens":  MAX_NEW_TOKENS,
        "temperature": 0.0,
        "stream":      True,
    }

    ttft_ms  = None
    chunks: list[str] = []
    t_gen    = time.perf_counter()

    with requests.post(
        f"{SERVER_URL}/v1/chat/completions",
        json=payload,
        stream=True,
        timeout=180,
    ) as resp:
        resp.raise_for_status()
        for raw in resp.iter_lines():
            if not raw:
                continue
            line = raw.decode() if isinstance(raw, bytes) else raw
            if not line.startswith("data: "):
                continue
            data = line[6:]
            if data.strip() == "[DONE]":
                break
            try:
                delta = json.loads(data)["choices"][0]["delta"].get("content", "")
            except (json.JSONDecodeError, KeyError, IndexError):
                continue
            if delta:
                if ttft_ms is None:
                    ttft_ms = (time.perf_counter() - t_gen) * 1000
                chunks.append(delta)

    total_s     = time.perf_counter() - t_gen
    description = "".join(chunks)
    n_out       = len(chunks)
    ttft_s      = (ttft_ms or 0.0) / 1000
    decode_s    = total_s - ttft_s
    tok_per_sec = n_out / decode_s if decode_s > 0 else 0.0

    return {
        "description":    description,
        "preprocess_ms":  preprocess_ms,
        "ttft_ms":        ttft_ms or 0.0,
        "decode_tok_s":   tok_per_sec,
        "total_s":        total_s,
        "n_out_tokens":   n_out,
        "target_met":     total_s <= 6.0,
    }


_, col_main, _ = st.columns([1, 10, 1])

with col_main:

    status_ph = st.empty()

    if not _is_server_up():
        _launch_server()
        status_ph.markdown(
            '<div class="model-loading">⏳ &nbsp;Starting llama.cpp server — '
            'first run downloads ~2.4 GB to /tmp …</div>',
            unsafe_allow_html=True,
        )
        time.sleep(4)
        st.rerun()
    else:
        status_ph.markdown(
            '<div class="model-loaded">✓ &nbsp;llama.cpp server ready · '
            'Qwen2-VL-2B · Q8_0 · CPU · port 8080</div>',
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown('<div class="section-label">Upload Image</div>', unsafe_allow_html=True)
    uploaded = st.file_uploader(
        "Choose a jewellery image",
        type=["jpg", "jpeg", "png", "webp"],
        label_visibility="collapsed",
    )

    if uploaded:
        img = Image.open(uploaded).convert("RGB")

        col_img, col_run = st.columns([3, 1])
        with col_img:
            st.markdown('<div class="section-label">Preview</div>', unsafe_allow_html=True)
            st.image(img, width=320)
            st.markdown(
                f'<span style="font-family:Jost,sans-serif;font-size:0.68rem;'
                f'color:#8C7B6B;letter-spacing:0.1em;">'
                f'{img.size[0]} × {img.size[1]} px</span>',
                unsafe_allow_html=True,
            )

        with col_run:
            st.markdown("<br><br><br>", unsafe_allow_html=True)
            run_btn = st.button("Generate Description", use_container_width=True)

        if run_btn:
            st.markdown("---")
            with st.spinner("Running inference via llama.cpp …"):
                result = run_caption(img)

            st.markdown('<div class="section-label">Timing Breakdown</div>', unsafe_allow_html=True)

            c1, c2, c3, c4, c5 = st.columns(5)

            with c1:
                st.markdown(f"""
                <div class="metric-card">
                  <div class="metric-value">{result['preprocess_ms']:.0f}</div>
                  <div class="metric-unit">ms</div>
                  <div class="metric-label">Preprocess</div>
                </div>""", unsafe_allow_html=True)

            with c2:
                st.markdown(f"""
                <div class="metric-card">
                  <div class="metric-value">{result['ttft_ms']:.0f}</div>
                  <div class="metric-unit">ms</div>
                  <div class="metric-label">TTFT</div>
                </div>""", unsafe_allow_html=True)

            with c3:
                st.markdown(f"""
                <div class="metric-card">
                  <div class="metric-value">{result['decode_tok_s']:.1f}</div>
                  <div class="metric-unit">tok / s</div>
                  <div class="metric-label">Decode Speed</div>
                </div>""", unsafe_allow_html=True)

            with c4:
                st.markdown(f"""
                <div class="metric-card">
                  <div class="metric-value">{result['total_s']:.1f}</div>
                  <div class="metric-unit">s</div>
                  <div class="metric-label">Total Time</div>
                </div>""", unsafe_allow_html=True)

            with c5:
                st.markdown(f"""
                <div class="metric-card">
                  <div class="metric-value">{result['n_out_tokens']}</div>
                  <div class="metric-unit">tokens</div>
                  <div class="metric-label">Output</div>
                </div>""", unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            badge = (
                '<span class="badge-pass">✓ Within 6s target</span>'
                if result["target_met"] else
                f'<span class="badge-fail">✗ Over target — {result["total_s"]:.1f}s vs 6s</span>'
            )
            st.markdown(badge, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown('<div class="section-label">Generated Description</div>', unsafe_allow_html=True)
            st.markdown(
                f'<div class="desc-box">{result["description"]}</div>',
                unsafe_allow_html=True,
            )
```

---

## Dependencies

```bash
# Check CPU AVX-512 support first
grep -o 'avx512[a-z_]*' /proc/cpuinfo | sort -u

# If AVX-512 available (e.g. c5/c6i/m5 instances):
TMPDIR=/tmp CMAKE_ARGS="-DGGML_AVX512=ON -DGGML_AVX512_VBMI=ON -DGGML_AVX512_VNNI=ON" \
  pip3.12 install "llama-cpp-python[server]" --no-cache-dir

# If no AVX-512 (drop the CMAKE_ARGS):
TMPDIR=/tmp pip3.12 install "llama-cpp-python[server]" --no-cache-dir

pip3.12 install streamlit pillow requests huggingface_hub
```

## Running

```bash
# Terminal 1 — start server (first run downloads ~2.36 GB to /tmp)
HF_TOKEN=<your_token> python3.12 serve_llama.py

# Terminal 2 — once server is ready (curl http://localhost:8080/v1/models returns 200)
python3.12 -m streamlit run caption_app.py --server.port 8501 --server.headless true
```

## Instance requirements
- `/tmp` with at least **3 GB free** (model: 1.6 GB + mmproj: 677 MB)
- Root disk: at least **1 GB free** for pip install + llama-cpp-python build
- RAM: at least **3 GB free** (mmap keeps model demand-paged, but needs headroom)
- Python 3.12
- `HF_TOKEN` env var — needed only on first run to download GGUFs from HuggingFace

## Key design decisions & known issues
- **`--chat_format qwen2.5-vl`** — this is the correct handler for Qwen2-VL in llama-cpp-python. `qwen2-vl` (with hyphen) is invalid and will throw `LlamaChatCompletionHandlerNotFoundException`.
- **`/v1/models` for health check** — llama-cpp-python server returns 404 on `/health`. Use `GET /v1/models` to check readiness.
- **`IMAGE_MAX_DIM = 224`** — intentionally capped at 224px (not 448px) to reduce image tokens from 256 to 64, cutting vision encoding time from ~20s to ~5s on CPU.
- **`--n_gpu_layers 0`** — CPU-only. If the instance has a GPU and llama-cpp-python is built with CUDA (`CMAKE_ARGS="-DGGML_CUDA=ON ..."`), set this to `99` for full GPU offload.
- **`--n_threads 4`** — set to match vCPU count of the instance (`nproc`).
