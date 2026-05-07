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
import logging
import logging.handlers
import subprocess
import sys
import time
from pathlib import Path

import requests
import streamlit as st
from PIL import Image

# ── Logging setup ─────────────────────────────────────────────────────────────
LOG_FILE = Path(__file__).parent / "caption.log"

_handler = logging.handlers.RotatingFileHandler(
    LOG_FILE, maxBytes=10 * 1024 * 1024, backupCount=3, encoding="utf-8"
)
_handler.setFormatter(logging.Formatter(
    "%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
))

log = logging.getLogger("caption_app")
log.setLevel(logging.DEBUG)
if not log.handlers:
    log.addHandler(_handler)

SERVER_URL      = "http://localhost:8080"
SERVE_SCRIPT    = Path(__file__).parent / "serve_llama.py"
MAX_NEW_TOKENS  = 400
IMAGE_MAX_DIM   = 384
CAPTION_PROMPT  = (
    # "Describe this jewellery product in detail. "
    "Include: type of jewellery, metal colour, gemstones or diamonds present, "
    "setting style, design features, and overall aesthetic."
)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Ivana × Devx — Caption",
    page_icon="💎",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Brand CSS ─────────────────────────────────────────────────────────────────
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

/* Navbar */
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

/* Section labels */
.section-label {
    font-family: 'Jost', sans-serif;
    font-size: 0.58rem;
    font-weight: 500;
    letter-spacing: 0.22em;
    color: #C9A96E;
    text-transform: uppercase;
    margin-bottom: 0.5rem;
}

/* Page heading */
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

/* Metric card */
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

/* Pass / fail badge */
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

/* Description box */
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

/* Model status pill */
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

# ── Navbar ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="ivana-nav">
  <span class="nav-brand">Ivana</span>
  <span class="nav-x">×</span>
  <span class="nav-brand">Devx</span>
  <span class="nav-sub">Caption Demo · SmolVLM-500M · llama.cpp · CPU Q8_0</span>
</div>
<div class="gold-rule"></div>
""", unsafe_allow_html=True)

# ── Heading ───────────────────────────────────────────────────────────────────
_, col_h, _ = st.columns([1, 10, 1])
with col_h:
    st.markdown("""
    <div class="page-heading">Image Caption Demo</div>
    <div class="page-sub">Upload a jewellery image — the model generates a detailed description and reports timing for every stage.</div>
    """, unsafe_allow_html=True)


# ── Server helpers ────────────────────────────────────────────────────────────

def _is_server_up() -> bool:
    try:
        r = requests.get(f"{SERVER_URL}/v1/models", timeout=1)
        up = r.ok
        if not up:
            log.warning("Health check failed: HTTP %s", r.status_code)
        return up
    except Exception as exc:
        log.debug("Health check exception: %s", exc)
        return False


@st.cache_resource(show_spinner=False)
def _launch_server() -> subprocess.Popen:
    log.info("Launching llama.cpp server: %s", SERVE_SCRIPT)
    proc = subprocess.Popen(
        [sys.executable, str(SERVE_SCRIPT)],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    log.info("Server process started (pid=%s)", proc.pid)
    return proc


# ── Image helpers ─────────────────────────────────────────────────────────────

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


# ── Inference ─────────────────────────────────────────────────────────────────

def run_caption(pil_img: Image.Image) -> dict:
    orig_w, orig_h = pil_img.size
    img = _resize(pil_img)
    resized_w, resized_h = img.size
    log.info(
        "REQUEST  original=%dx%d  resized=%dx%d  max_tokens=%d",
        orig_w, orig_h, resized_w, resized_h, MAX_NEW_TOKENS,
    )

    t_pre = time.perf_counter()
    data_uri = _to_data_uri(img)
    preprocess_ms = (time.perf_counter() - t_pre) * 1000
    log.debug("Preprocess (resize+encode): %.1f ms", preprocess_ms)

    messages = [{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": data_uri}},
        {"type": "text", "text": CAPTION_PROMPT},
    ]}]

    payload = {
        "model":            "smolvlm-500m",
        "messages":         messages,
        "max_tokens":       MAX_NEW_TOKENS,
        "temperature":      0.3,
        "repeat_penalty":   1.2,
        "stream":           True,
    }

    ttft_ms  = None
    chunks: list[str] = []
    t_gen    = time.perf_counter()

    log.debug("POST %s/v1/chat/completions (stream=True)", SERVER_URL)
    try:
        with requests.post(
            f"{SERVER_URL}/v1/chat/completions",
            json=payload,
            stream=True,
            timeout=180,
        ) as resp:
            log.debug("Response status: %s", resp.status_code)
            resp.raise_for_status()
            for raw in resp.iter_lines():
                if not raw:
                    continue
                line = raw.decode() if isinstance(raw, bytes) else raw
                if not line.startswith("data: "):
                    continue
                data = line[6:]
                if data.strip() == "[DONE]":
                    log.debug("Stream complete ([DONE] received)")
                    break
                try:
                    delta = json.loads(data)["choices"][0]["delta"].get("content", "")
                except (json.JSONDecodeError, KeyError, IndexError):
                    log.warning("Unparseable SSE chunk: %.120s", data)
                    continue
                if delta:
                    if ttft_ms is None:
                        ttft_ms = (time.perf_counter() - t_gen) * 1000
                        log.debug("First token: %.1f ms", ttft_ms)
                    chunks.append(delta)
    except requests.HTTPError as exc:
        log.error("HTTP error from llama.cpp: %s", exc)
        raise
    except requests.RequestException as exc:
        log.error("Request failed: %s", exc)
        raise

    total_s     = time.perf_counter() - t_gen
    description = "".join(chunks)
    n_out       = len(chunks)
    ttft_s      = (ttft_ms or 0.0) / 1000
    decode_s    = total_s - ttft_s
    tok_per_sec = n_out / decode_s if decode_s > 0 else 0.0

    target_met = total_s <= 6.0
    log.info(
        "RESULT   preprocess=%.0fms  ttft=%.0fms  decode=%.2ftok/s  "
        "total=%.2fs  tokens=%d  target=%s",
        preprocess_ms, ttft_ms or 0.0, tok_per_sec,
        total_s, n_out, "PASS" if target_met else "FAIL",
    )
    log.debug("Description: %.200s", description)

    return {
        "description":    description,
        "preprocess_ms":  preprocess_ms,
        "ttft_ms":        ttft_ms or 0.0,
        "decode_tok_s":   tok_per_sec,
        "total_s":        total_s,
        "n_out_tokens":   n_out,
        "target_met":     target_met,
    }


# ── Main UI ───────────────────────────────────────────────────────────────────

_, col_main, _ = st.columns([1, 10, 1])

with col_main:

    status_ph = st.empty()

    # ── Server status ──────────────────────────────────────────────────────────
    # Cache readiness in session_state so that inference reruns (which take
    # longer than the 1s health-check timeout) don't falsely trigger a relaunch.
    if not st.session_state.get("server_ready"):
        if _is_server_up():
            st.session_state["server_ready"] = True
            log.info("Server ready at %s", SERVER_URL)
        else:
            _launch_server()   # no-op after first call (cached)
            log.info("Server not ready — showing loading state, will rerun in 4s")
            status_ph.markdown(
                '<div class="model-loading">⏳ &nbsp;Starting llama.cpp server — '
                'first run downloads ~500 MB to /tmp …</div>',
                unsafe_allow_html=True,
            )
            time.sleep(4)
            st.rerun()

    status_ph.markdown(
        '<div class="model-loaded">✓ &nbsp;llama.cpp server ready · '
        'SmolVLM-500M · Q8_0 · CPU · port 8080</div>',
        unsafe_allow_html=True,
    )

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Upload ─────────────────────────────────────────────────────────────────
    st.markdown('<div class="section-label">Upload Image</div>', unsafe_allow_html=True)
    uploaded = st.file_uploader(
        "Choose a jewellery image",
        type=["jpg", "jpeg", "png", "webp"],
        label_visibility="collapsed",
    )

    if uploaded:
        log.info("Image uploaded: name=%s  size=%d bytes", uploaded.name, uploaded.size)
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
            log.info("Generate button clicked for %s (%dx%d)", uploaded.name, img.size[0], img.size[1])
            st.markdown("---")
            with st.spinner("Running inference via llama.cpp …"):
                try:
                    result = run_caption(img)
                except Exception as exc:
                    log.exception("Inference failed: %s", exc)
                    st.error(f"Inference error: {exc}")
                    st.stop()

            # ── Timing metrics row ─────────────────────────────────────────
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

            # ── Target badge ───────────────────────────────────────────────
            st.markdown("<br>", unsafe_allow_html=True)
            badge = (
                '<span class="badge-pass">✓ Within 6s target</span>'
                if result["target_met"] else
                f'<span class="badge-fail">✗ Over target — {result["total_s"]:.1f}s vs 6s</span>'
            )
            st.markdown(badge, unsafe_allow_html=True)

            # ── Description ────────────────────────────────────────────────
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown('<div class="section-label">Generated Description</div>', unsafe_allow_html=True)
            st.markdown(
                f'<div class="desc-box">{result["description"]}</div>',
                unsafe_allow_html=True,
            )
