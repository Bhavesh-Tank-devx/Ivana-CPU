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
import uuid
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

SERVER_URL     = "http://localhost:8080"
SERVE_SCRIPT   = Path(__file__).parent / "serve_llama.py"
MAX_NEW_TOKENS = 280
IMAGE_MAX_DIM  = 384

# ── Request archive ───────────────────────────────────────────────────────────
REQUEST_LOG_DIR  = Path(__file__).parent / "requests_log"
REQUEST_IMAGES   = REQUEST_LOG_DIR / "images"
REQUEST_JSONL    = REQUEST_LOG_DIR / "requests.jsonl"

REQUEST_LOG_DIR.mkdir(exist_ok=True)
REQUEST_IMAGES.mkdir(exist_ok=True)

SYSTEM_PROMPT = """\
You are an expert ring attribute extractor with deep knowledge of gemology and fine jewellery design. \
Every image you receive is a close-up photograph of a RING. Your task is to examine it precisely and \
output its visual attributes as a JSON object.

CRITICAL VISUAL CHECKS — read these before labeling every field:

stone_shape — measure the outline of the center stone carefully which is most prominant in the ring:
  • round    — perfectly circular; width and height are equal. If the stone is elongated AT ALL, it is not round.
  • oval     — ellipse longer than wide; length:width ratio roughly 1.3–1.5
  • princess — square outline with sharp 90-degree corners; no rounding
  • cushion  — square or rectangular with noticeably rounded corners; pillow-like silhouette
  • pear     — teardrop; one pointed tip and one fully rounded end
  • marquise — eye or football shape; two pointed ends, widest in the middle
  • emerald  — rectangle with clipped corners; step-cut facets visible as concentric tiers/steps
  • radiant  — rectangle or square with clipped corners; brilliant-cut (sparkly) not step-cut
  • heart    — clear heart silhouette with a cleft at the top centre
  • asscher  — square step-cut with heavily clipped corners; nearly octagonal outline
  Rule: if the stone is longer in any direction than it is wide, rule out "round" immediately.

stone_arrangement — how stones are laid out on the ring:
  • solitaire   — one center stone, plain or minimal band, no surrounding stones
  • halo        — center stone encircled by a ring of smaller accent stones
  • pavé        — band surface densely covered with small stones flush to the metal
  • cluster     — group of stones close together without a single dominant center
  • three-stone — exactly three prominent stones in a row (past-present-future style)
  • eternity    — stones run continuously all the way around the band
  • other       — any arrangement that doesn't fit the above categories, e.g. scattered random stones, asymmetrical designs, etc.
  • none        — no stones at all

band_metal — judge strictly by visible color:
  • yellow gold  — warm yellow/amber tone
  • white gold   — cool silver-white tone with slightly warm undertone
  • rose gold    — distinct pink or copper-rose tone
  • silver       — cool grey-white, less brilliant than white gold
  • platinum     — cool grey-white, slightly darker/heavier-looking than white gold
  • mixed        — two or more metal colors clearly visible on the same ring
  • unknown      — metal color is ambiguous or not visible

setting_style — how the center stone is secured:
  • prong     — 4 or 6 small metal claws grip the stone; stone is raised; visible from the side
  • bezel     — a continuous metal rim wraps entirely around the stone's edge
  • channel   — stones sit between two parallel metal rails; no prongs visible between stones
  • tension   — stone appears to float; held purely by spring pressure from two band ends
  • pavé      — tiny stones set very close with micro-prongs; surface looks paved/embedded
  • flush     — stone sits level with the metal surface, fully sunk in
  • invisible — stones fit edge-to-edge with no metal visible between them from above

Output ONLY a JSON object — no prose, no markdown fences, no explanation.

Allowed values per field (pick exactly one):
  band_metal:        yellow gold | white gold | rose gold | silver | platinum | mixed | unknown
  band_style:        plain | twisted | split-shank | bypass | tapered | braided | unknown
  band_width:        thin | medium | wide | statement
  band_texture:      smooth | hammered | engraved | milgrain | braided | none
  band_finish:       polished | matte | brushed | satin | unknown
  stone_arrangement: solitaire | halo | pavé | cluster | three-stone | eternity | none
  stone_count:       single | two-stone | three-stone | multi | eternity | none
  center_stone:      diamond | ruby | emerald | sapphire | pearl | opal | amethyst | garnet | topaz | none
  stone_shape:       round | oval | princess | cushion | pear | marquise | emerald | radiant | heart | asscher | none
  stone_color:       clear | red | blue | green | pink | purple | yellow | orange | multicolor | none
  setting_style:     prong | bezel | channel | tension | pavé | flush | invisible | none
  accent_stones:     diamond | sapphire | ruby | emerald | mixed | none
  filigree:          yes | no
  milgrain:          yes | no
  occasion:          engagement | wedding | everyday | cocktail | fashion
  profile:           flat | comfort-fit | domed | knife-edge | unknown

Now examine the ring image carefully and output only the JSON object.\
"""

RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "ring_attributes",
        "strict": True,
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "required": [
                "band_metal","band_style","band_width","band_texture",
                "band_finish","stone_arrangement","stone_count","center_stone",
                "stone_shape","stone_color","setting_style","accent_stones",
                "filigree","milgrain","occasion","profile",
            ],
            "properties": {
                "band_metal":        {"type":"string","enum":["yellow gold","white gold","rose gold","silver","platinum","mixed","unknown"]},
                "band_style":        {"type":"string","enum":["plain","twisted","split-shank","bypass","tapered","braided","unknown"]},
                "band_width":        {"type":"string","enum":["thin","medium","wide","statement"]},
                "band_texture":      {"type":"string","enum":["smooth","hammered","engraved","milgrain","braided","none"]},
                "band_finish":       {"type":"string","enum":["polished","matte","brushed","satin","unknown"]},
                "stone_arrangement": {"type":"string","enum":["solitaire","halo","pavé","cluster","three-stone","eternity","none"]},
                "stone_count":       {"type":"string","enum":["single","two-stone","three-stone","multi","eternity","none"]},
                "center_stone":      {"type":"string","enum":["diamond","ruby","emerald","sapphire","pearl","opal","amethyst","garnet","topaz","none"]},
                "stone_shape":       {"type":"string","enum":["round","oval","princess","cushion","pear","marquise","emerald","radiant","heart","asscher","none"]},
                "stone_color":       {"type":"string","enum":["clear","red","blue","green","pink","purple","yellow","orange","multicolor","none"]},
                "setting_style":     {"type":"string","enum":["prong","bezel","channel","tension","pavé","flush","invisible","none"]},
                "accent_stones":     {"type":"string","enum":["diamond","sapphire","ruby","emerald","mixed","none"]},
                "filigree":          {"type":"string","enum":["yes","no"]},
                "milgrain":          {"type":"string","enum":["yes","no"]},
                "occasion":          {"type":"string","enum":["engagement","wedding","everyday","cocktail","fashion"]},
                "profile":           {"type":"string","enum":["flat","comfort-fit","domed","knife-edge","unknown"]},
            },
        },
    },
}

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
def _launch_server() -> subprocess.Popen | None:
    check = subprocess.run(["pgrep", "-f", "serve_llama.py"], capture_output=True)
    if check.returncode == 0:
        pids = check.stdout.decode().strip().split()
        log.info("serve_llama.py already running (pids=%s), skipping launch", pids)
        return None
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


# ── Request archive ───────────────────────────────────────────────────────────

def _save_request_log(
    pil_img: Image.Image,
    original_filename: str,
    result: dict,
) -> str:
    """Save image + result to requests_log/; append one line to requests.jsonl.

    Returns the request_id so callers can reference it in log messages.
    """
    ts        = time.strftime("%Y%m%d_%H%M%S")
    req_id    = f"{ts}_{uuid.uuid4().hex[:6]}"

    img_path  = REQUEST_IMAGES / f"{req_id}.jpg"
    pil_img.save(img_path, format="JPEG", quality=95)

    record = {
        "request_id":      req_id,
        "timestamp":       time.strftime("%Y-%m-%dT%H:%M:%S"),
        "original_filename": original_filename,
        "image_path":      str(img_path.relative_to(REQUEST_LOG_DIR)),
        "image_size":      {"width": pil_img.width, "height": pil_img.height},
        "timing": {
            "preprocess_ms": round(result["preprocess_ms"], 1),
            "ttft_ms":       round(result["ttft_ms"], 1),
            "decode_tok_s":  round(result["decode_tok_s"], 2),
            "total_s":       round(result["total_s"], 3),
            "n_out_tokens":  result["n_out_tokens"],
        },
        "raw_response":  result["raw_text"],
        "attributes":    result["attributes"],
        "parse_ok":      result["attributes"] is not None,
    }

    with REQUEST_JSONL.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")

    log.info(
        "ARCHIVED  req_id=%s  image=%s  parse_ok=%s  total=%.2fs",
        req_id, img_path.name, record["parse_ok"], result["total_s"],
    )
    return req_id


# ── Inference ─────────────────────────────────────────────────────────────────

def run_caption(pil_img: Image.Image) -> dict:
    orig_w, orig_h = pil_img.size
    img = _resize(pil_img)
    log.info("REQUEST  original=%dx%d  resized=%dx%d", orig_w, orig_h, *img.size)

    t_pre = time.perf_counter()
    data_uri = _to_data_uri(img)
    preprocess_ms = (time.perf_counter() - t_pre) * 1000
    log.debug("Preprocess (resize+encode): %.1f ms", preprocess_ms)

    payload = {
        "model": "smolvlm-500m",
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": [
                {"type": "image_url", "image_url": {"url": data_uri}},
            ]},
        ],
        "max_tokens":      MAX_NEW_TOKENS,
        "temperature":     0.1,
        "repeat_penalty":  1.0,
        "response_format": RESPONSE_FORMAT,
        "stream":          True,
    }

    ttft_ms: float | None = None
    chunks: list[str] = []
    t_gen = time.perf_counter()

    try:
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

    total_s   = time.perf_counter() - t_gen
    raw_text  = "".join(chunks).strip()
    n_out     = len(chunks)
    ttft_s    = (ttft_ms or 0.0) / 1000
    decode_s  = total_s - ttft_s
    tok_per_s = n_out / decode_s if decode_s > 0 else 0.0

    log.info(
        "RESULT   preprocess=%.0fms  ttft=%.0fms  decode=%.2ftok/s  total=%.2fs  tokens=%d",
        preprocess_ms, ttft_ms or 0.0, tok_per_s, total_s, n_out,
    )
    log.debug("Raw model output: %.300s", raw_text)

    clean = raw_text
    if clean.startswith("```"):
        clean = clean.split("```")[1]
        if clean.startswith("json"):
            clean = clean[4:]
        clean = clean.strip()

    try:
        attributes = json.loads(clean)
    except json.JSONDecodeError as exc:
        log.error("Model returned invalid JSON: %s | raw=%.300s", exc, raw_text)
        attributes = None

    return {
        "attributes":   attributes,
        "raw_text":     raw_text,
        "preprocess_ms": preprocess_ms,
        "ttft_ms":       ttft_ms or 0.0,
        "decode_tok_s":  tok_per_s,
        "total_s":       total_s,
        "n_out_tokens":  n_out,
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

            req_id = _save_request_log(img, uploaded.name, result)
            log.info("Request complete  req_id=%s  file=%s", req_id, uploaded.name)

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

            # ── Attributes ────────────────────────────────────────────────
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown('<div class="section-label">Extracted Attributes</div>', unsafe_allow_html=True)

            attrs = result["attributes"]
            if attrs is None:
                st.error("Model did not return valid JSON.")
                st.code(result["raw_text"], language="text")
            else:
                FIELD_LABELS = {
                    "item_type":         "Item Type",
                    "band_metal":        "Band Metal",
                    "band_style":        "Band Style",
                    "band_width":        "Band Width",
                    "band_texture":      "Band Texture",
                    "band_finish":       "Band Finish",
                    "stone_arrangement": "Stone Arrangement",
                    "stone_count":       "Stone Count",
                    "center_stone":      "Center Stone",
                    "stone_shape":       "Stone Shape",
                    "stone_color":       "Stone Color",
                    "setting_style":     "Setting Style",
                    "accent_stones":     "Accent Stones",
                    "filigree":          "Filigree",
                    "milgrain":          "Milgrain",
                    "occasion":          "Occasion",
                    "profile":           "Profile",
                }
                rows = list(FIELD_LABELS.items())
                for i in range(0, len(rows), 3):
                    cols = st.columns(3)
                    for col, (key, label) in zip(cols, rows[i:i+3]):
                        value = attrs.get(key, "—")
                        with col:
                            st.markdown(f"""
                            <div class="metric-card" style="text-align:left;padding:0.9rem 1.2rem;">
                              <div class="metric-label">{label}</div>
                              <div style="font-family:'Jost',sans-serif;font-size:0.88rem;
                                          font-weight:400;color:#1A1410;margin-top:0.3rem;">
                                {value}
                              </div>
                            </div>""", unsafe_allow_html=True)
                    st.markdown("<div style='margin-bottom:0.6rem'></div>", unsafe_allow_html=True)
