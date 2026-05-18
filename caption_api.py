#!/usr/bin/env python3.12
"""
FastAPI caption API — wraps the llama.cpp SmolVLM-500M server.

POST /caption   multipart: file=<image>
GET  /health    { "llama_server": "up" | "starting" }

Run with:
    python3.12 -m uvicorn caption_api:app --host 0.0.0.0 --port 8000

On first run, serve_llama.py is auto-started in the background.
Downloads ~500 MB to /tmp and builds llama-server from source (~5 min, one-time).
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
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image

# ── Logging ───────────────────────────────────────────────────────────────────
LOG_FILE = Path(__file__).parent / "caption.log"
_handler = logging.handlers.RotatingFileHandler(
    LOG_FILE, maxBytes=10 * 1024 * 1024, backupCount=3, encoding="utf-8"
)
_handler.setFormatter(logging.Formatter(
    "%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
))
log = logging.getLogger("caption_api")
log.setLevel(logging.DEBUG)
if not log.handlers:
    log.addHandler(_handler)

# ── Config ────────────────────────────────────────────────────────────────────
SERVER_URL     = "http://localhost:8080"
SERVE_SCRIPT   = Path(__file__).parent / "serve_llama.py"
MAX_NEW_TOKENS = 280
IMAGE_MAX_DIM  = 384

SYSTEM_PROMPT = """\
You are a jewellery attribute extractor. Look at the image and output ONLY a JSON object — no prose, no markdown fences.

Allowed values per field (pick exactly one):
  item_type:         ring | earring | necklace | bracelet | pendant | bangle | brooch
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

Examples:

Input: yellow gold ring, plain thin band, single round diamond in four-prong solitaire setting, no accent stones, no filigree
Output: {"item_type":"ring","band_metal":"yellow gold","band_style":"plain","band_width":"thin","band_texture":"smooth","band_finish":"polished","stone_arrangement":"solitaire","stone_count":"single","center_stone":"diamond","stone_shape":"round","stone_color":"clear","setting_style":"prong","accent_stones":"none","filigree":"no","milgrain":"no","occasion":"engagement","profile":"comfort-fit"}

Input: white gold eternity band, medium width, channel-set round diamonds all around, no center stone, milgrain edges
Output: {"item_type":"ring","band_metal":"white gold","band_style":"plain","band_width":"medium","band_texture":"milgrain","band_finish":"polished","stone_arrangement":"eternity","stone_count":"eternity","center_stone":"none","stone_shape":"round","stone_color":"clear","setting_style":"channel","accent_stones":"diamond","filigree":"no","milgrain":"yes","occasion":"wedding","profile":"flat"}

Input: rose gold ring, split-shank band with pavé diamonds, large oval blue sapphire halo center, prong set
Output: {"item_type":"ring","band_metal":"rose gold","band_style":"split-shank","band_width":"medium","band_texture":"smooth","band_finish":"polished","stone_arrangement":"halo","stone_count":"single","center_stone":"sapphire","stone_shape":"oval","stone_color":"blue","setting_style":"prong","accent_stones":"diamond","filigree":"no","milgrain":"no","occasion":"engagement","profile":"comfort-fit"}

Now analyze the provided image and output only the JSON object.\
"""

# JSON schema used by llama.cpp response_format to grammar-enforce the output
RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "jewellery_attributes",
        "strict": True,
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "required": [
                "item_type","band_metal","band_style","band_width","band_texture",
                "band_finish","stone_arrangement","stone_count","center_stone",
                "stone_shape","stone_color","setting_style","accent_stones",
                "filigree","milgrain","occasion","profile",
            ],
            "properties": {
                "item_type":         {"type":"string","enum":["ring","earring","necklace","bracelet","pendant","bangle","brooch"]},
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

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(title="Ivana Caption API", version="1.0")


# ── Server management ─────────────────────────────────────────────────────────

def _is_llama_running() -> bool:
    r = subprocess.run(["pgrep", "-f", "serve_llama.py"], capture_output=True)
    return r.returncode == 0


def _is_llama_up() -> bool:
    try:
        return requests.get(f"{SERVER_URL}/v1/models", timeout=1).ok
    except Exception:
        return False


def _start_llama_if_needed() -> None:
    if _is_llama_running() or _is_llama_up():
        return
    log.info("API startup: launching serve_llama.py in background")
    subprocess.Popen(
        [sys.executable, str(SERVE_SCRIPT)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


@app.on_event("startup")
async def _on_startup() -> None:
    _start_llama_if_needed()


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

def _run_caption(pil_img: Image.Image) -> dict:
    orig_w, orig_h = pil_img.size
    img = _resize(pil_img)
    log.info("REQUEST  original=%dx%d  resized=%dx%d", orig_w, orig_h, *img.size)

    t_pre = time.perf_counter()
    data_uri = _to_data_uri(img)
    preprocess_ms = (time.perf_counter() - t_pre) * 1000

    payload = {
        "model": "smolvlm-500m",
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": [
                {"type": "image_url", "image_url": {"url": data_uri}},
            ]},
        ],
        "max_tokens":     MAX_NEW_TOKENS,
        "temperature":    0.1,
        "repeat_penalty": 1.0,
        "response_format": RESPONSE_FORMAT,
        "stream":         True,
    }

    ttft_ms: float | None = None
    chunks: list[str] = []
    t_gen = time.perf_counter()

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
                chunks.append(delta)

    total_s   = time.perf_counter() - t_gen
    raw_text  = "".join(chunks).strip()
    n_out     = len(chunks)
    ttft_s    = (ttft_ms or 0.0) / 1000
    decode_s  = total_s - ttft_s
    tok_per_s = n_out / decode_s if decode_s > 0 else 0.0

    log.info(
        "RESULT  preprocess=%.0fms  ttft=%.0fms  decode=%.2ftok/s  total=%.2fs  tokens=%d",
        preprocess_ms, ttft_ms or 0.0, tok_per_s, total_s, n_out,
    )
    log.debug("Raw model output: %.300s", raw_text)

    # Strip markdown code fences if the model wrapped the JSON anyway
    clean = raw_text
    if clean.startswith("```"):
        clean = clean.split("```")[1]
        if clean.startswith("json"):
            clean = clean[4:]
        clean = clean.strip()

    try:
        attributes = json.loads(clean)
    except json.JSONDecodeError as exc:
        log.error("Model did not return valid JSON: %s | raw=%.300s", exc, raw_text)
        attributes = {"_parse_error": str(exc), "_raw": raw_text}

    return {
        "attributes":    attributes,
        "preprocess_ms": round(preprocess_ms, 1),
        "ttft_ms":       round(ttft_ms or 0.0, 1),
        "decode_tok_s":  round(tok_per_s, 2),
        "total_s":       round(total_s, 3),
        "n_out_tokens":  n_out,
    }


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/health")
def health() -> dict:
    return {"llama_server": "up" if _is_llama_up() else "starting"}


@app.post("/caption")
async def caption(file: UploadFile = File(...)) -> JSONResponse:
    if not _is_llama_up():
        raise HTTPException(
            status_code=503,
            detail="llama.cpp server is not ready yet — check GET /health and retry",
        )

    content_type = file.content_type or ""
    if not content_type.startswith("image/"):
        raise HTTPException(400, f"Expected an image file, got {content_type!r}")

    data = await file.read()
    try:
        img = Image.open(io.BytesIO(data)).convert("RGB")
    except Exception as exc:
        raise HTTPException(400, f"Cannot decode image: {exc}")

    log.info("API /caption  filename=%s  size=%d bytes", file.filename, len(data))

    try:
        result = _run_caption(img)
    except requests.HTTPError as exc:
        log.error("llama.cpp HTTP error: %s", exc)
        raise HTTPException(502, f"llama.cpp error: {exc}")
    except Exception as exc:
        log.exception("Caption failed")
        raise HTTPException(500, str(exc))

    return JSONResponse(result)
