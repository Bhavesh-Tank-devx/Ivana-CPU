"""
Batch ring attribute extractor.

Processes every image in a folder against the llama.cpp server and writes a
structured report file with model metadata, the system prompt used, then each
image name and its JSON result.

Usage:
    python3.12 batch_caption.py [--input-dir DIR] [--output FILE] [--workers N]

Defaults:
    --input-dir  "data/SAM3 crop Rings"  (relative to this script)
    --output     "batch_results/results_<timestamp>.txt"
    --workers    1  (increase for parallel requests if the server can handle it)
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import logging
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
from PIL import Image

# ── Config ────────────────────────────────────────────────────────────────────
HERE           = Path(__file__).parent
SERVER_URL     = "http://localhost:8080"
MODEL_NAME     = "minicpmv-4.6"
MAX_NEW_TOKENS = 320          # 15 enum fields (~150 tok) + 2–3 sentence caption (~80 tok) + headroom
IMAGE_MAX_DIM  = 384          # SigLIP2 at 384px → ~64 vision tokens, fast on CPU
TEMPERATURE    = 0.1
IMAGE_EXTS     = {".jpg", ".jpeg", ".png", ".webp"}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("batch_caption")

# ── System prompt (ring-specific) ─────────────────────────────────────────────
SYSTEM_PROMPT = """\
You are an expert ring attribute extractor with deep knowledge of gemology and fine jewellery design. \
Every image you receive is a close-up photograph of a RING. Your task is to examine it precisely and \
output its visual attributes as a JSON object.

CRITICAL VISUAL CHECKS — read these before labeling every field:

has_biggest_stone — DECIDE THIS FIRST, before any biggest_stone_* field:
  • yes — the ring has ONE clearly dominant / largest focal stone (a center stone that stands out
          in size from everything else on the ring)
  • no  — there is NO single dominant stone. This includes: a plain metal band with no stones at all,
          AND bands covered only in uniform small stones with no larger focal stone (e.g. a full pavé
          or eternity band of equal-sized stones).
  RULE: if has_biggest_stone is "no", you MUST set biggest_stone_material, biggest_stone_shape, and
  biggest_stone_color all to "none". Only fill those three fields when has_biggest_stone is "yes".

biggest_stone_shape — measure the outline of the BIGGEST / MAIN (focal) stone carefully:
  • princess — square outline with sharp 90-degree corners; no rounding
  • round    — perfectly circular; width and height are equal. If the stone is elongated AT ALL, it is not round.
  • oval     — ellipse longer than wide; length:width ratio roughly 1.3–1.5
  • cushion  — square or rectangular with noticeably rounded corners; pillow-like silhouette
  • pear     — teardrop; one pointed tip and one fully rounded end
  • marquise — eye or football shape; two pointed ends, widest in the middle
  • emerald  — rectangle with clipped corners; step-cut facets visible as concentric tiers/steps
  • radiant  — rectangle or square with clipped corners; brilliant-cut (sparkly) not step-cut
  • heart    — clear heart silhouette with a cleft at the top centre
  • asscher  — square step-cut with heavily clipped corners; nearly octagonal outline
  • none     — there is no biggest/main stone (has_biggest_stone is "no")


stone_arrangement — how stones are laid out on the ring:
  • solitaire   — one center stone, plain or minimal band, no surrounding stones
  • halo        — center stone encircled by a ring of smaller accent stones
  • pavé        — band surface densely covered with small stones flush to the metal
  • cluster     — group of stones close together without a single dominant center
  • three-stone — exactly three prominent stones in a row (past-present-future style)
  • eternity    — stones run continuously all the way around the band
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

has_accent_stones — does the ring have small SECONDARY stones?
  • yes — small secondary stones are present in addition to (or instead of) the main stone
          (e.g. a halo, pavé, side stones, channel-set stones)
  • no  — no small secondary stones are present

band_texture — describes ONLY the surface treatment of the metal: smooth | hammered | engraved | none.
  Do NOT use band_texture for milgrain (that is its own yes/no field) or for braided
  (braided belongs to band_style). Keep those concepts out of band_texture.

caption — a short, factual VISUAL description of the ring (2–3 sentences), written for shoppers to
search by. Describe only what is clearly visible: metal color, band shape, the main stone
(shape/color/relative size), any accent stones, the setting, and notable detailing (filigree,
milgrain, engraving). Write it naturally, the way someone would search — e.g. "A rose gold halo
engagement-style ring with a round center stone surrounded by small accent stones on a pavé band."
Do NOT merely list the enum values, and do NOT invent sentiment, backstory, gemstone identity you
cannot confirm, or anything not visible in the image.

Output ONLY a JSON object — no prose, no markdown fences, no explanation.

Allowed values per field (pick exactly one):
  band_metal:        yellow gold | white gold | rose gold | silver | platinum | mixed | unknown
  band_style:        plain | twisted | split-shank | bypass | tapered | braided | unknown
  band_width:        thin | medium | wide | statement
  band_texture:      smooth | hammered | engraved | none
  band_finish:       polished | matte | brushed | satin | unknown
  stone_arrangement: solitaire | halo | pavé | cluster | three-stone | eternity | none
  stone_count:       single | two-stone | three-stone | multi | eternity | none
  has_biggest_stone: yes | no
  biggest_stone_material: diamond | ruby | emerald | sapphire | pearl | opal | amethyst | garnet | topaz | none
  biggest_stone_shape:    round | oval | princess | cushion | pear | marquise | emerald | radiant | heart | asscher | none
  biggest_stone_color:    clear | red | blue | green | pink | purple | yellow | orange | multicolor | none
  setting_style:     prong | bezel | channel | tension | pavé | flush | invisible | none
  has_accent_stones: yes | no

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
                "band_finish","stone_arrangement","stone_count","has_biggest_stone",
                "biggest_stone_material","biggest_stone_shape","biggest_stone_color",
                "setting_style","has_accent_stones",
                "filigree","milgrain","caption",
            ],
            "properties": {
                "band_metal":        {"type":"string","enum":["yellow gold","white gold","rose gold","silver","platinum","mixed","unknown"]},
                "band_style":        {"type":"string","enum":["plain","twisted","split-shank","bypass","tapered","braided","unknown"]},
                "band_width":        {"type":"string","enum":["thin","medium","wide","statement"]},
                "band_texture":      {"type":"string","enum":["smooth","hammered","engraved","none"]},
                "band_finish":       {"type":"string","enum":["polished","matte","brushed","satin","unknown"]},
                "stone_arrangement": {"type":"string","enum":["solitaire","halo","pavé","cluster","three-stone","eternity","none"]},
                "stone_count":       {"type":"string","enum":["single","two-stone","three-stone","multi","eternity","none"]},
                "has_biggest_stone":      {"type":"string","enum":["yes","no"]},
                "biggest_stone_material": {"type":"string","enum":["diamond","ruby","emerald","sapphire","pearl","opal","amethyst","garnet","topaz","none"]},
                "biggest_stone_shape":    {"type":"string","enum":["round","oval","princess","cushion","pear","marquise","emerald","radiant","heart","asscher","none"]},
                "biggest_stone_color":    {"type":"string","enum":["clear","red","blue","green","pink","purple","yellow","orange","multicolor","none"]},
                "setting_style":     {"type":"string","enum":["prong","bezel","channel","tension","pavé","flush","invisible","none"]},
                "has_accent_stones": {"type":"string","enum":["yes","no"]},
                "filigree":          {"type":"string","enum":["yes","no"]},
                "milgrain":          {"type":"string","enum":["yes","no"]},
                "caption":           {"type":"string","minLength":20,"maxLength":400},
            },
        },
    },
}


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

def _infer(img_path: Path) -> dict:
    pil = Image.open(img_path).convert("RGB")
    orig_size = pil.size
    pil = _resize(pil)

    t0       = time.perf_counter()
    data_uri = _to_data_uri(pil)

    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": [
                {"type": "image_url", "image_url": {"url": data_uri}},
            ]},
        ],
        "max_tokens":      MAX_NEW_TOKENS,
        "temperature":     TEMPERATURE,
        "repeat_penalty":  1.0,
        "response_format": RESPONSE_FORMAT,
        "stream":          True,
    }

    chunks: list[str] = []
    ttft_ms: float | None = None
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
                continue
            if delta:
                if ttft_ms is None:
                    ttft_ms = (time.perf_counter() - t_gen) * 1000
                chunks.append(delta)

    total_s  = time.perf_counter() - t0
    raw_text = "".join(chunks).strip()

    clean = raw_text
    if clean.startswith("```"):
        clean = clean.split("```")[1]
        if clean.startswith("json"):
            clean = clean[4:]
        clean = clean.strip()

    try:
        attributes = json.loads(clean)
        parse_ok   = True
    except json.JSONDecodeError:
        attributes = None
        parse_ok   = False

    return {
        "filename":   img_path.name,
        "orig_size":  orig_size,
        "total_s":    round(total_s, 2),
        "ttft_ms":    round(ttft_ms or 0.0, 1),
        "raw_text":   raw_text,
        "attributes": attributes,
        "parse_ok":   parse_ok,
    }


def _process_one(img_path: Path, index: int, total: int) -> dict:
    log.info("[%d/%d] Processing %s", index, total, img_path.name)
    try:
        result = _infer(img_path)
        status = "OK" if result["parse_ok"] else "PARSE_FAIL"
        log.info("[%d/%d] %s  %.2fs  %s", index, total, img_path.name, result["total_s"], status)
        return result
    except Exception as exc:
        log.error("[%d/%d] %s  ERROR: %s", index, total, img_path.name, exc)
        return {
            "filename":   img_path.name,
            "orig_size":  None,
            "total_s":    0.0,
            "ttft_ms":    0.0,
            "raw_text":   "",
            "attributes": None,
            "parse_ok":   False,
            "error":      str(exc),
        }


# ── Report writer ─────────────────────────────────────────────────────────────

def _write_report(results: list[dict], output_path: Path, input_dir: Path) -> None:
    ok_count   = sum(1 for r in results if r["parse_ok"])
    fail_count = len(results) - ok_count
    avg_time   = sum(r["total_s"] for r in results) / len(results) if results else 0.0

    lines: list[str] = []

    # ── Header ────────────────────────────────────────────────────────────────
    lines += [
        "=" * 80,
        "BATCH RING ATTRIBUTE EXTRACTION REPORT",
        "=" * 80,
        "",
        f"Generated:    {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"Model:        {MODEL_NAME}  Q4_K_M  (llama.cpp server @ {SERVER_URL})",
        f"Input dir:    {input_dir.resolve()}",
        f"Total images: {len(results)}",
        f"Parsed OK:    {ok_count}",
        f"Parse failed: {fail_count}",
        f"Avg time:     {avg_time:.2f}s / image",
        "",
        "─" * 80,
        "SYSTEM PROMPT",
        "─" * 80,
        "",
        SYSTEM_PROMPT,
        "",
        "=" * 80,
        "RESULTS",
        "=" * 80,
    ]

    # ── Per-image entries ─────────────────────────────────────────────────────
    for i, r in enumerate(results, 1):
        lines += [
            "",
            f"[{i}/{len(results)}]  {r['filename']}",
            f"  size: {r['orig_size'][0]}×{r['orig_size'][1]}px" if r["orig_size"] else "  size: unknown",
            f"  time: {r['total_s']:.2f}s   ttft: {r['ttft_ms']:.0f}ms   parse_ok: {r['parse_ok']}",
        ]
        if "error" in r:
            lines.append(f"  ERROR: {r['error']}")
        elif r["attributes"] is not None:
            lines.append("  attributes:")
            for k, v in r["attributes"].items():
                lines.append(f"    {k:<22} {v}")
        else:
            lines.append("  RAW OUTPUT (parse failed):")
            lines.append(f"    {r['raw_text']}")

    # ── Summary table (JSON parse failures) ───────────────────────────────────
    if fail_count:
        lines += [
            "",
            "─" * 80,
            f"PARSE FAILURES ({fail_count})",
            "─" * 80,
        ]
        for r in results:
            if not r["parse_ok"]:
                lines.append(f"  {r['filename']}")
                if r.get("raw_text"):
                    lines.append(f"    raw: {r['raw_text'][:200]}")

    lines += ["", "=" * 80, "END OF REPORT", "=" * 80, ""]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")
    log.info("Report written → %s", output_path)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Batch ring attribute extractor")
    parser.add_argument(
        "--input-dir", default=str(HERE / "data" / "SAM3 crop Rings"),
        help="Folder of ring images",
    )
    parser.add_argument(
        "--output", default="",
        help="Output report file (default: batch_results/results_<timestamp>.txt)",
    )
    parser.add_argument(
        "--workers", type=int, default=1,
        help="Parallel workers (increase only if the server supports concurrent requests)",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        log.error("Input directory not found: %s", input_dir)
        sys.exit(1)

    images = sorted(p for p in input_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    if not images:
        log.error("No images found in %s", input_dir)
        sys.exit(1)

    output_path = Path(args.output) if args.output else (
        HERE / "batch_results" / f"results_{time.strftime('%Y%m%d_%H%M%S')}.txt"
    )

    log.info("Found %d images in %s", len(images), input_dir)
    log.info("Output → %s", output_path)

    # Verify server is reachable before starting
    try:
        requests.get(f"{SERVER_URL}/v1/models", timeout=3).raise_for_status()
    except Exception as exc:
        log.error("llama.cpp server not reachable at %s: %s", SERVER_URL, exc)
        log.error("Start it first with:  python3.12 serve_llama.py")
        sys.exit(1)

    t_batch = time.perf_counter()
    results: list[dict] = [None] * len(images)  # type: ignore[list-item]

    if args.workers == 1:
        for i, img_path in enumerate(images, 1):
            results[i - 1] = _process_one(img_path, i, len(images))
    else:
        futures = {}
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            for i, img_path in enumerate(images, 1):
                fut = pool.submit(_process_one, img_path, i, len(images))
                futures[fut] = i - 1
            for fut in as_completed(futures):
                results[futures[fut]] = fut.result()

    elapsed = time.perf_counter() - t_batch
    log.info("Batch complete: %d images in %.1fs (%.2f s/img)", len(images), elapsed, elapsed / len(images))

    _write_report(results, output_path, input_dir)


if __name__ == "__main__":
    main()
