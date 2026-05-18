"""
Generate a self-contained HTML report from a batch_caption results .txt file.

Usage:
    python3.12 generate_html_report.py [results.txt] [--output report.html] [--thumb-size 320]

Defaults:
    results file  →  most recent file in batch_results/
    output        →  batch_results/report_<timestamp>.html
    thumb-size    →  320  (max dimension for embedded thumbnails)
"""

from __future__ import annotations

import argparse
import base64
import html
import io
import json
import re
import sys
import time
from pathlib import Path

from PIL import Image

HERE = Path(__file__).parent
IMAGE_DIR = HERE / "data" / "SAM3 crop Rings"

# ── Parser ────────────────────────────────────────────────────────────────────

def _parse_results(txt_path: Path) -> tuple[dict, list[dict]]:
    """Return (meta, records).  meta has keys from the header block."""
    text  = txt_path.read_text(encoding="utf-8")
    lines = text.splitlines()

    # ── header fields ────────────────────────────────────────────────────────
    meta: dict = {}
    for line in lines:
        for key in ("Generated", "Model", "Input dir", "Total images", "Parsed OK",
                    "Parse failed", "Avg time"):
            if line.startswith(f"{key}:"):
                meta[key] = line.split(":", 1)[1].strip()
                break

    # ── system prompt ─────────────────────────────────────────────────────────
    sp_start = text.find("SYSTEM PROMPT\n" + "─" * 80)
    sp_end   = text.find("=" * 80 + "\nRESULTS")
    if sp_start != -1 and sp_end != -1:
        block = text[sp_start:sp_end]
        # skip the header line and the rule beneath it
        sp_lines = block.splitlines()[2:]
        meta["system_prompt"] = "\n".join(sp_lines).strip()

    # ── records ───────────────────────────────────────────────────────────────
    records: list[dict] = []
    entry_re = re.compile(r"^\[(\d+)/\d+\]\s+(.+)$")
    attr_re  = re.compile(r"^\s{4}(\S+)\s+(.+)$")

    current: dict | None = None
    in_attrs = False

    for line in lines:
        m = entry_re.match(line)
        if m:
            if current:
                records.append(current)
            current  = {"index": int(m.group(1)), "filename": m.group(2).strip(),
                        "attributes": {}, "parse_ok": False, "error": None}
            in_attrs = False
            continue

        if current is None:
            continue

        if line.strip().startswith("size:"):
            parts = line.strip()[5:].strip().split("px")[0].split("×")
            try:
                current["width"], current["height"] = int(parts[0]), int(parts[1])
            except (ValueError, IndexError):
                current["width"] = current["height"] = 0

        elif line.strip().startswith("time:"):
            current["timing"] = line.strip()
            if "parse_ok: True" in line:
                current["parse_ok"] = True

        elif line.strip() == "attributes:":
            in_attrs = True

        elif in_attrs:
            am = attr_re.match(line)
            if am:
                current["attributes"][am.group(1)] = am.group(2).strip()
            else:
                in_attrs = False

        elif line.strip().startswith("ERROR:"):
            current["error"] = line.strip()[6:].strip()

    if current:
        records.append(current)

    return meta, records


# ── Image embedder ────────────────────────────────────────────────────────────

def _embed(filename: str, max_dim: int) -> str:
    path = IMAGE_DIR / filename
    if not path.exists():
        return ""
    try:
        img = Image.open(path).convert("RGB")
        w, h = img.size
        if max(w, h) > max_dim:
            scale = max_dim / max(w, h)
            img = img.resize((int(w * scale), int(h * scale)), Image.Resampling.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=82)
        return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()
    except Exception:
        return ""


# ── HTML builder ──────────────────────────────────────────────────────────────

_FIELD_LABELS = {
    "band_metal":        "Metal",
    "band_style":        "Band Style",
    "band_width":        "Width",
    "band_texture":      "Texture",
    "band_finish":       "Finish",
    "stone_arrangement": "Arrangement",
    "stone_count":       "Stone Count",
    "center_stone":      "Center Stone",
    "stone_shape":       "Shape",
    "stone_color":       "Color",
    "setting_style":     "Setting",
    "accent_stones":     "Accents",
    "filigree":          "Filigree",
    "milgrain":          "Milgrain",
    "occasion":          "Occasion",
    "profile":           "Profile",
}

# Color-coding for key attribute values
_VALUE_CLASSES = {
    "yellow gold": "metal-yg", "rose gold": "metal-rg", "white gold": "metal-wg",
    "platinum": "metal-pt", "silver": "metal-sv",
    "round": "shape-val", "oval": "shape-val", "princess": "shape-val",
    "cushion": "shape-val", "pear": "shape-val", "marquise": "shape-val",
    "heart": "shape-val", "emerald": "shape-val", "radiant": "shape-val",
    "asscher": "shape-val",
    "engagement": "occ-eng", "wedding": "occ-wed", "everyday": "occ-ev",
    "cocktail": "occ-ck", "fashion": "occ-fa",
    "yes": "flag-yes", "no": "flag-no",
}


def _card(rec: dict, thumb_size: int) -> str:
    fname  = html.escape(rec["filename"])
    label  = html.escape(rec["filename"].replace("-IVANA-JEWELS-", " · #").replace("-", " ").rsplit(".", 1)[0])
    src    = _embed(rec["filename"], thumb_size)
    img_tag = (f'<img src="{src}" alt="{fname}" loading="lazy">'
               if src else '<div class="no-img">image not found</div>')

    attr_rows = ""
    for key, display in _FIELD_LABELS.items():
        val = rec["attributes"].get(key, "—")
        vc  = _VALUE_CLASSES.get(val, "")
        cls = f' class="badge {vc}"' if vc else ' class="badge"'
        attr_rows += (
            f'<div class="attr-row">'
            f'<span class="attr-key">{html.escape(display)}</span>'
            f'<span{cls}>{html.escape(val)}</span>'
            f'</div>'
        )

    status = "card-ok" if rec["parse_ok"] else "card-fail"
    return f"""
<div class="card {status}" data-attrs='{json.dumps(rec["attributes"])}'>
  <div class="card-img">{img_tag}</div>
  <div class="card-body">
    <div class="card-title">{label}</div>
    <div class="attrs">{attr_rows}</div>
  </div>
</div>"""


def _build_html(meta: dict, records: list[dict], thumb_size: int) -> str:
    cards_html = "\n".join(_card(r, thumb_size) for r in records)

    # collect unique values per field for filter dropdowns
    filter_fields = ["band_metal", "stone_arrangement", "stone_shape",
                     "center_stone", "setting_style", "occasion"]
    filters_html = ""
    for field in filter_fields:
        vals = sorted({r["attributes"].get(field, "") for r in records if r["attributes"].get(field)})
        label = _FIELD_LABELS.get(field, field)
        options = "\n".join(f'<option value="{html.escape(v)}">{html.escape(v)}</option>' for v in vals)
        filters_html += f"""
        <div class="filter-group">
          <label>{label}</label>
          <select data-field="{field}" onchange="applyFilters()">
            <option value="">All</option>
            {options}
          </select>
        </div>"""

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Ivana Ring Captions — Batch Report</title>
<style>
@import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:ital,wght@0,300;0,400;1,300;1,400&family=Jost:wght@300;400;500&display=swap');

*, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}

body {{
  background: #FAF7F2;
  font-family: 'Jost', sans-serif;
  font-weight: 300;
  color: #2A2018;
}}

/* ── Nav ── */
.nav {{
  background: #1A1410;
  padding: 1.1rem 2.5rem;
  display: flex;
  align-items: center;
  gap: 0.5rem;
}}
.nav-brand {{
  font-family: 'Cormorant Garamond', serif;
  font-size: 1.05rem;
  letter-spacing: 0.35em;
  color: #E8D5B0;
  text-transform: uppercase;
}}
.nav-x {{ color: #C9A96E; font-family: 'Cormorant Garamond', serif; }}
.nav-sub {{
  margin-left: auto;
  font-size: 0.58rem;
  letter-spacing: 0.22em;
  color: rgba(201,169,110,0.5);
  text-transform: uppercase;
}}
.gold-rule {{
  height: 1px;
  background: linear-gradient(90deg, #C9A96E 0%, rgba(201,169,110,0.04) 100%);
  opacity: 0.4;
}}

/* ── Header ── */
.page-header {{
  padding: 2.2rem 2.5rem 1.2rem;
}}
.page-title {{
  font-family: 'Cormorant Garamond', serif;
  font-size: 2.2rem;
  font-weight: 300;
  font-style: italic;
  color: #1A1410;
}}
.meta-row {{
  display: flex;
  flex-wrap: wrap;
  gap: 1.5rem;
  margin-top: 0.9rem;
}}
.meta-chip {{
  background: #fff;
  border: 1px solid #EDE8E0;
  border-radius: 3px;
  padding: 0.35rem 0.8rem;
  font-size: 0.68rem;
  letter-spacing: 0.1em;
  color: #8C7B6B;
  text-transform: uppercase;
}}
.meta-chip strong {{ color: #1A1410; font-weight: 500; }}

/* ── Controls ── */
.controls {{
  background: #fff;
  border-top: 1px solid #EDE8E0;
  border-bottom: 1px solid #EDE8E0;
  padding: 0.9rem 2.5rem;
  display: flex;
  flex-wrap: wrap;
  gap: 1rem;
  align-items: flex-end;
}}
.search-wrap {{
  flex: 1 1 200px;
  min-width: 180px;
}}
.search-wrap label,
.filter-group label {{
  display: block;
  font-size: 0.58rem;
  letter-spacing: 0.18em;
  text-transform: uppercase;
  color: #C9A96E;
  margin-bottom: 0.3rem;
  font-weight: 500;
}}
.search-wrap input {{
  width: 100%;
  border: 1px solid #EDE8E0;
  border-radius: 3px;
  padding: 0.42rem 0.75rem;
  font-family: 'Jost', sans-serif;
  font-size: 0.8rem;
  background: #FAF7F2;
  color: #2A2018;
  outline: none;
}}
.search-wrap input:focus {{ border-color: #C9A96E; }}
.filter-group select {{
  border: 1px solid #EDE8E0;
  border-radius: 3px;
  padding: 0.42rem 0.6rem;
  font-family: 'Jost', sans-serif;
  font-size: 0.75rem;
  background: #FAF7F2;
  color: #2A2018;
  cursor: pointer;
  outline: none;
}}
.filter-group select:focus {{ border-color: #C9A96E; }}
.btn-reset {{
  border: 1px solid #C9A96E;
  background: transparent;
  color: #C9A96E;
  border-radius: 3px;
  padding: 0.42rem 1rem;
  font-family: 'Jost', sans-serif;
  font-size: 0.7rem;
  letter-spacing: 0.12em;
  text-transform: uppercase;
  cursor: pointer;
  align-self: flex-end;
}}
.btn-reset:hover {{ background: #C9A96E; color: #fff; }}
.count-label {{
  font-size: 0.68rem;
  color: #8C7B6B;
  letter-spacing: 0.08em;
  align-self: flex-end;
  padding-bottom: 0.45rem;
  white-space: nowrap;
}}

/* ── Grid ── */
.grid {{
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(310px, 1fr));
  gap: 1.4rem;
  padding: 1.8rem 2.5rem 3rem;
}}

/* ── Card ── */
.card {{
  background: #fff;
  border: 1px solid #EDE8E0;
  border-radius: 6px;
  overflow: hidden;
  display: flex;
  flex-direction: column;
  transition: box-shadow 0.18s;
}}
.card:hover {{ box-shadow: 0 4px 18px rgba(26,20,16,0.10); }}
.card-fail {{ border-left: 3px solid #BF360C; }}

.card-img {{
  background: #F5F1EB;
  display: flex;
  align-items: center;
  justify-content: center;
  overflow: hidden;
  height: 200px;
}}
.card-img img {{
  width: 100%;
  height: 100%;
  object-fit: contain;
  display: block;
}}
.no-img {{
  font-size: 0.7rem;
  color: #8C7B6B;
  letter-spacing: 0.1em;
}}

.card-body {{
  padding: 0.9rem 1rem 1rem;
  flex: 1;
}}
.card-title {{
  font-family: 'Cormorant Garamond', serif;
  font-size: 0.82rem;
  font-weight: 400;
  color: #1A1410;
  line-height: 1.35;
  margin-bottom: 0.7rem;
  letter-spacing: 0.02em;
}}

.attrs {{
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 0.3rem 0.5rem;
}}
.attr-row {{
  display: contents;
}}
.attr-key {{
  font-size: 0.6rem;
  letter-spacing: 0.12em;
  text-transform: uppercase;
  color: #8C7B6B;
  font-weight: 500;
  align-self: center;
}}
.badge {{
  font-size: 0.68rem;
  font-weight: 400;
  color: #2A2018;
  background: #F5F1EB;
  border-radius: 2px;
  padding: 0.18rem 0.45rem;
  letter-spacing: 0.04em;
  align-self: center;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}}

/* Metal badges */
.metal-yg {{ background: #FFF8DC; color: #7B5E00; }}
.metal-rg {{ background: #FDE8E0; color: #8B2500; }}
.metal-wg {{ background: #EEF2F7; color: #2A3A5C; }}
.metal-pt {{ background: #ECECEC; color: #333; }}
.metal-sv {{ background: #E8ECF0; color: #2C3E50; }}

/* Shape */
.shape-val {{ background: #EAF1FB; color: #1A3A6C; }}

/* Occasion */
.occ-eng {{ background: #F3E5F5; color: #6A1B9A; }}
.occ-wed {{ background: #FCE4EC; color: #880E4F; }}
.occ-ev  {{ background: #E8F5E9; color: #1B5E20; }}
.occ-ck  {{ background: #FFF3E0; color: #E65100; }}
.occ-fa  {{ background: #E0F2F1; color: #004D40; }}

/* Flags */
.flag-yes {{ background: #E8F5E9; color: #2E7D32; }}
.flag-no  {{ background: #F5F5F5; color: #9E9E9E; }}

.hidden {{ display: none !important; }}
</style>
</head>
<body>

<nav class="nav">
  <span class="nav-brand">Ivana</span>
  <span class="nav-x">&nbsp;×&nbsp;</span>
  <span class="nav-brand">Devx</span>
  <span class="nav-sub">Ring Caption Batch Report</span>
</nav>
<div class="gold-rule"></div>

<div class="page-header">
  <div class="page-title">Ring Attribute Extraction</div>
  <div class="meta-row">
    <span class="meta-chip">Generated <strong>{html.escape(meta.get("Generated","—"))}</strong></span>
    <span class="meta-chip">Model <strong>{html.escape(meta.get("Model","—"))}</strong></span>
    <span class="meta-chip">Images <strong>{html.escape(meta.get("Total images","—"))}</strong></span>
    <span class="meta-chip">Parsed OK <strong>{html.escape(meta.get("Parsed OK","—"))}</strong></span>
    <span class="meta-chip">Avg time <strong>{html.escape(meta.get("Avg time","—"))}</strong></span>
  </div>
</div>

<div class="controls">
  <div class="search-wrap">
    <label>Search by filename</label>
    <input type="text" id="search" placeholder="e.g. oval, halo, pear…" oninput="applyFilters()">
  </div>
  {filters_html}
  <button class="btn-reset" onclick="resetFilters()">Reset</button>
  <span class="count-label" id="count-label">{len(records)} rings</span>
</div>

<div class="grid" id="grid">
{cards_html}
</div>

<script>
const cards = Array.from(document.querySelectorAll('.card'));
const countLabel = document.getElementById('count-label');

function applyFilters() {{
  const search = document.getElementById('search').value.toLowerCase();
  const selects = Array.from(document.querySelectorAll('select[data-field]'));
  let visible = 0;
  cards.forEach(card => {{
    const attrs = JSON.parse(card.dataset.attrs || '{{}}');
    const title = card.querySelector('.card-title').textContent.toLowerCase();
    const matchSearch = !search || title.includes(search);
    const matchFilters = selects.every(sel => {{
      const val = sel.value;
      return !val || attrs[sel.dataset.field] === val;
    }});
    const show = matchSearch && matchFilters;
    card.classList.toggle('hidden', !show);
    if (show) visible++;
  }});
  countLabel.textContent = visible + ' ring' + (visible !== 1 ? 's' : '');
}}

function resetFilters() {{
  document.getElementById('search').value = '';
  document.querySelectorAll('select[data-field]').forEach(s => s.value = '');
  applyFilters();
}}
</script>

</body>
</html>"""


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("results_file", nargs="?", default="",
                        help="Path to batch results .txt (default: latest in batch_results/)")
    parser.add_argument("--output", default="",
                        help="Output HTML path (default: batch_results/report_<ts>.html)")
    parser.add_argument("--thumb-size", type=int, default=320,
                        help="Max thumbnail dimension in px (default: 320)")
    args = parser.parse_args()

    if args.results_file:
        txt_path = Path(args.results_file)
    else:
        candidates = sorted((HERE / "batch_results").glob("results_*.txt"))
        if not candidates:
            print("No results_*.txt found in batch_results/", file=sys.stderr)
            sys.exit(1)
        txt_path = candidates[-1]
    print(f"Parsing {txt_path} …")

    meta, records = _parse_results(txt_path)
    print(f"  {len(records)} records found, embedding thumbnails …")

    total = len(records)
    for i, rec in enumerate(records, 1):
        if i % 20 == 0 or i == total:
            print(f"  {i}/{total}", end="\r", flush=True)

    out_path = (Path(args.output) if args.output
                else HERE / "batch_results" / f"report_{time.strftime('%Y%m%d_%H%M%S')}.html")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    html_str = _build_html(meta, records, args.thumb_size)
    out_path.write_text(html_str, encoding="utf-8")
    size_mb = out_path.stat().st_size / 1_048_576
    print(f"\nReport written → {out_path}  ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
