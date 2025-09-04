# AGENTS.md

This document guides Codex CLI when working in this repository.

## Project Overview

Animated visualization of dbt runs using only Pillow + numpy + imageio. It reads dbt artifacts (`manifest.json`, `run_results.json`, and optional `debug.log`) and renders a timeline where icons appear as models build. No Graphviz/Matplotlib, no DAG edges — just beautiful, randomized icon placement with size, fade-in, and gentle rotation.

- Executed models: `images/jaffle.png`
- Fusion‑skipped models: `images/flower.png`

## Key Files

- `dbt_viz/parser.py` — Parses dbt artifacts into a simple structure (nodes + edges metadata). Detects dbt Core vs Fusion and marks Fusion‑skipped models (optionally via `debug.log`).
- `dbt_viz/renderer.py` — Pillow-based frame renderer with:
  - Random, reproducible placement (`--seed`)
  - Size scaling by execution time (executed) and dependency count (skipped)
  - Resolution-aware sizing + optional `--icon-scale`
  - Smooth fade-in and subtle per-node rotation
- `dbt_viz/video.py` — MP4 encoder using `imageio` + `libx264` backend. Ensures consistent frame size and converts RGBA → RGB for compatibility.
- `dbt_viz/cli.py` — CLI entry point. If `--outfile` is omitted, MP4s are named `dbt_core_YYYYMMDD_HHMMSS.mp4` or `dbt_fusion_YYYYMMDD_HHMMSS.mp4` automatically.
- `run_core.sh`, `run_fusion.sh` — 12s @ 30fps 4K runs with tuned visual flags.
- `requirements.txt` — `Pillow`, `numpy`, `imageio`, `imageio-ffmpeg`.

## Sample Artifacts

- `core/` — dbt Core artifacts (`manifest.json`, `run_results.json`).
- `fusion/` — dbt Fusion artifacts (`manifest.json`, `run_results.json`, `debug.log`).

## Quick Commands

Preview (fast, low-res):
```bash
python3 -m dbt_viz.cli \
  -m core/manifest.json \
  -r core/run_results.json \
  -o out_core_preview \
  --fps 5 --duration 1 \
  --width 640 --height 360
```

Full 4K runs (scripts already configured):
```bash
./run_core.sh
./run_fusion.sh
```

## CLI Options (Highlights)

- `--width`, `--height`: frame size (defaults 1920x1080)
- `--fps`, `--duration`: animation timing
- `--seed`: random placement seed
- `--fade-frames`: fade-in length in frames
- `--icon-scale`: multiplies sizes after resolution scaling
- `--exec-diam-min/--exec-diam-max`: executed diameter range at 1920 width (px)
- `--skip-diam-min/--skip-diam-max`: skipped diameter range at 1920 width (px)
- `--size-quantum`: round diameters to nearest multiple (px)
- `--executed-icon`, `--skipped-icon`: paths to icon PNGs
- `--encode`: write MP4 via imageio/ffmpeg
- `--outfile`: custom MP4 filename; otherwise auto‑labels with core/fusion + timestamp

## Development Notes

- No Graphviz/Matplotlib — keep renderer Pillow-only and performant.
- The parser loads JSON fully; artifacts are large but manageable. Avoid unnecessary copies.
- Rendering performance: cache resized icons and quantized rotations.
- Aesthetics: random placement is computed once per run; rotation/alpha animate per frame.
- Icons are required; renderer errors clearly if icons are missing.

## Common Tasks

- Tuning visuals for 4K: adjust `--icon-scale`, diameter ranges, and `--fade-frames` in `run_core.sh` / `run_fusion.sh`.
- Quick sanity checks: use preview settings (low fps/res) before full renders.
- Encoding robustness: the encoder normalizes frame sizes and strips alpha to avoid ffmpeg errors.

