Add `run_results.json` and `manifest.json` and get a simple PNG-sequenced movie.

This version removes Graphviz/Matplotlib and uses only:
- Pillow (PIL): image creation, drawing, compositing, resizing
- numpy: scaling and normalization math
- imageio: writes MP4 via ffmpeg (libx264)

## Quick Start

### dbt Core (orange execution):
```bash
rm -rf out_core && python3 -m dbt_viz.cli \
  -m core/manifest.json \
  -r core/run_results.json \
  -o out_core \
  --fps 15 --duration 4 \
  --encode
```

### dbt Fusion (green skipped + orange executed):
```bash
rm -rf out_fusion && python3 -m dbt_viz.cli \
  -m fusion/manifest.json \
  -r fusion/run_results.json \
  -d fusion/debug.log \
  -o out_fusion \
  --fps 15 --duration 4 \
  --encode
```

## Options
- `--width`, `--height`: output frame size (default 1920x1080)
- `--skip-zero-duration`: treat 0s tasks as reused (don’t highlight)
- `--outfile`: mp4 filename (default `timeline.mp4`)
- If `--encode` is used without `--outfile`, the CLI names videos as `dbt_core_YYYYMMDD_HHMMSS.mp4` or `dbt_fusion_YYYYMMDD_HHMMSS.mp4` automatically.
- `--executed-icon`: PNG for executed models (default `images/jaffle.png`)
- `--skipped-icon`: PNG for Fusion-skipped models (default `images/flower.png`)

## Samples
- Core artifacts in `/core/`
- Fusion artifacts in `/fusion/` (include `debug.log` to better detect reused models)

## Dependencies
```
pip install pillow numpy imageio imageio-ffmpeg
```
Requires `ffmpeg` installed and on PATH for MP4 encoding.

## Icons
Place your PNG icons at:
- `images/jaffle.png` for executed models
- `images/flower.png` for Fusion-skipped models

Or pass paths via `--executed-icon` and `--skipped-icon`. The renderer requires icons and will error if they’re missing.

## Agents
See `AGENTS.md` for Codex CLI agent guidance when working in this repo.
