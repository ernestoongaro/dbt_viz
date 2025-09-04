#!/bin/bash

# Clean frames directory but preserve existing videos
rm -rf out_fusion/frames

# Run dbt Fusion visualization (Pillow renderer + imageio encoder)
python3 -m dbt_viz.cli \
    -m fusion/manifest.json \
    -r fusion/run_results.json \
    -d fusion/debug.log \
    -o out_fusion \
    --fps 30 --duration 12 \
    --width 3840 --height 2160 \
    --fade-frames 12 \
    --icon-scale 1.0 \
    --exec-diam-min 72 --exec-diam-max 240 \
    --skip-diam-min 96 --skip-diam-max 280 \
    --seed 42 \
    --encode

echo "✅ dbt Fusion visualization complete! Check out_fusion/ for results."
