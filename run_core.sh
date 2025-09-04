#!/bin/bash

# Clean frames directory but preserve existing videos
rm -rf out_core/frames

# Run dbt Core visualization (Pillow renderer + imageio encoder)
python3 -m dbt_viz.cli \
    -m core/manifest.json \
    -r core/run_results.json \
    -o out_core \
    --fps 30 --duration 12 \
    --width 3840 --height 2160 \
    --fade-frames 12 \
    --icon-scale 2.2 \
    --exec-diam-min 160 --exec-diam-max 640 \
    --skip-diam-min 200 --skip-diam-max 720 \
    --seed 42 \
    --encode

echo "✅ dbt Core visualization complete! Check out_core/ for results."
