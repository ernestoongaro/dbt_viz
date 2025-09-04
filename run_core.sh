#!/bin/bash

# Clean frames directory but preserve existing videos
rm -rf out_core/frames

# Run dbt Core visualization with custom images
python3 dbt_dag_timeline_v2.py \
    -m core/manifest.json \
    -r core/run_results.json \
    -o out_core \
    --layout bubble \
    --mode cumulative \
    --fps 30 --duration 8 \
    --executed-image images/jaffle.svg \
    --skipped-image images/flower.svg \
    --encode mp4

echo "✅ dbt Core visualization complete! Check out_core/ for results."