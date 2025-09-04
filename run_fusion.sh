#!/bin/bash

# Clean frames directory but preserve existing videos
rm -rf out_fusion/frames

# Run dbt Fusion visualization with custom images and debug log
python3 dbt_dag_timeline_v2.py \
    -m fusion/manifest.json \
    -r fusion/run_results.json \
    -d fusion/debug.log \
    -o out_fusion \
    --layout bubble \
    --mode cumulative \
    --fps 30 --duration 8 \
    --executed-image images/jaffle.svg \
    --skipped-image images/flower.svg \
    --encode mp4

echo "✅ dbt Fusion visualization complete! Check out_fusion/ for results."