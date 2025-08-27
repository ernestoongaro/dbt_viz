```
python3 dbt_dag_timeline_v2.py \
  -m manifest.json -r run_results.json -o out_4k \
  --layout layered \
  --mode cumulative \
  --highlight-critical \
  --fps 60 --duration 12 \
  --title "" \
  --encode mp4
```
