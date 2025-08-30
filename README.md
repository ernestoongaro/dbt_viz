Add `run_results.json` and `manifest.json` and get some pretty movie

The visualization automatically detects dbt Fusion vs Core and highlights Fusion's intelligent skipping in cyan.

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

### dbt Fusion vs Core
- **Core**: `/core/` directory contains standard dbt Core artifacts
- **Fusion**: `/fusion/` directory contains dbt Fusion artifacts with intelligent skipping
- Fusion runs show cyan highlighting for skipped models and display skip counts

 
