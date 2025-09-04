Add `run_results.json` and `manifest.json` and get some pretty movie

The visualization automatically detects dbt Fusion vs Core and highlights Fusion's intelligent skipping with dramatic ghost effects.

## Quick Start

### Test dbt Core (busy orange activity):
```bash
rm -rf out_core && python3 dbt_dag_timeline_v2.py \
  -m core/manifest.json \
  -r core/run_results.json \
  -o out_core \
  --layout layered \
  --mode cumulative \
  --fps 15 --duration 4 \
  --title "dbt Core Timeline" \
  --encode mp4
```

### Test dbt Fusion (green efficiency + ghost skips):
```bash
rm -rf out_fusion && python3 dbt_dag_timeline_v2.py \
  -m fusion/manifest.json \
  -r fusion/run_results.json \
  -o out_fusion \
  --layout layered \
  --mode cumulative \
  --fps 15 --duration 4 \
  --title "dbt Fusion Timeline" \
  --encode mp4
```

## Advanced Usage

### High-quality 4K output:
```bash
python3 dbt_dag_timeline_v2.py \
  -m manifest.json -r run_results.json -o out_4k \
  --layout layered \
  --mode cumulative \
  --fps 60 --duration 12 \
  --title "dbt DAG Timeline" \
  --encode mp4
```

### Beautiful bubble chart layout (organic clustering):
```bash
python3 dbt_dag_timeline_v2.py \
  -m fusion/manifest.json -r fusion/run_results.json -d fusion/debug.log \
  -o out_fusion \
  --layout bubble \
  --mode cumulative \
  --fps 30 --duration 8 \
  --title "dbt Fusion Timeline" \
  --encode mp4
```

## Sample Data

This repository uses real dbt Cloud run artifacts for comparison:

- **dbt Core**: [Run #412885625](https://cloud.getdbt.com/deploy/1/projects/672/runs/412885625?tabId=artifacts) - stored in `/core/`
- **dbt Fusion**: [Run #425458971](https://cloud.getdbt.com/deploy/1/projects/672/runs/425458971) - stored in `/fusion/` - you need to get debug.log as well

### Visual Differences
- **Core**: `/core/` directory - shows all models building in orange
- **Fusion**: `/fusion/` directory - shows executed models in orange + skipped models as bright green nodes
- **Layouts**: Choose from `layered` (DAG structure), `spring`, `dot`, `sfdp`, or `bubble` (organic clustering)
- **Bubble layout**: Proportional sizing based on execution time (orange) and complexity (green)
- Fusion displays efficiency statistics and skip counts

 
