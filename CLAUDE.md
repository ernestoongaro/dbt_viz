# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a dbt visualization tool that creates animated timeline videos of dbt DAG execution from dbt artifacts (`manifest.json` and `run_results.json`). The main script processes dbt artifacts and generates frame-by-frame PNG images that can be compiled into MP4/MOV videos showing the execution timeline of dbt models.

The tool automatically detects dbt Core vs dbt Fusion and provides special visualization features for Fusion's intelligent skipping capabilities.

## Key Files

- `dbt_dag_timeline_v2.py` - Main visualization script
- `core/` - Contains sample dbt artifacts (manifest.json, run_results.json) for core dbt project
- `fusion/` - Contains sample dbt artifacts for fusion dbt project
- `out_core/` and `out_fusion/` - Generated output directories containing rendered frames

## Common Commands

### Basic Usage
```bash
python3 dbt_dag_timeline_v2.py \
  -m manifest.json -r run_results.json -o out_4k \
  --layout layered \
  --mode cumulative \
  --fps 60 --duration 12 \
  --title "dbt DAG Timeline" \
  --encode mp4
```

### dbt Fusion with Debug Log
For dbt Fusion runs, include the debug.log for accurate skip detection:
```bash
python3 dbt_dag_timeline_v2.py \
  -m fusion/manifest.json -r fusion/run_results.json -d fusion/debug.log \
  -o out_fusion \
  --layout layered \
  --mode cumulative \
  --fps 30 --duration 8 \
  --title "dbt Fusion Timeline" \
  --encode mp4
```


## Architecture

The codebase follows a modular structure:

### Core Data Processing
- `load_manifest_and_results()` - Parses dbt artifacts and builds NetworkX graph
- `parse_fusion_debug_log()` - Extracts "Reused" models from debug.log for Fusion skip detection
- `NodeInfo` dataclass - Represents individual dbt nodes with timing and metadata
- Timing extraction from run_results.json with fallback to timing blocks

### Layout Algorithms
- **Layered**: Hierarchical layout based on dependency depth (default)
- **Graphviz**: dot/sfdp layouts (requires pygraphviz)
- **Spring**: Force-directed layout for non-DAG graphs

### Rendering Pipeline
- `render_frames()` - Generates PNG frames using matplotlib
- Multiple visualization modes: cumulative, pulse, all_vs_needed
- Color coding by model owner (derived from file paths)
- Progress bar and timestamp overlay
- **Fusion-specific features:**
  - Automatic detection of dbt Fusion vs Core from run_results.json version strings
  - Accurate skip detection via debug.log parsing (looks for "Reused" entries)
  - Massive bright green nodes (500px) for intelligently skipped models
  - Orange nodes (75px) for executed models with all-orange connection lines
  - Skip count display with efficiency percentages
  - Visual differentiation between Core and Fusion runs

### Video Encoding
- `encode_video()` - Uses ffmpeg to create MP4/MOV from frames
- Supports transparent backgrounds with MOV format

## Dependencies

Required Python packages (inferred from imports):
- `networkx` - Graph processing
- `matplotlib` - Rendering and plotting
- `pygraphviz` (optional) - Advanced graph layouts

System dependencies:
- `ffmpeg` - Video encoding (optional, for --encode flag)

## Input Requirements

The script expects standard dbt artifacts:
- `manifest.json` - dbt project metadata and dependency graph
- `run_results.json` - Execution results with timing information
- `debug.log` (optional) - For dbt Fusion runs, provides accurate skip detection via "Reused" log entries

## Command Line Arguments

Key arguments for the main script:
- `-m, --manifest` - Path to manifest.json file (required)
- `-r, --run-results` - Path to run_results.json file (required)
- `-d, --debug-log` - Path to debug.log file (optional, for Fusion skip detection)
- `-o, --outdir` - Output directory for frames and video (required)
- `--layout` - Graph layout: layered (default), dot, sfdp, spring
- `--mode` - Visualization mode: cumulative (default), pulse, all_vs_needed
- `--fps` - Frame rate for video encoding (default: 60)
- `--duration` - Total animation duration in seconds (default: 24.0)
- `--encode` - Encode to video: mp4, mov (optional)

## Output Structure

Generated output directories contain:
- `frames/` - Individual PNG frames (frame_00001.png, frame_00002.png, etc.)
- `dag_timeline_v2.mp4` - Compiled video (if --encode mp4)
- `dag_timeline_v2_alpha.mov` - Compiled video with transparency (if --encode mov)

## dbt Fusion vs Core Visualization

The tool automatically detects dbt Fusion and provides enhanced visualization:

### Visual Indicators
- **Version Display**: Shows "dbt X.X.X 🚀 FUSION" for Fusion runs, "dbt X.X.X (Core)" for Core runs
- **Fusion Skipped Models**: Models with very short execution times (<0.05s) in Fusion are highlighted in cyan
- **Skip Counter**: Real-time count of models skipped by Fusion's intelligent skipping
- **Color Coding**: 
  - Cyan nodes/edges for Fusion-skipped models
  - Standard color-by-owner for executed models

### Detection Logic
- Fusion detection based on version string patterns and commit hashes
- Intelligent skip detection identifies models with execution times < 0.05 seconds
- Automatic fallback to Core visualization for non-Fusion runs