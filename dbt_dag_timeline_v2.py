#!/usr/bin/env python3
"""
DAG timeline animation for dbt artifacts (v2 + optional --skip-zero-duration)
"""

import argparse, json, math, os, shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import networkx as nx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import matplotlib.font_manager as fm
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from PIL import Image
import numpy as np

# Configure font and suppress all warnings since we're not rendering text
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")
warnings.filterwarnings("ignore", message=".*font.*")
plt.rcParams['font.family'] = 'DejaVu Sans'

try:
    import pygraphviz as pgv  # optional
    HAVE_PGV = True
except Exception:
    HAVE_PGV = False


# ---------------------- helpers ----------------------

def parse_iso8601(ts: str):
    if ts is None:
        raise ValueError("empty timestamp")
    if ts.endswith("Z"):
        ts = ts[:-1] + "+00:00"
    from datetime import datetime as _dt, timezone as _tz
    return _dt.fromisoformat(ts).astimezone(_tz.utc)

def to_epoch(ts: Optional[str]) -> Optional[int]:
    if not ts:
        return None
    try:
        return int(parse_iso8601(ts).timestamp())
    except Exception:
        return None

def stable_color(name: str) -> Tuple[float, float, float]:
    import hashlib, random as _r
    h = hashlib.sha1(name.encode()).hexdigest()
    rnd = _r.Random(int(h[:8], 16))
    r, g, b = rnd.random(), rnd.random(), rnd.random()
    def clamp(x): return 0.15 + 0.7 * x
    return (clamp(r), clamp(g), clamp(b))

def owner_from_path(original_file_path: Optional[str], fqn: List[str]) -> str:
    path = original_file_path or "/".join(fqn)
    parts = Path(path).as_posix().split("/")
    if "models" in parts:
        i = parts.index("models")
        if i + 1 < len(parts):
            return parts[i + 1]
    if len(fqn) >= 2:
        return fqn[1]
    return "dbt"


# ---------------------- core ----------------------

@dataclass
class NodeInfo:
    unique_id: str
    name: str
    resource_type: str
    fqn: List[str]
    original_file_path: Optional[str]
    completed_epoch: Optional[int]
    started_epoch: Optional[int]
    status: Optional[str]
    owner: str
    execution_time: float = 0.0  # For proportional bubble sizing
    dependency_count: int = 0  # For complexity-based sizing of skipped models

def detect_dbt_version(run_results_path: Path) -> Tuple[str, bool]:
    """Detect dbt version and whether it's Fusion from run_results.json"""
    rr = json.loads(run_results_path.read_text())
    version = rr.get("metadata", {}).get("dbt_version", "unknown")
    
    # Fusion detection: Look for "fusion" in version string or version >= 2025.8 (when Fusion was released)
    is_fusion = (
        "fusion" in version.lower() or 
        (version.startswith("2025.") and float(version.split("+")[0].replace("2025.", "")) >= 8.0)
    )
    return version, is_fusion

def parse_fusion_debug_log(debug_log_path: Path) -> set:
    """Parse debug.log to extract models that were 'Reused' (skipped by Fusion)"""
    reused_models = set()
    if not debug_log_path.exists():
        return reused_models
        
    try:
        with open(debug_log_path, 'r') as f:
            for line in f:
                if 'Reused' in line and 'model ' in line:
                    # Extract model name from: "Reused [  1.01s] model analytics.stg_intercom__tags"
                    parts = line.strip().split('model ')
                    if len(parts) > 1:
                        model_name = parts[1].split(' ')[0]  # Get just the model name
                        # Convert to full unique_id format: model.analytics.stg_intercom__tags
                        unique_id = f"model.{model_name}"
                        reused_models.add(unique_id)
    except Exception as e:
        print(f"Warning: Could not parse debug.log: {e}")
        
    return reused_models

def load_manifest_and_results(manifest_path: Path, run_results_path: Path, include_types: List[str], debug_log_path: Optional[Path] = None) -> Tuple[nx.DiGraph, Dict[str, NodeInfo], str, bool, set]:
    manifest = json.loads(manifest_path.read_text())
    rr = json.loads(run_results_path.read_text())
    
    # Detect dbt version and Fusion
    version, is_fusion = detect_dbt_version(run_results_path)
    
    # Parse debug.log to get Fusion reused models if available
    fusion_reused_models = set()
    if is_fusion and debug_log_path:
        fusion_reused_models = parse_fusion_debug_log(debug_log_path)
        if fusion_reused_models:
            # We'll calculate the proper efficiency after filtering models by include_types
            print(f"🚀 Fusion detected! {len(fusion_reused_models)} models intelligently skipped via debug.log.")
    timings: Dict[str, Tuple[Optional[int], Optional[int], Optional[str]]] = {}
    execution_times: Dict[str, float] = {}  # Store execution_time from run_results
    executed_models = set()  # Track which models actually executed (will be filtered later)
    
    for r in rr.get("results", []):
        started = r.get("started_at")
        completed = r.get("completed_at")
        status = r.get("status")
        unique_id = r.get("unique_id")
        execution_time = r.get("execution_time", 0)
        
        executed_models.add(unique_id)  # Record that this model executed (unfiltered)
        execution_times[unique_id] = execution_time  # Store execution time
        
        if (not started or not completed) and r.get("timing"):
            for t in r["timing"]:
                if t.get("name") == "execute":
                    started = started or t.get("started_at")
                    completed = completed or t.get("completed_at")
        timings[unique_id] = (to_epoch(started), to_epoch(completed), status)

    G = nx.DiGraph()
    info: Dict[str, NodeInfo] = {}
    all_manifest_models = set()  # Track all models that exist in manifest

    manifest_model_count = 0
    for section in ("nodes", "sources", "exposures", "metrics", "semantic_models"):
        for uid, n in manifest.get(section, {}).items():
            rtype = (n.get("resource_type") or "").lower()
            if rtype not in include_types:
                continue
            
            all_manifest_models.add(uid)  # Track all models in manifest
            manifest_model_count += 1
            
            name = n.get("name") or uid.split(".")[-1]
            fqn = n.get("fqn", [])
            ofp = n.get("original_file_path")
            owner = owner_from_path(ofp, fqn)
            
            # Calculate complexity metrics for bubble sizing
            depends_on = n.get("depends_on", {})
            dependency_count = len(depends_on.get("nodes", [])) + len(depends_on.get("macros", []))
            started_epoch, completed_epoch, status = None, None, None
            
            if uid in timings:
                started_epoch, completed_epoch, status = timings[uid]
                # For Fusion, only mark extremely fast models (< 0.01s) as skipped for visualization
                if is_fusion and uid in execution_times and execution_times[uid] < 0.01:
                    status = "fusion_skipped"  # Mark extremely fast models as skipped for visualization
            elif is_fusion:
                # Model exists in manifest but not in run_results = truly skipped
                status = "fusion_skipped"

            info[uid] = NodeInfo(
                unique_id=uid,
                name=name,
                resource_type=rtype,
                fqn=fqn,
                original_file_path=ofp,
                completed_epoch=completed_epoch,
                started_epoch=started_epoch,
                status=status,
                owner=owner,
                execution_time=execution_times.get(uid, 0.0),
                dependency_count=dependency_count,
            )
            G.add_node(uid)

    child_map = manifest.get("child_map") or {}
    if child_map:
        for parent_uid, children in child_map.items():
            if parent_uid not in info:
                continue
            for c in children:
                if c in info:
                    G.add_edge(parent_uid, c)
    else:
        parent_map = manifest.get("parent_map") or {}
        for uid, parents in parent_map.items():
            if uid not in info:
                continue
            for p in parents:
                if p in info:
                    G.add_edge(p, uid)

    # Return the fusion_reused_models from debug.log instead of recalculating
    fusion_skipped_models = fusion_reused_models
    
    return G, info, version, is_fusion, fusion_skipped_models


# -------- layouts --------

def compute_layout_graphviz(G: nx.DiGraph, prog: str) -> Dict[str, Tuple[float, float]]:
    A = nx.nx_agraph.to_agraph(G)
    A.layout(prog=prog)
    pos = {}
    xs, ys = [], []
    for n in A.nodes():
        x, y = map(float, (n.attr.get("pos") or "0,0").split(","))
        pos[n.get_name()] = (x, y)
        xs.append(x); ys.append(y)
    minx, maxx = min(xs), max(xs)
    miny, maxy = min(ys), max(ys)
    rx = max(1e-9, maxx - minx); ry = max(1e-9, maxy - miny)
    return {k: ((x - minx) / rx, (y - miny) / ry) for k, (x, y) in pos.items()}

def compute_layout_layered(G: nx.DiGraph) -> Dict[str, Tuple[float, float]]:
    assert nx.is_directed_acyclic_graph(G), "Graph must be a DAG for layered layout"
    dist = {n: 0 for n in G.nodes()}
    for n in nx.topological_sort(G):
        for succ in G.successors(n):
            dist[succ] = max(dist[succ], dist[n] + 1)
    levels: Dict[int, List[str]] = {}
    for n, d in dist.items():
        levels.setdefault(d, []).append(n)
    for lvl in levels:
        levels[lvl].sort(key=lambda u: (-(G.in_degree(u) + G.out_degree(u)), u))
    L = max(levels) if levels else 0
    pos: Dict[str, Tuple[float, float]] = {}
    for lvl, nodes in levels.items():
        x = 0.04 + 0.92 * (lvl / max(1, L))
        count = len(nodes)
        for i, n in enumerate(nodes):
            y = 0.04 + 0.92 * (i + 1) / (count + 1) if count > 1 else 0.5
            pos[n] = (x, y)
    return pos

def compute_layout_spring(G: nx.DiGraph) -> Dict[str, Tuple[float, float]]:
    k = 1.0 / math.sqrt(max(1, len(G.nodes())))
    pos = nx.spring_layout(G, k=k, seed=42)
    xs = [x for x, y in pos.values()]; ys = [y for x, y in pos.values()]
    minx, maxx = min(xs), max(xs); miny, maxy = min(ys), max(ys)
    rx = max(1e-9, maxx - minx); ry = max(1e-9, maxy - miny)
    return {k: ((x - minx) / rx, (y - miny) / ry) for k, (x, y) in pos.items()}

def compute_layout_bubble_pack(G: nx.DiGraph, info: Dict[str, 'NodeInfo']) -> Dict[str, Tuple[float, float]]:
    """Compute truly organic bubble pack layout filling entire canvas"""
    import random
    import math
    
    nodes = list(G.nodes())
    n = len(nodes)
    if n == 0:
        return {}
    
    random.seed(42)  # Reproducible randomness
    
    # Create truly random distribution across full canvas
    pos = {}
    for node in nodes:
        pos[node] = (random.random(), random.random())
    
    # Simple clustering by type to create some visual organization
    fusion_skipped = []
    executed = []
    
    for node in nodes:
        ni = info.get(node)
        if ni and (ni.status or '').lower() == "fusion_skipped":
            fusion_skipped.append(node)
        else:
            executed.append(node)
    
    # Keep completely random distribution - no artificial clustering
    # The natural story will emerge from the timing and sizing alone
    
    return pos

def choose_layout(G: nx.DiGraph, which: str, info: Dict[str, 'NodeInfo'] = None) -> Dict[str, Tuple[float, float]]:
    if which == "layered":
        return compute_layout_layered(G)
    if which == "bubble" and info:
        return compute_layout_bubble_pack(G, info)
    if which in ("dot", "sfdp") and HAVE_PGV:
        return compute_layout_graphviz(G, which)
    return compute_layout_spring(G)


# ---------------------- rendering ----------------------

@dataclass
class RenderConfig:
    """Configuration for rendering nodes and UI elements with beautiful styling"""
    # Node styling - Premium colors with perfect contrast
    executed_color: Tuple[float, float, float] = (1.0, 0.35, 0.0)  # Vibrant orange
    executed_size: int = 85  # Slightly larger for better visibility
    skipped_color: Tuple[float, float, float] = (0.0, 1.0, 0.3)  # Electric green  
    skipped_size: int = 600  # Massive for dramatic impact
    scheduled_color: Tuple[float, float, float] = (0.95, 0.95, 0.98)  # Subtle pearl white
    scheduled_size: int = 20  # Slightly larger for elegance
    scheduled_alpha: float = 0.15  # More visible but still subtle
    edge_color: Tuple[float, float, float, float] = (1.0, 0.35, 0.0, 0.7)  # Matching orange edges
    edge_width: float = 2.0  # Thicker edges for premium look
    
    # Progress bar - Elegant gradient-like effect
    progress_bg_color: Tuple[float, float, float, float] = (0.2, 0.2, 0.25, 0.4)  # Dark elegant background
    progress_fill_color: Tuple[float, float, float, float] = (1.0, 0.35, 0.0, 0.9)  # Vibrant orange
    progress_height: float = 0.025  # Thicker progress bar
    
    # Text styling - Premium typography
    version_color: Tuple[float, float, float] = (1.0, 0.35, 0.0)  # Vibrant orange
    version_prefix: str = "▼ dbt CORE"
    efficiency_color: Tuple[float, float, float] = (0.9, 0.25, 0.0)  # Rich orange
    efficiency_text: str = "▲ BUILDING ALL MODELS"
    stats_color: Tuple[float, float, float] = (0.75, 0.15, 0.0)  # Deep orange
    
    # Background and frame styling
    background_color: Tuple[float, float, float] = (1.0, 1.0, 1.0)  # Clean white background
    title_background_alpha: float = 0.85  # Subtle for clean look

class CoreRenderConfig(RenderConfig):
    """Core-specific rendering configuration"""
    pass  # Uses default RenderConfig values

class FusionRenderConfig(RenderConfig):
    """Fusion-specific rendering configuration - Stunning electric theme"""
    progress_fill_color: Tuple[float, float, float, float] = (0.0, 1.0, 0.4, 0.95)  # Electric green
    version_color: Tuple[float, float, float] = (0.0, 1.0, 0.35)  # Brilliant electric green
    version_prefix: str = "▶ dbt FUSION"
    efficiency_color: Tuple[float, float, float] = (0.0, 0.95, 0.25)  # Vibrant efficiency green
    stats_color: Tuple[float, float, float] = (0.0, 0.8, 0.15)  # Deep emerald

def create_circle_image(size: int, color: Tuple[float, float, float]) -> np.ndarray:
    """Create a colored circle image for use instead of scatter plot points"""
    # Create a square image with normalized values (0-1 for matplotlib)
    img = np.ones((size*2, size*2, 4), dtype=np.float32)  # RGBA, white background
    
    # Create circle mask
    center = size
    y, x = np.ogrid[:size*2, :size*2]
    mask = (x - center)**2 + (y - center)**2 <= size**2
    
    # Apply color to circle area (normalized 0-1)
    img[mask] = [color[0], color[1], color[2], 1.0]  # RGBA
    img[~mask] = [1.0, 1.0, 1.0, 0.0]  # Transparent outside circle
    
    return img

def load_and_resize_image(image_path: str, size: int, color_tint: Tuple[float, float, float] = None) -> np.ndarray:
    """Load SVG or PNG image and resize it with proper transparency support"""
    try:
        # Handle SVG files - try simpler approach using command line tools
        if image_path.lower().endswith('.svg'):
            import subprocess
            import tempfile
            from io import BytesIO
            
            # Try using command line tools to convert SVG to PNG
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp_png:
                try:
                    # Try using rsvg-convert - preserve native SVG aspect ratio
                    result = subprocess.run([
                        'rsvg-convert', '--width', str(size), '--keep-aspect-ratio',
                        '-o', temp_png.name, image_path
                    ], capture_output=True, timeout=10)
                    
                    if result.returncode == 0:
                        img = Image.open(temp_png.name).convert('RGBA')
                    else:
                        raise Exception("rsvg-convert failed")
                        
                except (subprocess.TimeoutExpired, subprocess.CalledProcessError, FileNotFoundError, Exception):
                    # Fallback: create a simple colored rectangle as placeholder
                    from PIL import Image as PILImage
                    color = (255, 140, 0) if 'jaffle' in image_path.lower() else (255, 105, 180)  # Orange or pink
                    img = PILImage.new('RGBA', (size, size), (*color, 255))
                
                finally:
                    import os
                    try:
                        os.unlink(temp_png.name)
                    except:
                        pass
        else:
            # Handle PNG/other raster formats
            img = Image.open(image_path).convert('RGBA')
            # Resize maintaining aspect ratio
            img.thumbnail((size, size), Image.Resampling.LANCZOS)
        
        # Convert to numpy array and normalize to 0-1 range for matplotlib
        img_array = np.array(img, dtype=np.float32) / 255.0
        
        # Apply color tint if specified (working with normalized values)
        if color_tint:
            # Simple tinting - multiply RGB channels, preserve alpha
            img_array[:,:,0] = img_array[:,:,0] * color_tint[0]
            img_array[:,:,1] = img_array[:,:,1] * color_tint[1]
            img_array[:,:,2] = img_array[:,:,2] * color_tint[2]
        
        return img_array
    except Exception as e:
        print(f"Could not load image {image_path}: {e}")
        # Fallback to circle
        return create_circle_image(size//2, color_tint or (0.5, 0.5, 0.5))

def calculate_bubble_size(node_info: 'NodeInfo', fusion_skipped: bool, executed: bool, 
                         execution_times_range: Tuple[float, float], dependency_range: Tuple[int, int]) -> int:
    """Calculate proportional bubble size based on execution time or complexity"""
    if fusion_skipped:
        # Size skipped models by their complexity (dependency count)
        min_deps, max_deps = dependency_range
        if max_deps > min_deps:
            # Scale from 400 to 2000 based on dependency complexity - MUCH BIGGER
            normalized = (node_info.dependency_count - min_deps) / (max_deps - min_deps)
            return int(400 + normalized * 1600)  # 400-2000 range
        return 800  # Default for skipped models - MUCH BIGGER
    elif executed:
        # Size executed models by their execution time
        min_time, max_time = execution_times_range
        if max_time > min_time and node_info.execution_time > 0:
            # Scale from 200 to 1500 based on execution time - MUCH BIGGER
            normalized = (node_info.execution_time - min_time) / (max_time - min_time)
            return int(200 + normalized * 1300)  # 200-1500 range
        return 400  # Default for executed models - MUCH BIGGER
    else:
        return 20  # Small for scheduled models

def render_node(config: RenderConfig, fusion_skipped: bool, executed: bool, scheduled: bool, 
                node_info: 'NodeInfo' = None, size_ranges: Tuple[Tuple[float, float], Tuple[int, int]] = None) -> Tuple[Tuple[float, float, float], int, float]:
    """Determine node color, size, and alpha based on state and config"""
    if fusion_skipped:
        size = calculate_bubble_size(node_info, True, False, *size_ranges) if node_info and size_ranges else config.skipped_size
        return config.skipped_color, size, 1.0
    elif executed:
        size = calculate_bubble_size(node_info, False, True, *size_ranges) if node_info and size_ranges else config.executed_size
        return config.executed_color, size, 1.0
    elif scheduled:
        return config.scheduled_color, config.scheduled_size, config.scheduled_alpha
    else:
        return (0.0, 0.0, 0.0), 0, 0.0  # Invisible

def render_progress_bar(ax, config: RenderConfig, progress: float):
    """Render elegant progress bar with beautiful styling"""
    # Background with rounded corners effect
    ax.add_patch(plt.Rectangle((0.02, 0.01), 0.96, config.progress_height, transform=ax.transAxes, 
                              color=config.progress_bg_color, lw=0))
    # Progress fill with slight glow effect
    ax.add_patch(plt.Rectangle((0.02, 0.01), 0.96*progress, config.progress_height, transform=ax.transAxes, 
                              color=config.progress_fill_color, lw=0))

def render_text_overlay(ax, config: RenderConfig, dbt_version: str, executed_models: int, 
                       skipped_models: int, total_models: int, is_fusion: bool):
    """Render beautiful text overlay with premium typography and styling"""
    # Elegant semi-transparent background with subtle gradient effect
    overlay_bg = plt.Rectangle((0.01, 1.02), 0.7, 0.15, transform=ax.transAxes, 
                              facecolor='white', alpha=config.title_background_alpha, zorder=10)
    ax.add_patch(overlay_bg)
    
    # Premium version text with enhanced styling
    version_text = f"{config.version_prefix} {dbt_version}"
    ax.text(0.03, 1.13, version_text, transform=ax.transAxes, ha="left", va="top", 
            fontsize=26, color=config.version_color, alpha=1.0, weight='bold', zorder=11)
    
    if is_fusion and skipped_models > 0:
        # Fusion efficiency display with dramatic styling
        efficiency = (skipped_models / max(1, skipped_models + executed_models)) * 100
        efficiency_text = f"⚡ {efficiency:.0f}% EFFICIENCY GAIN"
        ax.text(0.03, 1.09, efficiency_text, transform=ax.transAxes, ha="left", va="top", 
                fontsize=18, color=config.efficiency_color, alpha=1.0, weight='bold', zorder=11)
        stats_text = f"● Built: {executed_models:,}   ◉ Skipped: {skipped_models:,}"
    else:
        # Core work display with premium styling
        ax.text(0.03, 1.09, config.efficiency_text, transform=ax.transAxes, ha="left", va="top", 
                fontsize=18, color=config.efficiency_color, alpha=1.0, weight='bold', zorder=11)
        stats_text = f"● Built: {executed_models:,} / {total_models:,}"
    
    ax.text(0.03, 1.055, stats_text, transform=ax.transAxes, ha="left", va="top", 
            fontsize=16, color=config.stats_color, alpha=1.0, weight='bold', zorder=11)

def preload_images(executed_image_path: Optional[str], skipped_image_path: Optional[str]) -> Dict:
    """Pre-load and cache all image variations we'll need"""
    print("Pre-loading images...")
    image_cache = {}
    # Calculate dynamic sizes based on 4K canvas (3840x2160) and ~1700 nodes
    # Target range: 40-320 pixels for good visibility on 4K canvas
    common_sizes = [40, 60, 80, 100, 120, 160, 200, 240, 280, 320]
    
    # Pre-load all size variations for both image types
    for size in common_sizes:
        # Executed models - preserve original PNG transparency and colors
        if executed_image_path:
            cache_key = (size, False, executed_image_path)
            image_cache[cache_key] = load_and_resize_image(executed_image_path, size)
        
        # Skipped models - preserve original PNG transparency and colors
        if skipped_image_path:
            cache_key = (size, True, skipped_image_path)
            image_cache[cache_key] = load_and_resize_image(skipped_image_path, size)
        
        # Fallback circles for models without custom images
        cache_key = (size, False, None)
        image_cache[cache_key] = create_circle_image(size, (1.0, 0.5, 0.0))
        
        cache_key = (size, True, None)
        image_cache[cache_key] = create_circle_image(size, (0.0, 0.8, 0.0))
    
    print(f"Pre-loaded {len(image_cache)} image variations")
    return image_cache

def render_frames(
    G: nx.DiGraph,
    info: Dict[str, NodeInfo],
    pos: Dict[str, Tuple[float, float]],
    outdir: Path,
    fps: int,
    duration: float,
    mode: str,
    title: Optional[str],
    transparent: bool,
    skip_zero_duration: bool,
    dbt_version: str = "unknown",
    is_fusion: bool = False,
    fusion_skipped_models: set = None,
    executed_image_path: Optional[str] = None,
    skipped_image_path: Optional[str] = None,
):
    # Initialize beautiful render configuration
    config = FusionRenderConfig() if is_fusion else CoreRenderConfig()
    
    # Pre-load and cache all images once at startup
    image_cache = preload_images(executed_image_path, skipped_image_path)
    # Calculate dynamic sizes based on 4K canvas (3840x2160) and ~1700 nodes
    # Target range: 40-320 pixels for good visibility on 4K canvas  
    common_sizes = [40, 60, 80, 100, 120, 160, 200, 240, 280, 320]
    
    frames_dir = outdir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    times = [i.completed_epoch or i.started_epoch for i in info.values() if (i.completed_epoch or i.started_epoch)]
    if not times:
        raise SystemExit("No timestamps found in run_results.json")
    t_min, t_max = min(times), max(times)
    total_frames = int(fps * duration)
    ticks = [t_min + (t_max - t_min) * i / max(1, total_frames - 1) for i in range(total_frames)]

    owners = {i.owner for i in info.values()}
    color_map = {o: stable_color(o) for o in owners}

    edges = list(G.edges())

    def _dur(ni):
        if ni.completed_epoch and ni.started_epoch:
            return ni.completed_epoch - ni.started_epoch
        return None

    def _is_zero_duration(ni):
        dur = _dur(ni)
        return dur is not None and dur == 0
    
    def _is_fusion_skipped(ni):
        """Detect if a model was skipped by Fusion's intelligent skipping"""
        if not is_fusion:
            return False
        # True detection: model marked as fusion_skipped (in manifest but not run_results)
        return (ni.status or '').lower() == "fusion_skipped"

    if skip_zero_duration:
        actually_built = {
            uid for uid, ni in info.items()
            if (ni.status or '').lower() == "success" and (_dur(ni) is not None and _dur(ni) > 0)
        }
    else:
        actually_built = {uid for uid, ni in info.items() if (ni.status or '').lower() == "success"}

    for idx, t in enumerate(ticks):
        fig = plt.figure(figsize=(38.4, 21.6), dpi=100)  # 4K default
        ax = fig.add_axes([0, 0, 1, 1])  # Fill entire figure, no margins
        
        if transparent:
            fig.patch.set_alpha(0.0)
            ax.set_facecolor((1,1,1,0.0))
        else:
            bg_color = config.background_color
            fig.patch.set_facecolor(bg_color)
            ax.set_facecolor(bg_color)
        
        # Ensure absolutely no margins or padding and maintain aspect ratio
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_axis_off()
        ax.margins(0)
        ax.set_aspect('auto')  # Allow stretching to fill the 16:9 canvas
        fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

        def node_time(u):
            ni = info[u]
            return ni.completed_epoch or ni.started_epoch or 0

        # Add subtle arrows to show dependency flow in bubble layout
        edges = list(G.edges())
        
        # Draw dependency arrows - very subtle
        for u, v in edges:
            u_time = node_time(u)
            v_time = node_time(v)
            
            # Only show arrows if both nodes are visible at current time
            u_visible = (u_time <= t) or (u in fusion_skipped_models and t >= t_min + (t_max - t_min) * 0.25)
            v_visible = (v_time <= t) or (v in fusion_skipped_models and t >= t_min + (t_max - t_min) * 0.25)
            
            if u_visible and v_visible:
                x1, y1 = pos[u]
                x2, y2 = pos[v]
                
                # Draw very subtle dependency arrow
                ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                           arrowprops=dict(arrowstyle='->', 
                                         color=(0.6, 0.6, 0.6, 0.15), 
                                         lw=0.3,
                                         shrinkA=15, shrinkB=15))

        # Calculate ranges for proportional bubble sizing
        execution_times = [ni.execution_time for ni in info.values() if ni.execution_time > 0]
        dependency_counts = [ni.dependency_count for ni in info.values()]
        
        execution_range = (min(execution_times), max(execution_times)) if execution_times else (0.0, 0.0)
        dependency_range = (min(dependency_counts), max(dependency_counts)) if dependency_counts else (0, 0)
        size_ranges = (execution_range, dependency_range)
        
        # Beautiful nodes with proportional bubble sizing
        xs, ys, sizes, cols, ec = [], [], [], [], []
        
        for uid, ni in info.items():
            x, y = pos[uid]
            c_epoch = node_time(uid)
            fusion_skipped = _is_fusion_skipped(ni)
            
            # Determine node state with timing for skipped models
            executed = ((not skip_zero_duration or not _is_zero_duration(ni)) and c_epoch <= t)
            
            # For skipped models, show them when any of their children start building
            skipped_visible = False
            if fusion_skipped:
                # Find the earliest time any child of this skipped model starts building
                children = list(G.successors(uid))
                if children:
                    child_start_times = [node_time(child) for child in children if node_time(child) > 0]
                    if child_start_times:
                        earliest_child_time = min(child_start_times)
                        skipped_visible = t >= earliest_child_time
                    else:
                        # No children with timing data - show at 25% timeline
                        timeline_progress = (t - t_min) / max(1, (t_max - t_min))
                        skipped_visible = timeline_progress >= 0.25
                else:
                    # No children - show at 25% timeline  
                    timeline_progress = (t - t_min) / max(1, (t_max - t_min))
                    skipped_visible = timeline_progress >= 0.25
            
            scheduled = not fusion_skipped and not executed and not skipped_visible
            
            # Get node styling with proportional sizing - use skipped_visible instead of fusion_skipped for visibility
            base_col, size, alpha = render_node(config, skipped_visible, executed, scheduled, ni, size_ranges)
            
            if alpha > 0:  # Only add visible nodes
                xs.append(x); ys.append(y)
                cols.append((*base_col, alpha))
                ec.append(config.edge_color)
                sizes.append(size)
        
        # Draw nodes as custom SVG images with proper layering - flowers on top of jaffles
        if xs:
            # Separate executed and skipped models for layered rendering
            executed_models = []
            skipped_models = []
            
            for x, y, size, color, edge_color in zip(xs, ys, sizes, cols, ec):
                # Convert matplotlib size to pixel size - size is already in pixel-appropriate range (200-2000)
                # Scale down to our target range (40-320) for 4K canvas 
                raw_size = max(40, min(320, size / 6))  # Scale 200-2000 -> 33-333, clamp to 40-320
                img_size = min(common_sizes, key=lambda s: abs(s - raw_size))
                
                # Determine image type (green = skipped, orange = executed)
                is_green = color[1] > color[0]
                
                # Get pre-loaded image from cache
                if is_green and skipped_image_path:
                    cache_key = (img_size, True, skipped_image_path)
                elif not is_green and executed_image_path:
                    cache_key = (img_size, False, executed_image_path)
                elif is_green:
                    cache_key = (img_size, True, None)  # fallback circle
                else:
                    cache_key = (img_size, False, None)  # fallback circle
                
                img = image_cache.get(cache_key)
                if img is not None:
                    # Calculate extent based on actual image dimensions to preserve aspect ratio
                    img_height, img_width = img.shape[:2]
                    aspect_ratio = img_width / img_height
                    
                    # Base size in canvas coordinates
                    base_size = img_size / 2000
                    
                    if aspect_ratio > 1:  # Wider than tall
                        half_width = base_size
                        half_height = base_size / aspect_ratio
                    else:  # Taller than wide
                        half_width = base_size * aspect_ratio
                        half_height = base_size
                    
                    extent = [x - half_width, x + half_width, y - half_height, y + half_height]
                    
                    if is_green:
                        skipped_models.append((img, extent))
                    else:
                        executed_models.append((img, extent))
            
            # For true vector SVG output, we need to parse and embed the original SVG content
            import xml.etree.ElementTree as ET
            from matplotlib.patches import PathPatch
            from matplotlib.path import Path
            import numpy as np
            
            def add_svg_to_plot(svg_path, extent, zorder):
                """Add actual SVG content directly using matplotlib's SVG support"""
                try:
                    from matplotlib.offsetbox import OffsetImage, AnnotationBbox
                    
                    # Read the SVG content directly 
                    with open(svg_path, 'r', encoding='utf-8') as f:
                        svg_content = f.read()
                    
                    center_x = (extent[0] + extent[1]) / 2
                    center_y = (extent[2] + extent[3]) / 2
                    size = extent[1] - extent[0]
                    
                    # Create an annotation with the raw SVG data
                    # This should preserve vector content in the output SVG
                    from matplotlib.patches import Rectangle
                    from matplotlib import transforms
                    
                    # Create a custom artist that embeds SVG content
                    rect = Rectangle((center_x - size/2, center_y - size/2), size, size, 
                                   facecolor='none', edgecolor='none', zorder=zorder)
                    
                    # Add custom SVG data as metadata
                    rect.set_gid(f'svg_{zorder}_{center_x}_{center_y}')  # Group ID for SVG
                    rect.svg_content = svg_content  # Store SVG content
                    rect.svg_size = size
                    
                    ax.add_patch(rect)
                        
                except Exception as e:
                    print(f"Failed to add SVG {svg_path}: {e}")
                    # Fallback to simple circle if SVG processing fails
                    from matplotlib.patches import Circle
                    center_x = (extent[0] + extent[1]) / 2
                    center_y = (extent[2] + extent[3]) / 2
                    size = (extent[1] - extent[0]) / 2
                    circle = Circle((center_x, center_y), size, 
                                  facecolor='#FF8C00' if 'jaffle' in svg_path.lower() else '#FF69B4',
                                  edgecolor='#8B4513' if 'jaffle' in svg_path.lower() else '#C71585',
                                  linewidth=2, zorder=zorder, alpha=0.9)
                    ax.add_patch(circle)
            
            # Draw executed models first (bottom layer - jaffle images)
            for img, extent in executed_models:
                ax.imshow(img, extent=extent, aspect='auto', interpolation='bilinear', zorder=5)
                
            # Draw skipped models on top (top layer - flower images)
            for img, extent in skipped_models:
                ax.imshow(img, extent=extent, aspect='auto', interpolation='bilinear', zorder=10)

        # Pure bubble visualization - no text overlays, just beautiful bubbles filling the canvas

        # Save as PNG for reliable ffmpeg encoding - SVG has compatibility issues
        frame_path = outdir / "frames" / f"frame_{idx:05d}.png" 
        fig.savefig(frame_path, dpi=300, transparent=transparent)
        plt.close(fig)

    print(f"Wrote {total_frames} frames to {outdir/'frames'}")


def encode_video(outdir: Path, fps: int, alpha: bool, filename_prefix: str = "dag_timeline_v2"):
    frames_dir = outdir / "frames"
    if not frames_dir.exists():
        print("No frames directory found, skipping encode.")
        return
    
    # Check if we have SVG or PNG frames
    svg_frames = list(frames_dir.glob("frame_*.svg"))
    png_frames = list(frames_dir.glob("frame_*.png"))
    
    out_mp4 = outdir / f"{filename_prefix}.mp4"
    out_mov = outdir / f"{filename_prefix}_alpha.mov"
    
    if shutil.which("ffmpeg"):
        if svg_frames:
            # Try direct SVG encoding first
            print("Attempting direct SVG to video encoding...")
            if alpha:
                result = os.system(f"ffmpeg -y -framerate {fps} -i {frames_dir}/frame_%05d.svg -vcodec qtrle {out_mov}")
            else:
                result = os.system(f"ffmpeg -y -framerate {fps} -i {frames_dir}/frame_%05d.svg -vcodec libx264 -preset slow -crf 18 -pix_fmt yuv420p {out_mp4}")
            
            if result == 0:
                print("Direct SVG to video encoding successful!")
            else:
                print("Direct SVG encoding failed, falling back to SVG->PNG->video...")
                # Fallback to PNG conversion if direct SVG fails
                png_dir = frames_dir / "png_temp"
                png_dir.mkdir(exist_ok=True)
                
                # Convert SVG to PNG at exact 4K dimensions
                for svg_file in sorted(svg_frames):
                    png_file = png_dir / f"{svg_file.stem}.png"
                    os.system(f"rsvg-convert --width=3840 --height=2160 -o {png_file} {svg_file}")
                
                # Encode from PNG files
                if alpha:
                    os.system(f"ffmpeg -y -framerate {fps} -i {png_dir}/frame_%05d.png -vcodec qtrle {out_mov}")
                else:
                    os.system(f"ffmpeg -y -framerate {fps} -i {png_dir}/frame_%05d.png -vcodec libx264 -preset slow -crf 18 -pix_fmt yuv420p {out_mp4}")
                    
                # Clean up temporary PNG files
                import shutil as sh
                sh.rmtree(png_dir)
                print("Fallback SVG->PNG->video encoding complete!")
            
        elif png_frames:
            # Fallback PNG encoding
            if alpha:
                os.system(f"ffmpeg -y -framerate {fps} -i {frames_dir}/frame_%05d.png -vcodec qtrle {out_mov}")
            else:
                os.system(f"ffmpeg -y -framerate {fps} -i {frames_dir}/frame_%05d.png -vcodec libx264 -preset slow -crf 18 -pix_fmt yuv420p {out_mp4}")
            print("PNG to video encoding complete!")
        else:
            print("No frame files found for encoding.")
            return
    else:
        print("ffmpeg not found; skipping encode.")


# ---------------------- main ----------------------

def main():
    ap = argparse.ArgumentParser(description="Render a directed DAG animation from dbt artifacts.")
    ap.add_argument("-m","--manifest", type=Path, required=True)
    ap.add_argument("-r","--run-results", dest="run_results", type=Path, required=True)
    ap.add_argument("-d","--debug-log", dest="debug_log", type=Path, help="Optional debug.log file for Fusion skip detection")
    ap.add_argument("-o","--outdir", type=Path, required=True)
    ap.add_argument("--include", action="append", default=["model","seed","snapshot"], choices=["model","seed","snapshot","source"])
    ap.add_argument("--layout", choices=["layered","dot","sfdp","spring","bubble"], default="layered")
    ap.add_argument("--mode", choices=["cumulative"], default="cumulative")
    ap.add_argument("--fps", type=int, default=60)
    ap.add_argument("--duration", type=float, default=24.0)
    ap.add_argument("--title", default="dbt DAG timeline")
    ap.add_argument("--transparent", action="store_true")
    ap.add_argument("--encode", choices=["mp4","mov"], help="Encode after frame render (requires ffmpeg)")
    ap.add_argument("--executed-image", dest="executed_image", help="Image file for executed models (e.g., jaffle.png)")
    ap.add_argument("--skipped-image", dest="skipped_image", help="Image file for skipped models (e.g., flower.png)")
    ap.add_argument("--skip-zero-duration", action="store_true",
                    help="Treat tasks with 0s duration as skipped/reused (never highlight them)")
    args = ap.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)

    G, info, dbt_version, is_fusion, fusion_skipped_models = load_manifest_and_results(args.manifest, args.run_results, include_types=args.include, debug_log_path=args.debug_log)

    # Display detected dbt version and type
    version_type = "Fusion" if is_fusion else "Core"
    print(f"Detected dbt {dbt_version} ({version_type})")
    if is_fusion and fusion_skipped_models:
        total_relevant = len([ni for ni in info.values() if ni.resource_type in ["model", "seed", "snapshot"]])
        actually_executed = total_relevant - len(fusion_skipped_models)
        efficiency = (len(fusion_skipped_models) / total_relevant) * 100 if total_relevant > 0 else 0
        print(f"⚡ Efficiency gain: {efficiency:.1f}% of models avoided!")
        print(f"📊 {actually_executed} models executed vs {len(fusion_skipped_models)} skipped")

    layout_choice = args.layout
    if layout_choice == "layered" and not nx.is_directed_acyclic_graph(G):
        print("Graph is not a DAG; falling back to spring layout.")
        layout_choice = "spring"
    if layout_choice in ("dot","sfdp") and not HAVE_PGV:
        print("pygraphviz not available; falling back to layered layout.")
        layout_choice = "layered" if nx.is_directed_acyclic_graph(G) else "spring"
    pos = choose_layout(G, layout_choice, info)

    render_frames(
        G, info, pos, args.outdir, fps=args.fps, duration=args.duration,
        mode=args.mode, title=args.title, transparent=args.transparent,
        skip_zero_duration=args.skip_zero_duration, dbt_version=dbt_version, 
        is_fusion=is_fusion, fusion_skipped_models=fusion_skipped_models,
        executed_image_path=args.executed_image, skipped_image_path=args.skipped_image
    )

    if args.encode:
        # Generate filename with type and timestamp
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        dbt_type = "fusion" if is_fusion else "core"
        filename_prefix = f"dbt_{dbt_type}_timeline_{timestamp}"
        encode_video(args.outdir, fps=args.fps, alpha=(args.encode=="mov"), filename_prefix=filename_prefix)


if __name__ == "__main__":
    main()
