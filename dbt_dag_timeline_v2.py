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

# Configure font - use system default with fallback
try:
    # Check if Poppins is available
    available_fonts = [f.name for f in fm.fontManager.ttflist]
    if 'Poppins' in available_fonts:
        plt.rcParams['font.family'] = 'Poppins'
        print("✓ Using Poppins font")
    else:
        # Use clean system fonts
        plt.rcParams['font.family'] = ['SF Pro Display', 'Helvetica', 'Arial', 'sans-serif'] 
        print(f"✓ Using system font: {plt.rcParams['font.family'][0]}")
except Exception as e:
    # Final fallback
    plt.rcParams['font.family'] = 'sans-serif'
    print("✓ Using default sans-serif font")

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

def load_manifest_and_results(manifest_path: Path, run_results_path: Path, include_types: List[str]) -> Tuple[nx.DiGraph, Dict[str, NodeInfo], str, bool, set]:
    manifest = json.loads(manifest_path.read_text())
    rr = json.loads(run_results_path.read_text())
    
    # Detect dbt version and Fusion
    version, is_fusion = detect_dbt_version(run_results_path)

    timings: Dict[str, Tuple[Optional[int], Optional[int], Optional[str]]] = {}
    executed_models = set()  # Track which models actually executed
    
    for r in rr.get("results", []):
        started = r.get("started_at")
        completed = r.get("completed_at")
        status = r.get("status")
        unique_id = r.get("unique_id")
        executed_models.add(unique_id)  # Record that this model executed
        
        if (not started or not completed) and r.get("timing"):
            for t in r["timing"]:
                if t.get("name") == "execute":
                    started = started or t.get("started_at")
                    completed = completed or t.get("completed_at")
        timings[unique_id] = (to_epoch(started), to_epoch(completed), status)

    G = nx.DiGraph()
    info: Dict[str, NodeInfo] = {}
    all_manifest_models = set()  # Track all models that exist in manifest

    for section in ("nodes", "sources", "exposures", "metrics", "semantic_models"):
        for uid, n in manifest.get(section, {}).items():
            rtype = (n.get("resource_type") or "").lower()
            if rtype not in include_types:
                continue
            
            all_manifest_models.add(uid)  # Track all models in manifest
            
            name = n.get("name") or uid.split(".")[-1]
            fqn = n.get("fqn", [])
            ofp = n.get("original_file_path")
            owner = owner_from_path(ofp, fqn)
            started_epoch, completed_epoch, status = None, None, None
            
            if uid in timings:
                started_epoch, completed_epoch, status = timings[uid]
            elif is_fusion:
                # Model exists in manifest but not in run_results = Fusion skipped it!
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

    # Calculate skipped models for return
    fusion_skipped_models = all_manifest_models - executed_models if is_fusion else set()
    
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

def choose_layout(G: nx.DiGraph, which: str) -> Dict[str, Tuple[float, float]]:
    if which == "layered":
        return compute_layout_layered(G)
    if which in ("dot", "sfdp") and HAVE_PGV:
        return compute_layout_graphviz(G, which)
    return compute_layout_spring(G)


# ---------------------- rendering ----------------------

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
):
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
        ax = plt.axes([0, 0, 1, 1])
        if transparent:
            fig.patch.set_alpha(0.0); ax.set_facecolor((1,1,1,0.0))
        else:
            # DRAMATIC BACKGROUNDS for different versions
            if is_fusion:
                # Fusion: Very subtle green tint for "efficiency"
                fig.patch.set_facecolor((0.98, 1.0, 0.98))
                ax.set_facecolor((0.98, 1.0, 0.98))
            else:
                # Core: Very subtle warm tint for "work being done"
                fig.patch.set_facecolor((1.0, 0.98, 0.96))
                ax.set_facecolor((1.0, 0.98, 0.96))
        ax.set_axis_off(); ax.set_xlim(-0.05, 1.05); ax.set_ylim(-0.05, 1.15)  # More space for text overlay

        def node_time(u):
            ni = info[u]
            return ni.completed_epoch or ni.started_epoch or 0

        # DRAMATIC EDGES - Different styles for Core vs Fusion
        segs = [[[pos[u][0],pos[u][1]],[pos[v][0],pos[v][1]]] for (u,v) in edges]
        if segs:
            if is_fusion:
                # Fusion: Lighter, more elegant edges
                lc = LineCollection(segs, colors=(0.3, 0.3, 0.3, 0.05), linewidths=0.5)
            else:
                # Core: Heavier, busier edges showing "more work"
                lc = LineCollection(segs, colors=(0.7, 0.3, 0.0, 0.12), linewidths=1.2)
            ax.add_collection(lc)
        
        # Active edges with dramatic colors
        segs_recent = []
        win = (t_max - t_min) * 0.02
        for u,v in edges:
            if abs(t - node_time(v)) <= win:
                segs_recent.append([[pos[u][0],pos[u][1]],[pos[v][0],pos[v][1]]])
        
        if segs_recent:
            if is_fusion:
                # Fusion: Bright green "efficient" edges
                lc = LineCollection(segs_recent, colors=(0.0, 0.8, 0.2, 0.6), linewidths=2.0)
            else:
                # Core: Bright orange "working hard" edges
                lc = LineCollection(segs_recent, colors=(1.0, 0.4, 0.0, 0.8), linewidths=3.0)
            ax.add_collection(lc)

        # nodes - DRAMATIC REDESIGN for Core vs Fusion contrast
        xs, ys, sizes, cols, ec = [], [], [], [], []
        ghost_xs, ghost_ys, ghost_sizes, ghost_cols, ghost_ec = [], [], [], [], []  # For ghost effects
        
        for uid, ni in info.items():
            x, y = pos[uid]
            c_epoch = node_time(uid)
            fusion_skipped = _is_fusion_skipped(ni)
            
            # Default invisible state
            alpha = 0.0; size = 0
            
            if mode == "cumulative":
                if fusion_skipped:
                    # DRAMATIC FUSION AVOIDANCE EFFECT
                    # Show skipped models as permanent ghosts that pulse slowly
                    pulse_cycle = math.sin(t * 2) * 0.1 + 0.1  # Slow ghost pulse
                    alpha = 0.15 + pulse_cycle
                    size = 20 + pulse_cycle * 10
                    base_col = (0.4, 0.4, 0.6)  # Ghostly blue-gray
                    
                    ghost_xs.append(x); ghost_ys.append(y)
                    ghost_cols.append((*base_col, alpha))
                    ghost_ec.append((0.3, 0.3, 0.5, alpha * 0.8))
                    ghost_sizes.append(size)
                    
                elif ((not skip_zero_duration or not _is_zero_duration(ni)) and c_epoch <= t):
                    # DRAMATIC BUILT MODELS - Much more prominent
                    if is_fusion:
                        # Fusion built models - Bright green "efficiency"
                        base_col = (0.1, 0.9, 0.3)  # Bright green
                        size = 80  # Larger
                        alpha = 1.0
                        edge_col = (0.0, 0.6, 0.0, 0.8)  # Dark green edge
                    else:
                        # Core models - Dramatic orange/red "work being done"
                        base_col = (1.0, 0.4, 0.0)  # Bright orange
                        size = 75  # Large
                        alpha = 1.0
                        edge_col = (0.8, 0.2, 0.0, 0.9)  # Dark orange edge
                    
                    # Add pulsing effect for active builds
                    time_since = t - c_epoch
                    if time_since <= 1.0:  # Pulse for 1 second after execution
                        pulse = 0.8 + 0.2 * math.sin(time_since * 10)  # Fast pulse
                        size = int(size * pulse)
                        alpha = min(1.0, alpha * pulse)
                    
                    xs.append(x); ys.append(y)
                    cols.append((*base_col, alpha))
                    ec.append(edge_col)
                    sizes.append(size)
                else:
                    # Scheduled nodes - very faint
                    alpha = 0.05; size = 15
                    base_col = (0.8, 0.8, 0.8)  # Very light gray
                    xs.append(x); ys.append(y)
                    cols.append((*base_col, alpha))
                    ec.append((0.6, 0.6, 0.6, 0.1))
                    sizes.append(size)
            
            # Handle other modes with similar dramatic styling...
            elif mode == "pulse":
                if abs(t - c_epoch) <= win:
                    if fusion_skipped and is_fusion:
                        # Quick ghost flash
                        base_col = (0.7, 0.7, 0.9); size = 30; alpha = 0.3
                        ghost_xs.append(x); ghost_ys.append(y)
                        ghost_cols.append((*base_col, alpha))
                        ghost_ec.append((0.5, 0.5, 0.7, alpha))
                        ghost_sizes.append(size)
                    else:
                        # Dramatic pulse for built models
                        base_col = (0.1, 0.9, 0.3) if is_fusion else (1.0, 0.4, 0.0)
                        size = 85; alpha = 1.0
                        xs.append(x); ys.append(y)
                        cols.append((*base_col, alpha))
                        ec.append((0.0, 0.6, 0.0, 0.8) if is_fusion else (0.8, 0.2, 0.0, 0.9))
                        sizes.append(size)
        
        # Draw ghost nodes first (behind regular nodes)
        if ghost_xs:
            ax.scatter(ghost_xs, ghost_ys, s=ghost_sizes, c=ghost_cols, 
                      edgecolors=ghost_ec, linewidths=0.5, alpha=0.6)
        
        # Draw regular nodes on top
        if xs:
            ax.scatter(xs, ys, s=sizes, c=cols, edgecolors=ec, linewidths=1.5)

        # DRAMATIC PROGRESS BAR
        prog = (t - t_min) / max(1, (t_max - t_min))
        if is_fusion:
            # Fusion: Green progress bar
            ax.add_patch(plt.Rectangle((0.02, 0.01), 0.96, 0.015, transform=ax.transAxes, color=(0.8, 0.8, 0.8, 0.3), lw=0))
            ax.add_patch(plt.Rectangle((0.02, 0.01), 0.96*prog, 0.015, transform=ax.transAxes, color=(0.1, 0.8, 0.3, 0.8), lw=0))
        else:
            # Core: Orange progress bar
            ax.add_patch(plt.Rectangle((0.02, 0.01), 0.96, 0.015, transform=ax.transAxes, color=(0.8, 0.8, 0.8, 0.3), lw=0))
            ax.add_patch(plt.Rectangle((0.02, 0.01), 0.96*prog, 0.015, transform=ax.transAxes, color=(1.0, 0.4, 0.0, 0.8), lw=0))

        # Add semi-transparent background for text overlay
        overlay_bg = plt.Rectangle((0.01, 1.02), 0.6, 0.12, transform=ax.transAxes, 
                                  facecolor='white', alpha=0.85, zorder=10)
        ax.add_patch(overlay_bg)
        
        # title + timestamp + version info - BIGGER AND CLEAR
        if title:
            ax.text(0.03, 1.12, title, transform=ax.transAxes, ha="left", va="top", 
                    fontsize=24, color="black", alpha=0.95, weight='bold', zorder=11)
        
        # DRAMATIC STATISTICS AND VERSION DISPLAY - BIGGER
        
        # Count models by status - REAL FUSION SKIPPING
        total_models = len(info)
        executed_models = sum(1 for ni in info.values() if node_time(ni.unique_id) <= t and (ni.status or '').lower() == "success")
        # TRUE FUSION SKIPPED: models in manifest but not in run_results
        skipped_models = len(fusion_skipped_models) if fusion_skipped_models else 0
        
        if is_fusion:
            # FUSION - Dramatic efficiency display - BIGGER
            version_text = f"▶ dbt FUSION {dbt_version}"
            ax.text(0.03, 1.08, version_text, transform=ax.transAxes, ha="left", va="top", 
                    fontsize=20, color=(0.1, 0.9, 0.3), alpha=1.0, weight='bold', zorder=11)
            
            # Efficiency metrics - make this really dramatic
            efficiency = (skipped_models / max(1, skipped_models + executed_models)) * 100
            
            # Big efficiency percentage
            if skipped_models > 0:
                efficiency_text = f"⚡ {efficiency:.0f}% EFFICIENCY GAIN"
                ax.text(0.03, 1.05, efficiency_text, transform=ax.transAxes, ha="left", va="top", 
                        fontsize=16, color=(0.0, 0.8, 0.2), alpha=0.95, weight='bold', zorder=11)
            
            # Model counts with dramatic colors
            stats_text = f"■ Built: {executed_models}   ○ Skipped: {skipped_models}"
            ax.text(0.03, 1.025, stats_text, transform=ax.transAxes, ha="left", va="top", 
                    fontsize=14, color=(0.1, 0.7, 0.1), alpha=0.9, weight='bold', zorder=11)
            
        else:
            # CORE - Show it as "working hard" - BIGGER
            version_text = f"▼ dbt CORE {dbt_version}"
            ax.text(0.03, 1.08, version_text, transform=ax.transAxes, ha="left", va="top", 
                    fontsize=20, color=(1.0, 0.4, 0.0), alpha=1.0, weight='bold', zorder=11)
            
            # Show all the work being done
            work_text = f"▲ BUILDING ALL MODELS"
            ax.text(0.03, 1.05, work_text, transform=ax.transAxes, ha="left", va="top", 
                    fontsize=16, color=(0.9, 0.3, 0.0), alpha=0.95, weight='bold', zorder=11)
            
            # Model count
            stats_text = f"■ Built: {executed_models} / {total_models}"
            ax.text(0.03, 1.025, stats_text, transform=ax.transAxes, ha="left", va="top", 
                    fontsize=14, color=(0.8, 0.2, 0.0), alpha=0.9, weight='bold', zorder=11)
        
        ax.text(0.98, 0.015, datetime.fromtimestamp(t, timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
                transform=ax.transAxes, ha="right", va="bottom", fontsize=11, color="black", alpha=0.75)

        frame_path = outdir / "frames" / f"frame_{idx:05d}.png"
        fig.savefig(frame_path, dpi=100, transparent=transparent)
        plt.close(fig)

    print(f"Wrote {total_frames} frames to {outdir/'frames'}")


def encode_video(outdir: Path, fps: int, alpha: bool):
    frames_dir = outdir / "frames"
    if not frames_dir.exists():
        print("No frames directory found, skipping encode.")
        return
    out_mp4 = outdir / "dag_timeline_v2.mp4"
    out_mov = outdir / "dag_timeline_v2_alpha.mov"
    if shutil.which("ffmpeg"):
        if alpha:
            os.system(f"ffmpeg -y -framerate {fps} -i {frames_dir}/frame_%05d.png -vcodec qtrle {out_mov}")
        else:
            os.system(f"ffmpeg -y -framerate {fps} -i {frames_dir}/frame_%05d.png -vcodec libx264 -preset slow -crf 18 -pix_fmt yuv420p {out_mp4}")
        print("Encoding complete.")
    else:
        print("ffmpeg not found; skipping encode.")


# ---------------------- main ----------------------

def main():
    ap = argparse.ArgumentParser(description="Render a directed DAG animation from dbt artifacts.")
    ap.add_argument("-m","--manifest", type=Path, required=True)
    ap.add_argument("-r","--run-results", dest="run_results", type=Path, required=True)
    ap.add_argument("-o","--outdir", type=Path, required=True)
    ap.add_argument("--include", action="append", default=["model"], choices=["model","seed","snapshot","source"])
    ap.add_argument("--layout", choices=["layered","dot","sfdp","spring"], default="layered")
    ap.add_argument("--mode", choices=["cumulative","pulse","all_vs_needed"], default="cumulative")
    ap.add_argument("--fps", type=int, default=60)
    ap.add_argument("--duration", type=float, default=24.0)
    ap.add_argument("--title", default="dbt DAG timeline")
    ap.add_argument("--transparent", action="store_true")
    ap.add_argument("--encode", choices=["mp4","mov"], help="Encode after frame render (requires ffmpeg)")
    ap.add_argument("--skip-zero-duration", action="store_true",
                    help="Treat tasks with 0s duration as skipped/reused (never highlight them)")
    args = ap.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)

    G, info, dbt_version, is_fusion, fusion_skipped_models = load_manifest_and_results(args.manifest, args.run_results, include_types=args.include)

    # Display detected dbt version and type
    version_type = "Fusion" if is_fusion else "Core"
    print(f"Detected dbt {dbt_version} ({version_type})")
    if is_fusion:
        print(f"🚀 Fusion detected! {len(fusion_skipped_models)} models intelligently skipped.")
        efficiency = (len(fusion_skipped_models) / len(info)) * 100 if info else 0
        print(f"⚡ Efficiency gain: {efficiency:.1f}% of models avoided!")

    layout_choice = args.layout
    if layout_choice == "layered" and not nx.is_directed_acyclic_graph(G):
        print("Graph is not a DAG; falling back to spring layout.")
        layout_choice = "spring"
    if layout_choice in ("dot","sfdp") and not HAVE_PGV:
        print("pygraphviz not available; falling back to layered layout.")
        layout_choice = "layered" if nx.is_directed_acyclic_graph(G) else "spring"
    pos = choose_layout(G, layout_choice)

    render_frames(
        G, info, pos, args.outdir, fps=args.fps, duration=args.duration,
        mode=args.mode, title=args.title, transparent=args.transparent,
        skip_zero_duration=args.skip_zero_duration, dbt_version=dbt_version, 
        is_fusion=is_fusion, fusion_skipped_models=fusion_skipped_models
    )

    if args.encode:
        encode_video(args.outdir, fps=args.fps, alpha=(args.encode=="mov"))


if __name__ == "__main__":
    main()
