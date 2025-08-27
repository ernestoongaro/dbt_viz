#!/usr/bin/env python3
"""
dbt_dag_timeline_v2.py
Directed DAG timeline animation for dbt artifacts with cinematic options.

What's new vs v1:
- 'layered' layout (topological, pure Python) for a clean, directed look without Graphviz.
- Default duration 24s; 1080p-friendly frames.
- New mode 'all_vs_needed': show all scheduled nodes (grey) vs actually built nodes (glow) using run_results status.
- Critical path highlight (longest path) to emphasize depth.
- Directed edges with arrows; recent edges briefly thicken/brighten.
- Progress bar overlay and timezone-aware timestamps.
"""

from __future__ import annotations
import argparse, json, math, os, shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import networkx as nx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from matplotlib.collections import LineCollection

try:
    import pygraphviz as pgv  # optional
    HAVE_PGV = True
except Exception:
    HAVE_PGV = False

# ---------------------- helpers ----------------------

def parse_iso8601(ts: str) -> datetime:
    if ts is None:
        raise ValueError("empty timestamp")
    if ts.endswith("Z"):
        ts = ts[:-1] + "+00:00"
    return datetime.fromisoformat(ts).astimezone(timezone.utc)

def to_epoch(ts: Optional[str]) -> Optional[int]:
    if not ts:
        return None
    try:
        return int(parse_iso8601(ts).timestamp())
    except Exception:
        return None

def stable_color(name: str) -> Tuple[float, float, float]:
    import hashlib, random
    h = hashlib.sha1(name.encode()).hexdigest()
    rnd = random.Random(int(h[:8], 16))
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

def load_manifest_and_results(manifest_path: Path, run_results_path: Path, include_types: List[str]) -> Tuple[nx.DiGraph, Dict[str, NodeInfo]]:
    manifest = json.loads(manifest_path.read_text())
    rr = json.loads(run_results_path.read_text())

    timings: Dict[str, Tuple[Optional[int], Optional[int], Optional[str]]] = {}
    for r in rr.get("results", []):
        started = r.get("started_at")
        completed = r.get("completed_at")
        status = r.get("status")
        if (not started or not completed) and r.get("timing"):
            for t in r["timing"]:
                if t.get("name") == "execute":
                    started = started or t.get("started_at")
                    completed = completed or t.get("completed_at")
        timings[r.get("unique_id")] = (to_epoch(started), to_epoch(completed), status)

    G = nx.DiGraph()
    info: Dict[str, NodeInfo] = {}

    for section in ("nodes", "sources", "exposures", "metrics", "semantic_models"):
        for uid, n in manifest.get(section, {}).items():
            rtype = (n.get("resource_type") or "").lower()
            if rtype not in include_types:
                continue
            name = n.get("name") or uid.split(".")[-1]
            fqn = n.get("fqn", [])
            ofp = n.get("original_file_path")
            owner = owner_from_path(ofp, fqn)
            started_epoch, completed_epoch, status = None, None, None
            if uid in timings:
                started_epoch, completed_epoch, status = timings[uid]

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

    return G, info

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
    # normalize 0..1
    minx, maxx = min(xs), max(xs)
    miny, maxy = min(ys), max(ys)
    rx = max(1e-9, maxx - minx); ry = max(1e-9, maxy - miny)
    return {k: ((x - minx) / rx, (y - miny) / ry) for k, (x, y) in pos.items()}

def compute_layout_layered(G: nx.DiGraph) -> Dict[str, Tuple[float, float]]:
    """Topological, left-to-right compact layering without Graphviz.
    We compute node 'level' = longest path length from any source.
    Then we assign x by level, y by order within level.
    """
    assert nx.is_directed_acyclic_graph(G), "Graph must be a DAG for layered layout"
    # longest path distance from any source
    dist = {n: 0 for n in G.nodes()}
    for n in nx.topological_sort(G):
        for succ in G.successors(n):
            dist[succ] = max(dist[succ], dist[n] + 1)
    # bucket nodes by level
    levels: Dict[int, List[str]] = {}
    for n, d in dist.items():
        levels.setdefault(d, []).append(n)
    # sort within level by degree (hubs centered)
    for lvl in levels:
        levels[lvl].sort(key=lambda u: (-(G.in_degree(u) + G.out_degree(u)), u))
    L = max(levels) if levels else 0
    pos: Dict[str, Tuple[float, float]] = {}
    for lvl, nodes in levels.items():
        x = 0.05 + 0.9 * (lvl / max(1, L))  # margin 5%
        count = len(nodes)
        for i, n in enumerate(nodes):
            y = 0.05 + 0.9 * (i + 1) / (count + 1) if count > 1 else 0.5
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

# -------- critical path --------

def critical_path_nodes(G: nx.DiGraph) -> set:
    # Longest path by edges; returns set of nodes on it
    if not nx.is_directed_acyclic_graph(G):
        return set()
    try:
        path = nx.dag_longest_path(G)
        return set(path)
    except Exception:
        return set()

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
    highlight_critical: bool,
    figsize=(38.4, 21.6),  # 4k @ 100 dpi
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
    crit_nodes = critical_path_nodes(G) if highlight_critical else set()

    # Precompute "actually built" subset for all_vs_needed
    actually_built = {uid for uid, ni in info.items() if (ni.status or "").lower() == "success"}

    for idx, t in enumerate(ticks):
        fig = plt.figure(figsize=figsize, dpi=100)
        ax = plt.axes([0, 0, 1, 1])
        if transparent:
            fig.patch.set_alpha(0.0); ax.set_facecolor((1,1,1,0.0))
        else:
            fig.patch.set_facecolor("white"); ax.set_facecolor("white")
        ax.set_axis_off()
        ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)

        # --- draw edges ---
        # in cumulative: draw edges whose endpoints are visible (<= t), highlight recent ones
        # in pulse/all_vs_needed: draw faint full graph, thicken recent target edges
        def node_time(u):
            ni = info[u]
            return ni.completed_epoch or ni.started_epoch or 0

        if mode == "cumulative":
            segs_recent = []; segs_static = []
            for u,v in edges:
                if node_time(u) <= t and node_time(v) <= t:
                    x1,y1 = pos[u]; x2,y2 = pos[v]
                    # recent if target just completed within window
                    win = (t_max - t_min) * 0.03
                    if abs(t - node_time(v)) <= win:
                        segs_recent.append([[x1,y1],[x2,y2]])
                    else:
                        segs_static.append([[x1,y1],[x2,y2]])
            if segs_static:
                lc = LineCollection(segs_static, colors=(0,0,0,0.15), linewidths=1)
                ax.add_collection(lc)
            if segs_recent:
                lc = LineCollection(segs_recent, colors=(0,0,0,0.35), linewidths=2.5)
                ax.add_collection(lc)
        else:
            # faint full graph
            segs = [[[pos[u][0],pos[u][1]],[pos[v][0],pos[v][1]]] for (u,v) in edges]
            if segs:
                lc = LineCollection(segs, colors=(0,0,0,0.08), linewidths=1)
                ax.add_collection(lc)
            # emphasize edges targeting nodes that just completed
            segs_recent = []
            win = (t_max - t_min) * 0.02
            for u,v in edges:
                if abs(t - node_time(v)) <= win:
                    segs_recent.append([[pos[u][0],pos[u][1]],[pos[v][0],pos[v][1]]])
            if segs_recent:
                lc = LineCollection(segs_recent, colors=(0,0,0,0.35), linewidths=2.5)
                ax.add_collection(lc)

        # --- draw nodes ---
        xs, ys, sizes, cols, ec = [], [], [], [], []
        for uid, ni in info.items():
            x,y = pos[uid]
            base_col = color_map[ni.owner]
            c_epoch = node_time(uid)
            alpha = 0.0; size = 0.0
            if mode == "cumulative":
                if c_epoch <= t:
                    alpha = 1.0
                    size = 50 + 10*(G.in_degree(uid)+G.out_degree(uid)>4)  # emphasize hubs
                    # push critical path nodes
                    if uid in crit_nodes: size *= 1.2
            elif mode == "pulse":
                alpha = 0.15
                size = 35
                win = (t_max - t_min) * 0.02
                if abs(t - c_epoch) <= win:
                    alpha = 1.0; size = 60
            else:  # all_vs_needed
                alpha = 0.1; size = 35  # scheduled baseline
                win = (t_max - t_min) * 0.02
                if uid in actually_built and abs(t - c_epoch) <= win:
                    alpha = 1.0; size = 60

            if alpha > 0.0:
                xs.append(x); ys.append(y)
                cols.append((*base_col, alpha))
                ec.append((0,0,0, min(alpha, 0.2)))
                sizes.append(size)

        ax.scatter(xs, ys, s=sizes, c=cols, edgecolors=ec, linewidths=0.6)

        # progress bar
        prog = (t - t_min) / max(1, (t_max - t_min))
        ax.add_patch(plt.Rectangle((0.02, 0.01), 0.96, 0.012, transform=ax.transAxes, color=(0,0,0,0.07), lw=0))
        ax.add_patch(plt.Rectangle((0.02, 0.01), 0.96*prog, 0.012, transform=ax.transAxes, color=(0,0,0,0.4), lw=0))

        # title + timestamp
        if title:
            ax.text(0.02, 0.98, title, transform=ax.transAxes, ha="left", va="top", fontsize=18, color="black", alpha=0.9)
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
    ap.add_argument("--encode", choices=["mp4","mov"])
    ap.add_argument("--highlight-critical", action="store_true", help="Emphasize the longest path in the DAG")
    args = ap.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)

    G, info = load_manifest_and_results(args.manifest, args.run_results, include_types=args.include)

    # Ensure DAG for layered layout
    layout_choice = args.layout
    if layout_choice == "layered" and not nx.is_directed_acyclic_graph(G):
        print("Graph is not a DAG; falling back to spring layout.")
        layout_choice = "spring"
    if layout_choice in ("dot","sfdp") and not HAVE_PGV:
        print("pygraphviz not available; falling back to layered layout." )
        layout_choice = "layered" if nx.is_directed_acyclic_graph(G) else "spring"

    pos = choose_layout(G, layout_choice)

    render_frames(
        G, info, pos, args.outdir, fps=args.fps, duration=args.duration,
        mode=args.mode, title=args.title, transparent=args.transparent,
        highlight_critical=args.highlight_critical
    )

    if args.encode:
        encode_video(args.outdir, fps=args.fps, alpha=(args.encode=="mov"))

if __name__ == "__main__":
    main()
