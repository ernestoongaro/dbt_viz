from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import random
import math
import numpy as np
from PIL import Image, ImageDraw

from .parser import NodeInfo


@dataclass
class RenderStyle:
    bg_color: Tuple[int, int, int, int] = (255, 255, 255, 255)
    edge_color: Tuple[int, int, int, int] = (200, 200, 200, 80)
    edge_width: int = 2

    executed_color: Tuple[int, int, int, int] = (255, 89, 0, 220)    # orange
    skipped_color: Tuple[int, int, int, int] = (0, 220, 90, 220)      # green
    scheduled_color: Tuple[int, int, int, int] = (235, 235, 245, 150) # pearl

    # Node radii range in pixels
    # Deprecated (kept for backward compatibility). Prefer explicit diameter ranges.
    executed_radius_px: Tuple[int, int] = (20, 80)
    skipped_radius_px: Tuple[int, int] = (32, 120)
    scheduled_radius_px: int = 3


def compute_layout_random(
    nodes_to_place: List[Tuple[str, int]],  # (uid, radius)
    width: int,
    height: int,
    margin: float = 0.05,
    seed: int = 42,
    max_attempts: int = 60,
    padding: int = 3,
) -> Dict[str, Tuple[int, int]]:
    """Random placement for beauty. Greedy, big-first, limited collision avoidance.
    nodes_to_place: list of (uid, radius_px) sorted descending by radius is recommended.
    """
    rng = random.Random(seed)
    inner_w = int(width * (1 - 2 * margin))
    inner_h = int(height * (1 - 2 * margin))
    x0 = int(width * margin)
    y0 = int(height * margin)

    positions: Dict[str, Tuple[int, int]] = {}

    def collides(x, y, r):
        rr = r + padding
        for uid2, (ox, oy) in positions.items():
            # estimate other radius by distance to bounds not known; assume similar scale by using same list ordering
            # use conservative overlap check by treating other as same radius region
            dx = x - ox
            dy = y - oy
            if dx * dx + dy * dy < (rr + rr) * (rr + rr) // 2:  # relaxed
                return True
        return False

    for uid, r in nodes_to_place:
        placed = False
        for _ in range(max_attempts):
            x = x0 + rng.randint(r, inner_w - r)
            y = y0 + rng.randint(r, inner_h - r)
            if not collides(x, y, r):
                positions[uid] = (x, y)
                placed = True
                break
        if not placed:
            # accept overlap as last resort
            x = x0 + rng.randint(r, inner_w - r)
            y = y0 + rng.randint(r, inner_h - r)
            positions[uid] = (x, y)
    return positions


def _normalize_range(values: List[float], out_min: float, out_max: float) -> Dict[str, float]:
    arr = np.array(values, dtype=float)
    if arr.size == 0:
        return {}
    vmin, vmax = float(np.min(arr)), float(np.max(arr))
    if vmax <= vmin:
        return {"__const__": (out_min + out_max) / 2.0}
    scale = (arr - vmin) / (vmax - vmin)
    return {str(i): float(out_min + scale[i] * (out_max - out_min)) for i in range(len(values))}


def _color_tuple(rgb: Tuple[float, float, float]) -> Tuple[int, int, int, int]:
    r, g, b = rgb
    return (int(r * 255), int(g * 255), int(b * 255), 255)


def render_frames(
    nodes: Dict[str, NodeInfo],
    edges: List[Tuple[str, str]],
    outdir: Path,
    fps: int = 30,
    duration: float = 6.0,
    width: int = 1920,
    height: int = 1080,
    skip_zero_duration: bool = False,
    style: Optional[RenderStyle] = None,
    executed_icon_path: Optional[Path] = Path("images/jaffle.png"),
    skipped_icon_path: Optional[Path] = Path("images/flower.png"),
    seed: int = 42,
    fade_frames: Optional[int] = None,
    icon_scale: Optional[float] = None,
    # Visual sizing controls (diameters, in pixels, at 1920 width)
    exec_diam_range: Optional[Tuple[int, int]] = None,
    skip_diam_range: Optional[Tuple[int, int]] = None,
    size_quantum: int = 4,
    clamp_lo_hi_pct: Tuple[float, float] = (5.0, 95.0),
) -> None:
    style = style or RenderStyle()
    frames_dir = outdir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    # Time ticks
    times = [i.completed_epoch or i.started_epoch for i in nodes.values() if (i.completed_epoch or i.started_epoch)]
    if not times:
        raise SystemExit("No timestamps found in run_results.json")
    t_min, t_max = min(times), max(times)
    total_frames = int(fps * duration)
    ticks = [t_min + (t_max - t_min) * i / max(1, total_frames - 1) for i in range(total_frames)]
    if fade_frames is None:
        # ~0.4s default fade
        fade_frames = max(2, int(0.4 * fps))

    # Sizing ranges
    # Metrics
    exec_times_all = [n.execution_time for n in nodes.values() if (n.execution_time or 0) > 0]
    dep_counts_all = [n.dependency_count for n in nodes.values()]
    scale = float(icon_scale) if icon_scale is not None else max(0.5, width / 1920.0)

    # Diameter ranges (scaled by resolution)
    if exec_diam_range is None:
        exec_diam_range = (96, 384)  # at 1920 width
    if skip_diam_range is None:
        skip_diam_range = (128, 480)  # at 1920 width
    exec_diam_range = (int(exec_diam_range[0] * scale), int(exec_diam_range[1] * scale))
    skip_diam_range = (int(skip_diam_range[0] * scale), int(skip_diam_range[1] * scale))
    ex_min, ex_max = style.executed_radius_px
    sk_min, sk_max = style.skipped_radius_px

    # Normalize helpers
    # Robust normalization with percentile clamping
    def _clamp_norm(val: float, arr: List[float]) -> float:
        if not arr:
            return 0.5
        lo_pct, hi_pct = clamp_lo_hi_pct
        lo = float(np.percentile(arr, lo_pct))
        hi = float(np.percentile(arr, hi_pct))
        if hi <= lo:
            lo = min(arr)
            hi = max(arr)
            if hi <= lo:
                return 0.5
        v = max(lo, min(hi, float(val)))
        return (v - lo) / max(1e-9, (hi - lo))

    def exec_diam(n: NodeInfo) -> int:
        s = _clamp_norm(n.execution_time or 0.0, exec_times_all)
        dmin, dmax = exec_diam_range
        d = int(dmin + s * (dmax - dmin))
        if size_quantum > 1:
            d = int(round(d / size_quantum) * size_quantum)
        return max(8, d)

    def skip_diam(n: NodeInfo) -> int:
        s = _clamp_norm(n.dependency_count, dep_counts_all)
        dmin, dmax = skip_diam_range
        d = int(dmin + s * (dmax - dmin))
        if size_quantum > 1:
            d = int(round(d / size_quantum) * size_quantum)
        return max(8, d)

    def is_zero_duration(n: NodeInfo) -> bool:
        if n.started_epoch is not None and n.completed_epoch is not None:
            return (n.completed_epoch - n.started_epoch) == 0
        return False

    actually_built: set[str] = set(
        uid for uid, ni in nodes.items()
        if (ni.status or '').lower() == 'success' and (not skip_zero_duration or not is_zero_duration(ni))
    )

    # Load icons (required).
    if not executed_icon_path or not Path(executed_icon_path).exists():
        raise FileNotFoundError(f"Executed icon not found: {executed_icon_path} (place PNG at images/jaffle.png or pass --executed-icon)")
    if not skipped_icon_path or not Path(skipped_icon_path).exists():
        raise FileNotFoundError(f"Skipped icon not found: {skipped_icon_path} (place PNG at images/flower.png or pass --skipped-icon)")

    base_executed = Image.open(executed_icon_path).convert("RGBA")
    base_skipped = Image.open(skipped_icon_path).convert("RGBA")

    # Cache resized icons by pixel diameter (quantized to common sizes)
    icon_cache: Dict[Tuple[str, int], Image.Image] = {}
    # helper for consistent diameter rounding and caching

    def get_icon(kind: str, diameter: int) -> Image.Image:
        key = (kind, int(diameter))
        if key in icon_cache:
            return icon_cache[key]
        base = base_executed if kind == "executed" else base_skipped
        # preserve aspect, fit into square diameter
        img = base.copy()
        img.thumbnail((diameter, diameter), Image.Resampling.LANCZOS)
        icon_cache[key] = img
        return img

    # Pre-compute positions for visible nodes only (executed or fusion-skipped)
    visible_nodes: List[str] = []
    for uid, ni in nodes.items():
        if (ni.status or '').lower() == 'fusion_skipped':
            visible_nodes.append(uid)
        elif (ni.status or '').lower() == 'success':
            visible_nodes.append(uid)

    # Determine base radius per visible node (max of executed/skip prediction) for spacing
    allowed_diams = [16, 20, 24, 28, 32, 36, 40, 48, 56, 64, 72, 80]
    base_radii: Dict[str, int] = {}
    for uid in visible_nodes:
        ni = nodes[uid]
        if (ni.status or '').lower() == 'fusion_skipped':
            dia = skip_diam(ni)
        else:
            dia = exec_diam(ni)
        base_radii[uid] = max(4, dia // 2)

    # Place larger nodes first
    order = sorted([(uid, base_radii[uid]) for uid in visible_nodes], key=lambda t: -t[1])
    pos = compute_layout_random(order, width=width, height=height, seed=seed)

    # Determine first visible frame per node for fade-in
    def node_time(uid: str) -> int:
        ni = nodes[uid]
        return ni.completed_epoch or ni.started_epoch or 0

    first_idx: Dict[str, int] = {}
    # Executed nodes appear at their first tick >= node_time
    for uid, ni in nodes.items():
        if (ni.status or '').lower() == 'success':
            t0 = node_time(uid)
            fi = 0
            for i, tt in enumerate(ticks):
                if tt >= t0:
                    fi = i
                    break
            first_idx[uid] = fi

    # Skipped nodes get a randomized early appearance within first 20% frames
    rng = random.Random(seed)
    early = max(0, int(total_frames * 0.2))
    for uid, ni in nodes.items():
        if (ni.status or '').lower() == 'fusion_skipped':
            first_idx[uid] = rng.randint(0, early)

    # Rotation params per node (gentle oscillation)
    rot_rng = random.Random(seed ^ 0xA5A5)
    rot_params: Dict[str, Tuple[float, float, float]] = {}
    for uid in visible_nodes:
        amp = rot_rng.uniform(2.0, 9.0)  # degrees
        freq = rot_rng.uniform(0.03, 0.12)  # Hz
        phase = rot_rng.uniform(0.0, 2 * math.pi)
        rot_params[uid] = (amp, freq, phase)

    # Cache of resized base icons and rotated variants
    rotated_cache: Dict[Tuple[str, int, int], Image.Image] = {}

    # Draw loop
    for idx, t in enumerate(ticks):
        img = Image.new("RGBA", (width, height), style.bg_color)
        draw = ImageDraw.Draw(img, "RGBA")

        # No DAG edges — purely decorative random layout

        # Draw nodes
        for uid, ni in nodes.items():
            x, y = pos.get(uid, (width // 2, height // 2))
            fusion_skipped = (ni.status or '').lower() == 'fusion_skipped'
            executed = (uid in actually_built) and ((ni.completed_epoch or ni.started_epoch or 0) <= t)

            # Do not draw scheduled nodes
            if fusion_skipped:
                dia = skip_diam(ni)
                icon = get_icon("skipped", diameter=dia)
            elif executed:
                dia = exec_diam(ni)
                icon = get_icon("executed", diameter=dia)
            else:
                continue

            # Fade-in alpha based on first visible index
            fi = first_idx.get(uid, 0)
            if idx < fi:
                continue
            alpha = min(1.0, (idx - fi) / max(1, fade_frames))
            if alpha <= 0:
                continue
            # Rotation angle in degrees (gentle oscillation)
            amp, freq, phase = rot_params.get(uid, (0.0, 0.0, 0.0))
            theta = amp * math.sin(2 * math.pi * freq * (idx / max(1, fps)) + phase)
            theta_q = int(round(theta / 2.0) * 2)  # quantize to 2-degree steps

            # Retrieve or build rotated icon
            key = ("skipped" if fusion_skipped else "executed", icon.size[0], theta_q)
            icon_rot = rotated_cache.get(key)
            if icon_rot is None:
                if abs(theta_q) < 1:
                    icon_rot = icon
                else:
                    icon_rot = icon.rotate(theta_q, resample=Image.Resampling.BICUBIC, expand=True)
                rotated_cache[key] = icon_rot

            # Apply fade alpha to a copy to avoid mutating cache
            if alpha < 0.999:
                draw_img = icon_rot.copy()
                a = draw_img.split()[3]
                a = a.point(lambda p: int(p * alpha))
                draw_img.putalpha(a)
            else:
                draw_img = icon_rot

            w, h = draw_img.size
            img.alpha_composite(draw_img, (int(x - w / 2), int(y - h / 2)))

        frame_path = frames_dir / f"frame_{idx:05d}.png"
        img.save(frame_path, format="PNG")
