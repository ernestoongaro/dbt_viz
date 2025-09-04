from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import List

from .parser import load_manifest_and_results
from .renderer import render_frames


def main(argv: List[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Render a dbt DAG timeline using Pillow + imageio (no graphviz/mpl)")
    ap.add_argument("-m", "--manifest", type=Path, required=True)
    ap.add_argument("-r", "--run-results", dest="run_results", type=Path, required=True)
    ap.add_argument("-d", "--debug-log", dest="debug_log", type=Path)
    ap.add_argument("-o", "--outdir", type=Path, required=True)
    ap.add_argument("--include", action="append", default=["model", "seed", "snapshot"],
                    choices=["model", "seed", "snapshot", "source"])
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--duration", type=float, default=6.0)
    ap.add_argument("--width", type=int, default=1920)
    ap.add_argument("--height", type=int, default=1080)
    ap.add_argument("--seed", type=int, default=42, help="Random seed for image placement")
    ap.add_argument("--fade-frames", type=int, help="Fade-in duration in frames (default ~0.4s)")
    ap.add_argument("--icon-scale", type=float, help="Scale icons relative to 1920 width (default=width/1920)")
    ap.add_argument("--exec-diam-min", type=int, help="Executed min diameter at 1920 width (px)")
    ap.add_argument("--exec-diam-max", type=int, help="Executed max diameter at 1920 width (px)")
    ap.add_argument("--skip-diam-min", type=int, help="Skipped min diameter at 1920 width (px)")
    ap.add_argument("--skip-diam-max", type=int, help="Skipped max diameter at 1920 width (px)")
    ap.add_argument("--size-quantum", type=int, default=4, help="Round diameters to nearest multiple (px)")
    ap.add_argument("--skip-zero-duration", action="store_true",
                    help="Treat tasks with 0s duration as skipped/reused (never highlight them)")
    ap.add_argument("--encode", action="store_true", help="Encode frames to MP4 using imageio/ffmpeg")
    ap.add_argument("--outfile", type=str, help="Optional mp4 filename (default: timeline.mp4)")
    ap.add_argument("--executed-icon", type=Path, default=Path("images/jaffle.png"), help="PNG icon for executed models")
    ap.add_argument("--skipped-icon", type=Path, default=Path("images/flower.png"), help="PNG icon for fusion-skipped models")

    args = ap.parse_args(argv)

    args.outdir.mkdir(parents=True, exist_ok=True)

    parsed = load_manifest_and_results(
        manifest_path=args.manifest,
        run_results_path=args.run_results,
        include_types=args.include,
        debug_log_path=args.debug_log,
    )

    print(f"Detected dbt {parsed.dbt_version} ({'Fusion' if parsed.is_fusion else 'Core'})")

    exec_diam_range = None
    if args.exec_diam_min is not None or args.exec_diam_max is not None:
        emi = args.exec_diam_min if args.exec_diam_min is not None else 96
        ema = args.exec_diam_max if args.exec_diam_max is not None else 384
        exec_diam_range = (emi, ema)

    skip_diam_range = None
    if args.skip_diam_min is not None or args.skip_diam_max is not None:
        smi = args.skip_diam_min if args.skip_diam_min is not None else 128
        sma = args.skip_diam_max if args.skip_diam_max is not None else 480
        skip_diam_range = (smi, sma)

    render_frames(
        nodes=parsed.nodes,
        edges=parsed.edges,
        outdir=args.outdir,
        fps=args.fps,
        duration=args.duration,
        width=args.width,
        height=args.height,
        skip_zero_duration=args.skip_zero_duration,
        executed_icon_path=args.executed_icon,
        skipped_icon_path=args.skipped_icon,
        seed=args.seed,
        fade_frames=args.fade_frames,
        icon_scale=args.icon_scale,
        exec_diam_range=exec_diam_range,
        skip_diam_range=skip_diam_range,
        size_quantum=args.size_quantum,
    )

    if args.encode:
        # Lazy import to avoid requiring imageio when not encoding
        from .video import encode_mp4

        # Build default filename with dbt type label + timestamp if not provided
        if args.outfile:
            outfile = args.outfile
        else:
            label = "fusion" if parsed.is_fusion else "core"
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            outfile = f"dbt_{label}_{ts}.mp4"

        out_path = encode_mp4(args.outdir, fps=args.fps, filename=outfile)
        print(f"Wrote video: {out_path}")


if __name__ == "__main__":
    main()
