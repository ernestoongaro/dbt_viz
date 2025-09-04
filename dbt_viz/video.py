from __future__ import annotations

from pathlib import Path
from typing import Optional

import time
import numpy as np
import imageio.v2 as imageio
from PIL import Image


def encode_mp4(outdir: Path, fps: int = 30, filename: Optional[str] = None) -> Path:
    frames_dir = outdir / "frames"
    if not frames_dir.exists():
        raise FileNotFoundError("frames directory does not exist; render frames first")

    frame_paths = sorted(frames_dir.glob("frame_*.png"))
    if not frame_paths:
        raise FileNotFoundError("no frames found to encode")

    out = outdir / (filename or "timeline.mp4")

    # Determine target size from first frame
    with Image.open(frame_paths[0]) as im0:
        target_w, target_h = im0.size

    writer = imageio.get_writer(out, fps=fps, codec="libx264", macro_block_size=1)
    try:
        start = time.time()
        last = start
        for i, p in enumerate(frame_paths):
            with Image.open(p) as im:
                if im.size != (target_w, target_h):
                    im = im.resize((target_w, target_h), Image.Resampling.LANCZOS)
                # Drop alpha to RGB for compatibility
                im = im.convert("RGB")
                arr = np.asarray(im)
                writer.append_data(arr)

            now = time.time()
            if (now - last) >= 5 or i == 0 or i == (len(frame_paths) - 1):
                pct = 100.0 * (i + 1) / len(frame_paths)
                elapsed = now - start
                fps_eff = (i + 1) / elapsed if elapsed > 0 else 0.0
                remaining = len(frame_paths) - (i + 1)
                eta = remaining / fps_eff if fps_eff > 0 else 0.0
                print(f"Encoding video: {i + 1}/{len(frame_paths)} ({pct:.1f}%) — elapsed {elapsed:.1f}s, ETA {eta:.1f}s", flush=True)
                last = now
    finally:
        writer.close()

    return out
