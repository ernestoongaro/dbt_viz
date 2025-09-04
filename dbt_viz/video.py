from __future__ import annotations

from pathlib import Path
from typing import Optional

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
        for p in frame_paths:
            with Image.open(p) as im:
                if im.size != (target_w, target_h):
                    im = im.resize((target_w, target_h), Image.Resampling.LANCZOS)
                # Drop alpha to RGB for compatibility
                im = im.convert("RGB")
                arr = np.asarray(im)
                writer.append_data(arr)
    finally:
        writer.close()

    return out
