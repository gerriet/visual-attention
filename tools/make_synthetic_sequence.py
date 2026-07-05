#!/usr/bin/env python3
"""Generate a deterministic 3-frame motion sequence for the onset feature demo.

Frame 0 is a static textured background; in frame 1 a bright textured square
appears on the left; in frame 2 it jumps to the right. The onset feature should
light up where new structure appears (and not where it vanished).

Usage: python tools/make_synthetic_sequence.py [out_dir]
Writes f00.png, f01.png, f02.png (200x200 grayscale) into out_dir
(default: data/test_images/motion_seq/).
"""
import sys
from pathlib import Path

import numpy as np
from PIL import Image

SIZE = 200
SEED = 7


def main():
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/test_images/motion_seq")
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)

    background = rng.integers(60, 120, size=(SIZE, SIZE), dtype=np.uint8)

    def frame_with_square(cx):
        f = background.copy()
        square = rng.integers(180, 255, size=(40, 40), dtype=np.uint8)
        f[80:120, cx:cx + 40] = square
        return f

    Image.fromarray(background, mode="L").save(out / "f00.png")        # no object
    Image.fromarray(frame_with_square(30), mode="L").save(out / "f01.png")   # appears left
    Image.fromarray(frame_with_square(130), mode="L").save(out / "f02.png")  # jumps right
    print(f"wrote 3 frames to {out}")


if __name__ == "__main__":
    main()
