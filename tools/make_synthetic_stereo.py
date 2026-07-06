#!/usr/bin/env python3
"""Generate a deterministic synthetic stereo pair for the stereo-feature tests.

A textured background sits at zero disparity; a textured foreground rectangle
is shifted left by a known disparity in the right image (a nearer surface).
The stereo feature should light up the foreground (large |disparity|) and leave
the background dark.

Writes left.png and right.png (256x256 grayscale) into out_dir
(default: data/test_images/stereo/).
"""
import argparse
from pathlib import Path

SIZE = 256
FG = (80, 176, 64, 176)  # foreground rect: (x0, x1, y0, y1)
FG_DISPARITY = 10        # foreground shifts left by this many px in the right image
SEED = 20040521          # deterministic


def vertical_texture(rng, h, w):
    """Texture rich in vertical edges (what the near-vertical Gabor responds to)."""
    import numpy as np
    base = rng.integers(0, 256, size=(h, w // 4), dtype=np.uint8)
    tex = np.repeat(base, 4, axis=1)[:, :w]
    # Blend in a low-contrast smooth field so flat runs still vary a little.
    xr = np.linspace(0, 6 * np.pi, w)
    ramp = (64 + 32 * np.sin(xr))[None, :].repeat(h, axis=0)
    return (0.6 * tex + 0.4 * ramp).astype(np.uint8)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("out_dir", nargs="?", type=Path, default=Path("data/test_images/stereo"),
                        help="output directory (default: %(default)s)")
    out_dir = parser.parse_args().out_dir
    # Imported after parsing so --help works without numpy/PIL installed.
    import numpy as np
    from PIL import Image
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)

    background = vertical_texture(rng, SIZE, SIZE)
    foreground = vertical_texture(rng, SIZE, SIZE)

    x0, x1, y0, y1 = FG
    left = background.copy()
    left[y0:y1, x0:x1] = foreground[y0:y1, x0:x1]

    # Right image: background unchanged (disparity 0); foreground shifted left
    # by FG_DISPARITY (nearer surface). Fill the vacated strip with background.
    right = background.copy()
    d = FG_DISPARITY
    right[y0:y1, x0 - d:x1 - d] = foreground[y0:y1, x0:x1]

    Image.fromarray(left, mode="L").save(out_dir / "left.png")
    Image.fromarray(right, mode="L").save(out_dir / "right.png")
    print(f"wrote {out_dir/'left.png'} and {out_dir/'right.png'} "
          f"(foreground rect {FG}, disparity {FG_DISPARITY}px)")


if __name__ == "__main__":
    main()
