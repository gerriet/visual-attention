#!/usr/bin/env python3
"""Fetch the ONNX models the `dnn-classify` processor uses (roadmap M13).

Model weights are treated like datasets (v2 convention): pointed to and
downloaded on demand, never committed to the repository. Downloads land in
`models/` (gitignored) and are skipped when already present and plausible.

Default: SqueezeNet 1.1 (~5 MB, the smoke-test classifier) + the ImageNet
class names. `--all` adds MobileNetV2 (~14 MB, the documented more-accurate
alternative).

Usage:
  tools/fetch_models.py [--dest models/] [--all] [--force]
"""

import argparse
import sys
import urllib.request
from pathlib import Path

ONNX_ZOO = "https://github.com/onnx/models/raw/main/validated/vision/classification"

# name -> (url, minimum plausible size in bytes)
CORE = {
    "squeezenet1.1.onnx": (f"{ONNX_ZOO}/squeezenet/model/squeezenet1.1-7.onnx", 4_000_000),
    "imagenet_classes.txt": (
        "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt",
        10_000,
    ),
}
EXTRA = {
    "mobilenetv2-7.onnx": (f"{ONNX_ZOO}/mobilenet/model/mobilenetv2-7.onnx", 10_000_000),
}


def fetch(name: str, url: str, min_size: int, dest: Path, force: bool) -> bool:
    """Download one file; returns True on success. Skips plausible existing files."""
    target = dest / name
    if target.exists() and target.stat().st_size >= min_size and not force:
        print(f"  {name}: present ({target.stat().st_size} bytes), skipping")
        return True
    print(f"  {name}: downloading {url}")
    try:
        with urllib.request.urlopen(url, timeout=120) as response:
            data = response.read()
    except OSError as e:
        print(f"  {name}: FAILED ({e})", file=sys.stderr)
        return False
    if len(data) < min_size:
        print(
            f"  {name}: FAILED (got {len(data)} bytes, expected >= {min_size} — "
            "an LFS pointer or error page, not the model)",
            file=sys.stderr,
        )
        return False
    target.write_bytes(data)
    print(f"  {name}: ok ({len(data)} bytes)")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0], epilog="Files land in --dest (default: models/, gitignored)."
    )
    parser.add_argument("--dest", default="models", help="destination directory (default: models/)")
    parser.add_argument("--all", action="store_true", help="also fetch the optional models (MobileNetV2)")
    parser.add_argument("--force", action="store_true", help="re-download even if present")
    args = parser.parse_args()

    dest = Path(args.dest)
    dest.mkdir(parents=True, exist_ok=True)

    wanted = dict(CORE)
    if args.all:
        wanted.update(EXTRA)

    print(f"Fetching {len(wanted)} file(s) into {dest}/")
    # Attempt every file even if one fails — a transient error on the first
    # download must not silently skip the rest.
    results = [fetch(name, url, min_size, dest, args.force) for name, (url, min_size) in wanted.items()]
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
