"""Run a modern saliency model on an image, emitting the interchange format.

Usage:
    python -m attention_eval.models <model> <image> --emit-json out.json
    python -m attention_eval.models --list
"""

import argparse
import sys

from .base import MODELS, get_model, run_model


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("model", nargs="?", help="model name (see --list)")
    parser.add_argument("image", nargs="?", help="input image path")
    parser.add_argument("--emit-json", required=False, help="output result JSON path")
    parser.add_argument("--max-count", type=int, default=10)
    parser.add_argument("--min-distance", type=int, default=30)
    parser.add_argument("--threshold", type=float, default=0.2)
    parser.add_argument("--list", action="store_true", help="list available models and exit")
    args = parser.parse_args(argv)

    if args.list:
        print("Available models:")
        for name in sorted(MODELS):
            print(f"  {name}")
        return 0

    if not (args.model and args.image and args.emit_json):
        parser.error("model, image and --emit-json are required (or use --list)")

    model = get_model(args.model)
    out = run_model(model, args.image, args.emit_json,
                    max_count=args.max_count, min_distance=args.min_distance, threshold=args.threshold)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
