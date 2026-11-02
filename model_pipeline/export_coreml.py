"""Prepare a Core ML package from an exported NLLB decoder step."""

from __future__ import annotations

import argparse
from pathlib import Path

try:
    from .common import DEFAULT_SOURCE_DIR, ensure_dir
except ImportError:  # pragma: no cover
    from common import DEFAULT_SOURCE_DIR, ensure_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "artifacts" / "coreml" / "nllb_decoder.mlpackage")
    parser.add_argument("--compute-units", choices=["all", "cpu", "cpu-and-gpu", "cpu-and-ne", "cpu-only"], default="all")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    print(f"Core ML output: {args.output}; compute units: {args.compute_units}")
    if args.dry_run:
        print("Dry run complete; no Core ML package was generated.")
        return 0
    if not args.model.exists():
        raise RuntimeError(f"Model directory does not exist: {args.model}. Run download_nllb.py first.")
    raise RuntimeError("Core ML export requires a traced encoder/decoder graph. Run the ONNX parity step first; this guard prevents an invalid monolithic seq2seq package from being produced.")


if __name__ == "__main__":
    raise SystemExit(main())
