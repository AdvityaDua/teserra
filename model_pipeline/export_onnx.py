"""Export NLLB to ONNX encoder/decoder artifacts using Optimum."""

from __future__ import annotations

import argparse
from pathlib import Path

try:
    from .common import DEFAULT_MODEL_ID, DEFAULT_SOURCE_DIR, ensure_dir
except ImportError:  # pragma: no cover
    from common import DEFAULT_MODEL_ID, DEFAULT_SOURCE_DIR, ensure_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent / "artifacts" / "onnx" / "fp32")
    parser.add_argument("--task", default="text2text-generation-with-past")
    parser.add_argument("--opset", type=int, default=None)
    parser.add_argument("--optimize", choices=["none", "O1", "O2", "O3"], default="O2")
    parser.add_argument("--dtype", choices=["fp32", "fp16"], default="fp32", help="Export weight/activation precision. FP16 is recommended when disk is constrained.")
    parser.add_argument("--no-post-process", action="store_true", help="Keep separate encoder/decoder graphs and avoid decoder merging.")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    print(f"Exporting {args.model or DEFAULT_MODEL_ID} -> {args.output_dir}")
    print(f"Task: {args.task}; dtype: {args.dtype}; optimization: {args.optimize}")
    if args.dry_run:
        print("Dry run complete; no ONNX files were generated.")
        return 0
    if not args.model.exists():
        raise RuntimeError(f"Model directory does not exist: {args.model}. Run download_nllb.py first.")
    try:
        from optimum.exporters.onnx import main_export
    except ImportError as exc:
        raise RuntimeError("Install ONNX export dependencies with: pip install -r model_pipeline/requirements.txt") from exc
    ensure_dir(args.output_dir)
    options = {
        "model_name_or_path": str(args.model),
        "output": str(args.output_dir),
        "task": args.task,
        "framework": "pt",
        "optimize": None if args.optimize == "none" else args.optimize,
        "dtype": args.dtype,
        "do_validation": True,
        "no_post_process": args.no_post_process,
    }
    if args.opset is not None:
        options["opset"] = args.opset
    main_export(**options)
    print("ONNX export complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
