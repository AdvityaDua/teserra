"""Export NLLB to an ExecuTorch XNNPACK program for Android experiments."""

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
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "artifacts" / "executorch" / "nllb_xnnpack_int8.pte")
    parser.add_argument("--backend", choices=["portable", "xnnpack"], default="xnnpack")
    parser.add_argument("--quantization", choices=["none", "int8"], default="none")
    parser.add_argument("--max-seq-len", type=int, default=96)
    parser.add_argument("--max-hidden-seq-len", type=int, default=128)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    print(f"ExecuTorch backend: {args.backend}; quantization: {args.quantization}")
    print(f"Model: {args.model}; output: {args.output}")
    if args.dry_run:
        print("Dry run complete; no .pte file was generated.")
        return 0
    if not args.model.exists():
        raise RuntimeError(f"Model directory does not exist: {args.model}. Run download_nllb.py first.")
    try:
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("Install PyTorch and Transformers before exporting ExecuTorch.") from exc
    try:
        from optimum.exporters.executorch import main_export
    except ImportError as exc:
        raise RuntimeError("Install ExecuTorch and optimum-executorch before exporting.") from exc

    ensure_dir(args.output.parent)
    output_dir = args.output.parent
    kwargs = {
        "max_seq_len": args.max_seq_len,
        "max_hidden_seq_len": args.max_hidden_seq_len,
        "dtype": "float32",
        "local_files_only": True,
    }
    if args.quantization == "int8":
        kwargs["qlinear"] = "8w"
        kwargs["qlinear_encoder"] = "8w"
    main_export(
        model_name_or_path=str(args.model),
        task="text2text-generation",
        recipe=args.backend,
        output_dir=output_dir,
        **kwargs,
    )
    generated = sorted(output_dir.glob("*.pte"))
    if not generated:
        raise RuntimeError(f"ExecuTorch exporter completed without a .pte file in {output_dir}")
    if generated[0] != args.output:
        generated[0].replace(args.output)
    print(f"ExecuTorch export complete: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
