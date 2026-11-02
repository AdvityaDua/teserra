"""Quantize exported ONNX graphs for ONNX Runtime Mobile."""

from __future__ import annotations

import argparse
from pathlib import Path

try:
    from .common import ensure_dir
except ImportError:  # pragma: no cover
    from common import ensure_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path(__file__).parent / "artifacts" / "onnx" / "fp32")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent / "artifacts" / "onnx" / "int8-dynamic")
    parser.add_argument("--mode", choices=["dynamic"], default="dynamic", help="Static calibration will be added after representative inputs are finalized.")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    models = sorted(args.input_dir.glob("*.onnx"))
    print(f"Input graphs: {args.input_dir} ({len(models)} found)")
    print(f"Output graphs: {args.output_dir}")
    if args.dry_run:
        print("Dry run complete; no ONNX graphs were quantized.")
        return 0
    if not models:
        raise RuntimeError(f"No .onnx files found in {args.input_dir}. Run export_onnx.py first.")
    try:
        from onnxruntime.quantization import QuantType, quantize_dynamic
        import onnx
        import numpy as np
        from onnx import numpy_helper
    except ImportError as exc:
        raise RuntimeError("Install ONNX Runtime with: pip install onnxruntime") from exc
    ensure_dir(args.output_dir)
    for source in models:
        target = args.output_dir / source.name
        print(f"Quantizing {source.name} -> {target.name}")
        quantize_dynamic(str(source), str(target), weight_type=QuantType.QInt8, per_channel=True, reduce_range=False)
        # Android ONNX Runtime rejects FP16 DequantizeLinear scales. This can
        # happen when the source graph was exported from FP16 weights.
        graph = onnx.load(str(target), load_external_data=False)
        for index, initializer in enumerate(graph.graph.initializer):
            if initializer.name.endswith("_scale") and initializer.data_type == onnx.TensorProto.FLOAT16:
                graph.graph.initializer[index].CopyFrom(
                    numpy_helper.from_array(
                        numpy_helper.to_array(initializer).astype(np.float32),
                        initializer.name,
                    )
                )
        onnx.save(graph, str(target))
    print("Dynamic INT8 quantization complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
