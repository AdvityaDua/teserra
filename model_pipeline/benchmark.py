"""Run a dependency-light sanity benchmark or a model-runtime benchmark."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

try:
    from .common import PIPELINE_ROOT, ensure_dir, environment_manifest, utc_now, write_json
except ImportError:  # pragma: no cover
    from common import PIPELINE_ROOT, ensure_dir, environment_manifest, utc_now, write_json


DEFAULT_SENTENCES = [
    "The fastest path is the one we can measure.",
    "Good morning, how are you today?",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", choices=["sanity", "pytorch", "onnx"], default="sanity")
    parser.add_argument("--model-dir", type=Path, default=PIPELINE_ROOT / "artifacts" / "source")
    parser.add_argument("--output", type=Path, default=PIPELINE_ROOT / "manifests" / "benchmark.json")
    return parser.parse_args()


def sanity_result() -> dict:
    start = time.perf_counter_ns()
    token_count = sum(len(sentence.split()) for sentence in DEFAULT_SENTENCES)
    elapsed_ms = (time.perf_counter_ns() - start) / 1_000_000
    return {"runtime": "sanity", "sentences": len(DEFAULT_SENTENCES), "tokens_estimated": token_count, "pipeline_ms": elapsed_ms, "status": "pass"}


def main() -> int:
    args = parse_args()
    if args.runtime == "sanity":
        result = sanity_result()
    else:
        raise RuntimeError("Runtime benchmarking will be enabled after the model artifacts are downloaded and exported.")
    payload = {"created_at": utc_now(), "environment": environment_manifest(), "result": result}
    ensure_dir(args.output.parent)
    write_json(args.output, payload)
    print(json.dumps(result, indent=2))
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
