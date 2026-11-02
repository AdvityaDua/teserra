"""Download and fingerprint the NLLB checkpoint and tokenizer assets.

No credentials are stored by this script. Set HF_TOKEN in the environment when
the Hugging Face repository requires authentication.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

try:
    from .common import DEFAULT_MODEL_ID, DEFAULT_SOURCE_DIR, ensure_dir, environment_manifest, iter_files, sha256_file, utc_now, write_json
except ImportError:  # pragma: no cover - supports direct execution
    from common import DEFAULT_MODEL_ID, DEFAULT_SOURCE_DIR, ensure_dir, environment_manifest, iter_files, sha256_file, utc_now, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--revision", default="main", help="Pin this to a commit hash for reproducible artifacts.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--force", action="store_true", help="Allow an existing output directory to be reused.")
    parser.add_argument("--dry-run", action="store_true", help="Print the download plan without contacting Hugging Face.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    token_present = bool(os.environ.get("HF_TOKEN"))
    print(f"Model: {args.model_id}")
    print(f"Revision: {args.revision}")
    print(f"Output: {args.output_dir}")
    print(f"HF_TOKEN present: {'yes' if token_present else 'no'}")
    if args.dry_run:
        print("Dry run complete; no files were downloaded.")
        return 0

    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.force:
        raise RuntimeError(f"Output directory is not empty: {args.output_dir}. Use --force only after checking its contents.")

    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise RuntimeError("Missing dependency 'huggingface_hub'. Install with: pip install huggingface_hub") from exc

    ensure_dir(args.output_dir)
    snapshot_path = Path(snapshot_download(
        repo_id=args.model_id,
        revision=args.revision,
        local_dir=str(args.output_dir),
        local_dir_use_symlinks=False,
        token=os.environ.get("HF_TOKEN"),
        resume_download=True,
    ))
    files = []
    for path in sorted(iter_files(snapshot_path)):
        files.append({"path": str(path.relative_to(snapshot_path)), "bytes": path.stat().st_size, "sha256": sha256_file(path)})
    write_json(args.output_dir.parent / ".." / "manifests" / "source_manifest.json", {
        "model_id": args.model_id,
        "revision": args.revision,
        "downloaded_at": utc_now(),
        "license": "cc-by-nc-4.0",
        "environment": environment_manifest(),
        "files": files,
    })
    print(f"Downloaded and fingerprinted {len(files)} files.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
