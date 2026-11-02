"""Shared utilities used by the model pipeline command-line tools."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PIPELINE_ROOT = PROJECT_ROOT / "model_pipeline"
DEFAULT_MODEL_ID = "facebook/nllb-200-distilled-600M"
DEFAULT_SOURCE_DIR = PIPELINE_ROOT / "artifacts" / "source"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def require_optional(package: str, install_hint: str | None = None) -> Any:
    try:
        return __import__(package)
    except ImportError as exc:
        hint = install_hint or f"pip install {package}"
        raise RuntimeError(f"Missing optional dependency '{package}'. Install with: {hint}") from exc


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def command_version(command: list[str]) -> str | None:
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=False)
    except OSError:
        return None
    return (result.stdout or result.stderr).strip().splitlines()[0] if result.returncode == 0 else None


def environment_manifest() -> dict[str, Any]:
    return {
        "created_at": utc_now(),
        "python": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "packages": {
            "torch": command_version([sys.executable, "-c", "import torch; print(torch.__version__)"]),
            "transformers": command_version([sys.executable, "-c", "import transformers; print(transformers.__version__)"]),
            "onnx": command_version([sys.executable, "-c", "import onnx; print(onnx.__version__)"]),
            "onnxruntime": command_version([sys.executable, "-c", "import onnxruntime; print(onnxruntime.__version__)"]),
        },
    }


def iter_files(root: Path) -> Iterable[Path]:
    return (
        path
        for path in root.rglob("*")
        if path.is_file() and ".cache" not in path.relative_to(root).parts
    )
