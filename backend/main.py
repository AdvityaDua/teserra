"""Small FastAPI service for delivering approved Tessera model bundles.

The service deliberately exposes one application endpoint:
GET /v1/models/{model_key}/download

FastAPI's /docs and /openapi.json are generated automatically. Model files are
selected from an allowlist; callers cannot turn this into an arbitrary file
server.
"""

from __future__ import annotations

import hashlib
import json
import os
import struct
import zlib
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, Header, HTTPException, Path as ApiPath, status
from fastapi.responses import FileResponse, StreamingResponse


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ARTIFACT_ROOT = PROJECT_ROOT / "model_pipeline" / "artifacts"
ARTIFACT_ROOT = Path(os.getenv("TESSERA_ARTIFACT_ROOT", str(DEFAULT_ARTIFACT_ROOT))).resolve()
BACKEND_API_KEY = os.getenv("TESSERA_BACKEND_API_KEY")


class BundleSpec:
    def __init__(self, key: str, directory: Path, files: tuple[tuple[Path, str], ...], metadata: dict[str, str], raw: bool = False):
        self.key = key
        self.directory = directory
        self.files = files
        self.metadata = metadata
        self.raw = raw


ANDROID_INT8 = BundleSpec(
    key="android-onnx-int8",
    directory=ARTIFACT_ROOT / "onnx" / "int8-dynamic",
    files=(
        (Path("encoder_model.onnx"), "onnx/encoder_model.onnx"),
        (Path("decoder_model.onnx"), "onnx/decoder_model.onnx"),
        (Path("decoder_with_past_model.onnx"), "onnx/decoder_with_past_model.onnx"),
        (Path("../../source/tokenizer.json"), "source/tokenizer.json"),
        (Path("../../source/sentencepiece.bpe.model"), "source/sentencepiece.bpe.model"),
        (Path("../../source/tokenizer_config.json"), "source/tokenizer_config.json"),
        (Path("../../source/special_tokens_map.json"), "source/special_tokens_map.json"),
        (Path("../../source/config.json"), "config.json"),
        (Path("../../source/generation_config.json"), "generation_config.json"),
    ),
    metadata={
        "runtime": "onnxruntime-mobile",
        "platform": "android",
        "quantization": "dynamic-int8",
        "model_id": "facebook/nllb-200-distilled-600M",
    },
)

WHISPER_TINY = BundleSpec(
    key="whisper-tiny",
    directory=ARTIFACT_ROOT / "whisper",
    files=((Path("ggml-tiny.bin"), "ggml-tiny.bin"),),
    metadata={
        "runtime": "whisper.cpp",
        "platform": "android",
        "format": "ggml",
        "model_id": "ggerganov/whisper.cpp:ggml-tiny.bin",
    },
    raw=True,
)

BUNDLES = {ANDROID_INT8.key: ANDROID_INT8, WHISPER_TINY.key: WHISPER_TINY}

app = FastAPI(
    title="Tessera Model Delivery API",
    version="0.1.0",
    description="Local model delivery service for Tessera mobile runtimes.",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_file_path(spec: BundleSpec, relative_path: Path) -> Path:
    path = (spec.directory / relative_path).resolve()
    allowed = {(spec.directory / source).resolve() for source, _ in spec.files}
    if path not in allowed:
        raise HTTPException(status_code=500, detail="Invalid server bundle configuration")
    return path


def check_api_key(provided: str | None) -> None:
    if BACKEND_API_KEY and provided != BACKEND_API_KEY:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API key")


def resolve_bundle(spec: BundleSpec) -> list[tuple[Path, str]]:
    missing = []
    resolved_files: list[tuple[Path, str]] = []
    for source, archive_name in spec.files:
        path = safe_file_path(spec, source)
        if not path.is_file():
            missing.append(str(path))
        else:
            resolved_files.append((path, archive_name))
    if missing:
        raise HTTPException(status_code=503, detail={"message": "Model bundle is not ready", "missing": missing})

    return resolved_files


def zip_stream(spec: BundleSpec, resolved_files: list[tuple[Path, str]]):
    manifest = {
        **spec.metadata,
        "bundle_key": spec.key,
        "format": "zip",
        "files": [],
    }
    entries: list[tuple[str, int, int, int]] = []
    for source, archive_name in resolved_files:
        manifest["files"].append({
            "path": archive_name,
            "bytes": source.stat().st_size,
            "sha256": sha256_file(source),
        })
    manifest_bytes = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode()
    stream_files = [*resolved_files, (None, "manifest.json", manifest_bytes)]
    offset = 0
    for item in stream_files:
        if len(item) == 3:
            source, archive_name, data = item
            size = len(data)
            crc = zlib.crc32(data) & 0xffffffff
        else:
            source, archive_name = item
            size = source.stat().st_size
            crc = 0
            digest = 0
            with source.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest = zlib.crc32(chunk, digest)
            crc = digest & 0xffffffff
        name = archive_name.encode()
        header = struct.pack(
            "<IHHHHHIIIHH", 0x04034B50, 45, 0, 0, 0, 0, crc, size, size, len(name), 0
        ) + name
        yield header
        if len(item) == 3:
            yield data
        else:
            with source.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    yield chunk
        entries.append((archive_name, crc, size, offset))
        offset += len(header) + size

    central = bytearray()
    for archive_name, crc, size, entry_offset in entries:
        name = archive_name.encode()
        central.extend(struct.pack(
            "<IHHHHHHIIIHHHHHII",
            0x02014B50, 20, 45, 0, 0, 0, 0, crc, size, size,
            len(name), 0, 0, 0, 0, 0, entry_offset,
        ))
        central.extend(name)
    yield bytes(central)
    yield struct.pack(
        "<IHHHHIIH", 0x06054B50, 0, 0, len(entries), len(entries), len(central), offset, 0
    )


@app.get(
    "/v1/models/{model_key}/download",
    response_class=StreamingResponse,
    summary="Download an approved mobile model bundle",
    description="Streams a ZIP containing the selected runtime artifacts without creating a disk-sized temporary archive.",
)
def download_model(
    model_key: Annotated[str, ApiPath(description="Allowlisted bundle key, for example android-onnx-int8")],
    x_api_key: Annotated[str | None, Header()] = None,
) -> StreamingResponse:
    check_api_key(x_api_key)
    spec = BUNDLES.get(model_key)
    if spec is None:
        raise HTTPException(status_code=404, detail="Unknown model bundle")
    resolved_files = resolve_bundle(spec)
    if spec.raw:
        source, archive_name = resolved_files[0]
        return FileResponse(
            source,
            media_type="application/octet-stream",
            filename=archive_name,
            headers={"X-Tessera-Model": spec.key},
        )
    return StreamingResponse(
        zip_stream(spec, resolved_files),
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="{model_key}.zip"'},
    )
