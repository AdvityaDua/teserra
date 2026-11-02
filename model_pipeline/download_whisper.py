"""Download the multilingual Whisper Tiny GGML model for Tessera's backend."""

from __future__ import annotations

import hashlib
import os
import ssl
import urllib.request
from pathlib import Path

import certifi


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "artifacts" / "whisper" / "ggml-tiny.bin"
URL = (
    "https://huggingface.co/ggerganov/whisper.cpp/resolve/"
    "5359861c739e955e79d9a303bcbc70fb988958b1/ggml-tiny.bin"
)
SHA256 = "be07e048e1e599ad46341c8d2a135645097a538221678b7acdd1b1919c6e1b21"


def download() -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    if OUTPUT.exists() and hashlib.sha256(OUTPUT.read_bytes()).hexdigest() == SHA256:
        print(f"Whisper Tiny already verified: {OUTPUT}")
        return

    token = os.getenv("HF_TOKEN")
    request = urllib.request.Request(URL)
    if token:
        request.add_header("Authorization", f"Bearer {token}")
    partial = OUTPUT.with_suffix(".bin.part")
    print(f"Downloading multilingual Whisper Tiny -> {OUTPUT}")
    tls = ssl.create_default_context(cafile=certifi.where())
    with urllib.request.urlopen(request, context=tls) as response, partial.open("wb") as sink:
        while chunk := response.read(1024 * 1024):
            sink.write(chunk)
    actual = hashlib.sha256(partial.read_bytes()).hexdigest()
    if actual != SHA256:
        partial.unlink(missing_ok=True)
        raise RuntimeError(f"SHA-256 mismatch: expected {SHA256}, got {actual}")
    partial.replace(OUTPUT)
    print(f"Verified {OUTPUT}")


if __name__ == "__main__":
    download()
