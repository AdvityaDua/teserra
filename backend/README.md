# Tessera model delivery backend

This is a small local FastAPI service for delivering approved mobile model
bundles to the Tessera app. It does not run translation inference yet.

## Install

From the project root:

```bash
uv pip install --python .venv/bin/python -r backend/requirements.txt
```

## Run

```bash
uv run --python .venv/bin/python uvicorn backend.main:app --host 0.0.0.0 --port 8080
```

The OpenAPI document is available at:

```text
http://127.0.0.1:8080/openapi.json
```

The model endpoints are:

```text
GET /v1/models/android-onnx-int8/download
GET /v1/models/whisper-tiny/download
```

The first endpoint creates a temporary ZIP containing the Android dynamic-INT8
ONNX graphs, tokenizer, and model configuration. The second returns the
multilingual Whisper Tiny `ggml-tiny.bin` file directly for the on-device
`whisper.cpp` runtime. Neither endpoint performs inference on the server.

Download the Whisper artifact before starting the service:

```bash
uv run --python .venv/bin/python python model_pipeline/download_whisper.py
```

The Flutter app downloads each model once, verifies its SHA-256 checksum, and
then runs ASR or translation locally on the Android device. `whisper-tiny` is
the multilingual model; it is intentionally not `whisper-tiny.en`.

## Optional protection

Set `TESSERA_BACKEND_API_KEY` before starting the server. When set, clients
must send the value in the `X-API-Key` header.

For a phone on the same Wi-Fi network, use the Mac's LAN address instead of
`127.0.0.1`. Do not expose this development service directly to the public
internet.
