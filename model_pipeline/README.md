# Tessera model pipeline

This directory contains the reproducible download, export, quantization, and
benchmarking tools for `facebook/nllb-200-distilled-600M`.

## Whisper Tiny

Download the multilingual Whisper Tiny GGML artifact used by the Android ASR
client:

```bash
uv run --python ../.venv/bin/python python download_whisper.py
```

The verified file is written to `artifacts/whisper/ggml-tiny.bin`. The FastAPI
model service exposes it as `whisper-tiny`; the phone downloads and runs it
locally through `whisper.cpp`.

## Safety and licensing

The model is downloaded only when explicitly requested. Credentials are read
from `HF_TOKEN` and are never written to manifests or logs. The model card
lists the checkpoint as CC-BY-NC-4.0; review that license before commercial
distribution.

## Sanity check without downloading

From the project root:

```bash
python3 -m compileall -q model_pipeline
python3 model_pipeline/download_nllb.py --dry-run
python3 model_pipeline/export_onnx.py --dry-run
python3 model_pipeline/quantize_onnx.py --dry-run
python3 model_pipeline/export_executorch.py --dry-run
python3 model_pipeline/export_coreml.py --dry-run
python3 model_pipeline/benchmark.py --runtime sanity
```

## Download later

After setting the Hugging Face token in the shell, pin `--revision` to a
specific commit before creating release artifacts:

```bash
export HF_TOKEN='...'
python3 model_pipeline/download_nllb.py --revision <commit-sha>
```

The download script writes the source checkpoint to
`artifacts/source/` and records file hashes in `manifests/source_manifest.json`.

## Export order

1. Download and fingerprint the source checkpoint.
2. Run the PyTorch baseline.
3. Export and validate ONNX encoder/decoder graphs (use `--dtype fp16` when disk is constrained).
4. Quantize ONNX dynamically to INT8.
5. Add representative calibration data for static INT8.
6. Export the decoder-step graph to ExecuTorch and add the KV-cache loop.
7. Export separate Core ML encoder/decoder packages.
8. Benchmark all runtimes on the same physical devices.

ExecuTorch and Core ML exporters intentionally stop before producing a
misleading monolithic model. They need the exported encoder/decoder contract
and device-specific backend choices finalized first.
