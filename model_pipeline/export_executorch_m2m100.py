"""Custom ExecuTorch export for NLLB's M2M100 encoder-decoder architecture."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from transformers import AutoModelForSeq2SeqLM

from optimum.exporters.executorch.integrations import (
    Seq2SeqLMDecoderExportableModuleWithStaticCache,
    Seq2SeqLMEncoderExportableModule,
    Seq2SeqLMExportableModule,
)
from optimum.exporters.executorch.recipes.xnnpack import export_to_executorch_with_xnnpack


class M2M100ExportableModule(Seq2SeqLMExportableModule):
    """ExecuTorch's static-cache wrapper with M2M100 shape rules enabled."""

    def _export_encoder(self, encoder_input_ids):
        wrapped = Seq2SeqLMEncoderExportableModule(self.encoder).to(self.model.device).eval()
        encoder_seq_len = torch.export.Dim(
            "encoder_hidden_seq_length",
            min=1,
            max=self.max_hidden_seq_len,
        )
        with torch.no_grad():
            return torch.export.export(
                wrapped,
                (encoder_input_ids,),
                dynamic_shapes={"input_ids": {1: encoder_seq_len}},
                strict=True,
            )

    def _export_decoder(self, decoder_input_ids, encoder_hidden_states, cache_position):
        wrapped = Seq2SeqLMDecoderExportableModuleWithStaticCache(
            model=self.model,
            max_static_cache_length=self.max_seq_len,
            batch_size=self.batch_size,
        ).to(self.model.device).eval()
        encoder_seq_len = torch.export.Dim(
            "encoder_hidden_seq_length",
            min=1,
            max=self.max_hidden_seq_len,
        )
        with torch.nn.attention.sdpa_kernel([torch.nn.attention.SDPBackend.MATH]), torch.no_grad():
            return torch.export.export(
                wrapped,
                (decoder_input_ids, encoder_hidden_states, cache_position),
                dynamic_shapes={
                    "decoder_input_ids": None,
                    "encoder_hidden_states": {1: encoder_seq_len},
                    "cache_position": None,
                },
                strict=True,
            )

    def export(self, encoder_input_ids=None, decoder_input_ids=None, encoder_hidden_states=None, cache_position=None):
        if encoder_input_ids is None:
            encoder_input_ids = torch.ones((1, 10), dtype=torch.long, device=self.model.device)
        self.exported_encoder = self._export_encoder(encoder_input_ids)
        if encoder_hidden_states is None:
            encoder_hidden_states = self.exported_encoder.module()(encoder_input_ids)
        if decoder_input_ids is None:
            decoder_input_ids = torch.tensor(
                [[self.config.decoder_start_token_id]], dtype=torch.long, device=self.model.device
            )
        if cache_position is None:
            cache_position = torch.tensor([0], dtype=torch.long, device=self.model.device)
        self.exported_decoder = self._export_decoder(decoder_input_ids, encoder_hidden_states, cache_position)
        return {"encoder": self.exported_encoder, "text_decoder": self.exported_decoder}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=Path("model_pipeline/artifacts/source"))
    parser.add_argument("--output", type=Path, default=Path("model_pipeline/artifacts/executorch/nllb_m2m100_xnnpack"))
    parser.add_argument("--max-seq-len", type=int, default=96)
    parser.add_argument("--max-hidden-seq-len", type=int, default=128)
    args = parser.parse_args()

    model = AutoModelForSeq2SeqLM.from_pretrained(
        args.model,
        local_files_only=True,
    ).eval()
    if model.__class__.__name__ != "M2M100ForConditionalGeneration":
        raise RuntimeError(f"Expected M2M100ForConditionalGeneration, got {type(model).__name__}")

    wrapped = M2M100ExportableModule(
        model,
        batch_size=1,
        max_seq_len=args.max_seq_len,
        max_hidden_seq_len=args.max_hidden_seq_len,
    )
    program = export_to_executorch_with_xnnpack(wrapped)
    args.output.mkdir(parents=True, exist_ok=True)
    for name, executable in program.items():
        with (args.output / f"{name}.pte").open("wb") as handle:
            executable.write_to_file(handle)
        executable.write_tensor_data_to_file(args.output)
        print(f"Wrote {args.output / (name + '.pte')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
