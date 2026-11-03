# Tasks: Google Colab Auditor Fine-Tuning and llama.cpp GGUF Conversion

- [x] T1 — Add `IncidentAuditVerdict`, `TrafficTelemetryRecord`, `AuditorTrainingSample`, and `GGUFConversionMetadata` Pydantic schemas in `src/pipeline/schemas.py`. Covers: R7, R8, R12.
- [x] T2 — Define `AuditorFinetuningError`, `GGUFConversionError`, `HardwareAccelerationError`, and `TelemetryGenerationError` domain exceptions in `src/core/exceptions.py`. Covers: R4, R10, R15.
- [x] T3 — Implement deterministic synthetic telemetry dataset generator in `src/pipeline/telemetry_generator.py` supporting 1,000 balanced residential traffic infraction samples with JSONL formatting. Covers: R9, R10, R11, R12.
- [x] T4 — Implement LoRA fine-tuning engine and standalone FP16 weight merger in `src/engine/auditor_finetuning.py` with QLoRA NF4 configuration, prompt formatting, and parameter tracking. Covers: R4, R5, R6, R13.
- [x] T5 — Implement llama.cpp GGUF conversion engine in `src/engine/gguf_converter.py` supporting Hugging Face to GGUF conversion, tensor validation, and metadata preservation. Covers: R14, R15, R16.
- [x] T6 — Implement forensic auditor verification engine in `src/engine/auditor_verifier.py` computing JSON parse success rate, wrong-way accuracy, and risk level concordance. Covers: R17, R18, R19, R20.
- [x] T7 — Register CLI entrypoints `sva-generate-telemetry`, `sva-finetune-auditor`, and `sva-export-gguf` in `pyproject.toml` and expose corresponding command-line handlers. Covers: R21.
- [x] T8 — Author interactive Google Colab Pro GPU notebook in `colab_notebooks/02_auditor_finetuning_llamacpp.ipynb` with Drive sync, QLoRA fine-tuning, adapter merging, llama.cpp GGUF export, and verification cells. Covers: R1, R2, R3, R4, R5, R6, R13, R14.
- [x] T9 — Implement zero-egress mock telemetry and model fixtures in `tests/fixtures/mock_telemetry.py`. Covers: R22.
- [x] T10 — Implement unit tests for telemetry generation and schema validation in `tests/test_auditor_telemetry.py`. Covers: R7, R8, R9, R10, R11, R12, R22.
- [x] T11 — Implement unit tests for fine-tuning harness, LoRA configuration, and adapter merging in `tests/test_auditor_finetuning.py`. Covers: R4, R5, R6, R13, R22.
- [x] T12 — Implement unit tests for GGUF conversion, magic headers, and missing metadata traps in `tests/test_gguf_conversion.py`. Covers: R14, R15, R16, R22.
- [x] T13 — Implement unit tests for auditor verification, structured JSON parsing, and accuracy evaluation in `tests/test_auditor_verifier.py`. Covers: R17, R18, R19, R20, R22.
- [x] T14 — Execute `./init.sh` to verify complete green status across all new and existing test suites in under 5.0 seconds. Covers: R23.
