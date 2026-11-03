# Tasks: Hugging Face Teacher & Student Base Model Checkpoint Acquisition

## Feature Metadata
- **ID:** SVA-DATA-002
- **Name:** base_models_acquisition_and_registry
- **Status:** spec_ready

## Sequential Implementation Checklist

- [x] T1 — Define domain-specific exceptions in `src/core/exceptions.py`: `ModelRegistryError`, `ModelChecksumError`, `ModelNotFoundError`, and `ModelInferenceError` inheriting from `FrameworkError`. Covers: R4, R14.
- [x] T2 — Author Pydantic validation schemas in `src/pipeline/schemas.py`: `ModelBenchmarkMetrics`, `ModelRegistryEntry`, `ModelRegistryCatalog`, and `BaseModelProfile`. Covers: R7, R8, R9.
- [x] T3 — Implement `BaseModelDownloader` in `src/engine/download_base_models.py` supporting Hugging Face Hub download, resume download, persistent local storage in `models/base/<model_name>/`, `--offline` fallback mode, and SHA256 verification. Covers: R1, R2, R3, R4, R5, R6, R7.
- [x] T4 — Implement `ModelRegistry` catalog manager in `src/engine/model_registry.py` managing registration of models, updating `models/base/registry.json`, querying models by ID/role, and loading catalog profiles. Covers: R2, R7, R8, R9, R14.
- [x] T5 — Implement baseline inference runner and parameter counter in `src/engine/model_registry.py` executing forward passes on sample intersection frames. Covers: R9, R10.
- [x] T6 — Implement latency and memory benchmarking engine in `src/engine/model_registry.py` measuring mean and P95 inference latency over warm iterations and peak process RSS memory via `psutil`. Covers: R11, R12.
- [x] T7 — Implement baseline zero-shot mAP evaluation calculation (mAP@50 and mAP@50-95) in `src/engine/model_registry.py` comparing forward inference outputs against ground truth labels. Covers: R10, R13.
- [x] T8 — Update `pyproject.toml` with CLI entrypoint `sva-download-models` pointing to `src.engine.download_base_models:main` supporting download, verify, and benchmark commands. Covers: R1, R18.
- [x] T9 — Implement mock model fixtures in `tests/fixtures/mock_models/` containing synthetic miniature model weight files and test metadata. Covers: R15, R16.
- [x] T10 — Implement unit test suite in `tests/test_model_acquisition.py` validating download logic, SHA256 verification (success and failure with `ModelChecksumError`), offline mode, and resume flag. Covers: R1, R2, R3, R4, R5, R6, R14, R16, R17.
- [x] T11 — Implement unit test suite in `tests/test_model_registry.py` validating model registration, metadata persistence in `models/base/registry.json`, Pydantic schema validation, and parameter counting. Covers: R7, R8, R9, R14, R17.
- [x] T12 — Implement unit test suite in `tests/test_model_benchmarking.py` validating latency profiling, peak RSS memory measurement, and mAP metric computation. Covers: R10, R11, R12, R13, R17.
- [x] T13 — Implement end-to-end integration test in `tests/test_model_pipeline_integration.py` executing complete workflow (mock download -> SHA256 verify -> register -> benchmark -> update `registry.json`) in < 5.0s. Covers: R1, R2, R3, R7, R8, R9, R10, R11, R12, R13, R16, R17, R18.
- [x] T14 — Execute `./init.sh` and complete test suite `pytest tests/` ensuring 100% green verification without external network egress. Covers: R1, R2, R3, R4, R5, R6, R7, R8, R9, R10, R11, R12, R13, R14, R15, R16, R17, R18.

---

## Requirement Coverage Verification Matrix

| Requirement | Covering Tasks | Description |
|---|---|---|
| R1 | T3, T8, T10, T13, T14 | Base model acquisition subsystem via `huggingface_hub` |
| R2 | T3, T4, T10, T13, T14 | Structured storage under `models/base/<model_name>/` |
| R3 | T3, T10, T13, T14 | Cryptographic SHA256 checksum verification on download |
| R4 | T1, T3, T10, T14 | Deletion and `ModelChecksumError` upon hash mismatch |
| R5 | T3, T10, T14 | HTTP range resume capability and cache re-use |
| R6 | T3, T10, T14 | Offline execution mode utilizing local weights |
| R7 | T2, T3, T4, T11, T13, T14 | Predefined catalog profiles (Teacher, Student, LLM) |
| R8 | T2, T4, T11, T13, T14 | Centralized registry metadata file `models/base/registry.json` |
| R9 | T2, T4, T5, T11, T13, T14 | Model registration attributes and parameter count |
| R10 | T5, T7, T12, T13, T14 | Baseline inference execution on sample test frames |
| R11 | T6, T12, T13, T14 | Mean FP32 latency benchmarking over warm runs |
| R12 | T6, T12, T13, T14 | Peak process RSS memory profiling via `psutil` |
| R13 | T7, T12, T13, T14 | Baseline zero-shot mAP@50 and mAP@50-95 calculation |
| R14 | T1, T4, T10, T11, T14 | `ModelNotFoundError` for unregistered model profiles |
| R15 | T9, T14 | Synthetic miniature mock model fixtures |
| R16 | T9, T10, T13, T14 | Mocking network calls with zero external egress |
| R17 | T10, T11, T12, T13, T14 | Automated test suite execution under 5.0 seconds |
| R18 | T8, T13, T14 | CLI entrypoint `sva-download-models` |
