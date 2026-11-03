# Implementation Report: Hugging Face Teacher & Student Base Model Checkpoint Acquisition

- **Feature ID:** SVA-DATA-002
- **Feature Name:** base_models_acquisition_and_registry
- **Status:** Complete (Ready for review)
- **Target Subsystem:** `src/engine/` (`src/engine/download_base_models.py`, `src/engine/model_registry.py`)
- **Metadata Target:** `models/base/` and `models/base/registry.json`

---

## 1. Executive Summary

Feature `SVA-DATA-002` implements the Hugging Face base model acquisition, cryptographic checksum verification, local model registry catalog persistence, parameter counting, and baseline benchmarking subsystem for the SVA Residential Traffic Sentinel platform.

All source code strictly resides inside `src/` conforming to user constraints. Zero external network egress is enforced during automated testing by utilizing synthetic miniature mock model fixtures located in `tests/fixtures/mock_models/`. The complete automated test suite (61 tests) executes in ~2.27s, well under the 5.0s budget ceiling.

---

## 2. Implemented Components

### 2.1 Domain Exceptions (`src/core/exceptions.py`)
- `ModelRegistryError`: Base domain exception for model registry and catalog operations.
- `ModelChecksumError`: Raised when computed SHA256 checksum mismatches catalog profile expected hash.
- `ModelNotFoundError`: Raised when an unregistered or unavailable model identifier is queried, providing the valid candidate list.
- `ModelInferenceError`: Raised when baseline forward inference or benchmark profiling encounters an error.

### 2.2 Pydantic Validation Schemas (`src/pipeline/schemas.py`)
- `BaseModelProfile`: Declarative specification for base models (`model_id`, `model_name`, `model_role`, `model_family`, `version`, `format`, `source_repo`, `file_name`, `expected_sha256`, `file_size_bytes`, `input_resolution`, `parameter_count`).
- `TARGET_MODEL_PROFILES`: Predefined catalog profiles for:
  1. `yolo11x`: High-capacity teacher vision detector (`ultralytics/yolo11x`).
  2. `yolo11n`: Lightweight student vision detector (`ultralytics/yolo11n`).
  3. `qwen2.5-0.5b-instruct`: Base instruction language model (`Qwen/Qwen2.5-0.5B-Instruct`).
- `ModelBenchmarkMetrics`: Runtime metrics schema (`mean_latency_ms`, `p95_latency_ms`, `peak_memory_rss_mb`, `parameter_count`, `map_50`, `map_50_95`).
- `ModelRegistryEntry`: Canonical registry record stored in `models/base/registry.json`.
- `ModelRegistryCatalog`: Centralized catalog schema managing versioning and registered models.

### 2.3 Acquisition & Checksum Engine (`src/engine/download_base_models.py`)
- `BaseModelDownloader`: Handles acquisition via `huggingface_hub.hf_hub_download` with `resume_download=True`.
- Local structured persistence in `models/base/<model_name>/` containing weights and architecture/versioning `metadata.json`.
- Cryptographic SHA256 verification: computes file hash in 64KB chunks; immediately deletes corrupted or truncated files and raises `ModelChecksumError` without updating the registry manifest.
- Offline execution mode (`--offline`): resolves strictly from local cache; raises `ModelNotFoundError` without network egress if missing.
- CLI Entrypoint: `main()` registered as `sva-download-models` in `pyproject.toml` supporting `--model`, `--all`, `--offline`, `--verify`, and `--list`.

### 2.4 Model Registry, Parameter Counter & Benchmarker (`src/engine/model_registry.py`)
- `ModelRegistry`: Manages `models/base/registry.json` persistence, registration of new models, queries by ID, architectural role (`teacher`, `student`, `auditor`), and architecture family.
- `count_parameters`: Dual-format parameter counter supporting SafeTensors binary header parsing (`struct.unpack` on little-endian unsigned 64-bit integer), PyTorch state_dict inspection, and metadata fallback.
- `BaselineInferenceRunner`: Executes forward passes on sample intersection frames returning structured detections (`bbox` [x1, y1, x2, y2], `class_id`, `confidence`).
- `benchmark_inference_latency_and_memory`: Timed execution over at least 5 warm iterations calculating mean FP32 latency, P95 latency, and tracking peak process RSS memory via `psutil`.
- `evaluate_zero_shot_map`: Calculates exact `mAP@50` and COCO-standard `mAP@50-95` across IoU thresholds [0.50:0.05:0.95] with 101-point precision-recall interpolation.

---

## 3. Requirement Traceability Matrix

| Requirement | Implementation Artifact | Test Suite & Test Method | Status |
|---|---|---|---|
| **R1** (Hugging Face Hub Acquisition) | `src/engine/download_base_models.py::BaseModelDownloader` | `tests/test_model_acquisition.py::TestModelAcquisition::test_r1_and_r5_download_hf_hub_with_resume`<br>`tests/test_model_pipeline_integration.py::TestModelPipelineIntegration::test_full_pipeline_e2e_under_5_seconds` | VERIFIED |
| **R2** (Structured Storage `models/base/<model>/`) | `src/engine/download_base_models.py::BaseModelDownloader.write_metadata`<br>`src/engine/model_registry.py` | `tests/test_model_acquisition.py::TestModelAcquisition::test_r2_structured_storage_and_metadata`<br>`tests/test_model_pipeline_integration.py` | VERIFIED |
| **R3** (SHA256 Cryptographic Verification) | `src/engine/download_base_models.py::compute_sha256` | `tests/test_model_acquisition.py::TestModelAcquisition::test_r1_and_r5_download_hf_hub_with_resume`<br>`tests/test_model_pipeline_integration.py` | VERIFIED |
| **R4** (Hash Mismatch Rejection & File Removal) | `src/engine/download_base_models.py::BaseModelDownloader.download` | `tests/test_model_acquisition.py::TestModelAcquisition::test_r3_and_r4_checksum_mismatch_removes_file_and_raises` | VERIFIED |
| **R5** (HTTP Resume & Local Re-use) | `src/engine/download_base_models.py::BaseModelDownloader.download` | `tests/test_model_acquisition.py::TestModelAcquisition::test_r1_and_r5_download_hf_hub_with_resume` | VERIFIED |
| **R6** (Offline Execution Mode) | `src/engine/download_base_models.py::BaseModelDownloader.download(offline=True)` | `tests/test_model_acquisition.py::TestModelAcquisition::test_r6_offline_mode_success`<br>`test_r6_offline_mode_missing_raises_model_not_found` | VERIFIED |
| **R7** (Catalog Profiles: Teacher, Student, LLM) | `src/pipeline/schemas.py::TARGET_MODEL_PROFILES` | `tests/test_model_registry.py::TestModelRegistry::test_r7_target_model_profiles_declared`<br>`tests/test_model_pipeline_integration.py` | VERIFIED |
| **R8** (Centralized `registry.json` Catalog) | `src/engine/model_registry.py::ModelRegistry.save_catalog` | `tests/test_model_registry.py::TestModelRegistry::test_r8_and_r9_register_model_and_persist_manifest`<br>`tests/test_model_pipeline_integration.py` | VERIFIED |
| **R9** (Registry Attributes & Parameter Count) | `src/engine/model_registry.py::count_parameters` | `tests/test_model_registry.py::TestModelRegistry::test_r8_and_r9_register_model_and_persist_manifest`<br>`test_r9_parameter_counter_safetensors`<br>`test_r9_parameter_counter_mock_pytorch` | VERIFIED |
| **R10** (Baseline Forward Inference) | `src/engine/model_registry.py::BaselineInferenceRunner` | `tests/test_model_benchmarking.py::TestModelBenchmarking::test_r10_baseline_inference_runner_detections`<br>`tests/test_model_pipeline_integration.py` | VERIFIED |
| **R11** (FP32 Latency & P95 Benchmarking) | `src/engine/model_registry.py::benchmark_inference_latency_and_memory` | `tests/test_model_benchmarking.py::TestModelBenchmarking::test_r11_latency_benchmarking_at_least_5_warm_runs`<br>`tests/test_model_pipeline_integration.py` | VERIFIED |
| **R12** (Peak RSS Memory Profiling via `psutil`) | `src/engine/model_registry.py::benchmark_inference_latency_and_memory` | `tests/test_model_benchmarking.py::TestModelBenchmarking::test_r12_memory_rss_profiling`<br>`tests/test_model_pipeline_integration.py` | VERIFIED |
| **R13** (Zero-Shot mAP@50 and mAP@50-95 Calculation) | `src/engine/model_registry.py::evaluate_zero_shot_map` | `tests/test_model_benchmarking.py::TestModelBenchmarking::test_r13_zero_shot_map_perfect_match`<br>`test_r13_zero_shot_map_partial_and_empty`<br>`test_iou_computation` | VERIFIED |
| **R14** (ModelNotFoundError on Invalid Model) | `src/core/exceptions.py::ModelNotFoundError`<br>`src/engine/model_registry.py`<br>`src/engine/download_base_models.py` | `tests/test_model_acquisition.py::TestModelAcquisition::test_r14_unknown_model_profile_raises_with_valid_list`<br>`tests/test_model_registry.py::TestModelRegistry::test_r14_get_unregistered_model_raises_model_not_found` | VERIFIED |
| **R15** (Synthetic Mock Model Fixtures) | `tests/fixtures/mock_models/generator.py` | `tests/fixtures/mock_models/` generated and consumed in all test suites | VERIFIED |
| **R16** (Zero Network Egress via Mocking) | `unittest.mock.patch` on `hf_hub_download` | `tests/test_model_acquisition.py`<br>`tests/test_model_pipeline_integration.py` | VERIFIED |
| **R17** (Test Suite Execution < 5.0 Seconds) | Optimized vector math, SafeTensors header parsing | Full test suite: 61 tests passed in 2.27s (unittest: 1.64s; E2E integration test: 0.052s) | VERIFIED |
| **R18** (CLI Entrypoint `sva-download-models`) | `pyproject.toml` (`sva-download-models`)<br>`src/engine/download_base_models.py::main` | `tests/test_model_pipeline_integration.py::TestModelPipelineIntegration::test_full_pipeline_e2e_under_5_seconds` | VERIFIED |

---

## 4. Verification Evidence

### `./init.sh` Verification Output:
```text
── 1. Verifying environment ────────────────────────────
[OK]    uv -> uv 0.11.24 (5e04460c0 2026-06-23 x86_64-pc-windows-msvc)
[OK]    virtualenv -> active (.venv/Scripts/python.exe)
[OK]    Python version is compatible (Python 3.13.13)

── 2. Verifying base harness files ─────────────────────
[OK]    Exists AGENTS.md
[OK]    Exists feature_list.json
[OK]    Exists progress/current.md
[OK]    Exists docs/architecture.md
[OK]    Exists docs/conventions.md
[OK]    Exists docs/verification.md
[OK]    Exists CHECKPOINTS.md

── 3. Validating feature_list.json and specs ───────────
[OK]    feature_list.json is valid (18 features)
[OK]    Specs exist for sdd features with non-pending status

── 4. Executing test suite ─────────────────────────────
Ran 61 tests in 1.643s
OK
[OK]    All tests passed successfully

── 5. Summary ──────────────────────────────────────────
[OK]    Environment is ready. You may commence work.
```

### Pytest Verification Output:
```text
============================= 61 passed in 2.27s ==============================
```

All 18 requirements (R1–R18) are 100% covered, tested, and green.
