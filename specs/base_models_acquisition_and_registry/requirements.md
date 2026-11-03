# Requirements: Hugging Face Teacher & Student Base Model Checkpoint Acquisition

## Feature Metadata
- **ID:** SVA-DATA-002
- **Name:** base_models_acquisition_and_registry
- **Title:** Hugging Face Teacher & Student Base Model Checkpoint Acquisition
- **Target Subsystem:** `src/engine/` (Base Model Acquisition, Checksum Verification & Registry Engine)
- **Artifact Target:** `models/base/` and `models/base/registry.json`

## Acceptance Criteria Mapping
| Acceptance Criterion | Requirements |
|---|---|
| Create scripts/download_base_models.py fetching PyTorch/SafeTensors weights with SHA256 checksum verification | R1, R2, R3, R4, R5, R6, R7, R14, R18 |
| Store base models in structured artifact directory models/base/ with versioning metadata | R2, R7, R8, R9 |
| Validate baseline inference execution and zero-shot mAP on sample intersection test frames | R10, R13, R15, R16 |
| Log baseline FP32 latency, parameter count, and memory allocation into initial registry | R9, R11, R12, R13, R17 |

---

## Requirements (Strict EARS Notation)

### R1 (Ubiquitous)
The system SHALL implement a model acquisition and registry subsystem in `src/engine/model_registry.py` and `src/engine/download_base_models.py` downloading PyTorch and SafeTensors model checkpoints from Hugging Face Hub using `huggingface_hub`.

### R2 (Ubiquitous)
The system SHALL store acquired base model checkpoints into structured subdirectories under `models/base/<model_name>/` containing model weights, architecture configuration, and versioning metadata.

### R3 (Event-driven)
WHEN model checkpoint acquisition is initiated, the system SHALL compute the cryptographic SHA256 checksum of the downloaded file and verify it against the expected SHA256 hash defined in the model catalog profile.

### R4 (Unwanted)
IF the computed SHA256 checksum of a downloaded checkpoint does not match the catalog hash, THEN the system SHALL remove the invalid file, raise a `ModelChecksumError`, and abort registration without updating the registry manifest.

### R5 (State-driven)
WHILE downloading model checkpoints via `huggingface_hub`, the system SHALL support HTTP resume capability and local cache snapshot re-use.

### R6 (Optional)
WHERE the `--offline` flag is provided or external network egress is unavailable, the system SHALL resolve model checkpoints strictly from existing local files in `models/base/<model_name>/`.

### R7 (Ubiquitous)
The system SHALL declare predefined catalog profiles for target base models:
1. Heavy teacher vision detector: `yolo11x` (or `rt-detr-x`)
2. Lightweight student vision detector: `yolo11n`
3. Base instruction-tuned language model: `qwen2.5-0.5b-instruct` (or `llama-3.2-1b-instruct`) for subsequent forensic incident reasoning.

### R8 (Ubiquitous)
The system SHALL maintain a centralized registry metadata file `models/base/registry.json` validated against a Pydantic schema `ModelRegistryCatalog` recording all registered models and their evaluation benchmarks.

### R9 (Event-driven)
WHEN a base model is registered into `models/base/registry.json`, the system SHALL record its model identifier, version, model family, format (`safetensors` or `pt`), parameter count, file size in bytes, expected SHA256, and input tensor resolution.

### R10 (Event-driven)
WHEN baseline inference execution is evaluated on sample intersection test frames, the system SHALL execute forward inference generating detections (bounding box coordinates, class IDs, confidence scores) without runtime errors.

### R11 (Event-driven)
WHEN latency benchmarking is triggered, the system SHALL measure and record mean FP32 inference latency in milliseconds computed over at least 5 warm iterations.

### R12 (Event-driven)
WHEN memory profiling is executed during baseline inference, the system SHALL measure and record peak process RSS memory allocation in megabytes.

### R13 (Event-driven)
WHEN baseline zero-shot evaluation is executed on sample annotated test frames, the system SHALL compute and record intersection mAP@50 and mAP@50-95 in the model's registry entry in `models/base/registry.json`.

### R14 (Unwanted)
IF a requested base model profile name is not found in the predefined catalog, THEN the system SHALL raise a `ModelNotFoundError` with a list of available valid models.

### R15 (Ubiquitous)
The system SHALL provide mock model fixtures in `tests/fixtures/mock_models/` containing synthetic miniature model weights and manifests for fast verification.

### R16 (Event-driven)
WHEN automated unit tests run against model acquisition, the system SHALL mock `huggingface_hub` downloads using the local synthetic fixtures with zero external network egress.

### R17 (Ubiquitous)
The automated test suite for base model acquisition, checksum verification, registry persistence, and latency benchmarking SHALL execute in under 5.0 seconds.

### R18 (Ubiquitous)
The system SHALL expose a CLI entrypoint `sva-download-models` in `pyproject.toml` supporting commands to download, verify, inspect, and benchmark base models.
