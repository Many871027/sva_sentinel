# Requirements: Google Colab Auditor Fine-Tuning and llama.cpp GGUF Conversion

## Feature Metadata
- **ID:** SVA-DIST-002
- **Name:** colab_llamacpp_auditor_finetuning_and_export
- **Title:** Google Colab Auditor Fine-Tuning and llama.cpp GGUF Conversion
- **Target Subsystem:** `src/engine/` (Auditor Fine-Tuning & GGUF Conversion), `src/pipeline/` (Telemetry Ingestion & Generation), `colab_notebooks/` (Google Colab Pro GPU Notebook)
- **Artifact Targets:** `colab_notebooks/02_auditor_finetuning_llamacpp.ipynb`, `data/telemetry/synthetic_auditor_train.jsonl`, `models/auditor/`

## Acceptance Criteria Mapping
| Acceptance Criterion | Requirements |
|---|---|
| Create `colab_notebooks/02_auditor_finetuning_llamacpp.ipynb` utilizing Colab GPU (L4/A100) for LoRA/QLoRA fine-tuning | R1, R2, R3, R4, R5, R6 |
| Train model on 1,000 curated traffic violation telemetry prompts paired with verified structured JSON forensic verdicts | R7, R8, R9, R10, R11, R12 |
| Merge LoRA adapters and convert full model into GGUF base format using llama.cpp `convert_hf_to_gguf.py` | R13, R14, R15, R16 |
| Verify converted GGUF model produces valid structured JSON compliant with the incident schema | R17, R18, R19, R20, R21 |
| Zero-egress fast verification and test suite | R22, R23 |

---

## Requirements (Strict EARS Notation)

### R1 (Ubiquitous)
The system SHALL provide an end-to-end interactive training and conversion notebook in `colab_notebooks/02_auditor_finetuning_llamacpp.ipynb` configured for rented Google Colab GPU runtimes (NVIDIA L4 or A100 with >= 24GB VRAM).

### R2 (Event-driven)
WHEN executed in Google Colab, the notebook SHALL mount Google Drive (`/content/drive/MyDrive/sva_sentinel/`) to persist LoRA adapter weights, merged checkpoints, and GGUF artifacts across ephemeral sessions.

### R3 (Event-driven)
WHEN executed in Google Colab, the notebook SHALL install required fine-tuning dependencies (`unsloth` or `peft`, `bitsandbytes`, `transformers`, `triton`, `accelerate`) and clone the `llama.cpp` repository with build utilities.

### R4 (Unwanted)
IF GPU hardware acceleration is absent (CUDA device not detected), THEN the fine-tuning pipeline SHALL raise a `HardwareAccelerationError` and abort training execution before allocating VRAM.

### R5 (State-driven)
WHILE running LoRA/QLoRA fine-tuning, the system SHALL apply 4-bit normal float (`NF4`) quantization via `bitsandbytes` to constrain peak VRAM consumption below 12GB on the target Colab GPU.

### R6 (Ubiquitous)
The fine-tuning configuration SHALL configure target LoRA rank $r \ge 16$, alpha $\ge 32$, target modules (`q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`), and a maximum sequence length of 2048 tokens.

### R7 (Ubiquitous)
The system SHALL define Pydantic schemas in `src/pipeline/schemas.py` for structured forensic verdicts (`IncidentAuditVerdict`) and input telemetry events (`TrafficTelemetryRecord`).

### R8 (Ubiquitous)
The `IncidentAuditVerdict` schema SHALL require the following forensic fields:
1. `incident_id`: unique string identifier.
2. `timestamp`: ISO-8601 UTC timestamp string.
3. `entity_type`: one of `car`, `motorcycle`, `bicycle`, `e-scooter`, `pedestrian`.
4. `lane_id`: identifier of the occupied intersection lane.
5. `is_wrong_way`: boolean flag indicating counter-flow movement.
6. `heading_degrees`: detected movement angle relative to lane vector ($[0.0, 360.0)$).
7. `estimated_speed_kmh`: estimated ground speed in km/h ($\ge 0.0$).
8. `speed_limit_kmh`: posted residential speed limit ($\ge 0.0$).
9. `risk_level`: classification string among `low`, `medium`, `high`, `critical`.
10. `violation_detected`: boolean flag indicating whether any infraction occurred.
11. `summary`: concise narrative forensic explanation ($\le 300$ characters).
12. `reasoning`: detailed step-by-step forensic deduction.

### R9 (Ubiquitous)
The system SHALL implement a deterministic telemetry dataset generator in `src/pipeline/telemetry_generator.py` capable of synthesizing 1,000 balanced residential traffic infraction telemetry samples.

### R10 (State-driven)
WHILE generating synthetic telemetry samples, the generator SHALL synthesize realistic infraction distributions including:
- Wrong-way vehicular ingress (cars, motorcycles driving against one-way residential lanes).
- Sidewalk / wrong-way micromobility (bicycles, e-scooters traversing oncoming traffic or restricted pedestrian zones).
- Excessive speeding in residential school/intersection zones (> 30 km/h).
- Benign compliant traffic baseline control samples ($\ge 20\%$ of total records).

### R11 (Event-driven)
WHEN telemetry records are generated, the generator SHALL serialize samples into JSONL format (`data/telemetry/synthetic_auditor_train.jsonl`) pairing system instructions and telemetry prompts with ground-truth `IncidentAuditVerdict` JSON targets.

### R12 (Event-driven)
WHEN telemetry generation completes, the system SHALL validate 100% of generated sample targets against `IncidentAuditVerdict` schema and calculate class and violation distribution statistics.

### R13 (Event-driven)
WHEN fine-tuning concludes, the pipeline SHALL merge LoRA adapter weights with base Hugging Face weights (`Qwen/Qwen2.5-0.5B-Instruct`) into a single standalone 16-bit float (`fp16` or `bf16`) Hugging Face model directory.

### R14 (Event-driven)
WHEN model weights are merged, the conversion engine in `src/engine/gguf_converter.py` SHALL invoke or replicate `convert_hf_to_gguf.py` from `llama.cpp` to produce a base GGUF file (`models/auditor/qwen2.5-0.5b-auditor-f16.gguf`).

### R15 (Unwanted)
IF the input Hugging Face model directory lacks requisite tokenizer configuration files (`tokenizer.json`, `tokenizer_config.json`, `config.json`), THEN the conversion engine SHALL raise a `GGUFConversionError` detailing missing artifacts.

### R16 (State-driven)
WHILE converting to GGUF format, the system SHALL preserve model metadata including architecture family (`qwen2`), context window limit (2048), chat template, and EOS token ID.

### R17 (Ubiquitous)
The system SHALL provide a verification auditor engine in `src/engine/auditor_verifier.py` to evaluate whether model completions produce 100% valid, parsable JSON adhering to `IncidentAuditVerdict`.

### R18 (Event-driven)
WHEN evaluated against a validation evaluation set of 50 held-out telemetry prompts, the fine-tuned auditor SHALL achieve $\ge 95\%$ valid JSON parsing rate.

### R19 (Event-driven)
WHEN evaluated against the validation evaluation set, the fine-tuned auditor SHALL correctly classify `is_wrong_way` infraction status with $\ge 90\%$ accuracy.

### R20 (Event-driven)
WHEN evaluated against the validation evaluation set, the fine-tuned auditor SHALL classify `risk_level` with $\ge 85\%$ accuracy compared to ground-truth verdicts.

### R21 (Ubiquitous)
The system SHALL expose CLI entrypoints `sva-generate-telemetry`, `sva-finetune-auditor`, and `sva-export-gguf` registered in `pyproject.toml`.

### R22 (Ubiquitous)
Unit and integration test suites SHALL run fully offline without GPU hardware and without external Hugging Face network egress, executing in $< 5.0$ seconds via synthetic mock fixtures and mock model weight emulators.

### R23 (Event-driven)
WHEN `./init.sh` is executed, the entire test suite including SVA-DIST-002 tests SHALL pass with 100% green status and exit code 0.
