# Requirements: OpenVINO NNCF INT8 Quantization in Google Colab and Edge Artifact Export

## Feature Metadata
- **ID:** SVA-QUANT-001
- **Name:** colab_openvino_nncf_int8_quantization
- **Title:** OpenVINO NNCF INT8 Quantization in Google Colab and Edge Artifact Export
- **Target Subsystem:** `src/engine/` (OpenVINO NNCF Quantization Core & Benchmark), `src/pipeline/` (Calibration Dataset Preparation), `colab_notebooks/` (Google Colab Notebook)
- **Artifact Targets:** `colab_notebooks/03_openvino_int8_quantization.ipynb`, `models/quantized/openvino/yolo11n_int8.xml`, `models/quantized/openvino/yolo11n_int8.bin`

## Acceptance Criteria Mapping
| Acceptance Criterion | Requirements |
|---|---|
| Create `colab_notebooks/03_openvino_int8_quantization.ipynb` executing `nncf.quantize` on the distilled detector | R1, R2, R3, R4, R5 |
| Calibrate INT8 weights using representative unlabelled intersection frames across day, dusk, and night conditions (300 frames) | R6, R7, R8, R9, R10 |
| Demonstrate model size reduction $\ge 60\%$ with mAP@50-95 drop $\le 1.5\%$ compared to distilled FP32 model | R11, R12, R13, R14, R15 |
| Package and export calibrated OpenVINO IR artifacts ready for local deployment on laptop CPU achieving $\ge 25\text{ FPS}$ with $< 1.2\text{GB}$ RAM | R16, R17, R18, R19, R20 |
| Zero-egress fast verification and test suite | R21, R22 |

---

## Requirements (Strict EARS Notation)

### R1 (Ubiquitous)
The system SHALL provide an interactive Post-Training Quantization (PTQ) notebook in `colab_notebooks/03_openvino_int8_quantization.ipynb` configured for execution within Google Colab environments.

### R2 (Event-driven)
WHEN executed in Google Colab, the notebook SHALL mount Google Drive (`/content/drive/MyDrive/sva_sentinel/`) to load distilled FP32 student checkpoints (`models/distilled/yolo11n_distilled.onnx` or `.pt`) and persist calibrated OpenVINO IR artifacts across sessions.

### R3 (Event-driven)
WHEN executed in Google Colab, the notebook SHALL install required optimization dependencies (`openvino>=2024.4.0`, `nncf>=2.12.0`, `opencv-python-headless`, `psutil`).

### R4 (Unwanted)
IF the input distilled model artifact (`.onnx` or `.pt`) does not exist at the designated path, THEN the quantization engine SHALL raise a `QuantizationInputError` and abort execution before loading weights.

### R5 (Ubiquitous)
The quantization engine in `src/engine/openvino_quantizer.py` SHALL utilize OpenVINO Neural Network Compression Framework (`nncf.quantize` or equivalent OpenVINO optimization API) targeting 8-bit integer (`INT8`) quantization.

### R6 (Ubiquitous)
The system SHALL provide a calibration frame loader in `src/pipeline/calibration_loader.py` that selects 300 representative unlabelled intersection frames from curated datasets.

### R7 (State-driven)
WHILE assembling the 300-frame calibration set, the loader SHALL balance lighting representations containing:
- 120 daytime high-visibility frames ($40\%$).
- 90 dusk / golden-hour transition frames ($30\%$).
- 90 night / low-lux artificial illumination frames ($30\%$).

### R8 (Ubiquitous)
The calibration frame loader SHALL preprocess calibration images into normalized float32 tensors with resolution $[1, 3, 640, 640]$ and pixel values scaled to $[0.0, 1.0]$.

### R9 (State-driven)
WHILE iterating calibration batches, the loader SHALL wrap tensors in an `nncf.Dataset` generator yielding dictionary inputs compatible with the distilled detector's input layer name.

### R10 (Unwanted)
IF any calibration frame is corrupted, zero-byte, or incapable of being decoded as an image, THEN the calibration loader SHALL skip the defective frame, log a warning, and draw a replacement from the corresponding illumination partition.

### R11 (Event-driven)
WHEN Post-Training Quantization completes, the resulting INT8 OpenVINO IR model size (aggregate byte size of `.bin` and `.xml`) SHALL achieve at least a $60\%$ reduction compared to the baseline distilled FP32 ONNX/PyTorch model.

### R12 (Event-driven)
WHEN evaluated on the validation intersection dataset, the quantized INT8 detector SHALL maintain detection accuracy such that the absolute drop in `mAP@50-95` does not exceed $1.5\%$ compared to the distilled FP32 student baseline ($\Delta \text{mAP@50-95} \le 0.015$).

### R13 (Event-driven)
WHEN accuracy evaluation is performed, the system SHALL compute both `mAP@50` and `mAP@50-95` across all 5 canonical classes (`car`, `motorcycle`, `bicycle`, `e-scooter`, `pedestrian`).

### R14 (Unwanted)
IF the accuracy evaluation demonstrates an mAP degradation exceeding $1.5\%$ ($\Delta \text{mAP@50-95} > 0.015$), THEN the quantization engine SHALL flag an accuracy regression warning and record the non-compliance status in the quantization audit report.

### R15 (Ubiquitous)
The system SHALL define Pydantic schemas in `src/pipeline/schemas.py` for quantization configuration (`QuantizationConfig`), calibration metadata (`CalibrationDatasetManifest`), and quantization audit report (`QuantizationReport`).

### R16 (Event-driven)
WHEN quantization succeeds, the system SHALL serialize and package the calibrated model into OpenVINO Intermediate Representation format: an XML topology file (`models/quantized/openvino/yolo11n_int8.xml`) and a binary weights file (`models/quantized/openvino/yolo11n_int8.bin`).

### R17 (Event-driven)
WHEN the exported OpenVINO IR model is deployed on a laptop CPU using OpenVINO Runtime (`ov.Core().compile_model(..., device_name="CPU")`), the inference pipeline SHALL sustain a throughput of at least $25$ frames per second ($\ge 25\text{ FPS}$, latency $\le 40\text{ms}$ per $640\times 640$ frame).

### R18 (State-driven)
WHILE running edge CPU inference on the target OpenVINO IR model, the process resident set size (RSS) memory consumption SHALL NOT exceed $1.2\text{GB}$ RAM, adhering strictly to the $4.0\text{GB}$ container memory cgroup cap.

### R19 (Ubiquitous)
The system SHALL implement a CPU benchmark runner in `src/engine/openvino_benchmark.py` that executes a 100-frame warm benchmark on edge CPU measuring mean latency, P95 latency, FPS throughput, and peak RSS memory via `psutil`.

### R20 (Ubiquitous)
The system SHALL register CLI entrypoints `sva-quantize-openvino` and `sva-benchmark-openvino` in `pyproject.toml`.

### R21 (Ubiquitous)
Unit and integration test suites SHALL run fully offline without physical OpenVINO hardware accelerators, executing in $< 5.0$ seconds via synthetic mock calibration arrays and mock OpenVINO IR emulators.

### R22 (Event-driven)
WHEN `./init.sh` is executed, the entire test suite including SVA-QUANT-001 tests SHALL pass with 100% green status and exit code 0.
