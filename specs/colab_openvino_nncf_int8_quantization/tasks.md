# Tasks: OpenVINO NNCF INT8 Quantization in Google Colab and Edge Artifact Export

- [x] T1 — Add `QuantizationConfig`, `CalibrationDatasetManifest`, and `QuantizationReport` Pydantic schemas in `src/pipeline/schemas.py`. Covers: R15.
- [x] T2 — Define `QuantizationError`, `QuantizationInputError`, `CalibrationDataError`, and `QuantizationRegressionError` domain exceptions in `src/core/exceptions.py`. Covers: R4, R10, R14.
- [x] T3 — Implement representative 300-frame calibration frame loader with day/dusk/night balancing and `nncf.Dataset` adapter in `src/pipeline/calibration_loader.py`. Covers: R6, R7, R8, R9, R10.
- [x] T4 — Implement OpenVINO NNCF Post-Training Quantization engine and OpenVINO IR (.xml/.bin) exporter in `src/engine/openvino_quantizer.py` verifying size reduction $\ge 60\%$ and accuracy retention $\Delta \text{mAP} \le 1.5\%$. Covers: R4, R5, R11, R12, R13, R14, R16.
- [x] T5 — Implement edge CPU benchmark runner in `src/engine/openvino_benchmark.py` measuring latency, FPS throughput ($\ge 25\text{ FPS}$), and peak RSS memory ($< 1.2\text{GB}$). Covers: R17, R18, R19.
- [x] T6 — Register CLI entrypoints `sva-quantize-openvino` and `sva-benchmark-openvino` in `pyproject.toml` and implement command-line handlers. Covers: R20.
- [x] T7 — Author interactive Google Colab notebook in `colab_notebooks/03_openvino_int8_quantization.ipynb` with Drive sync, NNCF PTQ execution, OpenVINO IR packaging, and benchmark cells. Covers: R1, R2, R3, R4, R5, R16.
- [x] T8 — Implement zero-egress mock OpenVINO IR topologies and calibration fixtures in `tests/fixtures/mock_openvino.py`. Covers: R21.
- [x] T9 — Implement unit tests for calibration frame loader and illumination partitioning in `tests/test_calibration_loader.py`. Covers: R6, R7, R8, R9, R10, R21.
- [x] T10 — Implement unit tests for OpenVINO NNCF quantization engine, size reduction, and mAP tolerance checks in `tests/test_openvino_quantizer.py`. Covers: R4, R5, R11, R12, R13, R14, R16, R21.
- [x] T11 — Implement unit tests for edge CPU benchmark runner, FPS calculation, and memory budget assertions in `tests/test_openvino_benchmark.py`. Covers: R17, R18, R19, R21.
- [x] T12 — Implement end-to-end integration test validating the entire quantization, export, and benchmark cycle in `tests/test_quantization_pipeline_integration.py`. Covers: R11, R12, R16, R17, R18, R21.
- [x] T13 — Execute `./init.sh` to verify complete green status across all new and existing test suites in under 5.0 seconds. Covers: R22.
