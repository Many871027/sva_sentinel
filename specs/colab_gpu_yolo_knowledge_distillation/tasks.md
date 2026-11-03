# Tasks: Google Colab GPU Knowledge Distillation Pipeline (Teacher YOLO11x to Student YOLO11n)

## Feature Metadata
- **ID:** SVA-DIST-001
- **Name:** colab_gpu_yolo_knowledge_distillation
- **Status:** spec_ready

## Sequential Implementation Checklist

- [x] T1 — Define domain-specific exceptions in `src/core/exceptions.py`: `DistillationError`, `DistillationLossError`, and `ModelExportError` inheriting from `FrameworkError`. Covers: R10.
- [x] T2 — Author Pydantic validation schemas in `src/pipeline/schemas.py`: `DistillationHyperparameters`, `DistillationEpochLog`, and `DistillationSummaryReport`. Covers: R6, R14.
- [x] T3 — Implement `LogitDistillationLoss` in `src/engine/distillation_loss.py` computing temperature-scaled KL divergence scaled by $\tau^2$. Covers: R6, R8.
- [x] T4 — Implement `FeatureAlignmentLoss` in `src/engine/distillation_loss.py` with $1\times 1$ conv adaptation projecting student channels to teacher channels and computing MSE distance. Covers: R6, R9.
- [x] T5 — Implement `CompositeDistillationLoss` in `src/engine/distillation_loss.py` combining hard task loss, logit KD loss, and feature alignment loss ($\alpha, \beta, \gamma$) with `NaN`/`Inf` detection. Covers: R6, R7, R8, R9, R10.
- [x] T6 — Implement `DistillationTrainer` in `src/engine/distillation_trainer.py` supporting training loops, learning rate scheduling, mixed-precision (AMP) hooks, validation metric evaluation, and parameter reduction assertions (> 80%). Covers: R11, R12, R13, R14.
- [x] T7 — Implement ONNX export and validation functions (`export_student_to_onnx`, `verify_onnx_integrity`) in `src/engine/onnx_exporter.py` with dynamic batching $[B, 3, 640, 640]$. Covers: R15, R16.
- [x] T8 — Author Google Colab Pro notebook `colab_notebooks/01_yolo_knowledge_distillation.ipynb` orchestrating Drive mount, dataset streaming, GPU hardware check, distillation execution, evaluation, and export. Covers: R1, R2, R3, R4, R5, R12, R15, R17.
- [x] T9 — Update `pyproject.toml` with CLI entrypoint `sva-distill` pointing to `src.engine.distillation_trainer:main`. Covers: R18.
- [x] T10 — Implement unit test suite in `tests/test_distillation_loss.py` validating $\mathcal{L}_{KD}$ temperature scaling, $\mathcal{L}_{feat}$ channel adaptation, loss weights $\alpha, \beta, \gamma$, and `NaN`/`Inf` error handling. Covers: R6, R8, R9, R10, R19, R20.
- [x] T11 — Implement unit test suite in `tests/test_distillation_trainer.py` validating distillation step execution, parameter count reduction verification (>80%), and epoch logging. Covers: R11, R13, R14, R19, R20.
- [x] T12 — Implement unit test suite in `tests/test_onnx_export.py` validating ONNX export formatting, input tensor shape assertions, and integrity checks. Covers: R15, R16, R19.
- [x] T13 — Implement end-to-end integration test in `tests/test_distillation_pipeline_integration.py` executing mock distillation step through export in < 5.0s. Covers: R1, R6, R7, R8, R9, R12, R13, R15, R16, R19, R20.
- [x] T14 — Execute `./init.sh` and complete test suite `pytest tests/` ensuring 100% green verification without network egress. Covers: R1, R2, R3, R4, R5, R6, R7, R8, R9, R10, R11, R12, R13, R14, R15, R16, R17, R18, R19, R20.

---

## Requirement Coverage Verification Matrix

| Requirement | Covering Tasks | Description |
|---|---|---|
| R1 | T8, T13, T14 | Colab Pro distillation notebook in `colab_notebooks/` |
| R2 | T8, T14 | Google Drive mount for checkpoint persistence |
| R3 | T8, T14 | Dataset synchronization from HF / Drive |
| R4 | T8, T14 | Checksum verification of dataset manifest in Colab |
| R5 | T8, T14 | CUDA availability check and warning |
| R6 | T2, T3, T4, T5, T10, T13, T14 | Composite distillation loss $\mathcal{L}_{total}$ |
| R7 | T5, T10, T13, T14 | Hard task detection loss (CIoU + DFL + BCE) |
| R8 | T3, T5, T10, T13, T14 | Logit distillation loss with $\tau^2$ scaling |
| R9 | T4, T5, T10, T13, T14 | Feature alignment loss with $1\times 1$ conv adaptation |
| R10 | T1, T5, T10, T14 | `DistillationLossError` on `NaN`/`Inf` loss |
| R11 | T6, T11, T14 | Mixed precision and distributed training support |
| R12 | T6, T8, T13, T14 | Student mAP recovery >= 92% of teacher |
| R13 | T6, T11, T13, T14 | Parameter count reduction > 80% (Student <= 3.0M) |
| R14 | T2, T6, T11, T14 | Epoch evaluation logging and structured history |
| R15 | T7, T8, T12, T13, T14 | Student weights export to PyTorch `.pt` and `.onnx` |
| R16 | T7, T8, T12, T13, T14 | ONNX structure and tensor shape validation |
| R17 | T8, T14 | Remote MLFlow / HF Hub export hooks |
| R18 | T9, T14 | CLI entrypoint `sva-distill` in `pyproject.toml` |
| R19 | T10, T11, T12, T13, T14 | Fast unit test execution < 5.0s |
| R20 | T10, T11, T13, T14 | Mock tensor verification without physical GPU |
