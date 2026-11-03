# Tasks: Ground-Level Camera Ingestion and Planar Homography Calibration

## Feature Metadata
- **ID:** SVA-PRE-002
- **Name:** camera_rtsp_and_homography_calibration
- **Status:** spec_ready

## Sequential Implementation Checklist

- [ ] T1 — Define domain-specific exceptions in `src/core/exceptions.py`: `CalibrationError`, `CalibrationAccuracyError`, and `ConfigurationError` inheriting from `FrameworkError`. Covers: R5, R14, R16.
- [ ] T2 — Author Pydantic validation schemas in `src/pipeline/schemas.py` declaring `Point2D`, `LaneConfig`, `CalibrationMetadata`, and `HomographyConfig` with field validators for 3x3 matrices and unit vectors. Covers: R6, R7, R8, R9, R15.
- [ ] T3 — Implement `HomographyProjector` in `src/pipeline/homography.py` supporting forward coordinate projection $(u, v) \to (X, Y)$, analytical inverse projection $(X, Y) \to (u, v)$, and batch projection using normalized $\mathbf{H}$ and $\mathbf{H}^{-1}$. Covers: R1, R10.
- [ ] T4 — Implement bounding box foot-point extraction $((x_1 + x_2)/2, y_2)$ and metric projection within `HomographyProjector` in `src/pipeline/homography.py`. Covers: R10.
- [ ] T5 — Implement kinematic velocity vector calculation, legal unit vector normalization ($\|\mathbf{u}_{legal}\| = 1.0$), and directional cosine similarity metric engine ($\cos \theta = \frac{\mathbf{v} \cdot \mathbf{u}}{\|\mathbf{v}\|\|\mathbf{u}\|}$) with wrong-way condition thresholding ($\cos \theta < -0.7$) in `src/pipeline/homography.py`. Covers: R8, R11, R12.
- [ ] T6 — Implement 4-point Ground Control Point (GCP) planar homography computation using Direct Linear Transformation (DLT) with matrix normalization ($H_{3,3} = 1.0$) and singular/collinear point detection raising `CalibrationError` in `src/pipeline/calibration.py`. Covers: R2, R3, R5.
- [ ] T7 — Implement pavement calibration accuracy verification in `src/pipeline/calibration.py` calculating relative Euclidean distance error against surveyed ground truth measurements with strict `< 5.0%` threshold enforcement raising `CalibrationAccuracyError` upon violation. Covers: R13, R14.
- [ ] T8 — Implement configuration serialization and deserialization functions (`save_homography_config`, `load_homography_config`) with atomic file writes and Pydantic schema validation in `src/pipeline/calibration.py`. Covers: R6, R7, R8, R9, R15, R16.
- [ ] T9 — Implement optional interactive OpenCV calibration utility in `src/pipeline/calibration.py` with mouse callback to select 4 GCP corners on image frames and prompt for real-world metric dimensions. Covers: R4.
- [ ] T10 — Create canonical baseline configuration file `configs/homography_config.json` defining vehicular and micromobility lanes, 3x3 homography matrix, and directional thresholds conforming to `HomographyConfig`. Covers: R6, R7, R8, R9, R15.
- [ ] T11 — Implement unit test suite in `tests/test_homography_math.py` testing forward/inverse projections, foot-point calculations, batch transformations, velocity vectors, cosine similarity evaluation, and singular matrix rejection. Covers: R1, R5, R10, R11, R12.
- [ ] T12 — Implement unit test suite in `tests/test_calibration_engine.py` testing 4-point DLT calibration, distance accuracy verification (< 5% passing, >= 5% failing), schema validation, file persistence, and error handling. Covers: R2, R3, R5, R6, R7, R8, R9, R13, R14, R15, R16.
- [ ] T13 — Implement end-to-end integration test in `tests/test_calibration_pipeline_integration.py` validating the pipeline workflow from synthetic calibration points to foot-point tracking projection and counter-flow infraction detection. Covers: R1, R3, R8, R10, R11, R12, R13, R14, R15.
- [ ] T14 — Execute `./init.sh` and run full test suite `python -m pytest tests/` to confirm 100% green verification. Covers: R1, R2, R3, R4, R5, R6, R7, R8, R9, R10, R11, R12, R13, R14, R15, R16.

---

## Requirement Coverage Verification Matrix

| Requirement | Covering Tasks | Description |
|---|---|---|
| R1 | T3, T11, T13, T14 | Planar homography forward and inverse coordinate mapping |
| R2 | T6, T12, T14 | 4-point GCP homography matrix calculation via DLT |
| R3 | T6, T12, T13, T14 | Programmatic calibration with normalized matrix output |
| R4 | T9, T14 | Interactive OpenCV calibration GUI utility |
| R5 | T1, T6, T11, T12, T14 | Detection and rejection of degenerate/collinear points |
| R6 | T2, T8, T10, T12, T14 | Serialization to `configs/homography_config.json` |
| R7 | T2, T8, T10, T12, T14 | Persistence of lane polygon boundaries in pixel and metric space |
| R8 | T2, T5, T8, T10, T12, T13, T14 | Definition and storage of unit vector $\mathbf{u}_{legal}$ |
| R9 | T2, T8, T10, T12, T14 | Directional vector thresholds configuration |
| R10 | T3, T4, T11, T13, T14 | Object foot-point extraction and ground projection |
| R11 | T5, T11, T13, T14 | Velocity calculation and cosine similarity evaluation |
| R12 | T5, T11, T13, T14 | Counter-flow violation state evaluation ($\cos \theta < -0.7$) |
| R13 | T7, T12, T13, T14 | Pavement calibration test distance error calculation |
| R14 | T1, T7, T12, T13, T14 | Enforcement of `< 5.0%` distance accuracy error ceiling |
| R15 | T2, T8, T10, T12, T13, T14 | Runtime Pydantic validation during config loading |
| R16 | T1, T8, T12, T14 | Error handling for missing or malformed configuration |
