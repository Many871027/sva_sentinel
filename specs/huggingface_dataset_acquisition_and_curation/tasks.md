# Tasks: Hugging Face Multimodal Traffic & Micromobility Dataset Ingestion

## Feature Metadata
- **ID:** SVA-DATA-001
- **Name:** huggingface_dataset_acquisition_and_curation
- **Status:** spec_ready

## Sequential Implementation Checklist

- [x] T1 — Define domain-specific exceptions in `src/core/exceptions.py`: `DatasetIngestError`, `AnnotationFormatError`, and `CurationError` inheriting from `FrameworkError`. Covers: R5, R9, R12.
- [x] T2 — Author Pydantic validation schemas in `src/pipeline/schemas.py`: `BoundingBoxYOLO`, `AnnotationRecord`, `DatasetManifest`, `ClassDistributionReport`, and `IngestConfig`. Covers: R7, R8, R20, R21.
- [x] T3 — Implement Hugging Face Hub ingestion engine in `src/pipeline/dataset_ingest.py` supporting dataset snapshot download, local caching (`data/raw/<dataset_name>/`), HTTP resume capability, `--offline` local fallback, and error handling for unreachable datasets. Covers: R1, R2, R3, R4, R5, R6.
- [x] T4 — Implement annotation parsers in `src/pipeline/dataset_curation.py` for COCO JSON, Pascal VOC XML, and YOLO TXT formats, mapping source category names to canonical 5-class ontology (`0: car`, `1: motorcycle`, `2: bicycle`, `3: e-scooter`, `4: pedestrian`) and filtering unmapped labels. Covers: R7, R8, R10, R11, R12, R22.
- [x] T5 — Implement spatial coordinate normalizer, boundary clamper ($[0.0, 1.0]$), and degenerate box filter (< 16 px² area) in `src/pipeline/dataset_curation.py`. Covers: R8, R9, R17.
- [x] T6 — Implement viewpoint augmentation filters in `src/pipeline/dataset_curation.py`: perspective homography warp ($-15^\circ$ to $+15^\circ$ camera tilt), illumination variation (contrast, brightness, gamma for day/dusk/night), and rectangular cutout occlusion ($5\%$ to $25\%$ image area). Covers: R13, R14, R15, R16, R17.
- [x] T7 — Implement deterministic dataset partitioner in `src/pipeline/dataset_split.py` executing 70% train / 20% validation / 10% test splitting with seed control, preserving relative class distribution variance within $\pm 5\%$. Covers: R18, R19.
- [x] T8 — Implement class distribution statistics reporter in `src/pipeline/dataset_split.py` calculating class counts and split percentages serialized to `data/curated/class_distribution.json`. Covers: R20.
- [x] T9 — Implement cryptographic SHA256 integrity manifest generator in `src/pipeline/dataset_split.py` hashing all curated images and label files and saving to `data/curated/manifest.json`. Covers: R21.
- [x] T10 — Update `pyproject.toml` with `huggingface-hub>=0.25.0` dependency and provide CLI entrypoints in `src/pipeline/dataset_ingest.py`, `src/pipeline/dataset_curation.py`, and `src/pipeline/dataset_split.py`. Covers: R1, R4, R6, R18.
- [x] T11 — Implement mock dataset fixture generator in `tests/fixtures/mock_datasets/` generating synthetic miniature COCO, VOC, and raw YOLO samples. Covers: R23, R24.
- [x] T12 — Implement unit test suite for dataset ingestion and offline caching in `tests/test_dataset_ingest.py` mocking `huggingface_hub` downloads with zero external network egress. Covers: R1, R2, R3, R4, R5, R6, R24, R25.
- [x] T13 — Implement unit test suite for annotation parsing, normalization, coordinate clamping, and class mapping in `tests/test_dataset_curation.py`. Covers: R7, R8, R9, R10, R11, R12, R17, R22, R25.
- [x] T14 — Implement unit test suite for viewpoint augmentations (perspective warp, illumination, cutout occlusion) in `tests/test_dataset_augmentations.py`. Covers: R13, R14, R15, R16, R17, R25.
- [x] T15 — Implement unit test suite for deterministic 70/20/10 splitting, class balance verification, and SHA256 manifest generation in `tests/test_dataset_split.py`. Covers: R18, R19, R20, R21, R25.
- [x] T16 — Implement end-to-end integration test in `tests/test_dataset_pipeline_integration.py` validating the full synthetic data pipeline from mocked download to curated YOLO output and manifest in < 5.0 seconds. Covers: R1, R7, R8, R14, R18, R20, R21, R24, R25.
- [x] T17 — Execute `./init.sh` and complete test suite `pytest tests/` ensuring 100% green verification without network egress. Covers: R1, R2, R3, R4, R5, R6, R7, R8, R9, R10, R11, R12, R13, R14, R15, R16, R17, R18, R19, R20, R21, R22, R23, R24, R25.

---

## Requirement Coverage Verification Matrix

| Requirement | Covering Tasks | Description |
|---|---|---|
| R1 | T3, T10, T12, T16, T17 | Dataset ingestion pipeline via `huggingface_hub` |
| R2 | T3, T12, T17 | Persistent local caching in `data/raw/<dataset_name>/` |
| R3 | T3, T12, T17 | Resume capability with HTTP range requests & partial file validation |
| R4 | T3, T10, T12, T17 | Offline execution mode utilizing local cache snapshots |
| R5 | T1, T3, T12, T17 | `DatasetIngestError` for unreachable or invalid datasets |
| R6 | T3, T10, T12, T17 | Declarative ingestion profiles for target traffic datasets |
| R7 | T2, T4, T13, T16, T17 | Canonical 5-class ontology mapping (`0` to `4`) |
| R8 | T2, T4, T5, T13, T16, T17 | Normalized YOLO TXT format conversion ($[0.0, 1.0]$ bounds) |
| R9 | T1, T5, T13, T17 | Bounding box boundary clamping and invalid box handling |
| R10 | T4, T13, T17 | Generation of matching `.txt` label files per image stem |
| R11 | T4, T13, T17 | Heterogeneous annotation parsing (COCO, Pascal VOC, YOLO) |
| R12 | T1, T4, T13, T17 | Omission of unmapped class labels without pipeline failure |
| R13 | T6, T14, T17 | Viewpoint-specific augmentation engine |
| R14 | T6, T14, T16, T17 | Homographic perspective warp ($-15^\circ$ to $+15^\circ$ tilt) |
| R15 | T6, T14, T17 | Photometric illumination adjustments (contrast, brightness, gamma) |
| R16 | T6, T14, T17 | Synthetic rectangular cutout occlusion masks ($5\%$ to $25\%$) |
| R17 | T5, T6, T13, T14, T17 | Pruning degenerate bounding boxes (< 16 px² area) |
| R18 | T7, T10, T15, T16, T17 | Deterministic 70/20/10 split with seed control |
| R19 | T7, T15, T17 | Class frequency balance preservation within $\pm 5\%$ variance |
| R20 | T2, T8, T15, T16, T17 | Class distribution summary report `class_distribution.json` |
| R21 | T2, T9, T15, T16, T17 | Cryptographic SHA256 integrity manifest `manifest.json` |
| R22 | T4, T13, T17 | Negative background sample categorization or discard |
| R23 | T11, T17 | Miniature synthetic mock dataset fixtures |
| R24 | T11, T12, T16, T17 | Mocking `huggingface_hub` calls with zero network egress |
| R25 | T12, T13, T14, T15, T16, T17 | Test suite execution speed under 5.0 seconds |
