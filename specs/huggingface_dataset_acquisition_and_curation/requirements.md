# Requirements: Hugging Face Multimodal Traffic & Micromobility Dataset Ingestion

## Feature Metadata
- **ID:** SVA-DATA-001
- **Name:** huggingface_dataset_acquisition_and_curation
- **Title:** Hugging Face Multimodal Traffic & Micromobility Dataset Ingestion
- **Target Subsystem:** `src/pipeline/` (Dataset Ingestion, Annotation Normalization & Curation Engine)
- **Data Target:** `data/raw/` and `data/curated/`

## Acceptance Criteria Mapping
| Acceptance Criterion | Requirements |
|---|---|
| Ingest script using `huggingface_hub` downloading target subsets (`starwit/overhead-traffic-anomalies`, `UniqueData/electric-scooters-tracking`, and street-level vehicle subsets) with resume capability and offline caching | R1, R2, R3, R4, R5, R6 |
| Normalization and conversion of heterogeneous annotation formats into unified YOLO TXT format (`class_id x_center y_center width height` normalized 0..1) with canonical classes: 0: car, 1: motorcycle, 2: bicycle, 3: e-scooter, 4: pedestrian | R7, R8, R9, R10, R11, R12 |
| Perspective, illumination, and occlusion augmentation filters matching intersection camera viewpoints | R13, R14, R15, R16, R17 |
| Deterministic 70/20/10 train/val/test splitting with seed control, class distribution reporting, and SHA256 integrity manifest generation | R18, R19, R20, R21, R22 |
| Verification strategy supporting mocked dataset samples for fast, deterministic unit testing without requiring real gigabyte HuggingFace downloads | R23, R24, R25 |

---

## Requirements (Strict EARS Notation)

### R1 (Ubiquitous)
The system SHALL implement a dataset ingestion pipeline in `src/pipeline/dataset_ingest.py` downloading multimodal traffic datasets from Hugging Face Hub using `huggingface_hub`.

### R2 (Event-driven)
WHEN dataset download is triggered, the system SHALL store raw dataset artifacts into structured local directories under `data/raw/<dataset_name>/` using persistent local caching.

### R3 (State-driven)
WHILE downloading dataset files via `huggingface_hub`, the system SHALL maintain resume capability by utilizing HTTP range requests and local partial file verification.

### R4 (Optional)
WHERE the `--offline` flag is passed or the runtime environment lacks external network egress, the system SHALL operate exclusively from locally cached dataset snapshots without issuing outbound network calls.

### R5 (Unwanted)
IF a requested Hugging Face dataset identifier is unreachable or nonexistent and no local cache snapshot exists, THEN the system SHALL raise a `DatasetIngestError` with a diagnostic message and terminate without corrupting local directories.

### R6 (Ubiquitous)
The system SHALL provide declarative configuration profiles for target datasets: `starwit/overhead-traffic-anomalies`, `UniqueData/electric-scooters-tracking`, and a street-level vehicle detection subset.

### R7 (Ubiquitous)
The system SHALL map heterogeneous input class taxonomies to a canonical 5-class ontology: `0: car`, `1: motorcycle`, `2: bicycle`, `3: e-scooter`, and `4: pedestrian`.

### R8 (Event-driven)
WHEN converting bounding box annotations from heterogeneous formats (COCO JSON, Pascal VOC XML, or YOLO pixel space), the system SHALL convert spatial coordinates into normalized YOLO TXT format: `<class_id> <x_center> <y_center> <width> <height>` where all spatial coordinates are floating-point values bounded within $[0.0, 1.0]$.

### R9 (Unwanted)
IF an annotation contains bounding box coordinates outside the image frame ($x < 0$, $y < 0$, $x > W$, or $y > H$) or non-positive dimensions ($w \le 0$ or $h \le 0$), THEN the system SHALL clamp valid intersecting extents to $[0.0, 1.0]$ or discard unrecoverable boxes while logging a warning.

### R10 (Event-driven)
WHEN an image annotation conversion completes, the system SHALL generate a corresponding label file `.txt` matching the image file stem in the curated destination directory.

### R11 (Ubiquitous)
The system SHALL provide modular annotation parsers supporting COCO bounding boxes (`[x_min, y_min, width, height]`), Pascal VOC bounding boxes (`[xmin, ymin, xmax, ymax]`), and existing normalized/unnormalized YOLO format files.

### R12 (Unwanted)
IF an annotation record references an unrecognized class label not mapped in the canonical ontology, THEN the system SHALL omit that specific bounding box from the output label file without halting the dataset conversion process.

### R13 (Ubiquitous)
The system SHALL implement an augmentation engine in `src/pipeline/dataset_curation.py` applying viewpoint-specific transformations to curated training samples.

### R14 (Event-driven)
WHEN perspective augmentation is executed on an image and its bounding boxes, the system SHALL apply a homographic perspective warp simulating ground-level intersection camera tilt angles between $-15^\circ$ and $+15^\circ$ and update the bounding box coordinates to tightly enclose the transformed objects.

### R15 (Event-driven)
WHEN illumination augmentation is executed, the system SHALL apply randomized contrast adjustments (factor $0.6$ to $1.4$), brightness shifts (factor $0.7$ to $1.3$), and gamma corrections (factor $0.7$ to $1.4$) simulating intersection conditions across day, dusk, twilight, and night.

### R16 (Event-driven)
WHEN occlusion augmentation is executed, the system SHALL overlay synthetic rectangular cutout occlusions covering between $5\%$ and $25\%$ of the image area simulating roadside poles, foliage, or passing vehicles.

### R17 (Unwanted)
IF a spatial augmentation causes a bounding box's area to become less than $16$ square pixels ($4\times 4$) or fall completely outside the image frame, THEN the system SHALL remove the degenerate box from the augmented label file.

### R18 (Event-driven)
WHEN dataset splitting is triggered, the system SHALL partition curated samples into training ($70\%$), validation ($20\%$), and testing ($10\%$) sets using a deterministic pseudorandom seed.

### R19 (State-driven)
WHILE partitioning the dataset, the system SHALL preserve relative class frequency balance across the train, validation, and test splits within a $\pm 5\%$ class distribution variance.

### R20 (Event-driven)
WHEN split generation completes, the system SHALL compute and save a class distribution summary report `data/curated/class_distribution.json` containing total counts and split percentages per canonical class.

### R21 (Event-driven)
WHEN curation completes, the system SHALL generate a cryptographic integrity manifest `data/curated/manifest.json` containing SHA256 checksums, byte sizes, relative file paths, and record counts for all images and labels.

### R22 (Unwanted)
IF an image file has no matching label file or has zero valid bounding boxes after filtering, THEN the system SHALL categorize it as a negative background sample or discard it based on configuration without halting pipeline execution.

### R23 (Ubiquitous)
The system SHALL provide a test fixture generator creating miniature synthetic mocked datasets with known annotations in `tests/fixtures/mock_datasets/`.

### R24 (Event-driven)
WHEN unit tests execute against the ingestion engine, the system SHALL mock `huggingface_hub` network calls using local fixture files without performing external network egress.

### R25 (Ubiquitous)
The automated unit test suite for ingestion, normalization, augmentation, splitting, and manifest generation SHALL execute to completion in under $5.0$ seconds.
