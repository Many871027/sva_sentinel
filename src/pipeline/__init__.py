"""Dataset acquisition, curation, normalization, augmentation, and splitting pipeline."""

from src.pipeline.calibration_loader import (
    CalibrationFrameLoader,
    classify_frame_illumination,
    compute_frame_luminance,
)
from src.pipeline.dataset_curation import (
    AnnotationParser,
    ViewpointAugmentor,
    map_category_to_canonical,
    normalize_and_clamp_box,
    save_curated_sample,
)
from src.pipeline.dataset_ingest import (
    TARGET_PROFILES,
    HuggingFaceDatasetIngest,
)
from src.pipeline.dataset_split import (
    DatasetPartitioner,
    calculate_sha256,
    generate_class_distribution_report,
    generate_integrity_manifest,
    save_curated_split_artifacts,
)
from src.pipeline.schemas import (
    CANONICAL_CLASSES,
    CANONICAL_CLASS_NAMES,
    AnnotationRecord,
    BoundingBoxYOLO,
    ClassCount,
    ClassDistributionReport,
    DatasetFileManifestEntry,
    DatasetManifest,
    IngestConfig,
    ManifestSummary,
    SplitClassDistribution,
    SplitSummary,
)

__all__ = [
    "CANONICAL_CLASSES",
    "CANONICAL_CLASS_NAMES",
    "BoundingBoxYOLO",
    "AnnotationRecord",
    "IngestConfig",
    "ClassCount",
    "SplitClassDistribution",
    "ClassDistributionReport",
    "DatasetFileManifestEntry",
    "SplitSummary",
    "ManifestSummary",
    "DatasetManifest",
    "HuggingFaceDatasetIngest",
    "TARGET_PROFILES",
    "map_category_to_canonical",
    "normalize_and_clamp_box",
    "AnnotationParser",
    "ViewpointAugmentor",
    "save_curated_sample",
    "DatasetPartitioner",
    "calculate_sha256",
    "generate_class_distribution_report",
    "generate_integrity_manifest",
    "save_curated_split_artifacts",
]
