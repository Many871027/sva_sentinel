"""Pydantic validation schemas for dataset ingestion, curation, and manifest generation.

Covers:
- R7: Canonical 5-class ontology mapping (0: car, 1: motorcycle, 2: bicycle, 3: e-scooter, 4: pedestrian).
- R8: BoundingBoxYOLO normalized spatial representation in [0.0, 1.0].
- R20: ClassDistributionReport schema for split-level statistics.
- R21: DatasetManifest schema for cryptographic SHA256 integrity catalogs.
"""

from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Annotated, Any, Dict, List, Optional, Tuple
from pydantic import BaseModel, Field, field_validator, model_validator

CANONICAL_CLASSES: Dict[int, str] = {
    0: "car",
    1: "motorcycle",
    2: "bicycle",
    3: "e-scooter",
    4: "pedestrian",
}

CANONICAL_CLASS_NAMES: Dict[str, int] = {v: k for k, v in CANONICAL_CLASSES.items()}


class BoundingBoxYOLO(BaseModel):
    """Normalized axis-aligned bounding box adhering to YOLO format."""

    class_id: int = Field(ge=0, le=4, description="Canonical class identifier [0, 4]")
    x_center: float = Field(ge=0.0, le=1.0, description="Normalized horizontal center [0.0, 1.0]")
    y_center: float = Field(ge=0.0, le=1.0, description="Normalized vertical center [0.0, 1.0]")
    width: float = Field(gt=0.0, le=1.0, description="Normalized bounding box width (0.0, 1.0]")
    height: float = Field(gt=0.0, le=1.0, description="Normalized bounding box height (0.0, 1.0]")

    def to_yolo_line(self) -> str:
        """Serialize to standard space-delimited YOLO annotation format."""
        return (
            f"{self.class_id} {self.x_center:.6f} {self.y_center:.6f} "
            f"{self.width:.6f} {self.height:.6f}"
        )

    @classmethod
    def from_yolo_line(cls, line: str) -> "BoundingBoxYOLO":
        """Deserialize from space-delimited YOLO format line."""
        parts = line.strip().split()
        if len(parts) != 5:
            raise ValueError(f"Expected 5 space-delimited values in YOLO line, got {len(parts)}: '{line}'")
        return cls(
            class_id=int(parts[0]),
            x_center=float(parts[1]),
            y_center=float(parts[2]),
            width=float(parts[3]),
            height=float(parts[4]),
        )


class AnnotationRecord(BaseModel):
    """Collection of bounding boxes corresponding to a single image frame."""

    image_path: str = Field(description="Relative or absolute path to the image file")
    image_stem: str = Field(description="Stem identifier of the image without file extension")
    width: int = Field(gt=0, description="Image frame pixel width")
    height: int = Field(gt=0, description="Image frame pixel height")
    boxes: List[BoundingBoxYOLO] = Field(default_factory=list, description="List of validated YOLO boxes")
    is_negative: bool = Field(default=False, description="True if image is a background frame with zero boxes")

    def to_yolo_content(self) -> str:
        """Render all bounding boxes into full multi-line label file contents."""
        return "\n".join(box.to_yolo_line() for box in self.boxes)


class IngestConfig(BaseModel):
    """Declarative profile for downloading and caching Hugging Face Hub datasets."""

    dataset_name: str = Field(description="Hugging Face repository ID (e.g. 'starwit/overhead-traffic-anomalies')")
    subset: Optional[str] = Field(default=None, description="Dataset configuration or subset name")
    split: Optional[str] = Field(default=None, description="Repository split name if segregated upstream")
    raw_dir: Path = Field(default=Path("data/raw"), description="Destination root for raw cached assets")
    offline: bool = Field(default=False, description="Operate exclusively from local caches without network egress")
    revision: Optional[str] = Field(default=None, description="Specific git commit hash or branch tag")
    allow_patterns: Optional[List[str]] = Field(default=None, description="Wildcard match patterns for download")
    ignore_patterns: Optional[List[str]] = Field(default=None, description="Wildcard ignore patterns for download")


class ClassCount(BaseModel):
    """Per-class occurrence tally and relative percentage."""

    class_id: int = Field(ge=0, le=4)
    class_name: str
    count: int = Field(ge=0)
    percentage: float = Field(ge=0.0, le=100.0)


class SplitClassDistribution(BaseModel):
    """Class frequency distribution for a specific dataset split."""

    total_boxes: int = Field(ge=0)
    classes: Dict[str, ClassCount]


class ClassDistributionReport(BaseModel):
    """Master class distribution summary report across all splits."""

    canonical_classes: Dict[int, str] = Field(default=CANONICAL_CLASSES)
    total_annotations: int = Field(ge=0)
    splits: Dict[str, SplitClassDistribution]
    master_distribution: Dict[str, ClassCount]


class DatasetFileManifestEntry(BaseModel):
    """Individual cryptographic manifest record for an image or label file."""

    path: str = Field(description="Posix relative path from curated dataset root")
    sha256: str = Field(min_length=64, max_length=64, description="Hex-encoded SHA256 digest")
    bytes: int = Field(ge=0, description="Exact file size in bytes")


class SplitSummary(BaseModel):
    """High-level image and annotation tallies for a single partition."""

    images: int = Field(ge=0)
    annotations: int = Field(ge=0)


class ManifestSummary(BaseModel):
    """Aggregate volume and split metrics."""

    total_images: int = Field(ge=0)
    total_annotations: int = Field(ge=0)
    splits: Dict[str, SplitSummary]


class DatasetManifest(BaseModel):
    """Cryptographic integrity manifest conforming to R21."""

    dataset_version: str = Field(default="1.0.0")
    generated_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    seed: int = Field(default=42)
    summary: ManifestSummary
    files: List[DatasetFileManifestEntry]


# =============================================================================
# Base Model Catalog & Registry Schemas (SVA-DATA-002: R7, R8, R9)
# =============================================================================


class BaseModelProfile(BaseModel):
    """Declarative specification profile for target base models."""

    model_id: str = Field(description="Unique base model identifier (e.g. 'yolo11n')")
    model_name: str = Field(description="Human-readable descriptive model name")
    model_role: str = Field(description="Architectural role: 'teacher', 'student', or 'auditor'")
    model_family: str = Field(description="Architecture family: 'yolo11', 'qwen2.5', 'rtdetr'")
    version: str = Field(default="1.0.0", description="Model semantic version")
    format: str = Field(description="Model checkpoint serialization format: 'pt' or 'safetensors'")
    source_repo: str = Field(description="Hugging Face Hub repository ID")
    file_name: str = Field(description="Weight checkpoint filename in repo")
    expected_sha256: str = Field(min_length=64, max_length=64, description="Hex-encoded expected SHA256 checksum")
    file_size_bytes: int = Field(ge=0, description="Expected file size in bytes")
    input_resolution: List[int] = Field(description="Input tensor dimensions [height, width] or [context_length]")
    parameter_count: int = Field(ge=0, description="Total parameter count")
    description: Optional[str] = Field(default=None, description="Detailed profile description")


class ModelBenchmarkMetrics(BaseModel):
    """Runtime benchmark profiling metrics recorded during evaluation (R11, R12, R13)."""

    mean_latency_ms: float = Field(ge=0.0, description="Mean FP32 inference latency in milliseconds")
    p95_latency_ms: float = Field(ge=0.0, description="95th percentile inference latency in milliseconds")
    peak_memory_rss_mb: float = Field(ge=0.0, description="Peak resident set size memory in megabytes")
    parameter_count: int = Field(ge=0, description="Total verified model parameter count")
    map_50: Optional[float] = Field(default=None, ge=0.0, le=1.0, description="mAP@50 on sample intersection test set")
    map_50_95: Optional[float] = Field(default=None, ge=0.0, le=1.0, description="mAP@50-95 on sample intersection test set")


class ModelRegistryEntry(BaseModel):
    """Canonical registry entry stored in models/base/registry.json (R8, R9)."""

    model_id: str = Field(description="Unique model identifier")
    model_name: str = Field(description="Human-readable model name")
    model_role: str = Field(description="Role: teacher, student, auditor")
    model_family: str = Field(description="Model architecture family")
    version: str = Field(default="1.0.0", description="Model version")
    format: str = Field(description="Checkpoint format (pt, safetensors)")
    source_repo: str = Field(description="Source Hugging Face repository")
    file_name: str = Field(description="Checkpoint filename")
    sha256: str = Field(min_length=64, max_length=64, description="Verified SHA256 checksum")
    file_size_bytes: int = Field(ge=0, description="File size in bytes")
    input_resolution: List[int] = Field(description="Input tensor resolution")
    parameter_count: int = Field(ge=0, description="Total model parameter count")
    registered_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    weights_path: str = Field(description="Path to stored weights file")
    benchmarks: Optional[ModelBenchmarkMetrics] = Field(default=None, description="Baseline benchmark metrics")


class ModelRegistryCatalog(BaseModel):
    """Centralized catalog manifest for all registered base models (R8)."""

    catalog_version: str = Field(default="1.0.0")
    updated_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    models: Dict[str, ModelRegistryEntry] = Field(default_factory=dict)


# Predefined target model profiles (R7)
TARGET_MODEL_PROFILES: Dict[str, BaseModelProfile] = {
    "yolo11x": BaseModelProfile(
        model_id="yolo11x",
        model_name="YOLO11x High-Capacity Teacher Detector",
        model_role="teacher",
        model_family="yolo11",
        version="1.0.0",
        format="pt",
        source_repo="ultralytics/yolo11x",
        file_name="yolo11x.pt",
        expected_sha256="4a719c8f0e5b7c02b1f83c18b763ec18d86f7b11d35503080ff43f3c4db21d45",
        file_size_bytes=114000000,
        input_resolution=[640, 640],
        parameter_count=56900000,
        description="High-capacity teacher vision detector for distillation in Colab GPU",
    ),
    "yolo11n": BaseModelProfile(
        model_id="yolo11n",
        model_name="YOLO11n Lightweight Student Detector",
        model_role="student",
        model_family="yolo11",
        version="1.0.0",
        format="pt",
        source_repo="ultralytics/yolo11n",
        file_name="yolo11n.pt",
        expected_sha256="a6a16c7a950e82daeb77949bd9280d4c1cfd14e390c58bc3922d99723de4bca8",
        file_size_bytes=5621420,
        input_resolution=[640, 640],
        parameter_count=2610000,
        description="Lightweight student detector for OpenVINO INT8 quantization",
    ),
    "qwen2.5-0.5b-instruct": BaseModelProfile(
        model_id="qwen2.5-0.5b-instruct",
        model_name="Qwen2.5-0.5B-Instruct Forensic Auditor",
        model_role="auditor",
        model_family="qwen2.5",
        version="1.0.0",
        format="safetensors",
        source_repo="Qwen/Qwen2.5-0.5B-Instruct",
        file_name="model.safetensors",
        expected_sha256="c8c5ddb8ee9ffb732d84793f773b4e64f7df5f275e656d0289e6eb1325d72f10",
        file_size_bytes=980000000,
        input_resolution=[2048],
        parameter_count=490000000,
        description="Base instruction language model for incident reasoning and GGUF conversion",
    ),
}


# =============================================================================
# Knowledge Distillation Schemas (SVA-DIST-001: R6, R12, R13, R14, R17)
# =============================================================================


class DistillationHyperparameters(BaseModel):
    """Hyperparameters configuring composite distillation loss and training loop (R6, R17)."""

    alpha: float = Field(default=1.0, ge=0.0, description="Task detection hard loss weight alpha >= 0.0")
    beta: float = Field(default=2.5, ge=0.0, description="Logit response-based KD loss weight beta >= 0.0")
    gamma: float = Field(default=0.5, ge=0.0, description="Feature alignment loss weight gamma >= 0.0")
    temperature: float = Field(default=3.0, ge=1.0, description="Softmax distillation temperature tau >= 1.0")
    learning_rate: float = Field(default=1e-3, gt=0.0, description="Initial learning rate")
    batch_size: int = Field(default=32, gt=0, description="Training batch size")
    epochs: int = Field(default=50, gt=0, description="Total training epochs")
    student_channels: int = Field(default=128, gt=0, description="Student neck/backbone feature channels Cs")
    teacher_channels: int = Field(default=512, gt=0, description="Teacher neck/backbone feature channels Ct")
    imgsz: int = Field(default=640, gt=0, description="Input image resolution in pixels")
    mixed_precision: bool = Field(default=True, description="Enable Automatic Mixed Precision (AMP)")
    optimizer: str = Field(default="AdamW", description="Optimizer algorithm name")
    weight_decay: float = Field(default=1e-4, ge=0.0, description="Optimizer L2 weight decay")


class DistillationEpochLog(BaseModel):
    """Structured telemetry record captured at the conclusion of each training epoch (R14)."""

    epoch: int = Field(ge=1, description="1-indexed epoch number")
    total_loss: float = Field(ge=0.0, description="Total composite distillation loss L_total")
    hard_loss: float = Field(ge=0.0, description="Hard detection task loss L_hard")
    kd_loss: float = Field(ge=0.0, description="Response-based logit distillation loss L_KD")
    feat_loss: float = Field(ge=0.0, description="Feature representation alignment loss L_feat")
    map_50: float = Field(ge=0.0, le=1.0, description="Validation mean Average Precision at IoU=0.50")
    map_50_95: float = Field(ge=0.0, le=1.0, description="Validation mean Average Precision at IoU=0.50:0.95")
    val_loss: Optional[float] = Field(default=None, ge=0.0, description="Validation task evaluation loss")
    learning_rate: Optional[float] = Field(default=None, ge=0.0, description="Active learning rate")
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class DistillationSummaryReport(BaseModel):
    """Comprehensive post-training audit and artifact report (R12, R13, R14)."""

    experiment_name: str = Field(default="colab-gpu-distillation-yolo11", description="Experiment run name")
    teacher_model: str = Field(default="yolo11x", description="Teacher model identifier")
    student_model: str = Field(default="yolo11n", description="Student model identifier")
    teacher_parameters: int = Field(gt=0, description="Total teacher model parameters")
    student_parameters: int = Field(gt=0, description="Total student model parameters (<= 3.0e6)")
    parameter_reduction_pct: float = Field(ge=0.0, le=100.0, description="Parameter reduction percentage (> 80%)")
    teacher_map_50: float = Field(ge=0.0, le=1.0, description="Teacher baseline mAP@50")
    student_map_50: float = Field(ge=0.0, le=1.0, description="Distilled student final mAP@50")
    map_retention_pct: float = Field(ge=0.0, description="Student / Teacher mAP@50 retention percentage")
    achieved_target_retention: bool = Field(description="True if student achieves >= 92% teacher mAP@50 (R12)")
    achieved_target_reduction: bool = Field(description="True if student parameters <= 3.0M and reduction > 80% (R13)")
    hyperparameters: DistillationHyperparameters
    total_epochs: int = Field(ge=1, description="Total epochs trained")
    best_epoch: int = Field(ge=1, description="Epoch index yielding highest mAP@50")
    best_map_50: float = Field(ge=0.0, le=1.0, description="Highest mAP@50 achieved during training")
    exported_pt_path: Optional[str] = Field(default=None, description="Path to exported PyTorch checkpoint")
    exported_onnx_path: Optional[str] = Field(default=None, description="Path to exported ONNX artifact")
    history: List[DistillationEpochLog] = Field(default_factory=list, description="Epoch history records")
    completed_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())



# =============================================================================
# Forensic Auditor & llama.cpp Schemas (SVA-DIST-002: R7, R8, R12, R16)
# =============================================================================


class RiskLevel(str, Enum):
    """Categorical infraction severity levels for forensic auditing."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class TrafficTelemetryRecord(BaseModel):
    """Raw trajectory telemetry emitted by edge detector and tracker (R7, R8)."""

    incident_id: str = Field(description="Unique telemetry tracking identifier")
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    entity_type: str = Field(description="Detected object class: car, motorcycle, bicycle, e-scooter, pedestrian")
    lane_id: str = Field(description="Designated intersection lane identifier")
    lane_legal_heading: float = Field(ge=0.0, lt=360.0, description="Legal flow azimuth angle in degrees [0, 360)")
    observed_heading: float = Field(ge=0.0, lt=360.0, description="Observed movement azimuth angle in degrees [0, 360)")
    estimated_speed_kmh: float = Field(ge=0.0, description="Calculated metric ground speed in km/h")
    speed_limit_kmh: float = Field(ge=0.0, default=25.0, description="Posted lane or intersection speed limit")
    trajectory_points: List[Tuple[float, float]] = Field(default_factory=list, description="Sequence of (x, y) ground plane coordinates")
    dwell_time_seconds: float = Field(ge=0.0, default=1.0, description="Observed time duration within intersection zone")
    pedestrian_zone: bool = Field(default=False, description="Flag indicating if lane is restricted sidewalk or crosswalk")

    @field_validator("entity_type")
    @classmethod
    def validate_entity_type(cls, v: str) -> str:
        canonical_names = set(CANONICAL_CLASSES.values())
        if v.lower() not in canonical_names:
            raise ValueError(f"Entity type '{v}' not recognized. Must be one of {sorted(canonical_names)}")
        return v.lower()

    def format_telemetry_prompt(self) -> str:
        """Format raw telemetry into a structured text prompt for the auditor LLM."""
        angular_delta = abs(self.observed_heading - self.lane_legal_heading)
        if angular_delta > 180.0:
            angular_delta = 360.0 - angular_delta

        return (
            f"[TELEMETRY DATA]\n"
            f"Incident ID: {self.incident_id}\n"
            f"Timestamp: {self.timestamp}\n"
            f"Entity: {self.entity_type}\n"
            f"Lane: {self.lane_id} (Legal Heading: {self.lane_legal_heading:.1f}°, Angular Delta: {angular_delta:.1f}°)\n"
            f"Observed Heading: {self.observed_heading:.1f}°\n"
            f"Estimated Speed: {self.estimated_speed_kmh:.1f} km/h (Speed Limit: {self.speed_limit_kmh:.1f} km/h)\n"
            f"Dwell Time: {self.dwell_time_seconds:.1f}s, Trajectory Points: {len(self.trajectory_points)}\n"
            f"Pedestrian Zone: {'Yes' if self.pedestrian_zone else 'No'}"
        )


class IncidentAuditVerdict(BaseModel):
    """Structured forensic reasoning output from the auditor LLM (R8, R17)."""

    incident_id: str = Field(description="Unique telemetry identifier matching the incident")
    timestamp: str = Field(description="UTC timestamp of the incident analysis")
    entity_type: str = Field(description="Audited object class: car, motorcycle, bicycle, e-scooter, pedestrian")
    lane_id: str = Field(description="Intersection lane identifier")
    is_wrong_way: bool = Field(description="Flag indicating counter-flow traffic infraction")
    heading_degrees: float = Field(ge=0.0, lt=360.0, description="Audited heading direction in degrees [0, 360)")
    estimated_speed_kmh: float = Field(ge=0.0, description="Audited ground speed in km/h")
    speed_limit_kmh: float = Field(ge=0.0, description="Reference speed limit in km/h")
    risk_level: RiskLevel = Field(description="Categorical risk rating: low, medium, high, critical")
    violation_detected: bool = Field(description="Boolean flag indicating whether any infraction occurred")
    summary: str = Field(max_length=300, description="Concise forensic summary (<= 300 characters)")
    reasoning: str = Field(description="Detailed step-by-step forensic deduction")

    @field_validator("entity_type")
    @classmethod
    def validate_entity_type(cls, v: str) -> str:
        canonical_names = set(CANONICAL_CLASSES.values())
        if v.lower() not in canonical_names:
            raise ValueError(f"Entity type '{v}' not recognized. Must be one of {sorted(canonical_names)}")
        return v.lower()


class AuditorTrainingSample(BaseModel):
    """Training pair combining prompt input and structured JSON target for fine-tuning (R11)."""

    sample_id: str = Field(description="Unique training sample identifier")
    system_prompt: str = Field(description="System instruction defining role and JSON schema constraints")
    user_prompt: str = Field(description="Formatted telemetry prompt")
    ground_truth_verdict: IncidentAuditVerdict = Field(description="Ground truth forensic audit verdict")
    telemetry: TrafficTelemetryRecord = Field(description="Underlying raw telemetry record")

    def to_qwen_chat_format(self) -> Dict[str, Any]:
        """Convert sample into standard conversational chat format with system, user, assistant messages."""
        return {
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": self.user_prompt},
                {"role": "assistant", "content": self.ground_truth_verdict.model_dump_json(indent=2)},
            ]
        }

    def to_chat_completion_string(self) -> str:
        """Serialize sample into standard ChatML text format."""
        verdict_json = self.ground_truth_verdict.model_dump_json(indent=2)
        return (
            f"<|im_start|>system\n{self.system_prompt}<|im_end|>\n"
            f"<|im_start|>user\n{self.user_prompt}<|im_end|>\n"
            f"<|im_start|>assistant\n{verdict_json}<|im_end|>\n"
        )


class GGUFConversionMetadata(BaseModel):
    """Metadata tracking GGUF conversion parameters and exported model specs (R14, R16)."""

    model_name: str = Field(description="Name or ID of fine-tuned auditor model")
    architecture: str = Field(default="qwen2", description="GGUF architecture identifier (e.g. qwen2)")
    context_length: int = Field(default=2048, gt=0, description="Maximum context sequence length in tokens")
    quantization_type: str = Field(default="F16", description="GGUF quantization format (F16, BF16, Q4_K_M)")
    source_hf_dir: str = Field(description="Path to standalone merged Hugging Face model directory")
    output_gguf_path: str = Field(description="Path to exported GGUF artifact")
    parameter_count: int = Field(gt=0, description="Total active model parameter count")
    chat_template: str = Field(default="chatml", description="Chat template preserved in GGUF metadata")
    converted_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


# =============================================================================
# OpenVINO NNCF INT8 Quantization Schemas (SVA-QUANT-001: R11, R12, R15, R16)
# =============================================================================


class QuantizationConfig(BaseModel):
    """Configuration for OpenVINO NNCF Post-Training Quantization (R5, R15)."""

    model_input_path: str = Field(
        default="models/distilled/yolo11n_distilled.onnx",
        description="Path to baseline FP32 ONNX or PyTorch student detector",
    )
    output_dir: str = Field(
        default="models/quantized/openvino",
        description="Target directory for exported OpenVINO IR (.xml / .bin)",
    )
    target_device: str = Field(
        default="CPU",
        description="Target execution device for quantized IR (CPU, GPU, NPU)",
    )
    subset_size: int = Field(
        default=300,
        gt=0,
        description="Number of calibration frames (R6)",
    )
    model_type: str = Field(
        default="detector",
        description="Model architecture hint for NNCF quantization algorithms",
    )
    fast_bias_correction: bool = Field(
        default=True,
        description="Enable fast bias correction to recover activation shift",
    )
    preset: str = Field(
        default="PERFORMANCE",
        description="NNCF QuantizationPreset: PERFORMANCE (INT8 sym/asym) or MIXED",
    )
    min_reduction_pct: float = Field(
        default=60.0,
        ge=0.0,
        le=100.0,
        description="Minimum mandatory model size reduction percentage (R11)",
    )
    max_map_drop: float = Field(
        default=0.015,
        ge=0.0,
        description="Maximum permissible absolute mAP@50-95 accuracy drop (R12)",
    )


class CalibrationDatasetManifest(BaseModel):
    """Catalog manifest recording the 300-frame calibration partition (R6, R7, R15)."""

    total_frames: int = Field(default=300, ge=1, description="Total calibration frame count (R6)")
    day_frames: int = Field(default=120, ge=0, description="Daylight high-visibility frames (40%)")
    dusk_frames: int = Field(default=90, ge=0, description="Dusk / golden-hour transition frames (30%)")
    night_frames: int = Field(default=90, ge=0, description="Night / low-lux artificial light frames (30%)")
    input_resolution: List[int] = Field(default_factory=lambda: [1, 3, 640, 640], description="Tensor shape [B, C, H, W]")
    frame_paths: List[str] = Field(default_factory=list, description="Ordered paths to sampled calibration images")
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class QuantizationReport(BaseModel):
    """Post-quantization verification and audit report (R11, R12, R15, R17, R18)."""

    model_name: str = Field(default="yolo11n_int8", description="Quantized model identifier")
    baseline_fp32_size_bytes: int = Field(gt=0, description="Byte size of input FP32 model")
    quantized_int8_size_bytes: int = Field(gt=0, description="Aggregate byte size of exported IR (.xml + .bin)")
    size_reduction_pct: float = Field(ge=0.0, le=100.0, description="Model size reduction percentage")
    achieved_size_reduction: bool = Field(description="True if size_reduction_pct >= 60.0% (R11)")
    baseline_map_50_95: float = Field(ge=0.0, le=1.0, description="Baseline FP32 detector mAP@50-95")
    quantized_map_50_95: float = Field(ge=0.0, le=1.0, description="Quantized INT8 detector mAP@50-95")
    map_drop: float = Field(description="Absolute mAP@50-95 drop (baseline - quantized)")
    achieved_accuracy_tolerance: bool = Field(description="True if map_drop <= 0.015 (R12)")
    benchmark_fps: float = Field(ge=0.0, description="Measured laptop CPU throughput in FPS (R17)")
    benchmark_latency_ms: float = Field(ge=0.0, description="Mean inference latency per 640x640 frame in ms")
    peak_rss_mb: float = Field(ge=0.0, description="Peak resident set size memory in MB (R18)")
    within_memory_budget: bool = Field(description="True if peak_rss_mb < 1200.0 MB (R18)")
    ir_xml_path: str = Field(description="Filesystem path to exported OpenVINO IR XML topology")
    ir_bin_path: str = Field(description="Filesystem path to exported OpenVINO IR BIN binary weights")
    completed_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


# =============================================================================
# llama.cpp GGUF Quantization & CPU Benchmarking Schemas (SVA-QUANT-002: R15, R22)
# =============================================================================


class GGUFQuantizationConfig(BaseModel):
    """Configuration for GGUF model quantization via llama-quantize (R1, R2, R5, R22)."""

    input_gguf_path: str = Field(
        default="models/auditor/qwen2.5-0.5b-auditor-f16.gguf",
        description="Path to baseline F16 GGUF auditor model",
    )
    output_gguf_path: str = Field(
        default="models/quantized/llamacpp/qwen2.5-0.5b-auditor-q4_k_m.gguf",
        description="Path to exported quantized GGUF model artifact",
    )
    quantization_type: str = Field(
        default="Q4_K_M",
        description="Quantization format: Q4_K_M, Q4_K_S, Q5_K_M, Q8_0",
    )
    min_reduction_pct: float = Field(
        default=55.0,
        ge=0.0,
        le=100.0,
        description="Minimum mandatory model size reduction percentage (R3)",
    )


class AuditorBenchmarkConfig(BaseModel):
    """Execution parameters for laptop CPU auditor benchmark (R7, R12, R22)."""

    model_path: str = Field(
        default="models/quantized/llamacpp/qwen2.5-0.5b-auditor-q4_k_m.gguf",
        description="Path to quantized GGUF model",
    )
    n_threads: int = Field(default=4, gt=0, description="CPU worker thread count (R7)")
    n_ctx: int = Field(default=2048, gt=0, description="Context window length (R7)")
    warmup_iterations: int = Field(default=2, ge=0, description="Warm-up inference runs")
    benchmark_iterations: int = Field(default=10, gt=0, description="Measured benchmark iterations")
    max_latency_seconds: float = Field(default=3.5, gt=0.0, description="Permissible latency ceiling (R13)")
    max_rss_mb: float = Field(default=1200.0, gt=0.0, description="Permissible peak RSS memory ceiling (R14)")
    mock_mode: bool = Field(default=False, description="Operate in offline mock execution mode (R11)")


class AuditorCaseResult(BaseModel):
    """Evaluation result for an individual telemetry case (R18, R19, R21, R22)."""

    case_id: str = Field(description="Unique telemetry case identifier")
    parse_success: bool = Field(description="True if LLM completion parsed into valid IncidentAuditVerdict")
    is_wrong_way_expected: bool = Field(description="Ground truth counter-flow flag")
    is_wrong_way_predicted: Optional[bool] = Field(default=None, description="Auditor predicted counter-flow flag")
    risk_level_expected: str = Field(description="Ground truth risk level: low, medium, high, critical")
    risk_level_predicted: Optional[str] = Field(default=None, description="Auditor predicted risk level")
    generation_latency_seconds: float = Field(ge=0.0, description="Per-case inference latency in seconds")
    tokens_generated: int = Field(ge=0, description="Number of tokens generated")
    error_message: Optional[str] = Field(default=None, description="Error details if parsing failed")


class AuditorBenchmarkReport(BaseModel):
    """Consolidated post-quantization CPU benchmark report (R15, R16, R22)."""

    model_name: str = Field(default="qwen2.5-0.5b-auditor-q4_k_m", description="Model profile identifier")
    quantization_type: str = Field(default="Q4_K_M", description="Quantization format")
    baseline_f16_size_bytes: int = Field(gt=0, description="Original F16 GGUF byte size")
    quantized_q4_size_bytes: int = Field(gt=0, description="Quantized GGUF byte size")
    size_reduction_pct: float = Field(ge=0.0, le=100.0, description="Model file size reduction percentage")
    achieved_size_reduction: bool = Field(description="True if size_reduction_pct >= 55.0% (R3)")
    mean_latency_seconds: float = Field(ge=0.0, description="Mean inference latency per report (R13)")
    p95_latency_seconds: float = Field(ge=0.0, description="95th percentile latency in seconds")
    prompt_tokens_per_sec: float = Field(ge=0.0, description="Prompt ingestion speed")
    generation_tokens_per_sec: float = Field(ge=0.0, description="Token generation speed")
    peak_rss_mb: float = Field(ge=0.0, description="Peak resident set size memory in MB (R14)")
    within_memory_budget: bool = Field(description="True if peak_rss_mb <= 1200.0 MB (R14)")
    within_latency_budget: bool = Field(description="True if mean_latency_seconds <= 3.5s (R13)")
    total_cases_evaluated: int = Field(default=100, ge=0, description="Total infraction cases evaluated")
    schema_parse_success_rate: float = Field(ge=0.0, le=100.0, description="Percentage of valid JSON verdicts")
    wrong_way_accuracy_pct: float = Field(ge=0.0, le=100.0, description="Classification accuracy for wrong-way")
    risk_level_accuracy_pct: float = Field(ge=0.0, le=100.0, description="Classification accuracy for risk level")
    hardware_flags: Dict[str, bool] = Field(default_factory=dict, description="Detected CPU features: AVX2, AVX512")
    completed_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())




