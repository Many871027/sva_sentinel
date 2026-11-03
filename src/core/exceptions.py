"""Domain-specific exceptions for SVA Sentinel platform.

Covers:
- R5: DatasetIngestError for unreachable/invalid Hugging Face datasets.
- R9, R11: AnnotationFormatError for unparseable or corrupted annotation records.
- R12, R17: CurationError for dataset normalization, filtering, or augmentation failures.
"""


class FrameworkError(Exception):
    """Base exception for all SVA Sentinel framework and pipeline failures."""

    def __init__(self, message: str, details: dict | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = details or {}


class DatasetIngestError(FrameworkError):
    """Raised when dataset acquisition, network egress, or cache validation fails."""


class AnnotationFormatError(FrameworkError):
    """Raised when annotation structure parsing encounters malformed or invalid syntax."""


class CurationError(FrameworkError):
    """Raised when bounding box normalization, filtering, or augmentation fails."""


class DatasetSplitError(FrameworkError):
    """Raised when dataset partitioning, class balancing, or manifest generation fails."""


class ModelRegistryError(FrameworkError):
    """Base exception for model catalog and registry persistence operations."""


class ModelChecksumError(ModelRegistryError):
    """Raised when SHA256 checksum verification fails for a downloaded model checkpoint."""


class ModelNotFoundError(ModelRegistryError):
    """Raised when a requested model profile is not found in the catalog."""


class ModelInferenceError(FrameworkError):
    """Raised when baseline model inference execution or benchmark profiling fails."""


class DistillationError(FrameworkError):
    """Base exception for all knowledge distillation pipeline failures."""


class DistillationLossError(DistillationError):
    """Raised when loss evaluates to NaN/Inf or distillation loss computation diverges (R10)."""

    def __init__(self, message: str, loss_stats: dict | None = None) -> None:
        super().__init__(message, details=loss_stats)
        self.loss_stats = loss_stats or {}


class ModelExportError(DistillationError):
    """Raised when student model checkpoint or ONNX export fails validation (R15, R16)."""


# =============================================================================
# Auditor Fine-Tuning & GGUF Conversion Exceptions (SVA-DIST-002: R4, R10, R15)
# =============================================================================


class AuditorFinetuningError(FrameworkError):
    """Base exception for forensic auditor instruction fine-tuning failures."""


class HardwareAccelerationError(AuditorFinetuningError):
    """Raised when fine-tuning is attempted without required GPU acceleration (R4)."""


class TelemetryGenerationError(FrameworkError):
    """Raised when synthetic telemetry generation, distribution, or serialization fails (R9, R10)."""


class GGUFConversionError(FrameworkError):
    """Raised when llama.cpp GGUF conversion or tensor verification fails (R14, R15)."""


class AuditorVerificationError(FrameworkError):
    """Raised when auditor JSON output parsing or forensic accuracy validation fails (R17, R18)."""


# =============================================================================
# OpenVINO NNCF INT8 Quantization Exceptions (SVA-QUANT-001: R4, R10, R14)
# =============================================================================


class QuantizationError(FrameworkError):
    """Base exception for all OpenVINO NNCF quantization and calibration failures."""


class QuantizationInputError(QuantizationError):
    """Raised when the input model file to be quantized is missing or invalid (R4)."""


class CalibrationDataError(QuantizationError):
    """Raised when calibration dataset assembly, lighting partitions, or tensor generation fails (R10)."""


class QuantizationRegressionError(QuantizationError):
    """Raised when quantized model accuracy drop exceeds permissible tolerance (R14)."""


# =============================================================================
# llama.cpp GGUF Quantization & Benchmarking Exceptions (SVA-QUANT-002: R4, R10, R17)
# =============================================================================


class GGUFQuantizationError(FrameworkError):
    """Raised when llama.cpp GGUF quantization (e.g. Q4_K_M) fails or produces corrupt artifacts (R2, R10)."""


class AuditorBenchmarkError(FrameworkError):
    """Raised when CPU benchmarking encounters latency regression or memory budget violations (R17)."""




