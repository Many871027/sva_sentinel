"""OpenVINO and llama.cpp runtime engine wrappers, model registry, distillation, and fine-tuning."""

from src.engine.auditor_finetuning import AuditorFineTuner, LoRAConfigProfile
from src.engine.auditor_verifier import AuditorVerifier
from src.engine.distillation_loss import CompositeDistillationLoss
from src.engine.distillation_trainer import DistillationTrainer
from src.engine.download_base_models import BaseModelDownloader
from src.engine.gguf_converter import GGUFConverter
from src.engine.model_registry import ModelRegistry
from src.engine.onnx_exporter import export_student_to_onnx, verify_onnx_integrity
from src.engine.openvino_benchmark import OpenVINOBenchmarkRunner
from src.engine.openvino_quantizer import OpenVINOQuantizer
from src.engine.llamacpp_quantizer import GGUFQuantizer
from src.engine.auditor_engine import ForensicAuditorEngine, detect_cpu_vector_capabilities
from src.engine.auditor_benchmark import AuditorBenchmarkRunner
from src.engine.auditor_case_evaluator import AuditorCaseEvaluator

__all__ = [
    "BaseModelDownloader",
    "ModelRegistry",
    "CompositeDistillationLoss",
    "DistillationTrainer",
    "export_student_to_onnx",
    "verify_onnx_integrity",
    "LoRAConfigProfile",
    "AuditorFineTuner",
    "GGUFConverter",
    "AuditorVerifier",
    "OpenVINOQuantizer",
    "OpenVINOBenchmarkRunner",
    "GGUFQuantizer",
    "ForensicAuditorEngine",
    "detect_cpu_vector_capabilities",
    "AuditorBenchmarkRunner",
    "AuditorCaseEvaluator",
]
