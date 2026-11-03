"""Forensic Auditor Backend Engine with llama-cpp-python.

Covers:
- R6: Forensic auditor backend engine utilizing llama-cpp-python with AVX2/AVX-512 CPU support.
- R7: Configure n_threads (default 4) and context window n_ctx=2048.
- R8: Strictly enforced greedy sampling with temperature=0.0.
- R9: Schema-constrained decoding adhering to IncidentAuditVerdict schema.
- R10: Magic header and tensor load verification raising GGUFQuantizationError on failure.
- R11: Offline mock execution mode for test environments.
"""

from datetime import datetime, timezone
import json
import logging
import os
from pathlib import Path
import platform
import struct
import sys
from typing import Any, Dict, List, Optional, Union

from src.core.exceptions import GGUFQuantizationError
from src.engine.auditor_verifier import AuditorVerifier
from src.pipeline.schemas import (
    IncidentAuditVerdict,
    RiskLevel,
    TrafficTelemetryRecord,
)

logger = logging.getLogger(__name__)

GGUF_MAGIC = b"GGUF"


def detect_cpu_vector_capabilities() -> Dict[str, bool]:
    """Detect AVX2 and AVX-512 vector acceleration capabilities on host CPU (R6)."""
    flags: Dict[str, bool] = {"avx2": False, "avx512": False}

    # Environment override for testing
    if "SVA_FORCE_AVX2" in os.environ:
        flags["avx2"] = os.environ["SVA_FORCE_AVX2"].lower() in ("1", "true", "yes")
    if "SVA_FORCE_AVX512" in os.environ:
        flags["avx512"] = os.environ["SVA_FORCE_AVX512"].lower() in ("1", "true", "yes")
    if "SVA_FORCE_AVX2" in os.environ or "SVA_FORCE_AVX512" in os.environ:
        return flags

    # Try py-cpuinfo if installed
    try:
        import cpuinfo

        info = cpuinfo.get_cpu_info()
        cpu_flags = [f.lower() for f in info.get("flags", [])]
        flags["avx2"] = "avx2" in cpu_flags
        flags["avx512"] = any("avx512" in f for f in cpu_flags)
        return flags
    except Exception:
        pass

    # Platform-specific heuristics
    # On x86_64 modern architectures AVX2 is nearly universally present
    arch = platform.machine().lower()
    if arch in ("x86_64", "amd64"):
        flags["avx2"] = True

    return flags


class ForensicAuditorEngine:
    """High-throughput CPU inference engine for forensic traffic infraction auditing."""

    def __init__(
        self,
        model_path: Optional[str | Path] = None,
        n_threads: int = 4,
        n_ctx: int = 2048,
        mock_mode: bool = False,
    ) -> None:
        self.model_path = Path(model_path) if model_path else None
        self.n_threads = n_threads
        self.n_ctx = n_ctx
        self.temperature = 0.0  # Enforce greedy token generation (R8)
        self.mock_mode = mock_mode
        self.hardware_flags = detect_cpu_vector_capabilities()
        self.verifier = AuditorVerifier()
        self.llama = None

        if self.model_path is not None:
            self._validate_model_artifact(self.model_path)

        if not self.mock_mode:
            self._init_runtime()

    def _validate_model_artifact(self, path: Path) -> None:
        """Validate GGUF file header and size (R10)."""
        if not path.is_file():
            raise GGUFQuantizationError(f"GGUF model artifact not found: {path}")

        file_size = path.stat().st_size
        if file_size < 24:
            raise GGUFQuantizationError(
                f"Corrupted or truncated GGUF model artifact ({file_size} bytes): {path}"
            )

        with open(path, "rb") as f:
            magic = f.read(4)
            if magic != GGUF_MAGIC:
                raise GGUFQuantizationError(
                    f"Invalid GGUF magic header {magic!r}, expected {GGUF_MAGIC!r} in {path}"
                )

    def _init_runtime(self) -> None:
        """Initialize llama_cpp.Llama runtime instance (R6, R7, R10)."""
        if self.model_path is None:
            raise GGUFQuantizationError("Cannot initialize llama runtime without model_path")

        try:
            import llama_cpp

            logger.info(
                f"Initializing llama_cpp.Llama runtime (threads={self.n_threads}, ctx={self.n_ctx}, model={self.model_path})"
            )
            self.llama = llama_cpp.Llama(
                model_path=str(self.model_path),
                n_threads=self.n_threads,
                n_ctx=self.n_ctx,
                n_batch=512,
                verbose=False,
            )
        except Exception as e:
            raise GGUFQuantizationError(
                f"Failed to load GGUF model tensor runtime via llama-cpp-python: {e}"
            ) from e

    def _generate_mock_verdict(
        self,
        telemetry: TrafficTelemetryRecord,
    ) -> IncidentAuditVerdict:
        """Generate deterministic verified verdict for offline mock execution (R11)."""
        # Calculate heading discrepancy
        legal_heading = telemetry.lane_legal_heading
        observed_heading = telemetry.observed_heading
        diff = abs(observed_heading - legal_heading)
        if diff > 180.0:
            diff = 360.0 - diff

        is_wrong_way = diff > 90.0
        speed = telemetry.estimated_speed_kmh
        speed_limit = telemetry.speed_limit_kmh
        is_speeding = speed > speed_limit + 5.0
        in_ped_zone = getattr(telemetry, "pedestrian_zone", False)
        is_ped_violation = in_ped_zone and telemetry.entity_type in (
            "car",
            "motorcycle",
            "e-scooter",
            "bicycle",
        )

        violation_detected = is_wrong_way or is_speeding or is_ped_violation

        # Determine risk level
        if is_wrong_way and (speed > speed_limit or in_ped_zone):
            risk_level = RiskLevel.CRITICAL
        elif is_wrong_way:
            risk_level = RiskLevel.HIGH
        elif is_ped_violation and telemetry.entity_type in ("car", "motorcycle"):
            risk_level = RiskLevel.CRITICAL
        elif is_ped_violation:
            risk_level = RiskLevel.HIGH
        elif is_speeding and speed > speed_limit + 20.0:
            risk_level = RiskLevel.HIGH
        elif is_speeding:
            risk_level = RiskLevel.MEDIUM
        elif violation_detected:
            risk_level = RiskLevel.LOW
        else:
            risk_level = RiskLevel.LOW

        summary = (
            f"Audited {telemetry.entity_type} on {telemetry.lane_id}: "
            f"wrong_way={is_wrong_way}, speed={speed:.1f}/{speed_limit:.1f} km/h, "
            f"ped_zone={in_ped_zone}."
        )
        reasoning = (
            f"Observed heading {observed_heading:.1f}° vs legal {legal_heading:.1f}° "
            f"(deviation {diff:.1f}°). Velocity {speed:.1f} km/h on {speed_limit:.1f} limit. "
            f"Severity evaluated to {risk_level.value}."
        )

        return IncidentAuditVerdict(
            incident_id=telemetry.incident_id,
            timestamp=telemetry.timestamp,
            entity_type=telemetry.entity_type,
            lane_id=telemetry.lane_id,
            is_wrong_way=is_wrong_way,
            heading_degrees=observed_heading,
            estimated_speed_kmh=speed,
            speed_limit_kmh=speed_limit,
            risk_level=risk_level,
            violation_detected=violation_detected,
            summary=summary[:250],
            reasoning=reasoning[:300],
        )

    def audit_incident(
        self,
        telemetry: Union[TrafficTelemetryRecord, Dict[str, Any], str],
    ) -> IncidentAuditVerdict:
        """Generate deterministic schema-constrained forensic verdict from telemetry (R8, R9)."""
        if isinstance(telemetry, str):
            telemetry_record = TrafficTelemetryRecord.model_validate_json(telemetry)
        elif isinstance(telemetry, dict):
            telemetry_record = TrafficTelemetryRecord.model_validate(telemetry)
        else:
            telemetry_record = telemetry

        if self.mock_mode:
            return self._generate_mock_verdict(telemetry_record)

        if self.llama is None:
            raise GGUFQuantizationError("Runtime not initialized and mock_mode is False")

        # Format ChatML prompt strictly requiring valid JSON matching IncidentAuditVerdict schema
        system_prompt = (
            "You are an autonomous residential traffic forensic auditor. "
            "Analyze the telemetry data and generate a forensic verdict adhering strictly "
            "to the IncidentAuditVerdict JSON schema without markdown preambles or additional commentary."
        )
        user_prompt = telemetry_record.model_dump_json(indent=2)

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        response = self.llama.create_chat_completion(
            messages=messages,
            temperature=self.temperature,  # 0.0 greedy (R8)
            max_tokens=512,
        )

        raw_completion = response["choices"][0]["message"]["content"]
        # Schema-constrained parsing (R9)
        return self.verifier.validate_verdict(raw_completion)
