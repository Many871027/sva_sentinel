"""Mock telemetry records and test fixtures for zero-egress offline testing (R22)."""

import json
from pathlib import Path
import struct
from typing import Dict, List, Optional
from src.pipeline.schemas import (
    AuditorTrainingSample,
    IncidentAuditVerdict,
    RiskLevel,
    TrafficTelemetryRecord,
)


def create_mock_telemetry_record(
    incident_id: str = "INC-TEST-001",
    entity: str = "car",
    lane_id: str = "Lane-2-WB",
    legal_heading: float = 270.0,
    observed_heading: float = 90.0,
    speed: float = 35.0,
    speed_limit: float = 25.0,
    is_wrong_way: bool = True,
    pedestrian_zone: bool = False,
) -> TrafficTelemetryRecord:
    """Create a valid synthetic telemetry record for testing."""
    return TrafficTelemetryRecord(
        incident_id=incident_id,
        timestamp="2026-11-02T10:00:00Z",
        entity_type=entity,
        lane_id=lane_id,
        lane_legal_heading=legal_heading,
        observed_heading=observed_heading,
        estimated_speed_kmh=speed,
        speed_limit_kmh=speed_limit,
        trajectory_points=[(0.0, 0.0), (1.0, 2.0), (2.0, 4.0)],
        dwell_time_seconds=2.0,
        pedestrian_zone=pedestrian_zone,
    )


def create_mock_verdict(
    incident_id: str = "INC-TEST-001",
    entity: str = "car",
    lane_id: str = "Lane-2-WB",
    is_wrong_way: bool = True,
    speed: float = 35.0,
    speed_limit: float = 25.0,
    risk_level: RiskLevel = RiskLevel.HIGH,
    violation_detected: bool = True,
) -> IncidentAuditVerdict:
    """Create a matching ground truth forensic verdict."""
    return IncidentAuditVerdict(
        incident_id=incident_id,
        timestamp="2026-11-02T10:00:00Z",
        entity_type=entity,
        lane_id=lane_id,
        is_wrong_way=is_wrong_way,
        heading_degrees=90.0,
        estimated_speed_kmh=speed,
        speed_limit_kmh=speed_limit,
        risk_level=risk_level,
        violation_detected=violation_detected,
        summary=f"Vehicle infraction on {lane_id} detected.",
        reasoning="Trajectory and velocity violate residential traffic ordinance.",
    )


def create_mock_hf_model_dir(dir_path: Path) -> Path:
    """Create minimal valid Hugging Face model directory for offline GGUF conversion testing."""
    dir_path.mkdir(parents=True, exist_ok=True)

    config = {
        "architectures": ["Qwen2ForCausalLM"],
        "model_type": "qwen2",
        "hidden_size": 896,
        "max_position_embeddings": 2048,
        "torch_dtype": "float16",
    }
    with open(dir_path / "config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    tok_config = {
        "chat_template": "{% for m in messages %}{{m['content']}}{% endfor %}",
        "tokenizer_class": "Qwen2Tokenizer",
    }
    with open(dir_path / "tokenizer_config.json", "w", encoding="utf-8") as f:
        json.dump(tok_config, f, indent=2)

    with open(dir_path / "tokenizer.json", "w", encoding="utf-8") as f:
        json.dump({"version": "1.0"}, f, indent=2)

    weights_file = dir_path / "model.safetensors"
    if not weights_file.exists():
        header_json = json.dumps({"__metadata__": {"format": "pt"}}).encode("utf-8")
        pad = (8 - (len(header_json) % 8)) % 8
        header_json += b" " * pad
        weights_file.write_bytes(struct.pack("<Q", len(header_json)) + header_json)

    return dir_path
