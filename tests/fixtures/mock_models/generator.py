"""Synthetic miniature mock model fixtures and generator (R15, R16).

Provides deterministic mock model checkpoints and metadata for fast offline testing (<5.0s).
"""

from __future__ import annotations

import hashlib
import json
import struct
from pathlib import Path
from typing import Any, Dict, Tuple

FIXTURES_DIR = Path(__file__).parent


def create_mock_safetensors(
    file_path: Path | str,
    shapes: Dict[str, Tuple[int, ...]] | None = None,
) -> Tuple[Path, str, int]:
    """Create a minimal valid SafeTensors binary file."""
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if shapes is None:
        shapes = {
            "model.embed_tokens.weight": (100, 64),
            "model.layers.0.mlp.gate_proj.weight": (64, 64),
        }

    header_dict: Dict[str, Any] = {}
    current_offset = 0
    total_elements = 0

    for name, shape in shapes.items():
        num_elems = 1
        for dim in shape:
            num_elems *= dim
        total_elements += num_elems
        byte_len = num_elems * 4  # FP32
        header_dict[name] = {
            "dtype": "F32",
            "shape": list(shape),
            "data_offsets": [current_offset, current_offset + byte_len],
        }
        current_offset += byte_len

    header_bytes = json.dumps(header_dict).encode("utf-8")
    header_len = len(header_bytes)

    with open(path, "wb") as f:
        f.write(struct.pack("<Q", header_len))
        f.write(header_bytes)
        # Write dummy FP32 data
        f.write(b"\x00" * current_offset)

    with open(path, "rb") as f:
        sha256 = hashlib.sha256(f.read()).hexdigest()

    return path, sha256, total_elements


def create_mock_pytorch_weights(
    file_path: Path | str,
    parameter_count: int = 2610000,
) -> Tuple[Path, str]:
    """Create a synthetic miniature PyTorch checkpoint representation."""
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Write a structured mock weight manifest
    mock_payload = {
        "format": "pt",
        "parameter_count": parameter_count,
        "mock_layers": ["conv1", "bn1", "backbone", "head"],
        "bytes_padding": "X" * 1024,
    }
    content = json.dumps(mock_payload, indent=2).encode("utf-8")

    with open(path, "wb") as f:
        f.write(content)

    with open(path, "rb") as f:
        sha256 = hashlib.sha256(f.read()).hexdigest()

    return path, sha256


def setup_mock_fixtures(base_dir: Path | str = FIXTURES_DIR) -> Dict[str, Dict[str, Any]]:
    """Initialize synthetic miniature fixtures for all target models."""
    root = Path(base_dir)
    root.mkdir(parents=True, exist_ok=True)

    yolo11n_path, yolo11n_sha = create_mock_pytorch_weights(
        root / "yolo11n.pt", parameter_count=2610000
    )
    yolo11x_path, yolo11x_sha = create_mock_pytorch_weights(
        root / "yolo11x.pt", parameter_count=56900000
    )
    qwen_path, qwen_sha, qwen_params = create_mock_safetensors(
        root / "model.safetensors",
        shapes={"model.embed_tokens.weight": (1000, 490)},
    )

    return {
        "yolo11n": {
            "path": yolo11n_path,
            "sha256": yolo11n_sha,
            "file_name": "yolo11n.pt",
            "parameter_count": 2610000,
        },
        "yolo11x": {
            "path": yolo11x_path,
            "sha256": yolo11x_sha,
            "file_name": "yolo11x.pt",
            "parameter_count": 56900000,
        },
        "qwen2.5-0.5b-instruct": {
            "path": qwen_path,
            "sha256": qwen_sha,
            "file_name": "model.safetensors",
            "parameter_count": qwen_params,
        },
    }
