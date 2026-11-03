"""Mock OpenVINO IR and calibration test fixtures for zero-egress offline testing (R21)."""

from pathlib import Path
from typing import List, Tuple
import numpy as np


def create_mock_openvino_ir(
    dir_path: Path,
    model_name: str = "yolo11n_int8",
    bin_size_bytes: int = 2100000,
) -> Tuple[Path, Path]:
    """Create valid OpenVINO IR XML topology and matching BIN weights file."""
    dir_path.mkdir(parents=True, exist_ok=True)
    xml_path = dir_path / f"{model_name}.xml"
    bin_path = dir_path / f"{model_name}.bin"

    xml_content = (
        f'<?xml version="1.0" ?>\n'
        f'<net name="{model_name}" version="11">\n'
        f'  <layers>\n'
        f'    <layer id="0" name="images" type="Parameter" version="opset1">\n'
        f'      <data element_type="f32" shape="1,3,640,640"/>\n'
        f'      <output>\n'
        f'        <port id="0" precision="FP32" names="images">\n'
        f'          <dim>1</dim><dim>3</dim><dim>640</dim><dim>640</dim>\n'
        f'        </port>\n'
        f'      </output>\n'
        f'    </layer>\n'
        f'    <layer id="1" name="quantized_head" type="Result" version="opset1">\n'
        f'      <input>\n'
        f'        <port id="0" precision="I8"/>\n'
        f'      </input>\n'
        f'    </layer>\n'
        f'  </layers>\n'
        f'  <edges>\n'
        f'    <edge from-layer="0" from-port="0" to-layer="1" to-port="0"/>\n'
        f'  </edges>\n'
        f'</net>\n'
    )
    with open(xml_path, "w", encoding="utf-8") as f:
        f.write(xml_content)

    with open(bin_path, "wb") as f:
        f.write(b"\x00" * bin_size_bytes)

    return xml_path, bin_path


def create_mock_fp32_onnx(file_path: Path, size_bytes: int = 5600000) -> Path:
    """Create simulated baseline FP32 ONNX model file."""
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, "wb") as f:
        # Minimal ONNX magic header + dummy buffer
        f.write(b"\x08\x07\x12\x04ONNX" + b"\x00" * (size_bytes - 8))
    return file_path


def create_mock_calibration_frames(count: int = 10) -> List[np.ndarray]:
    """Create synthetic BGR frames for testing calibration loaders."""
    frames = []
    for i in range(count):
        # Varying luminance
        base_val = 20 + (i * 20) % 220
        frame = np.full((64, 64, 3), base_val, dtype=np.uint8)
        frames.append(frame)
    return frames
