"""Mock GGUF model fixtures and synthetic binary generator for offline testing (R24).

Covers:
- R1: Generation of synthetic base F16 GGUF model artifacts.
- R4: Generation of corrupt/truncated GGUF files to test fail-fast error handling.
- R5: Preservation of architecture and tokenizer metadata KV pairs.
"""

from pathlib import Path
import struct
from typing import Dict, Optional

GGUF_MAGIC = b"GGUF"
GGUF_VERSION = 3


def write_gguf_str(s: str) -> bytes:
    encoded = s.encode("utf-8")
    return struct.pack("<Q", len(encoded)) + encoded


def write_gguf_kv(key: str, val_type: int, raw_val: bytes) -> bytes:
    return write_gguf_str(key) + struct.pack("<I", val_type) + raw_val


def create_mock_f16_gguf(
    path: Path,
    file_size: int = 100_000,
    arch: str = "qwen2",
    context_length: int = 2048,
    chat_template: str = "{% for m in messages %}{{m['content']}}{% endfor %}",
) -> Path:
    """Create a synthetically valid GGUF F16 model file with metadata and padding."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    kv_pairs = [
        ("general.architecture", 8, write_gguf_str(arch)),  # 8 = STRING
        ("general.name", 8, write_gguf_str("qwen2.5-0.5b-auditor")),
        (f"{arch}.context_length", 4, struct.pack("<I", context_length)),  # 4 = UINT32
        ("general.quantization_version", 4, struct.pack("<I", 2)),
        ("tokenizer.chat_template", 8, write_gguf_str(chat_template)),
    ]

    tensor_count = 1
    kv_count = len(kv_pairs)

    header = (
        GGUF_MAGIC
        + struct.pack("<I", GGUF_VERSION)
        + struct.pack("<Q", tensor_count)
        + struct.pack("<Q", kv_count)
    )

    kv_bytes = bytearray()
    for k, v_type, v_data in kv_pairs:
        kv_bytes.extend(write_gguf_kv(k, v_type, v_data))

    header_and_kv = header + bytes(kv_bytes)

    target_size = max(len(header_and_kv) + 64, file_size)

    with open(path, "wb") as f:
        f.write(header_and_kv)
        remaining = target_size - len(header_and_kv)
        if remaining > 0:
            chunk = b"\x01" * 4096
            while remaining > 0:
                to_write = min(remaining, 4096)
                f.write(chunk[:to_write])
                remaining -= to_write

    return path


def create_mock_corrupted_gguf(
    path: Path,
    corruption_type: str = "invalid_magic",
) -> Path:
    """Create intentionally corrupted or truncated GGUF files to verify error handling."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if corruption_type == "truncated":
        # Less than 24 bytes (R4)
        path.write_bytes(b"GGUF\x03\x00\x00\x00")
    elif corruption_type == "invalid_magic":
        # Magic is not 'GGUF' (R4, R10), but size >= 24 bytes
        path.write_bytes(b"BADG\x03\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00")
    elif corruption_type == "unsupported_version":
        # Unsupported version 99, size >= 24 bytes
        path.write_bytes(b"GGUF" + struct.pack("<I", 99) + struct.pack("<Q", 0) + struct.pack("<Q", 0) + b"\x00" * 8)
    else:
        path.write_bytes(b"\x00" * 32)

    return path
