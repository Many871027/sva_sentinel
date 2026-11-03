"""Unit tests for llama.cpp GGUF converter, binary headers, and metadata preservation.

Covers:
- R14: Hugging Face to GGUF binary artifact generation.
- R15: GGUFConversionError raised when mandatory tokenizer/model configuration files are missing.
- R16: Metadata preservation (general.architecture=qwen2, context_length=2048, chat_template).
- R22: Zero-network sub-second offline execution.
"""

from pathlib import Path
import tempfile
import time
import unittest

from src.core.exceptions import GGUFConversionError
from src.engine.gguf_converter import GGUFConverter
from tests.fixtures.mock_telemetry import create_mock_hf_model_dir


class TestGGUFConversion(unittest.TestCase):
    """Test suite verifying GGUF binary generation, magic headers, and metadata validation."""

    def test_r14_and_r16_valid_conversion_and_metadata(self) -> None:
        """Verify R14 & R16: GGUF converter writes valid magic header and preserves metadata."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            hf_dir = tmp_path / "hf_model"
            gguf_file = tmp_path / "qwen2.5-0.5b-auditor-f16.gguf"

            create_mock_hf_model_dir(hf_dir)
            converter = GGUFConverter()
            meta = converter.generate_gguf_artifact(hf_dir=hf_dir, output_path=gguf_file)

            self.assertTrue(gguf_file.exists())
            self.assertEqual(meta.architecture, "qwen2")
            self.assertEqual(meta.context_length, 2048)
            self.assertEqual(meta.quantization_type, "F16")

            # Verify binary structure
            info = converter.verify_gguf(gguf_file)
            self.assertTrue(info["valid"])
            self.assertEqual(info["magic"], "GGUF")
            self.assertEqual(info["version"], 3)
            self.assertGreater(info["kv_count"], 0)

            keys = info["metadata_keys"]
            self.assertEqual(keys.get("general.architecture"), "qwen2")
            self.assertEqual(keys.get("general.name"), "qwen2.5-0.5b-auditor")
            self.assertEqual(keys.get("qwen2.context_length"), 2048)

    def test_r15_missing_tokenizer_files_raise_error(self) -> None:
        """Verify R15: Missing tokenizer.json or config.json raises GGUFConversionError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            hf_dir = tmp_path / "incomplete_model"
            hf_dir.mkdir(parents=True, exist_ok=True)
            # Only create config.json, omit tokenizer files
            (hf_dir / "config.json").write_text('{"model_type": "qwen2"}', encoding="utf-8")

            converter = GGUFConverter()
            with self.assertRaises(GGUFConversionError) as ctx:
                converter.generate_gguf_artifact(hf_dir=hf_dir, output_path=tmp_path / "out.gguf")

            self.assertIn("Missing required Hugging Face tokenizer/model configuration files", str(ctx.exception))
            self.assertIn("tokenizer.json", str(ctx.exception))

    def test_verify_corrupted_or_nonexistent_file(self) -> None:
        """Verify GGUF verification fails on missing or corrupted binary files."""
        converter = GGUFConverter()
        with self.assertRaises(GGUFConversionError):
            converter.verify_gguf("nonexistent_model.gguf")

        with tempfile.TemporaryDirectory() as tmpdir:
            bad_file = Path(tmpdir) / "corrupt.gguf"
            bad_file.write_bytes(b"NOT_A_VALID_GGUF_HEADER_DATA")

            with self.assertRaises(GGUFConversionError) as ctx:
                converter.verify_gguf(bad_file)
            self.assertIn("Invalid GGUF magic header", str(ctx.exception))

    def test_r22_fast_execution(self) -> None:
        """Verify R22: GGUF conversion and metadata validation completes in < 0.5s."""
        start = time.perf_counter()
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            hf_dir = tmp_path / "hf"
            gguf_path = tmp_path / "model.gguf"

            create_mock_hf_model_dir(hf_dir)
            converter = GGUFConverter()
            converter.generate_gguf_artifact(hf_dir, gguf_path)
            converter.verify_gguf(gguf_path)

        elapsed = time.perf_counter() - start
        self.assertLess(elapsed, 0.5, f"Expected conversion in < 0.5s, took {elapsed:.3f}s")


if __name__ == "__main__":
    unittest.main()
