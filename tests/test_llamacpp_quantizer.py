"""Unit tests for llama.cpp GGUF Quantizer (Q4_K_M).

Covers:
- R1: Execution of GGUF quantization from base F16 to compact Q4_K_M.
- R2: Q4_K_M quantization format support.
- R3: File size reduction >= 55% and binary size under 450MB.
- R4: QuantizationInputError raised on missing, truncated (< 24 bytes), or invalid magic files.
- R5: Preservation of architecture and tokenizer metadata KV pairs.
- R24: Fast offline execution (< 1.0s) without external dependencies.
"""

from pathlib import Path
import tempfile
import unittest

from src.core.exceptions import GGUFQuantizationError, QuantizationInputError
from src.engine.llamacpp_quantizer import GGUFQuantizer
from src.pipeline.schemas import GGUFQuantizationConfig
from tests.fixtures.mock_gguf_quant import (
    create_mock_corrupted_gguf,
    create_mock_f16_gguf,
)


class TestLLaMACppQuantizer(unittest.TestCase):
    """Test suite for GGUFQuantizer engine."""

    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.test_dir = Path(self.temp_dir.name)
        self.quantizer = GGUFQuantizer(llama_quantize_bin=None)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_r1_and_r2_and_r3_quantize_q4_k_m_size_reduction(self) -> None:
        """Verify R1, R2, R3: GGUF Q4_K_M quantization reduces model size by >= 55%."""
        base_f16 = self.test_dir / "qwen2.5-0.5b-auditor-f16.gguf"
        out_q4 = self.test_dir / "qwen2.5-0.5b-auditor-q4_k_m.gguf"

        # Create 200KB mock F16 file
        create_mock_f16_gguf(base_f16, file_size=200_000)
        base_size = base_f16.stat().st_size

        config = GGUFQuantizationConfig(
            input_gguf_path=str(base_f16),
            output_gguf_path=str(out_q4),
            quantization_type="Q4_K_M",
            min_reduction_pct=55.0,
        )

        res = self.quantizer.quantize(config=config)

        self.assertTrue(out_q4.exists(), "Output quantized GGUF file must exist")
        quant_size = out_q4.stat().st_size

        self.assertGreaterEqual(
            res["size_reduction_pct"],
            55.0,
            f"Size reduction {res['size_reduction_pct']}% must be >= 55.0%",
        )
        self.assertTrue(res["achieved_size_reduction"])
        self.assertLess(quant_size, base_size)
        self.assertLess(quant_size, 450 * 1024 * 1024, "Quantized model must be < 450MB")

    def test_r4_missing_input_file_raises_error(self) -> None:
        """Verify R4: QuantizationInputError raised when base GGUF file does not exist."""
        missing_path = self.test_dir / "non_existent.gguf"
        out_path = self.test_dir / "out.gguf"

        config = GGUFQuantizationConfig(
            input_gguf_path=str(missing_path),
            output_gguf_path=str(out_path),
        )

        with self.assertRaises(QuantizationInputError) as ctx:
            self.quantizer.quantize(config=config)
        self.assertIn("not found", str(ctx.exception).lower())

    def test_r4_truncated_file_raises_error(self) -> None:
        """Verify R4: QuantizationInputError raised when GGUF file is < 24 bytes."""
        trunc_path = self.test_dir / "truncated.gguf"
        create_mock_corrupted_gguf(trunc_path, corruption_type="truncated")
        out_path = self.test_dir / "out.gguf"

        config = GGUFQuantizationConfig(
            input_gguf_path=str(trunc_path),
            output_gguf_path=str(out_path),
        )

        with self.assertRaises(QuantizationInputError) as ctx:
            self.quantizer.quantize(config=config)
        self.assertIn("24 bytes", str(ctx.exception))

    def test_r4_invalid_magic_header_raises_error(self) -> None:
        """Verify R4: QuantizationInputError raised when GGUF file lacks GGUF magic header."""
        bad_magic_path = self.test_dir / "bad_magic.gguf"
        create_mock_corrupted_gguf(bad_magic_path, corruption_type="invalid_magic")
        out_path = self.test_dir / "out.gguf"

        config = GGUFQuantizationConfig(
            input_gguf_path=str(bad_magic_path),
            output_gguf_path=str(out_path),
        )

        with self.assertRaises(QuantizationInputError) as ctx:
            self.quantizer.quantize(config=config)
        self.assertIn("invalid gguf magic", str(ctx.exception).lower())

    def test_r5_metadata_preservation(self) -> None:
        """Verify R5: Architecture name, context length, and chat template are preserved."""
        base_f16 = self.test_dir / "model_f16.gguf"
        out_q4 = self.test_dir / "model_q4.gguf"

        custom_chat_template = "{% for message in messages %}{{ message['content'] }}{% endfor %}"
        create_mock_f16_gguf(
            base_f16,
            file_size=100_000,
            arch="qwen2",
            context_length=2048,
            chat_template=custom_chat_template,
        )

        self.quantizer.quantize(
            input_gguf_path=base_f16,
            output_gguf_path=out_q4,
            quantization_type="Q4_K_M",
        )

        # Inspect exported Q4 artifact metadata
        q4_meta = self.quantizer.read_gguf_metadata(out_q4)
        kv = q4_meta["kv_pairs"]

        self.assertEqual(kv.get("general.architecture"), "qwen2")
        self.assertEqual(kv.get("qwen2.context_length"), 2048)
        self.assertEqual(kv.get("tokenizer.chat_template"), custom_chat_template)
        # Verify quantization file type marker exists
        self.assertIn("general.file_type", kv)

    def test_r3_insufficient_reduction_raises_error(self) -> None:
        """Verify R3: GGUFQuantizationError raised if reduction percentage cannot be satisfied."""
        base_f16 = self.test_dir / "model_f16.gguf"
        out_q4 = self.test_dir / "model_q4.gguf"

        create_mock_f16_gguf(base_f16, file_size=100_000)

        # Require impossible 99.9% reduction
        config = GGUFQuantizationConfig(
            input_gguf_path=str(base_f16),
            output_gguf_path=str(out_q4),
            min_reduction_pct=99.9,
        )

        with self.assertRaises(GGUFQuantizationError) as ctx:
            self.quantizer.quantize(config=config)
        self.assertIn("reduction", str(ctx.exception).lower())


if __name__ == "__main__":
    unittest.main()
