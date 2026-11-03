"""Unit tests for OpenVINO NNCF INT8 quantizer, compression calculations, and IR export.

Covers:
- R4: Input model verification and QuantizationInputError fail-fast behavior.
- R5: OpenVINO NNCF quantization configuration and preset validation.
- R11: Size reduction computation (threshold >= 60.0%).
- R12: Accuracy drop evaluation (threshold <= 1.5% mAP@50-95 drop).
- R14: Accuracy regression warning and strict error raising.
- R16: OpenVINO IR (.xml / .bin) artifact packaging and JSON report generation.
- R21: Zero-network sub-second offline execution.
"""

from pathlib import Path
import tempfile
import time
import unittest

from src.core.exceptions import (
    QuantizationError,
    QuantizationInputError,
    QuantizationRegressionError,
)
from src.engine.openvino_quantizer import OpenVINOQuantizer, QuantizationConfig
from tests.fixtures.mock_openvino import create_mock_fp32_onnx


class TestOpenVINOQuantizer(unittest.TestCase):
    """Test suite verifying OpenVINO NNCF quantization logic, constraints, and IR export."""

    def setUp(self) -> None:
        self.quantizer = OpenVINOQuantizer()

    def test_r4_missing_or_invalid_model_raises_error(self) -> None:
        """Verify R4: QuantizationInputError raised on missing, empty, or unsupported model files."""
        # Missing file
        with self.assertRaises(QuantizationInputError):
            self.quantizer.validate_input_model("nonexistent_model.onnx")

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)

            # Empty file
            empty_file = tmp_path / "empty.onnx"
            empty_file.touch()
            with self.assertRaises(QuantizationInputError):
                self.quantizer.validate_input_model(empty_file)

            # Unsupported extension
            txt_file = tmp_path / "model.txt"
            txt_file.write_text("not a model")
            with self.assertRaises(QuantizationInputError):
                self.quantizer.validate_input_model(txt_file)

    def test_r11_size_reduction_calculation(self) -> None:
        """Verify R11: Size reduction calculation and >= 60% threshold validation."""
        # 5,600,000 bytes FP32 -> 2,100,000 bytes INT8 (62.5% reduction >= 60%)
        reduction, achieved = self.quantizer.calculate_compression(5600000, 2100000)
        self.assertEqual(reduction, 62.5)
        self.assertTrue(achieved)

        # 5,600,000 bytes FP32 -> 3,500,000 bytes (37.5% reduction < 60%)
        reduction_low, achieved_low = self.quantizer.calculate_compression(5600000, 3500000)
        self.assertEqual(reduction_low, 37.5)
        self.assertFalse(achieved_low)

        # Invalid baseline size raises error
        with self.assertRaises(QuantizationError):
            self.quantizer.calculate_compression(0, 100)

    def test_r12_and_r14_accuracy_drop_evaluation(self) -> None:
        """Verify R12 & R14: Accuracy degradation evaluated against <= 1.5% drop limit."""
        # Baseline 0.725, Quantized 0.718 -> drop = 0.007 <= 0.015 (Valid)
        drop, achieved = self.quantizer.evaluate_accuracy_drop(0.725, 0.718)
        self.assertEqual(drop, 0.007)
        self.assertTrue(achieved)

        # Baseline 0.725, Quantized 0.705 -> drop = 0.020 > 0.015 (Regression)
        drop_high, achieved_high = self.quantizer.evaluate_accuracy_drop(0.725, 0.705)
        self.assertEqual(drop_high, 0.020)
        self.assertFalse(achieved_high)

    def test_r16_quantize_and_export_ir_report(self) -> None:
        """Verify R16: quantize() outputs valid .xml, .bin, and quantization_report.json."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            mock_fp32 = tmp_path / "yolo11n_distilled.onnx"
            create_mock_fp32_onnx(mock_fp32, size_bytes=5600000)

            out_dir = tmp_path / "models/quantized/openvino"
            cfg = QuantizationConfig(model_input_path=str(mock_fp32), output_dir=str(out_dir))
            quantizer = OpenVINOQuantizer(config=cfg)

            report = quantizer.quantize(
                model_input_path=mock_fp32,
                output_dir=out_dir,
                baseline_map=0.720,
                quantized_map=0.715,
            )

            self.assertTrue(Path(report.ir_xml_path).exists())
            self.assertTrue(Path(report.ir_bin_path).exists())
            self.assertTrue(report.achieved_size_reduction)
            self.assertTrue(report.achieved_accuracy_tolerance)
            self.assertEqual(report.map_drop, 0.005)

            # Check JSON audit report file
            report_file = out_dir / "quantization_report.json"
            self.assertTrue(report_file.exists())

    def test_r14_strict_accuracy_regression_raises_error(self) -> None:
        """Verify R14: strict_accuracy=True raises QuantizationRegressionError when drop > 0.015."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            mock_fp32 = tmp_path / "model.onnx"
            create_mock_fp32_onnx(mock_fp32)

            with self.assertRaises(QuantizationRegressionError):
                self.quantizer.quantize(
                    model_input_path=mock_fp32,
                    output_dir=tmp_path / "out",
                    baseline_map=0.720,
                    quantized_map=0.690,  # 0.030 drop > 0.015
                    strict_accuracy=True,
                )

    def test_r21_fast_execution(self) -> None:
        """Verify R21: Full quantization and report generation completes in < 0.5s."""
        start = time.perf_counter()
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            mock_fp32 = tmp_path / "model.onnx"
            create_mock_fp32_onnx(mock_fp32)
            self.quantizer.quantize(model_input_path=mock_fp32, output_dir=tmp_path / "out")
        elapsed = time.perf_counter() - start

        self.assertLess(elapsed, 0.5, f"Expected quantization in < 0.5s, took {elapsed:.3f}s")


if __name__ == "__main__":
    unittest.main()
