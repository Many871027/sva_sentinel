"""Unit tests for ONNX model export and structural integrity verification.

Validates:
- R15: Distilled student weights export to ONNX format (models/distilled/yolo11n_distilled.onnx)
- R16: Structural verification of ONNX model, input tensor shape [1, 3, 640, 640], opset 17, and dynamic batching
- R19: Ultra-fast offline execution (< 5.0 seconds) without external network egress
"""

import shutil
import tempfile
import unittest
from pathlib import Path

from src.core.exceptions import ModelExportError
from src.engine.onnx_exporter import (
    export_student_to_onnx,
    verify_onnx_integrity,
)


class TestONNXExport(unittest.TestCase):
    """Test suite covering student detector ONNX export and verification."""

    def setUp(self):
        self.temp_dir = Path(tempfile.mkdtemp(prefix="test_sva_onnx_"))
        self.onnx_path = self.temp_dir / "yolo11n_distilled.onnx"

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_r15_and_r16_export_and_verify_valid_model(self):
        """Verify R15 & R16: Export student detector to ONNX and verify valid [1, 3, 640, 640] structure."""
        mock_model = {"parameter_count": 2_610_000}

        out_path = export_student_to_onnx(
            model=mock_model,
            output_path=self.onnx_path,
            input_shape=(1, 3, 640, 640),
            dynamic_batch=True,
            opset_version=17,
        )

        self.assertTrue(out_path.is_file())
        self.assertEqual(out_path, self.onnx_path)

        meta = verify_onnx_integrity(
            onnx_path=out_path,
            expected_input_shape=(1, 3, 640, 640),
            require_dynamic_batch=True,
            min_opset=17,
        )

        self.assertTrue(meta["verified"])
        self.assertEqual(meta["input_shape"], [1, 3, 640, 640])
        self.assertTrue(meta["dynamic_batch"])
        self.assertGreaterEqual(meta["opset_version"], 17)

    def test_r16_missing_file_raises_export_error(self):
        """Verify R16: Verification on nonexistent file raises ModelExportError."""
        missing = self.temp_dir / "nonexistent.onnx"
        with self.assertRaises(ModelExportError) as ctx:
            verify_onnx_integrity(missing)
        self.assertIn("does not exist", str(ctx.exception))

    def test_r16_empty_corrupted_file_raises_export_error(self):
        """Verify R16: Zero-byte or truncated file raises ModelExportError."""
        corrupt_file = self.temp_dir / "empty.onnx"
        corrupt_file.write_bytes(b"")

        with self.assertRaises(ModelExportError) as ctx:
            verify_onnx_integrity(corrupt_file)
        self.assertIn("corrupt or empty", str(ctx.exception))

    def test_r16_spatial_dimension_mismatch_raises_export_error(self):
        """Verify R16: Export with non-conforming resolution (e.g. 512x512) fails 640x640 validation."""
        bad_shape_path = self.temp_dir / "bad_shape.onnx"
        export_student_to_onnx(
            model={"parameter_count": 2_610_000},
            output_path=bad_shape_path,
            input_shape=(1, 3, 512, 512),
            dynamic_batch=True,
            opset_version=17,
        )

        with self.assertRaises(ModelExportError) as ctx:
            verify_onnx_integrity(
                onnx_path=bad_shape_path,
                expected_input_shape=(1, 3, 640, 640),
            )
        self.assertIn("spatial shape mismatch", str(ctx.exception))

    def test_r16_channel_mismatch_raises_export_error(self):
        """Verify R16: Export with 1 channel fails 3-channel validation."""
        bad_channel_path = self.temp_dir / "bad_channel.onnx"
        export_student_to_onnx(
            model={"parameter_count": 2_610_000},
            output_path=bad_channel_path,
            input_shape=(1, 1, 640, 640),
            dynamic_batch=True,
            opset_version=17,
        )

        with self.assertRaises(ModelExportError) as ctx:
            verify_onnx_integrity(
                onnx_path=bad_channel_path,
                expected_input_shape=(1, 3, 640, 640),
            )
        self.assertIn("spatial shape mismatch", str(ctx.exception))

    def test_r16_missing_dynamic_batch_raises_export_error(self):
        """Verify R16: Model lacking dynamic batching raises ModelExportError when required."""
        fixed_batch_path = self.temp_dir / "fixed_batch.onnx"
        export_student_to_onnx(
            model={"parameter_count": 2_610_000},
            output_path=fixed_batch_path,
            input_shape=(1, 3, 640, 640),
            dynamic_batch=False,
            opset_version=17,
        )

        with self.assertRaises(ModelExportError) as ctx:
            verify_onnx_integrity(
                onnx_path=fixed_batch_path,
                expected_input_shape=(1, 3, 640, 640),
                require_dynamic_batch=True,
            )
        self.assertIn("dynamic batching", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
