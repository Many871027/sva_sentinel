"""Unit tests for calibration frame loader, illumination partitioning, and NNCF dataset adapter.

Covers:
- R6: Assembly of 300 representative calibration frames.
- R7: Balanced illumination partitions (120 day / 40%, 90 dusk / 30%, 90 night / 30%).
- R8: Image normalization to float32 [1, 3, 640, 640] in [0.0, 1.0].
- R9: NNCF Dataset adapter generator yielding input dictionary.
- R10: Corrupted and unparseable frame skip and replacement behavior.
- R21: Zero-network sub-second offline execution.
"""

from pathlib import Path
import tempfile
import time
import unittest

import numpy as np

from src.pipeline.calibration_loader import (
    CalibrationFrameLoader,
    classify_frame_illumination,
    compute_frame_luminance,
)
from tests.fixtures.mock_openvino import create_mock_calibration_frames


class TestCalibrationLoader(unittest.TestCase):
    """Test suite verifying calibration frame loading, illumination balancing, and tensor preprocessing."""

    def setUp(self) -> None:
        self.loader = CalibrationFrameLoader(target_frames=300, seed=42)

    def test_luminance_calculation_and_classification(self) -> None:
        """Verify luminance calculation and partition thresholds."""
        # Bright day frame
        day_img = np.full((100, 100, 3), 150, dtype=np.uint8)
        day_luma = compute_frame_luminance(day_img)
        self.assertAlmostEqual(day_luma, 150.0, delta=1.0)
        self.assertEqual(classify_frame_illumination(day_luma), "day")

        # Dusk frame
        dusk_img = np.full((100, 100, 3), 65, dtype=np.uint8)
        dusk_luma = compute_frame_luminance(dusk_img)
        self.assertEqual(classify_frame_illumination(dusk_luma), "dusk")

        # Night frame
        night_img = np.full((100, 100, 3), 20, dtype=np.uint8)
        night_luma = compute_frame_luminance(night_img)
        self.assertEqual(classify_frame_illumination(night_luma), "night")

    def test_r6_and_r7_assemble_300_frames_balanced_partitions(self) -> None:
        """Verify R6 & R7: Assembly of 300 frames with 40% Day, 30% Dusk, 30% Night."""
        frames, manifest = self.loader.assemble_calibration_frames()

        self.assertEqual(len(frames), 300)
        self.assertEqual(manifest.total_frames, 300)
        self.assertEqual(manifest.day_frames, 120)  # 40%
        self.assertEqual(manifest.dusk_frames, 90)   # 30%
        self.assertEqual(manifest.night_frames, 90)  # 30%
        self.assertEqual(manifest.input_resolution, [1, 3, 640, 640])

    def test_r8_preprocess_normalization_and_shape(self) -> None:
        """Verify R8: Preprocessed frame has shape [1, 3, 640, 640], float32, in [0.0, 1.0]."""
        raw_img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        tensor = self.loader.preprocess_image(raw_img)

        self.assertEqual(tensor.shape, (1, 3, 640, 640))
        self.assertEqual(tensor.dtype, np.float32)
        self.assertGreaterEqual(float(tensor.min()), 0.0)
        self.assertLessEqual(float(tensor.max()), 1.0)

    def test_r9_nncf_dataset_adapter(self) -> None:
        """Verify R9: create_nncf_dataset yields dictionaries with 'images' key and [1, 3, 640, 640] tensor."""
        mock_frames = create_mock_calibration_frames(5)
        dataset = self.loader.create_nncf_dataset(mock_frames, input_name="images")

        count = 0
        for item in dataset:
            self.assertIn("images", item)
            self.assertEqual(item["images"].shape, (1, 3, 640, 640))
            count += 1

        self.assertEqual(count, 5)

    def test_r10_corrupted_frame_handling(self) -> None:
        """Verify R10: Empty or corrupted file paths are skipped and replaced automatically."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            empty_file = tmp_path / "empty_frame.jpg"
            empty_file.touch()  # 0 bytes

            # Supply 1 corrupted path
            frames, manifest = self.loader.assemble_calibration_frames(candidate_paths=[empty_file])
            # Must still deliver 300 valid frames
            self.assertEqual(len(frames), 300)
            self.assertEqual(manifest.total_frames, 300)

    def test_r21_fast_execution(self) -> None:
        """Verify R21: Assembling 300 frames and preprocessing 10 takes < 1.0s."""
        start = time.perf_counter()
        frames, _ = self.loader.assemble_calibration_frames()
        for f in frames[:10]:
            _ = self.loader.preprocess_image(f)
        elapsed = time.perf_counter() - start

        self.assertLess(elapsed, 1.0, f"Expected assembly in < 1.0s, took {elapsed:.3f}s")


if __name__ == "__main__":
    unittest.main()
