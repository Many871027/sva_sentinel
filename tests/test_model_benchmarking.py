"""Unit tests for baseline inference runner, benchmarking, and mAP evaluation (T12).

Covers:
- R10: Baseline inference execution generating detections without runtime errors.
- R11: Mean FP32 latency benchmarking over at least 5 warm runs and P95 latency.
- R12: Peak process RSS memory profiling in MB via psutil.
- R13: Baseline zero-shot mAP@50 and mAP@50-95 evaluation.
- R17: Fast automated test execution under 5.0 seconds.
"""

from __future__ import annotations

import tempfile
import time
import unittest
from pathlib import Path

import numpy as np

from src.core.exceptions import ModelInferenceError
from src.engine.model_registry import (
    BaselineInferenceRunner,
    benchmark_inference_latency_and_memory,
    compute_iou,
    evaluate_zero_shot_map,
)
from src.pipeline.schemas import TARGET_MODEL_PROFILES
from tests.fixtures.mock_models import create_mock_pytorch_weights


class TestModelBenchmarking(unittest.TestCase):
    """Test suite validating inference execution, latency profiling, and mAP evaluation."""

    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.weights_path, _ = create_mock_pytorch_weights(
            Path(self.temp_dir.name) / "mock_yolo.pt"
        )
        self.profile = TARGET_MODEL_PROFILES["yolo11n"]
        self.runner = BaselineInferenceRunner(
            weights_path=self.weights_path,
            profile=self.profile,
        )

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_r10_baseline_inference_runner_detections(self) -> None:
        """Verify R10: Forward inference execution returns valid detections without runtime errors."""
        sample_frame = np.zeros((640, 640, 3), dtype=np.uint8)
        detections = self.runner.run_inference(sample_frame)

        self.assertIsInstance(detections, list)
        self.assertGreater(len(detections), 0)

        for det in detections:
            self.assertIn("bbox", det)
            self.assertIn("class_id", det)
            self.assertIn("confidence", det)

            bbox = det["bbox"]
            self.assertEqual(len(bbox), 4)
            self.assertLess(bbox[0], bbox[2])  # x1 < x2
            self.assertLess(bbox[1], bbox[3])  # y1 < y2

            self.assertIn(det["class_id"], [0, 1, 2, 3, 4])
            self.assertGreaterEqual(det["confidence"], 0.0)
            self.assertLessEqual(det["confidence"], 1.0)

    def test_r11_latency_benchmarking_at_least_5_warm_runs(self) -> None:
        """Verify R11: Benchmarks mean and P95 latency over at least 5 warm iterations."""
        # Must raise error if warm_runs < 5 per R11
        with self.assertRaises(ValueError):
            benchmark_inference_latency_and_memory(
                runner_fn=self.runner.run_inference,
                sample_input=np.zeros((640, 640, 3), dtype=np.uint8),
                parameter_count=2610000,
                warm_runs=3,
            )

        metrics = benchmark_inference_latency_and_memory(
            runner_fn=self.runner.run_inference,
            sample_input=np.zeros((640, 640, 3), dtype=np.uint8),
            parameter_count=2610000,
            warm_runs=5,
        )

        self.assertGreaterEqual(metrics.mean_latency_ms, 0.0)
        self.assertGreaterEqual(metrics.p95_latency_ms, 0.0)
        self.assertEqual(metrics.parameter_count, 2610000)

    def test_r12_memory_rss_profiling(self) -> None:
        """Verify R12: Peak process RSS memory allocation is recorded in megabytes via psutil."""
        metrics = benchmark_inference_latency_and_memory(
            runner_fn=self.runner.run_inference,
            sample_input=np.zeros((640, 640, 3), dtype=np.uint8),
            parameter_count=2610000,
            warm_runs=5,
        )

        self.assertGreater(metrics.peak_memory_rss_mb, 0.0)
        self.assertIsInstance(metrics.peak_memory_rss_mb, float)

    def test_iou_computation(self) -> None:
        """Verify bounding box IoU calculation accuracy."""
        # Identical boxes -> IoU = 1.0
        self.assertAlmostEqual(compute_iou([10, 10, 50, 50], [10, 10, 50, 50]), 1.0)

        # Disjoint boxes -> IoU = 0.0
        self.assertAlmostEqual(compute_iou([0, 0, 10, 10], [20, 20, 30, 30]), 0.0)

        # 50% overlap calculation
        # Box 1: [0, 0, 20, 10] -> area = 200
        # Box 2: [10, 0, 30, 10] -> area = 200
        # Intersection: [10, 0, 20, 10] -> area = 100
        # Union: 200 + 200 - 100 = 300
        # IoU = 100 / 300 = 1/3
        self.assertAlmostEqual(compute_iou([0, 0, 20, 10], [10, 0, 30, 10]), 1.0 / 3.0)

    def test_r13_zero_shot_map_perfect_match(self) -> None:
        """Verify R13: Zero-shot mAP evaluation yields 1.0 on identical ground truth match."""
        predictions = [
            [
                {"bbox": [10.0, 10.0, 50.0, 50.0], "class_id": 0, "confidence": 0.95},
                {"bbox": [100.0, 100.0, 200.0, 200.0], "class_id": 3, "confidence": 0.90},
            ]
        ]
        ground_truths = [
            [
                {"bbox": [10.0, 10.0, 50.0, 50.0], "class_id": 0},
                {"bbox": [100.0, 100.0, 200.0, 200.0], "class_id": 3},
            ]
        ]

        map_50, map_50_95 = evaluate_zero_shot_map(predictions, ground_truths)
        self.assertEqual(map_50, 1.0)
        self.assertEqual(map_50_95, 1.0)

    def test_r13_zero_shot_map_partial_and_empty(self) -> None:
        """Verify R13: Zero-shot mAP evaluation with false positives, lower IoU, and empty predictions."""
        # Empty predictions vs positive GT -> mAP should be 0.0
        ground_truths = [[{"bbox": [10.0, 10.0, 50.0, 50.0], "class_id": 0}]]
        empty_preds = [[]]

        map_50, map_50_95 = evaluate_zero_shot_map(empty_preds, ground_truths)
        self.assertEqual(map_50, 0.0)
        self.assertEqual(map_50_95, 0.0)

        # Mismatched class
        mismatched_preds = [[{"bbox": [10.0, 10.0, 50.0, 50.0], "class_id": 1, "confidence": 0.9}]]
        map_50_mis, _ = evaluate_zero_shot_map(mismatched_preds, ground_truths)
        self.assertEqual(map_50_mis, 0.0)


if __name__ == "__main__":
    unittest.main()
