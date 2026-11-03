"""Unit tests for edge CPU benchmark runner, latency/FPS profiling, and RSS memory monitoring.

Covers:
- R17: Edge CPU throughput measurement and target >= 25 FPS verification.
- R18: Process resident set size (RSS) memory ceiling (< 1.2GB / 1200MB).
- R19: Benchmark loop execution (warmup and timed iterations) measuring mean/P95 latency.
- R21: Zero-network sub-second offline execution.
"""

from pathlib import Path
import tempfile
import time
import unittest

from src.core.exceptions import QuantizationInputError
from src.engine.openvino_benchmark import OpenVINOBenchmarkRunner
from tests.fixtures.mock_openvino import create_mock_openvino_ir


class TestOpenVINOBenchmark(unittest.TestCase):
    """Test suite verifying OpenVINO CPU benchmark runner, throughput checks, and memory limits."""

    def setUp(self) -> None:
        self.runner = OpenVINOBenchmarkRunner(
            device_name="CPU",
            warmup_iterations=2,
            benchmark_iterations=10,
            target_fps=25.0,
            max_rss_mb=1200.0,
        )

    def test_r17_and_r19_benchmark_execution_and_fps(self) -> None:
        """Verify R17 & R19: Benchmark measures latency, computes FPS, and asserts >= 25 FPS."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            xml_path, _ = create_mock_openvino_ir(tmp_path, model_name="yolo11n_int8")

            results = self.runner.benchmark_model(xml_path=xml_path, mock_mode=True)

            self.assertEqual(results["model_path"], str(xml_path))
            self.assertEqual(results["device"], "CPU")
            self.assertGreater(results["throughput_fps"], 0.0)
            self.assertTrue(results["achieved_target_fps"])
            self.assertGreater(results["mean_latency_ms"], 0.0)
            self.assertGreater(results["p95_latency_ms"], 0.0)

    def test_r18_memory_monitoring_and_limit(self) -> None:
        """Verify R18: Process peak RSS memory is monitored via psutil and within 1.2GB budget."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            xml_path, _ = create_mock_openvino_ir(tmp_path)

            results = self.runner.benchmark_model(xml_path=xml_path, mock_mode=True)

            self.assertGreater(results["peak_rss_mb"], 0.0)
            self.assertLess(results["peak_rss_mb"], 1200.0)
            self.assertTrue(results["within_memory_budget"])

    def test_missing_xml_or_bin_raises_error(self) -> None:
        """Verify QuantizationInputError raised on missing XML or missing matching BIN."""
        # Nonexistent XML
        with self.assertRaises(QuantizationInputError):
            self.runner.benchmark_model("missing_model.xml")

        # XML exists but BIN is missing
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            orphan_xml = tmp_path / "orphan.xml"
            orphan_xml.write_text("<net></net>")

            with self.assertRaises(QuantizationInputError) as ctx:
                self.runner.benchmark_model(orphan_xml)
            self.assertIn("Matching OpenVINO BIN weights file not found", str(ctx.exception))

    def test_r21_fast_execution(self) -> None:
        """Verify R21: Benchmark execution completes in < 0.5s."""
        start = time.perf_counter()
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            xml_path, _ = create_mock_openvino_ir(tmp_path)
            self.runner.benchmark_model(xml_path=xml_path, mock_mode=True)
        elapsed = time.perf_counter() - start

        self.assertLess(elapsed, 0.5, f"Expected benchmark in < 0.5s, took {elapsed:.3f}s")


if __name__ == "__main__":
    unittest.main()
