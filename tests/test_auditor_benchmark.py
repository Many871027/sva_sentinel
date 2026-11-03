"""Unit tests for AuditorBenchmarkRunner.

Covers:
- R12: Warmup and evaluation iteration execution.
- R13: Latency measurement and <= 3.5s ceiling verification.
- R14: Monitoring process peak RSS memory via psutil (<= 1200.0MB).
- R15: Metrics recording (mean latency, P95, prompt/gen tokens/sec, peak RSS).
- R16: Serialization of AuditorBenchmarkReport to JSON artifact.
- R17: Strict budget failure raising AuditorBenchmarkError.
- R24: Fast offline execution (< 1.0s).
"""

import json
from pathlib import Path
import tempfile
import unittest

from src.core.exceptions import AuditorBenchmarkError
from src.engine.auditor_benchmark import AuditorBenchmarkRunner
from src.engine.auditor_engine import ForensicAuditorEngine
from src.pipeline.schemas import AuditorBenchmarkConfig, AuditorBenchmarkReport, TrafficTelemetryRecord


class TestAuditorBenchmark(unittest.TestCase):
    """Test suite for AuditorBenchmarkRunner."""

    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.test_dir = Path(self.temp_dir.name)
        self.report_path = self.test_dir / "auditor_cpu_benchmark_report.json"
        self.engine = ForensicAuditorEngine(mock_mode=True)
        self.runner = AuditorBenchmarkRunner(engine=self.engine)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_r12_and_r13_and_r14_and_r15_benchmark_metrics(self) -> None:
        """Verify R12, R13, R14, R15: Warmup + iterations, latency <= 3.5s, RSS <= 1200MB, metrics recorded."""
        config = AuditorBenchmarkConfig(
            warmup_iterations=2,
            benchmark_iterations=5,
            max_latency_seconds=3.5,
            max_rss_mb=1200.0,
            mock_mode=True,
        )

        report = self.runner.run_benchmark(
            config=config,
            strict=False,
            output_report_path=self.report_path,
        )

        self.assertIsInstance(report, AuditorBenchmarkReport)
        self.assertLessEqual(report.mean_latency_seconds, 3.5, "Mean latency must be <= 3.5s (R13)")
        self.assertTrue(report.within_latency_budget)
        self.assertLessEqual(report.peak_rss_mb, 1200.0, "Peak RSS must be <= 1200.0MB (R14)")
        self.assertTrue(report.within_memory_budget)
        self.assertGreaterEqual(report.prompt_tokens_per_sec, 0.0)
        self.assertGreaterEqual(report.generation_tokens_per_sec, 0.0)
        self.assertEqual(report.total_cases_evaluated, 5)
        self.assertEqual(report.schema_parse_success_rate, 100.0)

    def test_r16_report_serialization(self) -> None:
        """Verify R16: Serialization of AuditorBenchmarkReport to JSON artifact."""
        config = AuditorBenchmarkConfig(
            warmup_iterations=1,
            benchmark_iterations=2,
            mock_mode=True,
        )

        self.runner.run_benchmark(
            config=config,
            output_report_path=self.report_path,
        )

        self.assertTrue(self.report_path.exists(), "Serialized JSON report artifact must exist")
        with open(self.report_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.assertEqual(data.get("model_name"), "qwen2.5-0.5b-auditor-q4_k_m")
        self.assertEqual(data.get("quantization_type"), "Q4_K_M")
        self.assertIn("mean_latency_seconds", data)
        self.assertIn("peak_rss_mb", data)
        self.assertIn("within_memory_budget", data)
        self.assertIn("within_latency_budget", data)

    def test_r17_strict_memory_budget_violation_raises_error(self) -> None:
        """Verify R17: AuditorBenchmarkError raised in strict mode when memory budget is exceeded."""
        # Enforce an impossible memory ceiling of 0.001 MB
        config = AuditorBenchmarkConfig(
            warmup_iterations=1,
            benchmark_iterations=2,
            max_rss_mb=0.001,
            mock_mode=True,
        )

        with self.assertRaises(AuditorBenchmarkError) as ctx:
            self.runner.run_benchmark(
                config=config,
                strict=True,
                output_report_path=self.report_path,
            )
        self.assertIn("rss memory", str(ctx.exception).lower())

    def test_r17_strict_latency_budget_violation_raises_error(self) -> None:
        """Verify R17: AuditorBenchmarkError raised in strict mode when latency budget is exceeded."""
        # Enforce an impossible latency ceiling of 0.0000001 seconds
        config = AuditorBenchmarkConfig(
            warmup_iterations=1,
            benchmark_iterations=2,
            max_latency_seconds=1e-9,
            mock_mode=True,
        )

        with self.assertRaises(AuditorBenchmarkError) as ctx:
            self.runner.run_benchmark(
                config=config,
                strict=True,
                output_report_path=self.report_path,
            )
        self.assertIn("latency", str(ctx.exception).lower())


if __name__ == "__main__":
    unittest.main()
