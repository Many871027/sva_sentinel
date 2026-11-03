"""End-to-end integration test for llama.cpp GGUF quantization and CPU benchmarking.

Covers:
- R1-R5: GGUF Q4_K_M quantization with metadata preservation and >= 55% size reduction.
- R6-R11: Forensic auditor engine initialization and schema-constrained decoding.
- R12-R17: CPU execution benchmarking with latency <= 3.5s and RSS <= 1200MB.
- R18-R21: 100-case deterministic infraction evaluation with 100% valid JSON parsing.
- R23: CLI console script execution (--help flag validation).
- R24: Fast offline execution (< 5.0 seconds) without external network egress.
"""

from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest

from src.engine.auditor_benchmark import AuditorBenchmarkRunner
from src.engine.auditor_case_evaluator import AuditorCaseEvaluator
from src.engine.auditor_engine import ForensicAuditorEngine
from src.engine.llamacpp_quantizer import GGUFQuantizer
from src.pipeline.schemas import AuditorBenchmarkConfig, GGUFQuantizationConfig
from tests.fixtures.mock_gguf_quant import create_mock_f16_gguf


class TestLLaMACppQuantIntegration(unittest.TestCase):
    """End-to-end integration suite for full quantization, engine, and benchmark lifecycle."""

    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.test_dir = Path(self.temp_dir.name)
        self.base_f16_path = self.test_dir / "qwen2.5-0.5b-auditor-f16.gguf"
        self.quant_q4_path = self.test_dir / "qwen2.5-0.5b-auditor-q4_k_m.gguf"
        self.report_path = self.test_dir / "auditor_cpu_benchmark_report.json"

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_full_quantization_and_benchmark_lifecycle_under_5_seconds(self) -> None:
        """Run complete E2E lifecycle: mock F16 -> Q4_K_M -> engine -> benchmark -> 100 cases in < 5.0s (R24)."""
        start_time = time.perf_counter()

        # Step 1: Create mock baseline F16 GGUF file
        create_mock_f16_gguf(self.base_f16_path, file_size=200_000)
        self.assertTrue(self.base_f16_path.exists())

        # Step 2: Quantize to compact Q4_K_M GGUF
        quantizer = GGUFQuantizer(llama_quantize_bin=None)
        q_cfg = GGUFQuantizationConfig(
            input_gguf_path=str(self.base_f16_path),
            output_gguf_path=str(self.quant_q4_path),
            quantization_type="Q4_K_M",
            min_reduction_pct=55.0,
        )
        quant_summary = quantizer.quantize(config=q_cfg)
        self.assertTrue(self.quant_q4_path.exists())
        self.assertGreaterEqual(quant_summary["size_reduction_pct"], 55.0)

        # Step 3: Initialize engine
        engine = ForensicAuditorEngine(
            model_path=self.quant_q4_path,
            n_threads=4,
            n_ctx=2048,
            mock_mode=True,
        )
        self.assertEqual(engine.n_threads, 4)
        self.assertEqual(engine.n_ctx, 2048)
        self.assertEqual(engine.temperature, 0.0)

        # Step 4: Run CPU Benchmark
        bench_cfg = AuditorBenchmarkConfig(
            model_path=str(self.quant_q4_path),
            n_threads=4,
            n_ctx=2048,
            warmup_iterations=2,
            benchmark_iterations=10,
            max_latency_seconds=3.5,
            max_rss_mb=1200.0,
            mock_mode=True,
        )
        runner = AuditorBenchmarkRunner(config=bench_cfg, engine=engine)
        bench_report = runner.run_benchmark(
            strict=True,
            output_report_path=self.report_path,
        )
        self.assertTrue(bench_report.within_latency_budget)
        self.assertTrue(bench_report.within_memory_budget)
        self.assertTrue(self.report_path.exists())

        # Step 5: Run 100-Case Deterministic Evaluator
        evaluator = AuditorCaseEvaluator(engine=engine, mock_mode=True)
        eval_summary = evaluator.evaluate()
        self.assertEqual(eval_summary["schema_parse_success_rate"], 100.0)
        self.assertGreaterEqual(eval_summary["wrong_way_accuracy_pct"], 90.0)
        self.assertGreaterEqual(eval_summary["risk_level_accuracy_pct"], 85.0)
        self.assertTrue(eval_summary["within_memory_budget"])

        elapsed = time.perf_counter() - start_time
        print(f"\n[E2E Quantization & Benchmark Timing] Completed in {elapsed:.3f}s")
        self.assertLess(elapsed, 5.0, f"Full integration lifecycle must execute in < 5.0s, took {elapsed:.3f}s (R24)")

    def test_cli_handlers_execution(self) -> None:
        """Verify CLI entrypoint handler modules execute --help without syntax errors (R23)."""
        modules = [
            "src.engine.llamacpp_quantizer",
            "src.engine.auditor_benchmark",
            "src.engine.auditor_case_evaluator",
        ]

        for mod in modules:
            res = subprocess.run(
                [sys.executable, "-m", mod, "--help"],
                capture_output=True,
                text=True,
            )
            self.assertEqual(
                res.returncode,
                0,
                f"CLI module {mod} failed to display --help:\n{res.stderr}",
            )
            self.assertIn("usage:", res.stdout.lower())


if __name__ == "__main__":
    unittest.main()
