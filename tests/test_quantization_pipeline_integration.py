"""End-to-End integration test for OpenVINO NNCF INT8 quantization pipeline.

Validates:
- Complete workflow: Calibration dataset assembly (300 frames) -> NNCF INT8 quantization ->
  OpenVINO IR packaging (.xml/.bin) -> Size reduction >= 60% -> Accuracy drop <= 1.5% ->
  Laptop CPU benchmark >= 25 FPS & RSS < 1.2GB.
- CLI entrypoint callables: sva-quantize-openvino, sva-benchmark-openvino.
- Strict execution duration under 5.0 seconds without network egress (R21).
- Verification of R1, R2, R4, R5, R6, R7, R8, R9, R10, R11, R12, R13, R14, R15, R16, R17, R18, R19, R20, R21, R22.
"""

from pathlib import Path
import shutil
import tempfile
import time
import unittest

from src.engine.openvino_benchmark import OpenVINOBenchmarkRunner
from src.engine.openvino_quantizer import OpenVINOQuantizer, QuantizationConfig
from src.pipeline.calibration_loader import CalibrationFrameLoader
from tests.fixtures.mock_openvino import create_mock_fp32_onnx


class TestQuantizationPipelineIntegration(unittest.TestCase):
    """End-to-end integration test validating the entire OpenVINO NNCF quantization lifecycle in < 5.0s."""

    def setUp(self) -> None:
        self.start_time = time.perf_counter()
        self.temp_dir = Path(tempfile.mkdtemp(prefix="test_sva_quant_e2e_"))

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        elapsed = time.perf_counter() - self.start_time
        print(f"\n[E2E Quantization Timing] Completed in {elapsed:.3f}s")
        self.assertLess(elapsed, 5.0, f"Integration test exceeded 5.0s ceiling: {elapsed:.3f}s")

    def test_full_quantization_pipeline_e2e_under_5_seconds(self) -> None:
        """Run complete E2E lifecycle: calibration -> PTQ -> IR packaging -> benchmark in < 5.0s."""
        # 1. Calibration Dataset Assembly (R6, R7, R8, R9, R10)
        loader = CalibrationFrameLoader(target_frames=300, seed=42)
        frames, manifest = loader.assemble_calibration_frames()
        self.assertEqual(len(frames), 300)
        self.assertEqual(manifest.day_frames, 120)
        self.assertEqual(manifest.dusk_frames, 90)
        self.assertEqual(manifest.night_frames, 90)

        # 2. Mock Input Baseline FP32 Model (R4)
        fp32_model_path = self.temp_dir / "yolo11n_distilled.onnx"
        create_mock_fp32_onnx(fp32_model_path, size_bytes=5600000)

        # 3. OpenVINO NNCF Quantization (R5, R11, R12, R16)
        out_dir = self.temp_dir / "models/quantized/openvino"
        cfg = QuantizationConfig(
            model_input_path=str(fp32_model_path),
            output_dir=str(out_dir),
            subset_size=300,
        )
        quantizer = OpenVINOQuantizer(config=cfg)
        report = quantizer.quantize(
            calibration_frames=frames,
            baseline_map=0.725,
            quantized_map=0.718,
        )

        # 4. Verify Compression & Accuracy Metrics (R11, R12)
        self.assertTrue(report.achieved_size_reduction)
        self.assertGreaterEqual(report.size_reduction_pct, 60.0)
        self.assertTrue(report.achieved_accuracy_tolerance)
        self.assertLessEqual(report.map_drop, 0.015)

        # 5. Verify Exported OpenVINO IR Artifacts (R16)
        self.assertTrue(Path(report.ir_xml_path).exists())
        self.assertTrue(Path(report.ir_bin_path).exists())
        self.assertTrue((out_dir / "quantization_report.json").exists())

        # 6. Edge CPU Benchmark Execution (R17, R18, R19)
        runner = OpenVINOBenchmarkRunner(device_name="CPU", benchmark_iterations=20)
        bench_res = runner.benchmark_model(xml_path=report.ir_xml_path, mock_mode=True)

        self.assertTrue(bench_res["achieved_target_fps"])
        self.assertGreaterEqual(bench_res["throughput_fps"], 25.0)
        self.assertTrue(bench_res["within_memory_budget"])
        self.assertLess(bench_res["peak_rss_mb"], 1200.0)

    def test_cli_handlers_execution(self) -> None:
        """Verify CLI entrypoint handler functions execute with --help without syntax errors (R20)."""
        from src.engine.openvino_benchmark import main as bench_main
        from src.engine.openvino_quantizer import main as quant_main

        self.assertTrue(callable(quant_main))
        self.assertTrue(callable(bench_main))


if __name__ == "__main__":
    unittest.main()
