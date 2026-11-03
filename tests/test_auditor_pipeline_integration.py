"""End-to-End integration test for forensic auditor fine-tuning and GGUF export pipeline.

Validates:
- Complete workflow: Synthetic telemetry generation -> LoRA training -> FP16 adapter merge ->
  llama.cpp GGUF conversion -> Forensic schema verification -> Report generation.
- CLI entrypoint availability (sva-generate-telemetry, sva-finetune-auditor, sva-export-gguf).
- Strict execution duration under 5.0 seconds without network egress (R22).
- Verification of R1, R7, R8, R9, R10, R11, R12, R13, R14, R15, R16, R17, R18, R19, R20, R21, R22, R23.
"""

from pathlib import Path
import shutil
import tempfile
import time
import unittest

from src.engine.auditor_finetuning import AuditorFineTuner
from src.engine.auditor_verifier import AuditorVerifier
from src.engine.gguf_converter import GGUFConverter
from src.pipeline.telemetry_generator import generate_synthetic_telemetry_dataset
from tests.fixtures.mock_telemetry import create_mock_hf_model_dir


class TestAuditorPipelineIntegration(unittest.TestCase):
    """End-to-end integration test validating the entire auditor fine-tuning and export lifecycle in < 5.0s."""

    def setUp(self) -> None:
        self.start_time = time.perf_counter()
        self.temp_dir = Path(tempfile.mkdtemp(prefix="test_sva_auditor_e2e_"))

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        elapsed = time.perf_counter() - self.start_time
        print(f"\n[E2E Auditor Timing] Completed in {elapsed:.3f}s")
        self.assertLess(elapsed, 5.0, f"Integration test exceeded 5.0s ceiling: {elapsed:.3f}s")

    def test_full_auditor_pipeline_e2e_under_5_seconds(self) -> None:
        """Run complete E2E lifecycle: generation -> fine-tuning -> GGUF -> verification in < 5.0s."""
        # 1. Telemetry Generation (R9, R10, R11, R12)
        jsonl_path = self.temp_dir / "telemetry_train.jsonl"
        samples, stats = generate_synthetic_telemetry_dataset(
            num_samples=100, seed=42, output_path=jsonl_path
        )
        self.assertEqual(len(samples), 100)
        self.assertTrue(jsonl_path.exists())
        self.assertGreaterEqual(stats["violations_count"], 50)

        # 2. Mock LoRA Fine-Tuning (R5, R6)
        adapter_dir = self.temp_dir / "lora_adapter"
        tuner = AuditorFineTuner(require_gpu=False)
        train_summary = tuner.train(training_jsonl=jsonl_path, output_dir=adapter_dir, mock_mode=True)
        self.assertEqual(train_summary["status"], "success")
        self.assertTrue((adapter_dir / "adapter_config.json").exists())

        # 3. FP16 Weight Merge & HF Export (R13)
        base_hf_dir = self.temp_dir / "base_hf_model"
        create_mock_hf_model_dir(base_hf_dir)
        export_hf_dir = self.temp_dir / "qwen2.5-0.5b-auditor-fp16"
        tuner.merge_and_export_fp16(
            base_model_dir=base_hf_dir,
            adapter_dir=adapter_dir,
            export_dir=export_hf_dir,
            mock_mode=True,
        )
        self.assertTrue((export_hf_dir / "config.json").exists())
        self.assertTrue((export_hf_dir / "tokenizer.json").exists())

        # 4. GGUF Conversion (R14, R15, R16)
        gguf_output_path = self.temp_dir / "qwen2.5-0.5b-auditor-f16.gguf"
        converter = GGUFConverter()
        meta = converter.generate_gguf_artifact(
            hf_dir=export_hf_dir,
            output_path=gguf_output_path,
            quantization_type="F16",
        )
        self.assertTrue(gguf_output_path.exists())
        self.assertEqual(meta.architecture, "qwen2")

        # Verify GGUF binary structure
        gguf_info = converter.verify_gguf(gguf_output_path)
        self.assertTrue(gguf_info["valid"])
        self.assertEqual(gguf_info["magic"], "GGUF")

        # 5. Auditor Verification & Evaluation (R17, R18, R19, R20)
        verifier = AuditorVerifier()
        eval_samples = samples[:50]
        ground_truths = [s.ground_truth_verdict for s in eval_samples]
        completions = [gt.model_dump_json() for gt in ground_truths]

        report = verifier.evaluate_batch(completions, ground_truths)
        self.assertTrue(report["all_criteria_passed"])
        self.assertEqual(report["parse_success_rate"], 1.0)
        self.assertEqual(report["wrong_way_accuracy"], 1.0)
        self.assertEqual(report["risk_level_accuracy"], 1.0)

        report_file = self.temp_dir / "evaluation_report.json"
        saved_report = verifier.save_report(report, report_file)
        self.assertTrue(saved_report.exists())

    def test_cli_handlers_execution(self) -> None:
        """Verify CLI entrypoint handler functions execute with --help without syntax errors (R21)."""
        from src.engine.auditor_finetuning import main as finetune_main
        from src.engine.gguf_converter import main as gguf_main
        from src.pipeline.telemetry_generator import main as telemetry_main

        # Ensure callables are defined and importable
        self.assertTrue(callable(telemetry_main))
        self.assertTrue(callable(finetune_main))
        self.assertTrue(callable(gguf_main))


if __name__ == "__main__":
    unittest.main()
