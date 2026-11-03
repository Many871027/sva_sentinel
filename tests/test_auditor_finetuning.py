"""Unit tests for auditor LoRA/QLoRA fine-tuning and adapter export engine.

Covers:
- R4: Hardware acceleration check and HardwareAccelerationError fail-fast behavior.
- R5: BitsAndBytes 4-bit NF4 configuration construction.
- R6: LoRA hyperparameters (r >= 16, alpha >= 32, max_seq_length=2048, 7 target modules).
- R13: Standalone FP16 weight merging, tokenizer preservation, and export validation.
- R22: Zero-network sub-second offline execution.
"""

import json
from pathlib import Path
import tempfile
import time
import unittest

from src.core.exceptions import AuditorFinetuningError, HardwareAccelerationError
from src.engine.auditor_finetuning import AuditorFineTuner, LoRAConfigProfile
from src.pipeline.telemetry_generator import generate_synthetic_telemetry_dataset
from tests.fixtures.mock_telemetry import create_mock_hf_model_dir


class TestAuditorFineTuning(unittest.TestCase):
    """Test suite verifying LoRA training harness, hardware checks, and adapter merge."""

    def test_r4_hardware_acceleration_fail_fast(self) -> None:
        """Verify R4: HardwareAccelerationError raised when require_gpu=True on non-CUDA system."""
        try:
            import torch
            has_cuda = torch.cuda.is_available()
        except ImportError:
            has_cuda = False

        if not has_cuda:
            with self.assertRaises(HardwareAccelerationError):
                AuditorFineTuner(require_gpu=True)
        else:
            # On CUDA system, initialization should succeed
            tuner = AuditorFineTuner(require_gpu=True)
            self.assertIsNotNone(tuner)

        # Default mode (require_gpu=False) must succeed on CPU
        cpu_tuner = AuditorFineTuner(require_gpu=False)
        self.assertIsNotNone(cpu_tuner)

    def test_r5_bitsandbytes_nf4_configuration(self) -> None:
        """Verify R5: NF4 quantization parameters constructed for peak VRAM < 12GB."""
        profile = LoRAConfigProfile(use_qlora_4bit=True, bnb_4bit_compute_dtype="float16")
        tuner = AuditorFineTuner(config=profile, require_gpu=False)
        bnb_cfg = tuner.get_bitsandbytes_config()

        self.assertTrue(bnb_cfg["load_in_4bit"])
        self.assertEqual(bnb_cfg["bnb_4bit_quant_type"], "nf4")
        self.assertTrue(bnb_cfg["bnb_4bit_use_double_quant"])
        self.assertEqual(bnb_cfg["bnb_4bit_compute_dtype"], "float16")

        # Disabled 4-bit returns empty
        profile_no_4bit = LoRAConfigProfile(use_qlora_4bit=False)
        tuner_no_4bit = AuditorFineTuner(config=profile_no_4bit, require_gpu=False)
        self.assertEqual(tuner_no_4bit.get_bitsandbytes_config(), {})

    def test_r6_lora_hyperparameter_profile(self) -> None:
        """Verify R6: LoRA config enforces r >= 16, alpha >= 32, max_seq_length=2048."""
        profile = LoRAConfigProfile()
        self.assertGreaterEqual(profile.lora_r, 16)
        self.assertGreaterEqual(profile.lora_alpha, 32)
        self.assertEqual(profile.max_seq_length, 2048)

        expected_targets = {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"}
        self.assertTrue(expected_targets.issubset(set(profile.target_modules)))

    def test_r6_mock_training_execution(self) -> None:
        """Verify R6: train() produces adapter artifacts and summary logs."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            jsonl_path = tmp_path / "telemetry_train.jsonl"
            out_adapter_dir = tmp_path / "lora_adapter"

            # Create small dataset
            generate_synthetic_telemetry_dataset(num_samples=10, seed=42, output_path=jsonl_path)

            tuner = AuditorFineTuner(require_gpu=False)
            summary = tuner.train(training_jsonl=jsonl_path, output_dir=out_adapter_dir, mock_mode=True)

            self.assertEqual(summary["status"], "success")
            self.assertEqual(summary["total_samples"], 10)
            self.assertTrue((out_adapter_dir / "adapter_config.json").exists())
            self.assertTrue((out_adapter_dir / "adapter_model.bin").exists())
            self.assertTrue((out_adapter_dir / "training_summary.json").exists())

    def test_r13_merge_and_export_fp16(self) -> None:
        """Verify R13: Standalone FP16 model export preserves configs and tokenizer files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            base_dir = tmp_path / "base_hf_model"
            adapter_dir = tmp_path / "lora_adapter"
            export_dir = tmp_path / "qwen2.5-0.5b-auditor-fp16"

            create_mock_hf_model_dir(base_dir)
            adapter_dir.mkdir(parents=True, exist_ok=True)
            (adapter_dir / "adapter_config.json").write_text("{}", encoding="utf-8")

            tuner = AuditorFineTuner(require_gpu=False)
            exported_path = tuner.merge_and_export_fp16(
                base_model_dir=base_dir,
                adapter_dir=adapter_dir,
                export_dir=export_dir,
                mock_mode=True,
            )

            self.assertTrue(exported_path.exists())
            self.assertTrue((export_dir / "config.json").exists())
            self.assertTrue((export_dir / "tokenizer.json").exists())
            self.assertTrue((export_dir / "tokenizer_config.json").exists())
            self.assertTrue((export_dir / "model.safetensors").exists())

    def test_r22_fast_execution(self) -> None:
        """Verify R22: Full mock training and export cycle completes in < 1.0s."""
        start = time.perf_counter()
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            jsonl_path = tmp_path / "data.jsonl"
            adapter_dir = tmp_path / "adapter"
            export_dir = tmp_path / "export"

            generate_synthetic_telemetry_dataset(num_samples=5, output_path=jsonl_path)
            tuner = AuditorFineTuner(require_gpu=False)
            tuner.train(training_jsonl=jsonl_path, output_dir=adapter_dir, mock_mode=True)
            tuner.merge_and_export_fp16(base_model_dir=tmp_path, adapter_dir=adapter_dir, export_dir=export_dir, mock_mode=True)

        elapsed = time.perf_counter() - start
        self.assertLess(elapsed, 1.0, f"Expected execution in < 1.0s, took {elapsed:.3f}s")


if __name__ == "__main__":
    unittest.main()
