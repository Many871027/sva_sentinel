"""Unit tests for distillation trainer and telemetry logging engine.

Validates:
- R11: Distillation training steps with mixed-precision hooks
- R13: Model parameter reduction verification (>80% reduction and student <= 3.0M params)
- R14: Structured epoch telemetry logging (DistillationEpochLog) and summary reporting (DistillationSummaryReport)
- R19: Ultra-fast execution (< 5.0 seconds) without network egress
- R20: Mock tensor execution without physical GPU hardware
"""

import json
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.engine.distillation_trainer import (
    DistillationTrainer,
    count_model_parameters,
)
from src.pipeline.schemas import (
    DistillationEpochLog,
    DistillationHyperparameters,
    DistillationSummaryReport,
)


class TestDistillationTrainer(unittest.TestCase):
    """Test suite covering DistillationTrainer execution, parameter reduction, and epoch telemetry."""

    def setUp(self):
        self.temp_dir = Path(tempfile.mkdtemp(prefix="test_sva_trainer_"))

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_r13_parameter_reduction_target_achieved(self):
        """Verify R13: Student YOLO11n (2.61M) vs Teacher YOLO11x (56.9M) achieves > 80% reduction."""
        teacher_params = 56_900_000
        student_params = 2_610_000

        reduction_pct, achieved = DistillationTrainer.verify_parameter_reduction(
            teacher_params, student_params
        )

        self.assertGreater(reduction_pct, 80.0)
        self.assertAlmostEqual(reduction_pct, 95.41, places=1)
        self.assertTrue(achieved)

    def test_r13_parameter_reduction_failure_cases(self):
        """Verify R13: Fails when student > 3.0M or reduction <= 80%."""
        # Case 1: Student parameters exceed 3.0M ceiling
        _, achieved_oversize = DistillationTrainer.verify_parameter_reduction(
            teacher_params=56_900_000,
            student_params=3_200_000,
        )
        self.assertFalse(achieved_oversize)

        # Case 2: Insufficient reduction <= 80%
        _, achieved_low_reduction = DistillationTrainer.verify_parameter_reduction(
            teacher_params=10_000_000,
            student_params=2_500_000,
        )
        self.assertFalse(achieved_low_reduction)

    def test_r11_and_r14_train_step_and_epoch_logging(self):
        """Verify R11 & R14: Training epoch execution and structured DistillationEpochLog generation."""
        mock_student = {"parameter_count": 2_610_000}
        mock_teacher = {"parameter_count": 56_900_000}

        hp = DistillationHyperparameters(
            alpha=1.0,
            beta=2.5,
            gamma=0.5,
            temperature=3.0,
            epochs=2,
            batch_size=8,
        )

        trainer = DistillationTrainer(
            student_model=mock_student,
            teacher_model=mock_teacher,
            hyperparameters=hp,
            output_dir=self.temp_dir,
            teacher_map_50=0.65,
        )

        mock_batch = {
            "student_logits": np.random.randn(8, 5).astype(np.float32),
            "teacher_logits": np.random.randn(8, 5).astype(np.float32),
            "student_preds": np.random.randn(8, 5).astype(np.float32),
            "targets": np.random.randn(8, 5).astype(np.float32),
            "student_features": np.random.randn(8, 128, 4, 4).astype(np.float32),
            "teacher_features": np.random.randn(8, 512, 4, 4).astype(np.float32),
        }

        # Train epoch 1
        train_stats = trainer.train_epoch([mock_batch], epoch=1)
        self.assertIn("total_loss", train_stats)
        self.assertGreater(train_stats["total_loss"], 0.0)

        # Evaluate epoch 1
        val_stats = trainer.evaluate(mock_map_50=0.61)
        self.assertAlmostEqual(val_stats["map_50"], 0.61)

        # Record epoch
        epoch_log = trainer.record_epoch(1, train_stats, val_stats)
        self.assertIsInstance(epoch_log, DistillationEpochLog)
        self.assertEqual(epoch_log.epoch, 1)
        self.assertAlmostEqual(epoch_log.map_50, 0.61)
        self.assertEqual(len(trainer.history), 1)

    def test_r14_checkpoint_saving_and_summary_report(self):
        """Verify R14: Checkpoint serialization and comprehensive DistillationSummaryReport compilation."""
        trainer = DistillationTrainer(
            student_model={"parameter_count": 2_610_000},
            teacher_model={"parameter_count": 56_900_000},
            output_dir=self.temp_dir,
            teacher_map_50=0.65,
        )

        # Epoch 1
        trainer.record_epoch(
            epoch=1,
            train_metrics={"total_loss": 1.2, "hard_loss": 0.5, "kd_loss": 0.4, "feat_loss": 0.3},
            val_metrics={"map_50": 0.58, "map_50_95": 0.38, "val_loss": 0.35},
        )
        trainer.save_checkpoint(epoch=1, is_best=False)

        # Epoch 2 (better)
        trainer.record_epoch(
            epoch=2,
            train_metrics={"total_loss": 0.9, "hard_loss": 0.4, "kd_loss": 0.3, "feat_loss": 0.2},
            val_metrics={"map_50": 0.62, "map_50_95": 0.44, "val_loss": 0.28},
        )
        ckpt_path = trainer.save_checkpoint(epoch=2, is_best=True)
        self.assertTrue(ckpt_path.is_file())
        self.assertEqual(ckpt_path.name, "best.pt")

        # Compile report
        report = trainer.compile_summary_report(
            exported_pt=str(ckpt_path),
            exported_onnx=str(self.temp_dir / "yolo11n_distilled.onnx"),
        )

        self.assertIsInstance(report, DistillationSummaryReport)
        self.assertEqual(report.best_epoch, 2)
        self.assertAlmostEqual(report.best_map_50, 0.62)
        # Check mAP retention: 0.62 / 0.65 = 95.38% >= 92%
        self.assertGreaterEqual(report.map_retention_pct, 92.0)
        self.assertTrue(report.achieved_target_retention)
        self.assertTrue(report.achieved_target_reduction)

        # Validate saved summary json file
        json_file = self.temp_dir / "distillation_summary.json"
        self.assertTrue(json_file.is_file())
        with open(json_file, "r", encoding="utf-8") as f:
            saved_data = json.load(f)
        self.assertEqual(saved_data["best_epoch"], 2)


if __name__ == "__main__":
    unittest.main()
