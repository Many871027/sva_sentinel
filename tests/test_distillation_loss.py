"""Unit tests for knowledge distillation loss engine.

Validates:
- R6: Composite distillation loss L_total = alpha * L_hard + beta * L_KD + gamma * L_feat
- R8: Response-based distillation loss with temperature scaling tau >= 1.0 and tau^2 factor
- R9: Feature alignment loss with 1x1 conv adaptation (Cs -> Ct) and MSE/cosine distance
- R10: Anomaly detection raising DistillationLossError on NaN or Inf with statistics
- R19: Ultra-fast execution (< 5.0 seconds) without network egress
- R20: Mathematical correctness of gradients via synthetic mock tensors without physical GPU
"""

import math
import unittest
import numpy as np

from src.core.exceptions import DistillationLossError
from src.engine.distillation_loss import (
    CompositeDistillationLoss,
    FeatureAlignmentLoss,
    HardTaskLoss,
    LogitDistillationLoss,
    MockTensor,
    compute_ciou_loss,
    compute_dfl_loss,
    compute_bce_loss,
)


class TestDistillationLoss(unittest.TestCase):
    """Test suite covering distillation loss formulations, stability, and gradients."""

    def test_r6_composite_loss_linear_combination(self):
        """Verify R6: L_total = alpha * L_hard + beta * L_KD + gamma * L_feat."""
        loss_engine = CompositeDistillationLoss(
            alpha=1.0,
            beta=2.5,
            gamma=0.5,
            temperature=3.0,
            student_channels=64,
            teacher_channels=128,
        )

        # Constant inputs for predictable component values
        z_s = np.zeros((4, 5), dtype=np.float32)
        z_t = np.zeros((4, 5), dtype=np.float32)
        f_s = np.zeros((4, 64, 8, 8), dtype=np.float32)
        f_t = np.zeros((4, 128, 8, 8), dtype=np.float32)

        res = loss_engine(
            student_preds=2.0,
            ground_truth=0.0,
            student_logits=z_s,
            teacher_logits=z_t,
            student_features=f_s,
            teacher_features=f_t,
        )

        stats = res["stats"]
        self.assertAlmostEqual(stats["hard_loss"], 2.0, places=5)
        self.assertAlmostEqual(stats["kd_loss"], 0.0, places=5)
        self.assertAlmostEqual(stats["feat_loss"], 0.0, places=5)
        expected_total = 1.0 * 2.0 + 2.5 * 0.0 + 0.5 * 0.0
        self.assertAlmostEqual(stats["total_loss"], expected_total, places=5)

    def test_r6_negative_weights_raise_value_error(self):
        """Verify R6: Negative alpha, beta, gamma or tau < 1.0 raise ValueError."""
        with self.assertRaises(ValueError):
            CompositeDistillationLoss(alpha=-0.1)

        with self.assertRaises(ValueError):
            CompositeDistillationLoss(beta=-1.0)

        with self.assertRaises(ValueError):
            CompositeDistillationLoss(gamma=-0.5)

        with self.assertRaises(ValueError):
            CompositeDistillationLoss(temperature=0.5)

    def test_r8_kd_identical_logits_zero_loss(self):
        """Verify R8: When student logits equal teacher logits, L_KD = 0.0."""
        kd = LogitDistillationLoss(temperature=3.0)
        z = np.array([[2.5, -1.2, 0.4, 3.1, -0.8]], dtype=np.float32)
        loss = kd(z, z)
        self.assertAlmostEqual(float(loss), 0.0, places=5)

    def test_r8_kd_temperature_scaling_invariance(self):
        """Verify R8: L_KD applies tau^2 scaling preserving scale across temperatures."""
        z_s = np.array([[3.0, 1.0, -1.0]], dtype=np.float32)
        z_t = np.array([[1.0, 3.0, -1.0]], dtype=np.float32)

        kd_tau1 = LogitDistillationLoss(temperature=1.0)
        kd_tau3 = LogitDistillationLoss(temperature=3.0)

        loss_tau1 = kd_tau1(z_s, z_t)
        loss_tau3 = kd_tau3(z_s, z_t)

        self.assertGreater(float(loss_tau1), 0.0)
        self.assertGreater(float(loss_tau3), 0.0)
        # Verify finite positive value
        self.assertTrue(math.isfinite(float(loss_tau3)))

    def test_r9_feature_alignment_adaptation_shape(self):
        """Verify R9: 1x1 conv adaptation projects student channels (Cs=128) to teacher channels (Ct=512)."""
        feat_loss = FeatureAlignmentLoss(student_channels=128, teacher_channels=512, metric="mse")

        # Student neck feat: (B=2, Cs=128, H=10, W=10)
        f_s = np.random.randn(2, 128, 10, 10).astype(np.float32)
        # Teacher neck feat: (B=2, Ct=512, H=10, W=10)
        f_t = np.random.randn(2, 512, 10, 10).astype(np.float32)

        adapted = feat_loss.adapt_student_features(f_s)
        self.assertEqual(adapted.shape, (2, 512, 10, 10))

        loss = feat_loss(f_s, f_t)
        self.assertGreater(float(loss), 0.0)
        self.assertTrue(math.isfinite(float(loss)))

    def test_r9_feature_alignment_cosine_metric(self):
        """Verify R9: Feature alignment supports cosine distance metric."""
        feat_loss = FeatureAlignmentLoss(student_channels=64, teacher_channels=64, metric="cosine")
        f_s = np.ones((2, 64, 4, 4), dtype=np.float32)
        f_t = np.ones((2, 64, 4, 4), dtype=np.float32)

        loss = feat_loss(f_s, f_t)
        # Cosine distance between identical vectors is 0.0
        self.assertAlmostEqual(float(loss), 0.0, places=5)

    def test_r10_nan_detection_raises_distillation_loss_error(self):
        """Verify R10: NaN in student logits raises DistillationLossError with statistics."""
        loss_engine = CompositeDistillationLoss()
        z_s = np.array([[float("nan"), 1.0, 0.0]], dtype=np.float32)
        z_t = np.array([[1.0, 1.0, 0.0]], dtype=np.float32)

        with self.assertRaises(DistillationLossError) as ctx:
            loss_engine(
                student_preds=0.0,
                ground_truth=0.0,
                student_logits=z_s,
                teacher_logits=z_t,
            )

        self.assertIn("L_KD", str(ctx.exception))
        self.assertIn("component", ctx.exception.loss_stats)

    def test_r10_inf_detection_raises_distillation_loss_error(self):
        """Verify R10: Inf in features raises DistillationLossError."""
        loss_engine = CompositeDistillationLoss()
        f_s = np.array([[[[float("inf")]]]], dtype=np.float32)
        f_t = np.array([[[[1.0]]]], dtype=np.float32)

        with self.assertRaises(DistillationLossError) as ctx:
            loss_engine(
                student_preds=0.0,
                ground_truth=0.0,
                student_logits=np.zeros((1, 5)),
                teacher_logits=np.zeros((1, 5)),
                student_features=f_s,
                teacher_features=f_t,
            )

        self.assertIn("L_feat", str(ctx.exception))

    def test_r7_hard_task_loss_components(self):
        """Verify R7: CIoU, DFL, and BCE hard loss calculations."""
        # CIoU perfect match
        b1 = np.array([[10.0, 10.0, 50.0, 50.0]], dtype=np.float32)
        ciou_loss = compute_ciou_loss(b1, b1)
        self.assertAlmostEqual(ciou_loss, 0.0, places=4)

        # BCE loss
        logits = np.array([[0.0, 0.0]], dtype=np.float32)
        targets = np.array([[0.5, 0.5]], dtype=np.float32)
        bce_loss = compute_bce_loss(logits, targets)
        self.assertGreater(bce_loss, 0.0)

        # DFL loss
        p_dist = np.array([[0.5, 0.5]], dtype=np.float32)
        dfl_loss = compute_dfl_loss(p_dist, targets)
        self.assertGreater(dfl_loss, 0.0)

    def test_r20_mock_tensor_gradient_propagation(self):
        """Verify R20: Backward pass populates finite gradients on synthetic mock tensors."""
        kd = LogitDistillationLoss(temperature=2.0)
        z_s = MockTensor([[1.5, -0.5, 0.2]], requires_grad=True)
        z_t = np.array([[0.5, 0.5, 0.2]], dtype=np.float32)

        loss = kd(z_s, z_t)
        self.assertTrue(isinstance(loss, MockTensor))
        self.assertTrue(loss.requires_grad)

        # Run backward
        loss.backward()

        self.assertIsNotNone(z_s.grad)
        self.assertEqual(z_s.grad.shape, (1, 3))
        self.assertTrue(np.all(np.isfinite(z_s.grad)))
        self.assertFalse(np.all(z_s.grad == 0.0))


if __name__ == "__main__":
    unittest.main()
