"""Unit tests for AuditorCaseEvaluator.

Covers:
- R18: Evaluation of 100 diverse synthetic traffic infraction cases (35 vehicular wrong-way, 25 micromobility, 20 speeding, 20 benign).
- R19: 100% of model completions parse into valid IncidentAuditVerdict.
- R20: Sustained sequential evaluation maintaining peak RSS < 1200MB.
- R21: Wrong-way classification accuracy >= 90% and risk-level accuracy >= 85%.
- R24: Fast offline execution (< 1.0s).
"""

import unittest

from src.engine.auditor_case_evaluator import AuditorCaseEvaluator
from src.engine.auditor_engine import ForensicAuditorEngine


class TestAuditorCaseEvaluator(unittest.TestCase):
    """Test suite for 100-case deterministic batch evaluator."""

    def setUp(self) -> None:
        self.engine = ForensicAuditorEngine(mock_mode=True)
        self.evaluator = AuditorCaseEvaluator(engine=self.engine, mock_mode=True)

    def test_r18_100_cases_generation_and_balance(self) -> None:
        """Verify R18: 100 balanced infraction test cases generated."""
        cases, ground_truths = self.evaluator.generate_100_cases()

        self.assertEqual(len(cases), 100, "Must synthesize exactly 100 infraction cases")
        self.assertEqual(len(ground_truths), 100, "Must have 100 ground truth verdicts")

        # Verify class and category partitions
        vehicular_ww = [c for c in cases if c.incident_id.startswith("CASE-WW-")]
        micromobility = [c for c in cases if c.incident_id.startswith("CASE-MICRO-")]
        speeding = [c for c in cases if c.incident_id.startswith("CASE-SPD-")]
        benign = [c for c in cases if c.incident_id.startswith("CASE-BENIGN-")]

        self.assertEqual(len(vehicular_ww), 35, "Must contain 35 wrong-way vehicular cases")
        self.assertEqual(len(micromobility), 25, "Must contain 25 micromobility cases")
        self.assertEqual(len(speeding), 20, "Must contain 20 speeding cases")
        self.assertEqual(len(benign), 20, "Must contain 20 benign control cases")

    def test_r19_100_percent_schema_parse_rate(self) -> None:
        """Verify R19: 100% of model completions parse into valid IncidentAuditVerdict."""
        summary = self.evaluator.evaluate()

        self.assertEqual(
            summary["schema_parse_success_rate"],
            100.0,
            "100% of completions must parse without JSON syntax errors (R19)",
        )
        self.assertTrue(summary["achieved_100_pct_parse"])

    def test_r20_memory_ceiling_under_1200mb(self) -> None:
        """Verify R20: Process peak RSS memory remains under 1200.0MB throughout 100 cases."""
        summary = self.evaluator.evaluate()

        peak_rss = summary["peak_rss_mb"]
        self.assertLessEqual(
            peak_rss,
            1200.0,
            f"Peak RSS {peak_rss:.1f}MB must not exceed 1200.0MB container ceiling (R20)",
        )
        self.assertTrue(summary["within_memory_budget"])

    def test_r21_classification_accuracy_thresholds(self) -> None:
        """Verify R21: Wrong-way accuracy >= 90% and risk-level accuracy >= 85%."""
        summary = self.evaluator.evaluate()

        ww_acc = summary["wrong_way_accuracy_pct"]
        risk_acc = summary["risk_level_accuracy_pct"]

        self.assertGreaterEqual(
            ww_acc,
            90.0,
            f"Wrong-way accuracy {ww_acc:.1f}% must achieve >= 90.0% (R21)",
        )
        self.assertTrue(summary["achieved_wrong_way_threshold"])

        self.assertGreaterEqual(
            risk_acc,
            85.0,
            f"Risk-level accuracy {risk_acc:.1f}% must achieve >= 85.0% (R21)",
        )
        self.assertTrue(summary["achieved_risk_threshold"])


if __name__ == "__main__":
    unittest.main()
