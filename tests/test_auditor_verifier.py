"""Unit tests for forensic auditor verification engine, schema compliance, and metric tracking.

Covers:
- R17: Markdown code fence stripping and Pydantic IncidentAuditVerdict parsing.
- R18: Parse success rate evaluation (threshold >= 95%).
- R19: Wrong-way infraction classification accuracy (threshold >= 90%).
- R20: Risk level concordance evaluation (threshold >= 85%).
- R22: Zero-network sub-second offline execution.
"""

from pathlib import Path
import tempfile
import time
import unittest

from src.core.exceptions import AuditorVerificationError
from src.engine.auditor_verifier import AuditorVerifier
from src.pipeline.schemas import IncidentAuditVerdict, RiskLevel
from tests.fixtures.mock_telemetry import create_mock_verdict


class TestAuditorVerifier(unittest.TestCase):
    """Test suite verifying auditor completion parsing, evaluation metrics, and reporting."""

    def setUp(self) -> None:
        self.verifier = AuditorVerifier()

    def test_r17_json_extraction_from_markdown_blocks(self) -> None:
        """Verify R17: Extraction of JSON payload from raw strings and markdown code fences."""
        verdict = create_mock_verdict(incident_id="INC-001", is_wrong_way=True, risk_level=RiskLevel.HIGH)
        raw_json = verdict.model_dump_json()

        # Direct raw JSON string
        parsed_direct = self.verifier.validate_verdict(raw_json)
        self.assertEqual(parsed_direct.incident_id, "INC-001")
        self.assertTrue(parsed_direct.is_wrong_way)

        # Markdown fenced code block
        fenced_json = f"```json\n{raw_json}\n```"
        parsed_fenced = self.verifier.validate_verdict(fenced_json)
        self.assertEqual(parsed_fenced.incident_id, "INC-001")

        # Conversational preamble around code block
        wrapped_json = f"Here is the forensic audit result:\n```json\n{raw_json}\n```\nHope this helps!"
        parsed_wrapped = self.verifier.validate_verdict(wrapped_json)
        self.assertEqual(parsed_wrapped.incident_id, "INC-001")

        # Invalid JSON string must raise AuditorVerificationError
        with self.assertRaises(AuditorVerificationError):
            self.verifier.validate_verdict("This is not valid JSON content.")

    def test_r18_parse_success_rate_threshold(self) -> None:
        """Verify R18: Parse success rate evaluated with target >= 95%."""
        ground_truths = [create_mock_verdict(f"INC-{i:03d}") for i in range(50)]

        # 48 valid completions, 2 invalid (48/50 = 96% >= 95%)
        completions = [gt.model_dump_json() for gt in ground_truths]
        completions[0] = "MALFORMED_OUTPUT_1"
        completions[1] = "MALFORMED_OUTPUT_2"

        report = self.verifier.evaluate_batch(completions, ground_truths)
        self.assertEqual(report["total_evaluated"], 50)
        self.assertEqual(report["valid_json_count"], 48)
        self.assertEqual(report["parse_success_rate"], 0.96)
        self.assertTrue(report["target_parse_rate_met"])

        # 40 valid, 10 invalid (40/50 = 80% < 95%)
        for i in range(2, 10):
            completions[i] = "MALFORMED"

        fail_report = self.verifier.evaluate_batch(completions, ground_truths)
        self.assertFalse(fail_report["target_parse_rate_met"])

    def test_r19_and_r20_accuracy_thresholds(self) -> None:
        """Verify R19 & R20: Wrong-way accuracy (>= 90%) and risk concordance (>= 85%)."""
        ground_truths = []
        completions = []

        # Create 50 samples
        for i in range(50):
            is_ww = i % 2 == 0
            risk = RiskLevel.HIGH if is_ww else RiskLevel.LOW
            gt = create_mock_verdict(f"INC-{i:03d}", is_wrong_way=is_ww, risk_level=risk)
            ground_truths.append(gt)

            # Create prediction with 92% wrong-way match (4 mismatch) and 88% risk match (6 mismatch)
            pred_ww = is_ww if i >= 4 else not is_ww
            pred_risk = risk if i >= 6 else RiskLevel.CRITICAL
            pred = create_mock_verdict(f"INC-{i:03d}", is_wrong_way=pred_ww, risk_level=pred_risk)
            completions.append(pred.model_dump_json())

        report = self.verifier.evaluate_batch(completions, ground_truths)
        self.assertEqual(report["wrong_way_accuracy"], 0.92)
        self.assertTrue(report["target_wrong_way_met"])
        self.assertEqual(report["risk_level_accuracy"], 0.88)
        self.assertTrue(report["target_risk_met"])
        self.assertTrue(report["all_criteria_passed"])

    def test_save_report(self) -> None:
        """Verify report persistence to JSON file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            out_file = Path(tmpdir) / "evaluation_report.json"
            sample_report = {"parse_success_rate": 0.98, "all_criteria_passed": True}
            saved = self.verifier.save_report(sample_report, out_file)

            self.assertTrue(saved.exists())
            with open(saved, "r", encoding="utf-8") as f:
                import json
                data = json.load(f)
            self.assertEqual(data["parse_success_rate"], 0.98)

    def test_r22_fast_execution(self) -> None:
        """Verify R22: 50-sample batch verification completes in < 0.2s."""
        ground_truths = [create_mock_verdict(f"INC-{i:03d}") for i in range(50)]
        completions = [gt.model_dump_json() for gt in ground_truths]

        start = time.perf_counter()
        report = self.verifier.evaluate_batch(completions, ground_truths)
        elapsed = time.perf_counter() - start

        self.assertTrue(report["all_criteria_passed"])
        self.assertLess(elapsed, 0.2, f"Expected verification in < 0.2s, took {elapsed:.3f}s")


if __name__ == "__main__":
    unittest.main()
