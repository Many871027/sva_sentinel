"""Forensic auditor verification and schema validation engine.

Covers:
- R17: Extraction and Pydantic validation of model output JSON adhering to IncidentAuditVerdict.
- R18: Evaluation of JSON parsing success rate (target >= 95%).
- R19: Evaluation of wrong-way infraction classification accuracy (target >= 90%).
- R20: Evaluation of risk-level categorization accuracy (target >= 85%).
"""

from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Tuple

from pydantic import ValidationError

from src.core.exceptions import AuditorVerificationError
from src.pipeline.schemas import IncidentAuditVerdict

logger = logging.getLogger(__name__)


class AuditorVerifier:
    """Verifies completion outputs from fine-tuned auditor LLM against forensic schema."""

    @staticmethod
    def extract_json_payload(raw_completion: str) -> str:
        """Extract JSON substring from LLM completion, stripping markdown code fences."""
        text = raw_completion.strip()

        # Handle ```json ... ``` or ``` ... ```
        fence_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
        if fence_match:
            return fence_match.group(1).strip()

        # Look for outermost balanced braces
        brace_match = re.search(r"(\{.*\})", text, re.DOTALL)
        if brace_match:
            return brace_match.group(1).strip()

        return text

    def validate_verdict(self, raw_completion: str) -> IncidentAuditVerdict:
        """Parse and validate a raw completion string into an IncidentAuditVerdict instance (R17)."""
        payload = self.extract_json_payload(raw_completion)
        try:
            return IncidentAuditVerdict.model_validate_json(payload)
        except (ValueError, ValidationError) as e:
            raise AuditorVerificationError(
                f"Failed to parse and validate IncidentAuditVerdict from completion: {e}\nRaw: {raw_completion[:200]}"
            ) from e

    def evaluate_batch(
        self,
        completions: List[str],
        ground_truths: List[IncidentAuditVerdict],
    ) -> Dict[str, Any]:
        """Compute parsing rate, wrong-way accuracy, and risk level accuracy across validation samples (R18, R19, R20)."""
        if len(completions) != len(ground_truths):
            raise AuditorVerificationError(
                f"Mismatch in count: {len(completions)} completions vs {len(ground_truths)} ground truth verdicts."
            )

        total = len(completions)
        if total == 0:
            raise AuditorVerificationError("Cannot evaluate empty batch.")

        valid_json_count = 0
        wrong_way_correct = 0
        risk_level_correct = 0
        errors: List[Dict[str, Any]] = []

        for idx, (comp, gt) in enumerate(zip(completions, ground_truths)):
            try:
                pred = self.validate_verdict(comp)
                valid_json_count += 1

                # Check wrong-way accuracy (R19)
                if pred.is_wrong_way == gt.is_wrong_way:
                    wrong_way_correct += 1

                # Check risk level accuracy (R20)
                if pred.risk_level == gt.risk_level:
                    risk_level_correct += 1

            except AuditorVerificationError as err:
                errors.append({"index": idx, "incident_id": gt.incident_id, "error": str(err)})

        parse_rate = valid_json_count / total
        wrong_way_acc = wrong_way_correct / total
        risk_acc = risk_level_correct / total

        meets_r18 = parse_rate >= 0.95
        meets_r19 = wrong_way_acc >= 0.90
        meets_r20 = risk_acc >= 0.85
        all_passed = meets_r18 and meets_r19 and meets_r20

        report = {
            "total_evaluated": total,
            "valid_json_count": valid_json_count,
            "parse_success_rate": round(parse_rate, 4),
            "target_parse_rate_met": meets_r18,
            "wrong_way_correct": wrong_way_correct,
            "wrong_way_accuracy": round(wrong_way_acc, 4),
            "target_wrong_way_met": meets_r19,
            "risk_level_correct": risk_level_correct,
            "risk_level_accuracy": round(risk_acc, 4),
            "target_risk_met": meets_r20,
            "all_criteria_passed": all_passed,
            "error_count": len(errors),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        logger.info(
            f"Verification Results: Parse={parse_rate:.1%}, Wrong-Way={wrong_way_acc:.1%}, Risk={risk_acc:.1%} (Passed={all_passed})"
        )
        return report

    def save_report(self, report: Dict[str, Any], output_path: str | Path) -> Path:
        """Persist verification report as JSON."""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        return path
