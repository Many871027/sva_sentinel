"""100-Case Deterministic Evaluator for Forensic Auditor.

Covers:
- R18: Evaluate quantized auditor on test suite of 100 diverse synthetic traffic infraction cases.
- R19: 100% of model completions parse into valid IncidentAuditVerdict models.
- R20: Peak process RSS memory remains under 1.2GB across all 100 cases.
- R21: Achieve >= 90% accuracy in wrong-way detection and >= 85% in risk-level categorization.
- R23: CLI console entrypoint `sva-evaluate-auditor`.
"""

import argparse
from datetime import datetime, timezone
import json
import logging
import os
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Tuple

import psutil

from src.core.exceptions import AuditorBenchmarkError, AuditorVerificationError
from src.engine.auditor_engine import ForensicAuditorEngine
from src.pipeline.schemas import (
    AuditorCaseResult,
    IncidentAuditVerdict,
    RiskLevel,
    TrafficTelemetryRecord,
)

logger = logging.getLogger(__name__)


class AuditorCaseEvaluator:
    """Evaluates quantized auditor over 100 diverse traffic infraction cases."""

    def __init__(
        self,
        engine: Optional[ForensicAuditorEngine] = None,
        mock_mode: bool = False,
    ) -> None:
        self.mock_mode = mock_mode
        self.engine = engine or ForensicAuditorEngine(mock_mode=mock_mode)

    @staticmethod
    def generate_100_cases() -> Tuple[List[TrafficTelemetryRecord], List[IncidentAuditVerdict]]:
        """Synthesize 100 diverse infraction test cases with ground truth labels (R18).

        Distribution:
        - 35 Wrong-way vehicular ingress (cars/motorcycles driving counter-flow)
        - 25 Sidewalk/wrong-way micromobility (e-scooters/bicycles)
        - 20 Excessive speed in school/intersection zones
        - 20 Benign compliant control traffic
        Total: 100 cases
        """
        cases: List[TrafficTelemetryRecord] = []
        ground_truths: List[IncidentAuditVerdict] = []

        now_iso = datetime.now(timezone.utc).isoformat()

        # 1. 35 Wrong-Way Vehicular Cases
        for i in range(35):
            cid = f"CASE-WW-{i+1:03d}"
            entity = "car" if i % 2 == 0 else "motorcycle"
            legal_hdg = 90.0
            obs_hdg = 270.0  # 180° opposing heading
            speed = 36.0 + (i % 10)
            limit = 30.0
            rec = TrafficTelemetryRecord(
                incident_id=cid,
                timestamp=now_iso,
                entity_type=entity,
                lane_id="Lane-1-EB",
                lane_legal_heading=legal_hdg,
                observed_heading=obs_hdg,
                estimated_speed_kmh=speed,
                speed_limit_kmh=limit,
                trajectory_points=[(0.0, 0.0), (2.0, 4.0)],
                dwell_time_seconds=2.5,
                pedestrian_zone=False,
            )
            cases.append(rec)
            # Wrong-way + over limit is CRITICAL
            risk = RiskLevel.CRITICAL if speed > limit else RiskLevel.HIGH
            ground_truths.append(
                IncidentAuditVerdict(
                    incident_id=cid,
                    timestamp=now_iso,
                    entity_type=entity,
                    lane_id="Lane-1-EB",
                    is_wrong_way=True,
                    heading_degrees=obs_hdg,
                    estimated_speed_kmh=speed,
                    speed_limit_kmh=limit,
                    risk_level=risk,
                    violation_detected=True,
                    summary=f"Counter-flow vehicle on Lane-1-EB.",
                    reasoning=f"Heading {obs_hdg}° violates {legal_hdg}° directive.",
                )
            )

        # 2. 25 Micromobility Cases (Wrong-way and/or Sidewalk encroachment)
        for i in range(25):
            cid = f"CASE-MICRO-{i+1:03d}"
            entity = "e-scooter" if i % 2 == 0 else "bicycle"
            in_ped_zone = (i % 3 != 0)  # 2 out of 3 in pedestrian zone
            legal_hdg = 180.0
            obs_hdg = 0.0 if (i % 2 == 0) else 180.0  # Some wrong-way, some right-way on sidewalk
            is_ww = obs_hdg != legal_hdg
            speed = 22.0 + (i % 5)
            limit = 20.0
            rec = TrafficTelemetryRecord(
                incident_id=cid,
                timestamp=now_iso,
                entity_type=entity,
                lane_id="PedZone-North" if in_ped_zone else "BikeLane-SB",
                lane_legal_heading=legal_hdg,
                observed_heading=obs_hdg,
                estimated_speed_kmh=speed,
                speed_limit_kmh=limit,
                trajectory_points=[(0.0, 0.0), (1.0, 2.0)],
                dwell_time_seconds=3.0,
                pedestrian_zone=in_ped_zone,
            )
            cases.append(rec)
            if is_ww and in_ped_zone:
                risk = RiskLevel.CRITICAL
            elif is_ww or in_ped_zone:
                risk = RiskLevel.HIGH
            else:
                risk = RiskLevel.MEDIUM

            ground_truths.append(
                IncidentAuditVerdict(
                    incident_id=cid,
                    timestamp=now_iso,
                    entity_type=entity,
                    lane_id="PedZone-North" if in_ped_zone else "BikeLane-SB",
                    is_wrong_way=is_ww,
                    heading_degrees=obs_hdg,
                    estimated_speed_kmh=speed,
                    speed_limit_kmh=limit,
                    risk_level=risk,
                    violation_detected=True,
                    summary=f"Micromobility infraction on {entity}.",
                    reasoning=f"Pedestrian zone encroachment or counter-flow.",
                )
            )

        # 3. 20 Speeding Infractions (Correct direction, high speed)
        for i in range(20):
            cid = f"CASE-SPD-{i+1:03d}"
            entity = "car" if i % 2 == 0 else "motorcycle"
            legal_hdg = 0.0
            obs_hdg = 0.0  # Aligned
            limit = 30.0
            speed = 52.0 + (i * 1.5)  # Well above speed limit
            rec = TrafficTelemetryRecord(
                incident_id=cid,
                timestamp=now_iso,
                entity_type=entity,
                lane_id="Lane-3-NB",
                lane_legal_heading=legal_hdg,
                observed_heading=obs_hdg,
                estimated_speed_kmh=speed,
                speed_limit_kmh=limit,
                trajectory_points=[(0.0, 0.0), (3.0, 6.0)],
                dwell_time_seconds=1.2,
                pedestrian_zone=False,
            )
            cases.append(rec)
            risk = RiskLevel.HIGH if speed > limit + 20.0 else RiskLevel.MEDIUM
            ground_truths.append(
                IncidentAuditVerdict(
                    incident_id=cid,
                    timestamp=now_iso,
                    entity_type=entity,
                    lane_id="Lane-3-NB",
                    is_wrong_way=False,
                    heading_degrees=obs_hdg,
                    estimated_speed_kmh=speed,
                    speed_limit_kmh=limit,
                    risk_level=risk,
                    violation_detected=True,
                    summary=f"Excessive speed infraction.",
                    reasoning=f"Speed {speed:.1f} km/h exceeds {limit:.1f} km/h limit.",
                )
            )

        # 4. 20 Benign Control Traffic
        for i in range(20):
            cid = f"CASE-BENIGN-{i+1:03d}"
            entity = ["car", "bicycle", "pedestrian", "motorcycle"][i % 4]
            legal_hdg = 270.0
            obs_hdg = 270.0
            limit = 30.0
            speed = 22.0 if entity != "pedestrian" else 4.5
            rec = TrafficTelemetryRecord(
                incident_id=cid,
                timestamp=now_iso,
                entity_type=entity,
                lane_id="Lane-4-WB" if entity != "pedestrian" else "Crosswalk-WB",
                lane_legal_heading=legal_hdg,
                observed_heading=obs_hdg,
                estimated_speed_kmh=speed,
                speed_limit_kmh=limit,
                trajectory_points=[(0.0, 0.0), (1.0, 2.0)],
                dwell_time_seconds=1.5,
                pedestrian_zone=False,
            )
            cases.append(rec)
            ground_truths.append(
                IncidentAuditVerdict(
                    incident_id=cid,
                    timestamp=now_iso,
                    entity_type=entity,
                    lane_id="Lane-4-WB" if entity != "pedestrian" else "Crosswalk-WB",
                    is_wrong_way=False,
                    heading_degrees=obs_hdg,
                    estimated_speed_kmh=speed,
                    speed_limit_kmh=limit,
                    risk_level=RiskLevel.LOW,
                    violation_detected=False,
                    summary=f"Compliant transit.",
                    reasoning=f"Normal heading and compliant speed.",
                )
            )

        return cases, ground_truths

    def evaluate(
        self,
        cases: Optional[List[TrafficTelemetryRecord]] = None,
        ground_truths: Optional[List[IncidentAuditVerdict]] = None,
    ) -> Dict[str, Any]:
        """Sequentially evaluate cases verifying parsing, accuracy, and memory ceiling (R19, R20, R21)."""
        if cases is None or ground_truths is None:
            cases, ground_truths = self.generate_100_cases()

        total = len(cases)
        if total == 0:
            raise AuditorVerificationError("Cannot evaluate empty test suite")

        process = psutil.Process(os.getpid())
        peak_rss_mb = process.memory_info().rss / (1024 * 1024)

        results: List[AuditorCaseResult] = []
        valid_json_count = 0
        wrong_way_correct = 0
        risk_level_correct = 0

        for idx, (rec, gt) in enumerate(zip(cases, ground_truths)):
            t0 = time.perf_counter()
            err_msg = None
            pred_verdict: Optional[IncidentAuditVerdict] = None

            try:
                pred_verdict = self.engine.audit_incident(rec)
                valid_json_count += 1
            except Exception as e:
                err_msg = str(e)

            t1 = time.perf_counter()
            latency = t1 - t0

            # Memory tracking
            current_rss = process.memory_info().rss / (1024 * 1024)
            peak_rss_mb = max(peak_rss_mb, current_rss)

            is_ww_pred = pred_verdict.is_wrong_way if pred_verdict else None
            risk_pred = pred_verdict.risk_level.value if pred_verdict else None

            if pred_verdict:
                if pred_verdict.is_wrong_way == gt.is_wrong_way:
                    wrong_way_correct += 1
                if pred_verdict.risk_level == gt.risk_level:
                    risk_level_correct += 1

            tok_est = len(pred_verdict.model_dump_json()) // 4 if pred_verdict else 0

            case_res = AuditorCaseResult(
                case_id=rec.incident_id,
                parse_success=pred_verdict is not None,
                is_wrong_way_expected=gt.is_wrong_way,
                is_wrong_way_predicted=is_ww_pred,
                risk_level_expected=gt.risk_level.value,
                risk_level_predicted=risk_pred,
                generation_latency_seconds=round(latency, 4),
                tokens_generated=tok_est,
                error_message=err_msg,
            )
            results.append(case_res)

        # Compute summary percentages
        parse_rate = (valid_json_count / total) * 100.0
        ww_acc = (wrong_way_correct / total) * 100.0
        risk_acc = (risk_level_correct / total) * 100.0
        within_memory = peak_rss_mb <= 1200.0

        summary = {
            "total_cases": total,
            "schema_parse_success_rate": round(parse_rate, 2),
            "achieved_100_pct_parse": parse_rate == 100.0,
            "wrong_way_accuracy_pct": round(ww_acc, 2),
            "achieved_wrong_way_threshold": ww_acc >= 90.0,
            "risk_level_accuracy_pct": round(risk_acc, 2),
            "achieved_risk_threshold": risk_acc >= 85.0,
            "peak_rss_mb": round(peak_rss_mb, 2),
            "within_memory_budget": within_memory,
            "case_results": [r.model_dump() for r in results],
        }

        logger.info(
            f"100-Case Evaluation Complete: Parse Rate={parse_rate:.1f}%, "
            f"Wrong-way Acc={ww_acc:.1f}%, Risk Acc={risk_acc:.1f}%, "
            f"Peak RSS={peak_rss_mb:.1f}MB (Budget Met={within_memory})"
        )

        return summary


def main() -> None:
    """CLI entrypoint for 100-case deterministic infraction evaluation (R23)."""
    parser = argparse.ArgumentParser(
        description="SVA Sentinel: 100-Case Deterministic Auditor Evaluator",
    )
    parser.add_argument(
        "--model-path",
        default="models/quantized/llamacpp/qwen2.5-0.5b-auditor-q4_k_m.gguf",
        help="Path to quantized GGUF model artifact",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Operate in offline mock mode without native llama runtime",
    )
    parser.add_argument(
        "--output-json",
        default=None,
        help="Destination path for evaluation summary JSON",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    engine = ForensicAuditorEngine(
        model_path=args.model_path if not args.mock else None,
        mock_mode=args.mock,
    )
    evaluator = AuditorCaseEvaluator(engine=engine, mock_mode=args.mock)
    summary = evaluator.evaluate()

    if args.output_json:
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        print(f"Summary written to {out_path}")
    else:
        # Print high level summary without 100 case array
        display_summary = dict(summary)
        display_summary.pop("case_results", None)
        print(json.dumps(display_summary, indent=2))


if __name__ == "__main__":
    main()
