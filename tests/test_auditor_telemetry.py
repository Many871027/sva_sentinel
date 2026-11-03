"""Unit tests for synthetic telemetry generation and forensic auditor schema validation.

Covers:
- R7: TrafficTelemetryRecord schema validation and canonical entity restrictions.
- R8: IncidentAuditVerdict forensic schema validation and field constraints.
- R9: Deterministic telemetry dataset generation with seed control.
- R10: Infraction scenario distribution balancing (wrong-way, micromobility, speeding, benign).
- R11: JSONL serialization formatting and ChatML chat template structure.
- R12: Statistical calculation and 100% schema compliance.
- R22: Zero-network sub-second offline execution.
"""

import json
from pathlib import Path
import tempfile
import time
import unittest

from pydantic import ValidationError

from src.core.exceptions import TelemetryGenerationError
from src.pipeline.schemas import (
    AuditorTrainingSample,
    IncidentAuditVerdict,
    RiskLevel,
    TrafficTelemetryRecord,
)
from src.pipeline.telemetry_generator import (
    TelemetryGenerator,
    generate_synthetic_telemetry_dataset,
)
from tests.fixtures.mock_telemetry import (
    create_mock_telemetry_record,
    create_mock_verdict,
)


class TestAuditorTelemetry(unittest.TestCase):
    """Test suite verifying telemetry generation, schema constraints, and distribution balancing."""

    def test_r7_and_r8_telemetry_schema_validation(self) -> None:
        """Verify R7 & R8: Valid telemetry serialization and rejection of invalid entities/angles."""
        valid_rec = create_mock_telemetry_record(
            incident_id="INC-001",
            entity="car",
            lane_id="Lane-1-EB",
            legal_heading=90.0,
            observed_heading=270.0,
            speed=32.5,
        )
        self.assertEqual(valid_rec.entity_type, "car")
        prompt = valid_rec.format_telemetry_prompt()
        self.assertIn("Incident ID: INC-001", prompt)
        self.assertIn("Entity: car", prompt)
        self.assertIn("Angular Delta: 180.0°", prompt)

        # Invalid entity type must fail validation (R7)
        with self.assertRaises(ValidationError):
            create_mock_telemetry_record(entity="drone")

        # Invalid heading degrees [0, 360) must fail validation (R8)
        with self.assertRaises(ValidationError):
            TrafficTelemetryRecord(
                incident_id="INC-BAD",
                entity_type="car",
                lane_id="Lane-1",
                lane_legal_heading=400.0,  # Invalid
                observed_heading=90.0,
                estimated_speed_kmh=20.0,
            )

    def test_r8_verdict_schema_validation(self) -> None:
        """Verify R8: IncidentAuditVerdict fields, summary length limit, and risk level enum."""
        verdict = create_mock_verdict(
            incident_id="INC-002",
            entity="motorcycle",
            risk_level=RiskLevel.CRITICAL,
        )
        self.assertTrue(verdict.is_wrong_way)
        self.assertEqual(verdict.risk_level, RiskLevel.CRITICAL)

        # Summary exceeding 300 chars must fail
        with self.assertRaises(ValidationError):
            IncidentAuditVerdict(
                incident_id="INC-BAD",
                timestamp="2026-11-02T10:00:00Z",
                entity_type="car",
                lane_id="Lane-1",
                is_wrong_way=True,
                heading_degrees=90.0,
                estimated_speed_kmh=30.0,
                speed_limit_kmh=25.0,
                risk_level=RiskLevel.HIGH,
                violation_detected=True,
                summary="A" * 305,  # Exceeds max_length=300
                reasoning="Reasoning test",
            )

    def test_r9_deterministic_dataset_generation(self) -> None:
        """Verify R9: Telemetry generator yields byte-identical output given identical seeds."""
        gen1 = TelemetryGenerator(seed=123)
        gen2 = TelemetryGenerator(seed=123)

        ds1 = gen1.generate_dataset(num_samples=50)
        ds2 = gen2.generate_dataset(num_samples=50)

        self.assertEqual(len(ds1), 50)
        self.assertEqual(len(ds2), 50)
        for s1, s2 in zip(ds1, ds2):
            self.assertEqual(s1.sample_id, s2.sample_id)
            self.assertEqual(s1.telemetry.observed_heading, s2.telemetry.observed_heading)
            self.assertEqual(s1.ground_truth_verdict.risk_level, s2.ground_truth_verdict.risk_level)

    def test_r10_infraction_distribution_balance(self) -> None:
        """Verify R10: 1,000 synthetic samples achieve required distribution quotas."""
        gen = TelemetryGenerator(seed=42)
        samples = gen.generate_dataset(num_samples=1000)
        self.assertEqual(len(samples), 1000)

        stats = gen.calculate_distribution_stats(samples)
        # Verify benign baseline >= 20% (so violations count <= 80%)
        benign_count = 1000 - stats["violations_count"]
        benign_pct = (benign_count / 1000) * 100
        self.assertGreaterEqual(benign_pct, 20.0, "Benign baseline must comprise >= 20% of samples")

        # Verify wrong-way count is approximately 30% of total
        wrong_way_pct = stats["wrong_way_percentage"]
        self.assertGreaterEqual(wrong_way_pct, 25.0)
        self.assertLessEqual(wrong_way_pct, 45.0)

        # Check entity variety
        entities = stats["entity_distribution"]
        self.assertIn("car", entities)
        self.assertIn("bicycle", entities)
        self.assertIn("e-scooter", entities)
        self.assertIn("motorcycle", entities)

    def test_r11_jsonl_serialization(self) -> None:
        """Verify R11: Dataset serializes to JSONL with required ChatML message structure."""
        with tempfile.TemporaryDirectory() as tmpdir:
            out_file = Path(tmpdir) / "telemetry_train.jsonl"
            gen = TelemetryGenerator(seed=42)
            samples = gen.generate_dataset(num_samples=25)
            saved_path = gen.save_jsonl(samples, out_file)

            self.assertTrue(saved_path.exists())
            with open(saved_path, "r", encoding="utf-8") as f:
                lines = f.readlines()

            self.assertEqual(len(lines), 25)
            first_record = json.loads(lines[0])
            self.assertIn("sample_id", first_record)
            self.assertIn("messages", first_record)
            self.assertIn("telemetry", first_record)
            self.assertIn("ground_truth", first_record)

            messages = first_record["messages"]
            self.assertEqual(len(messages), 3)
            self.assertEqual(messages[0]["role"], "system")
            self.assertEqual(messages[1]["role"], "user")
            self.assertEqual(messages[2]["role"], "assistant")

    def test_r12_distribution_stats_computation(self) -> None:
        """Verify R12: calculate_distribution_stats handles empty and populated sets accurately."""
        gen = TelemetryGenerator(seed=42)
        empty_stats = gen.calculate_distribution_stats([])
        self.assertEqual(empty_stats["total"], 0)

        samples = gen.generate_dataset(num_samples=100)
        stats = gen.calculate_distribution_stats(samples)
        self.assertEqual(stats["total_samples"], 100)
        self.assertIn("violations_percentage", stats)
        self.assertIn("risk_distribution", stats)

    def test_r22_fast_execution_speed(self) -> None:
        """Verify R22: Generating 1,000 synthetic samples executes in < 1.0s."""
        start = time.perf_counter()
        samples, _ = generate_synthetic_telemetry_dataset(num_samples=1000, seed=42)
        elapsed = time.perf_counter() - start

        self.assertEqual(len(samples), 1000)
        self.assertLess(elapsed, 1.0, f"Expected generation in < 1.0s, took {elapsed:.3f}s")


if __name__ == "__main__":
    unittest.main()
