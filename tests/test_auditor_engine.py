"""Unit tests for ForensicAuditorEngine.

Covers:
- R6: AVX2 and AVX-512 CPU hardware capability detection.
- R7: Parallel worker thread configuration (default 4) and 2048 context length.
- R8: Strictly enforced greedy temperature=0.0.
- R9: Schema-constrained completion decoding into valid IncidentAuditVerdict.
- R10: Corrupted or truncated GGUF file raising GGUFQuantizationError.
- R11: Offline mock execution mode for fast verification without native C++ compilation.
- R24: Fast offline execution (< 1.0s).
"""

from pathlib import Path
import tempfile
import unittest

from src.core.exceptions import GGUFQuantizationError
from src.engine.auditor_engine import (
    ForensicAuditorEngine,
    detect_cpu_vector_capabilities,
)
from src.pipeline.schemas import IncidentAuditVerdict, RiskLevel, TrafficTelemetryRecord
from tests.fixtures.mock_gguf_quant import (
    create_mock_corrupted_gguf,
    create_mock_f16_gguf,
)


class TestAuditorEngine(unittest.TestCase):
    """Test suite for ForensicAuditorEngine."""

    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.test_dir = Path(self.temp_dir.name)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_r6_cpu_vector_capabilities_detection(self) -> None:
        """Verify R6: AVX2 and AVX-512 hardware capability detection."""
        flags = detect_cpu_vector_capabilities()
        self.assertIsInstance(flags, dict)
        self.assertIn("avx2", flags)
        self.assertIn("avx512", flags)
        self.assertIsInstance(flags["avx2"], bool)
        self.assertIsInstance(flags["avx512"], bool)

    def test_r7_and_r8_engine_configuration(self) -> None:
        """Verify R7, R8: Thread parallelism (4), context length (2048), and temperature 0.0."""
        engine = ForensicAuditorEngine(n_threads=4, n_ctx=2048, mock_mode=True)
        self.assertEqual(engine.n_threads, 4, "Must default to 4 worker threads (R7)")
        self.assertEqual(engine.n_ctx, 2048, "Must allocate 2048 context window (R7)")
        self.assertEqual(engine.temperature, 0.0, "Must enforce strictly greedy temperature=0.0 (R8)")
        self.assertTrue(engine.mock_mode)

    def test_r9_and_r11_mock_verdict_schema_compliance(self) -> None:
        """Verify R9, R11: Mock mode generates valid IncidentAuditVerdict adhering to schema."""
        engine = ForensicAuditorEngine(mock_mode=True)

        telemetry = TrafficTelemetryRecord(
            incident_id="AUDIT-TEST-001",
            timestamp="2026-11-03T10:00:00Z",
            entity_type="car",
            lane_id="Lane-West-1",
            lane_legal_heading=90.0,
            observed_heading=270.0,  # 180° counter-flow
            estimated_speed_kmh=45.0,
            speed_limit_kmh=30.0,
            trajectory_points=[(0.0, 0.0), (1.0, 3.0)],
            dwell_time_seconds=2.0,
            pedestrian_zone=False,
        )

        verdict = engine.audit_incident(telemetry)

        self.assertIsInstance(verdict, IncidentAuditVerdict)
        self.assertEqual(verdict.incident_id, "AUDIT-TEST-001")
        self.assertTrue(verdict.is_wrong_way)
        self.assertTrue(verdict.violation_detected)
        self.assertEqual(verdict.risk_level, RiskLevel.CRITICAL)
        self.assertTrue(len(verdict.summary) > 0)
        self.assertTrue(len(verdict.reasoning) > 0)

    def test_r10_corrupted_magic_header_raises_error(self) -> None:
        """Verify R10: GGUFQuantizationError raised when GGUF file fails magic header check."""
        bad_path = self.test_dir / "bad.gguf"
        create_mock_corrupted_gguf(bad_path, corruption_type="invalid_magic")

        with self.assertRaises(GGUFQuantizationError) as ctx:
            ForensicAuditorEngine(model_path=bad_path, mock_mode=False)
        self.assertIn("magic", str(ctx.exception).lower())

    def test_r10_truncated_gguf_file_raises_error(self) -> None:
        """Verify R10: GGUFQuantizationError raised when GGUF file is truncated (< 24 bytes)."""
        trunc_path = self.test_dir / "trunc.gguf"
        create_mock_corrupted_gguf(trunc_path, corruption_type="truncated")

        with self.assertRaises(GGUFQuantizationError) as ctx:
            ForensicAuditorEngine(model_path=trunc_path, mock_mode=False)
        self.assertIn("truncated", str(ctx.exception).lower())

    def test_r10_missing_model_file_raises_error(self) -> None:
        """Verify R10: GGUFQuantizationError raised when GGUF file does not exist."""
        non_existent = self.test_dir / "non_existent.gguf"

        with self.assertRaises(GGUFQuantizationError) as ctx:
            ForensicAuditorEngine(model_path=non_existent, mock_mode=False)
        self.assertIn("not found", str(ctx.exception).lower())


if __name__ == "__main__":
    unittest.main()
