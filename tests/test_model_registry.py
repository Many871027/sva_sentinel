"""Unit tests for Model Registry catalog manager (T11).

Covers:
- R7: Predefined catalog profiles (Teacher yolo11x, Student yolo11n, LLM qwen2.5-0.5b-instruct).
- R8: Centralized registry persistence in models/base/registry.json.
- R9: Registry entry attributes (id, family, version, format, sha256, params, size, resolution).
- R14: ModelNotFoundError on unregistered model queries.
- R17: Automated test suite execution under 5.0 seconds.
"""

from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path

from src.core.exceptions import ModelChecksumError, ModelNotFoundError
from src.engine.model_registry import (
    ModelRegistry,
    count_parameters,
    count_parameters_safetensors,
)
from src.pipeline.schemas import (
    TARGET_MODEL_PROFILES,
    BaseModelProfile,
    ModelBenchmarkMetrics,
    ModelRegistryCatalog,
)
from tests.fixtures.mock_models import create_mock_safetensors, setup_mock_fixtures


class TestModelRegistry(unittest.TestCase):
    """Test suite validating model registration, catalog schemas, and parameter counting."""

    def setUp(self) -> None:
        self.temp_dir = tempfile.mkdtemp(prefix="test_sva_registry_")
        self.base_dir = Path(self.temp_dir) / "models" / "base"
        self.registry_file = self.base_dir / "registry.json"
        self.fixtures_dir = Path(self.temp_dir) / "fixtures"

        self.mock_data = setup_mock_fixtures(self.fixtures_dir)

        self.test_profiles = {
            "yolo11n": BaseModelProfile(
                model_id="yolo11n",
                model_name="YOLO11n Lightweight Student Detector",
                model_role="student",
                model_family="yolo11",
                version="1.0.0",
                format="pt",
                source_repo="ultralytics/yolo11n",
                file_name="yolo11n.pt",
                expected_sha256=self.mock_data["yolo11n"]["sha256"],
                file_size_bytes=self.mock_data["yolo11n"]["path"].stat().st_size,
                input_resolution=[640, 640],
                parameter_count=2610000,
            ),
            "yolo11x": BaseModelProfile(
                model_id="yolo11x",
                model_name="YOLO11x High-Capacity Teacher Detector",
                model_role="teacher",
                model_family="yolo11",
                version="1.0.0",
                format="pt",
                source_repo="ultralytics/yolo11x",
                file_name="yolo11x.pt",
                expected_sha256=self.mock_data["yolo11x"]["sha256"],
                file_size_bytes=self.mock_data["yolo11x"]["path"].stat().st_size,
                input_resolution=[640, 640],
                parameter_count=56900000,
            ),
            "qwen2.5-0.5b-instruct": BaseModelProfile(
                model_id="qwen2.5-0.5b-instruct",
                model_name="Qwen2.5-0.5B-Instruct Forensic Auditor",
                model_role="auditor",
                model_family="qwen2.5",
                version="1.0.0",
                format="safetensors",
                source_repo="Qwen/Qwen2.5-0.5B-Instruct",
                file_name="model.safetensors",
                expected_sha256=self.mock_data["qwen2.5-0.5b-instruct"]["sha256"],
                file_size_bytes=self.mock_data["qwen2.5-0.5b-instruct"]["path"].stat().st_size,
                input_resolution=[2048],
                parameter_count=self.mock_data["qwen2.5-0.5b-instruct"]["parameter_count"],
            ),
        }

        self.registry = ModelRegistry(
            registry_path=self.registry_file,
            models_dir=self.base_dir,
            profiles=self.test_profiles,
        )

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_r7_target_model_profiles_declared(self) -> None:
        """Verify R7: Predefined catalog profiles exist for yolo11x, yolo11n, and qwen2.5-0.5b-instruct."""
        declared_ids = set(TARGET_MODEL_PROFILES.keys())
        self.assertIn("yolo11x", declared_ids)
        self.assertIn("yolo11n", declared_ids)
        self.assertIn("qwen2.5-0.5b-instruct", declared_ids)

        teacher = TARGET_MODEL_PROFILES["yolo11x"]
        self.assertEqual(teacher.model_role, "teacher")
        self.assertEqual(teacher.format, "pt")

        student = TARGET_MODEL_PROFILES["yolo11n"]
        self.assertEqual(student.model_role, "student")
        self.assertEqual(student.format, "pt")

        auditor = TARGET_MODEL_PROFILES["qwen2.5-0.5b-instruct"]
        self.assertEqual(auditor.model_role, "auditor")
        self.assertEqual(auditor.format, "safetensors")

    def test_r8_and_r9_register_model_and_persist_manifest(self) -> None:
        """Verify R8 & R9: Register model attributes and persist valid registry.json."""
        weights_path = self.mock_data["yolo11n"]["path"]

        entry = self.registry.register_model(
            model_id="yolo11n",
            weights_path=weights_path,
        )

        # Validate R9 fields
        self.assertEqual(entry.model_id, "yolo11n")
        self.assertEqual(entry.model_family, "yolo11")
        self.assertEqual(entry.version, "1.0.0")
        self.assertEqual(entry.format, "pt")
        self.assertEqual(entry.parameter_count, 2610000)
        self.assertEqual(entry.file_size_bytes, weights_path.stat().st_size)
        self.assertEqual(entry.sha256, self.test_profiles["yolo11n"].expected_sha256)
        self.assertEqual(entry.input_resolution, [640, 640])
        self.assertTrue(entry.registered_at)

        # Validate R8 centralized persistence
        self.assertTrue(self.registry_file.is_file(), "registry.json must be written to disk")
        with open(self.registry_file, "r", encoding="utf-8") as f:
            raw_json = f.read()

        # Validate against Pydantic schema
        catalog = ModelRegistryCatalog.model_validate_json(raw_json)
        self.assertIn("yolo11n", catalog.models)
        self.assertEqual(catalog.models["yolo11n"].model_name, entry.model_name)

    def test_r9_parameter_counter_safetensors(self) -> None:
        """Verify R9: count_parameters parses SafeTensors binary header accurately."""
        safetensors_file = self.mock_data["qwen2.5-0.5b-instruct"]["path"]
        expected_params = self.mock_data["qwen2.5-0.5b-instruct"]["parameter_count"]

        counted = count_parameters_safetensors(safetensors_file)
        self.assertEqual(counted, expected_params)

        general_counted = count_parameters(safetensors_file, model_format="safetensors")
        self.assertEqual(general_counted, expected_params)

    def test_r9_parameter_counter_mock_pytorch(self) -> None:
        """Verify R9: count_parameters handles PyTorch/mock checkpoints."""
        pt_file = self.mock_data["yolo11n"]["path"]
        profile = self.test_profiles["yolo11n"]

        counted = count_parameters(pt_file, model_format="pt", profile=profile)
        self.assertEqual(counted, 2610000)

    def test_r14_get_unregistered_model_raises_model_not_found(self) -> None:
        """Verify R14: get_model on unregistered identifier raises ModelNotFoundError with registered list."""
        with self.assertRaises(ModelNotFoundError) as ctx:
            self.registry.get_model("unknown_model_123")

        self.assertIn("unknown_model_123", str(ctx.exception))
        self.assertIn("Registered models:", str(ctx.exception))

    def test_query_by_role_and_family(self) -> None:
        """Verify query filtering by role (teacher/student/auditor) and family."""
        self.registry.register_model("yolo11n", self.mock_data["yolo11n"]["path"])
        self.registry.register_model("yolo11x", self.mock_data["yolo11x"]["path"])
        self.registry.register_model("qwen2.5-0.5b-instruct", self.mock_data["qwen2.5-0.5b-instruct"]["path"])

        students = self.registry.query_by_role("student")
        self.assertEqual(len(students), 1)
        self.assertEqual(students[0].model_id, "yolo11n")

        teachers = self.registry.query_by_role("teacher")
        self.assertEqual(len(teachers), 1)
        self.assertEqual(teachers[0].model_id, "yolo11x")

        auditors = self.registry.query_by_role("auditor")
        self.assertEqual(len(auditors), 1)
        self.assertEqual(auditors[0].model_id, "qwen2.5-0.5b-instruct")

        yolo_family = self.registry.query_by_family("yolo11")
        self.assertEqual(len(yolo_family), 2)


if __name__ == "__main__":
    unittest.main()
