"""End-to-end integration test for model acquisition, verification, and registry pipeline (T13).

Covers:
- R1: Checkpoint acquisition via huggingface_hub.
- R2: Structured storage in models/base/<model_name>/.
- R3: SHA256 verification.
- R7: Full coverage of target model profiles (Teacher, Student, LLM).
- R8: Centralized registry catalog persistence in models/base/registry.json.
- R9: Complete registry attributes and parameter counts.
- R10: Baseline forward inference on intersection frames.
- R11: FP32 latency benchmarking over warm runs.
- R12: Peak process RSS memory profiling via psutil.
- R13: Baseline zero-shot mAP@50 and mAP@50-95 calculation.
- R16: Mock network calls with zero external network egress.
- R17: Complete pipeline execution well under 5.0 seconds.
- R18: CLI entrypoint execution (main()).
"""

from __future__ import annotations

import json
import shutil
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from src.engine.download_base_models import BaseModelDownloader, compute_sha256, main as cli_main
from src.engine.model_registry import ModelRegistry
from src.pipeline.schemas import BaseModelProfile, ModelRegistryCatalog
from tests.fixtures.mock_models import setup_mock_fixtures


class TestModelPipelineIntegration(unittest.TestCase):
    """End-to-end integration test executing the complete model lifecycle."""

    def setUp(self) -> None:
        self.temp_dir = tempfile.mkdtemp(prefix="test_sva_e2e_model_pipeline_")
        self.base_dir = Path(self.temp_dir) / "models" / "base"
        self.registry_file = self.base_dir / "registry.json"
        self.fixtures_dir = Path(self.temp_dir) / "fixtures"

        self.mock_data = setup_mock_fixtures(self.fixtures_dir)

        # Profiles configured with the mock fixtures' exact SHA256
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

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_full_pipeline_e2e_under_5_seconds(self) -> None:
        """Execute mock download -> verify -> register -> benchmark -> persist in < 5.0s (R17)."""
        start_time = time.perf_counter()

        downloader = BaseModelDownloader(
            base_dir=self.base_dir,
            profiles=self.test_profiles,
        )
        registry = ModelRegistry(
            registry_path=self.registry_file,
            models_dir=self.base_dir,
            profiles=self.test_profiles,
        )

        # 1. Acquire all models using mock hf_hub_download (zero external egress: R1, R16)
        def mock_hf_download(repo_id: str, filename: str, local_dir: str, resume_download: bool) -> str:
            # Locate matching fixture
            for mid, data in self.mock_data.items():
                if data["file_name"] == filename:
                    dest = Path(local_dir) / filename
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(data["path"], dest)
                    return str(dest)
            raise FileNotFoundError(f"No mock fixture for {filename}")

        with patch("src.engine.download_base_models.hf_hub_download", side_effect=mock_hf_download):
            acquired_paths = downloader.download_all()

        self.assertEqual(len(acquired_paths), 3)

        # 2. Verify SHA256 and structured storage for all models (R2, R3, R7)
        for model_id, weights_path in acquired_paths.items():
            profile = self.test_profiles[model_id]
            self.assertTrue(weights_path.is_file())
            self.assertEqual(compute_sha256(weights_path), profile.expected_sha256)
            self.assertTrue((weights_path.parent / "metadata.json").is_file())

        # 3. Create sample intersection test frames and ground truth annotations (R10, R13)
        sample_frames = [
            np.zeros((640, 640, 3), dtype=np.uint8),
            np.zeros((640, 640, 3), dtype=np.uint8),
        ]
        sample_gts = [
            [
                {"bbox": [96.0, 128.0, 288.0, 384.0], "class_id": 0},  # matches synthetic car
                {"bbox": [384.0, 224.0, 480.0, 448.0], "class_id": 3},  # matches synthetic scooter
            ],
            [
                {"bbox": [96.0, 128.0, 288.0, 384.0], "class_id": 0},
            ],
        ]

        # 4. Benchmark & Register Student Detector (yolo11n)
        entry_student = registry.benchmark_and_register(
            model_id="yolo11n",
            weights_path=acquired_paths["yolo11n"],
            sample_frames=sample_frames,
            ground_truths=sample_gts,
            warm_runs=5,
        )

        self.assertIsNotNone(entry_student.benchmarks)
        self.assertGreater(entry_student.benchmarks.mean_latency_ms, 0.0)
        self.assertGreater(entry_student.benchmarks.p95_latency_ms, 0.0)
        self.assertGreater(entry_student.benchmarks.peak_memory_rss_mb, 0.0)
        self.assertIsNotNone(entry_student.benchmarks.map_50)
        self.assertIsNotNone(entry_student.benchmarks.map_50_95)
        self.assertEqual(entry_student.parameter_count, 2610000)

        # 5. Register Teacher and LLM (R8, R9)
        entry_teacher = registry.register_model(
            model_id="yolo11x",
            weights_path=acquired_paths["yolo11x"],
        )
        self.assertEqual(entry_teacher.parameter_count, 56900000)

        entry_llm = registry.register_model(
            model_id="qwen2.5-0.5b-instruct",
            weights_path=acquired_paths["qwen2.5-0.5b-instruct"],
        )
        self.assertEqual(entry_llm.format, "safetensors")

        # 6. Verify Centralized Registry File (R8)
        self.assertTrue(self.registry_file.is_file())
        with open(self.registry_file, "r", encoding="utf-8") as f:
            catalog_dict = json.load(f)

        validated_catalog = ModelRegistryCatalog.model_validate(catalog_dict)
        self.assertEqual(len(validated_catalog.models), 3)
        self.assertIn("yolo11n", validated_catalog.models)
        self.assertIn("yolo11x", validated_catalog.models)
        self.assertIn("qwen2.5-0.5b-instruct", validated_catalog.models)

        # 7. Test CLI entrypoint (R18)
        with patch.dict("src.engine.download_base_models.TARGET_MODEL_PROFILES", self.test_profiles):
            rc_list = cli_main(["--list"])
            self.assertEqual(rc_list, 0)

            rc_verify = cli_main(["--model", "yolo11n", "--verify", "--base-dir", str(self.base_dir)])
            self.assertEqual(rc_verify, 0)

        # 8. Assert overall runtime duration < 5.0 seconds (R17)
        duration = time.perf_counter() - start_time
        print(f"\n[E2E Pipeline Timing] Completed in {duration:.3f}s")
        self.assertLess(
            duration,
            5.0,
            f"End-to-end integration test exceeded 5.0s budget: {duration:.3f}s",
        )


if __name__ == "__main__":
    unittest.main()
