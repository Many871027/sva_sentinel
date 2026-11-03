"""Unit tests for base model acquisition subsystem (T10).

Covers:
- R1: Base model acquisition via huggingface_hub.
- R2: Structured storage in models/base/<model_name>/ with weights and metadata.json.
- R3: SHA256 checksum verification on download.
- R4: Deletion and ModelChecksumError upon hash mismatch.
- R5: HTTP resume capability (resume_download=True).
- R6: Offline execution mode utilizing local weights.
- R14: ModelNotFoundError with valid available model list.
- R16: Mock network calls with zero external network egress.
- R17: Fast execution under 5.0 seconds.
"""

from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from src.core.exceptions import ModelChecksumError, ModelNotFoundError, ModelRegistryError
from src.engine.download_base_models import BaseModelDownloader, compute_sha256
from src.pipeline.schemas import BaseModelProfile
from tests.fixtures.mock_models import create_mock_pytorch_weights, setup_mock_fixtures


class TestModelAcquisition(unittest.TestCase):
    """Test suite validating checkpoint acquisition, checksum verification, and offline fallback."""

    def setUp(self) -> None:
        self.temp_dir = tempfile.mkdtemp(prefix="test_sva_model_acq_")
        self.base_dir = Path(self.temp_dir) / "models" / "base"
        self.fixtures_dir = Path(self.temp_dir) / "fixtures"

        self.mock_data = setup_mock_fixtures(self.fixtures_dir)

        # Custom mock profiles with checksums matching our mock fixtures
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
                parameter_count=490000,
            ),
        }

        self.downloader = BaseModelDownloader(
            base_dir=self.base_dir,
            profiles=self.test_profiles,
        )

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_r1_and_r5_download_hf_hub_with_resume(self) -> None:
        """Verify R1 & R5: Checkpoint downloaded via hf_hub_download with resume_download=True."""
        fixture_path = self.mock_data["yolo11n"]["path"]

        def mock_hf_download(repo_id: str, filename: str, local_dir: str, resume_download: bool) -> str:
            self.assertTrue(resume_download, "Per R5, resume_download must be explicitly enabled")
            dest = Path(local_dir) / filename
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(fixture_path, dest)
            return str(dest)

        with patch("src.engine.download_base_models.hf_hub_download", side_effect=mock_hf_download):
            downloaded = self.downloader.download("yolo11n")

            self.assertTrue(downloaded.is_file())
            self.assertEqual(downloaded.name, "yolo11n.pt")
            self.assertEqual(
                compute_sha256(downloaded),
                self.test_profiles["yolo11n"].expected_sha256,
            )

    def test_r2_structured_storage_and_metadata(self) -> None:
        """Verify R2: Checkpoints and metadata.json stored under models/base/<model_name>/."""
        fixture_path = self.mock_data["yolo11n"]["path"]

        with patch("src.engine.download_base_models.hf_hub_download") as mock_hf:
            target_dest = self.base_dir / "yolo11n" / "yolo11n.pt"
            target_dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(fixture_path, target_dest)
            mock_hf.return_value = str(target_dest)

            downloaded = self.downloader.download("yolo11n")

            # Check directory structure
            self.assertEqual(downloaded.parent, self.base_dir / "yolo11n")
            meta_path = downloaded.parent / "metadata.json"
            self.assertTrue(meta_path.is_file(), "metadata.json must be generated in model directory")

            with open(meta_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)

            self.assertEqual(metadata["model_id"], "yolo11n")
            self.assertEqual(metadata["format"], "pt")
            self.assertEqual(metadata["parameter_count"], 2610000)
            self.assertEqual(metadata["sha256"], self.test_profiles["yolo11n"].expected_sha256)

    def test_r3_and_r4_checksum_mismatch_removes_file_and_raises(self) -> None:
        """Verify R3 & R4: SHA256 mismatch raises ModelChecksumError and removes invalid file."""
        corrupted_path = Path(self.temp_dir) / "corrupted.pt"
        with open(corrupted_path, "wb") as f:
            f.write(b"CORRUPTED BYTES DATA")

        target_dest = self.base_dir / "yolo11n" / "yolo11n.pt"

        def mock_hf_download_corrupted(repo_id: str, filename: str, local_dir: str, resume_download: bool) -> str:
            dest = Path(local_dir) / filename
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(corrupted_path, dest)
            return str(dest)

        with patch("src.engine.download_base_models.hf_hub_download", side_effect=mock_hf_download_corrupted):
            with self.assertRaises(ModelChecksumError) as ctx:
                self.downloader.download("yolo11n")

            self.assertIn("SHA256 checksum mismatch", str(ctx.exception))
            # Confirm invalid file was unlinked per R4
            self.assertFalse(target_dest.exists(), "Corrupted file must be removed upon checksum failure")

    def test_r6_offline_mode_success(self) -> None:
        """Verify R6: Offline mode resolves strictly from local storage when valid file exists."""
        target_dest = self.base_dir / "yolo11n" / "yolo11n.pt"
        target_dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(self.mock_data["yolo11n"]["path"], target_dest)

        # Do not mock hf_hub_download to ensure it is NEVER called
        with patch("src.engine.download_base_models.hf_hub_download", side_effect=RuntimeError("Network access attempted")):
            downloaded = self.downloader.download("yolo11n", offline=True)
            self.assertEqual(downloaded, target_dest)
            self.assertTrue(downloaded.is_file())

    def test_r6_offline_mode_missing_raises_model_not_found(self) -> None:
        """Verify R6: Offline mode raises ModelNotFoundError when local file is missing."""
        with patch("src.engine.download_base_models.hf_hub_download", side_effect=RuntimeError("Network egress")):
            with self.assertRaises(ModelNotFoundError) as ctx:
                self.downloader.download("yolo11n", offline=True)

            self.assertIn("Offline checkpoint for 'yolo11n' does not exist", str(ctx.exception))

    def test_r14_unknown_model_profile_raises_with_valid_list(self) -> None:
        """Verify R14: Querying an unknown model profile raises ModelNotFoundError with available list."""
        with self.assertRaises(ModelNotFoundError) as ctx:
            self.downloader.get_profile("nonexistent_model_xyz")

        error_msg = str(ctx.exception)
        self.assertIn("nonexistent_model_xyz", error_msg)
        self.assertIn("Available valid models:", error_msg)
        self.assertIn("yolo11n", error_msg)


if __name__ == "__main__":
    unittest.main()
