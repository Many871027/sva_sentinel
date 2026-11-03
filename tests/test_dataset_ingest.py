"""Unit test suite for dataset ingestion and offline caching.

Validates:
- R1: Dataset ingestion pipeline via huggingface_hub.
- R2: Persistent local caching under data/raw/<dataset_name>/.
- R3: Resume capability using HTTP range requests / snapshot_download resume.
- R4: Offline execution mode operating exclusively from local cache.
- R5: DatasetIngestError for unreachable/invalid datasets without corrupting local state.
- R6: Declarative profiles for target traffic and micromobility datasets.
- R24: Mocking huggingface_hub calls with zero external network egress.
- R25: Ultra-fast execution under 5.0s.
"""

import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from src.core.exceptions import DatasetIngestError
from src.pipeline.dataset_ingest import TARGET_PROFILES, HuggingFaceDatasetIngest


class TestDatasetIngest(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.raw_root = Path(self.temp_dir) / "raw"
        self.raw_root.mkdir(parents=True, exist_ok=True)
        self.ingest = HuggingFaceDatasetIngest(raw_dir=self.raw_root, offline=False)

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    @patch("huggingface_hub.snapshot_download")
    def test_r1_and_r2_download_dataset_caching(self, mock_snapshot_download):
        """Verify R1 & R2: Dataset downloaded to data/raw/<dataset_name>/ and locally cached."""
        dataset_name = "starwit/overhead-traffic-anomalies"
        target_dir = self.ingest.get_dataset_dir(dataset_name)

        def mock_download_side_effect(**kwargs):
            # Simulate downloading files into target_dir
            target_path = Path(kwargs["local_dir"])
            target_path.mkdir(parents=True, exist_ok=True)
            (target_path / "sample.jpg").write_text("dummy image bytes")
            (target_path / "annotations.json").write_text("{}")
            return str(target_path)

        mock_snapshot_download.side_effect = mock_download_side_effect

        result_path = self.ingest.download_dataset(dataset_name)

        self.assertEqual(result_path, target_dir)
        self.assertTrue(result_path.is_dir())
        self.assertTrue((result_path / "sample.jpg").is_file())
        self.assertTrue(self.ingest.is_cached(dataset_name))
        mock_snapshot_download.assert_called_once()

    @patch("huggingface_hub.snapshot_download")
    def test_r3_resume_capability_flag(self, mock_snapshot_download):
        """Verify R3: resume_download is explicitly enabled in snapshot_download invocation."""
        mock_snapshot_download.return_value = "/path"
        dataset_name = "UniqueData/electric-scooters-tracking"

        self.ingest.download_dataset(dataset_name)

        mock_snapshot_download.assert_called_once()
        call_kwargs = mock_snapshot_download.call_args.kwargs
        self.assertTrue(
            call_kwargs.get("resume_download"),
            "snapshot_download must have resume_download=True for resume capability",
        )

    def test_r4_offline_mode_cached_snapshot(self):
        """Verify R4: Offline mode operates exclusively from local cached snapshot without network."""
        dataset_name = "test/offline-dataset"
        cached_dir = self.ingest.get_dataset_dir(dataset_name)
        cached_dir.mkdir(parents=True, exist_ok=True)
        (cached_dir / "frame.jpg").write_text("local frame")

        offline_ingest = HuggingFaceDatasetIngest(raw_dir=self.raw_root, offline=True)

        with patch("huggingface_hub.snapshot_download") as mock_download:
            res = offline_ingest.download_dataset(dataset_name)
            mock_download.assert_not_called()
            self.assertEqual(res, cached_dir)

    def test_r4_offline_mode_missing_cache_raises(self):
        """Verify R4: Offline mode raises DatasetIngestError when cache is missing."""
        offline_ingest = HuggingFaceDatasetIngest(raw_dir=self.raw_root, offline=True)
        with self.assertRaises(DatasetIngestError) as ctx:
            offline_ingest.download_dataset("nonexistent/dataset")
        self.assertIn("Offline mode active", str(ctx.exception))

    @patch("huggingface_hub.snapshot_download")
    def test_r5_unreachable_dataset_raises_ingest_error(self, mock_snapshot_download):
        """Verify R5: Unreachable dataset raises DatasetIngestError without directory corruption."""
        mock_snapshot_download.side_effect = RuntimeError("404 Client Error: Repository Not Found")

        dataset_name = "invalid/unreachable-dataset"
        target_dir = self.ingest.get_dataset_dir(dataset_name)

        with self.assertRaises(DatasetIngestError) as ctx:
            self.ingest.download_dataset(dataset_name)

        self.assertIn("unreachable or does not exist", str(ctx.exception))
        # Ensure target_dir was not left corrupted or partially populated
        if target_dir.exists():
            self.assertFalse(any(target_dir.iterdir()))

    def test_r6_target_declarative_profiles(self):
        """Verify R6: Declarative profiles are provided for target datasets."""
        required_profiles = [
            "starwit/overhead-traffic-anomalies",
            "UniqueData/electric-scooters-tracking",
            "sva/street-level-vehicles",
        ]
        for prof in required_profiles:
            self.assertIn(prof, TARGET_PROFILES)
            cfg = TARGET_PROFILES[prof]
            self.assertEqual(cfg.dataset_name, prof)
            self.assertTrue(len(cfg.allow_patterns) > 0)


if __name__ == "__main__":
    unittest.main()
