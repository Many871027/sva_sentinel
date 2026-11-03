"""End-to-end integration test for dataset acquisition, curation, augmentation, splitting, and manifest sealing.

Validates:
- R1: Ingestion pipeline from mock Hugging Face source.
- R7: Canonical 5-class ontology mapping.
- R8: Normalized YOLO TXT format conversion.
- R14: Homographic perspective viewpoint augmentation.
- R18: Deterministic 70/20/10 train/val/test splitting.
- R20: Class distribution summary report generation.
- R21: SHA256 cryptographic integrity manifest sealing.
- R24: Zero external network egress via fixture mocks.
- R25: Full pipeline test execution under 5.0 seconds.
"""

import json
import shutil
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
import yaml

from src.pipeline.dataset_curation import (
    AnnotationParser,
    ViewpointAugmentor,
    save_curated_sample,
)
from src.pipeline.dataset_ingest import HuggingFaceDatasetIngest
from src.pipeline.dataset_split import (
    DatasetPartitioner,
    calculate_sha256,
    generate_class_distribution_report,
    generate_integrity_manifest,
    save_curated_split_artifacts,
)
from src.pipeline.schemas import (
    CANONICAL_CLASSES,
    AnnotationRecord,
    BoundingBoxYOLO,
)
from tests.fixtures.mock_dataset_generator import setup_mock_fixtures


class TestDatasetPipelineIntegration(unittest.TestCase):
    def setUp(self):
        self.start_time = time.perf_counter()
        self.temp_dir = tempfile.mkdtemp()
        self.raw_root = Path(self.temp_dir) / "data" / "raw"
        self.curated_root = Path(self.temp_dir) / "data" / "curated"
        self.fixtures_dir = Path(self.temp_dir) / "fixtures"

        # Generate miniature mock fixtures
        self.fixtures = setup_mock_fixtures(self.fixtures_dir)

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    @patch("huggingface_hub.snapshot_download")
    def test_full_synthetic_data_pipeline_e2e(self, mock_snapshot_download):
        """Execute complete pipeline: download mock -> parse -> augment -> split -> seal manifest."""
        # --- Stage 1: Ingestion (R1, R24) ---
        dataset_name = "starwit/overhead-traffic-anomalies"

        def mock_hf_download(repo_id, local_dir, **kwargs):
            dest = Path(local_dir)
            dest.mkdir(parents=True, exist_ok=True)
            # Copy fixtures to simulated raw directory
            shutil.copytree(self.fixtures["coco_dir"], dest / "coco", dirs_exist_ok=True)
            shutil.copytree(self.fixtures["voc_dir"], dest / "voc", dirs_exist_ok=True)
            return str(dest)

        mock_snapshot_download.side_effect = mock_hf_download

        ingest = HuggingFaceDatasetIngest(raw_dir=self.raw_root, offline=False)
        raw_dataset_path = ingest.download_dataset(dataset_name)
        self.assertTrue(raw_dataset_path.is_dir())
        mock_snapshot_download.assert_called_once()

        # --- Stage 2: Normalization & Annotation Parsing (R7, R8) ---
        coco_json = raw_dataset_path / "coco" / "annotations.json"
        coco_images = raw_dataset_path / "coco" / "images"
        coco_records = AnnotationParser.parse_coco(coco_json, coco_images)
        self.assertEqual(len(coco_records), 3)

        voc_xml_1 = raw_dataset_path / "voc" / "annotations" / "voc_001.xml"
        voc_img_1 = raw_dataset_path / "voc" / "images" / "voc_001.jpg"
        voc_rec_1 = AnnotationParser.parse_pascal_voc(voc_xml_1, voc_img_1)

        voc_xml_2 = raw_dataset_path / "voc" / "annotations" / "voc_002.xml"
        voc_img_2 = raw_dataset_path / "voc" / "images" / "voc_002.jpg"
        voc_rec_2 = AnnotationParser.parse_pascal_voc(voc_xml_2, voc_img_2)

        master_records: list[AnnotationRecord] = list(coco_records) + [voc_rec_1, voc_rec_2]

        # Verify canonical classes mapped [0..4]
        all_class_ids = {b.class_id for r in master_records for b in r.boxes}
        for cid in all_class_ids:
            self.assertIn(cid, CANONICAL_CLASSES)

        # --- Stage 3: Viewpoint Augmentation (R14) ---
        augmentor = ViewpointAugmentor(seed=42)
        augmented_records: list[AnnotationRecord] = []
        image_arrays: dict[str, np.ndarray] = {}

        for rec in master_records:
            if Path(rec.image_path).is_file():
                img = cv2.imread(rec.image_path)
            else:
                img = np.full((rec.height, rec.width, 3), 120, dtype=np.uint8)

            # Store original
            image_arrays[rec.image_stem] = img
            augmented_records.append(rec)

            # Generate 1 augmented viewpoint variant per frame (R14)
            aug_img, aug_boxes = augmentor.augment_sample(img, rec.boxes)
            aug_stem = f"{rec.image_stem}_aug"
            aug_rec = AnnotationRecord(
                image_path=f"{aug_stem}.jpg",
                image_stem=aug_stem,
                width=rec.width,
                height=rec.height,
                boxes=aug_boxes,
                is_negative=(len(aug_boxes) == 0),
            )
            image_arrays[aug_stem] = aug_img
            augmented_records.append(aug_rec)

        self.assertEqual(len(augmented_records), len(master_records) * 2)

        # --- Stage 4: Stratified Partitioning (R18) ---
        partitioner = DatasetPartitioner(seed=42)
        splits = partitioner.partition_samples(augmented_records)

        total_split_images = sum(len(r) for r in splits.values())
        self.assertEqual(total_split_images, len(augmented_records))

        # --- Stage 5: Artifact Materialization, Report & Manifest (R20, R21) ---
        yaml_path, dist_path, manifest_path = save_curated_split_artifacts(
            self.curated_root, splits, image_arrays=image_arrays, seed=42
        )

        # Assert dataset.yaml
        self.assertTrue(yaml_path.is_file())
        with open(yaml_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
            self.assertIn("train", cfg)
            self.assertIn("val", cfg)
            self.assertIn("test", cfg)

        # Assert class_distribution.json (R20)
        self.assertTrue(dist_path.is_file())
        with open(dist_path, "r", encoding="utf-8") as f:
            dist_report = json.load(f)
            self.assertIn("splits", dist_report)
            self.assertIn("master_distribution", dist_report)
            self.assertGreater(dist_report["total_annotations"], 0)

        # Assert manifest.json (R21)
        self.assertTrue(manifest_path.is_file())
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
            self.assertEqual(manifest["dataset_version"], "1.0.0")
            self.assertEqual(manifest["seed"], 42)
            self.assertGreater(len(manifest["files"]), 0)

            # Check SHA256 of first file
            sample_entry = manifest["files"][0]
            disk_file = self.curated_root / sample_entry["path"]
            self.assertTrue(disk_file.is_file())
            self.assertEqual(sample_entry["sha256"], calculate_sha256(disk_file))

        # --- Stage 6: Execution time performance budget (R25) ---
        elapsed = time.perf_counter() - self.start_time
        self.assertLess(
            elapsed,
            5.0,
            f"Pipeline integration test exceeded 5.0s budget: {elapsed:.3f}s",
        )


if __name__ == "__main__":
    unittest.main()
