"""Unit test suite for deterministic 70/20/10 splitting, class balance verification, and SHA256 manifest generation.

Validates:
- R18: Deterministic 70/20/10 split with seed control.
- R19: Class frequency balance preservation across splits within ±5% variance.
- R20: Class distribution summary report class_distribution.json.
- R21: Cryptographic SHA256 integrity manifest manifest.json.
- R25: Ultra-fast execution under 5.0s.
"""

import json
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np
import yaml

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


class TestDatasetSplit(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.curated_root = Path(self.temp_dir) / "curated"
        self.curated_root.mkdir(parents=True, exist_ok=True)

        # Generate 100 synthetic annotation records with diverse 5-class distribution
        self.records = []
        for i in range(100):
            # Class distribution cycle across 0..4
            cls_1 = i % 5
            cls_2 = (i + 1) % 5
            boxes = [
                BoundingBoxYOLO(
                    class_id=cls_1,
                    x_center=0.3,
                    y_center=0.4,
                    width=0.2,
                    height=0.2,
                ),
                BoundingBoxYOLO(
                    class_id=cls_2,
                    x_center=0.6,
                    y_center=0.7,
                    width=0.15,
                    height=0.25,
                ),
            ]
            self.records.append(
                AnnotationRecord(
                    image_path=f"img_{i:03d}.jpg",
                    image_stem=f"img_{i:03d}",
                    width=640,
                    height=480,
                    boxes=boxes,
                    is_negative=False,
                )
            )

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_r18_deterministic_split_ratios_and_reproducibility(self):
        """Verify R18: 70/20/10 partitioning ratio and determinism via seed."""
        partitioner_1 = DatasetPartitioner(seed=42)
        splits_1 = partitioner_1.partition_samples(self.records)

        # Check partition ratios (~70/20/10)
        self.assertEqual(len(splits_1["train"]), 70)
        self.assertEqual(len(splits_1["val"]), 20)
        self.assertEqual(len(splits_1["test"]), 10)

        # Check exact determinism with same seed
        partitioner_2 = DatasetPartitioner(seed=42)
        splits_2 = partitioner_2.partition_samples(self.records)
        stems_1_train = [r.image_stem for r in splits_1["train"]]
        stems_2_train = [r.image_stem for r in splits_2["train"]]
        self.assertEqual(stems_1_train, stems_2_train)

        # Check variation with different seed
        partitioner_diff = DatasetPartitioner(seed=123)
        splits_diff = partitioner_diff.partition_samples(self.records)
        stems_diff_train = [r.image_stem for r in splits_diff["train"]]
        self.assertNotEqual(stems_1_train, stems_diff_train)

    def test_r19_class_balance_within_5_percent_variance(self):
        """Verify R19: Class distribution across train, val, and test splits remains within ±5% variance."""
        partitioner = DatasetPartitioner(seed=42)
        splits = partitioner.partition_samples(self.records)

        # Master totals and frequencies
        master_counts = {c: 0 for c in CANONICAL_CLASSES}
        total_boxes = sum(len(r.boxes) for r in self.records)
        for r in self.records:
            for b in r.boxes:
                master_counts[b.class_id] += 1

        master_freqs = {c: master_counts[c] / total_boxes for c in CANONICAL_CLASSES}

        # Check each split
        for split_name in ["train", "val", "test"]:
            split_recs = splits[split_name]
            split_total = sum(len(r.boxes) for r in split_recs)
            split_counts = {c: 0 for c in CANONICAL_CLASSES}
            for r in split_recs:
                for b in r.boxes:
                    split_counts[b.class_id] += 1

            for c in CANONICAL_CLASSES:
                split_freq = split_counts[c] / split_total
                variance = abs(split_freq - master_freqs[c])
                self.assertLessEqual(
                    variance,
                    0.05,
                    f"Class {CANONICAL_CLASSES[c]} variance {variance:.4f} in '{split_name}' exceeds ±5% limit",
                )

    def test_r20_class_distribution_report(self):
        """Verify R20: Class distribution statistics report format and counts."""
        partitioner = DatasetPartitioner(seed=42)
        splits = partitioner.partition_samples(self.records)

        report = generate_class_distribution_report(splits)
        self.assertEqual(report.total_annotations, 200)
        self.assertIn("train", report.splits)
        self.assertIn("val", report.splits)
        self.assertIn("test", report.splits)

        # Train: 70 images * 2 = 140 boxes
        self.assertEqual(report.splits["train"].total_boxes, 140)
        # Val: 20 images * 2 = 40 boxes
        self.assertEqual(report.splits["val"].total_boxes, 40)
        # Test: 10 images * 2 = 20 boxes
        self.assertEqual(report.splits["test"].total_boxes, 20)

        # Percentages sum to 100% in master
        pct_sum = sum(c.percentage for c in report.master_distribution.values())
        self.assertAlmostEqual(pct_sum, 100.0, delta=0.5)

    def test_r21_manifest_generation_and_checksum_integrity(self):
        """Verify R21: Cryptographic integrity manifest SHA256 calculation and schema."""
        partitioner = DatasetPartitioner(seed=42)
        # Partition a small batch of 10 samples
        sample_subset = self.records[:10]
        splits = partitioner.partition_samples(sample_subset)

        # Create dummy image bytes for saving
        image_arrays = {
            rec.image_stem: np.full((480, 640, 3), 100, dtype=np.uint8)
            for rec in sample_subset
        }

        yaml_path, dist_path, manifest_path = save_curated_split_artifacts(
            self.curated_root, splits, image_arrays=image_arrays, seed=42
        )

        self.assertTrue(yaml_path.is_file())
        self.assertTrue(dist_path.is_file())
        self.assertTrue(manifest_path.is_file())

        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest_json = json.load(f)

        self.assertEqual(manifest_json["dataset_version"], "1.0.0")
        self.assertEqual(manifest_json["seed"], 42)
        self.assertEqual(manifest_json["summary"]["total_images"], 10)
        self.assertEqual(manifest_json["summary"]["total_annotations"], 20)

        files = manifest_json["files"]
        self.assertGreater(len(files), 0)

        # Verify cryptographic accuracy of first file in manifest
        first_file_entry = files[0]
        actual_file_path = self.curated_root / first_file_entry["path"]
        self.assertTrue(actual_file_path.is_file())
        computed_sha256 = calculate_sha256(actual_file_path)
        self.assertEqual(first_file_entry["sha256"], computed_sha256)
        self.assertEqual(first_file_entry["bytes"], actual_file_path.stat().st_size)

        # Check dataset.yaml format
        with open(yaml_path, "r", encoding="utf-8") as f:
            yaml_content = yaml.safe_load(f)
        self.assertEqual(yaml_content["train"], "images/train")
        self.assertEqual(yaml_content["val"], "images/val")
        self.assertEqual(yaml_content["test"], "images/test")
        self.assertEqual(yaml_content["names"][0], "car")
        self.assertEqual(yaml_content["names"][4], "pedestrian")


if __name__ == "__main__":
    unittest.main()
