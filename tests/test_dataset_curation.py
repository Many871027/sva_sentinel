"""Unit test suite for annotation parsing, normalization, coordinate clamping, and canonical class mapping.

Validates:
- R7: Canonical 5-class ontology mapping (0: car, 1: motorcycle, 2: bicycle, 3: e-scooter, 4: pedestrian).
- R8: Normalized YOLO TXT format conversion within [0.0, 1.0].
- R9: Bounding box boundary clamping to [0.0, 1.0] and discarding invalid boxes.
- R10: Generation of corresponding .txt label file matching image stem.
- R11: Heterogeneous annotation parsing (COCO JSON, Pascal VOC XML, YOLO TXT).
- R12: Omission of unmapped class labels without failure.
- R17: Pruning degenerate bounding boxes (< 16 px² area).
- R22: Negative background sample categorization and handling.
- R25: Fast execution under 5.0s.
"""

import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.pipeline.dataset_curation import (
    AnnotationParser,
    map_category_to_canonical,
    normalize_and_clamp_box,
    save_curated_sample,
)
from src.pipeline.schemas import (
    CANONICAL_CLASSES,
    AnnotationRecord,
    BoundingBoxYOLO,
)
from tests.fixtures.mock_dataset_generator import setup_mock_fixtures


class TestDatasetCuration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp_dir = tempfile.mkdtemp()
        cls.fixture_root = Path(cls.temp_dir) / "fixtures"
        cls.fixtures = setup_mock_fixtures(cls.fixture_root)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.temp_dir, ignore_errors=True)

    def test_r7_canonical_class_ontology_mapping(self):
        """Verify R7: Canonical 5-class ontology mapping from synonyms."""
        # 0: car
        for syn in ["car", "automobile", "sedan", "suv", "van", "truck", "bus", "vehicle"]:
            self.assertEqual(map_category_to_canonical(syn), 0, f"Failed mapping '{syn}' to 0")

        # 1: motorcycle
        for syn in ["motorcycle", "motorbike", "moto", "scooter_gas", "moped"]:
            self.assertEqual(map_category_to_canonical(syn), 1, f"Failed mapping '{syn}' to 1")

        # 2: bicycle
        for syn in ["bicycle", "bike", "cyclist", "bicyclist"]:
            self.assertEqual(map_category_to_canonical(syn), 2, f"Failed mapping '{syn}' to 2")

        # 3: e-scooter
        for syn in ["e-scooter", "escooter", "electric_scooter", "kick_scooter", "scooter_electric"]:
            self.assertEqual(map_category_to_canonical(syn), 3, f"Failed mapping '{syn}' to 3")

        # 4: pedestrian
        for syn in ["pedestrian", "person", "walker", "ped"]:
            self.assertEqual(map_category_to_canonical(syn), 4, f"Failed mapping '{syn}' to 4")

    def test_r12_unmapped_class_omission(self):
        """Verify R12: Unrecognized categories return None and are omitted without error."""
        unmapped_classes = ["traffic_light", "tree", "dog", "building", "cloud", 999]
        for unmapped in unmapped_classes:
            self.assertIsNone(
                map_category_to_canonical(unmapped),
                f"Expected None for unmapped class '{unmapped}'",
            )

    def test_r8_and_r9_coordinate_clamping_and_normalization(self):
        """Verify R8 & R9: Clamping out-of-frame coordinates and normalizing to [0.0, 1.0]."""
        img_w, img_h = 1000, 500

        # Partially out-of-bounds box: xmin < 0, ymin < 0, xmax > img_w, ymax > img_h
        clamped = normalize_and_clamp_box(-50.0, -20.0, 1050.0, 550.0, img_w, img_h)
        self.assertIsNotNone(clamped)
        xc, yc, w, h = clamped
        self.assertAlmostEqual(xc, 0.5, places=5)
        self.assertAlmostEqual(yc, 0.5, places=5)
        self.assertAlmostEqual(w, 1.0, places=5)
        self.assertAlmostEqual(h, 1.0, places=5)

    def test_r17_degenerate_box_filtering(self):
        """Verify R17: Box with area < 16 px² or non-positive extents is pruned."""
        img_w, img_h = 640, 480

        # 2x2 box = 4 px² (< 16 px²)
        degenerate_small = normalize_and_clamp_box(100.0, 100.0, 102.0, 102.0, img_w, img_h)
        self.assertIsNone(degenerate_small, "Area < 16 px² must be discarded")

        # Non-positive extents (xmax <= xmin)
        degenerate_inverted = normalize_and_clamp_box(200.0, 200.0, 190.0, 250.0, img_w, img_h)
        self.assertIsNone(degenerate_inverted, "Inverted box must be discarded")

        # Valid 4x4 box = 16 px² (retained)
        valid_edge = normalize_and_clamp_box(100.0, 100.0, 104.0, 104.0, img_w, img_h)
        self.assertIsNotNone(valid_edge, "16 px² box must be retained")

    def test_r11_coco_parser(self):
        """Verify R11: COCO JSON parser correctly maps, normalizes, and filters categories."""
        coco_json = self.fixtures["coco_json"]
        records = AnnotationParser.parse_coco(coco_json, self.fixtures["coco_images"])

        self.assertEqual(len(records), 3)

        # Image 1 (coco_001.jpg): car (0), pedestrian (4); unmapped traffic_cone omitted
        rec1 = next(r for r in records if r.image_stem == "coco_001")
        self.assertFalse(rec1.is_negative)
        self.assertEqual(len(rec1.boxes), 2)
        classes = {b.class_id for b in rec1.boxes}
        self.assertIn(0, classes)
        self.assertIn(4, classes)

        # Image 2 (coco_002.jpg): bicycle (2), e-scooter (3), clamped car (0); tiny degenerate box pruned
        rec2 = next(r for r in records if r.image_stem == "coco_002")
        self.assertFalse(rec2.is_negative)
        self.assertEqual(len(rec2.boxes), 3)
        classes2 = {b.class_id for b in rec2.boxes}
        self.assertEqual(classes2, {0, 2, 3})

        # Image 3 (coco_003.jpg): negative background image
        rec3 = next(r for r in records if r.image_stem == "coco_003")
        self.assertTrue(rec3.is_negative)
        self.assertEqual(len(rec3.boxes), 0)

    def test_r11_voc_parser(self):
        """Verify R11: Pascal VOC XML parser correctly extracts boxes and canonical classes."""
        voc_001 = self.fixtures["voc_annotations"] / "voc_001.xml"
        rec = AnnotationParser.parse_pascal_voc(voc_001)

        self.assertEqual(rec.image_stem, "voc_001")
        self.assertFalse(rec.is_negative)
        # sedan -> car (0), motorcycle -> 1, tree -> omitted
        self.assertEqual(len(rec.boxes), 2)
        classes = [b.class_id for b in rec.boxes]
        self.assertIn(0, classes)
        self.assertIn(1, classes)

        voc_002 = self.fixtures["voc_annotations"] / "voc_002.xml"
        rec2 = AnnotationParser.parse_pascal_voc(voc_002)
        # kick_scooter -> 3, walker -> 4
        self.assertEqual(len(rec2.boxes), 2)
        classes2 = {b.class_id for b in rec2.boxes}
        self.assertEqual(classes2, {3, 4})

    def test_r11_yolo_parser(self):
        """Verify R11: YOLO TXT parser handles normalized space-delimited records."""
        yolo_001 = self.fixtures["yolo_labels"] / "yolo_001.txt"
        rec = AnnotationParser.parse_yolo_txt(yolo_001, 640, 480)

        self.assertEqual(rec.image_stem, "yolo_001")
        self.assertEqual(len(rec.boxes), 2)
        classes = {b.class_id for b in rec.boxes}
        self.assertEqual(classes, {0, 4})

    def test_r10_and_r22_save_curated_sample_and_negative_frame(self):
        """Verify R10 & R22: Matching .txt label generated, empty label for negative frame."""
        out_images = Path(self.temp_dir) / "test_out_images"
        out_labels = Path(self.temp_dir) / "test_out_labels"

        dummy_img = np.zeros((480, 640, 3), dtype=np.uint8)

        # 1. Positive sample
        pos_rec = AnnotationRecord(
            image_path="dummy_pos.jpg",
            image_stem="dummy_pos",
            width=640,
            height=480,
            boxes=[BoundingBoxYOLO(class_id=0, x_center=0.5, y_center=0.5, width=0.2, height=0.2)],
            is_negative=False,
        )
        img_dest, lbl_dest = save_curated_sample(dummy_img, pos_rec, out_images, out_labels)
        self.assertTrue(img_dest.is_file())
        self.assertIsNotNone(lbl_dest)
        self.assertTrue(lbl_dest.is_file())
        self.assertTrue(lbl_dest.stat().st_size > 0)

        # 2. Negative background sample (R22)
        neg_rec = AnnotationRecord(
            image_path="dummy_neg.jpg",
            image_stem="dummy_neg",
            width=640,
            height=480,
            boxes=[],
            is_negative=True,
        )
        neg_img_dest, neg_lbl_dest = save_curated_sample(
            dummy_img, neg_rec, out_images, out_labels, write_negative_labels=True
        )
        self.assertTrue(neg_img_dest.is_file())
        self.assertIsNotNone(neg_lbl_dest)
        self.assertTrue(neg_lbl_dest.is_file())
        self.assertEqual(neg_lbl_dest.stat().st_size, 0, "Negative frame label file must be 0 bytes (empty)")


if __name__ == "__main__":
    unittest.main()
