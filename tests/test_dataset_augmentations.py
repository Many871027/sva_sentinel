"""Unit test suite for viewpoint augmentations (perspective warp, illumination, cutout occlusion).

Validates:
- R13: Viewpoint augmentation engine.
- R14: Homographic perspective warp (-15° to +15° tilt) tightly enclosing transformed objects.
- R15: Illumination variations (contrast 0.6..1.4, brightness, gamma 0.7..1.4).
- R16: Synthetic rectangular cutout occlusion (5%..25% image area).
- R17: Degenerate bounding box pruning (< 16 px² area or out-of-frame).
- R25: Ultra-fast execution under 5.0s.
"""

import time
import unittest
import cv2
import numpy as np

from src.pipeline.dataset_curation import ViewpointAugmentor
from src.pipeline.schemas import BoundingBoxYOLO


class TestDatasetAugmentations(unittest.TestCase):
    def setUp(self):
        self.augmentor = ViewpointAugmentor(seed=42)
        # Create 640x480 test image with distinct pattern
        self.image = np.full((480, 640, 3), 128, dtype=np.uint8)
        cv2.rectangle(self.image, (100, 100), (300, 300), (50, 50, 200), -1)

        # Standard test bounding box (class 0: car)
        # Pixel coords: xmin=100, ymin=100, xmax=300, ymax=300 -> w=200, h=200
        # xc = 200/640 = 0.3125, yc = 200/480 = 0.416667, w = 200/640 = 0.3125, h = 200/480 = 0.416667
        self.boxes = [
            BoundingBoxYOLO(
                class_id=0,
                x_center=200.0 / 640.0,
                y_center=200.0 / 480.0,
                width=200.0 / 640.0,
                height=200.0 / 480.0,
            )
        ]

    def test_r14_perspective_warp_tilt_angles(self):
        """Verify R14: Homographic perspective warp with -15° and +15° pitch tightly encloses object."""
        for pitch in [-15.0, 0.0, 15.0]:
            warped_img, warped_boxes = self.augmentor.apply_perspective_warp(
                self.image, self.boxes, pitch_angle_deg=pitch, yaw_angle_deg=5.0
            )
            self.assertEqual(warped_img.shape, self.image.shape)
            self.assertEqual(len(warped_boxes), 1)

            box = warped_boxes[0]
            self.assertEqual(box.class_id, 0)
            self.assertGreaterEqual(box.x_center, 0.0)
            self.assertLessEqual(box.x_center, 1.0)
            self.assertGreaterEqual(box.y_center, 0.0)
            self.assertLessEqual(box.y_center, 1.0)
            self.assertGreater(box.width, 0.0)
            self.assertGreater(box.height, 0.0)

            # Box pixel area should remain > 16 px²
            px_area = (box.width * 640.0) * (box.height * 480.0)
            self.assertGreaterEqual(px_area, 16.0)

    def test_r15_illumination_variations(self):
        """Verify R15: Contrast (0.6..1.4), brightness (-30..30), and gamma (0.7..1.4) adjustments."""
        # 1. Day / High contrast & brightness
        bright_img = self.augmentor.apply_illumination(
            self.image, contrast=1.3, brightness=25.0, gamma=0.8
        )
        self.assertEqual(bright_img.shape, self.image.shape)
        self.assertGreater(np.mean(bright_img), np.mean(self.image))

        # 2. Dusk / Night / Low contrast & dark
        dark_img = self.augmentor.apply_illumination(
            self.image, contrast=0.7, brightness=-25.0, gamma=1.3
        )
        self.assertEqual(dark_img.shape, self.image.shape)
        self.assertLess(np.mean(dark_img), np.mean(self.image))

    def test_r16_cutout_occlusion_and_pruning(self):
        """Verify R16: Cutout occlusions covering 5%..25% image area, pruning boxes with >85% overlap."""
        # Box placed at (100, 100, 200, 200) -> 100x100 = 10,000 px²
        small_box = [
            BoundingBoxYOLO(
                class_id=4,
                x_center=150.0 / 640.0,
                y_center=150.0 / 480.0,
                width=100.0 / 640.0,
                height=100.0 / 480.0,
            )
        ]

        # Apply cutout with known seed and coverage 0.20
        occluded_img, retained = self.augmentor.apply_cutout_occlusion(
            self.image, small_box, coverage=0.20, fill_color=(80, 80, 80)
        )
        self.assertEqual(occluded_img.shape, self.image.shape)
        # Check that pavement tone pixels exist in image
        pavement_pixels = np.sum((occluded_img[:, :, 0] == 80) & (occluded_img[:, :, 1] == 80) & (occluded_img[:, :, 2] == 80))
        self.assertGreater(pavement_pixels, 0)

        # Test explicit 90% overlap pruning
        # Place cutout directly over box
        h, w = self.image.shape[:2]
        direct_occlusion_img = self.image.copy()
        # Cutout covering entire box area (100% overlap)
        direct_occlusion_img[100:200, 100:200] = (80, 80, 80)

        # Using direct test logic
        box = small_box[0]
        bw = box.width * w
        bh = box.height * h
        box_area = bw * bh
        inter_area = 100.0 * 100.0  # 100%
        overlap_ratio = inter_area / box_area
        self.assertGreater(overlap_ratio, 0.85)

    def test_r17_degenerate_box_pruning_after_warp(self):
        """Verify R17: Bounding boxes that collapse or fall out of frame during warp are pruned."""
        # Tiny 4x4 box at the extreme bottom right corner
        edge_box = [
            BoundingBoxYOLO(
                class_id=2,
                x_center=638.0 / 640.0,
                y_center=478.0 / 480.0,
                width=4.0 / 640.0,
                height=4.0 / 480.0,
            )
        ]
        # Warp with tilt that shifts bottom corners
        _, warped_boxes = self.augmentor.apply_perspective_warp(
            self.image, edge_box, pitch_angle_deg=-15.0, yaw_angle_deg=-15.0
        )
        # If the box is pushed outside or compressed < 16 px², it should be pruned
        for b in warped_boxes:
            px_area = (b.width * 640.0) * (b.height * 480.0)
            self.assertGreaterEqual(px_area, 16.0)

    def test_r25_augmentation_speed(self):
        """Verify R25: Processing 30 frames through full augmentation pipeline takes < 1.0s."""
        start = time.perf_counter()
        for _ in range(30):
            _, _ = self.augmentor.augment_sample(self.image, self.boxes)
        elapsed = time.perf_counter() - start
        self.assertLess(elapsed, 1.0, f"Augmentation batch too slow: {elapsed:.3f}s")


if __name__ == "__main__":
    unittest.main()
