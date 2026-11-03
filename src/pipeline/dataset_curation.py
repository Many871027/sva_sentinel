"""Dataset curation, heterogeneous annotation parsing, normalization, and viewpoint augmentations.

Covers:
- R7: Canonical 5-class ontology mapping (0: car, 1: motorcycle, 2: bicycle, 3: e-scooter, 4: pedestrian).
- R8: Bounding box normalization to YOLO TXT format within [0.0, 1.0].
- R9: Coordinate boundary clamping and non-positive dimension rejection.
- R10: Generation of corresponding .txt label files matching image file stems.
- R11: Heterogeneous annotation parsers (COCO, Pascal VOC, YOLO TXT).
- R12: Omission of unmapped class labels without pipeline failure.
- R13: Viewpoint-specific augmentation engine.
- R14: Homographic perspective warp simulating -15° to +15° camera tilt.
- R15: Illumination adjustments (contrast 0.6..1.4, brightness, gamma 0.7..1.4).
- R16: Synthetic rectangular cutout occlusion masks (5% to 25% area).
- R17: Pruning degenerate bounding boxes (< 16 px² area or outside frame).
- R22: Negative background frame categorization and handling.
"""

import argparse
import json
import logging
import math
import random
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import cv2
import numpy as np

from src.core.exceptions import AnnotationFormatError, CurationError
from src.pipeline.schemas import (
    CANONICAL_CLASSES,
    CANONICAL_CLASS_NAMES,
    AnnotationRecord,
    BoundingBoxYOLO,
)

logger = logging.getLogger("sva.pipeline.curation")

# Canonical synonym mapping
CANONICAL_SYNONYMS: Dict[int, Set[str]] = {
    0: {
        "car",
        "automobile",
        "sedan",
        "suv",
        "van",
        "truck",
        "bus",
        "vehicle",
        "auto",
        "traffic_car",
        "pickup",
    },
    1: {
        "motorcycle",
        "motorbike",
        "moto",
        "scooter_gas",
        "moped",
        "motorcyclist",
    },
    2: {
        "bicycle",
        "bike",
        "cyclist",
        "bicyclist",
        "cycle",
    },
    3: {
        "e-scooter",
        "escooter",
        "electric_scooter",
        "kick_scooter",
        "scooter_electric",
        "scooter",
        "micromobility",
    },
    4: {
        "pedestrian",
        "person",
        "walker",
        "ped",
        "people",
        "human",
    },
}


def map_category_to_canonical(raw_category: str | int) -> Optional[int]:
    """Map arbitrary source category name or numeric index to canonical 5-class ontology.

    Returns:
        Canonical class index (0..4) or None if unmapped (R12).
    """
    if isinstance(raw_category, int):
        if raw_category in CANONICAL_CLASSES:
            return raw_category
        return None

    cleaned = str(raw_category).strip().lower().replace("-", "_").replace(" ", "_")

    # Direct match on canonical class name
    if cleaned in CANONICAL_CLASS_NAMES:
        return CANONICAL_CLASS_NAMES[cleaned]

    # First check exact match across all canonical synonyms
    for class_id, synonyms in CANONICAL_SYNONYMS.items():
        if cleaned in synonyms:
            return class_id

    # Fallback to substring matching only if no exact match exists
    for class_id, synonyms in CANONICAL_SYNONYMS.items():
        for syn in synonyms:
            if syn in cleaned:
                return class_id

    return None


def normalize_and_clamp_box(
    xmin: float,
    ymin: float,
    xmax: float,
    ymax: float,
    img_w: int,
    img_h: int,
) -> Optional[Tuple[float, float, float, float]]:
    """Clamp pixel coordinates to image frame, filter degenerate boxes, and normalize to [0.0, 1.0].

    Returns:
        Tuple of (x_center, y_center, width, height) normalized to [0.0, 1.0],
        or None if box area < 16 px² or coordinates fall entirely outside frame.
    """
    if img_w <= 0 or img_h <= 0:
        return None

    # Coordinate boundary clamping (R9)
    clamped_xmin = max(0.0, min(float(img_w), float(xmin)))
    clamped_ymin = max(0.0, min(float(img_h), float(ymin)))
    clamped_xmax = max(0.0, min(float(img_w), float(xmax)))
    clamped_ymax = max(0.0, min(float(img_h), float(ymax)))

    w_px = clamped_xmax - clamped_xmin
    h_px = clamped_ymax - clamped_ymin

    # Degenerate box filtering (< 16 px² area or non-positive extents) (R9, R17)
    area = w_px * h_px
    if w_px <= 0.0 or h_px <= 0.0 or area < 16.0:
        return None

    x_center = (clamped_xmin + clamped_xmax) / (2.0 * img_w)
    y_center = (clamped_ymin + clamped_ymax) / (2.0 * img_h)
    norm_w = w_px / img_w
    norm_h = h_px / img_h

    # Guarantee bounded within [0.0, 1.0]
    x_center = max(0.0, min(1.0, x_center))
    y_center = max(0.0, min(1.0, y_center))
    norm_w = max(1e-6, min(1.0, norm_w))
    norm_h = max(1e-6, min(1.0, norm_h))

    return x_center, y_center, norm_w, norm_h


class AnnotationParser:
    """Parser suite for heterogeneous annotation formats (COCO JSON, Pascal VOC XML, YOLO TXT)."""

    @staticmethod
    def parse_coco(coco_json_path: Path, images_base_dir: Optional[Path] = None) -> List[AnnotationRecord]:
        """Parse standard COCO JSON instances file."""
        if not coco_json_path.is_file():
            raise AnnotationFormatError(f"COCO annotations file not found: {coco_json_path}")

        try:
            with open(coco_json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as exc:
            raise AnnotationFormatError(f"Failed to read COCO JSON '{coco_json_path}': {exc}") from exc

        # Map categories
        cat_map: Dict[int, Optional[int]] = {}
        for cat in data.get("categories", []):
            cat_id = cat.get("id")
            cat_name = cat.get("name", "")
            cat_map[cat_id] = map_category_to_canonical(cat_name)

        # Map images
        images_info: Dict[int, dict] = {}
        for img in data.get("images", []):
            img_id = img["id"]
            images_info[img_id] = {
                "file_name": img["file_name"],
                "width": img["width"],
                "height": img["height"],
                "stem": Path(img["file_name"]).stem,
            }

        # Collect boxes per image
        image_boxes: Dict[int, List[BoundingBoxYOLO]] = {img_id: [] for img_id in images_info}
        for ann in data.get("annotations", []):
            img_id = ann.get("image_id")
            if img_id not in images_info:
                continue

            raw_cat_id = ann.get("category_id")
            canonical_id = cat_map.get(raw_cat_id)
            if canonical_id is None:
                # Unmapped category omitted without error (R12)
                continue

            bbox = ann.get("bbox", [])
            if len(bbox) != 4:
                continue

            xmin, ymin, w, h = bbox
            xmax = xmin + w
            ymax = ymin + h
            img_meta = images_info[img_id]

            normalized = normalize_and_clamp_box(
                xmin, ymin, xmax, ymax, img_meta["width"], img_meta["height"]
            )
            if normalized is None:
                continue

            xc, yc, nw, nh = normalized
            image_boxes[img_id].append(
                BoundingBoxYOLO(
                    class_id=canonical_id,
                    x_center=xc,
                    y_center=yc,
                    width=nw,
                    height=nh,
                )
            )

        records: List[AnnotationRecord] = []
        for img_id, meta in images_info.items():
            boxes = image_boxes.get(img_id, [])
            file_name = meta["file_name"]
            img_path = (
                str(images_base_dir / file_name)
                if images_base_dir
                else file_name
            )
            records.append(
                AnnotationRecord(
                    image_path=img_path,
                    image_stem=meta["stem"],
                    width=meta["width"],
                    height=meta["height"],
                    boxes=boxes,
                    is_negative=(len(boxes) == 0),
                )
            )

        return records

    @staticmethod
    def parse_pascal_voc(voc_xml_path: Path, image_path: Optional[Path] = None) -> AnnotationRecord:
        """Parse a single Pascal VOC XML annotation file."""
        if not voc_xml_path.is_file():
            raise AnnotationFormatError(f"Pascal VOC XML file not found: {voc_xml_path}")

        try:
            tree = ET.parse(voc_xml_path)
            root = tree.getroot()
        except Exception as exc:
            raise AnnotationFormatError(f"Failed to parse VOC XML '{voc_xml_path}': {exc}") from exc

        filename_elem = root.find("filename")
        filename = filename_elem.text if filename_elem is not None and filename_elem.text else voc_xml_path.stem + ".jpg"
        stem = Path(filename).stem

        size_elem = root.find("size")
        if size_elem is not None:
            w_elem = size_elem.find("width")
            h_elem = size_elem.find("height")
            img_w = int(w_elem.text) if w_elem is not None and w_elem.text else 0
            img_h = int(h_elem.text) if h_elem is not None and h_elem.text else 0
        else:
            img_w, img_h = 0, 0

        # If dimension missing from XML, inspect image file if available
        if (img_w <= 0 or img_h <= 0) and image_path and image_path.is_file():
            img = cv2.imread(str(image_path))
            if img is not None:
                img_h, img_w = img.shape[:2]

        if img_w <= 0 or img_h <= 0:
            # Default fallback dimensions if undetermined
            img_w, img_h = 1920, 1080

        boxes: List[BoundingBoxYOLO] = []
        for obj in root.findall("object"):
            name_elem = obj.find("name")
            if name_elem is None or not name_elem.text:
                continue

            canonical_id = map_category_to_canonical(name_elem.text)
            if canonical_id is None:
                continue

            bndbox = obj.find("bndbox")
            if bndbox is None:
                continue

            try:
                xmin = float(bndbox.find("xmin").text)  # type: ignore[union-attr]
                ymin = float(bndbox.find("ymin").text)  # type: ignore[union-attr]
                xmax = float(bndbox.find("xmax").text)  # type: ignore[union-attr]
                ymax = float(bndbox.find("ymax").text)  # type: ignore[union-attr]
            except (TypeError, ValueError):
                continue

            normalized = normalize_and_clamp_box(xmin, ymin, xmax, ymax, img_w, img_h)
            if normalized is None:
                continue

            xc, yc, nw, nh = normalized
            boxes.append(
                BoundingBoxYOLO(
                    class_id=canonical_id,
                    x_center=xc,
                    y_center=yc,
                    width=nw,
                    height=nh,
                )
            )

        resolved_img_path = str(image_path) if image_path else filename
        return AnnotationRecord(
            image_path=resolved_img_path,
            image_stem=stem,
            width=img_w,
            height=img_h,
            boxes=boxes,
            is_negative=(len(boxes) == 0),
        )

    @staticmethod
    def parse_yolo_txt(
        txt_path: Path,
        img_w: int,
        img_h: int,
        image_path: Optional[Path] = None,
        is_normalized: bool = True,
    ) -> AnnotationRecord:
        """Parse YOLO space-delimited text label file."""
        stem = txt_path.stem
        resolved_img_path = str(image_path) if image_path else f"{stem}.jpg"

        if not txt_path.is_file():
            # Missing label file treated as negative background frame (R22)
            return AnnotationRecord(
                image_path=resolved_img_path,
                image_stem=stem,
                width=img_w,
                height=img_h,
                boxes=[],
                is_negative=True,
            )

        boxes: List[BoundingBoxYOLO] = []
        with open(txt_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 5:
                    continue

                try:
                    raw_cls = int(parts[0])
                    x_val = float(parts[1])
                    y_val = float(parts[2])
                    w_val = float(parts[3])
                    h_val = float(parts[4])
                except ValueError:
                    continue

                canonical_id = map_category_to_canonical(raw_cls)
                if canonical_id is None:
                    continue

                if is_normalized:
                    # Validate box area in pixels (R17)
                    box_px_area = (w_val * img_w) * (h_val * img_h)
                    if box_px_area < 16.0 or w_val <= 0.0 or h_val <= 0.0:
                        continue
                    boxes.append(
                        BoundingBoxYOLO(
                            class_id=canonical_id,
                            x_center=max(0.0, min(1.0, x_val)),
                            y_center=max(0.0, min(1.0, y_val)),
                            width=max(1e-6, min(1.0, w_val)),
                            height=max(1e-6, min(1.0, h_val)),
                        )
                    )
                else:
                    # Pixel coordinates
                    xmin = x_val - w_val / 2.0
                    ymin = y_val - h_val / 2.0
                    xmax = x_val + w_val / 2.0
                    ymax = y_val + h_val / 2.0
                    normalized = normalize_and_clamp_box(xmin, ymin, xmax, ymax, img_w, img_h)
                    if normalized is None:
                        continue
                    xc, yc, nw, nh = normalized
                    boxes.append(
                        BoundingBoxYOLO(
                            class_id=canonical_id,
                            x_center=xc,
                            y_center=yc,
                            width=nw,
                            height=nh,
                        )
                    )

        return AnnotationRecord(
            image_path=resolved_img_path,
            image_stem=stem,
            width=img_w,
            height=img_h,
            boxes=boxes,
            is_negative=(len(boxes) == 0),
        )


class ViewpointAugmentor:
    """Viewpoint-specific augmentation transforms simulating residential intersection camera conditions."""

    def __init__(self, seed: Optional[int] = None) -> None:
        self.rng = random.Random(seed)

    def apply_perspective_warp(
        self,
        image: np.ndarray,
        boxes: List[BoundingBoxYOLO],
        pitch_angle_deg: float = 0.0,
        yaw_angle_deg: float = 0.0,
    ) -> Tuple[np.ndarray, List[BoundingBoxYOLO]]:
        """Simulate intersection camera pitch and yaw homography (-15° to +15° tilt) (R14)."""
        h, w = image.shape[:2]
        if abs(pitch_angle_deg) < 1e-4 and abs(yaw_angle_deg) < 1e-4:
            return image.copy(), [b.model_copy() for b in boxes]

        # Constrain pitch and yaw to [-15.0, 15.0]
        pitch = max(-15.0, min(15.0, pitch_angle_deg))
        yaw = max(-15.0, min(15.0, yaw_angle_deg))

        # Project 4 corners with tilt
        pitch_rad = math.radians(pitch)
        yaw_rad = math.radians(yaw)

        dx = w * math.sin(yaw_rad) * 0.15
        dy = h * math.sin(pitch_rad) * 0.15

        src_pts = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
        dst_pts = np.float32([
            [max(0.0, dx), max(0.0, dy)],
            [min(float(w), w - max(0.0, -dx)), max(0.0, dy)],
            [min(float(w), w - max(0.0, dx)), min(float(h), h - max(0.0, -dy))],
            [max(0.0, -dx), min(float(h), h - max(0.0, -dy))],
        ])

        H_mat = cv2.getPerspectiveTransform(src_pts, dst_pts)
        warped_img = cv2.warpPerspective(
            image, H_mat, (w, h), borderMode=cv2.BORDER_REFLECT_101
        )

        transformed_boxes: List[BoundingBoxYOLO] = []
        for box in boxes:
            # Reconstruct 4 pixel corners of the bounding box
            bw = box.width * w
            bh = box.height * h
            bx1 = (box.x_center * w) - (bw / 2.0)
            by1 = (box.y_center * h) - (bh / 2.0)
            bx2 = bx1 + bw
            by2 = by1 + bh

            box_corners = np.float32([
                [[bx1, by1]],
                [[bx2, by1]],
                [[bx2, by2]],
                [[bx1, by2]],
            ])

            trans_pts = cv2.perspectiveTransform(box_corners, H_mat)
            xs = trans_pts[:, 0, 0]
            ys = trans_pts[:, 0, 1]

            min_x, max_x = float(np.min(xs)), float(np.max(xs))
            min_y, max_y = float(np.min(ys)), float(np.max(ys))

            normalized = normalize_and_clamp_box(min_x, min_y, max_x, max_y, w, h)
            if normalized is None:
                # Degenerate box < 16 px² or out of bounds pruned (R17)
                continue

            n_xc, n_yc, n_w, n_h = normalized
            transformed_boxes.append(
                BoundingBoxYOLO(
                    class_id=box.class_id,
                    x_center=n_xc,
                    y_center=n_yc,
                    width=n_w,
                    height=n_h,
                )
            )

        return warped_img, transformed_boxes

    def apply_illumination(
        self,
        image: np.ndarray,
        contrast: float = 1.0,
        brightness: float = 0.0,
        gamma: float = 1.0,
    ) -> np.ndarray:
        """Simulate intersection day/dusk/twilight/night conditions (R15).

        Args:
            image: Input uint8 BGR image array.
            contrast: Multiplicative contrast factor [0.6, 1.4].
            brightness: Additive brightness offset [-30, +30].
            gamma: Non-linear gamma factor [0.7, 1.4].
        """
        contrast = max(0.6, min(1.4, contrast))
        brightness = max(-30.0, min(30.0, brightness))
        gamma = max(0.7, min(1.4, gamma))

        # Linear contrast and brightness adjustment
        adjusted = np.clip(contrast * image.astype(np.float32) + brightness, 0, 255).astype(np.uint8)

        # Gamma adjustment using lookup table
        if abs(gamma - 1.0) > 1e-3:
            inv_gamma = 1.0 / gamma
            lut = np.array([
                np.clip(((i / 255.0) ** inv_gamma) * 255.0, 0, 255) for i in range(256)
            ], dtype=np.uint8)
            adjusted = cv2.LUT(adjusted, lut)

        return adjusted

    def apply_cutout_occlusion(
        self,
        image: np.ndarray,
        boxes: List[BoundingBoxYOLO],
        coverage: float = 0.15,
        fill_color: Tuple[int, int, int] = (80, 80, 80),
    ) -> Tuple[np.ndarray, List[BoundingBoxYOLO]]:
        """Overlay synthetic rectangular cutout occlusions covering 5% to 25% image area (R16)."""
        h, w = image.shape[:2]
        total_pixels = h * w
        cov = max(0.05, min(0.25, coverage))
        cutout_target_area = cov * total_pixels

        # Choose aspect ratio between 0.5 and 2.0
        aspect = float(self.rng.uniform(0.5, 2.0))
        cutout_w = int(math.sqrt(cutout_target_area * aspect))
        cutout_h = int(cutout_target_area / max(1, cutout_w))

        cutout_w = max(10, min(w - 1, cutout_w))
        cutout_h = max(10, min(h - 1, cutout_h))

        x0 = int(self.rng.randint(0, max(1, w - cutout_w)))
        y0 = int(self.rng.randint(0, max(1, h - cutout_h)))
        x1 = x0 + cutout_w
        y1 = y0 + cutout_h

        occluded_img = image.copy()
        occluded_img[y0:y1, x0:x1] = fill_color

        retained_boxes: List[BoundingBoxYOLO] = []
        for box in boxes:
            bw = box.width * w
            bh = box.height * h
            bx1 = (box.x_center * w) - (bw / 2.0)
            by1 = (box.y_center * h) - (bh / 2.0)
            bx2 = bx1 + bw
            by2 = by1 + bh
            box_area = bw * bh

            # Calculate intersection with cutout
            ix1 = max(bx1, x0)
            iy1 = max(by1, y0)
            ix2 = min(bx2, x1)
            iy2 = min(by2, y1)

            if ix2 > ix1 and iy2 > iy1:
                inter_area = (ix2 - ix1) * (iy2 - iy1)
                overlap_ratio = inter_area / max(1e-6, box_area)
                remaining_visible_area = box_area - inter_area
                # Prune if overlapping > 85% or visible area < 16 px² (R16, R17)
                if overlap_ratio > 0.85 or remaining_visible_area < 16.0:
                    continue

            retained_boxes.append(box.model_copy())

        return occluded_img, retained_boxes

    def augment_sample(
        self,
        image: np.ndarray,
        boxes: List[BoundingBoxYOLO],
    ) -> Tuple[np.ndarray, List[BoundingBoxYOLO]]:
        """Apply combined perspective, illumination, and occlusion augmentations."""
        pitch = float(self.rng.uniform(-15.0, 15.0))
        yaw = float(self.rng.uniform(-10.0, 10.0))
        warped_img, warped_boxes = self.apply_perspective_warp(image, boxes, pitch, yaw)

        contrast = float(self.rng.uniform(0.6, 1.4))
        brightness = float(self.rng.uniform(-25.0, 25.0))
        gamma = float(self.rng.uniform(0.7, 1.4))
        illum_img = self.apply_illumination(warped_img, contrast, brightness, gamma)

        coverage = float(self.rng.uniform(0.05, 0.20))
        aug_img, final_boxes = self.apply_cutout_occlusion(illum_img, warped_boxes, coverage)

        return aug_img, final_boxes


def save_curated_sample(
    image: np.ndarray,
    record: AnnotationRecord,
    images_dir: Path,
    labels_dir: Path,
    write_negative_labels: bool = True,
) -> Tuple[Path, Optional[Path]]:
    """Save curated image and its matching .txt label file (R10, R22)."""
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    img_filename = f"{record.image_stem}.jpg"
    img_dest = images_dir / img_filename
    cv2.imwrite(str(img_dest), image)

    label_filename = f"{record.image_stem}.txt"
    label_dest = labels_dir / label_filename

    if len(record.boxes) > 0 or write_negative_labels:
        with open(label_dest, "w", encoding="utf-8") as f:
            f.write(record.to_yolo_content())
            if len(record.boxes) > 0:
                f.write("\n")
        return img_dest, label_dest

    return img_dest, None


def main() -> int:
    """CLI entrypoint for dataset curation."""
    parser = argparse.ArgumentParser(description="Dataset Curation CLI")
    parser.add_argument("--format", choices=["coco", "voc", "yolo"], default="coco")
    parser.add_argument("--input", type=Path, required=True, help="Input annotation file or directory")
    parser.add_argument("--output-images", type=Path, default=Path("data/curated/images/raw"))
    parser.add_argument("--output-labels", type=Path, default=Path("data/curated/labels/raw"))
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    logger.info("Executing curation for format: %s from %s", args.format, args.input)
    return 0


if __name__ == "__main__":
    sys.exit(main())
