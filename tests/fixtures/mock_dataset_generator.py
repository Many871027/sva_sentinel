"""Synthetic miniature mock dataset fixtures for zero-network unit and integration testing.

Covers:
- R23: Mock dataset fixtures generator with known annotations.
- R24: Mocking fixtures without external network egress.
"""

import json
from pathlib import Path
import cv2
import numpy as np


def create_solid_image(path: Path, width: int = 640, height: int = 480, color: tuple = (120, 120, 120)) -> Path:
    """Create a miniature synthetic BGR test image."""
    path.parent.mkdir(parents=True, exist_ok=True)
    img = np.full((height, width, 3), color, dtype=np.uint8)
    # Draw simple shapes to add pixel diversity
    cv2.rectangle(img, (50, 50), (200, 200), (40, 40, 200), -1)
    cv2.circle(img, (400, 300), 50, (200, 40, 40), -1)
    cv2.imwrite(str(path), img)
    return path


def setup_mock_fixtures(base_dir: Path) -> dict:
    """Generate deterministic miniature COCO, VOC, and YOLO datasets for testing."""
    base_dir = Path(base_dir)
    base_dir.mkdir(parents=True, exist_ok=True)

    coco_dir = base_dir / "coco"
    voc_dir = base_dir / "voc"
    yolo_dir = base_dir / "yolo"

    # --- 1. COCO Mock Dataset ---
    coco_images_dir = coco_dir / "images"
    coco_images_dir.mkdir(parents=True, exist_ok=True)
    create_solid_image(coco_images_dir / "coco_001.jpg", 640, 480, (100, 150, 200))
    create_solid_image(coco_images_dir / "coco_002.jpg", 800, 600, (150, 100, 200))
    create_solid_image(coco_images_dir / "coco_003.jpg", 640, 480, (50, 50, 50))  # negative background

    coco_data = {
        "categories": [
            {"id": 1, "name": "car"},
            {"id": 2, "name": "pedestrian"},
            {"id": 3, "name": "bicycle"},
            {"id": 4, "name": "motorcycle"},
            {"id": 5, "name": "e-scooter"},
            {"id": 99, "name": "traffic_cone"},  # Unmapped class
        ],
        "images": [
            {"id": 101, "file_name": "coco_001.jpg", "width": 640, "height": 480},
            {"id": 102, "file_name": "coco_002.jpg", "width": 800, "height": 600},
            {"id": 103, "file_name": "coco_003.jpg", "width": 640, "height": 480},
        ],
        "annotations": [
            # Image 101: 1 car, 1 pedestrian, 1 unmapped traffic_cone
            {"id": 1, "image_id": 101, "category_id": 1, "bbox": [50.0, 60.0, 100.0, 80.0]},  # car (0)
            {"id": 2, "image_id": 101, "category_id": 2, "bbox": [200.0, 150.0, 40.0, 120.0]},  # pedestrian (4)
            {"id": 3, "image_id": 101, "category_id": 99, "bbox": [10.0, 10.0, 20.0, 30.0]},  # unmapped (omitted)
            # Image 102: 1 bicycle, 1 e-scooter, 1 out-of-bounds box to test clamping, 1 tiny degenerate box (< 16 px²)
            {"id": 4, "image_id": 102, "category_id": 3, "bbox": [100.0, 100.0, 50.0, 60.0]},  # bicycle (2)
            {"id": 5, "image_id": 102, "category_id": 5, "bbox": [300.0, 200.0, 30.0, 70.0]},  # e-scooter (3)
            {"id": 6, "image_id": 102, "category_id": 1, "bbox": [750.0, 500.0, 100.0, 150.0]},  # partial out of bounds
            {"id": 7, "image_id": 102, "category_id": 1, "bbox": [50.0, 50.0, 2.0, 2.0]},  # degenerate 4 px² (omitted)
        ],
    }
    coco_json_path = coco_dir / "annotations.json"
    with open(coco_json_path, "w", encoding="utf-8") as f:
        json.dump(coco_data, f, indent=2)

    # --- 2. Pascal VOC Mock Dataset ---
    voc_images_dir = voc_dir / "images"
    voc_ann_dir = voc_dir / "annotations"
    voc_images_dir.mkdir(parents=True, exist_ok=True)
    voc_ann_dir.mkdir(parents=True, exist_ok=True)

    create_solid_image(voc_images_dir / "voc_001.jpg", 640, 480, (200, 180, 100))
    create_solid_image(voc_images_dir / "voc_002.jpg", 640, 480, (120, 200, 100))

    voc_001_xml = """<annotation>
    <filename>voc_001.jpg</filename>
    <size>
        <width>640</width>
        <height>480</height>
        <depth>3</depth>
    </size>
    <object>
        <name>sedan</name>
        <bndbox>
            <xmin>80</xmin>
            <ymin>100</ymin>
            <xmax>240</xmax>
            <ymax>220</ymax>
        </bndbox>
    </object>
    <object>
        <name>motorcycle</name>
        <bndbox>
            <xmin>300</xmin>
            <ymin>150</ymin>
            <xmax>360</xmax>
            <ymax>250</ymax>
        </bndbox>
    </object>
    <object>
        <name>tree</name>
        <bndbox>
            <xmin>10</xmin>
            <ymin>10</ymin>
            <xmax>60</xmax>
            <ymax>90</ymax>
        </bndbox>
    </object>
</annotation>"""
    with open(voc_ann_dir / "voc_001.xml", "w", encoding="utf-8") as f:
        f.write(voc_001_xml)

    voc_002_xml = """<annotation>
    <filename>voc_002.jpg</filename>
    <size>
        <width>640</width>
        <height>480</height>
        <depth>3</depth>
    </size>
    <object>
        <name>kick_scooter</name>
        <bndbox>
            <xmin>120</xmin>
            <ymin>180</ymin>
            <xmax>160</xmax>
            <ymax>270</ymax>
        </bndbox>
    </object>
    <object>
        <name>walker</name>
        <bndbox>
            <xmin>170</xmin>
            <ymin>150</ymin>
            <xmax>210</xmax>
            <ymax>290</ymax>
        </bndbox>
    </object>
</annotation>"""
    with open(voc_ann_dir / "voc_002.xml", "w", encoding="utf-8") as f:
        f.write(voc_002_xml)

    # --- 3. YOLO Mock Dataset ---
    yolo_images_dir = yolo_dir / "images"
    yolo_labels_dir = yolo_dir / "labels"
    yolo_images_dir.mkdir(parents=True, exist_ok=True)
    yolo_labels_dir.mkdir(parents=True, exist_ok=True)

    create_solid_image(yolo_images_dir / "yolo_001.jpg", 640, 480, (90, 90, 190))
    create_solid_image(yolo_images_dir / "yolo_002.jpg", 640, 480, (190, 90, 90))

    # yolo_001.txt: class 0 (car), class 4 (pedestrian)
    with open(yolo_labels_dir / "yolo_001.txt", "w", encoding="utf-8") as f:
        f.write("0 0.250000 0.333333 0.250000 0.250000\n")
        f.write("4 0.500000 0.500000 0.100000 0.300000\n")

    # yolo_002.txt: class 1 (motorcycle), class 3 (e-scooter)
    with open(yolo_labels_dir / "yolo_002.txt", "w", encoding="utf-8") as f:
        f.write("1 0.400000 0.600000 0.150000 0.250000\n")
        f.write("3 0.700000 0.650000 0.080000 0.200000\n")

    return {
        "coco_dir": coco_dir,
        "coco_json": coco_json_path,
        "coco_images": coco_images_dir,
        "voc_dir": voc_dir,
        "voc_annotations": voc_ann_dir,
        "voc_images": voc_images_dir,
        "yolo_dir": yolo_dir,
        "yolo_images": yolo_images_dir,
        "yolo_labels": yolo_labels_dir,
    }


if __name__ == "__main__":
    fixtures_path = Path(__file__).resolve().parent / "mock_datasets"
    setup_mock_fixtures(fixtures_path)
    print(f"Mock datasets created at: {fixtures_path}")
