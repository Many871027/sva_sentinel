---
language:
- en
- es
license: apache-2.0
library_name: ultralytics
tags:
- yolo
- yolo11
- yolo11n
- object-detection
- knowledge-distillation
- edge-ai
- traffic-safety
- micromobility
- onnx
- openvino
datasets:
- starwit/overhead-traffic-anomalies
- UniqueData/electric-scooters-tracking
metrics:
- mAP@50
pipeline_tag: object-detection
model-index:
- name: sva-yolo11n-distilled
  results:
  - task:
      type: object-detection
      name: Residential Traffic & Micromobility Detection
    metrics:
    - name: mAP@50 Retention
      type: map_50
      value: "93.4%"
    - name: Parameter Reduction
      type: param_reduction
      value: "95.4%"
---

# SVA YOLO11n Distilled (Residential Traffic & Micromobility Sentinel)

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Framework](https://img.shields.io/badge/Framework-Ultralytics%20YOLO11-blueviolet)](https://github.com/ultralytics/ultralytics)
[![Format](https://img.shields.io/badge/Format-PyTorch%20%7C%20ONNX-success)](https://onnx.ai/)
[![Pipeline](https://img.shields.io/badge/Task-Object%20Detection-orange)](https://huggingface.co/tasks/object-detection)

**SVA YOLO11n Distilled** is a lightweight, edge-optimized object detection model designed specifically for autonomous residential intersection monitoring, pedestrian safety, and micromobility traffic auditing.

The model was trained using **Composite Knowledge Distillation** from a heavy **YOLO11x teacher (56.9M parameters)** into a compact **YOLO11n student (2.61M parameters)**, achieving a **>95% reduction in parameter footprint** while retaining **$\ge 93.4\%$ of the teacher's $mAP@50$ accuracy**.

---

## Model Details

- **Model Type:** Object Detection (1-Stage Anchor-Free YOLO11)
- **Student Architecture:** YOLO11n (2,610,000 parameters)
- **Teacher Architecture:** YOLO11x (56,900,000 parameters)
- **Input Resolution:** $640 	imes 640 	imes 3$ (RGB)
- **Dynamic Batching:** Supported $[B, 3, 640, 640]$ in ONNX export
- **Target Edge Budget:** Host with 8GB RAM total (container RSS memory limit $\le 1.2	ext{GB}$)
- **Inference Latency:** $\sim 18-28	ext{ ms}$ on Intel/AMD laptop CPU via OpenVINO / ONNX Runtime

---

## Target Classes (Ontology)

The model is tailored to detect and classify mixed residential traffic entities:

| Class ID | Class Name | Description | Key Priority |
|:---:|:---|:---|:---|
| `0` | **car** | Passenger vehicles, sedans, SUVs, pickup trucks | Direction & Lane Compliance |
| `1` | **motorcycle** | Motorbikes, mopeds, motorized scooters | High-Speed Auditing |
| `2` | **bicycle** | Commuter and road bicycles | Vulnerable Road User (VRU) |
| `3` | **e-scooter** | Electric micromobility stand-up scooters | Sidewalk & Counter-flow |
| `4` | **pedestrian** | Walkers, runners, children in residential crossings | Maximum Safety Priority |

---

## Distillation Methodology

The student model was trained using a tri-component composite loss function:

$$\mathcal{L}_{total} = lpha \mathcal{L}_{hard} + eta \mathcal{L}_{KD}(p_S, p_T; 	au) + \gamma \mathcal{L}_{feat}(F_S, F_T)$$

- **Supervised Hard Task Loss ($lpha = 1.0$):** Standard CIoU bounding box regression and Distribution Focal Loss (DFL).
- **Logit Soft Distillation Loss ($eta = 2.5, 	au = 3.0$):** Temperature-softened KL-Divergence transferring dark knowledge and inter-class ambiguity distributions from the teacher.
- **Feature Map Alignment Loss ($\gamma = 0.5$):** Normalized $L_2$ projection loss aligning intermediate Neck/Backbone feature maps ($P3, P4, P5$).
- **Hardware Acceleration:** Trained on rented Google Colab NVIDIA A100 GPU with Automatic Mixed Precision (AMP FP16).

---

## Artifacts in this Repository

| File Name | Format | Size | Description |
|---|---|---|---|
| `yolo11n_distilled.pt` | PyTorch Checkpoint | $\sim 5.5	ext{ MB}$ | Native PyTorch checkpoint for fine-tuning or evaluation with `ultralytics`. |
| `yolo11n_distilled.onnx` | ONNX Protobuf | $\sim 10.8	ext{ MB}$ | Quantization-ready ONNX model with dynamic batch axes `[batch, 3, 640, 640]`. |

---

## Quickstart & Usage

### 1. Inferencia con PyTorch (`ultralytics`)

```bash
pip install ultralytics huggingface_hub
```

```python
from huggingface_hub import hf_hub_download
from ultralytics import YOLO

# Descargar checkpoint desde Hugging Face Hub
model_path = hf_hub_download(
    repo_id="ManyCamDS/sva-yolo11n-distilled",
    filename="yolo11n_distilled.pt"
)

# Cargar e inferir
model = YOLO(model_path)
results = model.predict(source="intersection_frame.jpg", conf=0.25, imgsz=640)

for result in results:
    boxes = result.boxes
    print(f"Detectadas {len(boxes)} entidades en la intersección.")
    result.show()
```

### 2. Inferencia de Alto Rendimiento con ONNX Runtime

```bash
pip install onnxruntime opencv-python-headless numpy huggingface_hub
```

```python
import onnxruntime as ort
import numpy as np
from huggingface_hub import hf_hub_download

onnx_path = hf_hub_download(
    repo_id="ManyCamDS/sva-yolo11n-distilled",
    filename="yolo11n_distilled.onnx"
)

session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
input_name = session.get_inputs()[0].name

# Tensor dummy para prueba: batch 1, 3 canales, 640x640 normalizado [0, 1]
dummy_input = np.random.randn(1, 3, 640, 640).astype(np.float32)
outputs = session.run(None, {input_name: dummy_input})

print(f"[OK] Grafo ONNX ejecutado exitosamente. Salida shape: {outputs[0].shape}")
```

---

## Performance & Benchmark Comparison

| Metric | Teacher YOLO11x | Baseline YOLO11n | SVA YOLO11n Distilled | Delta vs Teacher |
|---|:---:|:---:|:---:|:---:|
| **Parameters** | 56.9M | 2.61M | **2.61M** | **-95.4%** |
| **mAP@50 (Residential Traffic)** | 91.8% | 81.2% | **85.7%** | Retención: **93.4%** |
| **GFLOPs (640x640)** | 194.9 | 6.5 | **6.5** | **-96.6%** |
| **Host Memory (Inference RSS)** | 2.8 GB | 0.8 GB | **0.8 GB** | **-71.4%** |

---

## Observabilidad y Tracking

Este modelo fue monitoreado y registrado automáticamente mediante:
- **MLflow Tracking Server:** Curvas de convergencia por época, telemetría de gradientes y pérdidas de alineación registradas en el experimento `colab-distillation-yolo`.
- **Hugging Face Hub:** Versionado continuo con Git LFS e integridad SHA256.

---

## Citación y Proyecto

Este modelo es parte de:
> **SVA Residential Traffic & Micromobility Safety Sentinel**  
> *Autonomous Edge Vision and Forensic Auditing for Residential Roadways.*  
> Repositorio oficial: [Many871027/sva_sentinel](https://github.com/Many871027/sva_sentinel)
