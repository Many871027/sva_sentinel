# Requirements: Google Colab GPU Knowledge Distillation Pipeline (Teacher YOLO11x to Student YOLO11n)

## Feature Metadata
- **ID:** SVA-DIST-001
- **Name:** colab_gpu_yolo_knowledge_distillation
- **Title:** Google Colab GPU Knowledge Distillation Pipeline (Teacher YOLO11x to Student YOLO11n)
- **Target Subsystem:** `src/engine/` (Distillation Loss & Training Core) and `colab_notebooks/` (Google Colab Pro GPU Notebook)
- **Artifact Targets:** `colab_notebooks/01_yolo_knowledge_distillation.ipynb`, `models/distilled/`

## Acceptance Criteria Mapping
| Acceptance Criterion | Requirements |
|---|---|
| Create colab_notebooks/01_yolo_knowledge_distillation.ipynb with automatic Google Drive / Hugging Face data synchronization | R1, R2, R3, R4, R5 |
| Implement composite distillation loss L_total = alpha * L_hard + beta * L_KD(teacher, student; temp=tau) + gamma * L_feat | R6, R7, R8, R9, R10 |
| Execute distributed/mixed-precision distillation on rented Colab GPU achieving >= 92% of teacher mAP@50 while reducing parameters by > 80% | R11, R12, R13, R14 |
| Export distilled student weights (PyTorch/ONNX) and push checkpoints to Hugging Face Model Hub / MLFlow remote registry | R15, R16, R17, R18 |
| Zero-egress fast verification and test suite | R19, R20 |

---

## Requirements (Strict EARS Notation)

### R1 (Ubiquitous)
The system SHALL provide an end-to-end distillation notebook in `colab_notebooks/01_yolo_knowledge_distillation.ipynb` configured for rented Google Colab Pro high-performance GPU runtimes (NVIDIA A100 or L4 with 24GB to 40GB VRAM).

### R2 (Event-driven)
WHEN executed in Google Colab, the notebook SHALL mount Google Drive (`/content/drive/MyDrive/sva_sentinel/`) to persist training checkpoints and epoch logs across runtime disconnections.

### R3 (Event-driven)
WHEN dataset preparation is invoked in Colab, the system SHALL synchronize curated training and validation datasets from Hugging Face Hub or Google Drive into Colab's high-speed local NVMe scratch disk (`/content/data/`).

### R4 (State-driven)
WHILE running dataset synchronization, the system SHALL verify file integrity manifests (`manifest.json`) using SHA256 checksums before initiating model training.

### R5 (Unwanted)
IF GPU hardware acceleration is unavailable (CUDA not detected) during Colab notebook execution, THEN the notebook SHALL output a diagnostic hardware warning and abort training before allocating memory.

### R6 (Ubiquitous)
The system SHALL implement a modular composite distillation loss engine in `src/engine/distillation_loss.py` calculating the total loss:
$$\mathcal{L}_{total} = \alpha \mathcal{L}_{hard} + \beta \mathcal{L}_{KD}(T, S; \tau) + \gamma \mathcal{L}_{feat}$$
where $\alpha$, $\beta$, $\gamma \ge 0.0$ and temperature $\tau \ge 1.0$.

### R7 (Ubiquitous)
The hard task loss $\mathcal{L}_{hard}$ SHALL compute standard detection loss combining Complete Intersection over Union (CIoU) bounding box regression loss, Distribution Focal Loss (DFL), and binary cross-entropy class loss between student predictions and ground truth labels.

### R8 (Ubiquitous)
The response-based distillation loss $\mathcal{L}_{KD}$ SHALL compute Kullback-Leibler (KL) divergence between temperature-scaled teacher logits and student logits:
$$\mathcal{L}_{KD} = \tau^2 \times \text{KL}\left(\text{Softmax}\left(\frac{\mathbf{z}_s}{\tau}\right) \parallel \text{Softmax}\left(\frac{\mathbf{z}_t}{\tau}\right)\right)$$
scaled by $\tau^2$ to maintain gradient magnitude consistency.

### R9 (Ubiquitous)
The feature-based distillation loss $\mathcal{L}_{feat}$ SHALL compute mean squared error (MSE) or cosine distance between intermediate neck/backbone feature maps of teacher and student projected through a $1\times 1$ convolutional adaptation layer.

### R10 (Unwanted)
IF any loss component evaluates to `NaN` or `Inf` during backward propagation, THEN the distillation loss engine SHALL raise a `DistillationLossError` with tensor statistics and abort the batch step.

### R11 (Ubiquitous)
The training engine SHALL support mixed-precision training (Automatic Mixed Precision via `torch.cuda.amp.autocast`) and distributed data parallel (DDP) execution across multi-GPU Colab configurations.

### R12 (Event-driven)
WHEN distillation completes, the student detector (YOLO11n) SHALL achieve at least $92\%$ of the teacher's (YOLO11x) baseline validation `mAP@50` on the intersection dataset.

### R13 (Ubiquitous)
The student detector model parameters SHALL NOT exceed $3.0\times 10^6$ parameters, achieving greater than an $80\%$ reduction in parameter count compared to the teacher model ($>50\times 10^6$ parameters).

### R14 (Event-driven)
WHEN each training epoch finishes, the distillation pipeline SHALL evaluate validation metrics (`mAP@50`, `mAP@50-95`, validation loss) and update a structured epoch log artifact.

### R15 (Event-driven)
WHEN distillation training concludes successfully, the system SHALL export the optimal student weights to PyTorch checkpoint format (`models/distilled/yolo11n_distilled.pt`) and Open Neural Network Exchange format (`models/distilled/yolo11n_distilled.onnx`).

### R16 (Event-driven)
WHEN ONNX export is executed, the system SHALL validate the exported ONNX model structure, checking input tensor shape ($[1, 3, 640, 640]$) and dynamic batching compatibility.

### R17 (Optional)
WHERE remote MLOps credentials are configured, the pipeline SHALL push exported model checkpoints, hyperparameters ($\alpha, \beta, \gamma, \tau$), and final evaluation metrics to the MLFlow Model Registry and Hugging Face Model Hub.

### R18 (Ubiquitous)
The system SHALL expose modular Python interfaces and CLI entrypoints in `src/engine/` (`sva-distill-loss`, `sva-export-onnx`) declared in `pyproject.toml`.

### R19 (Ubiquitous)
The automated unit test suite for distillation loss formulation, temperature scaling, feature projection, and ONNX export configuration SHALL execute without external network access in under $5.0$ seconds.

### R20 (Event-driven)
WHEN automated tests run, the system SHALL verify mathematical correctness of $\mathcal{L}_{KD}$ and $\mathcal{L}_{feat}$ gradients using synthetic mock tensors without requiring physical GPU hardware or full model training runs.
