# Design: Docker Environment and Dependency Scaffolding

## Feature Metadata
- **ID:** SVA-PRE-001
- **Name:** docker_environment_and_dependency_scaffolding
- **Status:** spec_ready

## 1. Architectural Overview

The SVA Sentinel system is an edge vision and forensic auditing platform targeted to execute on standard laptop/edge hardware with an aggregate memory budget of 8GB RAM. High-compute training, distillation (YOLO11x to YOLO11n), and model quantization (INT8 OpenVINO NNCF, Q4_K_M GGUF) are offloaded to external Google Colab GPU runtimes. 

The local runtime consists of three containerized services managed via Docker Compose:
1. `edge-api` (FastAPI + OpenVINO INT8 + llama-cpp-python CPU + ByteTrack)
2. `mlflow-server` (Lightweight tracking server with SQLite)
3. `nextjs-ui` (Next.js 15+ frontend in standalone node mode)

Strict cgroup memory limits are enforced in `docker-compose.yml` to prevent Out-Of-Memory (OOM) killing or system thrashing on an 8GB RAM machine.

```
+------------------------------------------------------------------------+
|                          HOST MACHINE (8GB RAM)                        |
|                                                                        |
|  +------------------------------------------------------------------+  |
|  |                  Docker Compose Runtime (Max 6.5GB)              |  |
|  |                                                                  |  |
|  |  +---------------------+  +-----------------+  +--------------+  |  |
|  |  |      edge-api       |  |  mlflow-server  |  |  nextjs-ui   |  |  |
|  |  |  mem_limit: 4.0GB   |  |  mem_limit: 1.5GB | | mem_limit: 1GB|  |
|  |  | (FastAPI, OpenVINO, |  | (SQLite backend,|  | (Standalone  |  |  |
|  |  |  llama.cpp, OpenCV) |  |  local artifacts)| |  Node.js)    |  |  |
|  |  +---------------------+  +-----------------+  +--------------+  |  |
|  +------------------------------------------------------------------+  |
|                                                                        |
|  +------------------------------------------------------------------+  |
|  |                 Host OS & Background Headroom Reserve             |  |
|  |                           Min: 1.5GB RAM                         |  |
|  +------------------------------------------------------------------+  |
+------------------------------------------------------------------------+
```

---

## 2. Workspace Layout & Files

### Files to Create

1. **Workspace Directory Layout:**
   - `src/api/__init__.py` — API routes and schema contracts
   - `src/core/__init__.py` — Orchestration, state graph, and domain exceptions
   - `src/engine/__init__.py` — OpenVINO and llama.cpp runtime engine wrappers
   - `src/pipeline/__init__.py` — Ingestion, tracking, and homography pipeline
   - `src/storage/__init__.py` — File storage, incident clips, and persistence
   - `src/utils/__init__.py` — Observability, metrics, and logging utilities
   - `configs/.gitkeep` — Configuration files (homography, sentinel config)
   - `data/.gitkeep` — Sample frames, test streams, and calibration data
   - `models/.gitkeep` — Local INT8 OpenVINO IR and GGUF model files
   - `colab_notebooks/.gitkeep` — Distillation, quantization, and fine-tuning notebooks
   - `frontend/.gitkeep` — Next.js frontend application root

2. **Containerization Configuration:**
   - `docker-compose.yml` — Multi-service composition with cgroup memory limits and health checks
   - `docker/Dockerfile.api` — Multi-stage edge API container (Python 3.11-slim, CPU inference)
   - `docker/Dockerfile.mlflow` — Lightweight MLflow container (SQLite backend)
   - `docker/Dockerfile.frontend` — Node.js multi-stage build for Next.js standalone server
   - `docker/.dockerignore` — Docker ignore file filtering caches, models, and virtual environments

3. **Verification Tests:**
   - `tests/test_workspace_layout.py` — Validates existence of project layout directories and modular Python packages
   - `tests/test_docker_compose_budget.py` — Validates compose syntax, service definitions, memory limit caps <= 6.5GB, and health checks
   - `tests/test_dependency_specs.py` — Validates declared edge runtime dependencies in `pyproject.toml`

### Files to Mutate

- `pyproject.toml` — Declare edge runtime dependencies and development tools under `[project.dependencies]` and `[tool.uv.dev-dependencies]`.

---

## 3. Container Memory Allocation & Budget Enforcement

| Service | Container Name | Memory Hard Limit (`mem_limit`) | Role & Memory Breakdown |
|---|---|---|---|
| `edge-api` | `sva-edge-api` | `4.0 GB` (4096m) | OpenVINO INT8 model (~1.2GB peak), llama-cpp-python GGUF (~1.2GB peak), Frame ring buffer & ByteTrack (~400MB), FastAPI/Python runtime (~300MB), Headroom (~900MB) |
| `mlflow-server` | `sva-mlflow` | `1.5 GB` (1536m) | MLflow tracking server + SQLite engine + Gunicorn worker pool |
| `nextjs-ui` | `sva-nextjs` | `1.0 GB` (1024m) | Next.js 15 standalone server + WebSocket proxy client |
| **Total Container Allocation** | — | **6.5 GB** (6656m) | Strict ceiling across all Docker containers |
| **Host System Reserve** | — | **>= 1.5 GB** (1536m) | Guaranteed headroom for Windows/Linux host OS, Docker daemon, and display server |
| **Total System RAM** | — | **8.0 GB** (8192m) | Physical memory hardware envelope |

---

## 4. Edge Runtime Dependencies (`pyproject.toml`)

Edge runtime dependencies are curated to avoid heavyweight training packages (e.g., CUDA-enabled PyTorch, TensorRT, TorchVision):

```toml
[project]
name = "sva-sentinel"
version = "0.1.0"
description = "SVA Residential Traffic & Micromobility Safety Sentinel"
readme = "README.md"
requires-python = ">=3.11"
dependencies = [
    "fastapi>=0.115.0",
    "uvicorn[standard]>=0.32.0",
    "pydantic>=2.9.0",
    "opencv-python-headless>=4.10.0",
    "openvino>=2024.4.0",
    "llama-cpp-python>=0.3.0",
    "mlflow>=2.17.0",
    "supervision>=0.24.0",
    "pyyaml>=6.0.2",
    "httpx>=0.27.0",
    "psutil>=6.0.0",
]

[tool.uv]
dev-dependencies = [
    "pytest>=8.3.0",
    "pytest-asyncio>=0.24.0",
]
```

### Dependency Rationale:
- `opencv-python-headless`: Video frame decoding and drawing without X11/GUI system dependencies.
- `openvino`: Optimized CPU inference for INT8 quantized YOLO models.
- `llama-cpp-python`: Low-footprint C++ GGUF inference for local forensic auditing with AVX/AVX2 acceleration.
- `supervision`: Fast polygon zone filtering, ByteTrack support, and visual annotation utilities.
- `mlflow`: Local and remote logging of metrics and telemetry.
- `httpx`: Asynchronous HTTP client for health checks and inter-service communication.
- `psutil`: Real-time container and host memory telemetry.

---

## 5. Interface Signatures & Health Checks

### Docker Compose Health Check Specifications
- **`edge-api` Health Check:**
  - Command: `["CMD-SHELL", "curl -f http://localhost:8000/health/live || exit 1"]`
  - Interval: `15s`, Timeout: `5s`, Retries: `3`, Start Period: `10s`
- **`mlflow-server` Health Check:**
  - Command: `["CMD-SHELL", "curl -f http://localhost:5000/health || exit 1"]`
  - Interval: `20s`, Timeout: `5s`, Retries: `3`, Start Period: `10s`
- **`nextjs-ui` Health Check:**
  - Command: `["CMD-SHELL", "wget -q --spider http://localhost:3000/api/health || exit 1"]`
  - Interval: `15s`, Timeout: `5s`, Retries: `3`, Start Period: `15s`

### Budget Verification Test Logic (`tests/test_docker_compose_budget.py`)
```python
def parse_memory_limit(limit_str: str) -> float:
    """Converts memory strings like '4g', '1.5g', '1024m' into gigabytes (float)."""
    ...

def test_docker_compose_memory_budget():
    """Asserts that each service respects its individual cap and total <= 6.5GB."""
    ...
```

---

## 6. Discarded Alternatives

### Discarded Alternative 1: Monolithic Container Architecture
- **Description:** Consolidating FastAPI inference, MLflow server, and Next.js frontend into a single multi-threaded container running supervisord.
- **Why Discarded:** Monolithic packaging prevents fine-grained cgroup memory control. An unexpected memory spike in Next.js or MLflow could trigger the Linux OOM-killer to terminate the primary vision inference pipeline. Multi-service Docker Compose isolates failures and allows strict per-component resource capping.

### Discarded Alternative 2: Including Full PyTorch (CUDA) in Edge Container
- **Description:** Shipping full PyTorch with CUDA/cuDNN libraries in the edge deployment container.
- **Why Discarded:** PyTorch with CUDA libraries creates an image size exceeding 10GB and consumes > 1.5GB of baseline RAM upon import, exceeding the edge memory envelope. In this architecture, all model distillation and quantization training happen off-device in Google Colab; the edge container deploys only lean CPU runtimes (`openvino` and `llama-cpp-python`).

### Discarded Alternative 3: Kubernetes / MicroK8s Deployment on Edge Device
- **Description:** Deploying an edge Kubernetes cluster (such as MicroK8s or K3s) to schedule the microservices.
- **Why Discarded:** The control plane of K3s/MicroK8s alone consumes 1.0 to 1.5GB of RAM just idling. On an 8GB host, this would consume the entire host reserve and squeeze inference workloads. Docker Compose provides deterministic containerization with negligible daemon overhead (< 150MB).
