# Requirements: Docker Environment and Dependency Scaffolding

## Feature Metadata
- **ID:** SVA-PRE-001
- **Name:** docker_environment_and_dependency_scaffolding
- **Title:** Containerized Runtime Architecture and 8GB RAM Budget Allocation
- **Target Memory Ceiling:** 8.0 GB RAM (Host Reserve: >= 1.5 GB, Containers: <= 6.5 GB)

## Acceptance Criteria Mapping
| Acceptance Criterion | Requirements |
|---|---|
| Create structured workspace layout separating edge inference, Google Colab training notebooks, and frontend | R1, R2 |
| Author docker-compose.yml with memory hard limits (mem_limit: 4g for inference, 1.5g for mlflow, 1g for nextjs) | R3, R4, R5, R6, R7, R8, R9, R10 |
| Configure pyproject.toml specifying edge runtime dependencies: fastapi, uvicorn, opencv-python-headless, openvino, llama-cpp-python, mlflow, supervision, pydantic | R14, R16 |
| Verify container startup health checks without exceeding the 8GB aggregate system RAM ceiling | R4, R5, R6, R7, R11, R12, R13, R15, R17 |

---

## Requirements (Strict EARS Notation)

### R1 (Ubiquitous)
The workspace layout SHALL provide discrete root-level directories for development source code (`src/`), frontend dashboard (`frontend/`), training notebooks (`colab_notebooks/`), system configurations (`configs/`), datasets (`data/`), model artifacts (`models/`), and container configurations (`docker/`).

### R2 (Ubiquitous)
The `src/` directory SHALL contain all Python development source code, structured into modular packages: `api/`, `core/`, `engine/`, `pipeline/`, `storage/`, and `utils/`.

### R3 (Ubiquitous)
The `docker-compose.yml` file SHALL define three isolated services: `edge-api`, `mlflow-server`, and `nextjs-ui`.

### R4 (Ubiquitous)
The `docker-compose.yml` file SHALL assign a hard memory limit of 4 gigabytes (`mem_limit: 4g`) to the `edge-api` service.

### R5 (Ubiquitous)
The `docker-compose.yml` file SHALL assign a hard memory limit of 1.5 gigabytes (`mem_limit: 1.5g`) to the `mlflow-server` service.

### R6 (Ubiquitous)
The `docker-compose.yml` file SHALL assign a hard memory limit of 1 gigabyte (`mem_limit: 1g`) to the `nextjs-ui` service.

### R7 (Ubiquitous)
The `docker-compose.yml` file SHALL enforce that the aggregate memory limit of all declared container services does not exceed 6.5 gigabytes, guaranteeing at least 1.5 gigabytes of host RAM reserve.

### R8 (Ubiquitous)
The `docker/Dockerfile.api` file SHALL specify a container build for the `edge-api` service without GPU CUDA drivers or heavy training frameworks.

### R9 (Ubiquitous)
The `docker/Dockerfile.mlflow` file SHALL specify a container build for the `mlflow-server` service utilizing SQLite backend storage and volume-mounted artifacts.

### R10 (Ubiquitous)
The `docker/Dockerfile.frontend` file SHALL specify a multi-stage container build for the `nextjs-ui` service producing an optimized standalone Node.js distribution.

### R11 (Ubiquitous)
The `docker-compose.yml` file SHALL configure an HTTP health check for the `edge-api` service querying `GET /health/live`.

### R12 (Ubiquitous)
The `docker-compose.yml` file SHALL configure an HTTP health check for the `mlflow-server` service querying `GET /health`.

### R13 (Ubiquitous)
The `docker-compose.yml` file SHALL configure an HTTP health check for the `nextjs-ui` service querying `GET /api/health`.

### R14 (Ubiquitous)
The `pyproject.toml` configuration SHALL declare edge runtime dependencies including `fastapi`, `uvicorn`, `opencv-python-headless`, `openvino`, `llama-cpp-python`, `mlflow`, `supervision`, and `pydantic`.

### R15 (Event-driven)
WHEN `docker compose config` is executed against `docker-compose.yml`, the system SHALL validate the configuration syntax with exit code 0.

### R16 (Event-driven)
WHEN dependency resolution is validated via package lock tooling, the system SHALL resolve all declared dependencies in `pyproject.toml` without version conflicts.

### R17 (Unwanted)
IF the sum of all container memory limits specified in `docker-compose.yml` exceeds 6.5GB, THEN the system verification test SHALL fail with an explicit budget violation message.
