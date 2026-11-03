# Tasks: Docker Environment and Dependency Scaffolding

## Feature Metadata
- **ID:** SVA-PRE-001
- **Name:** docker_environment_and_dependency_scaffolding
- **Status:** spec_ready

## Sequential Implementation Checklist

- [x] T1 — Create workspace directory layout: scaffold root development directory `src/` containing modular packages `src/{api,core,engine,pipeline,storage,utils}`, alongside `frontend/`, `configs/`, `data/`, `models/`, `colab_notebooks/`, and `docker/`. Covers: R1, R2.
- [x] T2 — Author `docker/Dockerfile.api` configuring a multi-stage Python 3.11-slim container for CPU edge inference omitting GPU/CUDA bloat. Covers: R8.
- [x] T3 — Author `docker/Dockerfile.mlflow` configuring a lightweight MLflow container backed by SQLite and local file volume storage. Covers: R9.
- [x] T4 — Author `docker/Dockerfile.frontend` configuring a Node.js multi-stage build producing an optimized standalone Next.js deployment. Covers: R10.
- [x] T5 — Author `docker/.dockerignore` to filter Python cache, virtual environments, datasets, and local model weights from image contexts. Covers: R8, R9, R10.
- [x] T6 — Author `docker-compose.yml` defining services `edge-api`, `mlflow-server`, and `nextjs-ui` with explicit memory caps (`mem_limit: 4g`, `1.5g`, `1g`), network isolation, volume mounts, and health check configurations. Covers: R3, R4, R5, R6, R7, R11, R12, R13.
- [x] T7 — Update `pyproject.toml` with edge runtime dependencies (`fastapi`, `uvicorn`, `opencv-python-headless`, `openvino`, `llama-cpp-python`, `mlflow`, `supervision`, `pydantic`, `httpx`, `psutil`) and test dependencies (`pytest`, `pytest-asyncio`, `pyyaml`). Covers: R14.
- [x] T8 — Implement automated test `tests/test_workspace_layout.py` to verify the presence of all required workspace root directories and modular subpackages in `src/`. Covers: R1, R2.
- [x] T9 — Implement automated test `tests/test_docker_compose_budget.py` validating compose configuration syntax, service definitions, per-container memory limits, total container budget <= 6.5GB, and health check specifications. Covers: R3, R4, R5, R6, R7, R11, R12, R13, R15, R17.
- [x] T10 — Implement automated test `tests/test_dependency_specs.py` verifying that all mandatory edge dependencies are declared in `pyproject.toml` and resolvable without version conflicts. Covers: R14, R16.
- [x] T11 — Execute `./init.sh` and run test suite `python -m pytest tests/` to confirm green verification. Covers: R15, R16, R17.

---

## Requirement Coverage Verification Matrix

| Requirement | Covering Tasks |
|---|---|
| R1 | T1, T8 |
| R2 | T1, T8 |
| R3 | T6, T9 |
| R4 | T6, T9 |
| R5 | T6, T9 |
| R6 | T6, T9 |
| R7 | T6, T9 |
| R8 | T2, T5 |
| R9 | T3, T5 |
| R10 | T4, T5 |
| R11 | T6, T9 |
| R12 | T6, T9 |
| R13 | T6, T9 |
| R14 | T7, T10 |
| R15 | T9, T11 |
| R16 | T10, T11 |
| R17 | T9, T11 |
