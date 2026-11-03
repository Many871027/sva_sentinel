# Tasks: llama.cpp GGUF Quantization (Q4_K_M) and Laptop CPU Execution Benchmarks

## Sequential Implementation Checklist

- [x] **T1 — Pydantic Schemas Definition (`src/pipeline/schemas.py`)**
  Define Pydantic schemas for GGUF quantization and CPU benchmarking:
  - `GGUFQuantizationConfig`: input/output GGUF paths, `Q4_K_M` type, minimum reduction percentage ($\ge 55\%$).
  - `AuditorBenchmarkConfig`: model path, thread count (default 4), context length (2048), latency ceiling ($3.5\text{s}$), RSS ceiling ($1200\text{MB}$), mock mode flag.
  - `AuditorCaseResult`: per-case parsing success, predicted vs. expected wrong-way flag and risk level, latency, tokens generated.
  - `AuditorBenchmarkReport`: consolidated benchmark metrics, throughput, latency distributions, peak RSS, and budget compliance booleans.
  *Covers: R15, R22.*

- [x] **T2 — Domain Exceptions Definition (`src/core/exceptions.py`)**
  Define domain exception classes:
  - `GGUFQuantizationError` inheriting from `FrameworkError` for quantization failures or corrupt GGUF files.
  - `AuditorBenchmarkError` inheriting from `FrameworkError` for CPU latency or RAM budget violations.
  *Covers: R4, R10, R17.*

- [x] **T3 — GGUF Quantizer Engine (`src/engine/llamacpp_quantizer.py`)**
  Implement `GGUFQuantizer` capable of:
  - Validating input GGUF existence and magic header `GGUF` (raising `QuantizationInputError` if missing or truncated).
  - Executing `llama-quantize` CLI when available or generating valid Q4_K_M GGUF binary artifacts preserving architecture metadata (`qwen2`, `context_length`, `chat_template`).
  - Verifying output model achieves $\ge 55\%$ file size reduction compared to base F16 GGUF model artifact.
  - Exposing CLI entrypoint `main()`.
  *Covers: R1, R2, R3, R4, R5.*

- [x] **T4 — Forensic Auditor Backend Engine (`src/engine/auditor_engine.py`)**
  Implement `ForensicAuditorEngine` wrapping `llama-cpp-python`:
  - Detect AVX2 and AVX-512 CPU SIMD capabilities on host system.
  - Configure `llama_cpp.Llama` with 4 worker threads, 2048 context length, and greedy decoding (`temperature=0.0`).
  - Implement schema-constrained JSON generation adhering to `IncidentAuditVerdict`.
  - Validate GGUF file header and raise `GGUFQuantizationError` if model loading fails.
  - Provide offline mock execution mode for fast test execution without native library compilation.
  *Covers: R6, R7, R8, R9, R10, R11.*

- [x] **T5 — CPU Benchmark Runner (`src/engine/auditor_benchmark.py`)**
  Implement `AuditorBenchmarkRunner` profiling CPU execution:
  - Execute warm-up iterations followed by measured benchmark iterations on sample telemetry prompts.
  - Measure generation latency ensuring mean response time $\le 3.5$ seconds.
  - Monitor process peak RSS memory via `psutil` ensuring RAM consumption $\le 1.2\text{GB}$ ($1200\text{MB}$).
  - Record prompt tokens/sec, generation tokens/sec, and serialize `AuditorBenchmarkReport` to `data/benchmarks/auditor_cpu_benchmark_report.json`.
  - Raise `AuditorBenchmarkError` if strict budget thresholds are violated.
  - Exposing CLI entrypoint `main()`.
  *Covers: R12, R13, R14, R15, R16, R17.*

- [x] **T6 — 100-Case Deterministic Evaluator (`src/engine/auditor_case_evaluator.py`)**
  Implement `AuditorCaseEvaluator` evaluating 100 infraction cases:
  - Generate or ingest 100 balanced telemetry cases (wrong-way vehicular, micromobility, speeding, benign control).
  - Execute sequential inference and verify $100\%$ valid JSON parsing into `IncidentAuditVerdict`.
  - Verify peak RSS memory remains under $1.2\text{GB}$ across all 100 cases without memory leaks.
  - Validate classification performance: $\ge 90\%$ wrong-way accuracy, $\ge 85\%$ risk-level accuracy.
  - Exposing CLI entrypoint `main()`.
  *Covers: R18, R19, R20, R21.*

- [x] **T7 — Google Colab Quantization Notebook Integration (`colab_notebooks/02_auditor_finetuning_llamacpp.ipynb`)**
  Add interactive cells to notebook:
  - Build `llama-quantize` tool from `llama.cpp` source in Colab GPU environment.
  - Quantize `models/auditor/qwen2.5-0.5b-auditor-f16.gguf` to `models/quantized/llamacpp/qwen2.5-0.5b-auditor-q4_k_m.gguf`.
  - Verify exported model size and compute initial CPU token generation benchmark.
  *Covers: R1, R2, R3.*

- [x] **T8 — CLI Entrypoints Registration (`pyproject.toml`)**
  Register CLI console scripts:
  - `sva-quantize-gguf = "src.engine.llamacpp_quantizer:main"`
  - `sva-benchmark-auditor = "src.engine.auditor_benchmark:main"`
  - `sva-evaluate-auditor = "src.engine.auditor_case_evaluator:main"`
  *Covers: R23.*

- [x] **T9 — GGUF Quantizer Unit Tests (`tests/test_llamacpp_quantizer.py` & `tests/fixtures/mock_gguf_quant.py`)**
  Implement unit tests verifying:
  - Q4_K_M quantization execution and $\ge 55\%$ file size reduction.
  - `QuantizationInputError` on missing or truncated input GGUF files.
  - Metadata preservation in output GGUF artifact.
  *Covers: R1, R2, R3, R4, R5, R24.*

- [x] **T10 — Auditor Engine Unit Tests (`tests/test_auditor_engine.py`)**
  Implement unit tests verifying:
  - AVX2/AVX-512 capability detection.
  - Configuration of 4 threads and 2048 context window.
  - Enforced `temperature=0.0` greedy sampling.
  - Schema-constrained parsing into `IncidentAuditVerdict`.
  - Corrupted GGUF file handling raising `GGUFQuantizationError`.
  *Covers: R6, R7, R8, R9, R10, R11, R24.*

- [x] **T11 — CPU Benchmark Runner Unit Tests (`tests/test_auditor_benchmark.py`)**
  Implement unit tests verifying:
  - Mean latency computation and $\le 3.5\text{s}$ threshold assertion.
  - `psutil` peak RSS memory tracking and $\le 1.2\text{GB}$ ceiling assertion.
  - Budget violation raising `AuditorBenchmarkError`.
  - Serialization of `AuditorBenchmarkReport` JSON artifact.
  *Covers: R12, R13, R14, R15, R16, R17, R24.*

- [x] **T12 — 100-Case Case Evaluator Unit Tests (`tests/test_auditor_case_evaluator.py`)**
  Implement unit tests verifying:
  - 100 synthetic test case generation and distribution balance.
  - $100\%$ valid JSON schema parse rate.
  - Memory stability under sustained 100-case sequential execution.
  - Classification accuracy: $\ge 90\%$ wrong-way, $\ge 85\%$ risk-level.
  *Covers: R18, R19, R20, R21, R24.*

- [x] **T13 — E2E Quantization & Benchmark Pipeline Integration Test (`tests/test_llamacpp_quant_integration.py`)**
  Implement end-to-end integration test validating complete workflow:
  - F16 GGUF $\to$ Q4_K_M quantization $\to$ engine initialization $\to$ CPU benchmark $\to$ case evaluation.
  - Sub-5.0 second offline execution speed with zero external network egress.
  *Covers: R24.*

- [x] **T14 — Test Suite and Environment Verification (`init.sh`)**
  Execute `./init.sh` and ensure 100% green test pass and spec compliance.
  *Covers: R25.*
