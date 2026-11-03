# Requirements: llama.cpp GGUF Quantization (Q4_K_M) and Laptop CPU Execution Benchmarks

## Feature Metadata
- **ID:** SVA-QUANT-002
- **Name:** llamacpp_gguf_quantization_and_laptop_cpu_benchmarking
- **Title:** llama.cpp GGUF Quantization (Q4_K_M) and Laptop CPU Execution Benchmarks
- **Target Subsystem:** `src/engine/` (llama.cpp GGUF Quantizer, Backend Engine, CPU Benchmark, Case Evaluator), `src/pipeline/` (Pydantic Schemas), `colab_notebooks/` (Google Colab Quantization Notebook)
- **Artifact Targets:** `models/quantized/llamacpp/qwen2.5-0.5b-auditor-q4_k_m.gguf`, `colab_notebooks/02_auditor_finetuning_llamacpp.ipynb`, `data/benchmarks/auditor_cpu_benchmark_report.json`

## Acceptance Criteria Mapping
| Acceptance Criterion | Requirements |
|---|---|
| Execute `llama-quantize` in Colab to generate compact Q4_K_M GGUF model artifact | R1, R2, R3, R4, R5 |
| Integrate quantized GGUF model into local backend using `llama-cpp-python` with CPU AVX2/AVX-512 optimization | R6, R7, R8, R9, R10, R11 |
| Benchmark forensic report generation ensuring peak RAM consumption $\le 1.2\text{GB}$ RSS and response generation time $\le 3.5$ seconds | R12, R13, R14, R15, R16, R17 |
| Validate deterministic JSON schema parsing on 100 test infraction cases without out-of-memory container crashes | R18, R19, R20, R21 |
| Enterprise schema definitions, CLI registration, and fast zero-egress test harness | R22, R23, R24, R25 |

---

## Requirements (Strict EARS Notation)

### R1 (Ubiquitous)
The system SHALL provide a GGUF quantization engine in `src/engine/llamacpp_quantizer.py` and interactive cells in `colab_notebooks/02_auditor_finetuning_llamacpp.ipynb` capable of executing `llama-quantize` to quantize the base F16 GGUF auditor model (`models/auditor/qwen2.5-0.5b-auditor-f16.gguf`) into a 4-bit medium k-quant model (`models/quantized/llamacpp/qwen2.5-0.5b-auditor-q4_k_m.gguf`).

### R2 (Event-driven)
WHEN quantization is executed, the quantization engine SHALL support the `Q4_K_M` quantization type implementing 4-bit medium k-quants with mixed block quantization.

### R3 (Event-driven)
WHEN GGUF Q4_K_M quantization completes, the exported model file size SHALL achieve at least a $55\%$ reduction compared to the baseline F16 GGUF model artifact, resulting in a binary file size under $450\text{MB}$.

### R4 (Unwanted)
IF the input base GGUF model file does not exist or has an invalid file size ($< 24\text{ bytes}$), THEN the quantization engine SHALL raise a `QuantizationInputError` and abort processing before allocating resources.

### R5 (State-driven)
WHILE generating the quantized GGUF artifact, the system SHALL preserve all model metadata key-value pairs (including architecture name `qwen2`, context length $2048$, and ChatML tokenizer chat template) while updating quantization version metadata.

### R6 (Ubiquitous)
The system SHALL implement a forensic auditor backend engine in `src/engine/auditor_engine.py` utilizing `llama-cpp-python` with AVX2 and AVX-512 CPU vector instruction acceleration support.

### R7 (State-driven)
WHILE initializing the `Llama` runtime instance, the engine SHALL configure thread parallelism (`n_threads`) matching detected physical CPU cores (defaulting to 4 threads) and allocate a context window (`n_ctx`) of 2048 tokens.

### R8 (State-driven)
WHILE generating forensic verdicts from telemetry inputs, the auditor engine SHALL enforce a temperature of $0.0$ to ensure strictly deterministic greedy token generation.

### R9 (Ubiquitous)
The auditor engine SHALL implement schema-constrained decoding enforcing that model completions adhere strictly to the JSON schema of `IncidentAuditVerdict`.

### R10 (Unwanted)
IF the specified GGUF model artifact fails magic header validation or tensor loading fails, THEN the auditor engine SHALL raise a `GGUFQuantizationError` detailing the load failure.

### R11 (Ubiquitous)
The auditor engine SHALL provide an offline mock execution mode enabling test execution and schema verification without requiring native C++ compilation or pre-built `llama.cpp` shared libraries.

### R12 (Ubiquitous)
The system SHALL implement a CPU benchmark runner in `src/engine/auditor_benchmark.py` that executes warm-up and evaluation iterations measuring on-demand inference performance on the host CPU.

### R13 (Event-driven)
WHEN the benchmark runner generates a forensic dossier from an incident telemetry prompt, the end-to-end generation latency SHALL NOT exceed 3.5 seconds ($\le 3.5\text{s}$) on host CPU.

### R14 (State-driven)
WHILE generating forensic dossiers during edge CPU execution, the process peak resident set size (RSS) memory consumption SHALL NOT exceed 1.2GB RAM ($\le 1200.0\text{MB}$), remaining well within the container 4.0GB cgroup memory cap.

### R15 (Ubiquitous)
The benchmark runner SHALL record runtime telemetry metrics including mean generation latency, P95 latency, prompt processing speed (tokens/sec), token generation speed (tokens/sec), and peak RSS memory via `psutil`.

### R16 (Event-driven)
WHEN benchmark execution concludes, the runner SHALL serialize an `AuditorBenchmarkReport` JSON artifact to `data/benchmarks/auditor_cpu_benchmark_report.json` documenting latency distributions and memory budget compliance.

### R17 (Unwanted)
IF peak RSS memory exceeds 1.2GB or mean generation latency exceeds 3.5 seconds during a strict benchmark verification run, THEN the benchmark runner SHALL raise an `AuditorBenchmarkError` documenting the non-compliant metrics.

### R18 (Ubiquitous)
The system SHALL implement a batch case evaluator in `src/engine/auditor_case_evaluator.py` that evaluates the quantized auditor on a test suite of 100 diverse synthetic traffic infraction cases.

### R19 (Event-driven)
WHEN the 100 test infraction cases are evaluated, 100% of model completions SHALL parse into valid `IncidentAuditVerdict` Pydantic models without JSON decoding syntax errors.

### R20 (State-driven)
WHILE sequentially evaluating the 100 test infraction cases, the system SHALL maintain process RSS memory under 1.2GB without cumulative memory leaks or container out-of-memory crashes.

### R21 (Event-driven)
WHEN evaluating the 100 test infraction cases, the quantized auditor SHALL achieve at least $90\%$ accuracy in detecting counter-flow violations (`is_wrong_way`) and at least $85\%$ accuracy in categorizing infraction risk severity (`risk_level`).

### R22 (Ubiquitous)
The system SHALL define Pydantic schemas in `src/pipeline/schemas.py` for quantization configuration (`GGUFQuantizationConfig`), benchmark configuration (`AuditorBenchmarkConfig`), case evaluation result (`AuditorCaseResult`), and benchmark audit report (`AuditorBenchmarkReport`).

### R23 (Ubiquitous)
The system SHALL register CLI entrypoints `sva-quantize-gguf`, `sva-benchmark-auditor`, and `sva-evaluate-auditor` in `pyproject.toml`.

### R24 (Ubiquitous)
The unit and integration test suites SHALL run fully offline without GPU hardware and without external network egress, completing execution in $< 5.0$ seconds via synthetic fixtures and mock model emulators.

### R25 (Event-driven)
WHEN `./init.sh` is executed, the entire test suite including SVA-QUANT-002 tests SHALL pass with 100% green status and exit code 0.
