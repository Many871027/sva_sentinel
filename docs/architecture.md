# Architecture — Defining 'Quality Work'

> This document establishes the absolute quality standard for the Agentic AI Generative System Framework. Reviewer agents evaluate all code against this file.

## Core Principles

1. **Modular Graph Orchestration.** The system must be implemented as a state-driven graph (e.g., LangGraph). Logic must be decoupled into discrete "Nodes" with a shared "State" object.
2. **Interface Abstraction.** All external integrations (LLMs, Vector Stores, Databases) must be wrapped in abstract base classes (Interfaces). The system should be "Model Agnostic" and "Store Agnostic."
3. **Asynchronous First.** All I/O-bound operations (API calls, DB queries, file reads) MUST be `async`. Synchronous calls in the main execution path are prohibited.
4. **Observable State.** Every transition in the Agentic Graph must be traceable. State changes must be logged with a unique `trace_id` compatible with observability tools (MLflow/LangSmith).
5. **Strong Typing.** Use Pydantic models for all data schemas, request/response payloads, and graph state definitions to ensure runtime validation.

## System Layers

The framework is divided into three primary functional planes:

1. **Ingestion Plane**: Document $\to$ Chunk $\to$ Vector $\to$ Index. (Focus: Throughput and Semantic Precision).
2. **Orchestration Plane**: User Intent $\to$ Routing $\to$ Planning $\to$ Execution $\to$ Synthesis. (Focus: Determinism and State Management).
3. **Delivery Plane**: API Gateway $\to$ Context Injection $\to$ Multi-modal Rendering. (Focus: Latency and User Experience).

## Anti-Patterns (What NOT to do)

- ❌ **Monolithic Chains**: Do not use linear "chains" for complex logic. Use a graph-based router.
- ❌ **Hardcoded Prompts**: Do not embed prompts directly in logic. Use a prompt management strategy or external configuration.
- ❌ **Synchronous Blocking**: Never use `requests` or `time.sleep()` in the orchestration flow; use `httpx` and `asyncio.sleep()`.
- ❌ **State Leakage**: Do not use global variables for session state. All state must be encapsulated within the Graph State or a persistent store (Redis).
- ❌ **Generic Exceptions**: Do not use `Exception`. Define a hierarchy of domain exceptions (e.g., `FrameworkError` $\to$ `RoutingError`).
