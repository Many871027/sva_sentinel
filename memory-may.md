# Project Memory: Agentic AI Generative System Framework

## Project Essence
A universal blueprint for building modular, agentic AI systems. This is not just a codebase, but a **Harness Engineering** exemplar, demonstrating how to use Multi-Agent Orchestration and Spec Driven Development (SDD) to build complex AI systems.

## Technical Architecture
- **Core Design**: Modular Agentic Graph (State Machine) rather than linear chains.
- **Key Components**:
    - **Ingestion Plane**: Document extraction $\to$ Semantic Chunking $\to$ Vectorization.
    - **Orchestration Plane**: Intent Router $\to$ Planner $\to$ RAG Retrieval $\to$ Synthesis.
    - **Delivery Plane**: API Gateway $\to$ Multi-modal Output $\to$ Session Persistence.
- **Standards**:
    - **Async-First**: Mandatory `asyncio` for all I/O.
    - **Model/Store Agnostic**: Interface abstraction for LLMs and VectorDBs.
    - **Observability**: Traceable state changes via `trace_id`.
    - **Type Safety**: Pydantic models for all schemas.

## The Harness (Operational Framework)
- **Orchestration Layer**:
    - `leader` $\to$ `spec_author` $\to$ `implementer` $\to$ `reviewer`.
- **SDD Workflow**: `pending` $\to$ `spec_author` $\to$ `spec_ready` $\to$ [Human Approval] $\to$ `in_progress` $\to$ `implementer` $\to$ `reviewer` $\to$ `done`.
- **Verification**: Strict traceability mapping from Requirements (`R<n>`) to concrete Tests.

## Current Roadmap (`feature_list.json`)
- **CORE-001**: Standardized API Gateway.
- **CORE-002**: Agentic Intent Router.
- **CORE-003**: Open-Source Embedding Integration.
- **CORE-004**: Modular RAG Node.
- **CORE-005**: Persistent Session Memory.
- **CORE-006**: System Observability & Tracing.
- **CORE-007**: Multi-Modal Output Generator.

## Architect
- **Manuel Camacho**: AI Engineer specializing in LLMs and autonomous agentic workflows.
