# Agentic AI Generative System Framework

A universal blueprint for building modular, agentic AI systems incorporating RAG, dynamic orchestration, and observable memory.

> This project is a masterclass in **Harness Engineering**. It demonstrates how to use a Multi-Agent Orchestration layer to build a complex, modular AI system using Spec Driven Development (SDD).

## 👤 About the Architect
This framework is designed and maintained by **Manuel Camacho**, an AI Engineer specializing in the intersection of Large Language Models (LLMs) and autonomous agentic workflows. With a focus on modularity, observability, and rigorous engineering standards, Manuel builds systems that move beyond simple prompt-chains into robust, state-driven autonomous architectures.

## The Framework Vision
The goal is to move away from monolithic LLM chains toward a **Modular Agentic Graph**, where specialized nodes handle specific intents, knowledge is decoupled via a standardized RAG pipeline, and memory is persistent and observable.

## The Harness Architecture
The repository itself is the "Control Plane" for the development of the framework:

| Pillar | Manifestation |
| :--- | :--- |
| **Orchestration** | `leader` $\to$ `spec_author` $\to$ `implementer` $\to$ `reviewer` |
| **Process** | SDD (EARS Requirements $\to$ Design $\to$ Tasks $\to$ Code) |
| **Truth** | `feature_list.json` (The Ledger), `specs/` (The Blueprint) |
| **Verification** | Traceability Mapping (Requirement $\to$ Test) |

## Blueprint Flow
1. **Knowledge Pipeline**: Document Extraction $\to$ Semantic Chunking $\to$ Embedding $\to$ Indexing.
2. **Agentic Graph**: Intent Router $\to$ Planner $\to$ RAG Retrieval $\to$ Specialized Processing $\to$ Synthesis.
3. **Interface Layer**: Context Injection $\to$ Multi-Modal Generation $\to$ Session Persistence.

## Getting Started
1. Execute `./init.sh` to verify the harness environment.
2. Review `feature_list.json` for the current roadmap.
3. Follow the SDD workflow: **Spec $\to$ Approval $\to$ Implement $\to$ Review**.
