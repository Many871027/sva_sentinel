# Coding Conventions

> Extreme homogeneity ensures that AI agents can predictably navigate and modify the codebase.

## Python Style

- **Version:** Python 3.10+ (Mandatory use of `asyncio` and type hinting).
- **Formatting:** PEP 8 compliance. Max line length 100 chars.
- **Async Patterns:** Prefer `async def` for all framework components. Use `await` for all I/O.
- **Strings:** Double quotes `"..."` universally.
- **Interpolation:** f-strings are mandated.

## Nomenclature

| Type                    | Convention        | Example               |
|-------------------------|-------------------|-----------------------|
| Modules                 | `snake_case`      | `intent_router.py`    |
| Classes                 | `PascalCase`      | `StateGraphManager`    |
| Functions / Variables   | `snake_case`      | `classify_intent`      |
| Constants               | `UPPER_SNAKE`     | `DEFAULT_EMBEDDING_MODEL` |
| State Keys               | `snake_case`      | `retrieved_context`    |

## Project Structure

The framework adheres to a domain-driven layout:

```
src/
├── api/            # Gateway, Middleware, Schemas (FastAPI)
├── core/           # Graph Logic, State Definitions, Router
├── engine/         # LLM Wrappers, Embedding Logic
├── pipeline/       # Ingestion, Chunking, Vectorization
├── storage/        # VectorStore Interfaces, Redis Memory
└── utils/          # Observability, Logging, Tracing
```

Every file must start with:
```python
"""A single-line string defining the module's core purpose."""
from __future__ import annotations
import asyncio
# Other imports...
```

## Testing Standard

- **Async Testing**: Use `unittest.IsolatedAsyncioTestCase` or `pytest-asyncio`.
- **Mocking**: Use `unittest.mock` to simulate LLM responses and Vector Store returns.
- **Traceability**: Tests must validate not just the final output, but the path taken through the graph (Node A $\to$ Node B).
- **Naming**: `test_<function>_<scenario>_<expected_result>` (e.g., `test_router_unknown_intent_returns_fallback`).

## Error Handling

Framework exceptions reside in `src/core/exceptions.py`:
```python
class FrameworkError(Exception):
    """Base class for all framework errors."""

class RoutingError(FrameworkError):
    """Raised when the intent router cannot resolve a path."""
```
The API Gateway traps these and returns standardized JSON error responses with appropriate HTTP status codes.
