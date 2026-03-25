# Code Review Rules — Dream-Agent

## Python General
- Python 3.11+ required — use modern syntax (match, type unions with |, etc.)
- Use type hints on all function signatures
- Use `from __future__ import annotations` for forward references
- Prefer dataclasses or NamedTuple over plain dicts for structured data
- No mutable default arguments
- No bare `except:` — always catch specific exceptions
- Use `logging` module, never `print()` in library code

## Architecture
- Dream layer (`dream/`) handles agent logic — inbox, wake, menubar, extraction
- Mycelium layer (`mycelium/`) is the cognitive substrate engine — cells, intersections, embeddings
- Dream imports from Mycelium, never the reverse
- All API endpoints go through FastAPI routers
- SQLite is the only persistence backend — use WAL mode for concurrent access

## Security
- NEVER hardcode API keys, tokens, or secrets
- NEVER include absolute paths with usernames
- NEVER include personal data (names, emails, hostnames)
- All secrets must come from environment variables via `os.environ.get()`
- No `subprocess.run` with `shell=True` unless absolutely necessary

## Code Quality
- No dead code — remove unused imports, functions, and variables
- No commented-out code blocks
- Keep functions under 50 lines when possible
- Use context managers for file and database operations
- Prefer early returns over deep nesting
- Use f-strings, not `.format()` or `%`

## Async
- FastAPI endpoints that call external services must be `async def`
- Use `asyncio.create_subprocess_exec` instead of `subprocess.run` in async contexts
- Never block the event loop with synchronous I/O

## Dependencies
- Minimize external dependencies
- Optional dependencies (like `rumps` for menubar) must be guarded with try/except ImportError
- Use `sentence-transformers` for embeddings, not raw transformers
