# Dream Agent — Development Instructions

## Project Structure

- `dream/` — The Dream agent layer (inbox, wake filter, menubar, router)
- `mycelium/` — Cognitive substrate engine (core primitives, embedding, storage)
- `scripts/` — Entry points for service, menubar, sync, wake verification
- `hooks/` — Git hooks for auto-ingestion
- `launchd/` — macOS LaunchAgent templates

## Architecture

Dream is the product layer. Mycelium is the internal cognitive engine.

The Three Primitives (Exist, Recognize, Move Toward the New) in `mycelium/core/` are the foundation. Everything else (consolidation, wake filter, energy metabolism) emerges from their composition.

## Import Conventions

- Dream layer: `from dream.X import Y`
- Mycelium engine: `from mycelium.core.X import Y`
- Cross-layer: Dream imports from mycelium, never the reverse

## Running

```bash
# Service
python scripts/run_service.py --db data/dream_substrate.db

# Menubar (requires rumps)
python scripts/dream_menubar.py

# Sync from engram
python scripts/dream_sync.py

# Manual wake verification
python scripts/run_wake_verify.py --limit 5
```

## Testing

```bash
pip install -e ".[dev]"
pytest
```
