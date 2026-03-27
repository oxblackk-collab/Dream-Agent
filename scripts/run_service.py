"""Mycelium Service — standalone substrate server.

Starts a FastAPI server with the REST API and optionally the
3D Microscope visualization.

Usage:
    uv run python scripts/run_service.py
    uv run python scripts/run_service.py --db data/substrate.db
    uv run python scripts/run_service.py --port 8080 --no-viz
"""

from __future__ import annotations

import argparse
import json
import logging
import sqlite3
import sys
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mycelium.consolidation.dreamer import Dreamer
    from mycelium.core.substrate import Substrate
    from mycelium.storage.store import SubstrateStore

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s — %(message)s",
)
logger = logging.getLogger(__name__)


def _load_substrate(
    db_path: Path | None,
) -> tuple[Substrate, SubstrateStore | None]:
    """Create substrate and optionally load state from DB.

    Returns (substrate, store_or_none).
    """
    from mycelium.core.substrate import Substrate
    from mycelium.embedding.embedder import SentenceTransformerEmbedder

    embedder = SentenceTransformerEmbedder()
    substrate = Substrate(
        embedder=embedder,
        initial_radius=0.57,
        promotion_threshold=0.55,
        recursion_depth_limit=1,
        dream_significance_threshold=0.51,
    )

    store = None
    if db_path is not None:
        from mycelium.storage.store import SubstrateStore

        if db_path.exists():
            store = SubstrateStore(db_path=db_path)
            logger.info("Loading substrate from %s", db_path)
            cells = store.load_cells()
            intersections = store.load_intersections()
            substrate.load_state(cells, intersections)
            logger.info(
                "Loaded %d cells, %d intersections",
                len(cells),
                len(intersections),
            )
        else:
            store = SubstrateStore(db_path=db_path)
            logger.info("DB %s does not exist, starting fresh", db_path)

    return substrate, store


def _get_last_session_context() -> str:
    """Fetch the most recent session context from Dream inbox."""
    try:
        from dream.inbox import InboxProcessor

        proc = InboxProcessor()
        return proc.last_session_context()
    except (ImportError, OSError, json.JSONDecodeError):
        return ""


def _persist_discoveries(
    discoveries: list,
    store: SubstrateStore,
    substrate: Substrate,
) -> None:
    """Save discoveries to disk and trigger auto-wake evaluation."""
    for ix in discoveries:
        store.save_intersection(ix)
        try:
            store.save_dream_log_entry(
                intersection_id=str(ix.id),
                discovered_at=ix.discovered_at.isoformat(),
            )
        except (sqlite3.Error, OSError) as exc:
            logger.warning("Failed to persist dream_log entry: %s", exc)
    logger.info("Persisted %d dream discoveries to disk", len(discoveries))

    from dream.wake import auto_wake

    ctx = _get_last_session_context()
    verified = auto_wake(discoveries, substrate, store, session_context=ctx)
    logger.info(
        "Auto-wake: %d verified from %d discoveries",
        verified,
        len(discoveries),
    )


def _start_dreamer(
    substrate: Substrate,
    store: SubstrateStore,
) -> Dreamer:
    """Start the Dreamer daemon for automatic background consolidation."""
    from mycelium.consolidation.dreamer import Dreamer

    dreamer = Dreamer(
        substrate=substrate,
        min_idle_seconds=30,
        pairs_per_cycle=5000,
        poll_interval_seconds=10,
        on_discoveries=lambda disc: _persist_discoveries(disc, store, substrate),
    )
    dreamer.start()
    logger.info("Dreamer daemon started (auto-consolidation)")
    return dreamer


def _parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the service."""
    parser = argparse.ArgumentParser(description="Mycelium Substrate Service")
    parser.add_argument(
        "--db",
        type=str,
        default=None,
        help="SQLite database to load/save substrate state",
    )
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--no-viz",
        action="store_true",
        help="Disable 3D Microscope, API only",
    )
    return parser.parse_args()


def _log_service_banner(
    args: argparse.Namespace,
    store: SubstrateStore | None,
    dreamer: Dreamer | None,
) -> None:
    """Log the startup banner with endpoint info."""
    logger.info("=" * 50)
    logger.info("MYCELIUM SERVICE")
    logger.info("=" * 50)
    logger.info("  API:  http://%s:%d/api/health", args.host, args.port)
    logger.info("  Docs: http://%s:%d/docs", args.host, args.port)
    if args.db:
        logger.info("  DB:   %s", args.db)
    if store is not None:
        logger.info("  Dream: http://%s:%d/api/dream/unseen", args.host, args.port)
    if dreamer is not None:
        logger.info("  Dreamer: active (30s idle, 5000 pairs/cycle, auto-wake)")
    logger.info("  Viz:  %s", "disabled" if args.no_viz else "TODO")
    logger.info("=" * 50)
    logger.info("  The future has to be free.")


def main() -> None:
    args = _parse_args()

    import uvicorn
    from fastapi import FastAPI

    from mycelium.api.router import create_api_router

    db_path = Path(args.db) if args.db else None
    substrate, store = _load_substrate(db_path)

    app = FastAPI(
        title="Mycelium", description="Cognitive Substrate Service", version="0.1.0"
    )
    app.include_router(create_api_router(substrate, store=store), prefix="/api")

    if store is not None:
        from dream.router import create_dream_router

        app.include_router(create_dream_router(substrate, store), prefix="/api")
        logger.info("Dream agent endpoints mounted")

    dreamer = None
    if store is not None:
        dreamer = _start_dreamer(substrate, store)

    _log_service_banner(args, store, dreamer)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
