"""Dream Agent Service — standalone substrate server.

Starts a FastAPI server with the REST API, Dream agent endpoints,
and automatic background consolidation (Dreamer daemon).

Usage:
    python scripts/run_service.py
    python scripts/run_service.py --db data/substrate.db
    python scripts/run_service.py --port 8080
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s -- %(message)s",
)
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Dream Agent Service"
    )
    parser.add_argument(
        "--db",
        type=str,
        default=None,
        help="SQLite database to load/save substrate state",
    )
    parser.add_argument(
        "--host", type=str, default="127.0.0.1"
    )
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    import uvicorn
    from fastapi import FastAPI

    from mycelium.api.router import create_api_router
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

    # Load existing state if --db provided
    store = None
    if args.db:
        from mycelium.storage.store import SubstrateStore

        db_path = Path(args.db)
        if db_path.exists():
            store = SubstrateStore(db_path=db_path)
            logger.info("Loading substrate from %s", db_path)
            cells = store.load_cells()
            intersections = store.load_intersections()
            substrate._cells.update(cells)
            substrate._intersections.update(intersections)
            logger.info(
                "Loaded %d cells, %d intersections",
                len(cells),
                len(intersections),
            )
        else:
            store = SubstrateStore(db_path=db_path)
            logger.info(
                "DB %s does not exist, starting fresh", db_path
            )

    app = FastAPI(
        title="Dream Agent",
        description="Cognitive Memory Agent — Mycelium Substrate Service",
        version="0.1.0",
    )
    app.include_router(
        create_api_router(substrate, store=store), prefix="/api"
    )

    # Mount Dream agent endpoints if store is available
    if store is not None:
        from dream.router import create_dream_router

        app.include_router(
            create_dream_router(substrate, store), prefix="/api"
        )
        logger.info("Dream agent endpoints mounted")

    # Start Dreamer daemon for automatic background consolidation
    dreamer = None
    if store is not None:
        from mycelium.consolidation.dreamer import Dreamer

        def _get_last_session_context() -> str:
            """Get session context from last processed commit."""
            try:
                from dream.inbox import InboxProcessor
                proc = InboxProcessor()
                return proc.last_session_context()
            except Exception:
                return ""

        def _persist_discoveries(discoveries):
            """Callback: persist dream discoveries to disk, then auto-wake."""
            for ix in discoveries:
                store.save_intersection(ix)
                # Also persist dream_log entry so auto-wake can find it
                try:
                    store.save_dream_log_entry(
                        str(ix.id), ix.discovered_at.isoformat(),
                    )
                except Exception as exc:
                    logger.warning("Failed to persist dream_log entry: %s", exc)
            logger.info("Persisted %d dream discoveries to disk", len(discoveries))

            # Auto-wake high-significance discoveries
            try:
                from dream.wake import auto_wake
                ctx = _get_last_session_context()
                verified = auto_wake(
                    discoveries, substrate, store,
                    session_context=ctx,
                )
                if verified > 0:
                    logger.info("Auto-wake: %d connections verified", verified)
            except Exception as exc:
                logger.error("Auto-wake failed: %s", exc)

        dreamer = Dreamer(
            substrate=substrate,
            min_idle_seconds=30,       # Dream after 30s of inactivity
            pairs_per_cycle=5000,      # Reasonable batch per dream
            poll_interval_seconds=10,  # Check every 10s
            on_discoveries=_persist_discoveries,
        )
        dreamer.start()
        logger.info("Dreamer daemon started (auto-consolidation)")

    logger.info("=" * 50)
    logger.info("DREAM AGENT SERVICE")
    logger.info("=" * 50)
    logger.info("API:  http://%s:%d/api/health", args.host, args.port)
    logger.info("Docs: http://%s:%d/docs", args.host, args.port)
    if args.db:
        logger.info("DB:   %s", args.db)
    if store is not None:
        logger.info("Dream: http://%s:%d/api/dream/unseen", args.host, args.port)
    if dreamer is not None:
        logger.info("Dreamer: active (30s idle, 5000 pairs/cycle, auto-wake)")
    logger.info("=" * 50)
    logger.info("The substrate dreams.")

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()

