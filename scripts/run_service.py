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
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s — %(message)s",
)
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Mycelium Substrate Service"
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
    parser.add_argument(
        "--no-viz",
        action="store_true",
        help="Disable 3D Microscope, API only",
    )
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
        title="Mycelium",
        description="Cognitive Substrate Service",
        version="0.1.0",
    )
    app.include_router(
        create_api_router(substrate, store=store), prefix="/api"
    )

    # Mount Dream agent endpoints if store is available
    if store is not None:
        from mycelium.dream.router import create_dream_router

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
                from mycelium.dream.inbox import InboxProcessor
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
                    with store._conn() as conn:
                        conn.execute(
                            "INSERT OR IGNORE INTO dream_log"
                            " (intersection_id, discovered_at, description)"
                            " VALUES (?,?,?)",
                            (str(ix.id), ix.discovered_at.isoformat(), ""),
                        )
                except Exception as exc:
                    logger.warning("Failed to persist dream_log entry: %s", exc)
            logger.info("Persisted %d dream discoveries to disk", len(discoveries))

            logger.info("Auto-wake disabled (using queue-based evaluation)")

        dreamer = Dreamer(
            substrate=substrate,
            min_idle_seconds=30,       # Dream after 30s of inactivity
            pairs_per_cycle=5000,      # Reasonable batch per dream
            poll_interval_seconds=10,  # Check every 10s
            on_discoveries=_persist_discoveries,
        )
        dreamer.start()
        logger.info("Dreamer daemon started (auto-consolidation)")

    print("\n" + "=" * 50)
    print("MYCELIUM SERVICE")
    print("=" * 50)
    print(f"  API:  http://{args.host}:{args.port}/api/health")
    print(f"  Docs: http://{args.host}:{args.port}/docs")
    if args.db:
        print(f"  DB:   {args.db}")
    if store is not None:
        print(f"  Dream: http://{args.host}:{args.port}/api/dream/unseen")
    if dreamer is not None:
        print(f"  Dreamer: active (30s idle, 5000 pairs/cycle, auto-wake)")
    print(f"  Viz:  {'disabled' if args.no_viz else 'TODO'}")
    print("=" * 50)
    print("  The future has to be free.\n")

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
