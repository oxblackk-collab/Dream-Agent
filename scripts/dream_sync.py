"""Dream Sync — Mycelium as subconscious layer over engram.

Reads new observations from engram's SQLite, ingests them into
a persistent Mycelium substrate, runs consolidation (dreaming),
and reports cross-domain discoveries.

Run daily, or after significant work sessions.

Usage:
    uv run python scripts/dream_sync.py
    uv run python scripts/dream_sync.py --report-only  # just show discoveries
"""

from __future__ import annotations

import argparse
import json
import logging
import sqlite3
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Generator

sys.path.insert(0, str(Path(__file__).parent.parent))

from mycelium.core.cell import CellState, OriginContext
from mycelium.core.substrate import Substrate
from mycelium.embedding.embedder import SentenceTransformerEmbedder
from mycelium.storage.store import SubstrateStore

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s — %(message)s",
)
logger = logging.getLogger(__name__)

ENGRAM_DB = Path.home() / ".engram" / "engram.db"
MYCELIUM_DB = Path("data/dream_substrate.db")
SYNC_STATE = Path("data/dream_sync_state.json")


def load_sync_state() -> dict:
    """Load last sync timestamp."""
    if SYNC_STATE.exists():
        return json.loads(SYNC_STATE.read_text())
    return {"last_sync": "2000-01-01T00:00:00"}


def save_sync_state(state: dict) -> None:
    SYNC_STATE.parent.mkdir(parents=True, exist_ok=True)
    SYNC_STATE.write_text(json.dumps(state, indent=2))


@contextmanager
def _engram_connection() -> Generator[sqlite3.Connection, None, None]:
    """Context manager that properly opens AND closes the engram DB connection."""
    conn = sqlite3.connect(str(ENGRAM_DB))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()


def fetch_new_observations(since: str) -> list[dict]:
    """Fetch engram observations newer than `since`."""
    if not ENGRAM_DB.exists():
        logger.warning("Engram DB not found: %s", ENGRAM_DB)
        return []

    with _engram_connection() as conn:
        rows = conn.execute(
            "SELECT id, type, title, content, project, scope, "
            "created_at FROM observations "
            "WHERE deleted_at IS NULL AND created_at > ? "
            "ORDER BY created_at",
            (since,),
        ).fetchall()

    return [dict(r) for r in rows]


def _process_observations(
    observations: list[dict],
    store: SubstrateStore,
    substrate: Substrate,
) -> int:
    """Ingest observations, run dreaming, and save. Return discovery count."""
    if MYCELIUM_DB.exists():
        cells = store.load_cells()
        intersections = store.load_intersections()
        if cells:
            substrate.load_state(cells, intersections)
            logger.info(
                "  Loaded existing substrate: %d cells, %d intersections",
                len(cells),
                len(intersections),
            )

    logger.info("  Ingesting...")
    for obs in observations:
        text = f"{obs['title']}: {obs['content']}"
        domain = obs.get("project", "general")
        substrate.ingest(
            text=text,
            source=f"engram:{domain}",
            participant_id="claude",
            domain=domain,
        )
        substrate.tick()

    logger.info(
        "  Substrate now: %d cells, %d intersections",
        substrate.cell_count,
        substrate.intersection_count,
    )

    logger.info("  Dreaming...")
    total_disc = 0
    for cycle in range(1, 4):
        disc = substrate.consolidate(pairs_per_cycle=10000)
        total_disc += len(disc)
        if disc:
            logger.info("    Cycle %d: %d discoveries", cycle, len(disc))

    snap = substrate.get_state_snapshot()
    store.save_snapshot(snap)
    return total_disc


def _collect_cross_domain(cells: dict, intersections: dict) -> list[tuple]:
    """Filter intersections to only cross-domain, non-synthesis discoveries."""
    discoveries = []
    for ix in intersections.values():
        ca = cells.get(ix.parent_a_id)
        cb = cells.get(ix.parent_b_id)
        if not ca or not cb:
            continue
        if ca.origin.context == OriginContext.SYNTHESIS:
            continue
        if cb.origin.context == OriginContext.SYNTHESIS:
            continue
        if ca.domain == cb.domain:
            continue
        if not ca.text or not cb.text:
            continue
        discoveries.append((ix, ca, cb))
    discoveries.sort(key=lambda t: t[0].significance, reverse=True)
    return discoveries


def _report_discoveries() -> None:
    """Load substrate and report cross-domain discoveries."""
    logger.info("=" * 60)
    logger.info("CROSS-DOMAIN DISCOVERIES")
    logger.info("Connections the conscious mind didn't make")
    logger.info("=" * 60)

    store = SubstrateStore(db_path=MYCELIUM_DB)
    cells = store.load_cells()
    intersections = store.load_intersections()
    discoveries = _collect_cross_domain(cells, intersections)

    if not discoveries:
        logger.info("  No cross-domain discoveries yet.")
        logger.info("  Feed more data and dream again.")
    else:
        for i, (ix, ca, cb) in enumerate(discoveries[:10], 1):
            title_a = ca.text.split(":")[0] if ":" in ca.text else ca.text[:60]
            title_b = cb.text.split(":")[0] if ":" in cb.text else cb.text[:60]
            logger.info(
                "  #%d [%s <-> %s] sig=%.4f", i, ca.domain, cb.domain, ix.significance
            )
            logger.info("    A: %s", title_a)
            logger.info("    B: %s", title_b)

    active = sum(1 for c in cells.values() if c.state == CellState.ACTIVE)
    promoted = sum(
        1 for c in cells.values() if c.origin.context == OriginContext.SYNTHESIS
    )
    logger.info(
        "  Total cells: %d (active=%d, promoted=%d)", len(cells), active, promoted
    )
    logger.info("  Total intersections: %d", len(intersections))
    logger.info("  Cross-domain discoveries: %d", len(discoveries))
    logger.info("  The substrate dreams.")


def run_dream_sync(report_only: bool = False) -> None:
    logger.info("=" * 60)
    logger.info("MYCELIUM — DREAM SYNC")
    logger.info("Subconscious processing of engram memories")
    logger.info("=" * 60)

    state = load_sync_state()
    last_sync = state["last_sync"]
    logger.info("  Last sync: %s", last_sync)

    if not report_only:
        new_obs = fetch_new_observations(last_sync)
        logger.info("  New observations: %d", len(new_obs))

        if not new_obs:
            logger.info("  Nothing new to process.")
        else:
            embedder = SentenceTransformerEmbedder()
            store = SubstrateStore(db_path=MYCELIUM_DB)
            substrate = Substrate(
                embedder=embedder,
                initial_radius=0.57,
                promotion_threshold=0.55,
                recursion_depth_limit=1,
                dream_significance_threshold=0.51,
            )

            total_disc = _process_observations(new_obs, store, substrate)

            latest = max(o["created_at"] for o in new_obs)
            state["last_sync"] = latest
            state["total_cells"] = substrate.cell_count
            state["total_intersections"] = substrate.intersection_count
            state["last_dream_discoveries"] = total_disc
            save_sync_state(state)

            logger.info("  Dream discoveries: %d", total_disc)
            logger.info("  State saved to: %s", MYCELIUM_DB)
    else:
        if not MYCELIUM_DB.exists():
            logger.info("  No substrate found. Run sync first.")
            return

    _report_discoveries()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dream Sync — Mycelium subconscious")
    parser.add_argument(
        "--report-only",
        action="store_true",
        help="Only show discoveries, don't sync",
    )
    args = parser.parse_args()
    run_dream_sync(report_only=args.report_only)
