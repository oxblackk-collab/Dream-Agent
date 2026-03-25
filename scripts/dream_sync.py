"""Dream Sync — Mycelium as subconscious layer over engram.

Reads new observations from engram's SQLite, ingests them into
a persistent Mycelium substrate, runs consolidation (dreaming),
and reports cross-domain discoveries.

Run daily, or after significant work sessions.

Usage:
    python scripts/dream_sync.py
    python scripts/dream_sync.py --report-only  # just show discoveries
"""

from __future__ import annotations

import argparse
import json
import logging
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from mycelium.core.cell import CellState, OriginContext
from mycelium.core.substrate import Substrate
from mycelium.embedding.embedder import SentenceTransformerEmbedder
from mycelium.storage.store import SubstrateStore

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s -- %(message)s",
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


def fetch_new_observations(since: str) -> list[dict]:
    """Fetch engram observations newer than `since`."""
    if not ENGRAM_DB.exists():
        logger.warning("Engram DB not found: %s", ENGRAM_DB)
        return []

    conn = sqlite3.connect(str(ENGRAM_DB))
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT id, type, title, content, project, scope, "
        "created_at FROM observations "
        "WHERE deleted_at IS NULL AND created_at > ? "
        "ORDER BY created_at",
        (since,),
    ).fetchall()
    conn.close()

    return [dict(r) for r in rows]


def run_dream_sync(report_only: bool = False) -> None:
    print("\n" + "=" * 60)
    print("DREAM AGENT -- DREAM SYNC")
    print("Subconscious processing of engram memories")
    print("=" * 60)

    # Load sync state
    state = load_sync_state()
    last_sync = state["last_sync"]
    print(f"\n  Last sync: {last_sync}")

    if not report_only:
        # Fetch new observations from engram
        new_obs = fetch_new_observations(last_sync)
        print(f"  New observations: {len(new_obs)}")

        if not new_obs:
            print("  Nothing new to process.")
            # Still run dreaming on existing substrate
        else:
            # Initialize or load substrate
            embedder = SentenceTransformerEmbedder()
            store = SubstrateStore(db_path=MYCELIUM_DB)

            substrate = Substrate(
                embedder=embedder,
                initial_radius=0.57,
                promotion_threshold=0.55,
                recursion_depth_limit=1,
                dream_significance_threshold=0.51,
            )

            # Load existing state if available
            if MYCELIUM_DB.exists():
                cells = store.load_cells()
                intersections = store.load_intersections()
                if cells:
                    substrate._cells.update(cells)
                    substrate._intersections.update(intersections)
                    print(
                        f"  Loaded existing substrate:"
                        f" {len(cells)} cells,"
                        f" {len(intersections)} intersections"
                    )

            # Ingest new observations
            print("\n  Ingesting...")
            for obs in new_obs:
                text = f"{obs['title']}: {obs['content']}"
                domain = obs.get("project", "general")

                substrate.ingest(
                    text=text,
                    source=f"engram:{domain}",
                    participant_id="claude",
                    domain=domain,
                )
                substrate.tick()

            print(
                f"  Substrate now:"
                f" {substrate.cell_count} cells,"
                f" {substrate.intersection_count} intersections"
            )

            # Dream
            print("\n  Dreaming...")
            total_disc = 0
            for cycle in range(1, 4):
                disc = substrate.consolidate(
                    pairs_per_cycle=10000
                )
                total_disc += len(disc)
                if disc:
                    print(
                        f"    Cycle {cycle}:"
                        f" {len(disc)} discoveries"
                    )

            # Save state
            snap = substrate.get_state_snapshot()
            store.save_snapshot(snap)

            # Update sync timestamp
            latest = max(o["created_at"] for o in new_obs)
            state["last_sync"] = latest
            state["total_cells"] = substrate.cell_count
            state["total_intersections"] = (
                substrate.intersection_count
            )
            state["last_dream_discoveries"] = total_disc
            save_sync_state(state)

            print(f"\n  Dream discoveries: {total_disc}")
            print(f"  State saved to: {MYCELIUM_DB}")
    else:
        # Report only — load existing substrate
        store = SubstrateStore(db_path=MYCELIUM_DB)
        if not MYCELIUM_DB.exists():
            print("  No substrate found. Run sync first.")
            return

    # -- REPORT: Cross-domain discoveries --
    print("\n" + "=" * 60)
    print("CROSS-DOMAIN DISCOVERIES")
    print("Connections the conscious mind didn't make")
    print("=" * 60)

    store = SubstrateStore(db_path=MYCELIUM_DB)
    cells = store.load_cells()
    intersections = store.load_intersections()

    # Find cross-domain pairs between original cells
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

    discoveries.sort(
        key=lambda t: t[0].significance, reverse=True
    )

    if not discoveries:
        print("\n  No cross-domain discoveries yet.")
        print("  Feed more data and dream again.")
    else:
        for i, (ix, ca, cb) in enumerate(discoveries[:10], 1):
            title_a = ca.text.split(":")[0] if ":" in ca.text else ca.text[:60]
            title_b = cb.text.split(":")[0] if ":" in cb.text else cb.text[:60]
            print(
                f"\n  #{i} [{ca.domain} <-> {cb.domain}]"
                f" sig={ix.significance:.4f}"
            )
            print(f"    A: {title_a}")
            print(f"    B: {title_b}")

    # Stats
    active = sum(
        1 for c in cells.values() if c.state == CellState.ACTIVE
    )
    promoted = sum(
        1
        for c in cells.values()
        if c.origin.context == OriginContext.SYNTHESIS
    )
    print(f"\n  Total cells: {len(cells)} (active={active},"
          f" promoted={promoted})")
    print(f"  Total intersections: {len(intersections)}")
    print(f"  Cross-domain discoveries: {len(discoveries)}")
    print("\n  The substrate dreams.\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Dream Sync -- Mycelium subconscious"
    )
    parser.add_argument(
        "--report-only",
        action="store_true",
        help="Only show discoveries, don't sync",
    )
    args = parser.parse_args()
    run_dream_sync(report_only=args.report_only)
