"""Manual wake verification — runs the async WakeFilter on high-significance dreams.

Usage:
    python scripts/run_wake_verify.py [--limit 5] [--min-sig 0.52]
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s -- %(message)s",
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = PROJECT_ROOT / "data" / "dream_substrate.db"


async def main() -> None:
    parser = argparse.ArgumentParser(description="Manual wake verification")
    parser.add_argument("--limit", type=int, default=5, help="Max connections to evaluate")
    parser.add_argument("--min-sig", type=float, default=0.52, help="Min significance threshold")
    parser.add_argument("--dry-run", action="store_true", help="Parse + print, don't persist")
    args = parser.parse_args()

    from mycelium.api.inspire import LateralConnection
    from dream.wake import WakeFilter
    from mycelium.storage.store import SubstrateStore

    store = SubstrateStore(db_path=DB_PATH)

    # Get distinct high-significance unverified dream entries
    rows = store.get_unverified_dreams(
        min_significance=args.min_sig, limit=args.limit,
    )

    if not rows:
        logger.info("No unverified dreams above significance %.2f", args.min_sig)
        return

    logger.info("Found %d distinct unverified pairs (sig >= %.2f)", len(rows), args.min_sig)

    # Load parent cells to build LateralConnection objects
    connections: list[LateralConnection] = []
    dream_ids: list[int] = []
    ix_ids: list[str] = []

    for row in rows:
        cell_a = store.get_cell_raw(row["parent_a"])
        cell_b = store.get_cell_raw(row["parent_b"])
        if not cell_a or not cell_b:
            logger.warning("Missing cell for pair %s / %s", row["parent_a"], row["parent_b"])
            continue

        connections.append(LateralConnection(
            cell_text=(cell_a.get("text") or "")[:300],
            cell_domain=cell_a.get("domain", ""),
            cell_source=cell_a.get("source", ""),
            cell_confidence=round(cell_a.get("confidence", 0.5), 3),
            cell_energy=round(cell_a.get("energy", 1.0), 3),
            connected_to_text=(cell_b.get("text") or "")[:300],
            connected_to_domain=cell_b.get("domain", ""),
            connected_to_source=cell_b.get("source", ""),
            intersection_significance=round(row["significance"], 4),
            intersection_overlap=round(row["overlap"], 4),
            intersection_novelty=round(row["novelty"], 4),
            cell_id=row["parent_a"],
            connected_to_id=row["parent_b"],
        ))
        dream_ids.append(row["dream_id"])
        ix_ids.append(row["intersection_id"])

    if not connections:
        logger.info("No valid connections to evaluate")
        return

    # Build query from domains
    domains = set()
    for c in connections:
        if c.cell_domain:
            domains.add(c.cell_domain)
        if c.connected_to_domain:
            domains.add(c.connected_to_domain)
    query = f"cross-domain connections: {', '.join(sorted(domains))}"

    logger.info("Evaluating %d connections via async WakeFilter...", len(connections))
    logger.info("Query: %s", query)

    # Run async wake filter
    wake = WakeFilter()
    results, backend = await wake.evaluate_async(query, connections)

    logger.info("Backend used: %s", backend)
    logger.info("Results: %d evaluations returned", len(results))

    if not results:
        logger.warning("No results -- check that claude CLI or ANTHROPIC_API_KEY is available")
        return

    verified_count = 0
    for wr in results:
        if wr.index >= len(connections):
            continue
        conn_obj = connections[wr.index]
        status = "VERIFIED" if wr.verified else "REJECTED"
        verified_count += wr.verified
        logger.info(
            "  [%d] %s (conf=%.2f): [%s] <-> [%s] -- %s",
            wr.index, status, wr.confidence,
            conn_obj.cell_domain, conn_obj.connected_to_domain,
            wr.reasoning,
        )

        if not args.dry_run and wr.index < len(dream_ids):
            # Persist verification to all dream_log entries with this intersection_id
            ix_id = ix_ids[wr.index]
            matching_ids = store.get_unverified_dream_log_ids(ix_id)
            for mid in matching_ids:
                store.set_wake_verification(mid, wr.verified, wr.reasoning)
            logger.info("    Persisted to %d dream_log entries (ix=%s)", len(matching_ids), ix_id[:20])

    logger.info(
        "Wake verification complete: %d/%d verified (backend=%s, dry_run=%s)",
        verified_count, len(results), backend, args.dry_run,
    )


if __name__ == "__main__":
    asyncio.run(main())

