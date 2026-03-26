"""Inspire — lateral connection oracle for the Mycelium substrate.

Queries the substrate for cross-domain connections related to a topic.
Returns structured results that Claude can interpret and present.

Usage as library:
    from mycelium.api.inspire import inspire
    results = inspire("Elliott Waves", k=5)

Usage as CLI:
    uv run python -m mycelium.api.inspire "Elliott Waves"
    uv run python -m mycelium.api.inspire "prediction markets" --k 3
"""

from __future__ import annotations

import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

from mycelium.core.cell import CellID, CognitiveCell
from mycelium.core.intersection import Intersection
from mycelium.core.substrate import Substrate
from mycelium.embedding.embedder import SentenceTransformerEmbedder
from mycelium.storage.store import SubstrateStore

MYCELIUM_DB = Path(__file__).parent.parent.parent / "data" / "dream_substrate.db"


@dataclass
class LateralConnection:
    """A cross-domain connection discovered by the substrate."""

    cell_text: str
    cell_domain: str
    cell_source: str
    cell_confidence: float
    cell_energy: float
    connected_to_text: str
    connected_to_domain: str
    connected_to_source: str
    intersection_significance: float
    intersection_overlap: float
    intersection_novelty: float
    cell_id: str = ""
    connected_to_id: str = ""


@dataclass
class InspireResult:
    """Result of an inspire query."""

    query: str
    nearest_cells: list[dict]  # direct matches
    lateral_connections: list[LateralConnection]  # cross-domain connections
    total_cells_in_substrate: int
    total_intersections_in_substrate: int


def _load_substrate(db_path: Path = MYCELIUM_DB) -> Substrate:
    """Load the substrate from SQLite."""
    store = SubstrateStore(db_path=db_path)
    cells = store.load_cells()
    intersections = store.load_intersections()

    embedder = SentenceTransformerEmbedder()
    substrate = Substrate(
        embedder=embedder,
        initial_radius=0.57,
        promotion_threshold=0.55,
        recursion_depth_limit=1,
    )
    substrate._cells.update(cells)
    substrate._intersections.update(intersections)
    return substrate


def inspire(
    query: str,
    k: int = 5,
    max_lateral: int = 10,
    db_path: Path = MYCELIUM_DB,
) -> InspireResult:
    """Query the substrate for lateral connections.

    1. Find the k nearest cells to the query
    2. For each cell, find cross-domain intersections
    3. Return the most significant lateral connections

    Args:
        query: The topic or problem to explore
        k: Number of nearest cells to search
        max_lateral: Maximum lateral connections to return
        db_path: Path to the substrate database
    """
    substrate = _load_substrate(db_path)

    # Find nearest cells — search wider, then separate originals from promoted
    raw_results = substrate.search_by_text(query, k=k * 10, active_only=True)

    # Prefer cells with domains (originals) for cross-domain search
    domain_results = [(c, d) for c, d in raw_results if c.domain]
    no_domain_results = [(c, d) for c, d in raw_results if not c.domain]

    # Take top k with domains, fill rest from promoted if needed
    results = domain_results[:k]
    if len(results) < k:
        results.extend(no_domain_results[:k - len(results)])

    nearest_cells = []
    for cell, distance in results:
        nearest_cells.append({
            "text": cell.text[:300] if cell.text else "",
            "domain": cell.domain,
            "source": cell.origin.source,
            "distance": round(distance, 4),
            "confidence": round(cell.confidence, 3),
            "energy": round(cell.energy, 3),
        })

    # Find cross-domain connections
    seen_pairs: set[frozenset[CellID]] = set()
    all_laterals: list[tuple[float, LateralConnection]] = []

    for cell, _ in results:
        intersections = substrate.get_intersections_for(cell.id)

        for ix in intersections:
            # Get the other cell
            other_id = (
                ix.parent_b_id if ix.parent_a_id == cell.id
                else ix.parent_a_id
            )
            other = substrate.get_cell(other_id)
            if other is None:
                continue

            # Only cross-domain connections (both must have domains)
            if not cell.domain or not other.domain:
                continue
            if cell.domain == other.domain:
                continue

            # Deduplicate
            pair = frozenset({cell.id, other.id})
            if pair in seen_pairs:
                continue
            seen_pairs.add(pair)

            lateral = LateralConnection(
                cell_text=cell.text[:300] if cell.text else "",
                cell_domain=cell.domain,
                cell_source=cell.origin.source,
                cell_confidence=round(cell.confidence, 3),
                cell_energy=round(cell.energy, 3),
                connected_to_text=other.text[:300] if other.text else "",
                connected_to_domain=other.domain,
                connected_to_source=other.origin.source,
                intersection_significance=round(ix.significance, 4),
                intersection_overlap=round(ix.overlap, 4),
                intersection_novelty=round(ix.novelty, 4),
            )
            all_laterals.append((ix.significance, lateral))

    # Sort by significance, take top N
    all_laterals.sort(key=lambda t: -t[0])
    top_laterals = [lat for _, lat in all_laterals[:max_lateral]]

    return InspireResult(
        query=query,
        nearest_cells=nearest_cells,
        lateral_connections=top_laterals,
        total_cells_in_substrate=substrate.cell_count,
        total_intersections_in_substrate=substrate.intersection_count,
    )


def format_for_claude(result: InspireResult) -> str:
    """Format InspireResult as structured text for Claude to interpret."""
    lines = [
        f"## Mycelium Lateral Connections for: \"{result.query}\"",
        f"Substrate: {result.total_cells_in_substrate} cells, "
        f"{result.total_intersections_in_substrate} intersections",
        "",
    ]

    if result.nearest_cells:
        lines.append("### Nearest cells")
        for i, cell in enumerate(result.nearest_cells, 1):
            lines.append(
                f"{i}. [{cell['domain']}] (dist={cell['distance']}) "
                f"— {cell['text'][:150]}..."
            )
        lines.append("")

    if result.lateral_connections:
        lines.append("### Cross-domain connections (by significance)")
        for i, lat in enumerate(result.lateral_connections, 1):
            lines.append(
                f"{i}. **{lat.cell_domain}** ↔ **{lat.connected_to_domain}** "
                f"(sig={lat.intersection_significance}, "
                f"novelty={lat.intersection_novelty})"
            )
            lines.append(f"   From: {lat.cell_text[:120]}...")
            lines.append(f"   To: {lat.connected_to_text[:120]}...")
            lines.append("")
    else:
        lines.append("No cross-domain connections found for this query.")

    return "\n".join(lines)


def main() -> None:
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Query Mycelium for lateral connections"
    )
    parser.add_argument("query", type=str, help="Topic or problem to explore")
    parser.add_argument("--k", type=int, default=5, help="Nearest cells (default: 5)")
    parser.add_argument("--max-lateral", type=int, default=10, help="Max connections (default: 10)")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    parser.add_argument("--db", type=str, default=None, help="DB path override")
    args = parser.parse_args()

    db = Path(args.db) if args.db else MYCELIUM_DB

    result = inspire(args.query, k=args.k, max_lateral=args.max_lateral, db_path=db)

    if args.json:
        out = {
            "query": result.query,
            "nearest_cells": result.nearest_cells,
            "lateral_connections": [asdict(lc) for lc in result.lateral_connections],
            "substrate_cells": result.total_cells_in_substrate,
            "substrate_intersections": result.total_intersections_in_substrate,
        }
        print(json.dumps(out, indent=2, ensure_ascii=False))
    else:
        print(format_for_claude(result))


if __name__ == "__main__":
    main()
