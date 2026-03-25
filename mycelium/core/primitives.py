"""The Three Primitives — the irreducible operations of the Mycelium.

Every behavior in the substrate (mitosis, immune response, supersession,
consolidation, cognitive bonds) emerges from the recursive composition
of these three operations. They cannot be decomposed further.

Primitive I:   Exist — create a cell at a point in semantic space
Primitive II:  Recognize — find overlapping fields, create intersections
Primitive III: Move Toward the New — promote significant intersections, recurse
"""

from __future__ import annotations

import hashlib
import logging

from mycelium.core.cell import (
    CellID,
    CellState,
    CognitiveCell,
    EmbeddingVector,
    Origin,
    OriginContext,
)
from mycelium.core.intersection import Intersection, IntersectionID

logger = logging.getLogger(__name__)

# Type alias for the substrate's cell registry (used before Substrate class exists)
CellRegistry = dict[CellID, CognitiveCell]
IntersectionRegistry = dict[IntersectionID, Intersection]

# Default config values (overridden by Substrate when wired up)
_DEFAULT_INITIAL_RADIUS = 0.15
_DEFAULT_PROMOTION_THRESHOLD = 0.6
_DEFAULT_RECURSION_DEPTH_LIMIT = 3
_DEFAULT_VITALITY_MINIMUM = 0.3


class Primitives:
    """The three irreducible operations of the Mycelium substrate.

    All methods are static — Primitives holds no state.
    State lives in the cell and intersection registries passed as arguments.
    """

    # ─────────────────────────────────────────────────────────────
    # Primitive I: Exist
    # ─────────────────────────────────────────────────────────────

    @staticmethod
    def exist(
        cells: CellRegistry,
        embedding: EmbeddingVector,
        origin: Origin,
        radius: float = _DEFAULT_INITIAL_RADIUS,
        confidence: float = 0.8,
        parent_ids: list[CellID] | None = None,
        generation: int = 0,
        domain: str = "",
    ) -> CognitiveCell:
        """Primitive I: Create a cell at a point in semantic space.

        1. Instantiate CognitiveCell with the embedding
        2. Set initial radius (provided or default)
        3. Set initial confidence from source authority
        4. Set energy to 1.0 (full vitality at birth)
        5. Register in the cell registry
        6. Return the new cell
        """
        cell = CognitiveCell.create(
            embedding=embedding,
            radius=radius,
            confidence=confidence,
            origin=origin,
            parent_ids=parent_ids,
            generation=generation,
            domain=domain,
        )
        cells[cell.id] = cell
        logger.debug("Exist: created cell %s at radius=%.3f", cell.id, radius)
        return cell

    # ─────────────────────────────────────────────────────────────
    # Primitive II: Recognize
    # ─────────────────────────────────────────────────────────────

    @staticmethod
    def recognize(
        cells: CellRegistry,
        intersections: IntersectionRegistry,
        cell: CognitiveCell,
        vitality_minimum: float = _DEFAULT_VITALITY_MINIMUM,
        max_neighbors: int | None = None,
        compared_pairs: set[frozenset[CellID]] | None = None,
    ) -> list[Intersection]:
        """Primitive II: Find all overlapping semantic fields.

        1. Query all cells within range (distance < cell.radius + other.radius)
        2. For each candidate, compute overlap_depth
        3. Filter to actual overlaps (depth > 0)
        4. Compute blended embedding and novelty for each overlap
        5. Register new intersections (skip if pair already recorded)
        6. Return list of Intersections (new + existing for this cell)

        When max_neighbors is given, only the k nearest active cells are
        considered — leaving distant pairs unexplored for consolidate() to find.
        When max_neighbors is None (default), all cells are compared (exhaustive).

        compared_pairs: the substrate's memory of which pairs have already been
        compared. If provided, newly compared pairs are added to this set.
        If None, a local set is built from the intersection registry (backward
        compatible but amnesic — the substrate forgets between calls).
        """
        if cell.energy < vitality_minimum:
            logger.debug(
                "Recognize: cell %s below vitality threshold (%.3f), skipping",
                cell.id,
                cell.energy,
            )
            return []

        # Use the substrate's memory of compared pairs, or build a local one
        if compared_pairs is None:
            compared_pairs = {
                frozenset({ix.parent_a_id, ix.parent_b_id})
                for ix in intersections.values()
            }

        # Build candidate_ids: top-k nearest or None (exhaustive)
        candidate_ids: set[CellID] | None = None
        if max_neighbors is not None and max_neighbors < len(cells) - 1:
            candidates_sorted = sorted(
                [
                    (c.id, cell.distance_to(c))
                    for cid, c in cells.items()
                    if cid != cell.id
                    and c.state in (CellState.ACTIVE, CellState.DISPUTED)
                    and c.energy >= vitality_minimum
                ],
                key=lambda t: t[1],
            )
            candidate_ids = {
                cid for cid, _ in candidates_sorted[:max_neighbors]
            }

        discovered: list[Intersection] = []

        for other_id, other in cells.items():
            if other_id == cell.id:
                continue
            if other.state not in (CellState.ACTIVE, CellState.DISPUTED):
                continue
            if other.energy < vitality_minimum:
                continue

            if candidate_ids is not None and other_id not in candidate_ids:
                continue

            pair = frozenset({cell.id, other_id})
            if pair in compared_pairs:
                continue

            if not cell.overlaps_with(other):
                continue

            ix = Intersection.compute(cell, other)
            intersections[ix.id] = ix
            discovered.append(ix)

            logger.debug(
                "Recognize: intersection %s "
                "(overlap=%.3f, novelty=%.3f, sig=%.3f)",
                ix.id,
                ix.overlap,
                ix.novelty,
                ix.significance,
            )

        return discovered

    # ─────────────────────────────────────────────────────────────
    # Primitive III: Move Toward the New
    # ─────────────────────────────────────────────────────────────

    @staticmethod
    def move_toward_new(
        cells: CellRegistry,
        intersections: IntersectionRegistry,
        discovered: list[Intersection],
        origin_context: Origin,
        promotion_threshold: float = _DEFAULT_PROMOTION_THRESHOLD,
        initial_radius: float = _DEFAULT_INITIAL_RADIUS,
        recursion_depth_limit: int = _DEFAULT_RECURSION_DEPTH_LIMIT,
        compared_pairs: set[frozenset[CellID]] | None = None,
        max_neighbors: int | None = 12,
        _depth: int = 0,
    ) -> list[CognitiveCell]:
        """Primitive III: Promote significant intersections and repeat.

        1. Sort intersections by significance (descending)
        2. For each intersection above promotion_threshold:
           a. Create new CognitiveCell from intersection embedding (exist)
           b. Call recognize() on the new cell
           c. Recursively process any new intersections (depth-limited)
        3. Return list of newly promoted cells
        """
        if _depth >= recursion_depth_limit:
            logger.debug("move_toward_new: recursion depth limit reached")
            return []

        # Sort by significance descending
        candidates = sorted(
            [
                ix
                for ix in discovered
                if ix.should_promote(promotion_threshold)
            ],
            key=lambda ix: ix.significance,
            reverse=True,
        )

        promoted_cells: list[CognitiveCell] = []

        for ix in candidates:
            parent_a = cells.get(ix.parent_a_id)
            parent_b = cells.get(ix.parent_b_id)
            if parent_a is None or parent_b is None:
                continue

            avg_confidence = (
                parent_a.confidence + parent_b.confidence
            ) / 2.0
            generation = (
                max(parent_a.generation, parent_b.generation) + 1
            )

            # Hash chain: child commits to parents' identity.
            # Tampering with either parent invalidates the child.
            hash_a = hashlib.sha256(
                parent_a.id.encode()
            ).hexdigest()
            hash_b = hashlib.sha256(
                parent_b.id.encode()
            ).hexdigest()
            parent_hash = hashlib.sha256(
                (hash_a + hash_b).encode()
            ).hexdigest()

            new_cell = Primitives.exist(
                cells=cells,
                embedding=ix.embedding,
                origin=Origin(
                    timestamp=origin_context.timestamp,
                    source=origin_context.source,
                    context=OriginContext.SYNTHESIS,
                    participant_id=origin_context.participant_id,
                ),
                radius=initial_radius,
                confidence=avg_confidence,
                parent_ids=[ix.parent_a_id, ix.parent_b_id],
                generation=generation,
            )
            new_cell.parent_hash = parent_hash
            # Promoted cells carry a blend of their parents' text
            pa_text = parent_a.text[:100] if parent_a.text else ""
            pb_text = parent_b.text[:100] if parent_b.text else ""
            if pa_text and pb_text:
                new_cell.text = f"[{pa_text}] + [{pb_text}]"
            ix.promoted = True
            promoted_cells.append(new_cell)

            new_intersections = Primitives.recognize(
                cells=cells,
                intersections=intersections,
                cell=new_cell,
                compared_pairs=compared_pairs,
                max_neighbors=max_neighbors,
            )

            if new_intersections:
                further_promoted = Primitives.move_toward_new(
                    cells=cells,
                    intersections=intersections,
                    discovered=new_intersections,
                    origin_context=origin_context,
                    promotion_threshold=promotion_threshold,
                    initial_radius=initial_radius,
                    recursion_depth_limit=recursion_depth_limit,
                    compared_pairs=compared_pairs,
                    max_neighbors=max_neighbors,
                    _depth=_depth + 1,
                )
                promoted_cells.extend(further_promoted)

        return promoted_cells
