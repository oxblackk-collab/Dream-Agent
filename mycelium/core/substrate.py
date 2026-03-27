"""Substrate — the cognitive substrate that orchestrates the Three Primitives.

The Substrate is the only stateful object in the core.
It holds all CognitiveCells, all Intersections, and the simulation clock.
Everything else (Primitives, Embedder, Metabolism) is stateless and injected.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING

import numpy as np

from mycelium.core.cell import (
    CellID,
    CellState,
    CognitiveCell,
    EmbeddingVector,
    Origin,
    OriginContext,
)
from mycelium.core.intersection import Intersection, IntersectionID
from mycelium.core.primitives import Primitives
from mycelium.energy.metabolism import Metabolism

if TYPE_CHECKING:
    from mycelium.embedding.embedder import BaseEmbedder
    from mycelium.identity.keys import ParticipantIdentity

logger = logging.getLogger(__name__)

# Substrate config defaults (energy constants come from Metabolism)
_VITALITY_MINIMUM = Metabolism.VITALITY_MINIMUM
_ARCHIVE_THRESHOLD = Metabolism.ARCHIVE_THRESHOLD
_QUARANTINE_THRESHOLD = _ARCHIVE_THRESHOLD  # back-compat alias

# Substrate config defaults
_DEFAULT_INITIAL_RADIUS = 0.15
_DEFAULT_PROMOTION_THRESHOLD = 0.55
_DEFAULT_RECURSION_DEPTH = 1
_DEFAULT_RADIUS_GROWTH_RATE = 0.01
# EF-008: Wake-filter — dream connections below this significance are registered
# but produce no energy boost. Modelled on human dreaming: the prefrontal
# cortex discards incoherent connections on waking, only genuine insights
# (high overlap + novelty + coherence) survive and energise the substrate.
# At 0.50 ≈ median significance → top 50% of dream connections boost energy.
_DEFAULT_DREAM_SIGNIFICANCE_THRESHOLD = 0.50


@dataclass
class DreamEntry:
    """A discovery from oneiric consolidation."""

    intersection_id: IntersectionID
    discovered_at: datetime
    description: str = ""


@dataclass
class SubstrateSnapshot:
    """Immutable snapshot of substrate state for visualization and storage."""

    tick_count: int
    cells: dict[CellID, CognitiveCell]
    intersections: dict[IntersectionID, Intersection]
    dream_log: list[DreamEntry]
    snapshot_at: datetime = field(default_factory=lambda: datetime.now(tz=UTC))


class Substrate:
    """The cognitive substrate. Orchestrates the Three Primitives.

    Thread safety: the substrate uses a lock for all mutations.
    The Dreamer (consolidation) accesses it from a background thread.
    """

    def __init__(
        self,
        embedder: BaseEmbedder,
        initial_radius: float = _DEFAULT_INITIAL_RADIUS,
        promotion_threshold: float = _DEFAULT_PROMOTION_THRESHOLD,
        recursion_depth_limit: int = _DEFAULT_RECURSION_DEPTH,
        radius_growth_rate: float = _DEFAULT_RADIUS_GROWTH_RATE,
        dream_significance_threshold: float = _DEFAULT_DREAM_SIGNIFICANCE_THRESHOLD,
    ) -> None:
        self._embedder = embedder
        self._initial_radius = initial_radius
        self._promotion_threshold = promotion_threshold
        self._recursion_depth_limit = recursion_depth_limit
        self._radius_growth_rate = radius_growth_rate
        self._dream_significance_threshold = dream_significance_threshold
        self._metabolism = Metabolism()
        self._ingest_neighbors: int = 12  # k for selective recognize during ingest

        self._cells: dict[CellID, CognitiveCell] = {}
        self._intersections: dict[IntersectionID, Intersection] = {}
        self._compared_pairs: set[frozenset[CellID]] = set()
        self._dream_log: list[DreamEntry] = []
        self._tick_count: int = 0
        self._lock = threading.Lock()

        # Dirty tracking — which cells/intersections changed since last save.
        # Enables incremental saves: only persist what's new or modified.
        self._dirty_cells: set[CellID] = set()
        self._dirty_intersections: set[IntersectionID] = set()
        self._dirty_dream_entries: list[DreamEntry] = []

        # Attentional context — where the substrate is currently focused.
        # EMA of recent cell embeddings. Implements Primitive III's
        # "direction": the substrate has a trajectory through semantic
        # space. Used for context gating of energy boosts — off-context
        # input doesn't nourish the tissue as strongly.
        self._context_embedding: EmbeddingVector | None = None
        self._context_alpha: float = 0.3  # ~last 3-4 cells dominate

    # ──────────────────────────────────────────────
    # Read-only properties
    # ──────────────────────────────────────────────

    @property
    def tick_count(self) -> int:
        return self._tick_count

    @property
    def cell_count(self) -> int:
        return len(self._cells)

    @property
    def intersection_count(self) -> int:
        return len(self._intersections)

    # ──────────────────────────────────────────────
    # Attentional context
    # ──────────────────────────────────────────────

    def _compute_context_gate(self, embedding: EmbeddingVector) -> float:
        """How aligned is this embedding with the substrate's trajectory?

        Returns a gate value in [0.1, 1.0]:
        - 1.0 = perfectly aligned with current context
        - 0.1 = orthogonal or opposed (floor prevents total suppression)

        The first cell always gets full gate (no context yet).
        """
        if self._context_embedding is None:
            return 1.0
        # Cosine similarity (embeddings are unit vectors)
        similarity = float(np.dot(embedding, self._context_embedding))
        # Map from [-1, 1] to [0, 1], then apply floor of 0.1
        gate = max(0.1, (similarity + 1.0) / 2.0)
        return gate

    def _update_context(self, embedding: EmbeddingVector) -> None:
        """Move the substrate's attention toward this embedding.

        Exponential moving average: recent cells dominate but older
        context persists. This IS Primitive III — the substrate moves
        toward the new, carrying memory of where it was.
        """
        if self._context_embedding is None:
            self._context_embedding = embedding.copy()
            return
        alpha = self._context_alpha
        self._context_embedding = (
            alpha * embedding + (1 - alpha) * self._context_embedding
        )
        # Re-normalize to unit vector
        norm = float(np.linalg.norm(self._context_embedding))
        if norm > 0:
            self._context_embedding = self._context_embedding / norm

    # ──────────────────────────────────────────────
    # Core API
    # ──────────────────────────────────────────────

    def _run_primitives(
        self,
        cell: CognitiveCell,
        embedding: EmbeddingVector,
        identity: ParticipantIdentity | None,
        context_gate: float,
    ) -> tuple[list[Intersection], list[CognitiveCell], set[CellID]]:
        """Execute the Three Primitives on a newly created cell.

        Handles signing, recognize, intersection boosts, context update,
        and move-toward-new. Must be called under self._lock.

        Returns (intersections, promoted_cells, accessed_neighbor_ids).
        """
        # Sign the cell if identity is provided
        if identity is not None:
            from mycelium.provenance.hasher import compute_cell_hash

            cell_hash = compute_cell_hash(cell)
            cell.origin.signature = identity.sign(cell_hash.encode("utf-8"))

        # II: Recognize (selective)
        intersections = Primitives.recognize(
            cells=self._cells,
            intersections=self._intersections,
            cell=cell,
            vitality_minimum=_VITALITY_MINIMUM,
            max_neighbors=self._ingest_neighbors,
            compared_pairs=self._compared_pairs,
        )

        accessed = self._apply_intersection_boosts(cell, intersections, context_gate)
        self._update_context(embedding)

        # III: Move Toward the New
        promoted = Primitives.move_toward_new(
            cells=self._cells,
            intersections=self._intersections,
            discovered=intersections,
            origin_context=cell.origin,
            promotion_threshold=self._promotion_threshold,
            initial_radius=self._initial_radius,
            recursion_depth_limit=self._recursion_depth_limit,
            compared_pairs=self._compared_pairs,
            max_neighbors=self._ingest_neighbors,
        )
        return intersections, promoted, accessed

    @staticmethod
    def _build_origin(
        source: str,
        participant_id: str,
        identity: ParticipantIdentity | None,
    ) -> Origin:
        """Create an Origin for an ingested cell."""
        pid = identity.participant_id if identity is not None else participant_id
        pub_key = identity.public_key_bytes if identity is not None else None
        return Origin(
            timestamp=datetime.now(tz=UTC),
            source=source,
            context=OriginContext.INTERACTION,
            participant_id=pid,
            public_key=pub_key,
        )

    def _mark_ingest_dirty(
        self,
        new_cells: list[CognitiveCell],
        accessed_neighbors: set[CellID],
        intersections: list[Intersection],
    ) -> None:
        """Record which cells/intersections changed during ingest.

        Enables incremental saves: only persist what's new or modified.
        Must be called under self._lock.
        """
        for c in new_cells:
            self._dirty_cells.add(c.id)
        for cid in accessed_neighbors:
            self._dirty_cells.add(cid)
        for ix in intersections:
            self._dirty_intersections.add(ix.id)
            self._dirty_cells.add(ix.parent_a_id)
            self._dirty_cells.add(ix.parent_b_id)

    def _apply_intersection_boosts(
        self,
        cell: CognitiveCell,
        intersections: list[Intersection],
        context_gate: float,
    ) -> set[CellID]:
        """Apply retrieval and intersection energy boosts, return accessed neighbor IDs.

        Must be called under self._lock.
        """
        accessed: set[CellID] = set()
        for ix in intersections:
            if ix.parent_a_id != cell.id:
                accessed.add(ix.parent_a_id)
            if ix.parent_b_id != cell.id:
                accessed.add(ix.parent_b_id)

        # Retrieval boost — gated by context alignment
        for cid in accessed:
            if cid in self._cells:
                neighbor = self._cells[cid]
                self._metabolism.on_retrieval(
                    neighbor,
                    cell_domain=neighbor.domain,
                    source_domain=cell.domain,
                    context_gate=context_gate,
                )

        # Intersection boost — also gated by context
        for ix in intersections:
            ca = self._cells.get(ix.parent_a_id)
            cb = self._cells.get(ix.parent_b_id)
            if ca:
                self._metabolism.on_intersection(
                    ca,
                    cell_domain=ca.domain,
                    partner_domain=cb.domain if cb else "",
                    context_gate=context_gate,
                )
            if cb:
                self._metabolism.on_intersection(
                    cb,
                    cell_domain=cb.domain,
                    partner_domain=ca.domain if ca else "",
                    context_gate=context_gate,
                )

        return accessed

    def ingest(
        self,
        text: str,
        source: str,
        participant_id: str = "",
        domain: str = "",
        identity: ParticipantIdentity | None = None,
    ) -> list[CognitiveCell]:
        """Process raw text into the substrate.

        If identity is provided, the cell's origin is signed with
        the participant's Ed25519 key — cryptographic proof of
        authorship. The participant_id is derived from the public key.
        """
        embedding = self._embedder.embed(text)
        origin = self._build_origin(source, participant_id, identity)

        with self._lock:
            context_gate = self._compute_context_gate(embedding)

            # I: Exist
            cell = Primitives.exist(
                cells=self._cells,
                embedding=embedding,
                origin=origin,
                radius=self._initial_radius,
                domain=domain,
            )
            cell.text = text

            # II & III: Recognize + Move Toward the New
            intersections, promoted, accessed = self._run_primitives(
                cell,
                embedding,
                identity,
                context_gate,
            )

            all_new = [cell, *promoted]
            self._mark_ingest_dirty(all_new, accessed, intersections)

            logger.info(
                "ingest: +%d cells, +%d intersections, +%d promoted",
                1,
                len(intersections),
                len(promoted),
            )
            return all_new

    def tick(self) -> None:
        """One time step of the substrate.

        1. Decay energy of all active cells
        2. EF-005: cells below archive threshold → ARCHIVED (not destroyed)
        3. Increment tick counter
        """
        with self._lock:
            to_archive: list[CellID] = []

            for cell in self._cells.values():
                if cell.state not in (CellState.ACTIVE, CellState.DISPUTED):
                    continue
                cell.decay_energy(self._metabolism.base_decay_rate)
                if cell.energy <= _ARCHIVE_THRESHOLD:
                    to_archive.append(cell.id)

            for cid in to_archive:
                self._cells[cid].state = CellState.ARCHIVED
                self._dirty_cells.add(cid)
                logger.debug("tick: cell %s archived (energy depleted)", cid)

            # Note: energy decay changes are tiny per tick. We only mark
            # archived cells as dirty here. Cells that merely decayed
            # will be saved on the next full snapshot or when they're
            # touched by ingest/consolidate. This keeps incremental
            # saves fast — otherwise every tick dirties the entire substrate.

            self._tick_count += 1

    def _find_consolidation_candidates(
        self,
        pairs_per_cycle: int,
    ) -> list[tuple[float, CognitiveCell, CognitiveCell]]:
        """Find candidate cell pairs for consolidation, sorted by distance.

        EF-005: Includes ARCHIVED cells — dreams can rescue them.
        EF-007: Ascending distance — closest uncompared pairs first.
        Must be called under self._lock.
        """
        candidate_cells = [
            c
            for c in self._cells.values()
            if c.state in (CellState.ACTIVE, CellState.ARCHIVED) and c.energy >= 0
        ]
        dream_cells = [
            c
            for c in candidate_cells
            if c.state == CellState.ARCHIVED or c.energy >= _VITALITY_MINIMUM
        ]

        if len(dream_cells) < 2:
            return []

        candidates: list[tuple[float, CognitiveCell, CognitiveCell]] = []
        for i, ca in enumerate(dream_cells):
            for cb in dream_cells[i + 1 :]:
                pair = frozenset({ca.id, cb.id})
                if pair in self._compared_pairs:
                    continue
                dist = ca.distance_to(cb)
                if dist >= (ca.radius + cb.radius):
                    continue
                candidates.append((dist, ca, cb))

        candidates.sort(key=lambda t: t[0])
        return candidates[:pairs_per_cycle]

    def _process_consolidation_discovery(
        self,
        ix: Intersection,
        ca: CognitiveCell,
        cb: CognitiveCell,
    ) -> None:
        """Handle a newly discovered intersection during consolidation.

        Logs the dream entry, handles archived cell rescue (EF-005),
        and applies energy boosts for significant connections (EF-008).
        Must be called under self._lock.
        """
        self._dirty_intersections.add(ix.id)

        text_a = ca.text or ca.origin.source
        text_b = cb.text or cb.origin.source
        desc_a = (text_a[:100] + "...") if len(text_a) > 100 else text_a
        desc_b = (text_b[:100] + "...") if len(text_b) > 100 else text_b
        domain_info = ""
        if ca.domain and cb.domain and ca.domain != cb.domain:
            domain_info = f" [{ca.domain} ↔ {cb.domain}]"
        dream_desc = f"{desc_a} ↔ {desc_b}{domain_info} (sig={ix.significance:.3f})"

        entry = DreamEntry(
            intersection_id=ix.id,
            discovered_at=datetime.now(tz=UTC),
            description=dream_desc,
        )
        self._dream_log.append(entry)
        self._dirty_dream_entries.append(entry)

        # EF-005: Handle archived cell rescue
        for archived, partner in [(ca, cb), (cb, ca)]:
            if archived.state != CellState.ARCHIVED:
                continue
            if (
                partner.confidence > archived.confidence * 1.2
                and ix.overlap > 0.6
                and partner.state == CellState.ACTIVE
            ):
                archived.state = CellState.SUPERSEDED
                self._dirty_cells.add(archived.id)
                logger.debug(
                    "consolidate: archived cell %s superseded by %s",
                    archived.id[:8],
                    partner.id[:8],
                )
            else:
                self._metabolism.reactivate(archived)
                self._dirty_cells.add(archived.id)
                logger.debug(
                    "consolidate: archived cell %s reactivated by dream",
                    archived.id[:8],
                )

        # EF-008: Wake-filter — only boost if significant enough
        if ix.significance >= self._dream_significance_threshold:
            self._metabolism.on_consolidation(ca)
            self._metabolism.on_consolidation(cb)
            self._dirty_cells.add(ca.id)
            self._dirty_cells.add(cb.id)

    def consolidate(self, pairs_per_cycle: int = 100) -> list[Intersection]:
        """Oneiric consolidation — discover latent connections.

        Systematically compare cell pairs that have never been compared,
        prioritizing overlapping pairs by ascending distance (closest first).
        Returns newly discovered intersections.
        """
        with self._lock:
            batch = self._find_consolidation_candidates(pairs_per_cycle)

            new_intersections: list[Intersection] = []
            for _, ca, cb in batch:
                pair = frozenset({ca.id, cb.id})
                self._compared_pairs.add(pair)
                if not ca.overlaps_with(cb):
                    continue
                ix = Intersection.compute(ca, cb)
                self._intersections[ix.id] = ix
                new_intersections.append(ix)

                self._process_consolidation_discovery(ix, ca, cb)

            logger.info(
                "consolidate: examined %d pairs, found %d intersections",
                len(batch),
                len(new_intersections),
            )
            return new_intersections

    def apply_bulk_decay(self, n_ticks: int) -> list[CellID]:
        """Apply n_ticks of adaptive energy decay in one shot.

        Each cell decays at its own rate based on access_count (EF-006).
        Frequently accessed cells lose less energy than rarely accessed ones,
        even across long temporal gaps — this is when it matters most.
        """
        with self._lock:
            archived = self._metabolism.apply_bulk_decay(self._cells, n_ticks)
            self._tick_count += n_ticks
            return archived

    def get_state_snapshot(self) -> SubstrateSnapshot:
        """Return immutable snapshot for visualization and storage."""
        with self._lock:
            return SubstrateSnapshot(
                tick_count=self._tick_count,
                cells=dict(self._cells),
                intersections=dict(self._intersections),
                dream_log=list(self._dream_log),
            )

    def get_incremental_snapshot(self) -> SubstrateSnapshot:
        """Return snapshot with only dirty (new/changed) cells and intersections.

        After calling this, dirty tracking is reset. Use this for fast
        incremental saves — only persist what changed since last save.
        """
        with self._lock:
            dirty_cells = {
                cid: self._cells[cid] for cid in self._dirty_cells if cid in self._cells
            }
            dirty_ix = {
                iid: self._intersections[iid]
                for iid in self._dirty_intersections
                if iid in self._intersections
            }
            dirty_dreams = list(self._dirty_dream_entries)

            # Reset dirty tracking
            self._dirty_cells.clear()
            self._dirty_intersections.clear()
            self._dirty_dream_entries.clear()

            return SubstrateSnapshot(
                tick_count=self._tick_count,
                cells=dirty_cells,
                intersections=dirty_ix,
                dream_log=dirty_dreams,
            )

    def get_bond(self, participant_id: str) -> list[CognitiveCell]:
        """Return cluster of cells associated with a participant.

        The cognitive bond is the emergent topological cluster formed
        through repeated interaction with a specific participant.
        """
        with self._lock:
            return [
                c
                for c in self._cells.values()
                if c.origin.participant_id == participant_id
                and c.state == CellState.ACTIVE
            ]

    # ──────────────────────────────────────────────
    # Query API
    # ──────────────────────────────────────────────

    def get_cell(self, cell_id: CellID) -> CognitiveCell | None:
        """Return a single cell by ID, or None if not found."""
        with self._lock:
            return self._cells.get(cell_id)

    def find_neighbors(
        self,
        embedding: EmbeddingVector,
        k: int = 10,
        active_only: bool = True,
    ) -> list[tuple[CognitiveCell, float]]:
        """Find k nearest cells to an embedding, with distances.

        Returns list of (cell, distance) sorted by distance ascending.
        Embeddings are unit vectors, so distance = euclidean ≈ cosine.
        """
        with self._lock:
            candidates = []
            for cell in self._cells.values():
                if active_only and cell.state != CellState.ACTIVE:
                    continue
                dist = float(np.linalg.norm(embedding - cell.embedding))
                candidates.append((cell, dist))

            candidates.sort(key=lambda t: t[1])
            return candidates[:k]

    def search_by_text(
        self, query: str, k: int = 10, active_only: bool = True
    ) -> list[tuple[CognitiveCell, float]]:
        """Embed query text, then find nearest cells.

        Convenience method: embed + find_neighbors in one call.
        """
        embedding = self._embedder.embed(query)
        return self.find_neighbors(embedding, k=k, active_only=active_only)

    def get_intersections_for(self, cell_id: CellID) -> list[Intersection]:
        """Return all intersections involving a given cell."""
        with self._lock:
            return [
                ix
                for ix in self._intersections.values()
                if ix.parent_a_id == cell_id or ix.parent_b_id == cell_id
            ]

    def find_cell_by_text_prefix(
        self,
        prefix: str,
        length: int = 100,
    ) -> CognitiveCell | None:
        """Find the first cell whose text starts with the given prefix.

        Compares up to `length` characters of each cell's text against
        the prefix (also truncated to `length`).
        Returns the cell or None if no match is found.
        """
        target = prefix[:length]
        with self._lock:
            for cell in self._cells.values():
                if cell.text and cell.text[:length] == target:
                    return cell
        return None

    def load_state(
        self,
        cells: dict[CellID, CognitiveCell],
        intersections: dict[str, Intersection],
    ) -> None:
        """Load pre-existing cells and intersections into the substrate."""
        with self._lock:
            self._cells.update(cells)
            self._intersections.update(intersections)

    def get_cross_domain_intersections(
        self,
        min_significance: float = 0.0,
    ) -> list[tuple[Intersection, CognitiveCell, CognitiveCell]]:
        """Return cross-domain intersections with their parent cells.

        Filters intersections by minimum significance and requires both
        parent cells to exist, have non-empty domains, and be from
        different domains.

        Returns list of (intersection, cell_a, cell_b) tuples sorted by
        significance descending.
        """
        results: list[tuple[Intersection, CognitiveCell, CognitiveCell]] = []
        with self._lock:
            for ix in self._intersections.values():
                if ix.significance < min_significance:
                    continue
                ca = self._cells.get(ix.parent_a_id)
                cb = self._cells.get(ix.parent_b_id)
                if not ca or not cb:
                    continue
                if not ca.domain or not cb.domain or ca.domain == cb.domain:
                    continue
                results.append((ix, ca, cb))
        results.sort(key=lambda t: t[0].significance, reverse=True)
        return results
