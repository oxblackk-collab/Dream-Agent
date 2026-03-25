"""Metabolism — the energy model of the Mycelium substrate.

Energy governs cell participation in substrate operations.
It is the self-regulation mechanism: no external commands needed.

Energy Sources:
- Retrieval: cell accessed during conversation
- Intersection: cell participates in new intersection
- Consolidation: cell discovered in dream cycle

Energy Drains:
- Base decay: continuous tick-based loss (adaptive — stabilises with use)

Thresholds:
- Archive:   energy below this -> cell excluded from active retrieval
             but preserved and accessible to dream consolidation
- Vitality:  energy below this -> cell skipped in Recognize
- Mitosis:   energy above this (+ diversity) -> cell can divide

EF-005 — Cross-domain multiplier (paracrine vs autocrine signals):
- Cross-domain boost x1.0: full signal (paracrine — from different domain)
- Same-domain boost  x0.5: half signal (autocrine — self-reinforcing echo)
- Empty domain defaults to cross-domain multiplier (safe fallback)

EF-006 — Adaptive decay (long-term memory stabilisation):
- Decay rate decreases with access_count: the more a cell is reactivated,
  the slower it decays. Mirrors the Ebbinghaus/Wixted power-law of forgetting.
- Formula: effective_decay = base_decay / (1 + access_count * STABILIZATION_FACTOR)
- touch() is now called on intersection, consolidation and reactivation events
  (not only retrieval), so any form of activation stabilises the cell.
"""

from __future__ import annotations

from mycelium.core.cell import CellID, CellState, CognitiveCell


class Metabolism:
    """Energy constants and operations for the substrate.

    All values are configurable; defaults match the spec.
    """

    # Energy sources
    RETRIEVAL_BOOST: float = 0.08
    INTERSECTION_BOOST: float = 0.12
    CONSOLIDATION_BOOST: float = 0.03

    # EF-005: Domain multipliers
    # Paracrine signal (cross-domain) — full boost: cell is useful to the organism
    CROSS_DOMAIN_MULTIPLIER: float = 1.0
    # Autocrine signal (same-domain) — half boost: valid but self-reinforcing echo
    SAME_DOMAIN_MULTIPLIER: float = 0.5

    # Energy drains
    BASE_DECAY_RATE: float = 0.001
    # EF-006: each access_count unit reduces effective decay by this fraction.
    # effective_decay = base_decay / (1 + access_count * STABILIZATION_FACTOR)
    # At access_count=10 -> 40% of base decay. At 20 -> 25%. Never zero.
    STABILIZATION_FACTOR: float = 0.15

    # Thresholds
    # EF-005: ARCHIVE replaces QUARANTINE — cells are preserved, not destroyed.
    # They are excluded from active retrieval but available to dream consolidation.
    ARCHIVE_THRESHOLD: float = 0.10
    MITOSIS_THRESHOLD: float = 0.70
    VITALITY_MINIMUM: float = 0.30

    # Back-compat alias (tests and external code may reference quarantine_threshold)
    QUARANTINE_THRESHOLD: float = ARCHIVE_THRESHOLD

    def __init__(
        self,
        retrieval_boost: float = RETRIEVAL_BOOST,
        intersection_boost: float = INTERSECTION_BOOST,
        consolidation_boost: float = CONSOLIDATION_BOOST,
        base_decay_rate: float = BASE_DECAY_RATE,
        stabilization_factor: float = STABILIZATION_FACTOR,
        quarantine_threshold: float = ARCHIVE_THRESHOLD,
        mitosis_threshold: float = MITOSIS_THRESHOLD,
        vitality_minimum: float = VITALITY_MINIMUM,
        same_domain_multiplier: float = SAME_DOMAIN_MULTIPLIER,
        cross_domain_multiplier: float = CROSS_DOMAIN_MULTIPLIER,
    ) -> None:
        self.retrieval_boost = retrieval_boost
        self.intersection_boost = intersection_boost
        self.consolidation_boost = consolidation_boost
        self.base_decay_rate = base_decay_rate
        self.stabilization_factor = stabilization_factor
        self.archive_threshold = quarantine_threshold  # canonical name
        self.mitosis_threshold = mitosis_threshold
        self.vitality_minimum = vitality_minimum
        self.same_domain_multiplier = same_domain_multiplier
        self.cross_domain_multiplier = cross_domain_multiplier

    @property
    def quarantine_threshold(self) -> float:
        """Back-compat alias for archive_threshold."""
        return self.archive_threshold

    # ──────────────────────────────────────────────
    # Domain multiplier helper
    # ──────────────────────────────────────────────

    def _domain_multiplier(self, cell_domain: str, other_domain: str) -> float:
        """Return boost multiplier based on domain relationship.

        Cross-domain (paracrine): full signal — cell is useful to the organism.
        Same-domain  (autocrine): half signal — valid but self-reinforcing echo.
        Empty domain: defaults to cross-domain (safe, no penalisation).
        """
        if cell_domain and other_domain and cell_domain == other_domain:
            return self.same_domain_multiplier
        return self.cross_domain_multiplier

    # ──────────────────────────────────────────────
    # Decay
    # ──────────────────────────────────────────────

    def _effective_decay(self, cell: CognitiveCell) -> float:
        """EF-006: adaptive decay rate — stabilises with accumulated access.

        Mirrors the Ebbinghaus/Wixted power-law of forgetting: the more a
        memory has been reactivated, the slower it fades.

        effective_decay = base_decay / (1 + access_count * stabilization_factor)
        """
        return self.base_decay_rate / (
            1.0 + cell.access_count * self.stabilization_factor
        )

    def apply_tick(self, cells: dict[CellID, CognitiveCell]) -> list[CellID]:
        """Apply one tick of energy decay to all active cells.

        EF-006: decay is adaptive — cells stabilise as their access_count grows.
        Returns the IDs of cells that transitioned to ARCHIVED this tick.
        """
        archived: list[CellID] = []
        for cell in cells.values():
            if cell.state not in (CellState.ACTIVE, CellState.DISPUTED):
                continue
            cell.decay_energy(self._effective_decay(cell))
            if cell.energy <= self.archive_threshold:
                cell.state = CellState.ARCHIVED
                archived.append(cell.id)
        return archived

    def apply_bulk_decay(
        self,
        cells: dict[CellID, CognitiveCell],
        n_ticks: int,
    ) -> list[CellID]:
        """Apply n_ticks of adaptive decay in one shot.

        Each cell decays at its own rate — frequently accessed cells
        lose less energy than rarely accessed ones, even over long gaps.
        This is the bulk equivalent of calling apply_tick() n times,
        but in O(cells) instead of O(cells * n_ticks).
        """
        archived: list[CellID] = []
        for cell in cells.values():
            if cell.state not in (CellState.ACTIVE, CellState.DISPUTED):
                continue
            total_decay = self._effective_decay(cell) * n_ticks
            cell.energy = max(0.0, cell.energy - total_decay)
            if cell.energy <= self.archive_threshold:
                cell.state = CellState.ARCHIVED
                archived.append(cell.id)
        return archived

    # ──────────────────────────────────────────────
    # Energy boosts — EF-005: domain-aware
    # ──────────────────────────────────────────────

    def on_retrieval(
        self,
        cell: CognitiveCell,
        cell_domain: str = "",
        source_domain: str = "",
        context_gate: float = 1.0,
    ) -> None:
        """Boost energy when a cell is accessed as a top-k neighbor.

        Cross-domain access (paracrine signal) -> full boost.
        Same-domain access (autocrine signal)  -> half boost.
        Context gate: off-context input dampens the boost — the
        substrate doesn't nourish what doesn't fit the trajectory.
        """
        multiplier = self._domain_multiplier(cell_domain, source_domain)
        cell.boost_energy(
            self.retrieval_boost * multiplier * context_gate
        )
        cell.touch()

    def on_intersection(
        self,
        cell: CognitiveCell,
        cell_domain: str = "",
        partner_domain: str = "",
        context_gate: float = 1.0,
    ) -> None:
        """Boost energy when a cell participates in a new intersection.

        Cross-domain intersection -> full boost.
        Same-domain intersection  -> half boost.
        Context gate: attentional modulation — only contextually
        coherent interactions fully potentiate.
        EF-006: touch() increments access_count.
        """
        multiplier = self._domain_multiplier(cell_domain, partner_domain)
        cell.boost_energy(
            self.intersection_boost * multiplier * context_gate
        )
        cell.touch()

    def on_consolidation(self, cell: CognitiveCell) -> None:
        """Boost energy when a cell is discovered in a dream cycle.

        EF-006: touch() increments access_count — dream discovery is an activation.
        """
        cell.boost_energy(self.consolidation_boost)
        cell.touch()

    # ──────────────────────────────────────────────
    # Archive / reactivation (EF-005)
    # ──────────────────────────────────────────────

    def should_archive(self, cell: CognitiveCell) -> bool:
        """True if cell energy dropped to archive threshold."""
        return cell.energy <= self.archive_threshold and cell.state in (
            CellState.ACTIVE,
            CellState.DISPUTED,
        )

    def reactivate(self, cell: CognitiveCell) -> None:
        """Reactivate an archived cell rescued by dream consolidation.

        The cell was preserved in the substrate and the dream cycle
        found a meaningful connection — it is relevant again.
        EF-006: touch() on reactivation — resurrection is the strongest activation.
        """
        if cell.state == CellState.ARCHIVED:
            cell.state = CellState.ACTIVE
            # Double consolidation boost on reactivation — the dream rescued it
            cell.boost_energy(self.consolidation_boost * 2)
            cell.touch()

    # ──────────────────────────────────────────────
    # Mitosis / vitality checks
    # ──────────────────────────────────────────────

    def can_undergo_mitosis(
        self, cell: CognitiveCell, min_intersections: int = 5
    ) -> bool:
        """True if cell has enough energy and diversity for mitosis."""
        return (
            cell.can_divide(min_intersections=min_intersections)
            and cell.energy >= self.mitosis_threshold
        )

    def is_vital(self, cell: CognitiveCell) -> bool:
        """True if cell has enough energy to participate in Recognize."""
        return cell.energy >= self.vitality_minimum
