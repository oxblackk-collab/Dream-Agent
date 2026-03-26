"""Intersection — emergent knowledge between two overlapping Cognitive Cells.

When two Cognitive Cells have overlapping semantic fields, the intersection
is a new emergent region that belongs to neither cell exclusively.
This is the Vesica Piscis of the Mycelium: the space where new understanding is born.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, NewType

import numpy as np

if TYPE_CHECKING:
    from mycelium.core.cell import CellID, CognitiveCell, EmbeddingVector

IntersectionID = NewType("IntersectionID", str)

# Promotion weights (novelty-biased per spec)
# EF-006: coherence factor added — genuine connections have aligned embeddings.
# Weights must sum to 1.0.
_WEIGHT_OVERLAP = 0.25
_WEIGHT_NOVELTY = 0.40
_WEIGHT_CONFIDENCE = 0.15
_WEIGHT_COHERENCE = 0.20


@dataclass
class Intersection:
    """The emergent region between two overlapping CognitiveCells.

    An Intersection is not merely a connection — it is a space.
    A region of meaning where both cells contribute but neither fully owns.
    """

    id: IntersectionID
    parent_a_id: CellID
    parent_b_id: CellID
    embedding: EmbeddingVector  # blended semantic position
    overlap: float  # degree of field overlap [0, 1]
    novelty: float  # cross-domain surprise [0, 1]
    significance: float  # computed: determines promotion eligibility
    discovered_at: datetime
    promoted: bool  # has this become a Cell?

    @classmethod
    def compute(
        cls,
        cell_a: CognitiveCell,
        cell_b: CognitiveCell,
    ) -> Intersection:
        """Compute a new Intersection from two overlapping cells."""
        overlap = cls._compute_overlap(cell_a, cell_b)
        novelty = cls._compute_novelty(cell_a, cell_b)
        coherence = cls._compute_coherence(cell_a, cell_b)
        blended = cls._compute_blended_embedding(cell_a, cell_b, overlap)
        avg_confidence = (cell_a.confidence + cell_b.confidence) / 2.0
        significance = cls._compute_significance(
            overlap, novelty, avg_confidence, coherence
        )

        return cls(
            id=IntersectionID(str(uuid.uuid4())),
            parent_a_id=cell_a.id,
            parent_b_id=cell_b.id,
            embedding=blended,
            overlap=overlap,
            novelty=novelty,
            significance=significance,
            discovered_at=datetime.now(tz=UTC),
            promoted=False,
        )

    @staticmethod
    def _compute_overlap(cell_a: CognitiveCell, cell_b: CognitiveCell) -> float:
        """Overlap depth between the two parent cells."""
        return cell_a.overlap_depth(cell_b)

    @staticmethod
    def _compute_novelty(cell_a: CognitiveCell, cell_b: CognitiveCell) -> float:
        """Cross-domain surprise: higher when cells are semantically distant.

        We invert distance (normalized by combined radii) to get novelty.
        Cells from very different domains overlapping = high novelty.
        """
        dist = cell_a.distance_to(cell_b)
        combined = cell_a.radius + cell_b.radius
        if combined == 0:
            return 0.0
        # Novelty increases with distance (up to a cap)
        raw = dist / (combined + dist)
        return float(min(raw, 1.0))

    @staticmethod
    def _compute_coherence(
        cell_a: CognitiveCell,
        cell_b: CognitiveCell,
    ) -> float:
        """EF-006: Semantic coherence of the intersection.

        Measures how well the two embeddings align when blended — the magnitude
        of the raw (pre-normalisation) weighted average. Two coherent, related
        embeddings produce a high-magnitude blend; a coherent cell crossed with
        an incoherent "noise" cell (pointing in many directions at once) produces
        a blend that partially cancels, yielding lower coherence.

        Returns a value in [0, 1].
        """
        total_conf = cell_a.confidence + cell_b.confidence
        weight_a = 0.5 if total_conf == 0 else cell_a.confidence / total_conf
        weight_b = 1.0 - weight_a
        raw = weight_a * cell_a.embedding + weight_b * cell_b.embedding
        norm = float(np.linalg.norm(raw))
        return norm if norm < 1.0 else 1.0

    @staticmethod
    def _compute_blended_embedding(
        cell_a: CognitiveCell,
        cell_b: CognitiveCell,
        overlap: float,
    ) -> EmbeddingVector:
        """Blended semantic position — weighted average biased toward overlap region.

        The blended embedding is not simply the midpoint: it's weighted by
        each cell's confidence, placing the intersection closer to the
        more confident cell's position.
        """
        total_conf = cell_a.confidence + cell_b.confidence
        weight_a = 0.5 if total_conf == 0 else cell_a.confidence / total_conf
        weight_b = 1.0 - weight_a

        blended = weight_a * cell_a.embedding + weight_b * cell_b.embedding
        norm = float(np.linalg.norm(blended))
        if norm > 0:
            blended = blended / norm
        return blended.astype(np.float32)

    @staticmethod
    def _compute_significance(
        overlap: float,
        novelty: float,
        avg_confidence: float,
        coherence: float,
    ) -> float:
        """Significance formula with depth-gated novelty.

        Novelty (cross-domain surprise) is only valuable when backed by
        real overlap depth. A superficial contact between distant cells
        is noise, not discovery. Deep overlap + high novelty = genuine
        insight. This mirrors Hebbian learning: both neurons must fire.

        The novelty contribution is scaled by overlap depth:
        at overlap >= 0.2, novelty contributes fully.
        Below that, it's proportionally dampened.

        Weights (overlap, gated_novelty, confidence, coherence) sum to 1.0.
        """
        # Gate novelty by overlap depth — superficial contacts
        # don't get novelty credit.
        # Floor at 0.10: below this, the overlap is negligible and
        # novelty is dampened proportionally. Above 0.10, full novelty.
        overlap_gate = min(1.0, overlap / 0.10)
        gated_novelty = novelty * overlap_gate

        return (
            overlap * _WEIGHT_OVERLAP
            + gated_novelty * _WEIGHT_NOVELTY
            + avg_confidence * _WEIGHT_CONFIDENCE
            + coherence * _WEIGHT_COHERENCE
        )

    def should_promote(self, threshold: float = 0.6) -> bool:
        """True if this intersection should become a full CognitiveCell."""
        return self.significance >= threshold and not self.promoted
