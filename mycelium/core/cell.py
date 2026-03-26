"""Cognitive Cell — the fundamental unit of the Mycelium substrate.

A CognitiveCell is a point in semantic space that defines a region around itself.
This is Primitive I (Exist): by occupying a region of semantic space, the cell
asserts territory of meaning.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import Enum
from typing import NewType

import numpy as np

# Type aliases for semantic clarity
CellID = NewType("CellID", str)
EmbeddingVector = np.ndarray  # shape: (dimensions,), dtype: float32


class CellState(Enum):
    """Lifecycle states of a CognitiveCell."""

    ACTIVE = "active"
    SUPERSEDED = "superseded"
    DISPUTED = "disputed"
    CONTEXTUALIZED = "contextualized"
    ARCHIVED = "archived"
    QUARANTINE = "quarantine"


class OriginContext(Enum):
    """How a cell came into existence."""

    INTERACTION = "interaction"
    CORRECTION = "correction"
    SYNTHESIS = "synthesis"
    CONSOLIDATION = "consolidation"
    MITOSIS = "mitosis"


@dataclass
class Origin:
    """Provenance record — birth certificate of a CognitiveCell."""

    timestamp: datetime
    source: str  # provenance hash or source identifier
    context: OriginContext
    participant_id: str = ""  # which human participant triggered this
    signature: bytes | None = None  # Ed25519 signature of cell hash
    public_key: bytes | None = None  # signer's public key (32 bytes)


@dataclass
class CognitiveCell:
    """A point in semantic space that defines a region around itself.

    The cell's embedding is its position in meaning-space.
    The cell's radius is the breadth of meanings it encompasses.
    Together they define the semantic field — the membrane of the cell.
    """

    id: CellID
    embedding: EmbeddingVector  # position in semantic space
    radius: float  # semantic field reach (> 0)
    confidence: float  # self-assessed certainty [0, 1]
    energy: float  # metabolic vitality [0, 1]
    state: CellState
    origin: Origin
    created_at: datetime
    last_accessed: datetime
    access_count: int
    parent_ids: list[CellID]  # lineage (mitosis / promotion)
    generation: int  # divisions from original cell
    domain: str = ""  # EF-005: semantic domain for cross-domain multiplier
    parent_hash: str = ""  # hash chain: SHA-256(parent_a_hash + parent_b_hash)
    text: str = ""  # original text that created this cell

    @classmethod
    def create(
        cls,
        embedding: EmbeddingVector,
        radius: float,
        confidence: float,
        origin: Origin,
        parent_ids: list[CellID] | None = None,
        generation: int = 0,
        domain: str = "",
    ) -> CognitiveCell:
        """Factory: instantiate a new CognitiveCell at full energy."""
        now = datetime.now(tz=UTC)
        normalized = embedding.astype(np.float32)
        norm = float(np.linalg.norm(normalized))
        if norm > 0:
            normalized = normalized / norm
        return cls(
            id=CellID(str(uuid.uuid4())),
            embedding=normalized,
            radius=radius,
            confidence=confidence,
            energy=1.0,
            state=CellState.ACTIVE,
            origin=origin,
            created_at=now,
            last_accessed=now,
            access_count=0,
            parent_ids=parent_ids or [],
            generation=generation,
            domain=domain,
        )

    def distance_to(self, other: CognitiveCell) -> float:
        """Euclidean distance between embedding positions."""
        return float(np.linalg.norm(self.embedding - other.embedding))

    def overlaps_with(self, other: CognitiveCell) -> bool:
        """True if semantic fields overlap (circles intersect)."""
        return self.distance_to(other) < (self.radius + other.radius)

    def overlap_depth(self, other: CognitiveCell) -> float:
        """Degree of field overlap, normalized to [0, 1].

        Returns 0 if no overlap. Returns 1 if one cell is fully inside the other.
        """
        dist = self.distance_to(other)
        combined_radii = self.radius + other.radius
        if dist >= combined_radii:
            return 0.0
        # How much the circles penetrate each other
        penetration = combined_radii - dist
        return float(min(penetration / combined_radii, 1.0))

    def absorb(
        self, information_vector: EmbeddingVector, absorption_rate: float = 0.1
    ) -> None:
        """Metabolic ingestion: shift embedding toward new information.

        The cell moves in semantic space as it absorbs new understanding.
        """
        delta = information_vector.astype(np.float32) - self.embedding
        self.embedding = self.embedding + absorption_rate * delta
        # Normalize to keep unit vector properties
        norm = float(np.linalg.norm(self.embedding))
        if norm > 0:
            self.embedding = self.embedding / norm

    def can_divide(self, min_intersections: int = 5) -> bool:
        """True if this cell has enough energy and diversity to undergo mitosis."""
        return (
            self.energy >= 0.7  # MITOSIS_THRESHOLD from config
            and self.access_count >= min_intersections
            and self.state == CellState.ACTIVE
        )

    def decay_energy(self, rate: float = 0.001) -> None:
        """Tick-based energy loss. Clamps to [0, 1]."""
        self.energy = max(0.0, self.energy - rate)

    def boost_energy(self, amount: float) -> None:
        """Add energy (from retrieval, intersection, consolidation). Clamps to 1."""
        self.energy = min(1.0, self.energy + amount)

    def touch(self) -> None:
        """Record an access event."""
        self.last_accessed = datetime.now(tz=UTC)
        self.access_count += 1
