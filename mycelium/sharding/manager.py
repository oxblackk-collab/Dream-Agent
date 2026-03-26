"""Substrate Manager — domain-based cognitive sharding.

Maps to The Cognitive Chain's Cognitive Sharding: the substrate is
partitioned by semantic domain. Each shard is a full Substrate with
its own cells, intersections, and memory. Cross-shard connections
are discovered by the CrossShardDreamer (D2).

The manager routes ingestion to the appropriate shard based on domain.
Shards are created on demand — no pre-configuration needed.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from mycelium.core.substrate import Substrate

if TYPE_CHECKING:
    from mycelium.core.cell import CognitiveCell
    from mycelium.embedding.embedder import BaseEmbedder
    from mycelium.identity.keys import ParticipantIdentity


logger = logging.getLogger(__name__)


class SubstrateManager:
    """Manages multiple domain-sharded substrates."""

    def __init__(
        self,
        embedder: BaseEmbedder,
        initial_radius: float = 0.57,
        promotion_threshold: float = 0.55,
        recursion_depth_limit: int = 1,
        dream_significance_threshold: float = 0.51,
    ) -> None:
        self._embedder = embedder
        self._initial_radius = initial_radius
        self._promotion_threshold = promotion_threshold
        self._recursion_depth_limit = recursion_depth_limit
        self._dream_significance_threshold = (
            dream_significance_threshold
        )
        self._shards: dict[str, Substrate] = {}

    def _create_shard(self, domain: str) -> Substrate:
        """Create a new shard for a domain."""
        shard = Substrate(
            embedder=self._embedder,
            initial_radius=self._initial_radius,
            promotion_threshold=self._promotion_threshold,
            recursion_depth_limit=self._recursion_depth_limit,
            dream_significance_threshold=(
                self._dream_significance_threshold
            ),
        )
        logger.info("Created shard: %s", domain)
        return shard

    def get_or_create_shard(self, domain: str) -> Substrate:
        """Get existing shard or create a new one for the domain."""
        if domain not in self._shards:
            self._shards[domain] = self._create_shard(domain)
        return self._shards[domain]

    def get_shard(self, domain: str) -> Substrate | None:
        """Get a shard by domain, or None if it doesn't exist."""
        return self._shards.get(domain)

    def ingest(
        self,
        text: str,
        source: str = "",
        participant_id: str = "",
        domain: str = "",
        identity: ParticipantIdentity | None = None,
    ) -> list[CognitiveCell]:
        """Ingest into the appropriate domain shard.

        If domain is empty, routes to a "_default" shard.
        """
        shard_key = domain or "_default"
        shard = self.get_or_create_shard(shard_key)
        cells = shard.ingest(
            text=text,
            source=source,
            participant_id=participant_id,
            domain=domain,
            identity=identity,
        )
        shard.tick()
        return cells

    def list_shards(self) -> list[str]:
        """Return all active shard domain names."""
        return list(self._shards.keys())

    def get_shard_stats(self) -> dict[str, dict[str, int]]:
        """Return cell/intersection counts per shard."""
        return {
            domain: {
                "cells": shard.cell_count,
                "intersections": shard.intersection_count,
                "ticks": shard.tick_count,
            }
            for domain, shard in self._shards.items()
        }

    def consolidate_all(
        self, pairs_per_cycle: int = 5000
    ) -> dict[str, int]:
        """Run consolidation on every shard. Returns discoveries."""
        results: dict[str, int] = {}
        for domain, shard in self._shards.items():
            discoveries = shard.consolidate(
                pairs_per_cycle=pairs_per_cycle
            )
            results[domain] = len(discoveries)
        return results

    def get_bond(
        self, participant_id: str
    ) -> dict[str, list[CognitiveCell]]:
        """Get a participant's bond across all shards."""
        bonds: dict[str, list[CognitiveCell]] = {}
        for domain, shard in self._shards.items():
            bond = shard.get_bond(participant_id)
            if bond:
                bonds[domain] = bond
        return bonds

    def total_cells(self) -> int:
        """Total cells across all shards."""
        return sum(s.cell_count for s in self._shards.values())

    def total_intersections(self) -> int:
        """Total intersections across all shards."""
        return sum(
            s.intersection_count for s in self._shards.values()
        )
