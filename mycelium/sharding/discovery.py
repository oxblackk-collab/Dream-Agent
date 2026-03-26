"""Cross-Shard Dreamer — discovers connections between domain shards.

Like deep sleep vs REM: intra-shard consolidation is REM (frequent,
within-domain). Cross-shard discovery is deep sleep (infrequent,
between-domain). It samples representative cells from each shard
and looks for overlaps that bridge different domains.

This is how the substrate discovers that "mitochondrial ATP production"
(biology) connects to "energy market pricing" (economics) — connections
that no single shard would find on its own.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from mycelium.core.cell import CellState
from mycelium.core.intersection import Intersection

if TYPE_CHECKING:
    from mycelium.core.substrate import Substrate
    from mycelium.sharding.manager import SubstrateManager

logger = logging.getLogger(__name__)


class CrossShardDreamer:
    """Discovers latent connections between domain shards."""

    def discover(
        self,
        shard_a: Substrate,
        shard_b: Substrate,
        k: int = 50,
    ) -> list[Intersection]:
        """Sample k cells from each shard, find cross-shard overlaps.

        Only considers ACTIVE cells with highest energy — these are
        the most representative of each shard's knowledge.

        Returns intersections that bridge the two shards. These are
        NOT added to either shard — they exist in the bridge space.
        """
        cells_a = self._sample_representative(shard_a, k)
        cells_b = self._sample_representative(shard_b, k)

        if not cells_a or not cells_b:
            return []

        discoveries: list[Intersection] = []
        for ca in cells_a:
            for cb in cells_b:
                if not ca.overlaps_with(cb):
                    continue
                ix = Intersection.compute(ca, cb)
                discoveries.append(ix)

        discoveries.sort(
            key=lambda ix: ix.significance, reverse=True
        )

        logger.info(
            "Cross-shard dream: %dx%d cells, %d discoveries",
            len(cells_a),
            len(cells_b),
            len(discoveries),
        )
        return discoveries

    def _sample_representative(
        self, shard: Substrate, k: int
    ) -> list:
        """Sample top-k cells by energy from a shard."""
        snap = shard.get_state_snapshot()
        active = [
            c
            for c in snap.cells.values()
            if c.state == CellState.ACTIVE
        ]
        active.sort(key=lambda c: c.energy, reverse=True)
        return active[:k]


def cross_shard_consolidate(
    manager: SubstrateManager,
    k: int = 50,
) -> dict[str, list[Intersection]]:
    """Run cross-shard discovery across all shard pairs.

    Returns a dict keyed by "domain_a↔domain_b" with the
    discovered intersections for each pair.
    """
    dreamer = CrossShardDreamer()
    domains = manager.list_shards()
    results: dict[str, list[Intersection]] = {}

    for i, domain_a in enumerate(domains):
        shard_a = manager.get_shard(domain_a)
        if shard_a is None:
            continue
        for domain_b in domains[i + 1 :]:
            shard_b = manager.get_shard(domain_b)
            if shard_b is None:
                continue
            discoveries = dreamer.discover(shard_a, shard_b, k=k)
            if discoveries:
                key = f"{domain_a}↔{domain_b}"
                results[key] = discoveries

    return results
