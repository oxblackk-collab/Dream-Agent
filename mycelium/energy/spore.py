"""SPORE Meter — measures computational cost of substrate operations.

SPORE is the metabolic unit of the Mycelium substrate: the verified
cost of computation measured in watt-hours. Not a speculative token —
a physical receipt for energy consumed.

This module measures wall time, CPU time, and estimates watt-hours
for each substrate operation. The data feeds the SPORE economy:
"this ingest cost 0.00003 Wh, this consolidation cost 0.0012 Wh."

TDP (Thermal Design Power) is configurable:
  - 15W for laptop
  - 65W for desktop
  - 250W for server GPU

1 SPORE = N watt-hours of verified computation.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mycelium.core.cell import CognitiveCell
    from mycelium.core.intersection import Intersection
    from mycelium.core.substrate import Substrate


@dataclass
class SPORECost:
    """Cost of a single substrate operation."""

    operation: str
    wall_time_ms: float
    cpu_time_ms: float
    estimated_wh: float  # watt-hours based on TDP

    @property
    def estimated_spore(self) -> float:
        """SPORE units (= watt-hours, 1:1 peg)."""
        return self.estimated_wh


@dataclass
class SPOREReport:
    """Aggregate cost report."""

    costs: list[SPORECost] = field(default_factory=list)

    @property
    def total_wh(self) -> float:
        return sum(c.estimated_wh for c in self.costs)

    @property
    def total_wall_ms(self) -> float:
        return sum(c.wall_time_ms for c in self.costs)

    @property
    def total_cpu_ms(self) -> float:
        return sum(c.cpu_time_ms for c in self.costs)

    def summary(self) -> dict[str, float]:
        """Per-operation-type summary."""
        by_op: dict[str, list[SPORECost]] = {}
        for c in self.costs:
            by_op.setdefault(c.operation, []).append(c)
        return {
            op: sum(c.estimated_wh for c in costs)
            for op, costs in by_op.items()
        }


class SPOREMeter:
    """Measures computational cost of substrate operations.

    Wraps substrate calls with timing. Does not modify behavior —
    pure measurement.
    """

    def __init__(self, tdp_watts: float = 15.0) -> None:
        self._tdp = tdp_watts

    def _to_wh(self, cpu_seconds: float) -> float:
        """Convert CPU seconds to watt-hours at configured TDP."""
        return cpu_seconds * self._tdp / 3600.0

    def measure_ingest(
        self,
        substrate: Substrate,
        text: str,
        source: str = "",
        participant_id: str = "",
        domain: str = "",
    ) -> tuple[list[CognitiveCell], SPORECost]:
        """Ingest with cost measurement."""
        t_wall = time.perf_counter()
        t_cpu = time.process_time()

        cells = substrate.ingest(
            text=text,
            source=source,
            participant_id=participant_id,
            domain=domain,
        )
        substrate.tick()

        wall_ms = (time.perf_counter() - t_wall) * 1000
        cpu_ms = (time.process_time() - t_cpu) * 1000
        wh = self._to_wh(cpu_ms / 1000)

        cost = SPORECost(
            operation="ingest",
            wall_time_ms=round(wall_ms, 3),
            cpu_time_ms=round(cpu_ms, 3),
            estimated_wh=wh,
        )
        return cells, cost

    def measure_consolidate(
        self,
        substrate: Substrate,
        pairs_per_cycle: int = 5000,
    ) -> tuple[list[Intersection], SPORECost]:
        """Consolidation with cost measurement."""
        t_wall = time.perf_counter()
        t_cpu = time.process_time()

        discoveries = substrate.consolidate(
            pairs_per_cycle=pairs_per_cycle
        )

        wall_ms = (time.perf_counter() - t_wall) * 1000
        cpu_ms = (time.process_time() - t_cpu) * 1000
        wh = self._to_wh(cpu_ms / 1000)

        cost = SPORECost(
            operation="consolidate",
            wall_time_ms=round(wall_ms, 3),
            cpu_time_ms=round(cpu_ms, 3),
            estimated_wh=wh,
        )
        return discoveries, cost

    def measure_search(
        self,
        substrate: Substrate,
        query: str,
        k: int = 10,
    ) -> tuple[list, SPORECost]:
        """Search with cost measurement."""
        t_wall = time.perf_counter()
        t_cpu = time.process_time()

        results = substrate.search_by_text(query, k=k)

        wall_ms = (time.perf_counter() - t_wall) * 1000
        cpu_ms = (time.process_time() - t_cpu) * 1000
        wh = self._to_wh(cpu_ms / 1000)

        cost = SPORECost(
            operation="search",
            wall_time_ms=round(wall_ms, 3),
            cpu_time_ms=round(cpu_ms, 3),
            estimated_wh=wh,
        )
        return results, cost
