"""FastAPI router for the Dream agent.

Exposes dream log queries, lateral inspire, and wake filter endpoints.
Factory function create_dream_router(substrate, store) returns a router
bound to specific instances — stateless and testable.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from fastapi import APIRouter, HTTPException

logger = logging.getLogger(__name__)
from pydantic import BaseModel, Field

from mycelium.api.inspire import (
    InspireResult,
    LateralConnection,
    format_for_claude,
)
from mycelium.core.cell import CellID

if TYPE_CHECKING:
    from mycelium.core.substrate import Substrate
    from mycelium.storage.store import SubstrateStore


# -- Request/Response schemas ----------------------------------------


class InspireRequest(BaseModel):
    q: str
    k: int = Field(default=5, ge=1, le=50)
    max_lateral: int = Field(default=10, ge=1, le=50)


class LateralConnectionResponse(BaseModel):
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


class InspireResponse(BaseModel):
    query: str
    nearest_cells: list[dict]
    lateral_connections: list[LateralConnectionResponse]
    formatted: str
    total_cells: int
    total_intersections: int


class DreamEntryResponse(BaseModel):
    id: int
    intersection_id: str
    discovered_at: str
    description: str
    significance: float
    overlap: float
    novelty: float
    parent_a: str
    parent_b: str
    seen_at: str | None = None
    wake_verified: bool | None = None
    wake_reasoning: str = ""


class MarkSeenRequest(BaseModel):
    dream_ids: list[int] | None = None


class WakeRequest(BaseModel):
    query: str
    session_context: str = ""
    top_k: int = Field(default=5, ge=1, le=20)


class WakeConnectionResult(BaseModel):
    connection: LateralConnectionResponse
    verified: bool | None = None  # None = no backend available
    confidence: float
    reasoning: str


class WakeResponse(BaseModel):
    query: str
    session_context: str
    connections_evaluated: int
    connections_verified: int
    results: list[WakeConnectionResult]
    backend: str = "none"  # "cli", "api", or "none"


# -- Helpers ---------------------------------------------------------


def _inspire_on_substrate(
    substrate: Substrate,
    query: str,
    k: int = 5,
    max_lateral: int = 10,
) -> InspireResult:
    """Run inspire logic on an already-loaded substrate instance."""
    raw_results = substrate.search_by_text(
        query, k=k * 10, active_only=True,
    )

    # Prefer cells with domains (originals) for cross-domain search
    domain_results = [(c, d) for c, d in raw_results if c.domain]
    no_domain_results = [(c, d) for c, d in raw_results if not c.domain]

    results = domain_results[:k]
    if len(results) < k:
        results.extend(no_domain_results[: k - len(results)])

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
            other_id = (
                ix.parent_b_id
                if ix.parent_a_id == cell.id
                else ix.parent_a_id
            )
            other = substrate.get_cell(other_id)
            if other is None:
                continue

            if not cell.domain or not other.domain:
                continue
            if cell.domain == other.domain:
                continue

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
                connected_to_text=(
                    other.text[:300] if other.text else ""
                ),
                connected_to_domain=other.domain,
                connected_to_source=other.origin.source,
                intersection_significance=round(ix.significance, 4),
                intersection_overlap=round(ix.overlap, 4),
                intersection_novelty=round(ix.novelty, 4),
                cell_id=str(cell.id),
                connected_to_id=str(other.id),
            )
            all_laterals.append((ix.significance, lateral))

    all_laterals.sort(key=lambda t: -t[0])
    top_laterals = [lat for _, lat in all_laterals[:max_lateral]]

    return InspireResult(
        query=query,
        nearest_cells=nearest_cells,
        lateral_connections=top_laterals,
        total_cells_in_substrate=substrate.cell_count,
        total_intersections_in_substrate=substrate.intersection_count,
    )


def _dream_row_to_response(row: dict) -> DreamEntryResponse:
    """Convert a dream log row dict to a DreamEntryResponse."""
    return DreamEntryResponse(
        id=row["id"],
        intersection_id=str(row["intersection_id"]),
        discovered_at=str(row["discovered_at"]),
        description=row.get("description", ""),
        significance=round(row.get("significance", 0.0), 4),
        overlap=round(row.get("overlap", 0.0), 4),
        novelty=round(row.get("novelty", 0.0), 4),
        parent_a=str(row.get("parent_a", "")),
        parent_b=str(row.get("parent_b", "")),
        seen_at=str(row["seen_at"]) if row.get("seen_at") else None,
        wake_verified=bool(row["wake_verified"])
        if row.get("wake_verified") is not None
        else None,
        wake_reasoning=row.get("wake_reasoning", "") or "",
    )


def _lateral_to_response(
    lat: LateralConnection,
) -> LateralConnectionResponse:
    """Convert a LateralConnection dataclass to its response model."""
    return LateralConnectionResponse(
        cell_text=lat.cell_text,
        cell_domain=lat.cell_domain,
        cell_source=lat.cell_source,
        cell_confidence=lat.cell_confidence,
        cell_energy=lat.cell_energy,
        connected_to_text=lat.connected_to_text,
        connected_to_domain=lat.connected_to_domain,
        connected_to_source=lat.connected_to_source,
        intersection_significance=lat.intersection_significance,
        intersection_overlap=lat.intersection_overlap,
        intersection_novelty=lat.intersection_novelty,
    )


# -- Router factory --------------------------------------------------


def create_dream_router(
    substrate: Substrate,
    store: SubstrateStore,
) -> APIRouter:
    """Create a FastAPI router bound to substrate and store instances."""
    router = APIRouter()

    @router.get("/inspire", response_model=InspireResponse)
    def inspire(
        q: str,
        k: int = 5,
        max_lateral: int = 10,
    ) -> InspireResponse:
        if not q.strip():
            raise HTTPException(
                status_code=400, detail="Query cannot be empty"
            )
        result = _inspire_on_substrate(
            substrate, query=q, k=k, max_lateral=max_lateral,
        )
        formatted = format_for_claude(result)
        return InspireResponse(
            query=result.query,
            nearest_cells=result.nearest_cells,
            lateral_connections=[
                _lateral_to_response(lat)
                for lat in result.lateral_connections
            ],
            formatted=formatted,
            total_cells=result.total_cells_in_substrate,
            total_intersections=result.total_intersections_in_substrate,
        )

    @router.get(
        "/dream/unseen",
        response_model=list[DreamEntryResponse],
    )
    def get_unseen_dreams(
        min_significance: float = 0.55,
        limit: int = 50,
        verified_only: bool = True,
    ) -> list[DreamEntryResponse]:
        rows = store.get_unseen_dreams(
            min_significance=min_significance,
            limit=limit,
            verified_only=verified_only,
        )
        return [_dream_row_to_response(r) for r in rows]

    @router.post("/dream/mark-seen")
    def mark_dreams_seen(req: MarkSeenRequest) -> dict:
        count = store.mark_dreams_seen(dream_ids=req.dream_ids)
        return {"marked": count}

    @router.get(
        "/dream/log",
        response_model=list[DreamEntryResponse],
    )
    def get_dream_log(
        limit: int = 100,
        offset: int = 0,
        wake_verified: bool | None = None,
    ) -> list[DreamEntryResponse]:
        rows = store.get_dream_log(
            limit=limit, offset=offset, wake_verified=wake_verified,
        )
        return [_dream_row_to_response(r) for r in rows]

    @router.post("/dream/wake", response_model=WakeResponse)
    async def wake_filter(req: WakeRequest) -> WakeResponse:
        if not req.query.strip():
            raise HTTPException(status_code=400, detail="Query cannot be empty")

        # Get lateral connections via inspire logic
        result = _inspire_on_substrate(
            substrate, req.query, k=req.top_k, max_lateral=req.top_k,
        )
        connections = result.lateral_connections

        if not connections:
            return WakeResponse(
                query=req.query,
                session_context=req.session_context,
                connections_evaluated=0,
                connections_verified=0,
                results=[],
                backend="none",
            )

        # Evaluate via Claude (CLI -> API -> none) — async to avoid blocking
        from dream.wake import WakeFilter as WF

        wake = WF()
        wake_results, backend = await wake.evaluate_async(
            req.query, connections, req.session_context,
        )

        # Apply energy effects and record results
        BOOST_VERIFIED = 0.09   # 3x consolidation boost
        DECAY_REJECTED = 0.03   # mild penalty

        results: list[WakeConnectionResult] = []
        for i, conn in enumerate(connections):
            # Find matching wake result
            wr = next((r for r in wake_results if r.index == i), None)
            if wr is not None:
                verified = wr.verified
                confidence = wr.confidence
                reasoning = wr.reasoning
            elif backend == "none":
                verified = None
                confidence = 0.0
                reasoning = "no backend available"
            else:
                verified = False
                confidence = 0.0
                reasoning = "no evaluation returned"

            # Apply energy effects to parent cells (match by text prefix)
            if verified is not None:
                for cell in substrate._cells.values():
                    if (
                        cell.text
                        and conn.cell_text
                        and cell.text[:100] == conn.cell_text[:100]
                    ):
                        if verified:
                            cell.boost_energy(BOOST_VERIFIED)
                        else:
                            cell.energy = max(0.0, cell.energy - DECAY_REJECTED)
                        break

                for cell in substrate._cells.values():
                    if (
                        cell.text
                        and conn.connected_to_text
                        and cell.text[:100] == conn.connected_to_text[:100]
                    ):
                        if verified:
                            cell.boost_energy(BOOST_VERIFIED)
                        else:
                            cell.energy = max(0.0, cell.energy - DECAY_REJECTED)
                        break

            # Persist wake verification to dream_log
            if verified is not None and conn.cell_id and conn.connected_to_id:
                try:
                    with store._conn() as db:
                        # Find dream_log entry by matching parent cells
                        row = db.execute(
                            """SELECT dl.id FROM dream_log dl
                               JOIN intersections ix ON dl.intersection_id = ix.id
                               WHERE (ix.parent_a = ? AND ix.parent_b = ?)
                                  OR (ix.parent_a = ? AND ix.parent_b = ?)
                               ORDER BY dl.id DESC LIMIT 1""",
                            (conn.cell_id, conn.connected_to_id,
                             conn.connected_to_id, conn.cell_id),
                        ).fetchone()
                        if row:
                            store.set_wake_verification(
                                row["id"], verified, reasoning,
                            )
                except Exception as exc:
                    logger.warning("Failed to persist wake result: %s", exc)

            results.append(WakeConnectionResult(
                connection=_lateral_to_response(conn),
                verified=verified,
                confidence=confidence,
                reasoning=reasoning,
            ))

        verified_count = sum(1 for r in results if r.verified)

        return WakeResponse(
            query=req.query,
            session_context=req.session_context,
            connections_evaluated=len(results),
            connections_verified=verified_count,
            results=results,
            backend=backend,
        )

    return router
