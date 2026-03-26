"""FastAPI router for the Mycelium substrate.

Exposes the substrate as a REST API. The router is a factory —
create_api_router(substrate) returns a router bound to a specific
substrate instance. This keeps the API stateless and testable.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from mycelium.core.substrate import Substrate


# ── Request/Response schemas ────────────────────────────


class IngestRequest(BaseModel):
    text: str
    source: str = ""
    participant_id: str = ""
    domain: str = ""


class ConsolidateRequest(BaseModel):
    pairs_per_cycle: int = Field(default=5000, ge=1, le=100000)


class CellResponse(BaseModel):
    id: str
    energy: float
    confidence: float
    radius: float
    state: str
    domain: str
    participant_id: str
    source: str
    origin_context: str
    generation: int
    access_count: int
    created_at: str
    text: str = ""

    model_config = {"from_attributes": True}


class IntersectionResponse(BaseModel):
    id: str
    parent_a_id: str
    parent_b_id: str
    overlap: float
    novelty: float
    significance: float
    promoted: bool
    discovered_at: str


class SearchResult(BaseModel):
    cell: CellResponse
    distance: float


class HealthResponse(BaseModel):
    status: str
    cells: int
    intersections: int
    ticks: int


class SnapshotResponse(BaseModel):
    tick_count: int
    cell_count: int
    intersection_count: int
    active_cells: int
    archived_cells: int
    promoted_intersections: int
    dream_log_entries: int


# ── Helpers ─────────────────────────────────────────────


def _cell_to_response(cell: Any) -> CellResponse:
    """Convert a CognitiveCell to API response."""
    return CellResponse(
        id=str(cell.id),
        energy=round(cell.energy, 4),
        confidence=round(cell.confidence, 4),
        radius=round(cell.radius, 4),
        state=cell.state.value,
        domain=cell.domain,
        participant_id=cell.origin.participant_id,
        source=cell.origin.source,
        origin_context=cell.origin.context.value,
        generation=cell.generation,
        access_count=cell.access_count,
        created_at=cell.created_at.isoformat(),
        text=getattr(cell, "text", ""),
    )


def _ix_to_response(ix: Any) -> IntersectionResponse:
    """Convert an Intersection to API response."""
    return IntersectionResponse(
        id=str(ix.id),
        parent_a_id=str(ix.parent_a_id),
        parent_b_id=str(ix.parent_b_id),
        overlap=round(ix.overlap, 4),
        novelty=round(ix.novelty, 4),
        significance=round(ix.significance, 4),
        promoted=ix.promoted,
        discovered_at=ix.discovered_at.isoformat(),
    )


# ── Router factory ──────────────────────────────────────


def create_api_router(
    substrate: Substrate,
    store: Any | None = None,
) -> APIRouter:
    """Create a FastAPI router bound to a substrate instance."""
    router = APIRouter()

    @router.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        return HealthResponse(
            status="ok",
            cells=substrate.cell_count,
            intersections=substrate.intersection_count,
            ticks=substrate.tick_count,
        )

    @router.post("/ingest", response_model=list[CellResponse])
    def ingest(req: IngestRequest) -> list[CellResponse]:
        cells = substrate.ingest(
            text=req.text,
            source=req.source,
            participant_id=req.participant_id,
            domain=req.domain,
        )
        substrate.tick()
        # Persist new cells to disk
        if store is not None:
            for cell in cells:
                store.save_cell(cell)
        return [_cell_to_response(c) for c in cells]

    @router.get("/cells", response_model=list[CellResponse])
    def list_cells(
        state: str | None = None,
        limit: int = 100,
    ) -> list[CellResponse]:
        from mycelium.core.cell import CellState

        snap = substrate.get_state_snapshot()
        cells = list(snap.cells.values())

        if state:
            try:
                filter_state = CellState(state)
            except ValueError as exc:
                valid = [s.value for s in CellState]
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid state. Valid: {valid}",
                ) from exc
            cells = [c for c in cells if c.state == filter_state]

        # Sort by energy descending
        cells.sort(key=lambda c: c.energy, reverse=True)
        return [_cell_to_response(c) for c in cells[:limit]]

    @router.get("/cells/{cell_id}", response_model=CellResponse)
    def get_cell(cell_id: str) -> CellResponse:
        from mycelium.core.cell import CellID

        cell = substrate.get_cell(CellID(cell_id))
        if cell is None:
            raise HTTPException(
                status_code=404, detail="Cell not found"
            )
        return _cell_to_response(cell)

    @router.get("/search", response_model=list[SearchResult])
    def search(q: str, k: int = 10) -> list[SearchResult]:
        if not q.strip():
            raise HTTPException(
                status_code=400, detail="Query cannot be empty"
            )
        results = substrate.search_by_text(q, k=k)
        return [
            SearchResult(
                cell=_cell_to_response(cell), distance=round(d, 4)
            )
            for cell, d in results
        ]

    @router.post(
        "/consolidate",
        response_model=list[IntersectionResponse],
    )
    def consolidate(
        req: ConsolidateRequest | None = None,
    ) -> list[IntersectionResponse]:
        pairs = req.pairs_per_cycle if req else 5000
        discoveries = substrate.consolidate(
            pairs_per_cycle=pairs
        )
        # Persist new intersections to disk
        if store is not None and discoveries:
            for ix in discoveries:
                store.save_intersection(ix)
        return [_ix_to_response(ix) for ix in discoveries]

    @router.get(
        "/snapshot", response_model=SnapshotResponse
    )
    def snapshot() -> SnapshotResponse:
        from mycelium.core.cell import CellState

        snap = substrate.get_state_snapshot()
        active = sum(
            1
            for c in snap.cells.values()
            if c.state == CellState.ACTIVE
        )
        archived = sum(
            1
            for c in snap.cells.values()
            if c.state == CellState.ARCHIVED
        )
        promoted = sum(
            1 for ix in snap.intersections.values() if ix.promoted
        )
        return SnapshotResponse(
            tick_count=snap.tick_count,
            cell_count=len(snap.cells),
            intersection_count=len(snap.intersections),
            active_cells=active,
            archived_cells=archived,
            promoted_intersections=promoted,
            dream_log_entries=len(snap.dream_log),
        )

    @router.get(
        "/bonds/{participant_id}",
        response_model=list[CellResponse],
    )
    def get_bond(participant_id: str) -> list[CellResponse]:
        bond = substrate.get_bond(participant_id)
        return [_cell_to_response(c) for c in bond]

    @router.get(
        "/cells/{cell_id}/intersections",
        response_model=list[IntersectionResponse],
    )
    def get_cell_intersections(
        cell_id: str,
    ) -> list[IntersectionResponse]:
        from mycelium.core.cell import CellID

        cell = substrate.get_cell(CellID(cell_id))
        if cell is None:
            raise HTTPException(
                status_code=404, detail="Cell not found"
            )
        ixs = substrate.get_intersections_for(CellID(cell_id))
        return [_ix_to_response(ix) for ix in ixs]

    return router
