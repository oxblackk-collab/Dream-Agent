"""FastAPI router for the Mycelium Dream agent.

Exposes dream log queries, lateral inspire, and wake filter endpoints.
Factory function create_dream_router(substrate, store) returns a router
bound to specific instances — stateless and testable.
"""

from __future__ import annotations

import logging
import sqlite3
from typing import TYPE_CHECKING

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from mycelium.api.inspire import (
    InspireResult,
    LateralConnection,
    _find_cross_domain_laterals,
    _results_to_dicts,
    _search_nearest,
    format_for_claude,
)

if TYPE_CHECKING:
    from mycelium.core.substrate import Substrate
    from mycelium.storage.store import SubstrateStore

logger = logging.getLogger(__name__)


# ── Request/Response schemas ────────────────────────────


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


class WakeSubmitItem(BaseModel):
    dream_log_id: int
    verified: bool
    confidence: float = 0.0
    reasoning: str = ""


class WakeSubmitRequest(BaseModel):
    results: list[WakeSubmitItem]


class WakePendingCell(BaseModel):
    id: str
    domain: str
    text: str


class WakePendingEntry(BaseModel):
    id: int
    intersection_id: str
    significance: float
    cell_a: WakePendingCell
    cell_b: WakePendingCell
    description: str


# ── Helpers ─────────────────────────────────────────────


def _inspire_on_substrate(
    substrate: Substrate,
    query: str,
    k: int = 5,
    max_lateral: int = 10,
) -> InspireResult:
    """Run inspire logic on an already-loaded substrate instance.

    Delegates to the canonical helpers in ``mycelium.api.inspire`` to
    avoid duplicating search/lateral logic (DRY).
    """
    results = _search_nearest(substrate, query, k)
    nearest_cells = _results_to_dicts(results)
    top_laterals = _find_cross_domain_laterals(substrate, results, max_lateral)

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


def _load_verified_domain_pairs(store: SubstrateStore) -> set[frozenset[str]]:
    """Load domain pairs that already have a verified wake insight."""
    try:
        return store.get_verified_domain_pairs()
    except (OSError, sqlite3.Error) as exc:
        logger.warning("Failed to load verified domain pairs: %s", exc)
        return set()


def _filter_unverified_connections(
    connections: list[LateralConnection],
    verified_pairs: set[frozenset[str]],
) -> list[LateralConnection]:
    """Remove connections whose domain pair is already verified."""
    return [
        c
        for c in connections
        if not c.cell_domain
        or not c.connected_to_domain
        or frozenset({c.cell_domain, c.connected_to_domain}) not in verified_pairs
    ]


def _resolve_wake_result(
    wake_results: list,
    index: int,
    backend: str,
) -> tuple[bool | None, float, str]:
    """Resolve verification status for a single connection."""
    wr = next((r for r in wake_results if r.index == index), None)
    if wr is not None:
        return wr.verified, wr.confidence, wr.reasoning
    if backend == "none":
        return None, 0.0, "no backend available"
    return False, 0.0, "no evaluation returned"


def _apply_energy_effect(
    substrate: Substrate,
    text_prefix: str,
    verified: bool,
    boost: float,
    decay: float,
) -> None:
    """Apply energy boost/decay to the substrate cell matching text_prefix."""
    if not text_prefix:
        return
    cell = substrate.find_cell_by_text_prefix(text_prefix)
    if cell is not None:
        if verified:
            cell.boost_energy(boost)
        else:
            cell.energy = max(0.0, cell.energy - decay)


def _persist_wake_verification(
    store: SubstrateStore,
    conn: LateralConnection,
    verified: bool,
    reasoning: str,
) -> None:
    """Persist wake verification result to dream_log."""
    if not conn.cell_id or not conn.connected_to_id:
        return
    try:
        dream_id = store.find_dream_log_by_parents(
            conn.cell_id,
            conn.connected_to_id,
        )
        if dream_id is not None:
            store.set_wake_verification(dream_id, verified, reasoning)
    except (OSError, sqlite3.Error) as exc:
        logger.warning("Failed to persist wake result: %s", exc)


def _build_wake_results(
    substrate: Substrate,
    store: SubstrateStore,
    connections: list[LateralConnection],
    wake_results: list,
    backend: str,
) -> list[WakeConnectionResult]:
    """Evaluate connections against wake results, apply energy effects, and persist."""
    BOOST_VERIFIED = 0.09
    DECAY_REJECTED = 0.03

    results: list[WakeConnectionResult] = []
    for i, conn in enumerate(connections):
        verified, confidence, reasoning = _resolve_wake_result(
            wake_results,
            i,
            backend,
        )

        if verified is not None:
            _apply_energy_effect(
                substrate, conn.cell_text, verified, BOOST_VERIFIED, DECAY_REJECTED
            )
            _apply_energy_effect(
                substrate,
                conn.connected_to_text,
                verified,
                BOOST_VERIFIED,
                DECAY_REJECTED,
            )
            _persist_wake_verification(store, conn, verified, reasoning)

        results.append(
            WakeConnectionResult(
                connection=_lateral_to_response(conn),
                verified=verified,
                confidence=confidence,
                reasoning=reasoning,
            )
        )
    return results


def _apply_wake_results(
    substrate: Substrate,
    store: SubstrateStore,
    items: list,
) -> dict[str, int]:
    """Apply wake verification results: persist, adjust energy, tally counts.

    Returns ``{"processed": N, "verified": N, "rejected": N}``.
    """
    BOOST_VERIFIED = 0.09
    DECAY_REJECTED = 0.03

    processed = 0
    verified_count = 0
    rejected_count = 0

    for item in items:
        try:
            store.set_wake_verification(
                item.dream_log_id,
                item.verified,
                item.reasoning,
            )
        except (OSError, sqlite3.Error) as exc:
            logger.warning(
                "Failed to persist wake result for %d: %s",
                item.dream_log_id,
                exc,
            )
            continue

        processed += 1
        if item.verified:
            verified_count += 1
        else:
            rejected_count += 1

        try:
            parents = store.get_parent_ids_for_dream(item.dream_log_id)
            if parents:
                for parent_id in parents:
                    cell = substrate.get_cell(parent_id)
                    if cell is None:
                        continue
                    if item.verified:
                        cell.boost_energy(BOOST_VERIFIED)
                    else:
                        cell.energy = max(0.0, cell.energy - DECAY_REJECTED)
        except (OSError, sqlite3.Error) as exc:
            logger.warning(
                "Failed to apply energy effects for %d: %s",
                item.dream_log_id,
                exc,
            )

    return {
        "processed": processed,
        "verified": verified_count,
        "rejected": rejected_count,
    }


def _collect_cross_domain_candidates(
    substrate: Substrate,
    min_significance: float,
    max_count: int,
) -> list[tuple[float, str, str, str, str, str, str, str]]:
    """Collect cross-domain intersection candidates from in-memory substrate."""
    cross_domain = substrate.get_cross_domain_intersections(
        min_significance=min_significance,
    )
    candidates: list[tuple[float, str, str, str, str, str, str, str]] = [
        (
            ix.significance,
            ix.id,
            ca.id,
            ca.domain,
            (ca.text or "")[:200],
            cb.id,
            cb.domain,
            (cb.text or "")[:200],
        )
        for ix, ca, cb in cross_domain
    ]
    return candidates[:max_count]


def _batch_lookup_unverified(
    store: SubstrateStore,
    ix_ids: list[str],
) -> dict[str, tuple[int, str]]:
    """Batch lookup unverified dream_log entries by intersection IDs."""
    return store.get_unverified_dream_entries_by_intersection(ix_ids)


def _deduplicate_pending_entries(
    rows: list[dict],
    verified_domain_pairs: set[frozenset[str]],
    limit: int,
) -> list[WakePendingEntry]:
    """Deduplicate rows by domain pair, skipping already-verified pairs."""
    entries: list[WakePendingEntry] = []
    seen_pairs: set[frozenset[str]] = set()
    for r in rows:
        pair = frozenset({r["domain_a"], r["domain_b"]})
        if pair in verified_domain_pairs or pair in seen_pairs:
            continue
        seen_pairs.add(pair)
        entries.append(
            WakePendingEntry(
                id=r["id"],
                intersection_id=str(r["intersection_id"]),
                significance=round(r["significance"], 4),
                cell_a=WakePendingCell(
                    id=str(r["parent_a"]),
                    domain=r["domain_a"],
                    text=(r["text_a"] or "")[:200],
                ),
                cell_b=WakePendingCell(
                    id=str(r["parent_b"]),
                    domain=r["domain_b"],
                    text=(r["text_b"] or "")[:200],
                ),
                description=r["description"] or "",
            )
        )
        if len(entries) >= limit:
            break
    return entries


# ── Router factory ──────────────────────────────────────


def create_dream_router(
    substrate: Substrate,
    store: SubstrateStore,
) -> APIRouter:
    """Factory — only contains @router endpoint handlers (closures over substrate/store).

    All business logic helpers are module-level functions that receive
    ``store`` and ``substrate`` as explicit parameters.
    Individual endpoints are kept under 50 lines each.
    """
    router = APIRouter()

    @router.get("/inspire", response_model=InspireResponse)
    # Sync acceptable — FastAPI runs sync endpoints in a threadpool automatically.
    # SQLite with WAL mode is local and non-blocking for reads.
    def inspire(
        q: str,
        k: int = 5,
        max_lateral: int = 10,
    ) -> InspireResponse:
        if not q.strip():
            raise HTTPException(status_code=400, detail="Query cannot be empty")
        result = _inspire_on_substrate(
            substrate,
            query=q,
            k=k,
            max_lateral=max_lateral,
        )
        formatted = format_for_claude(result)
        return InspireResponse(
            query=result.query,
            nearest_cells=result.nearest_cells,
            lateral_connections=[
                _lateral_to_response(lat) for lat in result.lateral_connections
            ],
            formatted=formatted,
            total_cells=result.total_cells_in_substrate,
            total_intersections=result.total_intersections_in_substrate,
        )

    @router.get(
        "/dream/unseen",
        response_model=list[DreamEntryResponse],
    )
    # Sync acceptable — FastAPI runs sync endpoints in a threadpool automatically.
    # SQLite with WAL mode is local and non-blocking for reads.
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
    # Sync acceptable — FastAPI runs sync endpoints in a threadpool automatically.
    # SQLite with WAL mode is local and non-blocking for reads/writes.
    def mark_dreams_seen(req: MarkSeenRequest) -> dict:
        count = store.mark_dreams_seen(dream_ids=req.dream_ids)
        return {"marked": count}

    @router.get(
        "/dream/log",
        response_model=list[DreamEntryResponse],
    )
    # Sync acceptable — FastAPI runs sync endpoints in a threadpool automatically.
    # SQLite with WAL mode is local and non-blocking for reads.
    def get_dream_log(
        limit: int = 100,
        offset: int = 0,
        wake_verified: bool | None = None,
    ) -> list[DreamEntryResponse]:
        rows = store.get_dream_log(
            limit=limit,
            offset=offset,
            wake_verified=wake_verified,
        )
        return [_dream_row_to_response(r) for r in rows]

    @router.post("/dream/wake", response_model=WakeResponse)
    # Sync — FastAPI runs in threadpool; SQLite/substrate ops are local
    def wake_filter(req: WakeRequest) -> WakeResponse:
        if not req.query.strip():
            raise HTTPException(status_code=400, detail="Query cannot be empty")

        result = _inspire_on_substrate(
            substrate,
            req.query,
            k=req.top_k,
            max_lateral=req.top_k,
        )

        verified_pairs = _load_verified_domain_pairs(store)
        connections = _filter_unverified_connections(
            result.lateral_connections,
            verified_pairs,
        )

        if not connections:
            return WakeResponse(
                query=req.query,
                session_context=req.session_context,
                connections_evaluated=0,
                connections_verified=0,
                results=[],
                backend="none",
            )

        from dream.wake import WakeFilter as WF

        wake = WF()
        wake_results, backend = wake.evaluate(
            req.query,
            connections,
            req.session_context,
        )

        results = _build_wake_results(
            substrate,
            store,
            connections,
            wake_results,
            backend,
        )
        verified_count = sum(1 for r in results if r.verified)

        return WakeResponse(
            query=req.query,
            session_context=req.session_context,
            connections_evaluated=len(results),
            connections_verified=verified_count,
            results=results,
            backend=backend,
        )

    @router.get("/dream/wake-pending", response_model=list[WakePendingEntry])
    # Sync — FastAPI runs in threadpool; SQLite/substrate ops are local
    def get_wake_pending(
        limit: int = Query(default=20, ge=1, le=100),
        min_significance: float = Query(default=0.55),
    ) -> list[WakePendingEntry]:
        verified_domain_pairs = _load_verified_domain_pairs(store)

        try:
            candidates = _collect_cross_domain_candidates(
                substrate,
                min_significance,
                limit * 20,
            )
            ix_ids = [c[1] for c in candidates]
            dl_map = _batch_lookup_unverified(store, ix_ids)

            rows = []
            for sig, ix_id, ca_id, da, ta, cb_id, db_domain, tb in candidates:
                if ix_id in dl_map:
                    dl_id, dl_desc = dl_map[ix_id]
                    rows.append(
                        {
                            "id": dl_id,
                            "intersection_id": ix_id,
                            "significance": sig,
                            "parent_a": ca_id,
                            "domain_a": da,
                            "text_a": ta,
                            "parent_b": cb_id,
                            "domain_b": db_domain,
                            "text_b": tb,
                            "description": dl_desc,
                        }
                    )
        except (OSError, sqlite3.Error) as exc:
            logger.warning("Failed to query wake-pending: %s", exc)
            return []

        return _deduplicate_pending_entries(rows, verified_domain_pairs, limit)

    @router.post("/dream/wake-submit")
    # Sync — FastAPI runs in threadpool; SQLite/substrate ops are local
    def submit_wake_results(req: WakeSubmitRequest) -> dict:
        return _apply_wake_results(substrate, store, req.results)

    return router
