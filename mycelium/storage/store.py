"""Store — SQLite persistence layer for the Mycelium substrate.

Phase 1: SQLite + pickle for embeddings.
         In-memory nearest-neighbor queries with numpy.
         Migration path: replace with pgvector when corpus > 100k cells.

Schema:
  cells: all CognitiveCells (embeddings as BLOB)
  intersections: all Intersections
  snapshots: substrate state snapshots (JSON metadata + tick count)
  dream_log: oneiric consolidation discoveries
"""

from __future__ import annotations

import logging
import pickle
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from mycelium.core.cell import (
    CellID,
    CellState,
    CognitiveCell,
    Origin,
    OriginContext,
)
from mycelium.core.intersection import Intersection, IntersectionID

if TYPE_CHECKING:
    from collections.abc import Generator

    from mycelium.core.substrate import SubstrateSnapshot
    from mycelium.provenance.anchor import AnchorRecord

logger = logging.getLogger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS cells (
    id          TEXT PRIMARY KEY,
    embedding   BLOB NOT NULL,
    radius      REAL NOT NULL,
    confidence  REAL NOT NULL,
    energy      REAL NOT NULL,
    state       TEXT NOT NULL,
    origin_ts   TEXT NOT NULL,
    origin_src  TEXT NOT NULL,
    origin_ctx  TEXT NOT NULL,
    participant TEXT NOT NULL DEFAULT '',
    created_at  TEXT NOT NULL,
    last_access TEXT NOT NULL,
    access_cnt  INTEGER NOT NULL DEFAULT 0,
    parent_ids  TEXT NOT NULL DEFAULT '',
    generation  INTEGER NOT NULL DEFAULT 0,
    domain      TEXT NOT NULL DEFAULT '',
    text        TEXT NOT NULL DEFAULT ''
);

CREATE TABLE IF NOT EXISTS intersections (
    id          TEXT PRIMARY KEY,
    parent_a    TEXT NOT NULL,
    parent_b    TEXT NOT NULL,
    embedding   BLOB NOT NULL,
    overlap     REAL NOT NULL,
    novelty     REAL NOT NULL,
    significance REAL NOT NULL,
    discovered  TEXT NOT NULL,
    promoted    INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS snapshots (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    tick_count  INTEGER NOT NULL,
    cell_count  INTEGER NOT NULL,
    ix_count    INTEGER NOT NULL,
    snapshot_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS dream_log (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    intersection_id TEXT NOT NULL,
    discovered_at   TEXT NOT NULL,
    description     TEXT NOT NULL DEFAULT ''
);

CREATE TABLE IF NOT EXISTS signatures (
    cell_id     TEXT PRIMARY KEY,
    cell_hash   TEXT NOT NULL,
    signature   BLOB NOT NULL,
    public_key  BLOB NOT NULL,
    signed_at   TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS merkle_anchors (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    merkle_root TEXT NOT NULL,
    cell_count  INTEGER NOT NULL,
    anchored_at TEXT NOT NULL,
    tx_hash     TEXT
);
"""


class SubstrateStore:
    """SQLite-backed persistence for the Mycelium substrate."""

    def __init__(self, db_path: str | Path = "data/substrate.db") -> None:
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()
        logger.info("SubstrateStore initialized at %s", self._db_path)

    @contextmanager
    def _conn(self) -> Generator[sqlite3.Connection, None, None]:
        conn = sqlite3.connect(self._db_path, timeout=30)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=30000")
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _init_schema(self) -> None:
        with self._conn() as conn:
            conn.executescript(_SCHEMA)
            # Dream agent migrations (idempotent)
            for col, typedef in [
                ("seen_at", "TEXT DEFAULT NULL"),
                ("wake_verified", "INTEGER DEFAULT NULL"),
                ("wake_reasoning", "TEXT DEFAULT ''"),
            ]:
                try:
                    conn.execute(f"ALTER TABLE dream_log ADD COLUMN {col} {typedef}")
                except sqlite3.OperationalError:
                    pass  # Column already exists

    # ──────────────────────────────────────────────
    # Cell persistence
    # ──────────────────────────────────────────────

    def save_cell(self, cell: CognitiveCell) -> None:
        """Upsert a CognitiveCell and its signature if present."""
        with self._conn() as conn:
            conn.execute(
                """INSERT OR REPLACE INTO cells VALUES
                   (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    cell.id,
                    pickle.dumps(cell.embedding),
                    cell.radius,
                    cell.confidence,
                    cell.energy,
                    cell.state.value,
                    cell.origin.timestamp.isoformat(),
                    cell.origin.source,
                    cell.origin.context.value,
                    cell.origin.participant_id,
                    cell.created_at.isoformat(),
                    cell.last_accessed.isoformat(),
                    cell.access_count,
                    ",".join(cell.parent_ids),
                    cell.generation,
                    cell.domain,
                    cell.text,
                ),
            )
            # Persist signature if cell is signed
            if cell.origin.signature and cell.origin.public_key:
                from mycelium.provenance.hasher import (
                    compute_cell_hash,
                )

                conn.execute(
                    """INSERT OR REPLACE INTO signatures
                       VALUES (?,?,?,?,?)""",
                    (
                        cell.id,
                        compute_cell_hash(cell),
                        cell.origin.signature,
                        cell.origin.public_key,
                        cell.created_at.isoformat(),
                    ),
                )

    def load_cells(self) -> dict[CellID, CognitiveCell]:
        """Load all cells from storage, including signatures."""
        cells: dict[CellID, CognitiveCell] = {}

        # Load signatures index
        sigs: dict[str, tuple[bytes, bytes]] = {}
        with self._conn() as conn:
            rows = conn.execute("SELECT * FROM cells").fetchall()
            sig_rows = conn.execute(
                "SELECT cell_id, signature, public_key "
                "FROM signatures"
            ).fetchall()
        for sr in sig_rows:
            sigs[sr["cell_id"]] = (
                sr["signature"],
                sr["public_key"],
            )

        for row in rows:
            r = dict(row)
            embedding: np.ndarray = pickle.loads(r["embedding"])
            parent_ids = [
                CellID(p) for p in r["parent_ids"].split(",") if p
            ]
            sig_data = sigs.get(r["id"])
            cell = CognitiveCell(
                id=CellID(r["id"]),
                embedding=embedding.astype(np.float32),
                radius=r["radius"],
                confidence=r["confidence"],
                energy=r["energy"],
                state=CellState(r["state"]),
                origin=Origin(
                    timestamp=datetime.fromisoformat(
                        r["origin_ts"]
                    ),
                    source=r["origin_src"],
                    context=OriginContext(r["origin_ctx"]),
                    participant_id=r["participant"],
                    signature=sig_data[0] if sig_data else None,
                    public_key=sig_data[1] if sig_data else None,
                ),
                created_at=datetime.fromisoformat(r["created_at"]),
                last_accessed=datetime.fromisoformat(
                    r["last_access"]
                ),
                access_count=r["access_cnt"],
                parent_ids=parent_ids,
                generation=r["generation"],
                domain=r.get("domain", ""),
                text=r.get("text", ""),
            )
            cells[cell.id] = cell
        return cells

    # ──────────────────────────────────────────────
    # Intersection persistence
    # ──────────────────────────────────────────────

    def save_intersection(self, ix: Intersection) -> None:
        """Upsert an Intersection."""
        with self._conn() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO intersections VALUES (?,?,?,?,?,?,?,?,?)",
                (
                    ix.id,
                    ix.parent_a_id,
                    ix.parent_b_id,
                    pickle.dumps(ix.embedding),
                    ix.overlap,
                    ix.novelty,
                    ix.significance,
                    ix.discovered_at.isoformat(),
                    int(ix.promoted),
                ),
            )

    def load_intersections(self) -> dict[IntersectionID, Intersection]:
        """Load all intersections from storage."""
        result: dict[IntersectionID, Intersection] = {}
        with self._conn() as conn:
            rows = conn.execute("SELECT * FROM intersections").fetchall()
        for row in rows:
            embedding: np.ndarray = pickle.loads(row["embedding"])
            ix = Intersection(
                id=IntersectionID(row["id"]),
                parent_a_id=CellID(row["parent_a"]),
                parent_b_id=CellID(row["parent_b"]),
                embedding=embedding.astype(np.float32),
                overlap=row["overlap"],
                novelty=row["novelty"],
                significance=row["significance"],
                discovered_at=datetime.fromisoformat(row["discovered"]),
                promoted=bool(row["promoted"]),
            )
            result[ix.id] = ix
        return result

    # ──────────────────────────────────────────────
    # Snapshot
    # ──────────────────────────────────────────────

    def save_snapshot(self, snapshot: SubstrateSnapshot) -> None:
        """Record a substrate state snapshot."""
        with self._conn() as conn:
            # Batch upsert cells
            cell_rows = []
            sig_rows = []
            for cell in snapshot.cells.values():
                cell_rows.append((
                    cell.id,
                    pickle.dumps(cell.embedding),
                    cell.radius,
                    cell.confidence,
                    cell.energy,
                    cell.state.value,
                    cell.origin.timestamp.isoformat(),
                    cell.origin.source,
                    cell.origin.context.value,
                    cell.origin.participant_id,
                    cell.created_at.isoformat(),
                    cell.last_accessed.isoformat(),
                    cell.access_count,
                    ",".join(cell.parent_ids),
                    cell.generation,
                    cell.domain,
                    cell.text,
                ))
                if cell.origin.signature and cell.origin.public_key:
                    from mycelium.provenance.hasher import (
                        compute_cell_hash,
                    )
                    sig_rows.append((
                        cell.id,
                        compute_cell_hash(cell),
                        cell.origin.signature,
                        cell.origin.public_key,
                        cell.created_at.isoformat(),
                    ))

            conn.executemany(
                "INSERT OR REPLACE INTO cells VALUES"
                " (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                cell_rows,
            )
            if sig_rows:
                conn.executemany(
                    "INSERT OR REPLACE INTO signatures VALUES"
                    " (?,?,?,?,?)",
                    sig_rows,
                )

            # Batch upsert intersections
            ix_rows = [
                (
                    ix.id,
                    ix.parent_a_id,
                    ix.parent_b_id,
                    pickle.dumps(ix.embedding),
                    ix.overlap,
                    ix.novelty,
                    ix.significance,
                    ix.discovered_at.isoformat(),
                    int(ix.promoted),
                )
                for ix in snapshot.intersections.values()
            ]
            conn.executemany(
                "INSERT OR REPLACE INTO intersections VALUES"
                " (?,?,?,?,?,?,?,?,?)",
                ix_rows,
            )

            # Snapshot metadata
            conn.execute(
                "INSERT INTO snapshots"
                " (tick_count, cell_count, ix_count, snapshot_at)"
                " VALUES (?,?,?,?)",
                (
                    snapshot.tick_count,
                    len(snapshot.cells),
                    len(snapshot.intersections),
                    snapshot.snapshot_at.isoformat(),
                ),
            )

            # Batch dream log
            if snapshot.dream_log:
                conn.executemany(
                    "INSERT OR IGNORE INTO dream_log"
                    " (intersection_id, discovered_at, description)"
                    " VALUES (?,?,?)",
                    [
                        (
                            entry.intersection_id,
                            entry.discovered_at.isoformat(),
                            entry.description,
                        )
                        for entry in snapshot.dream_log
                    ],
                )
        logger.info(
            "Snapshot saved: tick=%d, cells=%d, intersections=%d",
            snapshot.tick_count,
            len(snapshot.cells),
            len(snapshot.intersections),
        )

    # ──────────────────────────────────────────────
    # Dream log queries (Dream agent)
    # ──────────────────────────────────────────────

    def get_unseen_dreams(
        self, min_significance: float = 0.0, limit: int = 50,
        verified_only: bool = False,
    ) -> list[dict]:
        """Return dream_log entries that haven't been seen yet."""
        with self._conn() as conn:
            query = """
                SELECT dl.id, dl.intersection_id, dl.discovered_at, dl.description,
                       dl.wake_verified, dl.wake_reasoning,
                       ix.significance, ix.overlap, ix.novelty,
                       ix.parent_a, ix.parent_b
                FROM dream_log dl
                JOIN intersections ix ON dl.intersection_id = ix.id
                WHERE dl.seen_at IS NULL
                  AND ix.significance >= ?
            """
            if verified_only:
                query += " AND dl.wake_verified = 1"
            query += " ORDER BY ix.significance DESC LIMIT ?"
            rows = conn.execute(query, (min_significance, limit)).fetchall()
        return [dict(r) for r in rows]

    def mark_dreams_seen(self, dream_ids: list[int] | None = None) -> int:
        """Mark dream_log entries as seen. If dream_ids is None, mark all."""
        now = datetime.now().isoformat()
        with self._conn() as conn:
            if dream_ids is None:
                cursor = conn.execute(
                    "UPDATE dream_log SET seen_at = ? WHERE seen_at IS NULL",
                    (now,),
                )
            else:
                placeholders = ",".join("?" for _ in dream_ids)
                cursor = conn.execute(
                    f"UPDATE dream_log SET seen_at = ? WHERE id IN ({placeholders}) AND seen_at IS NULL",
                    [now, *dream_ids],
                )
            return cursor.rowcount

    def set_wake_verification(
        self, dream_id: int, verified: bool, reasoning: str = "",
    ) -> None:
        """Record wake filter verification result for a dream entry."""
        with self._conn() as conn:
            conn.execute(
                "UPDATE dream_log SET wake_verified = ?, wake_reasoning = ? WHERE id = ?",
                (int(verified), reasoning, dream_id),
            )

    def find_dream_log_by_cells(
        self, cell_a_id: str, cell_b_id: str,
    ) -> dict | None:
        """Find the most recent dream_log entry matching a pair of parent cells."""
        with self._conn() as conn:
            row = conn.execute(
                """SELECT dl.id FROM dream_log dl
                   JOIN intersections ix ON dl.intersection_id = ix.id
                   WHERE (ix.parent_a = ? AND ix.parent_b = ?)
                      OR (ix.parent_a = ? AND ix.parent_b = ?)
                   ORDER BY dl.id DESC LIMIT 1""",
                (cell_a_id, cell_b_id, cell_b_id, cell_a_id),
            ).fetchone()
        return dict(row) if row else None

    def find_dream_log_by_intersection(
        self, intersection_id: str,
    ) -> dict | None:
        """Find the most recent dream_log entry for an intersection ID."""
        with self._conn() as conn:
            row = conn.execute(
                "SELECT id FROM dream_log WHERE intersection_id = ? "
                "ORDER BY id DESC LIMIT 1",
                (intersection_id,),
            ).fetchone()
        return dict(row) if row else None

    def get_unverified_dreams(
        self, min_significance: float = 0.0, limit: int = 50,
    ) -> list[dict]:
        """Return unverified dream_log entries with cell data for wake evaluation."""
        with self._conn() as conn:
            rows = conn.execute(
                """
                SELECT dl.id AS dream_id, dl.intersection_id,
                       ix.significance, ix.overlap, ix.novelty,
                       ix.parent_a, ix.parent_b
                FROM dream_log dl
                JOIN intersections ix ON dl.intersection_id = ix.id
                WHERE dl.wake_verified IS NULL
                  AND ix.significance >= ?
                GROUP BY ix.parent_a, ix.parent_b
                ORDER BY ix.significance DESC
                LIMIT ?
                """,
                (min_significance, limit),
            ).fetchall()
        return [dict(r) for r in rows]

    def get_unverified_dream_log_ids(
        self, intersection_id: str,
    ) -> list[int]:
        """Return IDs of unverified dream_log entries for an intersection."""
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT id FROM dream_log WHERE intersection_id = ? AND wake_verified IS NULL",
                (intersection_id,),
            ).fetchall()
        return [r["id"] for r in rows]

    def get_cell_raw(self, cell_id: str) -> dict | None:
        """Return raw cell row as dict, or None."""
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM cells WHERE id = ?", (cell_id,),
            ).fetchone()
        return dict(row) if row else None

    def save_dream_log_entry(
        self, intersection_id: str, discovered_at: str, description: str = "",
    ) -> None:
        """Insert a dream_log entry (ignores duplicates)."""
        with self._conn() as conn:
            conn.execute(
                "INSERT OR IGNORE INTO dream_log"
                " (intersection_id, discovered_at, description)"
                " VALUES (?,?,?)",
                (intersection_id, discovered_at, description),
            )

    def get_dream_log(
        self,
        limit: int = 100,
        offset: int = 0,
        wake_verified: bool | None = None,
    ) -> list[dict]:
        """Return paginated dream log with optional filter on wake_verified."""
        with self._conn() as conn:
            query = """
                SELECT dl.id, dl.intersection_id, dl.discovered_at, dl.description,
                       dl.seen_at, dl.wake_verified, dl.wake_reasoning,
                       ix.significance, ix.overlap, ix.novelty,
                       ix.parent_a, ix.parent_b
                FROM dream_log dl
                JOIN intersections ix ON dl.intersection_id = ix.id
            """
            params: list = []
            if wake_verified is not None:
                query += " WHERE dl.wake_verified = ?"
                params.append(int(wake_verified))
            query += " ORDER BY dl.id DESC LIMIT ? OFFSET ?"
            params.extend([limit, offset])
            rows = conn.execute(query, params).fetchall()
        return [dict(r) for r in rows]

    # ──────────────────────────────────────────────
    # Merkle anchor persistence
    # ──────────────────────────────────────────────

    def save_anchor(self, record: AnchorRecord) -> None:
        """Store a Merkle root anchor."""
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO merkle_anchors"
                " (merkle_root, cell_count, anchored_at, tx_hash)"
                " VALUES (?,?,?,?)",
                (
                    record.merkle_root,
                    record.cell_count,
                    record.anchored_at.isoformat(),
                    record.tx_hash,
                ),
            )

    def load_anchors(self) -> list:
        """Load all stored Merkle anchors."""
        from mycelium.provenance.anchor import AnchorRecord

        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM merkle_anchors "
                "ORDER BY anchored_at DESC"
            ).fetchall()

        return [
            AnchorRecord(
                merkle_root=row["merkle_root"],
                cell_count=row["cell_count"],
                anchored_at=datetime.fromisoformat(
                    row["anchored_at"]
                ),
                tx_hash=row["tx_hash"],
            )
            for row in rows
        ]

