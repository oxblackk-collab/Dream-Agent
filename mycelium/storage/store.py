"""Store — SQLite persistence layer for the Mycelium substrate.

Embeddings stored as raw float32 bytes via numpy tobytes/frombuffer.
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

CREATE INDEX IF NOT EXISTS idx_dream_log_intersection_id ON dream_log(intersection_id);

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
        except BaseException:
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

            # Index on wake_verified — must be created AFTER the ALTER TABLE migration
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_dream_log_wake_verified"
                " ON dream_log(wake_verified)"
            )

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
                    cell.embedding.astype(np.float32).tobytes(),
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

    @staticmethod
    def _deserialize_cell_row(
        row: dict,
        sigs: dict[str, tuple[bytes, bytes]],
    ) -> CognitiveCell:
        """Deserialize a single cell row into a CognitiveCell."""
        embedding = np.frombuffer(row["embedding"], dtype=np.float32).copy()
        parent_ids = [CellID(p) for p in row["parent_ids"].split(",") if p]
        sig_data = sigs.get(row["id"])
        return CognitiveCell(
            id=CellID(row["id"]),
            embedding=embedding.astype(np.float32),
            radius=row["radius"],
            confidence=row["confidence"],
            energy=row["energy"],
            state=CellState(row["state"]),
            origin=Origin(
                timestamp=datetime.fromisoformat(row["origin_ts"]),
                source=row["origin_src"],
                context=OriginContext(row["origin_ctx"]),
                participant_id=row["participant"],
                signature=sig_data[0] if sig_data else None,
                public_key=sig_data[1] if sig_data else None,
            ),
            created_at=datetime.fromisoformat(row["created_at"]),
            last_accessed=datetime.fromisoformat(row["last_access"]),
            access_count=row["access_cnt"],
            parent_ids=parent_ids,
            generation=row["generation"],
            domain=row.get("domain", ""),
            text=row.get("text", ""),
        )

    def load_cells(self) -> dict[CellID, CognitiveCell]:
        """Load all cells from storage, including signatures."""
        sigs: dict[str, tuple[bytes, bytes]] = {}
        with self._conn() as conn:
            rows = conn.execute("SELECT * FROM cells").fetchall()
            sig_rows = conn.execute(
                "SELECT cell_id, signature, public_key FROM signatures"
            ).fetchall()
        for sr in sig_rows:
            sigs[sr["cell_id"]] = (sr["signature"], sr["public_key"])

        cells: dict[CellID, CognitiveCell] = {}
        for row in rows:
            cell = self._deserialize_cell_row(dict(row), sigs)
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
                    ix.embedding.astype(np.float32).tobytes(),
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
            embedding = np.frombuffer(row["embedding"], dtype=np.float32).copy()
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

    @staticmethod
    def _serialize_cells(
        cells: dict,
    ) -> tuple[list[tuple], list[tuple]]:
        """Serialize cells into row tuples for batch insert.

        Returns (cell_rows, signature_rows).
        """
        cell_rows: list[tuple] = []
        sig_rows: list[tuple] = []
        for cell in cells.values():
            cell_rows.append(
                (
                    cell.id,
                    cell.embedding.astype(np.float32).tobytes(),
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
                )
            )
            if cell.origin.signature and cell.origin.public_key:
                from mycelium.provenance.hasher import compute_cell_hash

                sig_rows.append(
                    (
                        cell.id,
                        compute_cell_hash(cell),
                        cell.origin.signature,
                        cell.origin.public_key,
                        cell.created_at.isoformat(),
                    )
                )
        return cell_rows, sig_rows

    @staticmethod
    def _serialize_intersections(
        intersections: dict,
    ) -> list[tuple]:
        """Serialize intersections into row tuples for batch insert."""
        return [
            (
                ix.id,
                ix.parent_a_id,
                ix.parent_b_id,
                ix.embedding.astype(np.float32).tobytes(),
                ix.overlap,
                ix.novelty,
                ix.significance,
                ix.discovered_at.isoformat(),
                int(ix.promoted),
            )
            for ix in intersections.values()
        ]

    @staticmethod
    def _write_snapshot_tables(
        conn: sqlite3.Connection,
        snapshot: SubstrateSnapshot,
        cell_rows: list[tuple],
        sig_rows: list[tuple],
        ix_rows: list[tuple],
    ) -> None:
        """Batch-write all snapshot data into the open connection.

        Inserts/replaces cells, signatures, intersections, the snapshot
        metadata row, and any dream_log entries.
        """
        conn.executemany(
            "INSERT OR REPLACE INTO cells VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            cell_rows,
        )
        if sig_rows:
            conn.executemany(
                "INSERT OR REPLACE INTO signatures VALUES (?,?,?,?,?)",
                sig_rows,
            )
        conn.executemany(
            "INSERT OR REPLACE INTO intersections VALUES (?,?,?,?,?,?,?,?,?)",
            ix_rows,
        )
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

    def save_snapshot(self, snapshot: SubstrateSnapshot) -> None:
        """Record a substrate state snapshot.

        Uses batch executemany for cells and intersections — much faster
        than individual save_cell/save_intersection calls.
        """
        cell_rows, sig_rows = self._serialize_cells(snapshot.cells)
        ix_rows = self._serialize_intersections(snapshot.intersections)

        with self._conn() as conn:
            self._write_snapshot_tables(
                conn,
                snapshot,
                cell_rows,
                sig_rows,
                ix_rows,
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

    def save_dream_log_entry(
        self,
        intersection_id: str,
        discovered_at: str,
        description: str = "",
    ) -> None:
        """Persist a dream_log entry.

        Inserts OR IGNOREs into dream_log — safe for duplicate calls.
        """
        with self._conn() as conn:
            conn.execute(
                "INSERT OR IGNORE INTO dream_log"
                " (intersection_id, discovered_at, description)"
                " VALUES (?,?,?)",
                (intersection_id, discovered_at, description),
            )

    def get_unseen_dreams(
        self,
        min_significance: float = 0.0,
        limit: int = 50,
        verified_only: bool = False,
    ) -> list[dict]:
        """Return dream_log entries that haven't been seen yet.

        Joins with intersections to get significance and parent cell info.
        If verified_only=True, only returns wake-verified entries.
        """
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
        self,
        dream_id: int,
        verified: bool,
        reasoning: str = "",
    ) -> None:
        """Record wake filter verification result for a dream entry."""
        with self._conn() as conn:
            conn.execute(
                "UPDATE dream_log SET wake_verified = ?, wake_reasoning = ? WHERE id = ?",
                (int(verified), reasoning, dream_id),
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
    # Wake verification queries (Dream agent)
    # ──────────────────────────────────────────────

    def get_verified_domain_pairs(self) -> set[frozenset[str]]:
        """Return domain pairs that already have a wake-verified insight.

        Used for deduplication — skip domain pairs that have already been
        verified so the wake filter focuses on genuinely new connections.
        """
        with self._conn() as conn:
            rows = conn.execute(
                """SELECT DISTINCT c1.domain, c2.domain
                   FROM dream_log dl
                   JOIN intersections ix ON dl.intersection_id = ix.id
                   JOIN cells c1 ON ix.parent_a = c1.id
                   JOIN cells c2 ON ix.parent_b = c2.id
                   WHERE dl.wake_verified = 1
                   AND c1.domain != '' AND c2.domain != ''""",
            ).fetchall()
        return {frozenset({r[0], r[1]}) for r in rows}

    def find_dream_log_by_parents(
        self,
        cell_id_a: str,
        cell_id_b: str,
    ) -> int | None:
        """Find the most recent dream_log entry matching parent cell IDs.

        Checks both orderings (a,b) and (b,a). Returns the dream_log id
        or None if no matching entry exists.
        """
        with self._conn() as conn:
            row = conn.execute(
                """SELECT dl.id FROM dream_log dl
                   JOIN intersections ix ON dl.intersection_id = ix.id
                   WHERE (ix.parent_a = ? AND ix.parent_b = ?)
                      OR (ix.parent_a = ? AND ix.parent_b = ?)
                   ORDER BY dl.id DESC LIMIT 1""",
                (cell_id_a, cell_id_b, cell_id_b, cell_id_a),
            ).fetchone()
        return row["id"] if row else None

    def get_unverified_dream_entries_by_intersection(
        self,
        intersection_ids: list[str],
    ) -> dict[str, tuple[int, str]]:
        """Batch lookup unverified dream_log entries by intersection IDs.

        Returns a dict mapping intersection_id -> (dream_log_id, description)
        for entries where wake_verified IS NULL.
        """
        result: dict[str, tuple[int, str]] = {}
        with self._conn() as conn:
            for batch_start in range(0, len(intersection_ids), 100):
                batch = intersection_ids[batch_start : batch_start + 100]
                placeholders = ",".join("?" * len(batch))
                rows = conn.execute(
                    f"""SELECT intersection_id, MIN(id) AS id,
                               MIN(description) AS description
                        FROM dream_log
                        WHERE intersection_id IN ({placeholders})
                          AND wake_verified IS NULL
                        GROUP BY intersection_id""",
                    batch,
                ).fetchall()
                for dr in rows:
                    result[dr["intersection_id"]] = (
                        dr["id"],
                        dr["description"] or "",
                    )
        return result

    def get_parent_ids_for_dream(
        self,
        dream_log_id: int,
    ) -> tuple[str, str] | None:
        """Get parent cell IDs (parent_a, parent_b) for a dream_log entry.

        Returns a tuple (parent_a, parent_b) or None if not found.
        """
        with self._conn() as conn:
            row = conn.execute(
                """SELECT ix.parent_a, ix.parent_b
                   FROM dream_log dl
                   JOIN intersections ix ON dl.intersection_id = ix.id
                   WHERE dl.id = ?""",
                (dream_log_id,),
            ).fetchone()
        if row:
            return (row["parent_a"], row["parent_b"])
        return None

    # ──────────────────────────────────────────────
    # Merkle anchor persistence
    # ──────────────────────────────────────────────

    def save_anchor(self, record: object) -> None:
        """Store a Merkle root anchor."""
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO merkle_anchors"
                " (merkle_root, cell_count, anchored_at, tx_hash)"
                " VALUES (?,?,?,?)",
                (
                    record.merkle_root,  # type: ignore[union-attr]
                    record.cell_count,  # type: ignore[union-attr]
                    record.anchored_at.isoformat(),  # type: ignore[union-attr]
                    record.tx_hash,  # type: ignore[union-attr]
                ),
            )

    def load_anchors(self) -> list:
        """Load all stored Merkle anchors."""
        from mycelium.provenance.anchor import AnchorRecord

        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM merkle_anchors ORDER BY anchored_at DESC"
            ).fetchall()

        return [
            AnchorRecord(
                merkle_root=row["merkle_root"],
                cell_count=row["cell_count"],
                anchored_at=datetime.fromisoformat(row["anchored_at"]),
                tx_hash=row["tx_hash"],
            )
            for row in rows
        ]
