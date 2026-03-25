"""Proof of Provenance — periodic Merkle root anchoring.

Computes the Merkle root of the substrate state and stores it locally
with a timestamp. This is the cryptographic fingerprint of the entire
substrate at a point in time — any modification to any cell would
produce a different root.

The local anchor is always stored. Blockchain anchoring (writing the
root to a testnet) is optional and gated behind the 'web3' package.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from mycelium.provenance.hasher import compute_snapshot_merkle

if TYPE_CHECKING:

    from mycelium.core.substrate import SubstrateSnapshot
    from mycelium.storage.store import SubstrateStore

logger = logging.getLogger(__name__)


@dataclass
class AnchorRecord:
    """A Merkle root anchored at a point in time."""

    merkle_root: str
    cell_count: int
    anchored_at: datetime
    tx_hash: str | None = None  # blockchain tx hash, if anchored


class ProvenanceAnchor:
    """Computes and stores Merkle root anchors."""

    def __init__(self, store: SubstrateStore) -> None:
        self._store = store

    def anchor(
        self, snapshot: SubstrateSnapshot
    ) -> AnchorRecord:
        """Compute Merkle root and store locally.

        Returns the AnchorRecord with the root and metadata.
        """
        merkle_root = compute_snapshot_merkle(snapshot)
        now = datetime.now(tz=UTC)
        cell_count = len(snapshot.cells)

        record = AnchorRecord(
            merkle_root=merkle_root,
            cell_count=cell_count,
            anchored_at=now,
        )

        self._store.save_anchor(record)

        logger.info(
            "Anchor: root=%s... cells=%d",
            merkle_root[:16] if merkle_root else "(empty)",
            cell_count,
        )
        return record

    def get_anchors(self) -> list[AnchorRecord]:
        """Load all stored anchors."""
        return self._store.load_anchors()

    def verify_snapshot(
        self,
        snapshot: SubstrateSnapshot,
        expected_root: str,
    ) -> bool:
        """Verify a snapshot matches a previously anchored root."""
        current_root = compute_snapshot_merkle(snapshot)
        return current_root == expected_root

