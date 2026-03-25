"""Hasher — cryptographic provenance for CognitiveCells.

Every cell's birth certificate is a SHA-256 hash of its initial state.
Periodic Merkle root snapshots provide verifiable substrate integrity.
"""

from __future__ import annotations

import hashlib
import json
import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mycelium.core.cell import CognitiveCell
    from mycelium.core.substrate import SubstrateSnapshot

logger = logging.getLogger(__name__)


def compute_cell_hash(cell: CognitiveCell) -> str:
    """Compute a deterministic SHA-256 hash for a cell's origin state.

    The hash encodes: initial embedding, radius, confidence, origin timestamp,
    source, context, participant_id, and parent lineage.

    This is the cell's birth certificate — immutable from creation.
    """
    payload = {
        "id": cell.id,
        "embedding": cell.embedding.tolist(),
        "radius": cell.radius,
        "confidence": cell.confidence,
        "origin_timestamp": cell.origin.timestamp.isoformat(),
        "origin_source": cell.origin.source,
        "origin_context": cell.origin.context.value,
        "participant_id": cell.origin.participant_id,
        "parent_ids": sorted(cell.parent_ids),
        "generation": cell.generation,
        "parent_hash": cell.parent_hash,
    }
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode()).hexdigest()


def compute_merkle_root(cell_hashes: list[str]) -> str:
    """Compute a Merkle root from a list of cell hashes.

    Standard binary Merkle tree:
    - Leaf nodes: cell hashes
    - Internal nodes: SHA-256(left_hash + right_hash)
    - If odd number of leaves, duplicate the last leaf

    Returns the root hash as a hex string.
    Returns empty string for empty input.
    """
    if not cell_hashes:
        return ""

    layer = [h.encode() for h in sorted(cell_hashes)]  # sorted for determinism

    while len(layer) > 1:
        if len(layer) % 2 == 1:
            layer.append(layer[-1])  # duplicate last leaf
        next_layer: list[bytes] = []
        for i in range(0, len(layer), 2):
            combined = hashlib.sha256(layer[i] + layer[i + 1]).digest()
            next_layer.append(combined)
        layer = next_layer

    return layer[0].hex()


def compute_snapshot_merkle(snapshot: SubstrateSnapshot) -> str:
    """Compute a Merkle root for an entire substrate snapshot.

    The root is a cryptographic fingerprint of the substrate state.
    Any modification to any cell would change this root.
    """
    hashes = [compute_cell_hash(cell) for cell in snapshot.cells.values()]
    root = compute_merkle_root(hashes)
    logger.debug(
        "Merkle root computed for %d cells: %s...",
        len(hashes),
        root[:16] if root else "(empty)",
    )
    return root

