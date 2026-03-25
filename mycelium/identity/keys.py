"""Participant Identity — Ed25519 keypairs for cognitive bonds.

Each participant (human or AI) has a keypair. The public key IS the
participant's identity — no external registry needed. The private key
signs cell hashes, proving authorship.

This maps to The Cognitive Chain's "key sovereignty": the bond
between a human and an AI is cryptographically portable. Neither party
depends on a platform to prove the relationship exists.
"""

from __future__ import annotations

from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)
from cryptography.hazmat.primitives.serialization import (
    Encoding,
    NoEncryption,
    PrivateFormat,
    PublicFormat,
)


class ParticipantIdentity:
    """Ed25519 identity for a substrate participant.

    The participant_id is the hex-encoded public key (64 chars).
    This is deterministic: same keypair always produces the same ID.
    """

    def __init__(self, private_key: Ed25519PrivateKey) -> None:
        self._private_key = private_key
        self._public_key = private_key.public_key()
        self._public_bytes = self._public_key.public_bytes(
            Encoding.Raw, PublicFormat.Raw
        )

    @classmethod
    def generate(cls) -> ParticipantIdentity:
        """Generate a new random Ed25519 keypair."""
        return cls(Ed25519PrivateKey.generate())

    @classmethod
    def from_private_bytes(
        cls, private_bytes: bytes
    ) -> ParticipantIdentity:
        """Restore identity from raw private key bytes (32 bytes)."""
        from cryptography.hazmat.primitives.asymmetric.ed25519 import (
            Ed25519PrivateKey,
        )

        key = Ed25519PrivateKey.from_private_bytes(private_bytes)
        return cls(key)

    @property
    def participant_id(self) -> str:
        """Hex-encoded public key — the participant's identity."""
        return self._public_bytes.hex()

    @property
    def public_key_bytes(self) -> bytes:
        """Raw 32-byte public key."""
        return self._public_bytes

    @property
    def private_key_bytes(self) -> bytes:
        """Raw 32-byte private key. Handle with care."""
        return self._private_key.private_bytes(
            Encoding.Raw, PrivateFormat.Raw, NoEncryption()
        )

    @property
    def public_key(self) -> Ed25519PublicKey:
        """The public key object for verification."""
        return self._public_key

    def sign(self, data: bytes) -> bytes:
        """Sign arbitrary data with the private key.

        Returns a 64-byte Ed25519 signature.
        """
        return self._private_key.sign(data)

    @staticmethod
    def verify(
        data: bytes,
        signature: bytes,
        public_key_bytes: bytes,
    ) -> bool:
        """Verify a signature against a public key.

        Returns True if valid, False if tampered or wrong key.
        """
        from cryptography.exceptions import InvalidSignature
        from cryptography.hazmat.primitives.asymmetric.ed25519 import (
            Ed25519PublicKey,
        )

        try:
            pub = Ed25519PublicKey.from_public_bytes(
                public_key_bytes
            )
            pub.verify(signature, data)
        except (InvalidSignature, ValueError):
            return False
        return True
