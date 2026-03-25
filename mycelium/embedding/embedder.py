"""Embedder — converts text to semantic vectors.

Default model: paraphrase-multilingual-MiniLM-L12-v2 (384 dimensions).
Chosen for multilingual semantic discrimination — same-language text from
different domains produces measurably different embeddings, which is
critical for the substrate's immune system (noise rejection).

Previous model (all-MiniLM-L6-v2) clustered all same-language text too
tightly, making noise indistinguishable from coherent content.

Design: BaseEmbedder is a Protocol — the substrate depends on the abstraction,
not the concrete implementation. Swap to any model without touching the core.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Protocol, cast, runtime_checkable

import numpy as np

if TYPE_CHECKING:
    from mycelium.core.cell import EmbeddingVector

logger = logging.getLogger(__name__)


@runtime_checkable
class BaseEmbedder(Protocol):
    """Protocol for all embedding implementations."""

    @property
    def dimensions(self) -> int:
        """Dimensionality of the embedding vectors."""
        ...

    def embed(self, text: str) -> EmbeddingVector:
        """Convert a single text to a normalized embedding vector."""
        ...

    def embed_batch(self, texts: list[str]) -> list[EmbeddingVector]:
        """Batch embedding for efficiency."""
        ...

    def semantic_distance(self, a: EmbeddingVector, b: EmbeddingVector) -> float:
        """Cosine distance between two embeddings. Range [0, 2]."""
        ...


_DEFAULT_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"


class SentenceTransformerEmbedder:
    """Wraps sentence-transformers for local embedding generation.

    Default: paraphrase-multilingual-MiniLM-L12-v2 (384D).
    Multilingual, good semantic discrimination, no API costs.
    """

    def __init__(self, model_name: str = _DEFAULT_MODEL) -> None:
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(model_name)
        self._dimensions = self._model.get_sentence_embedding_dimension() or 384
        logger.info(
            "SentenceTransformerEmbedder initialized: %s (%d dims)",
            model_name,
            self._dimensions,
        )

    @property
    def dimensions(self) -> int:
        return self._dimensions

    def embed(self, text: str) -> EmbeddingVector:
        """Embed a single text, normalize to unit vector."""
        raw = cast("np.ndarray", self._model.encode(text, normalize_embeddings=True))
        return raw.astype(np.float32)

    def embed_batch(self, texts: list[str]) -> list[EmbeddingVector]:
        """Batch embed for efficiency."""
        raw: np.ndarray = self._model.encode(texts, normalize_embeddings=True)
        return [row.astype(np.float32) for row in raw]

    def semantic_distance(self, a: EmbeddingVector, b: EmbeddingVector) -> float:
        """Cosine distance: 1 - cosine_similarity. Range [0, 2]."""
        dot = float(np.dot(a, b))
        return float(1.0 - dot)


class MockEmbedder:
    """Deterministic embedder for tests.

    Maps text to a fixed vector based on a hash of the text.
    No model loading — instantaneous. Produces normalized vectors.
    """

    def __init__(self, dimensions: int = 8) -> None:
        self._dimensions = dimensions

    @property
    def dimensions(self) -> int:
        return self._dimensions

    def embed(self, text: str) -> EmbeddingVector:
        """Deterministic embedding from text hash."""
        rng = np.random.default_rng(seed=hash(text) % (2**32))
        v = rng.standard_normal(self._dimensions).astype(np.float32)
        norm = float(np.linalg.norm(v))
        if norm > 0:
            v = v / norm
        return v

    def embed_batch(self, texts: list[str]) -> list[EmbeddingVector]:
        return [self.embed(t) for t in texts]

    def semantic_distance(self, a: EmbeddingVector, b: EmbeddingVector) -> float:
        dot = float(np.dot(a, b))
        return float(1.0 - dot)

