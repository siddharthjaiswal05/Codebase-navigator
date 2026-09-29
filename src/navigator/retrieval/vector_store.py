"""Vector storage and search.

FAISS `IndexFlatIP` over L2-normalised vectors, which makes inner product exact
cosine similarity. Flat is the right choice at this corpus size: it is exact, so
retrieval quality is never confounded by an approximate index, and a repository
of a few tens of thousands of chunks searches in milliseconds.

A NumPy store with the same interface stands in when FAISS is unavailable, so
the pipeline runs anywhere. Both are exact, so they return the same ranking.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

# FAISS and PyTorch each bundle their own OpenMP runtime. On macOS, loading
# both into one process aborts with a hard segfault the moment either is used,
# which surfaces as an unexplained crash rather than an exception. Allowing the
# duplicate runtime is the documented workaround and must be set before faiss
# is imported.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

try:  # pragma: no cover - import guard
    import faiss

    _FAISS_AVAILABLE = True
except Exception:  # pragma: no cover
    _FAISS_AVAILABLE = False


def which_backend() -> str:
    return "faiss" if _FAISS_AVAILABLE else "numpy"


class VectorStore:
    """Exact inner-product search over unit vectors."""

    def __init__(self, dimension: int, backend: str | None = None):
        self.dimension = dimension
        self.ids: list[str] = []

        use_faiss = _FAISS_AVAILABLE if backend is None else backend == "faiss"
        if use_faiss and not _FAISS_AVAILABLE:
            raise RuntimeError("faiss backend requested but not installed")

        self.backend = "faiss" if use_faiss else "numpy"
        self._index = faiss.IndexFlatIP(dimension) if use_faiss else None
        self._matrix: np.ndarray | None = None

    def add(self, ids: list[str], vectors: np.ndarray) -> None:
        if vectors.dtype != np.float32:
            vectors = vectors.astype(np.float32)
        if vectors.shape[1] != self.dimension:
            raise ValueError(
                f"expected dimension {self.dimension}, got {vectors.shape[1]}"
            )

        self.ids.extend(ids)
        if self.backend == "faiss":
            self._index.add(vectors)
        else:
            self._matrix = (
                vectors if self._matrix is None
                else np.vstack([self._matrix, vectors])
            )

    def search(self, query_vectors: np.ndarray,
               top_k: int = 20) -> list[list[tuple[str, float]]]:
        """One ranked list per query row."""
        if not self.ids:
            return [[] for _ in range(len(query_vectors))]
        if query_vectors.dtype != np.float32:
            query_vectors = query_vectors.astype(np.float32)

        k = min(top_k, len(self.ids))
        if self.backend == "faiss":
            scores, indices = self._index.search(query_vectors, k)
        else:
            sims = query_vectors @ self._matrix.T
            indices = np.argsort(-sims, axis=1)[:, :k]
            scores = np.take_along_axis(sims, indices, axis=1)

        out: list[list[tuple[str, float]]] = []
        for row_scores, row_indices in zip(scores, indices):
            out.append(
                [
                    (self.ids[i], float(s))
                    for s, i in zip(row_scores, row_indices)
                    if i >= 0
                ]
            )
        return out

    def search_one(self, query_vector: np.ndarray,
                   top_k: int = 20) -> list[tuple[str, float]]:
        return self.search(query_vector.reshape(1, -1), top_k=top_k)[0]

    # -- persistence ------------------------------------------------------
    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        if self.backend == "faiss":
            faiss.write_index(self._index, str(path.with_suffix(".faiss")))
            np.save(path.with_suffix(".ids.npy"), np.array(self.ids, dtype=object),
                    allow_pickle=True)
        else:
            np.savez(
                path.with_suffix(".npz"),
                matrix=self._matrix,
                ids=np.array(self.ids, dtype=object),
            )

    @classmethod
    def load(cls, path: Path, dimension: int, backend: str) -> "VectorStore":
        store = cls(dimension, backend=backend)
        if backend == "faiss":
            store._index = faiss.read_index(str(path.with_suffix(".faiss")))
            store.ids = list(
                np.load(path.with_suffix(".ids.npy"), allow_pickle=True)
            )
        else:
            data = np.load(path.with_suffix(".npz"), allow_pickle=True)
            store._matrix = data["matrix"]
            store.ids = list(data["ids"])
        return store

    def __len__(self) -> int:
        return len(self.ids)
