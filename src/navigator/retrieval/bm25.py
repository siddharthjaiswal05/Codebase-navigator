"""Okapi BM25 over identifier-level tokens.

Implemented directly rather than taken from a library because the tokenization
is the point: the index is built over camelCase and snake_case parts, and the
scoring has to see the same terms the query produces.
"""

from __future__ import annotations

import math
import pickle
from collections import Counter
from pathlib import Path

from .tokenize import tokenize, tokenize_query


class BM25Index:
    """Sparse lexical retrieval with the standard Okapi parameters."""

    def __init__(self, k1: float = 1.2, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.doc_ids: list[str] = []
        self.doc_lengths: list[int] = []
        self.term_freqs: list[dict[str, int]] = []
        self.doc_freqs: Counter[str] = Counter()
        self.postings: dict[str, list[int]] = {}
        self.avg_doc_length: float = 0.0

    # -- build ------------------------------------------------------------
    def add(self, doc_id: str, text: str) -> None:
        tokens = tokenize(text)
        counts = Counter(tokens)
        index = len(self.doc_ids)

        self.doc_ids.append(doc_id)
        self.term_freqs.append(dict(counts))
        self.doc_lengths.append(len(tokens))

        for term in counts:
            self.doc_freqs[term] += 1
            self.postings.setdefault(term, []).append(index)

    def finalize(self) -> "BM25Index":
        total = sum(self.doc_lengths)
        self.avg_doc_length = total / len(self.doc_lengths) if self.doc_lengths else 0.0
        return self

    @classmethod
    def build(cls, documents: list[tuple[str, str]], **kwargs) -> "BM25Index":
        index = cls(**kwargs)
        for doc_id, text in documents:
            index.add(doc_id, text)
        return index.finalize()

    # -- query ------------------------------------------------------------
    def _idf(self, term: str) -> float:
        n = len(self.doc_ids)
        df = self.doc_freqs.get(term, 0)
        if df == 0:
            return 0.0
        # Standard BM25 idf with the +1 that keeps common terms non-negative.
        return math.log(1 + (n - df + 0.5) / (df + 0.5))

    def search(self, query: str, top_k: int = 20) -> list[tuple[str, float]]:
        """Top-k (doc_id, score), highest first.

        Only documents containing at least one query term are scored, via the
        postings lists, so cost tracks the query rather than the corpus.
        """
        terms = tokenize_query(query)
        if not terms or not self.doc_ids:
            return []

        scores: dict[int, float] = {}
        for term in terms:
            idf = self._idf(term)
            if idf == 0.0:
                continue
            for doc_index in self.postings.get(term, ()):
                freq = self.term_freqs[doc_index].get(term, 0)
                length = self.doc_lengths[doc_index] or 1
                norm = 1 - self.b + self.b * (length / (self.avg_doc_length or 1))
                contribution = idf * (freq * (self.k1 + 1)) / (freq + self.k1 * norm)
                scores[doc_index] = scores.get(doc_index, 0.0) + contribution

        ranked = sorted(scores.items(), key=lambda kv: (-kv[1], self.doc_ids[kv[0]]))
        return [(self.doc_ids[i], score) for i, score in ranked[:top_k]]

    # -- persistence ------------------------------------------------------
    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as handle:
            pickle.dump(
                {
                    "k1": self.k1,
                    "b": self.b,
                    "doc_ids": self.doc_ids,
                    "doc_lengths": self.doc_lengths,
                    "term_freqs": self.term_freqs,
                    "doc_freqs": self.doc_freqs,
                    "postings": self.postings,
                    "avg_doc_length": self.avg_doc_length,
                },
                handle,
                protocol=pickle.HIGHEST_PROTOCOL,
            )

    @classmethod
    def load(cls, path: Path) -> "BM25Index":
        with open(path, "rb") as handle:
            data = pickle.load(handle)
        index = cls(k1=data["k1"], b=data["b"])
        index.doc_ids = data["doc_ids"]
        index.doc_lengths = data["doc_lengths"]
        index.term_freqs = data["term_freqs"]
        index.doc_freqs = data["doc_freqs"]
        index.postings = data["postings"]
        index.avg_doc_length = data["avg_doc_length"]
        return index

    def __len__(self) -> int:
        return len(self.doc_ids)
