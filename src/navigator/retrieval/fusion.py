"""Reciprocal rank fusion.

BM25 scores and cosine similarities are not comparable: one is unbounded and
corpus-dependent, the other sits in [-1, 1]. Normalising them onto a shared
scale means inventing a relationship between two different distributions, and
the choice of normalisation then quietly decides the ranking.

RRF sidesteps that by discarding the scores and combining ranks:

    score(d) = sum over retrievers of  weight / (k + rank(d))

A document ranked highly by either arm surfaces; one ranked highly by both wins.
`k` damps the influence of the very top ranks, so a single retriever cannot
dominate the fused list on one confident hit. k=60 is the value from the
original formulation and is used here unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass, field

RRF_K = 60


@dataclass
class FusedResult:
    doc_id: str
    score: float
    # Per-retriever rank, 1-indexed, for results that arm returned at all.
    ranks: dict[str, int] = field(default_factory=dict)
    raw_scores: dict[str, float] = field(default_factory=dict)

    def sources(self) -> list[str]:
        return sorted(self.ranks)

    def found_by_both(self) -> bool:
        return len(self.ranks) > 1


def reciprocal_rank_fusion(
    ranked_lists: dict[str, list[tuple[str, float]]],
    k: int = RRF_K,
    weights: dict[str, float] | None = None,
    top_k: int | None = None,
) -> list[FusedResult]:
    """Fuse named ranked lists into one.

    `ranked_lists` maps a retriever name to its (doc_id, score) list, best
    first. `weights` lets one arm count for more without changing the fusion.
    """
    weights = weights or {}
    accumulated: dict[str, FusedResult] = {}

    for retriever, results in ranked_lists.items():
        weight = weights.get(retriever, 1.0)
        for position, (doc_id, raw_score) in enumerate(results, start=1):
            entry = accumulated.get(doc_id)
            if entry is None:
                entry = FusedResult(doc_id=doc_id, score=0.0)
                accumulated[doc_id] = entry
            entry.score += weight / (k + position)
            entry.ranks[retriever] = position
            entry.raw_scores[retriever] = raw_score

    fused = sorted(accumulated.values(), key=lambda r: (-r.score, r.doc_id))
    return fused[:top_k] if top_k else fused
