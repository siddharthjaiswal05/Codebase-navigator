"""Retrieval and citation metrics.

Localisation is scored at two granularities, because they answer different
questions. File-level asks whether the system pointed at the right file, which
is what a developer needs to start reading. Function-level asks whether it
pointed at the right symbol, which is what an automated patcher needs.

Citation validity is scored alongside accuracy rather than beneath it. An answer
that names the right function but cites a location that does not exist has not
been verified by anything, and counting it as correct would reward exactly the
behaviour this system is built to prevent.
"""

from __future__ import annotations

from dataclasses import dataclass, field


def recall_at_k(retrieved: list[str], gold: set[str], k: int) -> float:
    """Share of gold items appearing in the top k."""
    if not gold:
        return 0.0
    top = set(retrieved[:k])
    return len(top & gold) / len(gold)


def hit_at_k(retrieved: list[str], gold: set[str], k: int) -> float:
    """1.0 if any gold item is in the top k."""
    if not gold:
        return 0.0
    return 1.0 if set(retrieved[:k]) & gold else 0.0


def precision_at_k(retrieved: list[str], gold: set[str], k: int) -> float:
    if not retrieved[:k]:
        return 0.0
    return len(set(retrieved[:k]) & gold) / len(retrieved[:k])


def reciprocal_rank(retrieved: list[str], gold: set[str]) -> float:
    """1/rank of the first gold item, 0 if none appears."""
    for position, item in enumerate(retrieved, start=1):
        if item in gold:
            return 1.0 / position
    return 0.0


def average_precision(retrieved: list[str], gold: set[str]) -> float:
    if not gold:
        return 0.0
    hits, total = 0, 0.0
    for position, item in enumerate(retrieved, start=1):
        if item in gold:
            hits += 1
            total += hits / position
    return total / len(gold)


@dataclass
class MetricAccumulator:
    """Averages a metric family over a question set."""

    ks: tuple[int, ...] = (1, 3, 5, 10)
    _hit: dict[int, list[float]] = field(default_factory=dict)
    _recall: dict[int, list[float]] = field(default_factory=dict)
    _precision: dict[int, list[float]] = field(default_factory=dict)
    _mrr: list[float] = field(default_factory=list)
    _map: list[float] = field(default_factory=list)
    n: int = 0

    def add(self, retrieved: list[str], gold: set[str]) -> None:
        self.n += 1
        for k in self.ks:
            self._hit.setdefault(k, []).append(hit_at_k(retrieved, gold, k))
            self._recall.setdefault(k, []).append(recall_at_k(retrieved, gold, k))
            self._precision.setdefault(k, []).append(
                precision_at_k(retrieved, gold, k)
            )
        self._mrr.append(reciprocal_rank(retrieved, gold))
        self._map.append(average_precision(retrieved, gold))

    @staticmethod
    def _mean(values: list[float]) -> float:
        return round(sum(values) / len(values), 4) if values else 0.0

    def summary(self) -> dict:
        out: dict[str, float | int] = {"n": self.n}
        for k in self.ks:
            out[f"hit@{k}"] = self._mean(self._hit.get(k, []))
            out[f"recall@{k}"] = self._mean(self._recall.get(k, []))
            out[f"precision@{k}"] = self._mean(self._precision.get(k, []))
        out["mrr"] = self._mean(self._mrr)
        out["map"] = self._mean(self._map)
        return out


@dataclass
class CitationAccumulator:
    """Citation quality, tracked independently of retrieval rank."""

    total_answers: int = 0
    supported_answers: int = 0
    total_citations: int = 0
    valid_citations: int = 0
    answers_all_valid: int = 0
    answers_with_no_citation: int = 0
    failure_reasons: dict[str, int] = field(default_factory=dict)

    def add(self, citations: list) -> None:
        self.total_answers += 1
        if not citations:
            self.answers_with_no_citation += 1
            return

        valid = [c for c in citations if c.valid]
        self.total_citations += len(citations)
        self.valid_citations += len(valid)
        if valid:
            self.supported_answers += 1
        if len(valid) == len(citations):
            self.answers_all_valid += 1
        for citation in citations:
            if not citation.valid:
                reason = citation.reason or "unspecified"
                self.failure_reasons[reason] = self.failure_reasons.get(reason, 0) + 1

    def summary(self) -> dict:
        return {
            "answers": self.total_answers,
            "answers_with_no_citation": self.answers_with_no_citation,
            "total_citations": self.total_citations,
            "valid_citations": self.valid_citations,
            "citation_validity_rate": round(
                self.valid_citations / self.total_citations, 4
            ) if self.total_citations else 0.0,
            "supported_answer_rate": round(
                self.supported_answers / self.total_answers, 4
            ) if self.total_answers else 0.0,
            "fully_valid_answer_rate": round(
                self.answers_all_valid / self.total_answers, 4
            ) if self.total_answers else 0.0,
            "failure_reasons": dict(sorted(self.failure_reasons.items())),
        }
