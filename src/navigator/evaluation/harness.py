"""Evaluation over the hand-verified question set.

Three things are measured, and kept apart on purpose:

  Retrieval        does the ranked list contain the right file and the right
                   symbol, at file and function granularity.
  Ablations        BM25 alone, dense alone, fused, and fused plus graph
                   expansion, over the identical question set. Reporting the
                   full system without its components says nothing about which
                   part is doing the work.
  Agent behaviour  does the end-to-end loop produce an answer whose citations
                   survive verification.

Results are broken out by question type, because a system that is strong on
questions sharing vocabulary with the code and weak on questions that do not
has not solved the problem the project is about.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from ..agent.react import CodeNavigatorAgent
from ..index import CodeIndex
from .metrics import CitationAccumulator, MetricAccumulator


@dataclass
class Question:
    id: str
    repo: str
    type: str
    question: str
    gold_files: list[str] = field(default_factory=list)
    gold_symbols: list[str] = field(default_factory=list)


@dataclass
class QuestionSet:
    repos: dict[str, dict]
    questions: list[Question]

    @classmethod
    def load(cls, path: Path) -> "QuestionSet":
        with open(path, encoding="utf-8") as handle:
            data = yaml.safe_load(handle)
        return cls(
            repos=data["repos"],
            questions=[Question(**q) for q in data["questions"]],
        )

    def for_repo(self, repo: str) -> list[Question]:
        return [q for q in self.questions if q.repo == repo]

    def types(self) -> list[str]:
        return sorted({q.type for q in self.questions})


def validate_gold(question_set: QuestionSet, indexes: dict[str, CodeIndex]) -> dict:
    """Confirm every gold answer resolves to something real.

    A question set that drifts out of sync with the code silently turns into
    noise, so this runs before any scoring and reports each break by id.
    """
    problems: list[dict] = []
    for question in question_set.questions:
        index = indexes.get(question.repo)
        if index is None:
            problems.append({"id": question.id, "issue": f"no index for repo '{question.repo}'"})
            continue

        known_files = set(index.files())
        for path in question.gold_files:
            if path not in known_files:
                problems.append({"id": question.id, "issue": f"gold file not indexed: {path}"})

        known_symbols = {c.qualified_name for c in index.chunks}
        for symbol in question.gold_symbols:
            if symbol not in known_symbols:
                problems.append({"id": question.id, "issue": f"gold symbol not indexed: {symbol}"})

    return {
        "questions": len(question_set.questions),
        "problems": len(problems),
        "ok": not problems,
        "details": problems,
    }


# --------------------------------------------------------------------------
# Retrieval ablations
# --------------------------------------------------------------------------

ABLATIONS = ("bm25", "dense", "hybrid", "hybrid+graph")


def _retrieve(index: CodeIndex, query: str, mode: str, top_k: int):
    """Return chunks in rank order under one retrieval configuration."""
    if mode == "bm25":
        ids = [cid for cid, _ in index.search_bm25(query, top_k=top_k)]
        return [index.get(cid) for cid in ids if index.get(cid)]
    if mode == "dense":
        ids = [cid for cid, _ in index.search_dense(query, top_k=top_k)]
        return [index.get(cid) for cid in ids if index.get(cid)]
    if mode == "hybrid":
        return [r.chunk for r in index.search_hybrid(query, top_k=top_k)]
    if mode == "hybrid+graph":
        results, _ = index.search_with_expansion(
            query, top_k=top_k, seed_k=max(3, top_k // 2), token_budget=6000
        )
        return [r.chunk for r in results]
    raise ValueError(f"unknown retrieval mode: {mode}")


def evaluate_retrieval(
    question_set: QuestionSet,
    indexes: dict[str, CodeIndex],
    modes: tuple[str, ...] = ABLATIONS,
    top_k: int = 10,
) -> dict:
    """Score every ablation at both granularities, overall and by question type."""
    report: dict = {"top_k": top_k, "modes": {}}

    for mode in modes:
        file_overall = MetricAccumulator()
        symbol_overall = MetricAccumulator()
        by_type: dict[str, dict[str, MetricAccumulator]] = {}
        per_question: list[dict] = []

        for question in question_set.questions:
            index = indexes.get(question.repo)
            if index is None:
                continue

            chunks = _retrieve(index, question.question, mode, top_k)
            # De-duplicate files while preserving rank order: the first time a
            # file appears is the rank a developer would actually experience.
            ranked_files, seen = [], set()
            for chunk in chunks:
                if chunk.path not in seen:
                    seen.add(chunk.path)
                    ranked_files.append(chunk.path)
            ranked_symbols = [c.qualified_name for c in chunks]

            gold_files = set(question.gold_files)
            gold_symbols = set(question.gold_symbols)

            file_overall.add(ranked_files, gold_files)
            symbol_overall.add(ranked_symbols, gold_symbols)

            slot = by_type.setdefault(
                question.type, {"file": MetricAccumulator(), "symbol": MetricAccumulator()}
            )
            slot["file"].add(ranked_files, gold_files)
            slot["symbol"].add(ranked_symbols, gold_symbols)

            from .metrics import hit_at_k, reciprocal_rank

            per_question.append(
                {
                    "id": question.id,
                    "type": question.type,
                    "repo": question.repo,
                    "file_hit@1": hit_at_k(ranked_files, gold_files, 1),
                    "file_hit@5": hit_at_k(ranked_files, gold_files, 5),
                    "symbol_hit@1": hit_at_k(ranked_symbols, gold_symbols, 1),
                    "symbol_hit@5": hit_at_k(ranked_symbols, gold_symbols, 5),
                    "symbol_mrr": round(reciprocal_rank(ranked_symbols, gold_symbols), 4),
                    "top_symbol": ranked_symbols[0] if ranked_symbols else None,
                }
            )

        report["modes"][mode] = {
            "file_level": file_overall.summary(),
            "function_level": symbol_overall.summary(),
            "by_type": {
                qtype: {
                    "file_level": slots["file"].summary(),
                    "function_level": slots["symbol"].summary(),
                }
                for qtype, slots in sorted(by_type.items())
            },
            "per_question": per_question,
        }

    return report


# --------------------------------------------------------------------------
# End-to-end agent evaluation
# --------------------------------------------------------------------------

def evaluate_agent(
    question_set: QuestionSet,
    indexes: dict[str, CodeIndex],
    llm_backend: str = "auto",
    max_steps: int = 6,
) -> dict:
    """Run the full loop and score answers on citation validity and localisation."""
    from ..agent.llm import build_llm
    from .metrics import hit_at_k

    citations = CitationAccumulator()
    file_acc = MetricAccumulator()
    per_question: list[dict] = []
    tool_usage: dict[str, int] = {}
    total_steps = 0
    llm_calls = 0

    agents = {
        repo: CodeNavigatorAgent(
            index, llm=build_llm(llm_backend), max_steps=max_steps
        )
        for repo, index in indexes.items()
    }

    for question in question_set.questions:
        agent = agents.get(question.repo)
        if agent is None:
            continue

        answer = agent.answer(question.question)
        citations.add(answer.citations)

        cited_paths = answer.cited_paths()
        gold_files = set(question.gold_files)
        file_acc.add(cited_paths, gold_files)

        for tool in answer.tools_used:
            tool_usage[tool] = tool_usage.get(tool, 0) + 1
        total_steps += len(answer.steps)
        llm_calls += answer.llm_calls

        per_question.append(
            {
                "id": question.id,
                "type": question.type,
                "steps": len(answer.steps),
                "tools_used": answer.tools_used,
                "citations": len(answer.citations),
                "valid_citations": len(answer.valid_citations),
                "citation_validity": round(answer.citation_validity, 4),
                "supported": answer.is_supported,
                "cited_gold_file": bool(hit_at_k(cited_paths, gold_files, 10)),
                "stopped_reason": answer.stopped_reason,
            }
        )

    n = len(per_question) or 1
    return {
        "citations": citations.summary(),
        "answer_file_localisation": file_acc.summary(),
        "cited_gold_file_rate": round(
            sum(1 for q in per_question if q["cited_gold_file"]) / n, 4
        ),
        "mean_steps": round(total_steps / n, 2),
        "llm_calls": llm_calls,
        "tool_usage": dict(sorted(tool_usage.items(), key=lambda kv: -kv[1])),
        "per_question": per_question,
    }


def build_indexes(question_set: QuestionSet, base_dir: Path,
                  encoder: str = "auto") -> dict[str, CodeIndex]:
    """Index every repository the question set references."""
    indexes: dict[str, CodeIndex] = {}
    for name, spec in question_set.repos.items():
        root = (base_dir / spec["root"]).resolve()
        if not root.exists():
            continue
        indexes[name] = CodeIndex.build(root, encoder_preference=encoder)
    return indexes


def write_report(report: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, default=str)
