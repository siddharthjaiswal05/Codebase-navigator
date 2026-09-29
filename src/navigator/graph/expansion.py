"""Graph-expansion retrieval.

Hybrid search supplies the seeds. This walks outward along caller and callee
edges to reach code that no text retriever would return, because the target
shares no vocabulary with the question.

Two properties keep the walk useful rather than merely large:

  A token budget, not a depth constant. Expansion stops when the retrieved set
  would exceed the budget the agent allocated, so the traversal adapts to how
  large the surrounding functions actually are.

  Distance decay. A node two hops out is weaker evidence than one hop out, so
  its score is discounted. Without decay a hub function with many callers
  floods the result set.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..chunking.models import Chunk
from .call_graph import CodeGraph


@dataclass
class ExpansionResult:
    chunk_id: str
    score: float
    hops: int
    reached_via: str              # seed | caller | callee | sibling
    from_chunk: str | None = None

    def is_seed(self) -> bool:
        return self.hops == 0


@dataclass
class ExpansionTrace:
    """What the walk did, so the agent can report it and a test can assert it."""

    seeds: int = 0
    expanded: int = 0
    max_hops_reached: int = 0
    tokens_used: int = 0
    token_budget: int = 0
    stopped_on_budget: bool = False
    frontier_exhausted: bool = False
    by_hop: dict[int, int] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "seeds": self.seeds,
            "expanded": self.expanded,
            "max_hops_reached": self.max_hops_reached,
            "tokens_used": self.tokens_used,
            "token_budget": self.token_budget,
            "stopped_on_budget": self.stopped_on_budget,
            "frontier_exhausted": self.frontier_exhausted,
            "by_hop": dict(sorted(self.by_hop.items())),
        }


def estimate_tokens(chunk: Chunk) -> int:
    """Cheap, deterministic token estimate.

    Four characters per token is the usual rule of thumb for code. The budget
    only needs to be approximately right to keep the walk bounded, and a real
    tokenizer would tie the graph layer to a specific model.
    """
    return max(1, len(chunk.source) // 4)


def expand(
    seeds: list[tuple[str, float]],
    graph: CodeGraph,
    chunks_by_id: dict[str, Chunk],
    token_budget: int = 6000,
    max_hops: int = 2,
    decay: float = 0.5,
    include_siblings: bool = False,
    max_neighbours_per_node: int = 12,
) -> tuple[list[ExpansionResult], ExpansionTrace]:
    """Breadth-first walk from scored seeds, bounded by tokens.

    `seeds` is (chunk_id, score) best-first, normally the fused hybrid result.
    Returns results ordered by score, seeds first, and a trace of the walk.
    """
    trace = ExpansionTrace(token_budget=token_budget)
    results: dict[str, ExpansionResult] = {}
    tokens_used = 0

    # Seeds are admitted first and always count against the budget.
    frontier: list[tuple[str, float, int, str | None]] = []
    for chunk_id, score in seeds:
        chunk = chunks_by_id.get(chunk_id)
        if chunk is None or chunk_id in results:
            continue
        cost = estimate_tokens(chunk)
        if tokens_used + cost > token_budget and results:
            trace.stopped_on_budget = True
            break
        tokens_used += cost
        results[chunk_id] = ExpansionResult(chunk_id, score, 0, "seed")
        frontier.append((chunk_id, score, 0, None))
        trace.by_hop[0] = trace.by_hop.get(0, 0) + 1

    trace.seeds = len(results)

    while frontier and not trace.stopped_on_budget:
        chunk_id, score, hops, _ = frontier.pop(0)
        if hops >= max_hops:
            continue

        callees = graph.callees(chunk_id)
        callers = graph.callers(chunk_id)
        sibling_ids = graph.siblings(chunk_id) if include_siblings else []

        candidates: list[tuple[str, str]] = (
            [(cid, "callee") for cid in callees]
            + [(cid, "caller") for cid in callers]
            + [(cid, "sibling") for cid in sibling_ids]
        )[:max_neighbours_per_node]

        for neighbour_id, relation in candidates:
            if neighbour_id in results:
                continue
            chunk = chunks_by_id.get(neighbour_id)
            if chunk is None:
                continue

            cost = estimate_tokens(chunk)
            if tokens_used + cost > token_budget:
                trace.stopped_on_budget = True
                break

            tokens_used += cost
            next_hops = hops + 1
            neighbour_score = score * (decay ** next_hops)

            results[neighbour_id] = ExpansionResult(
                neighbour_id, neighbour_score, next_hops, relation, from_chunk=chunk_id
            )
            frontier.append((neighbour_id, neighbour_score, next_hops, chunk_id))
            trace.by_hop[next_hops] = trace.by_hop.get(next_hops, 0) + 1
            trace.max_hops_reached = max(trace.max_hops_reached, next_hops)

        if trace.stopped_on_budget:
            break

    if not frontier and not trace.stopped_on_budget:
        trace.frontier_exhausted = True

    trace.tokens_used = tokens_used
    trace.expanded = len(results) - trace.seeds

    ordered = sorted(results.values(), key=lambda r: (-r.score, r.hops, r.chunk_id))
    return ordered, trace
