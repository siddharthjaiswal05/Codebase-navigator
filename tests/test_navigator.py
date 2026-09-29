"""Behavioural tests for the navigator.

These assert the properties the design actually depends on, not line coverage:
that chunks are syntactically whole, that the tokenizer closes the camelCase
and snake_case gap, that fusion is rank-based, that graph expansion respects
its budget and reaches code no text retriever would, and that a bad citation is
rejected.

Runs standalone or under pytest:

    python3 tests/test_navigator.py
    pytest tests/test_navigator.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from navigator.agent import CodeNavigatorAgent, ToolBox, build_llm  # noqa: E402
from navigator.agent.cache import ResponseCache, prompt_hash  # noqa: E402
from navigator.agent.llm import extract_json  # noqa: E402
from navigator.chunking import chunk_repository  # noqa: E402
from navigator.evaluation.metrics import (  # noqa: E402
    hit_at_k, reciprocal_rank, recall_at_k,
)
from navigator.evaluation.swebench import parse_patch  # noqa: E402
from navigator.graph.call_graph import _parse_imports, _resolve_relative  # noqa: E402
from navigator.graph.expansion import expand  # noqa: E402
from navigator.index import CodeIndex  # noqa: E402
from navigator.retrieval.fusion import reciprocal_rank_fusion  # noqa: E402
from navigator.retrieval.tokenize import split_identifier, tokenize_query  # noqa: E402

SRC = ROOT / "src"
_INDEX: CodeIndex | None = None


def index() -> CodeIndex:
    """Built once; every test shares it."""
    global _INDEX
    if _INDEX is None:
        _INDEX = CodeIndex.build(SRC, encoder_preference="tfidf-svd")
    return _INDEX


# --- chunking -----------------------------------------------------------

def test_chunks_are_syntactically_whole():
    """The property that makes a citation trustworthy."""
    for chunk in chunk_repository(SRC):
        if chunk.kind in ("function", "method"):
            first = chunk.source.lstrip().split("\n")[0]
            assert first.startswith(("def ", "async def ", "@")), chunk.qualified_name


def test_chunk_line_spans_match_the_file():
    """A citation is only as good as its line span."""
    for chunk in chunk_repository(SRC)[:60]:
        lines = (SRC / chunk.path).read_text(encoding="utf-8").split("\n")
        assert 1 <= chunk.start_line <= chunk.end_line <= len(lines)
        assert lines[chunk.start_line - 1] == chunk.source.split("\n")[0]


def test_every_chunk_knows_its_module():
    assert all(c.module for c in chunk_repository(SRC))


def test_methods_are_distinguished_from_functions():
    chunks = chunk_repository(SRC)
    kinds = {c.kind for c in chunks}
    assert {"function", "method", "class"} <= kinds
    for chunk in chunks:
        if chunk.kind == "method":
            assert chunk.parent is not None


def test_no_chunk_is_empty():
    assert all(c.source.strip() for c in chunk_repository(SRC))


# --- tokenization -------------------------------------------------------

def test_camel_and_snake_converge_on_the_same_parts():
    """The whole point of identifier-level tokenization."""
    camel = set(split_identifier("parseConfigFile"))
    snake = set(split_identifier("parse_config_file"))
    assert {"parse", "config", "file"} <= camel
    assert {"parse", "config", "file"} <= snake


def test_acronym_runs_split_correctly():
    parts = split_identifier("HTTPServerHandler")
    assert "http" in parts and "server" in parts and "handler" in parts


def test_intact_identifier_is_kept():
    assert "parseconfigfile" in split_identifier("parseConfigFile")


def test_query_tokenization_drops_filler():
    tokens = set(tokenize_query("how does the parser handle a config file?"))
    assert "the" not in tokens and "does" not in tokens
    assert {"parser", "handle", "config", "file"} <= tokens


# --- fusion -------------------------------------------------------------

def test_fusion_rewards_agreement_between_arms():
    fused = reciprocal_rank_fusion({
        "bm25": [("a", 99.0), ("b", 98.0)],
        "dense": [("b", 0.9), ("c", 0.8)],
    })
    assert fused[0].doc_id == "b", "found by both arms should win"
    assert fused[0].found_by_both()


def test_fusion_ignores_raw_score_magnitude():
    """Rank-based by construction: rescaling one arm must not change the order."""
    first = reciprocal_rank_fusion({
        "bm25": [("a", 1000.0), ("b", 1.0)],
        "dense": [("b", 0.9), ("a", 0.1)],
    })
    second = reciprocal_rank_fusion({
        "bm25": [("a", 0.002), ("b", 0.001)],
        "dense": [("b", 0.9), ("a", 0.1)],
    })
    assert [r.doc_id for r in first] == [r.doc_id for r in second]


def test_fusion_weights_shift_the_result():
    weighted = reciprocal_rank_fusion(
        {"bm25": [("a", 1.0), ("b", 1.0)], "dense": [("b", 1.0), ("a", 1.0)]},
        weights={"bm25": 5.0, "dense": 1.0},
    )
    assert weighted[0].doc_id == "a"


# --- import and call resolution ----------------------------------------

def test_relative_imports_resolve_against_the_package():
    assert _resolve_relative(".models", "navigator.chunking.ast_chunker") == \
        "navigator.chunking.models"
    assert _resolve_relative("..index", "navigator.graph.call_graph") == \
        "navigator.index"
    assert _resolve_relative("os.path", "navigator.index") == "os.path"


def test_import_table_handles_aliases():
    table = _parse_imports(
        ["from pkg.mod import helper as h", "import numpy as np"], "some.module"
    )
    assert table["h"] == "pkg.mod.helper"
    assert table["np"] == "numpy"


def test_call_resolution_separates_external_from_failed():
    """A call to len() is not a resolution failure."""
    stats = index().graph.stats
    assert stats.external > 0
    assert stats.in_repo_call_sites == stats.total_call_sites - stats.external
    assert stats.resolution_rate > 0.7, stats.to_dict()


def test_graph_has_both_call_and_containment_edges():
    summary = index().graph.summary()
    assert summary["call_edges"] > 0
    assert summary["contains_edges"] > 0


def test_callers_and_callees_are_inverse():
    graph = index().graph
    for chunk in index().chunks[:40]:
        for callee in graph.callees(chunk.chunk_id):
            assert chunk.chunk_id in graph.callers(callee)


# --- graph expansion ----------------------------------------------------

def test_expansion_respects_the_token_budget():
    idx = index()
    seeds = [(r.chunk.chunk_id, r.score) for r in idx.search_hybrid("index", top_k=5)]
    _, trace = expand(seeds, idx.graph, idx.by_id, token_budget=1200, max_hops=2)
    assert trace.tokens_used <= 1200


def test_larger_budget_reaches_more_code():
    idx = index()
    seeds = [(r.chunk.chunk_id, r.score) for r in idx.search_hybrid("search", top_k=5)]
    small, _ = expand(seeds, idx.graph, idx.by_id, token_budget=1500, max_hops=2)
    large, _ = expand(seeds, idx.graph, idx.by_id, token_budget=12000, max_hops=2)
    assert len(large) >= len(small)


def test_expansion_reaches_code_with_no_query_term_overlap():
    """The reason the graph exists at all."""
    idx = index()
    question = "how is the token budget enforced during traversal?"
    terms = set(tokenize_query(question))
    results, _ = idx.search_with_expansion(
        question, top_k=12, seed_k=5, token_budget=4000
    )
    zero_overlap = [
        r for r in results
        if r.hops > 0 and not (terms & set(tokenize_query(r.chunk.retrieval_text())))
    ]
    assert zero_overlap, "graph expansion returned nothing beyond text matches"


def test_expanded_results_score_below_their_seeds():
    idx = index()
    results, _ = idx.search_with_expansion("fusion", top_k=15, token_budget=8000)
    seeds = [r.score for r in results if r.hops == 0]
    hops = [r.score for r in results if r.hops > 0]
    if seeds and hops:
        assert max(hops) <= max(seeds)


# --- agent --------------------------------------------------------------

def test_tool_surface_is_exactly_seven():
    assert len(ToolBox(index()).names()) == 7


def test_every_tool_runs():
    box = ToolBox(index())
    chunk_id = index().chunks[0].chunk_id
    path = index().files()[0]
    calls = {
        "search_code": {"query": "fusion", "top_k": 3},
        "expand_context": {"query": "fusion", "token_budget": 2000},
        "read_chunk": {"chunk_id": chunk_id},
        "find_callers": {"chunk_id": chunk_id},
        "find_callees": {"chunk_id": chunk_id},
        "list_symbols": {"path": path},
        "grep_repo": {"pattern": "def ", "max_results": 3},
    }
    assert set(calls) == set(box.names())
    for name, arguments in calls.items():
        assert box.call(name, arguments).ok, name


def test_unknown_tool_fails_without_raising():
    result = ToolBox(index()).call("no_such_tool", {})
    assert not result.ok and "unknown tool" in result.error


def test_agent_answers_with_verified_citations():
    agent = CodeNavigatorAgent(index(), llm=build_llm("scripted"))
    answer = agent.answer("How are two ranked lists combined?")
    assert answer.citations
    assert answer.is_supported
    assert answer.citation_validity == 1.0


def test_traversal_depth_varies_with_the_question():
    """The claim that the model picks depth, rather than a fixed pipeline."""
    agent = CodeNavigatorAgent(index(), llm=build_llm("scripted"))
    relational = agent.answer("Which functions call the fusion helper?")
    direct = agent.answer("Where is BM25 scoring implemented?")
    assert "expand_context" in relational.tools_used
    assert len(relational.steps) != len(direct.steps)


def test_bad_citations_are_rejected():
    agent = CodeNavigatorAgent(index(), llm=build_llm("scripted"))
    assert not agent.verify_citation("navigator/nope.py:1-2").valid
    assert not agent.verify_citation("garbage").valid
    assert not agent.verify_citation("navigator/index.py:99999-100000").valid


def test_a_real_span_is_accepted():
    chunk = index().chunks[0]
    citation = agent_citation(chunk)
    assert citation.valid, citation.reason


def agent_citation(chunk):
    agent = CodeNavigatorAgent(index(), llm=build_llm("scripted"))
    return agent.verify_citation(f"{chunk.path}:{chunk.start_line}-{chunk.end_line}")


# --- caching and provider handling --------------------------------------

def test_cache_round_trips_and_counts_hits(tmp_path=None):
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        cache = ResponseCache(Path(tmp) / "c.db")
        key = prompt_hash("m", [{"role": "user", "content": "hi"}], temperature=0.0)
        assert cache.get(key) is None
        cache.put(key, "m", {"messages": []}, {"text": "hello"})
        assert cache.get(key)["text"] == "hello"
        assert cache.stats()["session_hits"] == 1
        cache.close()


def test_prompt_hash_is_sensitive_to_every_input():
    base = prompt_hash("m", [{"role": "user", "content": "a"}], temperature=0.0)
    assert base != prompt_hash("other", [{"role": "user", "content": "a"}], temperature=0.0)
    assert base != prompt_hash("m", [{"role": "user", "content": "b"}], temperature=0.0)
    assert base != prompt_hash("m", [{"role": "user", "content": "a"}], temperature=0.7)


def test_json_is_recovered_from_messy_model_output():
    assert extract_json('{"tool": "search_code"}')["tool"] == "search_code"
    assert extract_json('```json\n{"tool": "a"}\n```')["tool"] == "a"
    assert extract_json('Sure! {"tool": "b"} hope that helps')["tool"] == "b"
    assert extract_json("no json here") is None


# --- metrics and SWE-bench parsing --------------------------------------

def test_metric_edge_cases():
    assert hit_at_k(["a"], {"a"}, 1) == 1.0
    assert hit_at_k(["b"], {"a"}, 1) == 0.0
    assert hit_at_k([], set(), 5) == 0.0
    assert reciprocal_rank(["x", "y", "a"], {"a"}) == 1 / 3
    assert recall_at_k(["a", "b"], {"a", "b"}, 2) == 1.0


def test_patch_parsing_extracts_files_and_line_ranges():
    patch = (
        "diff --git a/src/mod.py b/src/mod.py\n"
        "--- a/src/mod.py\n"
        "+++ b/src/mod.py\n"
        "@@ -10,3 +10,4 @@\n"
        "+new line\n"
        "--- a/src/other.py\n"
        "+++ b/src/other.py\n"
        "@@ -1,2 +5,2 @@\n"
    )
    targets = {t.path: t for t in parse_patch(patch)}
    assert set(targets) == {"src/mod.py", "src/other.py"}
    assert targets["src/mod.py"].overlaps(10, 13)
    assert not targets["src/mod.py"].overlaps(100, 200)


# --- persistence --------------------------------------------------------

def test_index_survives_a_save_and_load_round_trip():
    import tempfile

    idx = index()
    with tempfile.TemporaryDirectory() as tmp:
        idx.save(Path(tmp))
        restored = CodeIndex.load(Path(tmp))
        assert len(restored.chunks) == len(idx.chunks)
        before = [r.chunk.qualified_name for r in idx.search_hybrid("fusion", top_k=5)]
        after = [r.chunk.qualified_name for r in restored.search_hybrid("fusion", top_k=5)]
        assert before == after


def _main() -> int:
    tests = [(n, o) for n, o in sorted(globals().items())
             if n.startswith("test_") and callable(o)]
    failed = []
    for name, fn in tests:
        try:
            fn()
            print(f"  pass  {name}")
        except AssertionError as exc:
            failed.append(name)
            print(f"  FAIL  {name}: {exc}")
        except Exception as exc:
            failed.append(name)
            print(f"  ERROR {name}: {type(exc).__name__}: {exc}")
    print(f"\n{len(tests) - len(failed)}/{len(tests)} tests passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(_main())
