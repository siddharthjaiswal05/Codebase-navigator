"""The tool surface the agent controls.

Seven tools, chosen so that the model decides *how* to look rather than being
walked through a fixed pipeline:

  search_code      hybrid BM25 + dense retrieval, fused with RRF
  expand_context   graph walk from the hybrid seeds under a token budget
  read_chunk       full source of one chunk, for citation before answering
  find_callers     inbound call edges
  find_callees     outbound call edges
  list_symbols     every symbol defined in one file
  grep_repo        literal or regex search, for strings the index cannot rank

Every tool returns a `ToolResult` whose payload is JSON-serialisable, because
observations are fed back to the model as text. Retrieval tools always return
citations alongside content, so the answer step has something verifiable to
quote and the loop cannot invent a location.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any, Callable

from ..index import CodeIndex

MAX_SOURCE_CHARS = 4000


@dataclass
class ToolResult:
    tool: str
    ok: bool
    payload: dict[str, Any] = field(default_factory=dict)
    error: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ToolSpec:
    name: str
    description: str
    arguments: dict[str, str]
    handler: Callable[..., ToolResult]

    def signature(self) -> str:
        args = ", ".join(f"{k}: {v}" for k, v in self.arguments.items())
        return f"{self.name}({args})"


class ToolBox:
    """Binds the tool surface to one index and records every call."""

    def __init__(self, index: CodeIndex, default_token_budget: int = 6000):
        self.index = index
        self.default_token_budget = default_token_budget
        self.call_log: list[dict] = []
        self.last_expansion_trace: dict | None = None
        self._specs = self._build_specs()

    # -- surface ----------------------------------------------------------
    def _build_specs(self) -> dict[str, ToolSpec]:
        specs = [
            ToolSpec(
                "search_code",
                "Hybrid lexical and dense search over the repository. Use first, "
                "and whenever you need new candidates.",
                {"query": "string", "top_k": "int (default 8)"},
                self.search_code,
            ),
            ToolSpec(
                "expand_context",
                "Walk the call graph outward from search seeds under a token "
                "budget. Use when the answer depends on how functions relate, or "
                "when search results look weak.",
                {"query": "string", "token_budget": "int (default 6000)",
                 "max_hops": "int (default 2)"},
                self.expand_context,
            ),
            ToolSpec(
                "read_chunk",
                "Full source of one chunk by id. Read before citing.",
                {"chunk_id": "string"},
                self.read_chunk,
            ),
            ToolSpec(
                "find_callers",
                "Functions that call the given chunk.",
                {"chunk_id": "string"},
                self.find_callers,
            ),
            ToolSpec(
                "find_callees",
                "Functions the given chunk calls.",
                {"chunk_id": "string"},
                self.find_callees,
            ),
            ToolSpec(
                "list_symbols",
                "Every symbol defined in one file, in line order.",
                {"path": "string (repo-relative)"},
                self.list_symbols,
            ),
            ToolSpec(
                "grep_repo",
                "Literal or regex search across source. Use for exact strings, "
                "error messages and config keys that ranking may bury.",
                {"pattern": "string", "max_results": "int (default 20)"},
                self.grep_repo,
            ),
        ]
        return {spec.name: spec for spec in specs}

    def specs(self) -> list[ToolSpec]:
        return list(self._specs.values())

    def describe(self) -> str:
        return "\n".join(
            f"- {spec.signature()}\n    {spec.description}"
            for spec in self._specs.values()
        )

    def names(self) -> list[str]:
        return list(self._specs)

    def call(self, name: str, arguments: dict[str, Any]) -> ToolResult:
        spec = self._specs.get(name)
        if spec is None:
            result = ToolResult(
                name, False,
                error=f"unknown tool '{name}'; available: {', '.join(self._specs)}",
            )
        else:
            try:
                result = spec.handler(**(arguments or {}))
            except TypeError as exc:
                result = ToolResult(name, False, error=f"bad arguments: {exc}")
            except Exception as exc:
                result = ToolResult(name, False, error=f"{type(exc).__name__}: {exc}")

        self.call_log.append(
            {"tool": name, "arguments": arguments, "ok": result.ok,
             "error": result.error}
        )
        return result

    # -- handlers ---------------------------------------------------------
    def search_code(self, query: str, top_k: int = 8) -> ToolResult:
        results = self.index.search_hybrid(query, top_k=int(top_k))
        return ToolResult(
            "search_code", True,
            {
                "query": query,
                "count": len(results),
                "results": [
                    {
                        "chunk_id": r.chunk.chunk_id,
                        "citation": r.citation(),
                        "qualified_name": r.chunk.qualified_name,
                        "kind": r.chunk.kind,
                        "signature": r.chunk.signature,
                        "score": round(r.score, 6),
                        "matched_by": r.sources,
                    }
                    for r in results
                ],
            },
        )

    def expand_context(self, query: str, token_budget: int | None = None,
                       max_hops: int = 2, top_k: int = 12) -> ToolResult:
        budget = int(token_budget or self.default_token_budget)
        results, trace = self.index.search_with_expansion(
            query, top_k=int(top_k), token_budget=budget, max_hops=int(max_hops)
        )
        self.last_expansion_trace = trace.to_dict()
        return ToolResult(
            "expand_context", True,
            {
                "query": query,
                "trace": self.last_expansion_trace,
                "results": [
                    {
                        "chunk_id": r.chunk.chunk_id,
                        "citation": r.citation(),
                        "qualified_name": r.chunk.qualified_name,
                        "hops": r.hops,
                        "reached_via": r.reached_via,
                        "score": round(r.score, 6),
                    }
                    for r in results
                ],
            },
        )

    def read_chunk(self, chunk_id: str) -> ToolResult:
        chunk = self.index.get(chunk_id)
        if chunk is None:
            # Accept a symbol name as a fallback, since a model will sometimes
            # pass one instead of an opaque id.
            matches = self.index.find_by_name(chunk_id)
            if not matches:
                return ToolResult(
                    "read_chunk", False, error=f"no chunk with id or name '{chunk_id}'"
                )
            chunk = matches[0]

        source = chunk.source
        truncated = len(source) > MAX_SOURCE_CHARS
        return ToolResult(
            "read_chunk", True,
            {
                "chunk_id": chunk.chunk_id,
                "citation": chunk.citation,
                "qualified_name": chunk.qualified_name,
                "kind": chunk.kind,
                "signature": chunk.signature,
                "docstring": chunk.docstring,
                "source": source[:MAX_SOURCE_CHARS],
                "truncated": truncated,
            },
        )

    def _edge_tool(self, tool: str, chunk_id: str, ids: list[str]) -> ToolResult:
        chunk = self.index.get(chunk_id)
        if chunk is None:
            return ToolResult(tool, False, error=f"no chunk with id '{chunk_id}'")
        related = []
        for other_id in ids:
            other = self.index.get(other_id)
            if other is not None:
                related.append(
                    {
                        "chunk_id": other.chunk_id,
                        "citation": other.citation,
                        "qualified_name": other.qualified_name,
                        "kind": other.kind,
                    }
                )
        return ToolResult(
            tool, True,
            {"of": chunk.qualified_name, "count": len(related), "results": related},
        )

    def find_callers(self, chunk_id: str) -> ToolResult:
        return self._edge_tool(
            "find_callers", chunk_id, self.index.graph.callers(chunk_id)
        )

    def find_callees(self, chunk_id: str) -> ToolResult:
        return self._edge_tool(
            "find_callees", chunk_id, self.index.graph.callees(chunk_id)
        )

    def list_symbols(self, path: str) -> ToolResult:
        chunks = self.index.chunks_in_file(path)
        if not chunks:
            candidates = [f for f in self.index.files() if path in f]
            if not candidates:
                return ToolResult(
                    "list_symbols", False, error=f"no indexed file matching '{path}'"
                )
            chunks = self.index.chunks_in_file(candidates[0])
            path = candidates[0]
        return ToolResult(
            "list_symbols", True,
            {
                "path": path,
                "count": len(chunks),
                "symbols": [
                    {
                        "chunk_id": c.chunk_id,
                        "citation": c.citation,
                        "kind": c.kind,
                        "qualified_name": c.qualified_name,
                        "signature": c.signature,
                    }
                    for c in chunks
                ],
            },
        )

    def grep_repo(self, pattern: str, max_results: int = 20) -> ToolResult:
        try:
            regex = re.compile(pattern)
        except re.error:
            regex = re.compile(re.escape(pattern))

        matches = []
        for chunk in self.index.chunks:
            for offset, line in enumerate(chunk.source.split("\n")):
                if regex.search(line):
                    matches.append(
                        {
                            "chunk_id": chunk.chunk_id,
                            "citation": f"{chunk.path}:{chunk.start_line + offset}",
                            "qualified_name": chunk.qualified_name,
                            "line": line.strip()[:200],
                        }
                    )
                    if len(matches) >= int(max_results):
                        break
            if len(matches) >= int(max_results):
                break

        return ToolResult(
            "grep_repo", True,
            {"pattern": pattern, "count": len(matches), "results": matches},
        )
