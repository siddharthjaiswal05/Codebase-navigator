"""Call and module-dependency graphs.

Retrieval that only matches text can never find a function whose body shares no
vocabulary with the question. The graph is what makes such a function reachable:
it is retrieved because of how it relates to code that did match, not because of
what it says.

Call sites are resolved in three passes, cheapest first:

  1. exact match against a qualified name
  2. import-table resolution, which turns `from pkg.mod import helper` plus a
     call to `helper(...)` into an edge to `pkg.mod.helper`
  3. scope-aware suffix matching, which handles `self.method(...)` and
     `module.function(...)`

jedi refines the residue. It is precise but costs a subprocess-grade analysis
per call site, so it runs only on sites the cheap passes left unresolved and
under an explicit budget. `resolution_stats()` reports what each pass settled,
so the graph's reliability is measurable rather than assumed.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import networkx as nx

from ..chunking.models import Chunk

CALLS = "calls"
CONTAINS = "contains"
IMPORTS = "imports"


@dataclass
class ResolutionStats:
    """Resolution outcomes, with external calls kept out of the failure count.

    A call to `len()` or `Path()` has no in-repo target, so counting it as a
    resolution failure would understate the graph. `external` holds those, and
    the headline rate is measured against in-repo call sites only. `ambiguous`
    holds sites where several same-named symbols were plausible and none was
    chosen, because inventing an edge is worse than omitting one.
    """

    total_call_sites: int = 0
    external: int = 0
    resolved_exact: int = 0
    resolved_import: int = 0
    resolved_suffix: int = 0
    resolved_jedi: int = 0
    ambiguous: int = 0
    unresolved: int = 0
    jedi_attempted: int = 0
    jedi_available: bool = False

    @property
    def resolved(self) -> int:
        return (
            self.resolved_exact
            + self.resolved_import
            + self.resolved_suffix
            + self.resolved_jedi
        )

    @property
    def in_repo_call_sites(self) -> int:
        """Call sites whose target could plausibly live in this repository."""
        return self.total_call_sites - self.external

    @property
    def resolution_rate(self) -> float:
        """Resolved share of in-repo call sites."""
        denominator = self.in_repo_call_sites
        return self.resolved / denominator if denominator else 0.0

    def to_dict(self) -> dict:
        return {
            "total_call_sites": self.total_call_sites,
            "external_call_sites": self.external,
            "in_repo_call_sites": self.in_repo_call_sites,
            "resolved_exact": self.resolved_exact,
            "resolved_import": self.resolved_import,
            "resolved_suffix": self.resolved_suffix,
            "resolved_jedi": self.resolved_jedi,
            "ambiguous": self.ambiguous,
            "unresolved": self.unresolved,
            "jedi_attempted": self.jedi_attempted,
            "jedi_available": self.jedi_available,
            "resolution_rate": round(self.resolution_rate, 4),
        }


@dataclass
class CodeGraph:
    """Chunks as nodes, structural relationships as typed edges."""

    graph: nx.MultiDiGraph = field(default_factory=nx.MultiDiGraph)
    module_graph: nx.DiGraph = field(default_factory=nx.DiGraph)
    stats: ResolutionStats = field(default_factory=ResolutionStats)

    # -- queries ----------------------------------------------------------
    def callees(self, chunk_id: str) -> list[str]:
        """What this chunk calls."""
        if chunk_id not in self.graph:
            return []
        return sorted({
            target
            for _, target, data in self.graph.out_edges(chunk_id, data=True)
            if data.get("type") == CALLS
        })

    def callers(self, chunk_id: str) -> list[str]:
        """What calls this chunk."""
        if chunk_id not in self.graph:
            return []
        return sorted({
            source
            for source, _, data in self.graph.in_edges(chunk_id, data=True)
            if data.get("type") == CALLS
        })

    def siblings(self, chunk_id: str) -> list[str]:
        """Other members of the same enclosing class or module."""
        if chunk_id not in self.graph:
            return []
        parents = [
            source
            for source, _, data in self.graph.in_edges(chunk_id, data=True)
            if data.get("type") == CONTAINS
        ]
        out: set[str] = set()
        for parent in parents:
            for _, child, data in self.graph.out_edges(parent, data=True):
                if data.get("type") == CONTAINS and child != chunk_id:
                    out.add(child)
        return sorted(out)

    def neighbours(self, chunk_id: str, include_siblings: bool = False) -> list[str]:
        out = set(self.callees(chunk_id)) | set(self.callers(chunk_id))
        if include_siblings:
            out |= set(self.siblings(chunk_id))
        return sorted(out)

    def degree(self, chunk_id: str) -> int:
        return len(self.callers(chunk_id)) + len(self.callees(chunk_id))

    def resolution_stats(self) -> dict:
        return self.stats.to_dict()

    def summary(self) -> dict:
        call_edges = sum(
            1 for _, _, d in self.graph.edges(data=True) if d.get("type") == CALLS
        )
        return {
            "nodes": self.graph.number_of_nodes(),
            "call_edges": call_edges,
            "contains_edges": sum(
                1 for _, _, d in self.graph.edges(data=True)
                if d.get("type") == CONTAINS
            ),
            "modules": self.module_graph.number_of_nodes(),
            "module_edges": self.module_graph.number_of_edges(),
            **self.resolution_stats(),
        }


# --------------------------------------------------------------------------
# Construction
# --------------------------------------------------------------------------

def _resolve_relative(module_token: str, current_module: str) -> str:
    """Turn a relative import target into an absolute dotted path.

    `from .models import Chunk` inside `navigator.chunking.ast_chunker` refers
    to `navigator.chunking.models`. One leading dot means the current package,
    each additional dot climbs one level. Without this, every relative import
    in a `src/` layout resolves to a path that matches nothing.
    """
    dots = len(module_token) - len(module_token.lstrip("."))
    if dots == 0:
        return module_token

    package_parts = current_module.split(".")[:-1]      # drop the module itself
    climb = dots - 1
    if climb:
        package_parts = package_parts[:-climb] if climb <= len(package_parts) else []

    remainder = module_token[dots:]
    parts = package_parts + ([remainder] if remainder else [])
    return ".".join(p for p in parts if p)


def _parse_imports(import_statements: list[str],
                   current_module: str = "") -> dict[str, str]:
    """Map a locally visible name to the dotted path it refers to.

    `from pkg.mod import helper as h` gives {h: pkg.mod.helper}
    `import pkg.mod as m`             gives {m: pkg.mod}
    Relative targets are resolved against `current_module`.
    """
    table: dict[str, str] = {}
    for statement in import_statements:
        tokens = statement.split()
        if not tokens:
            continue

        if tokens[0] == "from" and "import" in tokens:
            module = _resolve_relative(tokens[1], current_module)
            rest = " ".join(tokens[tokens.index("import") + 1:])
            for piece in rest.split(","):
                piece = piece.strip().strip("()").strip()
                if not piece:
                    continue
                parts = piece.split()
                name = parts[0]
                alias = parts[2] if len(parts) >= 3 and parts[1] == "as" else name
                table[alias] = f"{module}.{name}" if module else name

        elif tokens[0] == "import":
            rest = " ".join(tokens[1:])
            for piece in rest.split(","):
                piece = piece.strip()
                if not piece:
                    continue
                parts = piece.split()
                dotted = parts[0]
                alias = parts[2] if len(parts) >= 3 and parts[1] == "as" else dotted
                table[alias] = dotted

    return table


def _module_of(qualified_name: str, name: str) -> str:
    """Strip the trailing symbol (and enclosing class) off a qualified name."""
    parts = qualified_name.split(".")
    if parts and parts[-1] == name:
        parts = parts[:-1]
    return ".".join(parts)


def build_code_graph(
    chunks: list[Chunk],
    repo_root: Path | None = None,
    use_jedi: bool = True,
    jedi_budget: int = 300,
) -> CodeGraph:
    """Build the call and module graphs over a chunk set."""
    cg = CodeGraph()
    stats = cg.stats

    by_id = {c.chunk_id: c for c in chunks}
    by_qualified: dict[str, str] = {}
    by_name: dict[str, list[str]] = defaultdict(list)

    for chunk in chunks:
        cg.graph.add_node(
            chunk.chunk_id,
            path=chunk.path,
            name=chunk.name,
            kind=chunk.kind,
            qualified_name=chunk.qualified_name,
            start_line=chunk.start_line,
            end_line=chunk.end_line,
        )
        by_qualified.setdefault(chunk.qualified_name, chunk.chunk_id)
        by_name[chunk.name].append(chunk.chunk_id)

    # Containment: class to method, so siblings are reachable.
    for chunk in chunks:
        if chunk.parent and chunk.parent in by_qualified:
            cg.graph.add_edge(
                by_qualified[chunk.parent], chunk.chunk_id, type=CONTAINS
            )

    # Module dependency graph, independent of call resolution.
    for chunk in chunks:
        module = chunk.module
        if not module:
            continue
        cg.module_graph.add_node(module, path=chunk.path)
        for target in _parse_imports(chunk.imports, module).values():
            root = target.split(".")[0]
            if root and root != module:
                cg.module_graph.add_edge(module, target, via=root)

    deferred: list[tuple[Chunk, str]] = []

    for chunk in chunks:
        module = chunk.module
        import_table = _parse_imports(chunk.imports, module)
        enclosing = chunk.parent or module

        for raw_call in chunk.calls:
            stats.total_call_sites += 1
            target_id = None

            base = raw_call.split("(")[0].strip()
            if not base:
                stats.unresolved += 1
                continue

            # A call whose leaf name matches nothing in the repository targets
            # a builtin or a third-party package. That is not a failure to
            # resolve, so it is counted separately and skipped.
            if base.split(".")[-1] not in by_name:
                stats.external += 1
                continue

            # Pass 1: the call text already is a qualified name.
            if base in by_qualified:
                target_id = by_qualified[base]
                stats.resolved_exact += 1

            # Pass 2: import table.
            if target_id is None:
                head, _, tail = base.partition(".")
                if head in import_table:
                    candidate = import_table[head] + (f".{tail}" if tail else "")
                    if candidate in by_qualified:
                        target_id = by_qualified[candidate]
                        stats.resolved_import += 1

            # Pass 3: scope-aware suffix match.
            if target_id is None:
                leaf = base.split(".")[-1]
                candidates = by_name.get(leaf, [])
                if candidates:
                    if base.startswith("self."):
                        # Prefer a method of the same class.
                        same_class = [
                            cid for cid in candidates
                            if by_id[cid].parent == enclosing
                        ]
                        candidates = same_class or candidates
                    same_module = [
                        cid for cid in candidates if by_id[cid].module == module
                    ]
                    pool = same_module or candidates
                    # Only accept an unambiguous target; guessing between many
                    # same-named symbols would add edges that are not real.
                    if len(pool) == 1:
                        target_id = pool[0]
                        stats.resolved_suffix += 1

            if target_id is None:
                if len(by_name.get(base.split(".")[-1], [])) > 1:
                    stats.ambiguous += 1
                deferred.append((chunk, base))
                continue

            if target_id != chunk.chunk_id:
                cg.graph.add_edge(
                    chunk.chunk_id, target_id, type=CALLS, call=base
                )

    # Pass 4: jedi, on the residue only and under budget.
    if use_jedi and repo_root is not None and deferred:
        resolved = _resolve_with_jedi(
            deferred, by_qualified, repo_root, cg, budget=jedi_budget, stats=stats
        )
        stats.unresolved += len(deferred) - resolved
    else:
        stats.unresolved += len(deferred)

    return cg


def _resolve_with_jedi(
    deferred: list[tuple[Chunk, str]],
    by_qualified: dict[str, str],
    repo_root: Path,
    cg: CodeGraph,
    budget: int,
    stats: ResolutionStats,
) -> int:
    """Refine unresolved call sites with jedi's inference, under a budget."""
    try:
        import jedi
    except Exception:
        return 0

    stats.jedi_available = True
    project = jedi.Project(path=str(repo_root))
    source_cache: dict[str, str] = {}
    resolved_count = 0

    for chunk, base in deferred[:budget]:
        stats.jedi_attempted += 1
        leaf = base.split(".")[-1]
        try:
            full_path = repo_root / chunk.path
            source = source_cache.get(chunk.path)
            if source is None:
                source = full_path.read_text(encoding="utf-8", errors="replace")
                source_cache[chunk.path] = source

            script = jedi.Script(code=source, path=str(full_path), project=project)
            # Search the chunk's own line span for the call site.
            hit = None
            lines = source.split("\n")
            for lineno in range(chunk.start_line, min(chunk.end_line, len(lines)) + 1):
                line = lines[lineno - 1]
                column = line.find(leaf)
                if column == -1:
                    continue
                try:
                    definitions = script.goto(
                        lineno, column + 1, follow_imports=True
                    )
                except Exception:
                    continue
                for definition in definitions:
                    if definition.full_name and definition.full_name in by_qualified:
                        hit = by_qualified[definition.full_name]
                        break
                    if definition.name in by_qualified:
                        hit = by_qualified[definition.name]
                        break
                if hit:
                    break

            if hit and hit != chunk.chunk_id:
                cg.graph.add_edge(chunk.chunk_id, hit, type=CALLS, call=base,
                                  resolver="jedi")
                stats.resolved_jedi += 1
                resolved_count += 1
        except Exception:
            continue

    return resolved_count
