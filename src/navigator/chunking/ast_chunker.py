"""AST-based chunk extraction.

Conventional RAG cuts source on a token window, which splits functions in half
and yields chunks that cannot be cited or read. This module walks a real parse
tree instead and emits functions, methods and classes as atomic units, each
annotated with its signature, enclosing scope, import table and line span.

Two backends produce identical `Chunk` output:

  tree-sitter   the primary parser, and the one the design targets
  stdlib ast    a zero-dependency fallback, so the system still indexes a
                repository on a machine where tree-sitter is unavailable

`which_backend()` reports which one is live, and the index records it, so a
result set always says how its chunks were produced.
"""

from __future__ import annotations

import ast as py_ast
import os
from pathlib import Path
from typing import Iterator

from .models import Chunk, make_chunk_id

try:  # pragma: no cover - import guard
    import tree_sitter_python as tsp
    from tree_sitter import Language, Parser

    _TS_LANGUAGE = Language(tsp.language())
    _TS_AVAILABLE = True
except Exception:  # pragma: no cover
    _TS_AVAILABLE = False

SKIP_DIRS = {
    ".git", "__pycache__", ".venv", "venv", "node_modules", ".mypy_cache",
    ".pytest_cache", "build", "dist", ".eggs", ".tox", ".idea", ".ruff_cache",
}


def which_backend() -> str:
    return "tree-sitter" if _TS_AVAILABLE else "stdlib-ast"


# --------------------------------------------------------------------------
# File discovery
# --------------------------------------------------------------------------

def iter_source_files(root: Path, extensions: tuple[str, ...] = (".py",)) -> Iterator[Path]:
    """Every source file under `root`, skipping vendored and generated trees."""
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        for filename in sorted(filenames):
            if filename.endswith(extensions):
                yield Path(dirpath) / filename


def module_name_for(path: Path, root: Path) -> str:
    """Dotted module path, so qualified names match how code imports itself."""
    rel = path.relative_to(root).with_suffix("")
    parts = [p for p in rel.parts if p != "__init__"]
    # A `src/` layout is a packaging detail, not part of the import path.
    if parts and parts[0] == "src":
        parts = parts[1:]
    return ".".join(parts) if parts else rel.stem


# --------------------------------------------------------------------------
# tree-sitter backend
# --------------------------------------------------------------------------

_DEF_NODES = {"function_definition", "class_definition"}


def _ts_text(node, src: bytes) -> str:
    return src[node.start_byte:node.end_byte].decode("utf-8", errors="replace")


def _ts_child_field(node, field: str):
    return node.child_by_field_name(field)


def _ts_signature(node, src: bytes) -> str:
    """Everything up to the body, collapsed to one line."""
    body = _ts_child_field(node, "body")
    end = body.start_byte if body else node.end_byte
    text = src[node.start_byte:end].decode("utf-8", errors="replace")
    text = text.rstrip().rstrip(":")
    return " ".join(text.split())


def _ts_docstring(node, src: bytes) -> str:
    body = _ts_child_field(node, "body")
    if not body or not body.children:
        return ""
    first = body.children[0]
    if first.type == "expression_statement" and first.children:
        literal = first.children[0]
        if literal.type == "string":
            raw = _ts_text(literal, src)
            return raw.strip("\"'").strip()
    return ""


def _ts_collect_calls(node, src: bytes) -> list[str]:
    """Call targets inside a node, not descending into nested definitions.

    Nested definitions get their own chunk and their own call list, so counting
    their calls here would attribute a child's edges to its parent.
    """
    found: list[str] = []
    root_id = node.id

    def visit(n) -> None:
        if n.id != root_id and n.type in _DEF_NODES:
            return
        if n.type == "call":
            fn = _ts_child_field(n, "function")
            if fn is not None:
                found.append(_ts_text(fn, src).strip())
        for child in n.children:
            visit(child)

    visit(node)
    # Preserve first-seen order while de-duplicating.
    return list(dict.fromkeys(found))


def _ts_decorators(node, src: bytes) -> list[str]:
    parent = node.parent
    if parent is None or parent.type != "decorated_definition":
        return []
    out = []
    for child in parent.children:
        if child.type == "decorator":
            out.append(_ts_text(child, src).strip())
    return out


def _ts_imports(root, src: bytes) -> list[str]:
    out: list[str] = []

    def visit(n) -> None:
        if n.type in ("import_statement", "import_from_statement"):
            out.append(" ".join(_ts_text(n, src).split()))
        for child in n.children:
            visit(child)

    visit(root)
    return out


def _ts_chunks(path: Path, root_dir: Path, source: str) -> list[Chunk]:
    parser = Parser(_TS_LANGUAGE)
    src = source.encode("utf-8")
    tree = parser.parse(src)
    rel = str(path.relative_to(root_dir))
    module = module_name_for(path, root_dir)
    imports = _ts_imports(tree.root_node, src)

    chunks: list[Chunk] = []

    def visit(node, scope: list[str], parent_qual: str | None) -> None:
        if node.type in _DEF_NODES:
            name_node = _ts_child_field(node, "name")
            name = _ts_text(name_node, src) if name_node else "<anonymous>"
            qualified = ".".join([module] + scope + [name])

            # A def whose nearest enclosing scope is a class is a method.
            kind = "class" if node.type == "class_definition" else (
                "method" if scope else "function"
            )

            start = node.start_point[0] + 1
            end = node.end_point[0] + 1
            deco = _ts_decorators(node, src)
            if deco:
                # Decorators belong to the definition they wrap, so the chunk
                # starts at the first decorator and the citation covers it.
                start = node.parent.start_point[0] + 1

            chunks.append(
                Chunk(
                    chunk_id=make_chunk_id(rel, qualified, start),
                    path=rel,
                    kind=kind,
                    name=name,
                    qualified_name=qualified,
                    module=module,
                    start_line=start,
                    end_line=end,
                    source="\n".join(source.split("\n")[start - 1:end]),
                    signature=_ts_signature(node, src),
                    docstring=_ts_docstring(node, src),
                    parent=parent_qual,
                    imports=imports,
                    calls=_ts_collect_calls(node, src),
                    decorators=deco,
                )
            )

            inner_scope = scope + [name]
            for child in node.children:
                visit(child, inner_scope, qualified)
            return

        for child in node.children:
            visit(child, scope, parent_qual)

    visit(tree.root_node, [], None)
    return chunks


# --------------------------------------------------------------------------
# stdlib ast backend
# --------------------------------------------------------------------------

def _py_signature(node: py_ast.AST, lines: list[str]) -> str:
    start = node.lineno - 1
    end = start
    for i in range(start, min(start + 25, len(lines))):
        if lines[i].rstrip().endswith(":"):
            end = i
            break
    return " ".join("\n".join(lines[start:end + 1]).split()).rstrip(":")


def _py_calls(node: py_ast.AST) -> list[str]:
    found: list[str] = []
    for child in py_ast.walk(node):
        if isinstance(child, py_ast.Call):
            fn = child.func
            if isinstance(fn, py_ast.Name):
                found.append(fn.id)
            elif isinstance(fn, py_ast.Attribute):
                parts, cur = [fn.attr], fn.value
                while isinstance(cur, py_ast.Attribute):
                    parts.append(cur.attr)
                    cur = cur.value
                if isinstance(cur, py_ast.Name):
                    parts.append(cur.id)
                found.append(".".join(reversed(parts)))
    return list(dict.fromkeys(found))


def _py_chunks(path: Path, root_dir: Path, source: str) -> list[Chunk]:
    try:
        tree = py_ast.parse(source)
    except SyntaxError:
        return []

    lines = source.split("\n")
    rel = str(path.relative_to(root_dir))
    module = module_name_for(path, root_dir)

    imports = [
        " ".join(py_ast.unparse(n).split())
        for n in py_ast.walk(tree)
        if isinstance(n, (py_ast.Import, py_ast.ImportFrom))
    ]

    chunks: list[Chunk] = []
    defs = (py_ast.FunctionDef, py_ast.AsyncFunctionDef, py_ast.ClassDef)

    def visit(node: py_ast.AST, scope: list[str], parent_qual: str | None) -> None:
        for child in py_ast.iter_child_nodes(node):
            if isinstance(child, defs):
                name = child.name
                qualified = ".".join([module] + scope + [name])
                kind = "class" if isinstance(child, py_ast.ClassDef) else (
                    "method" if scope else "function"
                )
                start = min(
                    [child.lineno] + [d.lineno for d in child.decorator_list]
                )
                end = getattr(child, "end_lineno", child.lineno)

                chunks.append(
                    Chunk(
                        chunk_id=make_chunk_id(rel, qualified, start),
                        path=rel,
                        kind=kind,
                        name=name,
                        qualified_name=qualified,
                        module=module,
                        start_line=start,
                        end_line=end,
                        source="\n".join(lines[start - 1:end]),
                        signature=_py_signature(child, lines),
                        docstring=py_ast.get_docstring(child) or "",
                        parent=parent_qual,
                        imports=imports,
                        calls=_py_calls(child),
                        decorators=[py_ast.unparse(d) for d in child.decorator_list],
                    )
                )
                visit(child, scope + [name], qualified)
            else:
                visit(child, scope, parent_qual)

    visit(tree, [], None)
    return chunks


# --------------------------------------------------------------------------
# Public surface
# --------------------------------------------------------------------------

def chunk_source(path: Path, root_dir: Path, source: str,
                 backend: str | None = None) -> list[Chunk]:
    """Chunk one file. `backend` forces a parser, otherwise the best available."""
    use_ts = _TS_AVAILABLE if backend is None else backend == "tree-sitter"
    if use_ts and not _TS_AVAILABLE:
        raise RuntimeError("tree-sitter backend requested but not installed")
    if use_ts:
        try:
            return _ts_chunks(path, root_dir, source)
        except Exception:
            # A parser failure on one file must not abandon the whole index.
            return _py_chunks(path, root_dir, source)
    return _py_chunks(path, root_dir, source)


def chunk_repository(root_dir: Path, backend: str | None = None,
                     min_lines: int = 1) -> list[Chunk]:
    """Every chunk in a repository, in stable path order."""
    root_dir = Path(root_dir).resolve()
    out: list[Chunk] = []
    for path in iter_source_files(root_dir):
        try:
            source = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for chunk in chunk_source(path, root_dir, source, backend=backend):
            if chunk.line_count >= min_lines:
                out.append(chunk)
    return out
