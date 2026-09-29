"""The unit of retrieval.

A chunk is a syntactically complete piece of code, never a token window. That
property is what makes a citation trustworthy: every chunk has a real start and
end line in a real file, so `path:line` always points at something a reader can
open and verify.
"""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class Chunk:
    """One function, method, class, or module-level region."""

    chunk_id: str
    path: str                     # repo-relative
    kind: str                     # function | method | class | module
    name: str                     # bare symbol name
    qualified_name: str           # module.Class.method
    start_line: int               # 1-indexed, inclusive
    end_line: int                 # 1-indexed, inclusive
    source: str
    module: str = ""              # dotted module path, set at chunk time
    signature: str = ""
    docstring: str = ""
    parent: str | None = None     # qualified name of the enclosing scope
    imports: list[str] = field(default_factory=list)
    calls: list[str] = field(default_factory=list)
    decorators: list[str] = field(default_factory=list)
    language: str = "python"

    @property
    def citation(self) -> str:
        """The form every answer has to cite in."""
        return f"{self.path}:{self.start_line}-{self.end_line}"

    @property
    def line_count(self) -> int:
        return self.end_line - self.start_line + 1

    def header(self) -> str:
        """A compact description used when the agent lists candidates."""
        sig = self.signature or self.name
        return f"{self.citation}  [{self.kind}] {sig}"

    def retrieval_text(self) -> str:
        """What the retrievers index.

        The qualified name and signature are repeated ahead of the body so that
        identifier matches on the symbol itself outweigh incidental matches
        deep inside an unrelated function.
        """
        parts = [self.qualified_name, self.signature, self.docstring, self.source]
        return "\n".join(p for p in parts if p)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Chunk":
        allowed = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in data.items() if k in allowed})


def make_chunk_id(path: str, qualified_name: str, start_line: int) -> str:
    """Stable across runs, so a cached index and a fresh one agree."""
    raw = f"{path}::{qualified_name}::{start_line}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]
