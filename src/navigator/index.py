"""The repository index: build, persist, load, and search.

One object owns the chunk store, both retrieval arms, and the graph, so a query
path is the same whether it comes from the CLI, the agent, or the evaluation
harness. The manifest records which backends produced the index, so a result is
always attributable to a configuration.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .chunking import chunk_repository
from .chunking.ast_chunker import which_backend as chunker_backend
from .chunking.models import Chunk
from .graph.call_graph import CodeGraph, build_code_graph
from .graph.expansion import ExpansionTrace, expand
from .retrieval.bm25 import BM25Index
from .retrieval.dense import Encoder, build_encoder, load_encoder, save_encoder
from .retrieval.fusion import FusedResult, reciprocal_rank_fusion
from .retrieval.vector_store import VectorStore
from .retrieval.vector_store import which_backend as vector_backend


@dataclass
class SearchResult:
    chunk: Chunk
    score: float
    sources: list[str]
    ranks: dict[str, int]
    hops: int = 0
    reached_via: str = "seed"

    def citation(self) -> str:
        return self.chunk.citation

    def to_dict(self) -> dict:
        return {
            "chunk_id": self.chunk.chunk_id,
            "citation": self.chunk.citation,
            "qualified_name": self.chunk.qualified_name,
            "kind": self.chunk.kind,
            "score": round(self.score, 6),
            "sources": self.sources,
            "ranks": self.ranks,
            "hops": self.hops,
            "reached_via": self.reached_via,
        }


class CodeIndex:
    """Everything needed to answer a question about one repository."""

    def __init__(
        self,
        repo_root: Path,
        chunks: list[Chunk],
        bm25: BM25Index,
        encoder: Encoder,
        store: VectorStore,
        graph: CodeGraph,
        manifest: dict,
    ):
        self.repo_root = Path(repo_root)
        self.chunks = chunks
        self.by_id = {c.chunk_id: c for c in chunks}
        self.bm25 = bm25
        self.encoder = encoder
        self.store = store
        self.graph = graph
        self.manifest = manifest

    # -- build ------------------------------------------------------------
    @classmethod
    def build(
        cls,
        repo_root: Path,
        encoder_preference: str = "auto",
        dense_dimension: int = 256,
        use_jedi: bool = True,
        jedi_budget: int = 300,
    ) -> "CodeIndex":
        repo_root = Path(repo_root).resolve()
        started = time.time()

        chunks = chunk_repository(repo_root)
        if not chunks:
            raise ValueError(f"no indexable source found under {repo_root}")

        texts = [c.retrieval_text() for c in chunks]
        ids = [c.chunk_id for c in chunks]

        bm25 = BM25Index.build(list(zip(ids, texts)))

        encoder = build_encoder(encoder_preference, dimension=dense_dimension)
        encoder.fit(texts)
        vectors = encoder.encode(texts)

        store = VectorStore(encoder.dimension)
        store.add(ids, vectors)

        graph = build_code_graph(
            chunks, repo_root=repo_root, use_jedi=use_jedi, jedi_budget=jedi_budget
        )

        manifest = {
            "repo_root": str(repo_root),
            "built_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "build_seconds": round(time.time() - started, 2),
            "chunks": len(chunks),
            "files": len({c.path for c in chunks}),
            "chunker_backend": chunker_backend(),
            "encoder": encoder.name,
            "encoder_dimension": encoder.dimension,
            "vector_backend": store.backend,
            "vector_backend_available": vector_backend(),
            "graph": graph.summary(),
            "chunk_kinds": {
                kind: sum(1 for c in chunks if c.kind == kind)
                for kind in sorted({c.kind for c in chunks})
            },
        }

        return cls(repo_root, chunks, bm25, encoder, store, graph, manifest)

    # -- retrieval --------------------------------------------------------
    def search_bm25(self, query: str, top_k: int = 20) -> list[tuple[str, float]]:
        return self.bm25.search(query, top_k=top_k)

    def search_dense(self, query: str, top_k: int = 20) -> list[tuple[str, float]]:
        vector = self.encoder.encode([query])
        return self.store.search_one(vector[0], top_k=top_k)

    def search_hybrid(
        self,
        query: str,
        top_k: int = 10,
        candidates_per_arm: int = 30,
        weights: dict[str, float] | None = None,
    ) -> list[SearchResult]:
        """BM25 and dense, fused with reciprocal rank fusion."""
        fused = self._fuse(query, candidates_per_arm, weights)
        return [self._to_result(f) for f in fused[:top_k]]

    def _fuse(
        self,
        query: str,
        candidates_per_arm: int,
        weights: dict[str, float] | None = None,
    ) -> list[FusedResult]:
        return reciprocal_rank_fusion(
            {
                "bm25": self.search_bm25(query, top_k=candidates_per_arm),
                "dense": self.search_dense(query, top_k=candidates_per_arm),
            },
            weights=weights,
        )

    def _to_result(self, fused: FusedResult, hops: int = 0,
                   reached_via: str = "seed") -> SearchResult:
        return SearchResult(
            chunk=self.by_id[fused.doc_id],
            score=fused.score,
            sources=fused.sources(),
            ranks=fused.ranks,
            hops=hops,
            reached_via=reached_via,
        )

    def search_with_expansion(
        self,
        query: str,
        top_k: int = 10,
        seed_k: int = 8,
        token_budget: int = 6000,
        max_hops: int = 2,
        candidates_per_arm: int = 30,
        include_siblings: bool = False,
    ) -> tuple[list[SearchResult], ExpansionTrace]:
        """Hybrid seeds, then a graph walk under a token budget."""
        fused = self._fuse(query, candidates_per_arm)
        by_doc = {f.doc_id: f for f in fused}
        seeds = [(f.doc_id, f.score) for f in fused[:seed_k]]

        expanded, trace = expand(
            seeds,
            self.graph,
            self.by_id,
            token_budget=token_budget,
            max_hops=max_hops,
            include_siblings=include_siblings,
        )

        results: list[SearchResult] = []
        for item in expanded:
            chunk = self.by_id.get(item.chunk_id)
            if chunk is None:
                continue
            source = by_doc.get(item.chunk_id)
            results.append(
                SearchResult(
                    chunk=chunk,
                    score=item.score,
                    sources=source.sources() if source else ["graph"],
                    ranks=source.ranks if source else {},
                    hops=item.hops,
                    reached_via=item.reached_via,
                )
            )
        return results[:top_k], trace

    # -- lookups ----------------------------------------------------------
    def get(self, chunk_id: str) -> Chunk | None:
        return self.by_id.get(chunk_id)

    def find_by_name(self, name: str) -> list[Chunk]:
        lowered = name.lower()
        return [
            c for c in self.chunks
            if c.name.lower() == lowered or c.qualified_name.lower().endswith(lowered)
        ]

    def chunks_in_file(self, path: str) -> list[Chunk]:
        return sorted(
            (c for c in self.chunks if c.path == path),
            key=lambda c: c.start_line,
        )

    def files(self) -> list[str]:
        return sorted({c.path for c in self.chunks})

    # -- persistence ------------------------------------------------------
    def save(self, index_dir: Path) -> None:
        index_dir = Path(index_dir)
        index_dir.mkdir(parents=True, exist_ok=True)

        with open(index_dir / "chunks.jsonl", "w", encoding="utf-8") as handle:
            for chunk in self.chunks:
                handle.write(json.dumps(chunk.to_dict()) + "\n")

        self.bm25.save(index_dir / "bm25.pkl")
        save_encoder(self.encoder, index_dir / "encoder.pkl")
        self.store.save(index_dir / "vectors")

        import pickle

        with open(index_dir / "graph.pkl", "wb") as handle:
            pickle.dump(self.graph, handle, protocol=pickle.HIGHEST_PROTOCOL)

        with open(index_dir / "manifest.json", "w", encoding="utf-8") as handle:
            json.dump(self.manifest, handle, indent=2)

    @classmethod
    def load(cls, index_dir: Path) -> "CodeIndex":
        index_dir = Path(index_dir)

        with open(index_dir / "manifest.json", encoding="utf-8") as handle:
            manifest = json.load(handle)

        chunks = []
        with open(index_dir / "chunks.jsonl", encoding="utf-8") as handle:
            for line in handle:
                chunks.append(Chunk.from_dict(json.loads(line)))

        bm25 = BM25Index.load(index_dir / "bm25.pkl")
        encoder = load_encoder(index_dir / "encoder.pkl")
        store = VectorStore.load(
            index_dir / "vectors",
            dimension=manifest["encoder_dimension"],
            backend=manifest["vector_backend"],
        )

        import pickle

        with open(index_dir / "graph.pkl", "rb") as handle:
            graph = pickle.load(handle)

        return cls(
            Path(manifest["repo_root"]), chunks, bm25, encoder, store, graph, manifest
        )
