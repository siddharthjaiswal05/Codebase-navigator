"""Dense retrieval backends.

The design targets `jinaai/jina-embeddings-v2-base-code`, a code-specialised
encoder. That model is a ~600MB download, so the encoder is an interface with
two implementations and the index records which one produced its vectors:

  jina      the code encoder, used whenever sentence-transformers and the
            weights are present
  tfidf-svd a local, deterministic encoder: TF-IDF over identifier-level tokens
            reduced by truncated SVD, which is latent semantic indexing. It
            needs no download and no network, so the pipeline is always
            runnable and evaluation is always reproducible.

Reported numbers always name the backend that produced them. `describe()`
returns that name, and it is written into the index manifest.
"""

from __future__ import annotations

import os
import pickle
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np

from .tokenize import tokenize

JINA_MODEL = "jinaai/jina-embeddings-v2-base-code"


class Encoder(ABC):
    """Maps text to unit-norm vectors."""

    name: str = "encoder"
    dimension: int = 0

    @abstractmethod
    def fit(self, corpus: list[str]) -> "Encoder":
        """Backends that learn from the corpus use this; others no-op."""

    @abstractmethod
    def encode(self, texts: list[str]) -> np.ndarray:
        """(len(texts), dimension) float32, L2-normalised."""

    def describe(self) -> str:
        return f"{self.name} (dim={self.dimension})"


def _l2_normalise(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (matrix / norms).astype(np.float32)


class JinaEncoder(Encoder):
    """The code-specialised encoder the design targets."""

    name = "jina-embeddings-v2-base-code"

    def __init__(
        self,
        model_name: str = JINA_MODEL,
        batch_size: int = 8,
        device: str = "cpu",
        max_seq_length: int = 1024,
    ):
        # The jina v2 code models ship custom modelling code written against
        # the transformers 4.x API. On installs that also carry TensorFlow,
        # transformers eagerly imports it and an unrelated TF/protobuf mismatch
        # surfaces here as a jina failure. Disabling the TF path keeps the
        # import on torch, which is the only backend this needs.
        os.environ.setdefault("USE_TF", "0")
        os.environ.setdefault("TRANSFORMERS_NO_TF", "1")

        from sentence_transformers import SentenceTransformer

        # CPU by default, deliberately. This model accepts very long inputs and
        # attention is quadratic in sequence length, so a single long source
        # file can ask for several GB in one allocation. On Apple MPS that is a
        # hard out-of-memory abort rather than a catchable error, which makes
        # the whole index build die. CPU degrades gracefully instead.
        self.model = SentenceTransformer(
            model_name, trust_remote_code=True, device=device
        )
        # Cap the window for the same reason. Chunks are functions and classes,
        # so 1024 tokens covers the overwhelming majority intact, and the
        # signal that matters (signature, docstring, opening body) sits early.
        self.model.max_seq_length = min(max_seq_length, self.model.max_seq_length)

        self.device = device
        self.batch_size = batch_size
        self.dimension = int(self.model.get_sentence_embedding_dimension())

    def describe(self) -> str:
        return (
            f"{self.name} (dim={self.dimension}, device={self.device}, "
            f"max_seq={self.model.max_seq_length})"
        )

    def fit(self, corpus: list[str]) -> "JinaEncoder":
        return self  # pretrained

    def encode(self, texts: list[str]) -> np.ndarray:
        vectors = self.model.encode(
            texts,
            batch_size=self.batch_size,
            convert_to_numpy=True,
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        return vectors.astype(np.float32)


class TfidfSvdEncoder(Encoder):
    """Latent semantic indexing over identifier-level tokens.

    Deterministic and dependency-light. It shares the code-aware tokenizer with
    BM25, so its vocabulary is the same camelCase and snake_case parts, but the
    SVD projection lets it match documents that share no literal term with the
    query, which is the job the dense arm exists to do.
    """

    name = "tfidf-svd"

    def __init__(self, dimension: int = 256, random_state: int = 0):
        self.requested_dimension = dimension
        self.random_state = random_state
        self.vectorizer = None
        self.svd = None
        self.dimension = 0

    def fit(self, corpus: list[str]) -> "TfidfSvdEncoder":
        from sklearn.decomposition import TruncatedSVD
        from sklearn.feature_extraction.text import TfidfVectorizer

        self.vectorizer = TfidfVectorizer(
            analyzer=lambda text: tokenize(text),
            min_df=1,
            sublinear_tf=True,
        )
        matrix = self.vectorizer.fit_transform(corpus)

        # SVD cannot produce more components than the smaller matrix dimension.
        n_components = max(
            2, min(self.requested_dimension, min(matrix.shape) - 1)
        )
        self.svd = TruncatedSVD(
            n_components=n_components, random_state=self.random_state
        )
        self.svd.fit(matrix)
        self.dimension = n_components
        return self

    def encode(self, texts: list[str]) -> np.ndarray:
        if self.vectorizer is None or self.svd is None:
            raise RuntimeError("TfidfSvdEncoder.fit must run before encode")
        matrix = self.vectorizer.transform(texts)
        return _l2_normalise(self.svd.transform(matrix))

    # The fitted vectorizer holds a lambda, which pickle cannot serialise, so
    # the analyzer is dropped here and restored on load. The vectorizer is
    # shallow-copied first: clearing the attribute on the live object would
    # leave the in-memory encoder unusable the moment an index was saved.
    def __getstate__(self) -> dict:
        import copy

        state = self.__dict__.copy()
        vectorizer = state.get("vectorizer")
        if vectorizer is not None:
            detached = copy.copy(vectorizer)
            detached.analyzer = None
            state["vectorizer"] = detached
        return state

    def __setstate__(self, state: dict) -> None:
        self.__dict__.update(state)
        if self.vectorizer is not None:
            self.vectorizer.analyzer = lambda text: tokenize(text)


def build_encoder(prefer: str = "auto", dimension: int = 256) -> Encoder:
    """Return the requested encoder.

    `auto` resolves to the local encoder, and the neural one is opt-in. That
    looks backwards until you try it on a constrained machine: loading a
    transformer that does not fit takes the process down with SIGSEGV or an
    allocator abort, neither of which is a catchable exception, so a `try`
    around it cannot fall back. An opt-in flag fails in the one place the
    operator is expecting it to.

    `prefer="jina"` therefore raises rather than downgrading, so a run meant to
    use the code encoder never reports numbers under the wrong backend name.
    """
    if prefer in ("auto", "tfidf-svd"):
        return TfidfSvdEncoder(dimension=dimension)
    if prefer == "jina":
        return JinaEncoder()
    raise ValueError(f"unknown encoder preference: {prefer}")


def save_encoder(encoder: Encoder, path: Path) -> None:
    """Only learned encoders need persisting; pretrained ones reload by name."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(encoder, TfidfSvdEncoder):
        with open(path, "wb") as handle:
            pickle.dump(encoder, handle, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        with open(path, "wb") as handle:
            pickle.dump({"pretrained": encoder.name}, handle)


def load_encoder(path: Path) -> Encoder:
    with open(path, "rb") as handle:
        obj = pickle.load(handle)
    if isinstance(obj, dict) and "pretrained" in obj:
        return JinaEncoder(model_name=obj["pretrained"])
    return obj
