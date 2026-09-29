# Architecture

How the pieces fit, and why each one is shaped the way it is.

## The pipeline

```
repository
   |
   |  tree-sitter parse, one chunk per function / method / class
   v
chunks ----------------------------+
   |                               |
   |  identifier-level BM25        |  jina or TF-IDF+SVD vectors
   v                               v
lexical ranks                 dense ranks          call + module graph
   |                               |                       |
   +---------- reciprocal rank fusion ---------+           |
                        |                                  |
                     seeds ------- graph walk under a token budget
                                          |
                                     candidate set
                                          |
                          ReAct loop over a seven-tool surface
                                          |
                            answer + citations, each verified
```

## Why chunks are syntactic

A token-window chunker produces spans that start and end arbitrarily. Two
consequences follow, and the second is the one that matters here.

The retrieval consequence is well known: half a function embeds poorly, because
the signature that says what it does may sit in a different chunk from the body
that does it.

The citation consequence is worse. If a chunk is an arbitrary window, then
`path:120-160` is an artefact of the chunker rather than a statement about the
code. It cannot be checked, because there is nothing it is supposed to line up
with. Every chunk here is a complete definition with a real line span, so a
citation is a claim that can be verified against the index, and
`verify_citation` does exactly that.

## Why two retrievers

They fail differently, which is the only good reason to run both.

BM25 over identifier parts is precise when the question happens to use the
project's vocabulary. Ask "where is BM25 scoring implemented" and it wins
outright.

Dense retrieval is what has any chance when the question and the code share no
words. That is the normal case for an issue report: a user describes a symptom,
and the fix lands in a function whose name the user never saw.

## Why reciprocal rank fusion, and not score blending

BM25 scores are unbounded and depend on corpus statistics; cosine similarities
sit in a fixed range. Putting them on a common scale means choosing a
normalisation, and that choice silently decides the ranking. RRF discards the
magnitudes and combines ranks:

```
score(d) = sum over retrievers of  weight / (k + rank(d))
```

A document ranked well by either arm surfaces; one ranked well by both wins.
`test_fusion_ignores_raw_score_magnitude` asserts the property directly: rescale
one arm's scores by any factor and the fused order does not move.

## Why the graph

Text retrieval of either kind can only return code that resembles the query. A
function that shares no vocabulary with the question is unreachable, no matter
how good the encoder.

The graph makes it reachable through a different relation: this function is
called by something that did match. Expansion walks caller and callee edges
outward from the fused seeds, and
`test_expansion_reaches_code_with_no_query_term_overlap` asserts that the walk
returns results with zero query-term overlap, which neither retriever could
have produced.

Two properties keep the walk from degenerating. A token budget, rather than a
depth constant, so the traversal adapts to how large the surrounding functions
actually are. And distance decay, so a hub function with many callers cannot
flood the result set.

### Resolving call sites

An edge is only useful if it is real. Call sites resolve in four passes,
cheapest first: exact qualified name, import table, scope-aware suffix match,
then jedi on whatever is left, under a budget because jedi is expensive.

Two accounting decisions make the resulting number meaningful:

**External calls are not failures.** A call to `len()` or `Path()` has no
in-repo target. Counting those as unresolved put the measured rate at 31% when
the true figure against resolvable call sites was 91%. They are now counted
separately.

**Ambiguity is not resolved by guessing.** Where several same-named symbols are
plausible and nothing distinguishes them, no edge is added. A wrong edge sends
the traversal somewhere unrelated, which is worse than a missing one.

## Why the agent, rather than a fixed pipeline

Different questions need different work. "Where is BM25 implemented" needs one
search. "Which functions call the fusion helper" needs a search, a graph walk,
and a caller lookup. A fixed pipeline has to be tuned for one of those and is
wrong for the other.

The loop gives the model the tool surface and the observations so far, and it
returns one action per step. Whether to expand, how large a token budget to
spend, and when to stop are decisions per question, not constants.

The seven tools:

| Tool | Purpose |
|---|---|
| `search_code` | hybrid retrieval, the usual entry point |
| `expand_context` | graph walk under a chosen token budget |
| `read_chunk` | full source, so a citation covers inspected code |
| `find_callers` | inbound call edges |
| `find_callees` | outbound call edges |
| `list_symbols` | everything defined in one file |
| `grep_repo` | literal strings that ranking would bury |

## Why the answer step is not trusted

The loop verifies every citation against the index before returning. A model
that answers correctly but points at the wrong location has produced something
worse than a wrong answer, because it looks checkable and is not.

`verify_citation` rejects five distinct failure modes: malformed strings, paths
absent from the index, lines beyond a file's extent, spans that cross chunk
boundaries, and citations with no line numbers at all.

## Why every heavy dependency has a fallback

The system runs on an 8GB laptop with no API key, and it runs in a configuration
with a GPU, a neural encoder and a paid model. The same code does both:

| Component | Default | Opt-in |
|---|---|---|
| Parser | tree-sitter | stdlib `ast` if unavailable |
| Encoder | TF-IDF + SVD | `--encoder jina` |
| Vector store | FAISS | exact NumPy store if unavailable |
| Model | offline policy | LiteLLM over Groq and Gemini |

The encoder default deserves a note, because it looks backwards. Loading a
transformer that does not fit in memory takes the process down with SIGSEGV,
which is not a catchable exception, so a `try/except` around it cannot fall
back. An opt-in flag fails where the operator is looking.

## Determinism

The SQLite cache keys on a hash of the model, the full message list and the
sampling parameters. A repeated prompt replays byte-identically, so a change in
a retrieval metric is attributable to the retrieval change rather than to
sampling noise, and re-running the suite after a change costs nothing.
