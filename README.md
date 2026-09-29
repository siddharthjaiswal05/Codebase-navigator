# Codebase-navigator

An agentic code retrieval system that answers natural-language questions about
unfamiliar repositories and backs every answer with verifiable `file:line`
citations.

Conventional RAG fails on source code for two reasons: token-window chunking splits
functions in half, and embedding similarity misses the code you need when the question
and the implementation share no vocabulary. This project addresses both, then adds an
agent that decides how far to traverse.

EE656 course project, Prof. Nischal K Verma.

```bash
pip install -r requirements.txt
python3 scripts/ask.py --repo src --trace "Which functions call the rank fusion helper?"
python3 scripts/run_eval.py          # 40-question suite, retrieval ablations, agent
python3 tests/test_navigator.py      # 34/34
```

It runs with no API key and no model download. Both are opt-in.

## How it works

### AST-based chunking

Chunks are extracted with **tree-sitter** rather than cut on a token window. Functions,
methods and classes come out as atomic units, each annotated with its signature,
enclosing scope, import table and line span. Nothing is ever split mid-function, so a
retrieved chunk is always something you can actually read and cite.

That second property is the one the rest of the system leans on. A token-window chunk
makes `path:120-160` an artefact of the chunker, not a statement about the code, so
there is nothing to check it against. A syntactic chunk has a real line span, which is
what makes a citation falsifiable. `test_chunk_line_spans_match_the_file` asserts every
span against the file on disk.

A zero-dependency `ast` backend produces identical output when tree-sitter is
unavailable, and the index records which parser produced it.

### Structural graphs

Call and module-dependency graphs are built in **NetworkX**, resolving call sites
through import tables and **jedi** symbol resolution. This makes it possible to retrieve
a function because of how it relates to other code, not because its text happens to
overlap the query.

Call sites resolve in four passes, cheapest first: exact qualified name, import table,
scope-aware suffix match, then jedi on the residue under a budget. Two accounting
decisions make the resulting number mean something:

- **External calls are not resolution failures.** A call to `len()` has no in-repo
  target. Counting those as unresolved reported 31% when the true rate against
  resolvable call sites was 91%.
- **Ambiguity is never resolved by guessing.** Where several same-named symbols are
  equally plausible, no edge is added, because a wrong edge sends the traversal
  somewhere unrelated.

Measured on this repository: 747 call sites, 482 external, **265 in-repo of which 80%
resolve** (37 by import table, 143 by scope, 32 by jedi), leaving 83 ambiguous and 53
unresolved.

### Hybrid retrieval

Issue reports and implementation code rarely use the same words. Two retrievers run
against that gap and are combined with **reciprocal rank fusion**:

- **BM25** at identifier level, tokenized on camelCase and snake_case boundaries
- **jina-embeddings-v2-base-code** dense vectors over **FAISS**

Fusion is rank-based rather than score-blending on purpose. BM25 scores are unbounded
and corpus-dependent; cosine sits in a fixed range. Normalising them onto one scale
means inventing a relationship between two distributions, and that choice quietly
decides the ranking. RRF discards magnitudes:

```
score(d) = sum over retrievers of  weight / (k + rank(d))
```

`test_fusion_ignores_raw_score_magnitude` asserts the property: rescale either arm by
any factor and the fused order does not move.

### Graph-expansion retrieval

Hybrid search supplies the seeds; the system then walks caller and callee edges outward
under an agent-controlled token budget. This is what reaches target functions containing
zero query terms, which pure lexical or pure vector search cannot do.

That is not a claim about the design, it is a test. On the question *"how is the token
budget enforced during traversal?"*, expansion returns **7 of 12 results with zero
query-term overlap**, none of which either retriever returned. The budget is a token
budget rather than a hop constant, so the walk adapts to how large the surrounding
functions are, and scores decay with distance so a hub function cannot flood the result
set.

### Agent loop

A **ReAct-style control loop** sits over a seven-tool surface. The model chooses which
tools to call and how deep to traverse at runtime, rather than executing a fixed
pipeline. Traversal depth is therefore a function of the question, not a constant.

| Tool | Purpose |
|---|---|
| `search_code` | hybrid retrieval, the usual entry point |
| `expand_context` | graph walk under a chosen token budget |
| `read_chunk` | full source, so a citation covers inspected code |
| `find_callers` | inbound call edges |
| `find_callees` | outbound call edges |
| `list_symbols` | everything defined in one file |
| `grep_repo` | literal strings that ranking would bury |

Depth varying with the question is also a test rather than an assertion: a relational
question spends 5 steps and invokes `expand_context` and `find_callers`; a direct lookup
answers in 3 and touches neither. Across the 40-question suite the mean is **3.25 steps**,
with `expand_context` invoked on 5 of 40 questions.

## Evaluation

Bug localisation is measured against **SWE-bench Lite** gold patches at both file and
function granularity, alongside a hand-verified 40-question set.

**Citation validity is treated as a first-class correctness criterion**, on equal footing
with answer accuracy. An answer that is right but cites the wrong location is not
counted as correct.

### The question set

40 questions across two repositories: this one, and an unrelated Python project, so the
system is not being scored only on the source it was written against. Every gold answer
is checked to resolve to a real indexed symbol before scoring, and the run fails loudly
if one drifts.

Questions are typed, because they measure different things:

| Type | n | What it tests |
|---|---|---|
| `lookup` | 6 | question shares vocabulary with the target |
| `vocabulary_gap` | 31 | question deliberately avoids the identifier it seeks |
| `relational` | 3 | answer depends on how code connects to other code |

### Retrieval ablations

Reporting the full system without its components says nothing about which part is doing
the work, so all four configurations run over the identical question set.

| Configuration | file hit@1 | file hit@5 | file MRR | function hit@5 | function MRR |
|---|---|---|---|---|---|
| BM25 only | **0.550** | 0.775 | **0.638** | **0.450** | 0.326 |
| Dense only | 0.425 | **0.800** | 0.573 | 0.400 | 0.263 |
| Hybrid (RRF) | 0.525 | 0.775 | 0.622 | 0.425 | 0.313 |
| Hybrid + graph | 0.525 | **0.800** | 0.629 | 0.425 | **0.333** |

Read honestly, this says two things.

**Graph expansion earns its place.** It produces the best function-level MRR overall
(0.333) and on the vocabulary-gap subset specifically (0.363 against BM25's 0.357),
while recovering the file-level hit@5 that fusion alone gives up. It is the component
that retrieves code text search cannot reach.

**Fusion does not beat BM25 alone in this configuration, and the reason is the dense
arm.** These numbers come from the local TF-IDF + SVD encoder, not from jina. That
encoder is weaker than BM25 at every granularity, and RRF at equal weight lets the
weaker arm pull the stronger one down. A weight sweep from 1:1 to 6:1 in BM25's favour
recovers file-level MRR to 0.650 but never restores function-level hit@5. Fusion is
worth having when the dense arm is a genuine code encoder; it is not worth having for
its own sake, and the ablation is what makes that visible rather than assumed.

The jina backend is implemented and verified working: it loads at 768 dimensions and
scores cosine 0.60 between a code snippet and a natural-language question sharing no
vocabulary, which is exactly the signal the dense arm exists to supply. Re-run any of
the above with `--encoder jina` on a machine with the memory for it to get the
corresponding numbers.

### Citations

| Metric | Result |
|---|---|
| Citation validity rate | **1.000** (120 of 120) |
| Answers with at least one valid citation | **1.000** (40 of 40) |
| Answers where every citation is valid | **1.000** |
| Answers citing the gold file | 0.700 |

Validity is measured against the index, not the filesystem, so a citation into a file
the index never read cannot pass. The verifier rejects five distinct failure modes:
malformed strings, paths absent from the index, lines past a file's extent, spans
crossing chunk boundaries, and citations with no line numbers. Each is a test.

The gap between 1.000 validity and 0.700 gold-file accuracy is the honest shape of this
result: every citation points at real code that was actually retrieved and read, and on
70% of questions that code is in the right file. Validity and accuracy are different
claims and are reported separately.

### SWE-bench Lite

The adapter parses gold patches into changed files and post-image line ranges, and
scores localisation at both granularities: whether the ranked list contains a file the
patch modified, and whether it contains a symbol whose line span overlaps a patch hunk.
Patch parsing is unit-tested against a synthetic multi-file diff.

Run it with:

```bash
python3 scripts/run_eval.py --swebench-path data/swebench_lite.jsonl \
                            --swebench-checkouts data/swebench_repos
```

Each instance needs its repository checked out at the base commit; the harness reports
which instances are ready and scores only those, and every report states how many
instances it covered so the numbers are never read as covering more of the benchmark
than they do.

## Running cost

The full system runs at zero infrastructure cost:

- Multi-provider LLM failover across **Groq** and **Gemini** through **LiteLLM**
- A **SQLite** prompt-hash response cache, making re-evaluation deterministic and free

The cache keys on the model, the full message list and the sampling parameters, so a
repeated prompt replays byte-identically. That is what makes an ablation trustworthy: a
change in a retrieval metric is attributable to the retrieval change rather than to
sampling noise.

With no key configured, a deterministic offline policy drives the same seven-tool
surface, so the loop, the citation checks and the whole evaluation run anywhere. The
numbers above were produced that way, which is why they are reproducible on any machine
by running the command at the top of this file.

## Layout

```
src/navigator/
  chunking/     tree-sitter and ast backends, the Chunk model
  graph/        call and module graphs, budgeted expansion
  retrieval/    identifier tokenizer, BM25, encoders, FAISS store, RRF
  agent/        seven tools, ReAct loop, LiteLLM failover, SQLite cache
  evaluation/   metrics, question-set harness, SWE-bench Lite adapter
  index.py      binds chunks, both retrievers and the graph together
scripts/        ask.py, build_index.py, run_eval.py
eval/           questions.yaml, report.json
tests/          34 behavioural tests
docs/           architecture.md
```

## Configuration

| Component | Default | Opt-in |
|---|---|---|
| Parser | tree-sitter | stdlib `ast` fallback |
| Encoder | TF-IDF + SVD | `--encoder jina` |
| Vector store | FAISS | exact NumPy fallback |
| Model | offline policy | `--llm litellm` with `GROQ_API_KEY` or `GEMINI_API_KEY` |

The encoder default is deliberate and looks backwards. Loading a transformer that does
not fit in memory takes the process down with SIGSEGV, which is not a catchable
exception, so a `try/except` cannot fall back to anything. An explicit flag fails where
the operator is expecting it to.

## Author

Siddharth Jaiswal
