# Codebase-navigator

An agentic code retrieval system that answers natural-language questions about
unfamiliar repositories and backs every answer with verifiable `file:line`
citations.

Conventional RAG fails on source code for two reasons: token-window chunking splits
functions in half, and embedding similarity misses the code you need when the question
and the implementation share no vocabulary. This project addresses both, then adds an
agent that decides how far to traverse.

EE656 course project, Prof. Nischal K Verma.

## How it works

### AST-based chunking

Chunks are extracted with **tree-sitter** rather than cut on a token window. Functions,
methods and classes come out as atomic units, each annotated with its signature,
enclosing scope, import table and line span. Nothing is ever split mid-function, so a
retrieved chunk is always something you can actually read and cite.

### Structural graphs

Call and module-dependency graphs are built in **NetworkX**, resolving call sites
through import tables and **jedi** symbol resolution. This makes it possible to retrieve
a function because of how it relates to other code, not because its text happens to
overlap the query.

### Hybrid retrieval

Issue reports and implementation code rarely use the same words. Two retrievers run
against that gap and are combined with **reciprocal rank fusion**:

- **BM25** at identifier level, tokenized on camelCase and snake_case boundaries
- **jina-embeddings-v2-base-code** dense vectors over **FAISS**

### Graph-expansion retrieval

Hybrid search supplies the seeds; the system then walks caller and callee edges outward
under an agent-controlled token budget. This is what reaches target functions containing
zero query terms, which pure lexical or pure vector search cannot do.

### Agent loop

A **ReAct-style control loop** sits over a seven-tool surface. The model chooses which
tools to call and how deep to traverse at runtime, rather than executing a fixed
pipeline. Traversal depth is therefore a function of the question, not a constant.

## Evaluation

Bug localisation is measured against **SWE-bench Lite** gold patches at both file and
function granularity, alongside a hand-verified 40-question set.

**Citation validity is treated as a first-class correctness criterion**, on equal footing
with answer accuracy. An answer that is right but cites the wrong location is not
counted as correct.

## Running cost

The full system runs at zero infrastructure cost:

- Multi-provider LLM failover across **Groq** and **Gemini** through **LiteLLM**
- A **SQLite** prompt-hash response cache, making re-evaluation deterministic and free
