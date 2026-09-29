#!/usr/bin/env python3
"""Ask a question about a repository.

    python3 scripts/ask.py --repo src "How are two ranked lists combined?"
    python3 scripts/ask.py --repo /path/to/project --encoder jina "..." 
    python3 scripts/ask.py --repo src --index-dir .index --trace "..."
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from navigator.agent import CodeNavigatorAgent, build_llm  # noqa: E402
from navigator.index import CodeIndex  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Ask a question about a repository.")
    parser.add_argument("question")
    parser.add_argument("--repo", type=Path, default=ROOT / "src")
    parser.add_argument("--index-dir", type=Path, default=None,
                        help="Load a saved index, or save a fresh one here.")
    parser.add_argument("--encoder", default="auto",
                        choices=["auto", "jina", "tfidf-svd"])
    parser.add_argument("--llm", default="auto",
                        choices=["auto", "litellm", "scripted"])
    parser.add_argument("--max-steps", type=int, default=6)
    parser.add_argument("--trace", action="store_true", help="Show every step.")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    if args.index_dir and (args.index_dir / "manifest.json").exists():
        index = CodeIndex.load(args.index_dir)
    else:
        index = CodeIndex.build(args.repo, encoder_preference=args.encoder)
        if args.index_dir:
            index.save(args.index_dir)

    llm = build_llm(args.llm)
    agent = CodeNavigatorAgent(index, llm=llm, max_steps=args.max_steps)
    answer = agent.answer(args.question)

    if args.json:
        print(json.dumps(answer.to_dict(), indent=2))
        return 0

    m = index.manifest
    print(f"index: {m['chunks']} chunks, {m['files']} files, "
          f"parser={m['chunker_backend']}, encoder={m['encoder']}, "
          f"vectors={m['vector_backend']}, llm={llm.name}\n")

    if args.trace:
        for step in answer.steps:
            print(f"  [{step.index}] {step.tool}")
            if step.thought:
                print(f"       {step.thought}")
            if step.error:
                print(f"       error: {step.error}")
        print()

    print(answer.answer, "\n")
    print("Citations:")
    for citation in answer.citations:
        mark = "ok  " if citation.valid else "BAD "
        note = "" if citation.valid else f"   ({citation.reason})"
        print(f"  {mark}{citation.raw}{note}")

    print(f"\n{len(answer.valid_citations)}/{len(answer.citations)} citations verified"
          f" | {len(answer.steps)} steps | tools: {', '.join(answer.tools_used)}")
    return 0 if answer.is_supported else 1


if __name__ == "__main__":
    raise SystemExit(main())
