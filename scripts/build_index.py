#!/usr/bin/env python3
"""Build and persist an index for a repository.

    python3 scripts/build_index.py --repo /path/to/project --out .index
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from navigator.index import CodeIndex  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Index a repository.")
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path(".index"))
    parser.add_argument("--encoder", default="auto",
                        choices=["auto", "jina", "tfidf-svd"])
    parser.add_argument("--no-jedi", action="store_true",
                        help="Skip jedi refinement of unresolved call sites.")
    parser.add_argument("--jedi-budget", type=int, default=300)
    args = parser.parse_args()

    index = CodeIndex.build(
        args.repo,
        encoder_preference=args.encoder,
        use_jedi=not args.no_jedi,
        jedi_budget=args.jedi_budget,
    )
    index.save(args.out)
    print(json.dumps(index.manifest, indent=2))
    print(f"\nSaved to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
