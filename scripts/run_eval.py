#!/usr/bin/env python3
"""Run the evaluation suite and write a report.

    python3 scripts/run_eval.py                     # retrieval + agent
    python3 scripts/run_eval.py --validate-only     # check gold answers resolve
    python3 scripts/run_eval.py --swebench-path data/swebench_lite.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from navigator.evaluation import (  # noqa: E402
    QuestionSet, build_indexes, evaluate_agent, evaluate_retrieval,
    validate_gold, write_report,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate the code navigator.")
    parser.add_argument("--questions", type=Path, default=ROOT / "eval/questions.yaml")
    parser.add_argument("--out", type=Path, default=ROOT / "eval/report.json")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--encoder", default="auto",
                        choices=["auto", "jina", "tfidf-svd"])
    parser.add_argument("--llm", default="auto",
                        choices=["auto", "litellm", "scripted"])
    parser.add_argument("--max-steps", type=int, default=6)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--skip-agent", action="store_true")
    parser.add_argument("--swebench-path", type=Path, default=None)
    parser.add_argument("--swebench-checkouts", type=Path,
                        default=ROOT / "data/swebench_repos")
    parser.add_argument("--swebench-limit", type=int, default=None)
    args = parser.parse_args()

    question_set = QuestionSet.load(args.questions)
    print(f"Loaded {len(question_set.questions)} questions "
          f"across {len(question_set.repos)} repositories.")

    indexes = build_indexes(question_set, ROOT, encoder=args.encoder)
    for name, index in indexes.items():
        m = index.manifest
        print(f"  {name:12s} {m['chunks']:5d} chunks  {m['files']:3d} files  "
              f"parser={m['chunker_backend']}  encoder={m['encoder']}  "
              f"vectors={m['vector_backend']}")

    validation = validate_gold(question_set, indexes)
    print(f"\nGold validation: {'OK' if validation['ok'] else 'PROBLEMS'} "
          f"({validation['problems']} issues)")
    for detail in validation["details"][:20]:
        print(f"   {detail['id']}: {detail['issue']}")
    if args.validate_only:
        print(json.dumps(validation, indent=2))
        return 0 if validation["ok"] else 1

    report: dict = {
        "question_set": str(args.questions),
        "questions": len(question_set.questions),
        "indexes": {n: i.manifest for n, i in indexes.items()},
        "gold_validation": validation,
    }

    print("\nRetrieval ablations...")
    report["retrieval"] = evaluate_retrieval(
        question_set, indexes, top_k=args.top_k
    )
    for mode, res in report["retrieval"]["modes"].items():
        f, s = res["file_level"], res["function_level"]
        print(f"  {mode:14s} file hit@1={f['hit@1']:.3f} hit@5={f['hit@5']:.3f} "
              f"| function hit@1={s['hit@1']:.3f} hit@5={s['hit@5']:.3f} "
              f"mrr={s['mrr']:.3f}")

    if not args.skip_agent:
        print("\nAgent loop...")
        report["agent"] = evaluate_agent(
            question_set, indexes, llm_backend=args.llm, max_steps=args.max_steps
        )
        c = report["agent"]["citations"]
        print(f"  citation validity      {c['citation_validity_rate']:.3f}")
        print(f"  supported answers      {c['supported_answer_rate']:.3f}")
        print(f"  cited the gold file    {report['agent']['cited_gold_file_rate']:.3f}")
        print(f"  mean steps per answer  {report['agent']['mean_steps']}")

    if args.swebench_path or args.swebench_checkouts.exists():
        from navigator.evaluation.swebench import (
            evaluate_localisation, load_instances, prepare,
        )
        print("\nSWE-bench Lite...")
        instances, provenance = load_instances(
            args.swebench_path, limit=args.swebench_limit
        )
        print(f"  source: {provenance}")
        if instances:
            readiness = prepare(instances, args.swebench_checkouts)
            print(f"  checkouts ready: {readiness['ready']}/{readiness['instances']}")
            report["swebench"] = {
                "provenance": provenance,
                "readiness": readiness,
                "results": evaluate_localisation(
                    instances, args.swebench_checkouts, top_k=args.top_k
                ) if readiness["ready"] else None,
            }
        else:
            report["swebench"] = {"provenance": provenance, "results": None}

    write_report(report, args.out)
    print(f"\nWrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
