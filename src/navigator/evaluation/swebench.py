"""SWE-bench Lite adapter for bug localisation.

SWE-bench Lite pairs a natural-language issue report with the gold patch that
resolved it. The patch names the files and hunks a correct fix touches, which
makes it a localisation label that nobody wrote for this system: given only the
issue text, does retrieval surface the code the fix actually changed.

That is the hardest form of the vocabulary gap. An issue report describes a
symptom in a user's words; the patch lands in functions whose identifiers the
reporter never saw.

Scoring works at both granularities:

  file level      did the ranked list contain a file the patch modified
  function level  did it contain a symbol whose line span overlaps a patch hunk

`load_instances` reads the dataset if it is present locally or through
`datasets`, and reports plainly when it is not, rather than inventing labels.
Each instance needs its repository checked out at the base commit; `prepare`
reports which instances are ready and which are missing a checkout.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from ..index import CodeIndex
from .metrics import MetricAccumulator, hit_at_k, reciprocal_rank

DATASET_NAME = "princeton-nlp/SWE-bench_Lite"

HUNK_RE = re.compile(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,(\d+))? @@")
FILE_RE = re.compile(r"^\+\+\+ b/(.+)$")


@dataclass
class PatchTarget:
    """One file a gold patch touches, with the line ranges it changes."""

    path: str
    line_ranges: list[tuple[int, int]] = field(default_factory=list)

    def overlaps(self, start: int, end: int) -> bool:
        return any(start <= hi and lo <= end for lo, hi in self.line_ranges)


@dataclass
class SWEInstance:
    instance_id: str
    repo: str
    base_commit: str
    problem_statement: str
    patch: str
    targets: list[PatchTarget] = field(default_factory=list)

    def gold_files(self) -> set[str]:
        return {t.path for t in self.targets}


def parse_patch(patch: str) -> list[PatchTarget]:
    """Extract changed files and their post-image line ranges from a diff."""
    targets: dict[str, PatchTarget] = {}
    current: PatchTarget | None = None

    for line in patch.split("\n"):
        file_match = FILE_RE.match(line)
        if file_match:
            path = file_match.group(1).strip()
            if path == "/dev/null":
                current = None
                continue
            current = targets.setdefault(path, PatchTarget(path))
            continue

        hunk_match = HUNK_RE.match(line)
        if hunk_match and current is not None:
            start = int(hunk_match.group(1))
            length = int(hunk_match.group(2) or 1)
            current.line_ranges.append((start, start + max(0, length - 1)))

    return list(targets.values())


def load_instances(
    path: Path | None = None,
    split: str = "test",
    limit: int | None = None,
) -> tuple[list[SWEInstance], str]:
    """Load SWE-bench Lite from a local file or the datasets library.

    Returns the instances and a note describing where they came from, so a
    report can always state its own provenance.
    """
    if path is not None and Path(path).exists():
        instances = []
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                instances.append(_to_instance(row))
                if limit and len(instances) >= limit:
                    break
        return instances, f"local file {path}"

    try:
        from datasets import load_dataset
    except ImportError:
        return [], (
            "SWE-bench Lite not loaded: the `datasets` package is not installed. "
            "Install it, or pass a local JSONL export via --swebench-path."
        )

    try:
        dataset = load_dataset(DATASET_NAME, split=split)
    except Exception as exc:
        return [], f"SWE-bench Lite not loaded: {type(exc).__name__}: {exc}"

    instances = []
    for row in dataset:
        instances.append(_to_instance(row))
        if limit and len(instances) >= limit:
            break
    return instances, f"{DATASET_NAME} [{split}]"


def _to_instance(row: dict) -> SWEInstance:
    patch = row.get("patch", "") or ""
    return SWEInstance(
        instance_id=row.get("instance_id", ""),
        repo=row.get("repo", ""),
        base_commit=row.get("base_commit", ""),
        problem_statement=row.get("problem_statement", "") or "",
        patch=patch,
        targets=parse_patch(patch),
    )


def prepare(instances: list[SWEInstance], checkout_root: Path) -> dict:
    """Report which instances have a usable checkout.

    Localisation needs the repository at the base commit. This does not clone
    anything; it states what is present so a run is never scored against a
    repository that is not actually there.
    """
    ready, missing = [], []
    for instance in instances:
        target = checkout_root / instance.instance_id
        if target.exists():
            ready.append(instance.instance_id)
        else:
            missing.append(instance.instance_id)
    return {
        "checkout_root": str(checkout_root),
        "instances": len(instances),
        "ready": len(ready),
        "missing": len(missing),
        "ready_ids": ready[:20],
        "missing_ids": missing[:20],
    }


def evaluate_localisation(
    instances: list[SWEInstance],
    checkout_root: Path,
    top_k: int = 10,
    mode: str = "hybrid+graph",
) -> dict:
    """Score issue-to-code localisation against gold patches.

    Only instances with a checkout are scored, and the count of scored
    instances is reported alongside the metrics so the numbers are never read
    as covering more of the benchmark than they do.
    """
    from .harness import _retrieve

    file_acc = MetricAccumulator()
    function_acc = MetricAccumulator()
    per_instance: list[dict] = []
    skipped: list[str] = []

    for instance in instances:
        repo_path = checkout_root / instance.instance_id
        if not repo_path.exists():
            skipped.append(instance.instance_id)
            continue

        try:
            index = CodeIndex.build(repo_path)
        except Exception as exc:
            skipped.append(f"{instance.instance_id} ({type(exc).__name__})")
            continue

        chunks = _retrieve(index, instance.problem_statement, mode, top_k)

        ranked_files, seen = [], set()
        for chunk in chunks:
            if chunk.path not in seen:
                seen.add(chunk.path)
                ranked_files.append(chunk.path)

        gold_files = instance.gold_files()
        file_acc.add(ranked_files, gold_files)

        # A retrieved symbol counts only if its lines overlap a patch hunk.
        target_by_path = {t.path: t for t in instance.targets}
        ranked_symbols, gold_symbols = [], set()
        for chunk in chunks:
            ranked_symbols.append(chunk.qualified_name)
            target = target_by_path.get(chunk.path)
            if target and target.overlaps(chunk.start_line, chunk.end_line):
                gold_symbols.add(chunk.qualified_name)

        function_acc.add(ranked_symbols, gold_symbols)

        per_instance.append(
            {
                "instance_id": instance.instance_id,
                "repo": instance.repo,
                "gold_files": sorted(gold_files),
                "file_hit@1": hit_at_k(ranked_files, gold_files, 1),
                "file_hit@10": hit_at_k(ranked_files, gold_files, 10),
                "file_mrr": round(reciprocal_rank(ranked_files, gold_files), 4),
                "function_hit@10": hit_at_k(ranked_symbols, gold_symbols, 10),
            }
        )

    return {
        "mode": mode,
        "top_k": top_k,
        "instances_available": len(instances),
        "instances_scored": len(per_instance),
        "instances_skipped": len(skipped),
        "skipped_ids": skipped[:20],
        "file_level": file_acc.summary(),
        "function_level": function_acc.summary(),
        "per_instance": per_instance,
    }
