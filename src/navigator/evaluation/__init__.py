"""Evaluation: metrics, the hand-verified question set, and SWE-bench Lite."""

from .harness import (
    Question, QuestionSet, build_indexes, evaluate_agent, evaluate_retrieval,
    validate_gold, write_report,
)
from .metrics import CitationAccumulator, MetricAccumulator, hit_at_k, recall_at_k
from .swebench import SWEInstance, evaluate_localisation, load_instances, parse_patch

__all__ = [
    "Question", "QuestionSet", "build_indexes", "evaluate_agent",
    "evaluate_retrieval", "validate_gold", "write_report",
    "CitationAccumulator", "MetricAccumulator", "hit_at_k", "recall_at_k",
    "SWEInstance", "evaluate_localisation", "load_instances", "parse_patch",
]
