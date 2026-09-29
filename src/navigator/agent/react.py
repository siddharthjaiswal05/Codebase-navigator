"""The ReAct control loop.

The model sees the question, the tool surface, and the observations so far, and
returns one action as JSON. The loop executes it, appends the observation, and
asks again. Nothing about the order of tools or the depth of traversal is fixed
in code: whether to expand the graph, how large a token budget to spend, and
when to stop are all decisions the model makes per question.

The answer step is not trusted. Every citation a model returns is checked
against the index, and one that does not resolve to real indexed lines is
rejected rather than passed through. An answer that cites nothing verifiable is
marked unsupported, because a confident answer pointing at the wrong location is
the failure mode this system exists to avoid.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field

from ..index import CodeIndex
from .llm import LLMClient, build_llm
from .tools import ToolBox, ToolResult

CITATION_RE = re.compile(r"^(?P<path>[^\s:]+):(?P<start>\d+)(?:-(?P<end>\d+))?$")

SYSTEM_PROMPT = """You are a code navigator. You answer questions about an \
unfamiliar repository and you support every claim with a file:line citation.

You work in a loop. At each step you return exactly one JSON object:

{"thought": "...", "tool": "<tool name>", "arguments": {...}}

Available tools:
{tools}

Plus one terminal action:
  final_answer(answer: string, citations: list of "path:start-end")

Rules:
- Call search_code before answering anything.
- If the question is about how code relates to other code, use expand_context \
to walk the call graph. Choose the token budget yourself: a narrow question \
deserves a small budget, a broad one a larger budget.
- read_chunk before you cite, so your citation covers code you actually saw.
- Cite only locations that appeared in a tool observation. Never invent a path \
or a line number.
- Stop as soon as you can answer. Do not spend steps you do not need.

Return only the JSON object, with no surrounding prose."""


@dataclass
class Step:
    index: int
    thought: str
    tool: str
    arguments: dict
    observation: dict
    ok: bool
    error: str | None = None

    def summary(self) -> str:
        status = "ok" if self.ok else f"error: {self.error}"
        return f"[{self.index}] {self.tool}({self.arguments}) -> {status}"


@dataclass
class Citation:
    raw: str
    path: str
    start_line: int
    end_line: int
    valid: bool
    reason: str = ""

    def to_dict(self) -> dict:
        return {
            "citation": self.raw,
            "path": self.path,
            "start_line": self.start_line,
            "end_line": self.end_line,
            "valid": self.valid,
            "reason": self.reason,
        }


@dataclass
class AgentAnswer:
    question: str
    answer: str
    citations: list[Citation] = field(default_factory=list)
    steps: list[Step] = field(default_factory=list)
    tools_used: list[str] = field(default_factory=list)
    stopped_reason: str = "final_answer"
    elapsed_seconds: float = 0.0
    llm_calls: int = 0
    expansion_trace: dict | None = None

    @property
    def valid_citations(self) -> list[Citation]:
        return [c for c in self.citations if c.valid]

    @property
    def is_supported(self) -> bool:
        """An answer with no citation that survives checking is unsupported."""
        return bool(self.valid_citations)

    @property
    def citation_validity(self) -> float:
        if not self.citations:
            return 0.0
        return len(self.valid_citations) / len(self.citations)

    def cited_paths(self) -> list[str]:
        return sorted({c.path for c in self.valid_citations})

    def to_dict(self) -> dict:
        return {
            "question": self.question,
            "answer": self.answer,
            "citations": [c.to_dict() for c in self.citations],
            "valid_citations": len(self.valid_citations),
            "citation_validity": round(self.citation_validity, 4),
            "is_supported": self.is_supported,
            "steps": [
                {"index": s.index, "tool": s.tool, "arguments": s.arguments,
                 "thought": s.thought, "ok": s.ok, "error": s.error}
                for s in self.steps
            ],
            "tools_used": self.tools_used,
            "stopped_reason": self.stopped_reason,
            "elapsed_seconds": round(self.elapsed_seconds, 3),
            "llm_calls": self.llm_calls,
            "expansion_trace": self.expansion_trace,
        }


class CodeNavigatorAgent:
    """ReAct loop over the seven-tool surface."""

    def __init__(
        self,
        index: CodeIndex,
        llm: LLMClient | None = None,
        max_steps: int = 6,
        token_budget: int = 6000,
    ):
        self.index = index
        self.llm = llm or build_llm("auto")
        self.max_steps = max_steps
        self.tools = ToolBox(index, default_token_budget=token_budget)

    # -- citation checking -------------------------------------------------
    def verify_citation(self, raw: str) -> Citation:
        """A citation is valid only if it points at real indexed lines.

        Checked against the chunk store rather than the filesystem, so a
        citation into a file the index never read cannot pass.
        """
        text = str(raw).strip().strip("`")
        match = CITATION_RE.match(text)
        if not match:
            return Citation(text, "", 0, 0, False, "malformed, expected path:start-end")

        path = match.group("path")
        start = int(match.group("start"))
        end = int(match.group("end") or start)

        chunks = self.index.chunks_in_file(path)
        if not chunks:
            return Citation(text, path, start, end, False, "path not in index")

        for chunk in chunks:
            if chunk.start_line <= start and end <= chunk.end_line:
                return Citation(text, path, start, end, True, "")

        file_end = max(c.end_line for c in chunks)
        if start > file_end:
            return Citation(
                text, path, start, end, False,
                f"line {start} beyond indexed extent ({file_end})",
            )
        return Citation(
            text, path, start, end, False, "span does not fall inside one chunk"
        )

    # -- prompting ---------------------------------------------------------
    def _state_payload(self, question: str, step: int, steps: list[Step],
                       top: list[dict]) -> str:
        observations = []
        for s in steps[-4:]:
            observations.append(
                {
                    "tool": s.tool,
                    "ok": s.ok,
                    "error": s.error,
                    "result": _condense(s.observation),
                }
            )
        return json.dumps(
            {
                "question": question,
                "step": step,
                "max_steps": self.max_steps,
                "observations": observations,
                "top_candidates": top[:8],
            },
            indent=2,
        )

    # -- main loop ---------------------------------------------------------
    def answer(self, question: str) -> AgentAnswer:
        started = time.time()
        system = SYSTEM_PROMPT.replace("{tools}", self.tools.describe())
        steps: list[Step] = []
        top: list[dict] = []
        llm_calls = 0
        result = AgentAnswer(question=question, answer="")

        for step_index in range(1, self.max_steps + 1):
            messages = [
                {"role": "system", "content": system},
                {"role": "user", "content": self._state_payload(
                    question, step_index, steps, top)},
            ]

            response = self.llm.complete(messages, temperature=0.0)
            llm_calls += 1
            action = response.json()

            if not action or "tool" not in action:
                steps.append(Step(step_index, "", "<malformed>", {},
                                  {"raw": response.text[:400]}, False,
                                  "model did not return a usable action"))
                continue

            tool = str(action.get("tool", ""))
            arguments = action.get("arguments") or {}
            thought = str(action.get("thought", ""))

            if tool == "final_answer":
                raw_citations = arguments.get("citations") or []
                result.citations = [self.verify_citation(c) for c in raw_citations]
                result.answer = str(arguments.get("answer", "")).strip()
                result.stopped_reason = "final_answer"
                steps.append(Step(step_index, thought, tool, arguments,
                                  {"citations": raw_citations}, True))
                break

            tool_result: ToolResult = self.tools.call(tool, arguments)
            steps.append(Step(step_index, thought, tool, arguments,
                              tool_result.payload, tool_result.ok,
                              tool_result.error))

            for key in ("results",):
                for item in tool_result.payload.get(key, []) or []:
                    if "chunk_id" in item and item not in top:
                        top.append(item)
        else:
            result.stopped_reason = "step_budget_exhausted"
            if not result.answer:
                result.answer = (
                    "Step budget exhausted before a final answer was produced."
                )

        result.steps = steps
        result.tools_used = list(dict.fromkeys(s.tool for s in steps))
        result.elapsed_seconds = time.time() - started
        result.llm_calls = llm_calls
        result.expansion_trace = self.tools.last_expansion_trace
        return result


def _condense(payload: dict, max_items: int = 5) -> dict:
    """Trim an observation so the prompt stays small as steps accumulate."""
    out = {}
    for key, value in payload.items():
        if isinstance(value, list):
            out[key] = value[:max_items]
            if len(value) > max_items:
                out[f"{key}_truncated"] = len(value) - max_items
        elif isinstance(value, str) and len(value) > 600:
            out[key] = value[:600] + " ..."
        else:
            out[key] = value
    return out
