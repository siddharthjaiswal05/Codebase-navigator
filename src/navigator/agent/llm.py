"""LLM access: multi-provider failover, caching, and an offline backend.

Provider routing goes through LiteLLM, which gives one call signature across
Groq and Gemini. Both offer free tiers, and both rate-limit aggressively, so the
client walks an ordered chain and moves to the next provider on failure. That is
what keeps the system running at zero infrastructure cost: when one free tier
throttles, the next serves the request.

Every call passes through the SQLite cache first, so a repeated prompt costs
nothing and replays identically.

`ScriptedLLM` is a deterministic offline backend. It runs the full agent loop
with no API key and no network by choosing tools from the question and the
retrieval state, which makes the control loop testable in CI and makes an
evaluation run reproducible on any machine.
"""

from __future__ import annotations

import json
import os
import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path

from .cache import ResponseCache, prompt_hash

# Ordered failover chain. Free tiers first, cheapest-to-slowest within each.
DEFAULT_CHAIN = [
    "groq/llama-3.3-70b-versatile",
    "groq/llama-3.1-8b-instant",
    "gemini/gemini-2.0-flash",
    "gemini/gemini-1.5-flash",
]

PROVIDER_KEYS = {
    "groq": "GROQ_API_KEY",
    "gemini": "GEMINI_API_KEY",
}


@dataclass
class LLMResponse:
    text: str
    model: str
    cached: bool = False
    attempts: list[str] = field(default_factory=list)
    latency_seconds: float = 0.0

    def json(self) -> dict | None:
        """Parse the response as JSON, tolerating a fenced code block."""
        return extract_json(self.text)


def extract_json(text: str) -> dict | None:
    """Pull the first JSON object out of a model response.

    Models wrap JSON in prose or fences often enough that requiring a bare
    object would fail on output that is otherwise correct.
    """
    if not text:
        return None

    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fenced:
        try:
            return json.loads(fenced.group(1))
        except json.JSONDecodeError:
            pass

    start = text.find("{")
    while start != -1:
        depth, in_string, escaped = 0, False, False
        for i in range(start, len(text)):
            char = text[i]
            if escaped:
                escaped = False
                continue
            if char == "\\":
                escaped = True
                continue
            if char == '"':
                in_string = not in_string
                continue
            if in_string:
                continue
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start:i + 1])
                    except json.JSONDecodeError:
                        break
        start = text.find("{", start + 1)
    return None


def available_providers() -> list[str]:
    return [name for name, env in PROVIDER_KEYS.items() if os.environ.get(env)]


class LLMClient(ABC):
    name: str = "llm"

    @abstractmethod
    def complete(self, messages: list[dict], temperature: float = 0.0,
                 max_tokens: int = 1024) -> LLMResponse:
        ...

    def stats(self) -> dict:
        return {"backend": self.name}


class LiteLLMClient(LLMClient):
    """Provider-agnostic client with an ordered failover chain and a cache."""

    name = "litellm"

    def __init__(
        self,
        chain: list[str] | None = None,
        cache: ResponseCache | None = None,
        cache_path: Path | str = ".navigator_cache/responses.db",
        max_retries_per_model: int = 1,
    ):
        self.chain = chain or DEFAULT_CHAIN
        self.cache = cache if cache is not None else ResponseCache(cache_path)
        self.max_retries_per_model = max_retries_per_model
        self.failover_events: list[dict] = []

    def _usable_chain(self) -> list[str]:
        """Drop models whose provider has no key configured."""
        usable = []
        for model in self.chain:
            provider = model.split("/")[0]
            env = PROVIDER_KEYS.get(provider)
            if env is None or os.environ.get(env):
                usable.append(model)
        return usable

    def complete(self, messages: list[dict], temperature: float = 0.0,
                 max_tokens: int = 1024) -> LLMResponse:
        import litellm

        chain = self._usable_chain()
        if not chain:
            raise RuntimeError(
                "No provider key found. Set GROQ_API_KEY or GEMINI_API_KEY, "
                "or run with the offline backend."
            )

        attempts: list[str] = []
        started = time.time()

        for model in chain:
            key = prompt_hash(
                model, messages, temperature=temperature, max_tokens=max_tokens
            )
            hit = self.cache.get(key)
            if hit is not None:
                return LLMResponse(
                    text=hit["text"], model=model, cached=True,
                    attempts=attempts + [model],
                    latency_seconds=time.time() - started,
                )

            for _ in range(self.max_retries_per_model + 1):
                attempts.append(model)
                try:
                    response = litellm.completion(
                        model=model,
                        messages=messages,
                        temperature=temperature,
                        max_tokens=max_tokens,
                    )
                    text = response["choices"][0]["message"]["content"] or ""
                    self.cache.put(
                        key,
                        model,
                        {
                            "messages": messages,
                            "temperature": temperature,
                            "max_tokens": max_tokens,
                        },
                        {"text": text},
                    )
                    return LLMResponse(
                        text=text, model=model, cached=False, attempts=attempts,
                        latency_seconds=time.time() - started,
                    )
                except Exception as exc:
                    self.failover_events.append(
                        {"model": model, "error": type(exc).__name__, "detail": str(exc)[:200]}
                    )
                    break  # move to the next provider rather than hammering this one

        raise RuntimeError(
            f"All {len(chain)} providers failed. Attempts: {attempts}. "
            f"Last errors: {self.failover_events[-3:]}"
        )

    def stats(self) -> dict:
        return {
            "backend": self.name,
            "chain": self.chain,
            "usable_chain": self._usable_chain(),
            "failover_events": len(self.failover_events),
            "cache": self.cache.stats(),
        }


class ScriptedLLM(LLMClient):
    """Deterministic offline policy over the same tool surface.

    This is not a language model. It is a rule-based controller that emits the
    same JSON actions the prompt asks a model for, so the ReAct loop, the tool
    surface, the citation checks and the evaluation harness can all be exercised
    end to end with no key, no network and no variance between runs.

    Its policy mirrors the instructions given to a real model: search first,
    widen with graph expansion when the top hits look weak or the question is
    about relationships, read the strongest candidate, then answer with
    citations.
    """

    name = "scripted"

    # Deliberately excludes bare interrogatives such as "where" and "who":
    # they appear in almost every question and would make every question look
    # relational, which collapses the traversal decision into a constant.
    RELATIONAL = (
        "call", "calls", "called", "caller", "callers", "callee", "invoke",
        "invoked", "uses", "used by", "depend", "depends", "flow", "trace",
        "reach", "chain", "downstream", "upstream", "relate",
    )

    def __init__(self):
        self.calls = 0

    def complete(self, messages: list[dict], temperature: float = 0.0,
                 max_tokens: int = 1024) -> LLMResponse:
        self.calls += 1
        state = extract_json(messages[-1]["content"]) or {}
        question = str(state.get("question", ""))
        step = int(state.get("step", 1))
        observations = state.get("observations", [])
        top = state.get("top_candidates", [])

        lowered = question.lower()
        relational = any(word in lowered for word in self.RELATIONAL)
        tools_used = {o.get("tool") for o in observations}

        if "search_code" not in tools_used:
            action = {
                "thought": "Locate candidate code with hybrid retrieval before "
                           "committing to any file.",
                "tool": "search_code",
                "arguments": {"query": question, "top_k": 8},
            }
        elif relational and "expand_context" not in tools_used:
            action = {
                "thought": "The question is about relationships between "
                           "functions, so walk the call graph outward from the "
                           "seeds rather than trusting text matches alone.",
                "tool": "expand_context",
                "arguments": {"query": question, "token_budget": 4000, "max_hops": 2},
            }
        elif top and "read_chunk" not in tools_used:
            action = {
                "thought": "Read the strongest candidate in full so the citation "
                           "covers code actually inspected.",
                "tool": "read_chunk",
                "arguments": {"chunk_id": top[0]["chunk_id"]},
            }
        elif relational and top and "find_callers" not in tools_used:
            action = {
                "thought": "Confirm the relationship direction by listing callers.",
                "tool": "find_callers",
                "arguments": {"chunk_id": top[0]["chunk_id"]},
            }
        else:
            citations = [c["citation"] for c in top[:3]]
            names = [c["qualified_name"] for c in top[:3]]
            if names:
                answer = (
                    f"The relevant implementation is {names[0]}"
                    + (f", with supporting code in {', '.join(names[1:])}" if names[1:] else "")
                    + "."
                )
            else:
                answer = "No code in this repository matches the question."
            action = {
                "thought": "Enough evidence gathered; answer with citations.",
                "tool": "final_answer",
                "arguments": {"answer": answer, "citations": citations},
            }

        if step >= int(state.get("max_steps", 6)) and action["tool"] != "final_answer":
            citations = [c["citation"] for c in top[:3]]
            action = {
                "thought": "Step budget reached; answer from what is gathered.",
                "tool": "final_answer",
                "arguments": {
                    "answer": "Answering from the evidence gathered so far.",
                    "citations": citations,
                },
            }

        return LLMResponse(text=json.dumps(action), model="scripted", cached=False)

    def stats(self) -> dict:
        return {"backend": self.name, "calls": self.calls}


def build_llm(
    backend: str = "auto",
    chain: list[str] | None = None,
    cache_path: Path | str = ".navigator_cache/responses.db",
) -> LLMClient:
    """Pick a backend.

    "auto" uses a real provider when a key is present and the offline policy
    otherwise, so the system runs everywhere without configuration.
    """
    if backend == "scripted":
        return ScriptedLLM()
    if backend == "litellm":
        return LiteLLMClient(chain=chain, cache_path=cache_path)
    if backend != "auto":
        raise ValueError(f"unknown LLM backend: {backend}")

    if available_providers():
        try:
            import litellm  # noqa: F401
            return LiteLLMClient(chain=chain, cache_path=cache_path)
        except Exception:
            pass
    return ScriptedLLM()
