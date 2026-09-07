"""LLM-assisted hint extractor for HGC.

Extracts location / entity / strategy hints from a ReAct trajectory.
Philosophy: "Where, Not What" — hints NEVER contain the final answer text.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

_EXTRACTION_PROMPT_TMPL = """\
You are a hint extractor for an information-seeking agent.
Given a query and the agent's tool-use trajectory, extract navigation hints.

CRITICAL RULE: Do NOT include the final answer or any answer text in hints.
Hints are navigation aids only -- they point to WHERE information was found,
WHAT entities were encountered, and WHAT search strategies worked.

Return JSON with exactly this structure:
{{"location": [{{"content": "<docid or file ref>", "content_meta": {{}}}}],
 "entity":   [{{"content": "<named entity>",       "content_meta": {{}}}}],
 "strategy": [{{"content": "<tool call pattern>",  "content_meta": {{}}}}]}}

Rules:
- location: document IDs or resource references accessed during the trajectory
- entity: proper nouns, named entities seen in tool outputs (NOT the answer itself)
- strategy: search keywords or tool-call patterns that produced useful results
- If a list has no items, return an empty list []
- Return ONLY the JSON object, no prose

Query: {query}

Trajectory:
{trajectory_text}
"""


def _build_trajectory_text(trajectory: dict) -> str:
    """Render trajectory dict into a compact text block."""
    lines = []
    i = 0
    while True:
        thought = trajectory.get(f"thought_{i}")
        tool_name = trajectory.get(f"tool_name_{i}")
        tool_args = trajectory.get(f"tool_args_{i}")
        observation = trajectory.get(f"observation_{i}")
        if thought is None and tool_name is None:
            break
        if thought:
            lines.append(f"[{i}] Thought: {thought}")
        if tool_name:
            lines.append(f"[{i}] Tool: {tool_name}({tool_args})")
        if observation is not None:
            obs_str = str(observation)[:300]
            lines.append(f"[{i}] Obs: {obs_str}")
        i += 1
    return "\n".join(lines) if lines else "(empty trajectory)"


def _make_default_lm(deployment: str | None = None) -> Any:
    """Construct a dspy.LM for Azure gpt-4.1-mini (or gpt-5-mini via SUB env)."""
    import dspy  # deferred so tests can import without dspy installed

    dep = deployment or os.environ.get("AZURE_OPENAI_SUB_DEPLOYMENT", "gpt-4.1-mini")
    is_reasoning = dep.startswith("gpt-5")
    return dspy.LM(
        model=f"azure/{dep}",
        api_key=os.environ["AZURE_OPENAI_API_KEY"],
        api_base=os.environ["AZURE_OPENAI_ENDPOINT"].rstrip("/"),
        api_version=os.environ["AZURE_OPENAI_API_VERSION"],
        temperature=1.0 if is_reasoning else 0,
        max_tokens=16000 if is_reasoning else 800,
    )


_EMPTY: dict = {"location": [], "entity": [], "strategy": []}


class HintExtractor:
    """Extract structured hints from a ReAct trajectory using an LLM."""

    def __init__(self, lm: Any = None, deployment: str | None = None) -> None:
        self._lm = lm  # may be None — constructed lazily on first extract()
        self._deployment = deployment

    def _get_lm(self) -> Any:
        if self._lm is None:
            self._lm = _make_default_lm(self._deployment)
        return self._lm

    def extract(self, query: str, trajectory: dict, correct: bool) -> dict:
        """Return location/entity/strategy hints extracted from *trajectory*.

        Returns empty lists on LLM or parse failure.
        """
        traj_text = _build_trajectory_text(trajectory)
        prompt = _EXTRACTION_PROMPT_TMPL.format(
            query=query,
            trajectory_text=traj_text,
        )

        lm = self._get_lm()
        try:
            raw = lm(prompt)
            # dspy.LM returns a list of strings; take the first completion
            if isinstance(raw, list):
                raw = raw[0] if raw else ""
            if hasattr(raw, "completions"):
                raw = raw.completions[0]
        except Exception as exc:
            logger.warning("HintExtractor: LM call failed: %s", exc)
            return dict(_EMPTY)

        try:
            text = str(raw).strip()
            if text.startswith("```"):
                text = text.split("```")[1]
                if text.startswith("json"):
                    text = text[4:]
            parsed = json.loads(text)
        except (json.JSONDecodeError, ValueError) as exc:
            logger.warning("HintExtractor: JSON parse failed: %s | raw=%r", exc, str(raw)[:200])
            return dict(_EMPTY)

        result: dict = {"location": [], "entity": [], "strategy": []}
        for key in result:
            items = parsed.get(key, [])
            if not isinstance(items, list):
                continue
            for item in items:
                if not isinstance(item, dict):
                    continue
                content = str(item.get("content", "")).strip()
                if not content:
                    continue
                meta: dict = item.get("content_meta", {})
                if not isinstance(meta, dict):
                    meta = {}
                if not correct:
                    meta = {**meta, "polarity": "negative"}
                result[key].append({"content": content, "content_meta": meta})

        return result
