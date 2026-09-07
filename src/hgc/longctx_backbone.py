"""Long-context backbone — no retrieval, all provided docs injected into prompt.

Used for the FinanceBench × Oracle experiment (Patronus Islam et al., 2023):
evidence pages are pre-identified at dataset construction, so the backbone
skips any retrieval step and concatenates every doc into the prompt. This
matches Patronus's "Oracle" baseline setting and isolates memory-layer
attacks (cache / hint poisoning) from retrieval-layer confounds.

Exposes the same ``.run(question, docs)`` interface as
:class:`hgc.rag_backbone.RAGBackbone` so ``HGCRAGAgent`` can compose it in
place of RAG without any agent-side change.
"""

from __future__ import annotations

import time
from typing import Any

_DEFAULT_PROMPT = """\
Use the following documents to answer the question. If the answer is not
present in the documents, say "I don't know.".

Documents:
{context}

Question: {question}

Answer:"""


def _format_docs(docs: list[dict], max_chars: int) -> tuple[str, list[str]]:
    """Concatenate every doc's text into a single context string, bounded by max_chars."""
    chunks: list[str] = []
    docids: list[str] = []
    used = 0
    for d in docs:
        docid = str(d.get("docid", "unknown"))
        text = str(d.get("text", ""))
        piece = f"[docid={docid}] {text}"
        if used + len(piece) > max_chars:
            # Truncate the final piece and stop.
            remaining = max(0, max_chars - used)
            if remaining > 50:  # only add if meaningful
                chunks.append(piece[:remaining])
                docids.append(docid)
            break
        chunks.append(piece)
        docids.append(docid)
        used += len(piece)
    return "\n\n".join(chunks), docids


class LongCtxBackbone:
    """All-docs-injected backbone. No retrieval, single LLM call per query.

    Parameters
    ----------
    llm:
        LangChain ``.invoke``-style LM (or any callable returning a message with
        ``content`` / ``usage_metadata``).
    prompt_template:
        Prompt with ``{context}`` and ``{question}`` placeholders.
    max_context_chars:
        Hard cap on concatenated doc chars before truncation. FinanceBench
        evidence pages are typically 1–5k chars, so the default 60k is plenty
        for almost all queries; oversized docs get a truncated tail.
    """

    def __init__(
        self,
        llm: Any,
        prompt_template: str = _DEFAULT_PROMPT,
        max_context_chars: int = 60000,
    ) -> None:
        self._llm = llm
        self._prompt = prompt_template
        self._max_chars = max_context_chars

    def _call_llm(self, prompt: str) -> tuple[str, int]:
        result = self._llm.invoke(prompt) if hasattr(self._llm, "invoke") else self._llm(prompt)
        if hasattr(result, "content"):
            text = str(result.content)
        elif isinstance(result, list) and result:
            text = str(result[0])
        else:
            text = str(result)
        usage = getattr(result, "usage_metadata", {}) or {}
        tokens = int(usage.get("total_tokens", max(1, len(text) // 4)))
        return text.strip(), tokens

    def run(self, question: str, docs: list[dict] | None = None) -> dict:
        t0 = time.monotonic()
        if not docs:
            return {
                "answer": "",
                "trajectory": {"retrieved_docids": []},
                "tokens": 0,
                "wall_time": time.monotonic() - t0,
                "n_iters": 0,
                "cache_hit": False,
            }

        context, docids = _format_docs(docs, self._max_chars)
        prompt = self._prompt.format(context=context, question=question)
        answer, tokens = self._call_llm(prompt)
        return {
            "answer": answer,
            "trajectory": {"retrieved_docids": docids, "context": context},
            "tokens": tokens,
            "wall_time": time.monotonic() - t0,
            "n_iters": 1,
            "cache_hit": False,
        }
