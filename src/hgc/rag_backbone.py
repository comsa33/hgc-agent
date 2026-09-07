"""Classical RAG backbone on LangChain (FAISS retriever + AzureChatOpenAI).

A single-shot retrieve-then-generate pipeline: given a question, embed it,
pull top-K passages from a FAISS vectorstore, and call an LLM with the
concatenated passages as context. No agentic iteration — this is the
"Naive RAG" counterpart to the DSPy ReAct backbone.

The wrapper deliberately exposes a :class:`RAGBackbone.run(question, docs)`
interface identical to the agent backbones so that HGCRAGAgent can compose
it the same way HGCAgent composes HGCCoreAgent.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

_DEFAULT_PROMPT = """\
Use the following passages to answer the question. If the answer is not
present in the passages, say "I don't know."

Passages:
{context}

Question: {question}

Answer:"""


@dataclass
class RAGResult:
    """Structured result from a RAG pipeline run."""

    answer: str
    retrieved_docids: list[str]
    tokens: int
    wall_time: float
    n_iters: int = 1  # RAG is single-shot by definition


def build_faiss_index(docs: list[dict], embedder: Callable[[str], list[float]]) -> Any:
    """Build an in-memory FAISS store from a list of ``{"docid": str, "text": str}`` dicts.

    The embedder must return a list[float] per call. For Azure we can use
    ``AzureOpenAIEmbeddings(...).embed_query`` / ``embed_documents``.
    """
    from langchain_community.vectorstores import FAISS

    texts = [d["text"] for d in docs]
    metas = [{"docid": str(d["docid"])} for d in docs]
    embed_docs = getattr(embedder, "embed_documents", None)
    if embed_docs is None:
        raise TypeError(
            "embedder must expose embed_documents(texts) -> list[list[float]]; "
            "use LangChain AzureOpenAIEmbeddings or a compatible shim."
        )
    # FAISS.from_texts wants the embedder instance with embed_query/embed_documents.
    store = FAISS.from_texts(texts=texts, embedding=embedder, metadatas=metas)
    return store


def format_context(retrieved: list[Any]) -> tuple[str, list[str]]:
    """Concatenate retrieved Documents into a single context string.

    Returns ``(context_text, docids_list)``.
    """
    chunks: list[str] = []
    docids: list[str] = []
    for d in retrieved:
        md = getattr(d, "metadata", {}) or {}
        docid = str(md.get("docid", "unknown"))
        docids.append(docid)
        text = getattr(d, "page_content", str(d))
        chunks.append(f"[docid={docid}] {text}")
    return "\n\n".join(chunks), docids


class RAGBackbone:
    """Single-shot retrieve + generate pipeline with an ``.run()`` interface.

    Parameters
    ----------
    llm:
        Callable ``(prompt: str) -> str`` (or LangChain LLM with ``.invoke``).
    embedder:
        Embedder instance exposing ``embed_query`` / ``embed_documents``.
    top_k:
        Number of passages to retrieve per query. Default 5.
    prompt_template:
        Prompt containing ``{context}`` and ``{question}`` placeholders.
    """

    def __init__(
        self,
        llm: Any,
        embedder: Any,
        top_k: int = 5,
        prompt_template: str = _DEFAULT_PROMPT,
    ) -> None:
        self._llm = llm
        self._embedder = embedder
        self._top_k = top_k
        self._prompt = prompt_template

    def _call_llm(self, prompt: str) -> tuple[str, int]:
        """Call the LLM and return ``(answer_text, token_estimate)``."""
        result = self._llm.invoke(prompt) if hasattr(self._llm, "invoke") else self._llm(prompt)
        if hasattr(result, "content"):
            text = str(result.content)
        elif isinstance(result, list) and result:
            text = str(result[0])
        else:
            text = str(result)
        token_estimate = getattr(result, "usage_metadata", {}).get("total_tokens", len(text) // 4)
        return text.strip(), int(token_estimate)

    def run(self, question: str, docs: list[dict] | None = None) -> dict:
        """Execute the RAG pipeline and return an agent-compatible dict.

        A fresh per-query FAISS index is built when ``docs`` is provided.
        This mirrors the BCP / QASPER / FinanceBench data model where each
        query ships its own candidate document pool.
        """
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

        store = build_faiss_index(docs, self._embedder)
        retrieved = store.similarity_search(question, k=self._top_k)
        context, docids = format_context(retrieved)
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
