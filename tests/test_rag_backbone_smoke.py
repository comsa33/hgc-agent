"""Offline smoke tests for hgc.rag_backbone.RAGBackbone.

FAISS + AzureChatOpenAI are replaced with lightweight stubs so the tests
run without network or Azure credentials.
"""

from __future__ import annotations

from types import SimpleNamespace

from langchain_core.embeddings import Embeddings

from hgc.rag_backbone import RAGBackbone, format_context


class _StubEmbedder(Embeddings):
    """Hash-based embedder satisfying the LangChain Embeddings interface."""

    def embed_query(self, text: str) -> list[float]:
        h = abs(hash(text)) % 10_000
        return [float(h % 100), float((h // 100) % 100), 0.0]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_query(t) for t in texts]


class _StubLLM:
    """Deterministic LLM stub returning a content-bearing result object."""

    def __init__(self, answer: str = "Madrid"):
        self._answer = answer
        self.last_prompt: str | None = None

    def invoke(self, prompt: str):
        self.last_prompt = prompt
        return SimpleNamespace(content=self._answer, usage_metadata={"total_tokens": 42})


def test_format_context_concatenates_docid_prefix():
    d1 = SimpleNamespace(page_content="Madrid is the capital.", metadata={"docid": "5412"})
    d2 = SimpleNamespace(page_content="Lisbon is in Portugal.", metadata={"docid": "7777"})
    ctx, ids = format_context([d1, d2])
    assert "[docid=5412]" in ctx
    assert "[docid=7777]" in ctx
    assert ids == ["5412", "7777"]


def test_run_returns_empty_on_missing_docs():
    agent = RAGBackbone(llm=_StubLLM(), embedder=_StubEmbedder(), top_k=3)
    result = agent.run("What is X?", docs=None)
    assert result["answer"] == ""
    assert result["n_iters"] == 0
    assert result["trajectory"]["retrieved_docids"] == []


def test_run_builds_index_and_answers(tmp_path):
    """End-to-end call through FAISS + stub LLM, exercising the full path."""
    docs = [
        {"docid": "1", "text": "Madrid is the capital of Spain."},
        {"docid": "2", "text": "Paris is the capital of France."},
        {"docid": "3", "text": "Lisbon is the capital of Portugal."},
    ]
    llm = _StubLLM(answer="Madrid")
    agent = RAGBackbone(llm=llm, embedder=_StubEmbedder(), top_k=2)
    result = agent.run("What is the capital of Spain?", docs=docs)
    assert result["answer"] == "Madrid"
    assert len(result["trajectory"]["retrieved_docids"]) == 2
    assert result["n_iters"] == 1
    assert result["tokens"] == 42
    assert "Passages:" in llm.last_prompt
    assert "What is the capital of Spain?" in llm.last_prompt
