"""Shared factory helpers for BCP and HotpotQA experiment phases (US-025).

Extracted from run_smoke_n10.py and run_smoke_hotpot.py so that both the new
unified run_experiment.py and (temporarily) the legacy smoke shims can import
common logic.  No API calls here — just pure factory builders.
"""

from __future__ import annotations

import logging
import os
import shutil
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Cost constants (shared across scripts)
# ---------------------------------------------------------------------------

COST_PER_1K_TOKENS = 0.005
INPUT_COST_PER_1M = 2.50
OUTPUT_COST_PER_1M = 10.00

# ---------------------------------------------------------------------------
# Ablation knobs (set by run.py via configure_* helpers before phase execution)
# ---------------------------------------------------------------------------


def _enabled_gate_set() -> set[str]:
    """Read the current gate variant from the ``HGC_GATE_VARIANT`` env var.

    Values: ``full`` (default) → {g1,g2,g3}; ``g1g2`` → drop G3;
    ``g1g3`` → drop G2. Set before ``run.py`` starts to ablate.
    """
    variant = os.environ.get("HGC_GATE_VARIANT", "full")
    if variant == "g1g2":
        return {"g1", "g2"}
    if variant == "g1g3":
        return {"g1", "g3"}
    return {"g1", "g2", "g3"}


def _contamination_fraction() -> float:
    """Read contamination fraction from ``HGC_CONTAMINATION_FRACTION`` env var.

    Default 0.20. Overrides all hardcoded ``fraction=0.20`` call sites when
    set.
    """
    return float(os.environ.get("HGC_CONTAMINATION_FRACTION", "0.20"))


def _cache_sim_threshold() -> float:
    """Answer-cache similarity threshold tau, from ``HGC_CACHE_SIM_THRESHOLD``.

    Default ``0.85``. Read at serve time only: the warm-up seeder appends
    every judge-correct P1 answer without consulting it, so changing tau never
    requires re-running P1 — the seeded cache can be copied between runs.
    """
    return float(os.environ.get("HGC_CACHE_SIM_THRESHOLD", "0.85"))


def _top_k_hints() -> int:
    """Location hints retrieved per query, from ``HGC_TOP_K_HINTS``.

    Default ``3``. The agent over-fetches ``3 * top_k`` candidates and keeps
    the first ``top_k`` of type ``location``, so this bounds what reaches G1.
    """
    return int(os.environ.get("HGC_TOP_K_HINTS", "3"))


def _verifier_max_doc_chars() -> int:
    """G3 prompt truncation, from ``HGC_VERIFIER_MAX_DOC_CHARS``. Default 2500.

    Applies to the LLM prompt only — the containment fast path scans the whole
    document — so this knob moves nothing on queries that short-circuit.
    """
    return int(os.environ.get("HGC_VERIFIER_MAX_DOC_CHARS", "2500"))


def _verifier_containment_min_chars() -> int:
    """G3 containment floor, from ``HGC_VERIFIER_CONTAINMENT_MIN_CHARS``.

    Default ``5``. Answers shorter than this skip the substring fast path and
    go to the LLM, which keeps "yes"/"no" from matching spuriously.
    """
    return int(os.environ.get("HGC_VERIFIER_CONTAINMENT_MIN_CHARS", "5"))


def _verifier_predicate() -> str:
    """G3 verification predicate, from ``HGC_VERIFIER_PREDICATE``.

    Default ``support`` (released behaviour): does the document carry evidence
    consistent with the cached answer? ``answerhood`` asks instead whether the
    document shows that answer to be responsive to the question. The two agree
    on foreign text and diverge on a poison lifted verbatim from the victim's
    own document, which is grounded but answers nothing.
    """
    return os.environ.get("HGC_VERIFIER_PREDICATE", "support")


def _gate_lm():
    """Optional dedicated LM for the G3 support verifier.

    When ``HGC_GATE_LM`` is set (e.g. ``gemma4:31b``), build a DSPy LM
    pointing at the Ollama Cloud OpenAI-compatible endpoint so the gate
    verification runs on an open-weight model while the backbone keeps the
    Azure model configured by ``run.py``. Unset → ``None`` (global LM,
    released default, unchanged). ``cache=False`` mirrors ``configure_lm``
    to avoid cross-run contamination.
    """
    model = os.environ.get("HGC_GATE_LM")
    if not model:
        return None
    import dspy

    api_base = os.environ.get("OLLAMA_CLOUD_BASE_URL", "https://ollama.com/v1")
    api_key = os.environ.get("OLLAMA_CLOUD_API_KEY", "")
    # Fail fast on a misconfigured gate: the hosted Ollama Cloud endpoint
    # requires a key. Without one every gate call would raise, and
    # SupportVerifier swallows LM errors into a conservative ``False`` — which
    # would silently reject every cache hit and corrupt the experiment rather
    # than error out. A local/self-hosted endpoint needs no key, so only
    # enforce this for the cloud host.
    if not api_key and "ollama.com" in api_base:
        raise RuntimeError(
            f"HGC_GATE_LM={model!r} targets the Ollama Cloud endpoint "
            f"({api_base}) but OLLAMA_CLOUD_API_KEY is empty. Set the key, or "
            "point OLLAMA_CLOUD_BASE_URL at a keyless local endpoint."
        )

    return dspy.LM(
        f"openai/{model}",
        api_base=api_base,
        api_key=api_key,
        cache=False,
    )


def _strict_no_hint() -> bool:
    """Strict-fallback variant for the ``cache_hit_no_hint`` path.

    When ``HGC_STRICT_NO_HINT=1``, an uncovered cache hit is rerouted through
    the core agent instead of returning the cached answer (strict-fallback ablation,
    requested in the previous review round). Default (unset / ``0``) preserves
    the permissive released policy.
    """
    return os.environ.get("HGC_STRICT_NO_HINT", "0").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


# ---------------------------------------------------------------------------
# Containment judge helper
# ---------------------------------------------------------------------------


def make_containment_judge(gold: str):
    def judge(question: str, answer: str) -> bool:  # noqa: ARG001
        return gold.strip().lower() in answer.lower()

    return judge


# ---------------------------------------------------------------------------
# LangChain embedder adapter
# ---------------------------------------------------------------------------


def _make_langchain_embedder(emb: Any) -> Any:
    """Wrap an ``hgc.embeddings.Embedder`` as a LangChain ``Embeddings`` instance.

    FAISS dispatches on ``isinstance(embedding, Embeddings)``: if it doesn't
    match, FAISS falls back to calling the embedder as a function, which
    raises ``'object is not callable'``. Subclassing ``Embeddings`` routes
    queries/docs through ``embed_query``/``embed_documents`` instead.
    """
    from langchain_core.embeddings import Embeddings

    class _LangChainEmbedderAdapter(Embeddings):
        def embed_query(self, text: str) -> list[float]:
            v = emb.embed(text)
            return v.tolist() if hasattr(v, "tolist") else list(v)

        def embed_documents(self, texts: list[str]) -> list[list[float]]:
            return [self.embed_query(t) for t in texts]

    return _LangChainEmbedderAdapter()


# ---------------------------------------------------------------------------
# Tool factories
# ---------------------------------------------------------------------------


def make_tools_bcp(qr: dict) -> list:
    """Build the 4 standard document tools for a BCP query record.

    BCP records carry docs across three keys: gold_docs, evidence_docs,
    negative_docs.
    """
    docs = qr.get("gold_docs", []) + qr.get("evidence_docs", []) + qr.get("negative_docs", [])
    doc_by_id = {d["docid"]: d for d in docs}

    def list_document_ids() -> list:
        return [d["docid"] for d in docs]

    def get_document_title(docid: str) -> str:
        d = doc_by_id.get(docid)
        if not d:
            return f"ERROR: docid {docid} not found"
        return d.get("text", "").split("\n", 1)[0][:300]

    def get_document_snippet(docid: str, char_start: int = 0, char_len: int = 2000) -> str:
        d = doc_by_id.get(docid)
        if not d:
            return f"ERROR: docid {docid} not found"
        return d.get("text", "")[char_start : char_start + char_len]

    def search_documents(keyword: str, max_hits: int = 10) -> list:
        """Substring search; returns list of {docid, snippet}."""
        hits = []
        kl = keyword.lower()
        for d in docs:
            idx = d.get("text", "").lower().find(kl)
            if idx != -1:
                t = d["text"]
                hits.append(
                    {
                        "docid": d["docid"],
                        "snippet": t[max(0, idx - 80) : idx + 120],
                    }
                )
                if len(hits) >= max_hits:
                    break
        return hits

    return [list_document_ids, get_document_title, get_document_snippet, search_documents]


def make_tools_hotpot(qr: dict) -> list:
    """Build the 4 standard document tools for a HotpotQA query record.

    HotpotQA records carry docs directly under "docs".
    """
    docs = qr.get("docs", [])
    doc_by_id = {d["docid"]: d for d in docs}

    def list_document_ids() -> list:
        return [d["docid"] for d in docs]

    def get_document_title(docid: str) -> str:
        d = doc_by_id.get(docid)
        if not d:
            return f"ERROR: docid {docid!r} not found"
        return d.get("text", "").split("\n", 1)[0][:300]

    def get_document_snippet(docid: str, char_start: int = 0, char_len: int = 2000) -> str:
        d = doc_by_id.get(docid)
        if not d:
            return f"ERROR: docid {docid!r} not found"
        return d.get("text", "")[char_start : char_start + char_len]

    def search_documents(keyword: str, max_hits: int = 10) -> list:
        """Substring search; returns list of {docid, snippet}."""
        hits = []
        kl = keyword.lower()
        for d in docs:
            idx = d.get("text", "").lower().find(kl)
            if idx != -1:
                t = d["text"]
                hits.append(
                    {
                        "docid": d["docid"],
                        "snippet": t[max(0, idx - 80) : idx + 120],
                    }
                )
                if len(hits) >= max_hits:
                    break
        return hits

    return [list_document_ids, get_document_title, get_document_snippet, search_documents]


# ---------------------------------------------------------------------------
# Phase factory builders — BCP
# ---------------------------------------------------------------------------


def build_hgc_core_factory_bcp(
    embedder: Any,
    store: Any,
    extractor: Any,
    scope_of=None,
):
    """Return a HGCCoreAgent factory for BCP queries.

    Parameters
    ----------
    scope_of:
        Optional callable ``(qr) -> str`` that returns the scope_id string.
        Defaults to ``f"bcp_query_{qr['query_id']}"``.
    """
    from hgc.hgc_core import HGCCoreAgent

    def factory(qr: dict, tools: list):
        judge_fn = make_containment_judge(qr.get("answer", ""))
        scope_id = scope_of(qr) if scope_of else f"bcp_query_{qr['query_id']}"
        return HGCCoreAgent(
            tools=tools,
            store=store,
            embedder=embedder,
            extractor=extractor,
            judge=judge_fn,
            max_iters=15,
            scope_id=scope_id,
        )

    return factory


def build_rag_factory_bcp(embedder: Any, doc_emb_cache: Any = None):
    """NaiveRAGAgent factory for BCP (with optional doc embedding cache)."""
    from hgc.baselines import NaiveRAGAgent

    def factory(qr: dict, tools: list):
        judge_fn = make_containment_judge(qr.get("answer", ""))
        return NaiveRAGAgent(embedder=embedder, judge=judge_fn, doc_emb_cache=doc_emb_cache)

    return factory


def build_ac_factory(embedder: Any, p1_traj_dir: Path):
    """AnswerCacheAgent factory seeded from P1 trajectories."""
    from hgc.baselines import AnswerCacheAgent
    from hgc.runner import seed_answer_cache_from_p1

    seed_agent = AnswerCacheAgent(
        tools=[],
        embedder=lambda text: embedder.embed(text),
        judge=lambda q, a: False,
        sim_threshold=_cache_sim_threshold(),
        max_iters=15,
    )
    p1_trajs = sorted(p1_traj_dir.glob("trajectory_q*.json")) if p1_traj_dir.exists() else []
    seed_answer_cache_from_p1(seed_agent, p1_trajs, lambda text: embedder.embed(text))
    seeded_cache = list(seed_agent._cache)

    def factory(qr: dict, tools: list):
        judge_fn = make_containment_judge(qr.get("answer", ""))
        agent = AnswerCacheAgent(
            tools=tools,
            embedder=lambda text: embedder.embed(text),
            judge=judge_fn,
            sim_threshold=_cache_sim_threshold(),
            max_iters=15,
        )
        agent._cache = list(seeded_cache)
        return agent

    return factory


def build_tc_factory(embedder: Any, p1_traj_dir: Path):
    """TrajectoryCacheAgent factory seeded from P1 trajectories."""
    from hgc.baselines import TrajectoryCacheAgent
    from hgc.runner import seed_trajectory_cache_from_p1

    seed_agent = TrajectoryCacheAgent(
        tools=[],
        embedder=lambda text: embedder.embed(text),
        judge=lambda q, a: False,
        sim_threshold=0.7,
        top_k=3,
        max_iters=15,
    )
    p1_trajs = sorted(p1_traj_dir.glob("trajectory_q*.json")) if p1_traj_dir.exists() else []
    seed_trajectory_cache_from_p1(seed_agent, p1_trajs, lambda text: embedder.embed(text))
    seeded_store = list(seed_agent._store)

    def factory(qr: dict, tools: list):
        judge_fn = make_containment_judge(qr.get("answer", ""))
        agent = TrajectoryCacheAgent(
            tools=tools,
            embedder=lambda text: embedder.embed(text),
            judge=judge_fn,
            sim_threshold=0.7,
            top_k=3,
            max_iters=15,
        )
        agent._store = list(seeded_store)
        return agent

    return factory


def build_m0_factory(
    p1_traj_dir: Path, mem0_config: dict | None = None, user_id: str = "p3_m0_smoke"
):
    """Mem0ReActAgent factory seeded from P1.

    Single shared mem0.Memory instance to avoid Qdrant lock collisions.
    """
    from hgc.baselines import Mem0ReActAgent
    from hgc.runner import seed_mem0_from_p1

    if mem0_config is not None:
        _config = mem0_config
        if isinstance(_config, dict):
            from mem0.configs.base import MemoryConfig as _MemoryConfig

            _config = _MemoryConfig(**_config)
        from mem0 import Memory as _Memory

        shared_memory = _Memory(config=_config)
    else:
        from mem0 import Memory as _Memory

        shared_memory = _Memory()

    p1_trajs = sorted(p1_traj_dir.glob("trajectory_q*.json")) if p1_traj_dir.exists() else []
    seed_mem0_from_p1(shared_memory, p1_trajs, user_id=user_id)

    def factory(qr: dict, tools: list):
        judge_fn = make_containment_judge(qr.get("answer", ""))
        return Mem0ReActAgent(
            tools=tools,
            judge=judge_fn,
            user_id=user_id,
            mem0_memory=shared_memory,
            max_iters=15,
        )

    return factory


def build_p4_ac_factory(embedder: Any, p1_traj_dir: Path, seed: int = 42, fraction: float = 0.20):
    """P4-AC: AnswerCacheAgent seeded from P1 then contaminated via 20% cross-swap."""
    from hgc.baselines import AnswerCacheAgent
    from hgc.contaminators.cross_swap import contaminate_answer_cache_from_p1

    seed_agent = AnswerCacheAgent(
        tools=[],
        embedder=lambda text: embedder.embed(text),
        judge=lambda q, a: False,
        sim_threshold=_cache_sim_threshold(),
        max_iters=15,
    )
    p1_trajs = sorted(p1_traj_dir.glob("trajectory_q*.json")) if p1_traj_dir.exists() else []
    corrupted_indices = contaminate_answer_cache_from_p1(
        seed_agent, p1_trajs, lambda text: embedder.embed(text), seed=seed, fraction=fraction
    )
    logger.info("P4-AC: contaminated %d cache entries", len(corrupted_indices))
    seeded_cache = list(seed_agent._cache)

    def factory(qr: dict, tools: list):
        judge_fn = make_containment_judge(qr.get("answer", ""))
        agent = AnswerCacheAgent(
            tools=tools,
            embedder=lambda text: embedder.embed(text),
            judge=judge_fn,
            sim_threshold=_cache_sim_threshold(),
            max_iters=15,
        )
        agent._cache = list(seeded_cache)
        return agent

    return factory


def build_p4_tc_factory(embedder: Any, p1_traj_dir: Path, seed: int = 42, fraction: float = 0.20):
    """P4-TC: TrajectoryCacheAgent seeded from P1 then contaminated via 20% cross-swap."""
    from hgc.baselines import TrajectoryCacheAgent
    from hgc.contaminators.cross_swap import contaminate_trajectory_cache_from_p1

    seed_agent = TrajectoryCacheAgent(
        tools=[],
        embedder=lambda text: embedder.embed(text),
        judge=lambda q, a: False,
        sim_threshold=0.7,
        top_k=3,
        max_iters=15,
    )
    p1_trajs = sorted(p1_traj_dir.glob("trajectory_q*.json")) if p1_traj_dir.exists() else []
    corrupted_indices = contaminate_trajectory_cache_from_p1(
        seed_agent, p1_trajs, lambda text: embedder.embed(text), seed=seed, fraction=fraction
    )
    logger.info("P4-TC: contaminated %d store entries", len(corrupted_indices))
    seeded_store = list(seed_agent._store)

    def factory(qr: dict, tools: list):
        judge_fn = make_containment_judge(qr.get("answer", ""))
        agent = TrajectoryCacheAgent(
            tools=tools,
            embedder=lambda text: embedder.embed(text),
            judge=judge_fn,
            sim_threshold=0.7,
            top_k=3,
            max_iters=15,
        )
        agent._store = list(seeded_store)
        return agent

    return factory


def build_p4_m0_factory(
    p1_traj_dir: Path,
    mem0_config: dict | None = None,
    user_id: str = "p4_m0",
    seed: int = 42,
    fraction: float = 0.20,
):
    """P4-M0: Mem0ReActAgent seeded from P1 then contaminated via 20% cross-swap."""
    from hgc.baselines import Mem0ReActAgent
    from hgc.contaminators.cross_swap import contaminate_mem0_from_p1

    if mem0_config is not None:
        _config = mem0_config
        if isinstance(_config, dict):
            from mem0.configs.base import MemoryConfig as _MemoryConfig

            _config = _MemoryConfig(**_config)
        from mem0 import Memory as _Memory

        shared_memory = _Memory(config=_config)
    else:
        from mem0 import Memory as _Memory

        shared_memory = _Memory()

    p1_trajs = sorted(p1_traj_dir.glob("trajectory_q*.json")) if p1_traj_dir.exists() else []
    corrupted_ids = contaminate_mem0_from_p1(
        shared_memory, p1_trajs, seed=seed, fraction=fraction, user_id=user_id
    )
    logger.info("P4-M0: contaminated %d mem0 entries", len(corrupted_ids))

    def factory(qr: dict, tools: list):
        judge_fn = make_containment_judge(qr.get("answer", ""))
        return Mem0ReActAgent(
            tools=tools,
            judge=judge_fn,
            user_id=user_id,
            mem0_memory=shared_memory,
            max_iters=15,
        )

    return factory


def build_lcs_factory(doc_order: str = "primacy"):
    """LongContextStuffAgent factory (primacy or middle doc ordering).

    Uses max_input_tokens=500_000 to cap per-call cost.
    """
    from hgc.baselines import LongContextStuffAgent

    def factory(qr: dict, tools: list):  # noqa: ARG001
        judge_fn = make_containment_judge(qr.get("answer", ""))
        return LongContextStuffAgent(
            judge=judge_fn,
            max_input_tokens=500_000,
            max_chars_per_doc=8_000,
            doc_order=doc_order,
        )

    return factory


def build_hybrid_factory_bcp(
    embedder: Any,
    p1_db: Path,
    p1_traj_dir: Path,
    extractor: Any,
    scope_of=None,
) -> Callable:
    """Clean Hybrid (P3-Hybrid): AC fast path + hint re-verification.

    Uses the same AnswerCacheAgent seed logic as build_ac_factory (sim_threshold=0.85,
    seeded from P1 trajectories) AND the same HGCCoreAgent wiring as
    build_hgc_core_factory_bcp (HintStore opened from p1_db). Returns a factory that
    builds a HGCAgent per query.
    """
    from hgc.baselines import AnswerCacheAgent
    from hgc.gate import CompositeGate, SupportVerifier
    from hgc.hgc_agent import HGCAgent
    from hgc.hgc_core import HGCCoreAgent
    from hgc.memory import HintStore
    from hgc.runner import seed_answer_cache_from_p1

    # Seed AC template from P1 trajectories
    seed_agent = AnswerCacheAgent(
        tools=[],
        embedder=lambda text: embedder.embed(text),
        judge=lambda q, a: False,
        sim_threshold=_cache_sim_threshold(),
        max_iters=15,
    )
    p1_trajs = sorted(p1_traj_dir.glob("trajectory_q*.json")) if p1_traj_dir.exists() else []
    seed_answer_cache_from_p1(seed_agent, p1_trajs, lambda text: embedder.embed(text))
    seeded_cache = list(seed_agent._cache)
    logger.info("P3-Hybrid: seeded AC with %d cache entries from P1", len(seeded_cache))

    # Shared HintStore from P1 DB + one shared LLM verifier (DSPy Predict reused).
    store = HintStore(db_path=str(p1_db))
    verifier = SupportVerifier(
        lm=_gate_lm(),
        max_doc_chars=_verifier_max_doc_chars(),
        containment_min_chars=_verifier_containment_min_chars(),
        predicate=_verifier_predicate(),
    )

    def factory(qr: dict, tools: list):
        judge_fn = make_containment_judge(qr.get("answer", ""))
        scope_id = scope_of(qr) if scope_of else f"bcp_query_{qr['query_id']}"

        ac_agent = AnswerCacheAgent(
            tools=[],
            embedder=lambda text: embedder.embed(text),
            judge=make_containment_judge(qr.get("answer", "")),
            sim_threshold=_cache_sim_threshold(),
            max_iters=15,
        )
        ac_agent._cache = list(seeded_cache)

        hgc_agent = HGCCoreAgent(
            tools=tools,
            store=store,
            embedder=embedder,
            extractor=extractor,
            judge=judge_fn,
            max_iters=15,
            scope_id=scope_id,
        )

        return HGCAgent(
            ac_agent=ac_agent,
            core_agent=hgc_agent,
            store=store,
            embedder=lambda text: embedder.embed(text),
            gate=CompositeGate(support_verifier=verifier, enabled=_enabled_gate_set()),
            top_k_hints=_top_k_hints(),
            strict_no_hint=_strict_no_hint(),
        )

    return factory


def build_e4_ac_factory(embedder: Any, p1_traj_dir: Path, seed: int = 42, fraction: float = 0.20):
    """E4-AC: AnswerCacheAgent seeded from P1 then contaminated via 20% entity-swap."""
    from hgc.baselines import AnswerCacheAgent
    from hgc.contaminators.entity_swap import corrupt_answer_cache_entity_swap

    seed_agent = AnswerCacheAgent(
        tools=[],
        embedder=lambda text: embedder.embed(text),
        judge=lambda q, a: False,
        sim_threshold=_cache_sim_threshold(),
        max_iters=15,
    )
    p1_trajs = sorted(p1_traj_dir.glob("trajectory_q*.json")) if p1_traj_dir.exists() else []
    corrupted_indices = corrupt_answer_cache_entity_swap(
        seed_agent, p1_trajs, lambda text: embedder.embed(text), seed=seed, fraction=fraction
    )
    logger.info("E4-AC: contaminated %d cache entries (entity-swap)", len(corrupted_indices))
    seeded_cache = list(seed_agent._cache)

    def factory(qr: dict, tools: list):
        judge_fn = make_containment_judge(qr.get("answer", ""))
        agent = AnswerCacheAgent(
            tools=tools,
            embedder=lambda text: embedder.embed(text),
            judge=judge_fn,
            sim_threshold=_cache_sim_threshold(),
            max_iters=15,
        )
        agent._cache = list(seeded_cache)
        return agent

    return factory


def build_e4_hybrid_factory(
    embedder: Any,
    p1_db: Path,
    p1_traj_dir: Path,
    extractor: Any,
    e4_db: Path,
    scope_of=None,
    seed: int = 42,
    fraction: float = 0.20,
) -> Callable:
    """E4-Hybrid: entity-swap on BOTH AnswerCache AND HintStore."""
    from hgc.baselines import AnswerCacheAgent
    from hgc.contaminators.entity_swap import (
        corrupt_answer_cache_entity_swap,
        corrupt_hint_store_entity_swap,
    )
    from hgc.gate import CompositeGate, SupportVerifier
    from hgc.hgc_agent import HGCAgent
    from hgc.hgc_core import HGCCoreAgent
    from hgc.memory import HintStore

    # Step 1: Seed + contaminate AC (entity-swap)
    seed_agent = AnswerCacheAgent(
        tools=[],
        embedder=lambda text: embedder.embed(text),
        judge=lambda q, a: False,
        sim_threshold=_cache_sim_threshold(),
        max_iters=15,
    )
    p1_trajs = sorted(p1_traj_dir.glob("trajectory_q*.json")) if p1_traj_dir.exists() else []
    corrupted_indices = corrupt_answer_cache_entity_swap(
        seed_agent, p1_trajs, lambda text: embedder.embed(text), seed=seed, fraction=fraction
    )
    logger.info("E4-Hybrid AC: contaminated %d cache entries (entity-swap)", len(corrupted_indices))
    seeded_cache = list(seed_agent._cache)

    # Step 2: Copy p1_db to e4_db and contaminate HintStore (entity-swap)
    e4_db.parent.mkdir(parents=True, exist_ok=True)
    if p1_db.exists() and not e4_db.exists():
        shutil.copy2(str(p1_db), str(e4_db))
    elif not p1_db.exists():
        logger.warning("E4-Hybrid: P1 store missing at %s; using empty store", p1_db)

    contaminated_store = HintStore(db_path=str(e4_db))
    corrupted_ids = corrupt_hint_store_entity_swap(contaminated_store, seed=seed, fraction=fraction)
    logger.info("E4-Hybrid HintStore: contaminated %d hints (entity-swap)", len(corrupted_ids))

    verifier = SupportVerifier(
        lm=_gate_lm(),
        max_doc_chars=_verifier_max_doc_chars(),
        containment_min_chars=_verifier_containment_min_chars(),
        predicate=_verifier_predicate(),
    )

    def factory(qr: dict, tools: list):
        judge_fn = make_containment_judge(qr.get("answer", ""))
        scope_id = scope_of(qr) if scope_of else f"bcp_query_{qr['query_id']}"

        ac_agent = AnswerCacheAgent(
            tools=[],
            embedder=lambda text: embedder.embed(text),
            judge=make_containment_judge(qr.get("answer", "")),
            sim_threshold=_cache_sim_threshold(),
            max_iters=15,
        )
        ac_agent._cache = list(seeded_cache)

        hgc_agent = HGCCoreAgent(
            tools=tools,
            store=contaminated_store,
            embedder=embedder,
            extractor=extractor,
            judge=judge_fn,
            max_iters=15,
            scope_id=scope_id,
        )

        return HGCAgent(
            ac_agent=ac_agent,
            core_agent=hgc_agent,
            store=contaminated_store,
            embedder=lambda text: embedder.embed(text),
            gate=CompositeGate(support_verifier=verifier, enabled=_enabled_gate_set()),
            top_k_hints=_top_k_hints(),
            strict_no_hint=_strict_no_hint(),
        )

    return factory


def build_t4_ac_factory(embedder: Any, p1_traj_dir: Path, seed: int = 42, fraction: float = 0.20):
    """T4-AC: AnswerCacheAgent seeded from P1 then contaminated via 20% typo-mutation."""
    from hgc.baselines import AnswerCacheAgent
    from hgc.contaminators.typo_mutation import corrupt_answer_cache_typo

    seed_agent = AnswerCacheAgent(
        tools=[],
        embedder=lambda text: embedder.embed(text),
        judge=lambda q, a: False,
        sim_threshold=_cache_sim_threshold(),
        max_iters=15,
    )
    p1_trajs = sorted(p1_traj_dir.glob("trajectory_q*.json")) if p1_traj_dir.exists() else []
    corrupted_indices = corrupt_answer_cache_typo(
        seed_agent, p1_trajs, lambda text: embedder.embed(text), seed=seed, fraction=fraction
    )
    logger.info("T4-AC: contaminated %d cache entries (typo-mutation)", len(corrupted_indices))
    seeded_cache = list(seed_agent._cache)

    def factory(qr: dict, tools: list):
        judge_fn = make_containment_judge(qr.get("answer", ""))
        agent = AnswerCacheAgent(
            tools=tools,
            embedder=lambda text: embedder.embed(text),
            judge=judge_fn,
            sim_threshold=_cache_sim_threshold(),
            max_iters=15,
        )
        agent._cache = list(seeded_cache)
        return agent

    return factory


def build_t4_hybrid_factory(
    embedder: Any,
    p1_db: Path,
    p1_traj_dir: Path,
    extractor: Any,
    t4_db: Path,
    scope_of=None,
    seed: int = 42,
    fraction: float = 0.20,
) -> Callable:
    """T4-Hybrid: typo-mutation on BOTH AnswerCache AND HintStore."""
    from hgc.baselines import AnswerCacheAgent
    from hgc.contaminators.typo_mutation import corrupt_answer_cache_typo, corrupt_hint_store_typo
    from hgc.gate import CompositeGate, SupportVerifier
    from hgc.hgc_agent import HGCAgent
    from hgc.hgc_core import HGCCoreAgent
    from hgc.memory import HintStore

    # Step 1: Seed + contaminate AC (typo-mutation)
    seed_agent = AnswerCacheAgent(
        tools=[],
        embedder=lambda text: embedder.embed(text),
        judge=lambda q, a: False,
        sim_threshold=_cache_sim_threshold(),
        max_iters=15,
    )
    p1_trajs = sorted(p1_traj_dir.glob("trajectory_q*.json")) if p1_traj_dir.exists() else []
    corrupted_indices = corrupt_answer_cache_typo(
        seed_agent, p1_trajs, lambda text: embedder.embed(text), seed=seed, fraction=fraction
    )
    logger.info(
        "T4-Hybrid AC: contaminated %d cache entries (typo-mutation)", len(corrupted_indices)
    )
    seeded_cache = list(seed_agent._cache)

    # Step 2: Copy p1_db to t4_db and contaminate HintStore (typo-mutation)
    t4_db.parent.mkdir(parents=True, exist_ok=True)
    if p1_db.exists() and not t4_db.exists():
        shutil.copy2(str(p1_db), str(t4_db))
    elif not p1_db.exists():
        logger.warning("T4-Hybrid: P1 store missing at %s; using empty store", p1_db)

    contaminated_store = HintStore(db_path=str(t4_db))
    corrupted_ids = corrupt_hint_store_typo(contaminated_store, seed=seed, fraction=fraction)
    logger.info("T4-Hybrid HintStore: contaminated %d hints (typo-mutation)", len(corrupted_ids))

    verifier = SupportVerifier(
        lm=_gate_lm(),
        max_doc_chars=_verifier_max_doc_chars(),
        containment_min_chars=_verifier_containment_min_chars(),
        predicate=_verifier_predicate(),
    )

    def factory(qr: dict, tools: list):
        judge_fn = make_containment_judge(qr.get("answer", ""))
        scope_id = scope_of(qr) if scope_of else f"bcp_query_{qr['query_id']}"

        ac_agent = AnswerCacheAgent(
            tools=[],
            embedder=lambda text: embedder.embed(text),
            judge=make_containment_judge(qr.get("answer", "")),
            sim_threshold=_cache_sim_threshold(),
            max_iters=15,
        )
        ac_agent._cache = list(seeded_cache)

        hgc_agent = HGCCoreAgent(
            tools=tools,
            store=contaminated_store,
            embedder=embedder,
            extractor=extractor,
            judge=judge_fn,
            max_iters=15,
            scope_id=scope_id,
        )

        return HGCAgent(
            ac_agent=ac_agent,
            core_agent=hgc_agent,
            store=contaminated_store,
            embedder=lambda text: embedder.embed(text),
            gate=CompositeGate(support_verifier=verifier, enabled=_enabled_gate_set()),
            top_k_hints=_top_k_hints(),
            strict_no_hint=_strict_no_hint(),
        )

    return factory


def build_p4_hybrid_factory(
    embedder: Any,
    p1_db: Path,
    p1_traj_dir: Path,
    extractor: Any,
    p4_db: Path,
    scope_of=None,
    seed: int = 42,
    fraction: float = 0.20,
) -> Callable:
    """Contaminated Hybrid (P4-Hybrid).

    BOTH the AnswerCache AND the HintStore are contaminated at the same seed/fraction
    used in P4-AC and P4-Hybrid. This is a fair apples-to-apples contamination stress
    test: under 20% cross-swap corruption of both backing stores, does the hybrid
    layer still recover answers?

    Flow:
      1. Seed AC cache via contaminate_answer_cache_from_p1(seed_ac, trajs, embedder,
         seed=42, fraction=0.20) — mirrors build_p4_ac_factory.
      2. Copy p1_db to p4_db (shutil.copy2), open as HintStore, run HintContaminator
         with seed=42 fraction=0.20 — mirrors build_p4_factory.
      3. Return factory(qr, tools) that builds a HGCAgent wrapping (fresh AC agent
         with contaminated seed_cache, HGCCoreAgent bound to contaminated HintStore).
    """
    from hgc.baselines import AnswerCacheAgent
    from hgc.contaminators.cross_swap import HintContaminator, contaminate_answer_cache_from_p1
    from hgc.gate import CompositeGate, SupportVerifier
    from hgc.hgc_agent import HGCAgent
    from hgc.hgc_core import HGCCoreAgent
    from hgc.memory import HintStore

    # Step 1: Seed + contaminate AC cache (mirrors build_p4_ac_factory)
    seed_agent = AnswerCacheAgent(
        tools=[],
        embedder=lambda text: embedder.embed(text),
        judge=lambda q, a: False,
        sim_threshold=_cache_sim_threshold(),
        max_iters=15,
    )
    p1_trajs = sorted(p1_traj_dir.glob("trajectory_q*.json")) if p1_traj_dir.exists() else []
    corrupted_indices = contaminate_answer_cache_from_p1(
        seed_agent, p1_trajs, lambda text: embedder.embed(text), seed=seed, fraction=fraction
    )
    logger.info("P4-Hybrid AC: contaminated %d cache entries", len(corrupted_indices))
    seeded_cache = list(seed_agent._cache)

    # Step 2: Copy p1_db to p4_db and contaminate HintStore (mirrors build_p4_factory)
    p4_db.parent.mkdir(parents=True, exist_ok=True)
    if p1_db.exists() and not p4_db.exists():
        shutil.copy2(str(p1_db), str(p4_db))
    elif not p1_db.exists():
        logger.warning("P4-Hybrid: P1 store missing at %s; using empty store", p1_db)

    contaminated_store = HintStore(db_path=str(p4_db))
    contaminator = HintContaminator(contaminated_store, seed=seed, fraction=fraction)
    corrupted_ids = contaminator.corrupt()
    logger.info("P4-Hybrid HintStore: contaminated %d hints in copied store", len(corrupted_ids))

    verifier = SupportVerifier(
        lm=_gate_lm(),
        max_doc_chars=_verifier_max_doc_chars(),
        containment_min_chars=_verifier_containment_min_chars(),
        predicate=_verifier_predicate(),
    )

    def factory(qr: dict, tools: list):
        judge_fn = make_containment_judge(qr.get("answer", ""))
        scope_id = scope_of(qr) if scope_of else f"bcp_query_{qr['query_id']}"

        ac_agent = AnswerCacheAgent(
            tools=[],
            embedder=lambda text: embedder.embed(text),
            judge=make_containment_judge(qr.get("answer", "")),
            sim_threshold=_cache_sim_threshold(),
            max_iters=15,
        )
        ac_agent._cache = list(seeded_cache)

        hgc_agent = HGCCoreAgent(
            tools=tools,
            store=contaminated_store,
            embedder=embedder,
            extractor=extractor,
            judge=judge_fn,
            max_iters=15,
            scope_id=scope_id,
        )

        return HGCAgent(
            ac_agent=ac_agent,
            core_agent=hgc_agent,
            store=contaminated_store,
            embedder=lambda text: embedder.embed(text),
            gate=CompositeGate(support_verifier=verifier, enabled=_enabled_gate_set()),
            top_k_hints=_top_k_hints(),
            strict_no_hint=_strict_no_hint(),
        )

    return factory


def build_p4_factory(
    embedder: Any,
    p1_db: Path,
    extractor: Any,
    p4_db: Path,
    scope_of=None,
):
    """P4: HGCCoreAgent with contaminated copy of P1's store.

    Parameters
    ----------
    scope_of:
        Optional callable ``(qr) -> str`` returning the scope_id.
        Defaults to ``f"bcp_query_{qr['query_id']}"``.
    """
    from hgc.contaminators.cross_swap import HintContaminator
    from hgc.hgc_core import HGCCoreAgent
    from hgc.memory import HintStore

    p4_db.parent.mkdir(parents=True, exist_ok=True)
    if p1_db.exists() and not p4_db.exists():
        shutil.copy2(str(p1_db), str(p4_db))
    elif not p1_db.exists():
        logger.warning("P4: P1 store missing at %s; using empty store", p1_db)

    contaminated_store = HintStore(db_path=str(p4_db))
    contaminator = HintContaminator(contaminated_store, seed=42, fraction=0.20)
    corrupted_ids = contaminator.corrupt()
    logger.info("P4: contaminated %d hints in copied store", len(corrupted_ids))

    def factory(qr: dict, tools: list):
        judge_fn = make_containment_judge(qr.get("answer", ""))
        scope_id = scope_of(qr) if scope_of else f"bcp_query_{qr['query_id']}"
        return HGCCoreAgent(
            tools=tools,
            store=contaminated_store,
            embedder=embedder,
            extractor=extractor,
            judge=judge_fn,
            max_iters=15,
            scope_id=scope_id,
        )

    return factory


# ---------------------------------------------------------------------------
# Phase factory builders — HotpotQA
# ---------------------------------------------------------------------------


def build_hgc_core_factory_hotpot(
    embedder: Any,
    store: Any,
    extractor: Any,
    scope_of=None,
):
    """Return a HGCCoreAgent factory for HotpotQA queries.

    Parameters
    ----------
    scope_of:
        Optional callable ``(qr) -> str`` for the scope_id.
        Defaults to ``f"hotpot_cluster_{qr['cluster_idx']}"``.
    """
    from hgc.hgc_core import HGCCoreAgent

    def factory(qr: dict, tools: list):
        judge_fn = make_containment_judge(qr.get("answer", ""))
        scope_id = scope_of(qr) if scope_of else f"hotpot_cluster_{qr.get('cluster_idx', 0)}"
        return HGCCoreAgent(
            tools=tools,
            store=store,
            embedder=embedder,
            extractor=extractor,
            judge=judge_fn,
            max_iters=15,
            scope_id=scope_id,
        )

    return factory


def build_rag_factory_hotpot(embedder: Any):
    from hgc.baselines import NaiveRAGAgent

    def factory(qr: dict, tools: list):
        judge_fn = make_containment_judge(qr.get("answer", ""))
        return NaiveRAGAgent(embedder=embedder, judge=judge_fn)

    return factory


# ---------------------------------------------------------------------------
# Tool factory — QASPER
# ---------------------------------------------------------------------------


def make_tools_qasper(qr: dict) -> list:
    """Build the 4 standard document tools for a QASPER query record.

    QASPER records carry docs directly under "docs" (same as HotpotQA).
    """
    docs = qr.get("docs", [])
    doc_by_id = {d["docid"]: d for d in docs}

    def list_document_ids() -> list:
        return [d["docid"] for d in docs]

    def get_document_title(docid: str) -> str:
        d = doc_by_id.get(docid)
        if not d:
            return f"ERROR: docid {docid!r} not found"
        return d.get("text", "").split("\n", 1)[0][:300]

    def get_document_snippet(docid: str, char_start: int = 0, char_len: int = 2000) -> str:
        d = doc_by_id.get(docid)
        if not d:
            return f"ERROR: docid {docid!r} not found"
        return d.get("text", "")[char_start : char_start + char_len]

    def search_documents(keyword: str, max_hits: int = 10) -> list:
        """Substring search; returns list of {docid, snippet}."""
        hits = []
        kl = keyword.lower()
        for d in docs:
            idx = d.get("text", "").lower().find(kl)
            if idx != -1:
                t = d["text"]
                hits.append(
                    {
                        "docid": d["docid"],
                        "snippet": t[max(0, idx - 80) : idx + 120],
                    }
                )
                if len(hits) >= max_hits:
                    break
        return hits

    return [list_document_ids, get_document_title, get_document_snippet, search_documents]


# ---------------------------------------------------------------------------
# Phase factory builders — QASPER
# ---------------------------------------------------------------------------


def build_hgc_core_factory_qasper(
    embedder: Any,
    store: Any,
    extractor: Any,
    scope_of=None,
):
    """Return a HGCCoreAgent factory for QASPER queries.

    Parameters
    ----------
    scope_of:
        Optional callable ``(qr) -> str`` for the scope_id.
        Defaults to ``f"qasper_paper_{qr['paper_id']}"``.
        Paper-level scope is used because multiple questions share a paper.
    """
    from hgc.hgc_core import HGCCoreAgent

    def factory(qr: dict, tools: list):
        judge_fn = make_containment_judge(qr.get("answer", ""))
        scope_id = (
            scope_of(qr) if scope_of else f"qasper_paper_{qr.get('paper_id', qr['query_id'])}"
        )
        return HGCCoreAgent(
            tools=tools,
            store=store,
            embedder=embedder,
            extractor=extractor,
            judge=judge_fn,
            max_iters=15,
            scope_id=scope_id,
        )

    return factory


def build_ac_factory_qasper(embedder: Any, p1_traj_dir: Path):
    """AnswerCacheAgent factory seeded from QASPER P1 trajectories."""
    from hgc.baselines import AnswerCacheAgent
    from hgc.runner import seed_answer_cache_from_p1

    seed_agent = AnswerCacheAgent(
        tools=[],
        embedder=lambda text: embedder.embed(text),
        judge=lambda q, a: False,
        sim_threshold=_cache_sim_threshold(),
        max_iters=15,
    )
    p1_trajs = sorted(p1_traj_dir.glob("trajectory_q*.json")) if p1_traj_dir.exists() else []
    seed_answer_cache_from_p1(seed_agent, p1_trajs, lambda text: embedder.embed(text))
    seeded_cache = list(seed_agent._cache)

    def factory(qr: dict, tools: list):
        judge_fn = make_containment_judge(qr.get("answer", ""))
        agent = AnswerCacheAgent(
            tools=tools,
            embedder=lambda text: embedder.embed(text),
            judge=judge_fn,
            sim_threshold=_cache_sim_threshold(),
            max_iters=15,
        )
        agent._cache = list(seeded_cache)
        return agent

    return factory


def build_m0_factory_qasper(
    p1_traj_dir: Path, mem0_config: dict | None = None, user_id: str = "p3_m0_qasper"
):
    """Mem0ReActAgent factory seeded from QASPER P1."""
    from hgc.baselines import Mem0ReActAgent
    from hgc.runner import seed_mem0_from_p1

    if mem0_config is not None:
        _config = mem0_config
        if isinstance(_config, dict):
            from mem0.configs.base import MemoryConfig as _MemoryConfig

            _config = _MemoryConfig(**_config)
        from mem0 import Memory as _Memory

        shared_memory = _Memory(config=_config)
    else:
        from mem0 import Memory as _Memory

        shared_memory = _Memory()

    p1_trajs = sorted(p1_traj_dir.glob("trajectory_q*.json")) if p1_traj_dir.exists() else []
    seed_mem0_from_p1(shared_memory, p1_trajs, user_id=user_id)

    def factory(qr: dict, tools: list):
        judge_fn = make_containment_judge(qr.get("answer", ""))
        return Mem0ReActAgent(
            tools=tools,
            judge=judge_fn,
            user_id=user_id,
            mem0_memory=shared_memory,
            max_iters=15,
        )

    return factory


def build_p4_factory_qasper(
    embedder: Any,
    p1_db: Path,
    extractor: Any,
    p4_db: Path,
    scope_of=None,
):
    """P4: HGCCoreAgent with contaminated copy of QASPER P1's store."""
    from hgc.contaminators.cross_swap import HintContaminator
    from hgc.hgc_core import HGCCoreAgent
    from hgc.memory import HintStore

    p4_db.parent.mkdir(parents=True, exist_ok=True)
    if p1_db.exists() and not p4_db.exists():
        shutil.copy2(str(p1_db), str(p4_db))
    elif not p1_db.exists():
        logger.warning("P4-QASPER: P1 store missing at %s; using empty store", p1_db)

    contaminated_store = HintStore(db_path=str(p4_db))
    contaminator = HintContaminator(contaminated_store, seed=42, fraction=0.20)
    corrupted_ids = contaminator.corrupt()
    logger.info("P4-QASPER: contaminated %d hints in copied store", len(corrupted_ids))

    def factory(qr: dict, tools: list):
        judge_fn = make_containment_judge(qr.get("answer", ""))
        scope_id = (
            scope_of(qr) if scope_of else f"qasper_paper_{qr.get('paper_id', qr['query_id'])}"
        )
        return HGCCoreAgent(
            tools=tools,
            store=contaminated_store,
            embedder=embedder,
            extractor=extractor,
            judge=judge_fn,
            max_iters=15,
            scope_id=scope_id,
        )

    return factory


def build_p4_ac_factory_qasper(
    embedder: Any, p1_traj_dir: Path, seed: int = 42, fraction: float = 0.20
):
    """P4-AC: AnswerCacheAgent seeded from QASPER P1 then contaminated via 20% cross-swap."""
    from hgc.baselines import AnswerCacheAgent
    from hgc.contaminators.cross_swap import contaminate_answer_cache_from_p1

    seed_agent = AnswerCacheAgent(
        tools=[],
        embedder=lambda text: embedder.embed(text),
        judge=lambda q, a: False,
        sim_threshold=_cache_sim_threshold(),
        max_iters=15,
    )
    p1_trajs = sorted(p1_traj_dir.glob("trajectory_q*.json")) if p1_traj_dir.exists() else []
    corrupted_indices = contaminate_answer_cache_from_p1(
        seed_agent, p1_trajs, lambda text: embedder.embed(text), seed=seed, fraction=fraction
    )
    logger.info("P4-AC-QASPER: contaminated %d cache entries", len(corrupted_indices))
    seeded_cache = list(seed_agent._cache)

    def factory(qr: dict, tools: list):
        judge_fn = make_containment_judge(qr.get("answer", ""))
        agent = AnswerCacheAgent(
            tools=tools,
            embedder=lambda text: embedder.embed(text),
            judge=judge_fn,
            sim_threshold=_cache_sim_threshold(),
            max_iters=15,
        )
        agent._cache = list(seeded_cache)
        return agent

    return factory


def build_p4_m0_factory_qasper(
    p1_traj_dir: Path,
    mem0_config: dict | None = None,
    user_id: str = "p4_m0_qasper",
    seed: int = 42,
    fraction: float = 0.20,
):
    """P4-M0: Mem0ReActAgent seeded from QASPER P1 then contaminated via 20% cross-swap."""
    from hgc.baselines import Mem0ReActAgent
    from hgc.contaminators.cross_swap import contaminate_mem0_from_p1

    if mem0_config is not None:
        _config = mem0_config
        if isinstance(_config, dict):
            from mem0.configs.base import MemoryConfig as _MemoryConfig

            _config = _MemoryConfig(**_config)
        from mem0 import Memory as _Memory

        shared_memory = _Memory(config=_config)
    else:
        from mem0 import Memory as _Memory

        shared_memory = _Memory()

    p1_trajs = sorted(p1_traj_dir.glob("trajectory_q*.json")) if p1_traj_dir.exists() else []
    corrupted_ids = contaminate_mem0_from_p1(
        shared_memory, p1_trajs, seed=seed, fraction=fraction, user_id=user_id
    )
    logger.info("P4-M0-QASPER: contaminated %d mem0 entries", len(corrupted_ids))

    def factory(qr: dict, tools: list):
        judge_fn = make_containment_judge(qr.get("answer", ""))
        return Mem0ReActAgent(
            tools=tools,
            judge=judge_fn,
            user_id=user_id,
            mem0_memory=shared_memory,
            max_iters=15,
        )

    return factory


def build_hybrid_factory_qasper(
    embedder: Any,
    p1_db: Path,
    p1_traj_dir: Path,
    extractor: Any,
    scope_of=None,
):
    """Clean Hybrid (P3-Hybrid) for QASPER: AC fast path + hint re-verification."""
    from hgc.baselines import AnswerCacheAgent
    from hgc.hgc_agent import HGCAgent
    from hgc.hgc_core import HGCCoreAgent
    from hgc.memory import HintStore
    from hgc.runner import seed_answer_cache_from_p1

    seed_agent = AnswerCacheAgent(
        tools=[],
        embedder=lambda text: embedder.embed(text),
        judge=lambda q, a: False,
        sim_threshold=_cache_sim_threshold(),
        max_iters=15,
    )
    p1_trajs = sorted(p1_traj_dir.glob("trajectory_q*.json")) if p1_traj_dir.exists() else []
    seed_answer_cache_from_p1(seed_agent, p1_trajs, lambda text: embedder.embed(text))
    seeded_cache = list(seed_agent._cache)
    logger.info("P3-Hybrid-QASPER: seeded AC with %d cache entries from P1", len(seeded_cache))

    store = HintStore(db_path=str(p1_db))

    def factory(qr: dict, tools: list):
        judge_fn = make_containment_judge(qr.get("answer", ""))
        scope_id = (
            scope_of(qr) if scope_of else f"qasper_paper_{qr.get('paper_id', qr['query_id'])}"
        )

        ac_agent = AnswerCacheAgent(
            tools=[],
            embedder=lambda text: embedder.embed(text),
            judge=make_containment_judge(qr.get("answer", "")),
            sim_threshold=_cache_sim_threshold(),
            max_iters=15,
        )
        ac_agent._cache = list(seeded_cache)

        hgc_agent = HGCCoreAgent(
            tools=tools,
            store=store,
            embedder=embedder,
            extractor=extractor,
            judge=judge_fn,
            max_iters=15,
            scope_id=scope_id,
        )

        return HGCAgent(
            ac_agent=ac_agent,
            core_agent=hgc_agent,
            store=store,
            embedder=lambda text: embedder.embed(text),
            top_k_hints=_top_k_hints(),
        )

    return factory


def build_p4_hybrid_factory_qasper(
    embedder: Any,
    p1_db: Path,
    p1_traj_dir: Path,
    extractor: Any,
    p4_db: Path,
    scope_of=None,
    seed: int = 42,
    fraction: float = 0.20,
):
    """Contaminated Hybrid (P4-Hybrid) for QASPER."""
    from hgc.baselines import AnswerCacheAgent
    from hgc.contaminators.cross_swap import HintContaminator, contaminate_answer_cache_from_p1
    from hgc.hgc_agent import HGCAgent
    from hgc.hgc_core import HGCCoreAgent
    from hgc.memory import HintStore

    seed_agent = AnswerCacheAgent(
        tools=[],
        embedder=lambda text: embedder.embed(text),
        judge=lambda q, a: False,
        sim_threshold=_cache_sim_threshold(),
        max_iters=15,
    )
    p1_trajs = sorted(p1_traj_dir.glob("trajectory_q*.json")) if p1_traj_dir.exists() else []
    corrupted_indices = contaminate_answer_cache_from_p1(
        seed_agent, p1_trajs, lambda text: embedder.embed(text), seed=seed, fraction=fraction
    )
    logger.info("P4-Hybrid-QASPER AC: contaminated %d cache entries", len(corrupted_indices))
    seeded_cache = list(seed_agent._cache)

    p4_db.parent.mkdir(parents=True, exist_ok=True)
    if p1_db.exists() and not p4_db.exists():
        shutil.copy2(str(p1_db), str(p4_db))
    elif not p1_db.exists():
        logger.warning("P4-Hybrid-QASPER: P1 store missing at %s; using empty store", p1_db)

    contaminated_store = HintStore(db_path=str(p4_db))
    contaminator = HintContaminator(contaminated_store, seed=seed, fraction=fraction)
    corrupted_ids = contaminator.corrupt()
    logger.info(
        "P4-Hybrid-QASPER HintStore: contaminated %d hints in copied store", len(corrupted_ids)
    )

    def factory(qr: dict, tools: list):
        judge_fn = make_containment_judge(qr.get("answer", ""))
        scope_id = (
            scope_of(qr) if scope_of else f"qasper_paper_{qr.get('paper_id', qr['query_id'])}"
        )

        ac_agent = AnswerCacheAgent(
            tools=[],
            embedder=lambda text: embedder.embed(text),
            judge=make_containment_judge(qr.get("answer", "")),
            sim_threshold=_cache_sim_threshold(),
            max_iters=15,
        )
        ac_agent._cache = list(seeded_cache)

        hgc_agent = HGCCoreAgent(
            tools=tools,
            store=contaminated_store,
            embedder=embedder,
            extractor=extractor,
            judge=judge_fn,
            max_iters=15,
            scope_id=scope_id,
        )

        return HGCAgent(
            ac_agent=ac_agent,
            core_agent=hgc_agent,
            store=contaminated_store,
            embedder=lambda text: embedder.embed(text),
            top_k_hints=_top_k_hints(),
        )

    return factory


# ---------------------------------------------------------------------------
# Phase factory builders — QASPER RAG
# ---------------------------------------------------------------------------


def build_rag_naive_factory_qasper(embedder: Any, qasper_dataset: Any = None):
    """V-RAG: Vanilla RAG baseline for QASPER using RAGBackbone.

    Parameters
    ----------
    embedder:
        Embedder instance (hgc.embeddings.Embedder).
    qasper_dataset:
        Unused — kept for API symmetry. Tools carry per-query docs.
    """
    from langchain_openai import AzureChatOpenAI

    from hgc.rag_backbone import RAGBackbone

    deployment = os.environ.get("AZURE_OPENAI_DEPLOYMENT", "gpt-4.1")
    llm = AzureChatOpenAI(
        azure_deployment=deployment,
        azure_endpoint=os.environ.get("AZURE_OPENAI_ENDPOINT", ""),
        api_key=os.environ.get("AZURE_OPENAI_API_KEY", ""),
        api_version=os.environ.get("AZURE_OPENAI_API_VERSION", ""),
        temperature=0,
        max_tokens=1024,
    )

    backbone = RAGBackbone(llm=llm, embedder=_make_langchain_embedder(embedder))

    def factory(qr: dict, tools: list):  # noqa: ARG001
        _ = tools
        docs = qr.get("docs", [])
        judge_fn = make_containment_judge(qr.get("answer", ""))

        class _RAGAgent:
            def run(self, question: str) -> dict:
                result = backbone.run(question, docs=docs)
                result["judgment_correct"] = judge_fn(question, result.get("answer", ""))
                return result

        return _RAGAgent()

    return factory


def build_p1_rag_factory_qasper(
    embedder: Any,
    store: Any,
    scope_of=None,
    llm_judge: Any = None,
):
    """P1-RAG: Seed phase — runs vanilla RAG and writes location hints to HintStore.

    Mirrors the role of ReAct's P1 in the BCP suite: a first-encounter pass
    whose trajectories + populated HintStore serve as the seed source for
    AC-RAG / HGC-RAG / C-AC-RAG / C-HGC-RAG.

    Location hints are synthesised directly from the RAG retriever's top-k
    docids. Polarity is decided by an ``LLMJudge`` (semantic equivalence) so
    that QASPER answers like ``"I don't know."`` against gold ``"UNANSWERABLE"``
    register as positive. A containment check would miss these and mis-label
    every hint as negative, defeating the HGC gate.
    """
    import time
    import uuid

    import numpy as np
    from langchain_openai import AzureChatOpenAI

    from hgc.judge import LLMJudge
    from hgc.memory import HintRecord
    from hgc.rag_backbone import RAGBackbone

    deployment = os.environ.get("AZURE_OPENAI_DEPLOYMENT", "gpt-4.1")
    llm = AzureChatOpenAI(
        azure_deployment=deployment,
        azure_endpoint=os.environ.get("AZURE_OPENAI_ENDPOINT", ""),
        api_key=os.environ.get("AZURE_OPENAI_API_KEY", ""),
        api_version=os.environ.get("AZURE_OPENAI_API_VERSION", ""),
        temperature=0,
        max_tokens=1024,
    )
    backbone = RAGBackbone(llm=llm, embedder=_make_langchain_embedder(embedder))
    judge = llm_judge if llm_judge is not None else LLMJudge()

    def factory(qr: dict, tools: list):  # noqa: ARG001
        _ = tools
        docs = qr.get("docs", [])
        gold = qr.get("answer", "")
        scope_id = (
            scope_of(qr) if scope_of else f"qasper_paper_{qr.get('paper_id', qr['query_id'])}"
        )

        class _P1RAGAgent:
            def run(self, question: str) -> dict:
                result = backbone.run(question, docs=docs)
                answer = result.get("answer", "")

                if answer.strip():
                    verdict = judge.judge(question, gold, answer)
                    judgment = bool(verdict.correct)
                else:
                    judgment = False

                q_emb = embedder.embed(question)
                q_emb_bytes: bytes = q_emb.astype(np.float32).tobytes()
                polarity = "positive" if judgment else "negative"
                now = time.time()

                retrieved_docids = result.get("trajectory", {}).get("retrieved_docids", [])
                added_ids: list[str] = []
                for docid in retrieved_docids:
                    record = HintRecord(
                        hint_id=str(uuid.uuid4()),
                        hint_type="location",
                        polarity=polarity,
                        content=str(docid),
                        content_meta={},
                        query_ctx=question,
                        query_ctx_embedding=q_emb_bytes,
                        trajectory_step=0,
                        created_at=now,
                        last_validated_at=now,
                        success_count=0,
                        failure_count=0,
                        retrieval_count=0,
                        scope_id=scope_id,
                    )
                    added_ids.append(store.add(record))

                result["judgment_correct"] = judgment
                result["added_hints"] = added_ids
                result["n_added_hints"] = len(added_ids)
                return result

        return _P1RAGAgent()

    return factory


def build_ac_rag_factory_qasper(embedder: Any, p1_traj_dir: Path):
    """AC-RAG: ACRAGAgent (cache + single-shot RAG fallback) for QASPER.

    Uses RAGBackbone as the cache-miss fallback to mirror HGC-RAG's
    architecture, so the AC-vs-HGC token comparison reflects the gate's
    contribution rather than the difference between a ReAct loop and a
    single-shot RAG pipeline.
    """
    from langchain_openai import AzureChatOpenAI

    from hgc.baselines import AnswerCacheAgent
    from hgc.hgc_rag import ACRAGAgent
    from hgc.rag_backbone import RAGBackbone
    from hgc.runner import seed_answer_cache_from_p1

    deployment = os.environ.get("AZURE_OPENAI_DEPLOYMENT", "gpt-4.1")
    llm = AzureChatOpenAI(
        azure_deployment=deployment,
        azure_endpoint=os.environ.get("AZURE_OPENAI_ENDPOINT", ""),
        api_key=os.environ.get("AZURE_OPENAI_API_KEY", ""),
        api_version=os.environ.get("AZURE_OPENAI_API_VERSION", ""),
        temperature=0,
        max_tokens=1024,
    )
    backbone = RAGBackbone(llm=llm, embedder=_make_langchain_embedder(embedder))

    seed_agent = AnswerCacheAgent(
        tools=[],
        embedder=lambda text: embedder.embed(text),
        judge=lambda q, a: False,
        sim_threshold=_cache_sim_threshold(),
        max_iters=15,
    )
    p1_trajs = sorted(p1_traj_dir.glob("trajectory_q*.json")) if p1_traj_dir.exists() else []
    seed_answer_cache_from_p1(seed_agent, p1_trajs, lambda text: embedder.embed(text))
    seeded_cache = list(seed_agent._cache)
    logger.info("AC-RAG-QASPER: seeded AC with %d cache entries from P1", len(seeded_cache))

    def factory(qr: dict, tools: list):  # noqa: ARG001
        _ = tools
        judge_fn = make_containment_judge(qr.get("answer", ""))
        qr_docs = qr.get("docs", [])

        ac_agent = AnswerCacheAgent(
            tools=[],
            embedder=lambda text: embedder.embed(text),
            judge=judge_fn,
            sim_threshold=_cache_sim_threshold(),
            max_iters=15,
        )
        ac_agent._cache = list(seeded_cache)

        class _BoundRAGBackbone:
            def run(self, question: str, docs: list[dict] | None = None) -> dict:
                return backbone.run(question, docs=docs if docs is not None else qr_docs)

        return ACRAGAgent(ac_agent=ac_agent, rag_backbone=_BoundRAGBackbone())

    return factory


def build_hgc_rag_factory_qasper(
    embedder: Any,
    p1_db: Path,
    p1_traj_dir: Path,
    extractor: Any,
    scope_of=None,
):
    """HGC-RAG: HGCRAGAgent (AC + RAG fallback + gate) for QASPER."""
    from langchain_openai import AzureChatOpenAI

    from hgc.baselines import AnswerCacheAgent
    from hgc.gate import CompositeGate, SupportVerifier
    from hgc.hgc_rag import HGCRAGAgent
    from hgc.memory import HintStore
    from hgc.rag_backbone import RAGBackbone
    from hgc.runner import seed_answer_cache_from_p1

    deployment = os.environ.get("AZURE_OPENAI_DEPLOYMENT", "gpt-4.1")
    llm = AzureChatOpenAI(
        azure_deployment=deployment,
        azure_endpoint=os.environ.get("AZURE_OPENAI_ENDPOINT", ""),
        api_key=os.environ.get("AZURE_OPENAI_API_KEY", ""),
        api_version=os.environ.get("AZURE_OPENAI_API_VERSION", ""),
        temperature=0,
        max_tokens=1024,
    )
    backbone = RAGBackbone(llm=llm, embedder=_make_langchain_embedder(embedder))

    seed_agent = AnswerCacheAgent(
        tools=[],
        embedder=lambda text: embedder.embed(text),
        judge=lambda q, a: False,
        sim_threshold=_cache_sim_threshold(),
        max_iters=15,
    )
    p1_trajs = sorted(p1_traj_dir.glob("trajectory_q*.json")) if p1_traj_dir.exists() else []
    seed_answer_cache_from_p1(seed_agent, p1_trajs, lambda text: embedder.embed(text))
    seeded_cache = list(seed_agent._cache)
    logger.info("HGC-RAG-QASPER: seeded AC with %d cache entries from P1", len(seeded_cache))

    store = HintStore(db_path=str(p1_db))
    verifier = SupportVerifier(
        lm=_gate_lm(),
        max_doc_chars=_verifier_max_doc_chars(),
        containment_min_chars=_verifier_containment_min_chars(),
        predicate=_verifier_predicate(),
    )

    def factory(qr: dict, tools: list):  # noqa: ARG001
        _ = tools
        judge_fn = make_containment_judge(qr.get("answer", ""))
        scope_id = (
            scope_of(qr) if scope_of else f"qasper_paper_{qr.get('paper_id', qr['query_id'])}"
        )
        qr_docs = qr.get("docs", [])

        ac_agent = AnswerCacheAgent(
            tools=[],
            embedder=lambda text: embedder.embed(text),
            judge=judge_fn,
            sim_threshold=_cache_sim_threshold(),
            max_iters=15,
        )
        ac_agent._cache = list(seeded_cache)

        class _BoundRAGBackbone:
            def run(self, question: str, docs: list[dict] | None = None) -> dict:
                return backbone.run(question, docs=docs if docs is not None else qr_docs)

        return HGCRAGAgent(
            ac_agent=ac_agent,
            rag_backbone=_BoundRAGBackbone(),
            store=store,
            embedder=lambda text: embedder.embed(text),
            hint_retrieval_params={"scope_id": scope_id},
            gate=CompositeGate(support_verifier=verifier, enabled=_enabled_gate_set()),
            top_k_hints=_top_k_hints(),
            strict_no_hint=_strict_no_hint(),
        )

    return factory


def build_p4_ac_rag_factory_qasper(
    embedder: Any, p1_traj_dir: Path, seed: int = 42, fraction: float = 0.20
):
    """C-AC-RAG: ACRAGAgent (single-shot RAG fallback) on a contaminated cache.

    Mirrors :func:`build_ac_rag_factory_qasper` but contaminates the cache
    via cross-swap before serve. Uses RAGBackbone fallback for apples-to-apples
    comparison against C-HGC-RAG.
    """
    from langchain_openai import AzureChatOpenAI

    from hgc.baselines import AnswerCacheAgent
    from hgc.contaminators.cross_swap import contaminate_answer_cache_from_p1
    from hgc.hgc_rag import ACRAGAgent
    from hgc.rag_backbone import RAGBackbone

    deployment = os.environ.get("AZURE_OPENAI_DEPLOYMENT", "gpt-4.1")
    llm = AzureChatOpenAI(
        azure_deployment=deployment,
        azure_endpoint=os.environ.get("AZURE_OPENAI_ENDPOINT", ""),
        api_key=os.environ.get("AZURE_OPENAI_API_KEY", ""),
        api_version=os.environ.get("AZURE_OPENAI_API_VERSION", ""),
        temperature=0,
        max_tokens=1024,
    )
    backbone = RAGBackbone(llm=llm, embedder=_make_langchain_embedder(embedder))

    seed_agent = AnswerCacheAgent(
        tools=[],
        embedder=lambda text: embedder.embed(text),
        judge=lambda q, a: False,
        sim_threshold=_cache_sim_threshold(),
        max_iters=15,
    )
    p1_trajs = sorted(p1_traj_dir.glob("trajectory_q*.json")) if p1_traj_dir.exists() else []
    corrupted_indices = contaminate_answer_cache_from_p1(
        seed_agent, p1_trajs, lambda text: embedder.embed(text), seed=seed, fraction=fraction
    )
    logger.info("C-AC-RAG-QASPER: contaminated %d cache entries", len(corrupted_indices))
    seeded_cache = list(seed_agent._cache)

    def factory(qr: dict, tools: list):  # noqa: ARG001
        _ = tools
        judge_fn = make_containment_judge(qr.get("answer", ""))
        qr_docs = qr.get("docs", [])

        ac_agent = AnswerCacheAgent(
            tools=[],
            embedder=lambda text: embedder.embed(text),
            judge=judge_fn,
            sim_threshold=_cache_sim_threshold(),
            max_iters=15,
        )
        ac_agent._cache = list(seeded_cache)

        class _BoundRAGBackbone:
            def run(self, question: str, docs: list[dict] | None = None) -> dict:
                return backbone.run(question, docs=docs if docs is not None else qr_docs)

        return ACRAGAgent(ac_agent=ac_agent, rag_backbone=_BoundRAGBackbone())

    return factory


def build_p4_hgc_rag_factory_qasper(
    embedder: Any,
    p1_db: Path,
    p1_traj_dir: Path,
    extractor: Any,
    p4_db: Path,
    scope_of=None,
    seed: int = 42,
    fraction: float = 0.20,
):
    """C-HGC-RAG: Contaminated HGCRAGAgent for QASPER.

    AC and HintStore both contaminated via cross-swap (mirrors build_p4_hybrid_factory_qasper
    but uses RAGBackbone as the fallback instead of HGCCoreAgent).
    """
    from langchain_openai import AzureChatOpenAI

    from hgc.baselines import AnswerCacheAgent
    from hgc.contaminators.cross_swap import HintContaminator, contaminate_answer_cache_from_p1
    from hgc.gate import CompositeGate, SupportVerifier
    from hgc.hgc_rag import HGCRAGAgent
    from hgc.memory import HintStore
    from hgc.rag_backbone import RAGBackbone

    deployment = os.environ.get("AZURE_OPENAI_DEPLOYMENT", "gpt-4.1")
    llm = AzureChatOpenAI(
        azure_deployment=deployment,
        azure_endpoint=os.environ.get("AZURE_OPENAI_ENDPOINT", ""),
        api_key=os.environ.get("AZURE_OPENAI_API_KEY", ""),
        api_version=os.environ.get("AZURE_OPENAI_API_VERSION", ""),
        temperature=0,
        max_tokens=1024,
    )
    backbone = RAGBackbone(llm=llm, embedder=_make_langchain_embedder(embedder))

    # Contaminate AC
    seed_agent = AnswerCacheAgent(
        tools=[],
        embedder=lambda text: embedder.embed(text),
        judge=lambda q, a: False,
        sim_threshold=_cache_sim_threshold(),
        max_iters=15,
    )
    p1_trajs = sorted(p1_traj_dir.glob("trajectory_q*.json")) if p1_traj_dir.exists() else []
    corrupted_indices = contaminate_answer_cache_from_p1(
        seed_agent, p1_trajs, lambda text: embedder.embed(text), seed=seed, fraction=fraction
    )
    logger.info("C-HGC-RAG-QASPER AC: contaminated %d cache entries", len(corrupted_indices))
    seeded_cache = list(seed_agent._cache)

    # Contaminate HintStore
    p4_db.parent.mkdir(parents=True, exist_ok=True)
    if p1_db.exists() and not p4_db.exists():
        shutil.copy2(str(p1_db), str(p4_db))
    elif not p1_db.exists():
        logger.warning("C-HGC-RAG-QASPER: P1 store missing at %s; using empty store", p1_db)

    contaminated_store = HintStore(db_path=str(p4_db))
    contaminator = HintContaminator(contaminated_store, seed=seed, fraction=fraction)
    corrupted_ids = contaminator.corrupt()
    logger.info(
        "C-HGC-RAG-QASPER HintStore: contaminated %d hints in copied store", len(corrupted_ids)
    )

    verifier = SupportVerifier(
        lm=_gate_lm(),
        max_doc_chars=_verifier_max_doc_chars(),
        containment_min_chars=_verifier_containment_min_chars(),
        predicate=_verifier_predicate(),
    )

    def factory(qr: dict, tools: list):  # noqa: ARG001
        _ = tools
        judge_fn = make_containment_judge(qr.get("answer", ""))
        scope_id = (
            scope_of(qr) if scope_of else f"qasper_paper_{qr.get('paper_id', qr['query_id'])}"
        )

        ac_agent = AnswerCacheAgent(
            tools=[],
            embedder=lambda text: embedder.embed(text),
            judge=judge_fn,
            sim_threshold=_cache_sim_threshold(),
            max_iters=15,
        )
        ac_agent._cache = list(seeded_cache)

        class _BoundRAGBackbone:
            def run(self, question: str, docs: list[dict] | None = None) -> dict:
                return backbone.run(question, docs=docs)

        return HGCRAGAgent(
            ac_agent=ac_agent,
            rag_backbone=_BoundRAGBackbone(),
            store=contaminated_store,
            embedder=lambda text: embedder.embed(text),
            hint_retrieval_params={"scope_id": scope_id},
            gate=CompositeGate(support_verifier=verifier, enabled=_enabled_gate_set()),
            top_k_hints=_top_k_hints(),
            strict_no_hint=_strict_no_hint(),
        )

    return factory


# ---------------------------------------------------------------------------
# FinanceBench × Long-Context (Oracle) factories
# Same structure as QASPER-RAG factories, but swap RAGBackbone →
# LongCtxBackbone (no retrieval, inject all evidence docs directly).
# ---------------------------------------------------------------------------


def _make_longctx_llm():
    from langchain_openai import AzureChatOpenAI

    deployment = os.environ.get("AZURE_OPENAI_DEPLOYMENT", "gpt-4.1")
    return AzureChatOpenAI(
        azure_deployment=deployment,
        azure_endpoint=os.environ.get("AZURE_OPENAI_ENDPOINT", ""),
        api_key=os.environ.get("AZURE_OPENAI_API_KEY", ""),
        api_version=os.environ.get("AZURE_OPENAI_API_VERSION", ""),
        temperature=0,
        max_tokens=1024,
    )


def build_lc_naive_factory_financebench(embedder: Any):
    """V-LC: vanilla Long-Context baseline on FinanceBench (no cache, no gate)."""
    from hgc.longctx_backbone import LongCtxBackbone

    backbone = LongCtxBackbone(llm=_make_longctx_llm())

    def factory(qr: dict, tools: list):  # noqa: ARG001
        _ = tools
        docs = qr.get("docs", [])
        judge_fn = make_containment_judge(qr.get("answer", ""))

        class _LCAgent:
            def run(self, question: str) -> dict:
                result = backbone.run(question, docs=docs)
                result["judgment_correct"] = judge_fn(question, result.get("answer", ""))
                return result

        return _LCAgent()

    return factory


def build_p1_lc_factory_financebench(
    embedder: Any,
    store: Any,
    scope_of=None,
    llm_judge: Any = None,
):
    """P1-LC: seed phase — run Long-Context backbone and write location hints.

    Writes each provided docid as a HintRecord tagged positive/negative based
    on an LLMJudge verdict (mirrors build_p1_rag_factory_qasper).
    """
    import time
    import uuid

    import numpy as np

    from hgc.judge import LLMJudge
    from hgc.longctx_backbone import LongCtxBackbone
    from hgc.memory import HintRecord

    backbone = LongCtxBackbone(llm=_make_longctx_llm())
    judge = llm_judge if llm_judge is not None else LLMJudge()

    def factory(qr: dict, tools: list):  # noqa: ARG001
        _ = tools
        docs = qr.get("docs", [])
        gold = qr.get("answer", "")
        scope_id = scope_of(qr) if scope_of else f"financebench_doc_{qr.get('doc_name', '?')}"

        class _P1LCAgent:
            def run(self, question: str) -> dict:
                result = backbone.run(question, docs=docs)
                answer = result.get("answer", "")
                judgment = (
                    bool(judge.judge(question, gold, answer).correct) if answer.strip() else False
                )

                q_emb = embedder.embed(question)
                q_emb_bytes: bytes = q_emb.astype(np.float32).tobytes()
                polarity = "positive" if judgment else "negative"
                now = time.time()

                retrieved_docids = result.get("trajectory", {}).get("retrieved_docids", [])
                added_ids: list[str] = []
                for docid in retrieved_docids:
                    record = HintRecord(
                        hint_id=str(uuid.uuid4()),
                        hint_type="location",
                        polarity=polarity,
                        content=str(docid),
                        content_meta={},
                        query_ctx=question,
                        query_ctx_embedding=q_emb_bytes,
                        trajectory_step=0,
                        created_at=now,
                        last_validated_at=now,
                        success_count=0,
                        failure_count=0,
                        retrieval_count=0,
                        scope_id=scope_id,
                    )
                    added_ids.append(store.add(record))

                result["judgment_correct"] = judgment
                result["added_hints"] = added_ids
                result["n_added_hints"] = len(added_ids)
                return result

        return _P1LCAgent()

    return factory


def build_ac_lc_factory_financebench(embedder: Any, p1_traj_dir: Path):
    """AC-LC: AnswerCache seeded from P1-LC, Long-Context fallback on cache miss."""
    from hgc.baselines import AnswerCacheAgent, _best_similarity
    from hgc.longctx_backbone import LongCtxBackbone
    from hgc.runner import seed_answer_cache_from_p1

    backbone = LongCtxBackbone(llm=_make_longctx_llm())

    seed_agent = AnswerCacheAgent(
        tools=[],
        embedder=lambda text: embedder.embed(text),
        judge=lambda q, a: False,
        sim_threshold=_cache_sim_threshold(),
        max_iters=15,
    )
    p1_trajs = sorted(p1_traj_dir.glob("trajectory_q*.json")) if p1_traj_dir.exists() else []
    seed_answer_cache_from_p1(seed_agent, p1_trajs, lambda text: embedder.embed(text))
    seeded_cache = list(seed_agent._cache)
    logger.info("AC-LC: seeded AC with %d cache entries", len(seeded_cache))

    def factory(qr: dict, tools: list):  # noqa: ARG001
        _ = tools
        docs = qr.get("docs", [])
        judge_fn = make_containment_judge(qr.get("answer", ""))

        ac_agent = AnswerCacheAgent(
            tools=[],
            embedder=lambda text: embedder.embed(text),
            judge=judge_fn,
            sim_threshold=_cache_sim_threshold(),
            max_iters=15,
        )
        ac_agent._cache = list(seeded_cache)

        class _ACLCAgent:
            def run(self, question: str) -> dict:
                # Read the cache directly rather than calling ac_agent.run().
                # AnswerCacheAgent's miss path runs a tool-less ReAct rollout
                # and a judge call, and we then discard that answer in favour
                # of the long-context backbone below — so it cost a wasted
                # rollout per miss and inflated the measured per-query time.
                # ACRAGAgent (Round 5) already fixed this for the RAG cells.
                t0 = time.monotonic()
                q_emb = ac_agent._embedder(question)
                best_sim, best_idx = _best_similarity(q_emb, ac_agent._cache)
                if best_sim >= ac_agent._sim_threshold:
                    cached = ac_agent._cache[best_idx][1]
                    return {
                        "answer": cached,
                        "trajectory": {},
                        "tokens": 0,
                        "wall_time": time.monotonic() - t0,
                        "n_iters": 0,
                        "cache_hit": True,
                        "path": "cache_hit",
                        "judgment_correct": judge_fn(question, cached),
                    }
                result = backbone.run(question, docs=docs)
                result["judgment_correct"] = judge_fn(question, result.get("answer", ""))
                result["cache_hit"] = False
                result["path"] = "cache_miss"
                result["wall_time"] = time.monotonic() - t0
                return result

        return _ACLCAgent()

    return factory


def build_hgc_lc_factory_financebench(
    embedder: Any,
    p1_db: Path,
    p1_traj_dir: Path,
    scope_of=None,
):
    """HGC-LC: AC + Long-Context fallback + gate, seeded from P1-LC (mirrors HGC-RAG)."""
    from hgc.baselines import AnswerCacheAgent
    from hgc.gate import CompositeGate, SupportVerifier
    from hgc.hgc_rag import HGCRAGAgent
    from hgc.longctx_backbone import LongCtxBackbone
    from hgc.memory import HintStore
    from hgc.runner import seed_answer_cache_from_p1

    backbone = LongCtxBackbone(llm=_make_longctx_llm())

    seed_agent = AnswerCacheAgent(
        tools=[],
        embedder=lambda text: embedder.embed(text),
        judge=lambda q, a: False,
        sim_threshold=_cache_sim_threshold(),
        max_iters=15,
    )
    p1_trajs = sorted(p1_traj_dir.glob("trajectory_q*.json")) if p1_traj_dir.exists() else []
    seed_answer_cache_from_p1(seed_agent, p1_trajs, lambda text: embedder.embed(text))
    seeded_cache = list(seed_agent._cache)
    logger.info("HGC-LC: seeded AC with %d cache entries from P1-LC", len(seeded_cache))

    store = HintStore(db_path=str(p1_db))
    verifier = SupportVerifier(
        lm=_gate_lm(),
        max_doc_chars=_verifier_max_doc_chars(),
        containment_min_chars=_verifier_containment_min_chars(),
        predicate=_verifier_predicate(),
    )

    def factory(qr: dict, tools: list):  # noqa: ARG001
        _ = tools
        judge_fn = make_containment_judge(qr.get("answer", ""))
        scope_id = scope_of(qr) if scope_of else f"financebench_doc_{qr.get('doc_name', '?')}"
        qr_docs = qr.get("docs", [])

        ac_agent = AnswerCacheAgent(
            tools=[],
            embedder=lambda text: embedder.embed(text),
            judge=judge_fn,
            sim_threshold=_cache_sim_threshold(),
            max_iters=15,
        )
        ac_agent._cache = list(seeded_cache)

        class _BoundLCBackbone:
            def run(self, question: str, docs: list[dict] | None = None) -> dict:
                return backbone.run(question, docs=docs if docs is not None else qr_docs)

        return HGCRAGAgent(
            ac_agent=ac_agent,
            rag_backbone=_BoundLCBackbone(),
            store=store,
            embedder=lambda text: embedder.embed(text),
            hint_retrieval_params={"scope_id": scope_id},
            gate=CompositeGate(support_verifier=verifier, enabled=_enabled_gate_set()),
            top_k_hints=_top_k_hints(),
            strict_no_hint=_strict_no_hint(),
        )

    return factory


def build_p4_ac_lc_factory_financebench(
    embedder: Any, p1_traj_dir: Path, seed: int = 42, fraction: float = 0.20
):
    """C-AC-LC: AnswerCache seeded from P1-LC then cross-swap contaminated."""
    from hgc.baselines import AnswerCacheAgent, _best_similarity
    from hgc.contaminators.cross_swap import contaminate_answer_cache_from_p1
    from hgc.longctx_backbone import LongCtxBackbone

    backbone = LongCtxBackbone(llm=_make_longctx_llm())

    seed_agent = AnswerCacheAgent(
        tools=[],
        embedder=lambda text: embedder.embed(text),
        judge=lambda q, a: False,
        sim_threshold=_cache_sim_threshold(),
        max_iters=15,
    )
    p1_trajs = sorted(p1_traj_dir.glob("trajectory_q*.json")) if p1_traj_dir.exists() else []
    try:
        corrupted = contaminate_answer_cache_from_p1(
            seed_agent, p1_trajs, lambda text: embedder.embed(text), seed=seed, fraction=fraction
        )
        logger.info("C-AC-LC: contaminated %d cache entries", len(corrupted))
    except ValueError as e:
        logger.warning("C-AC-LC contamination skipped: %s", e)
    seeded_cache = list(seed_agent._cache)

    def factory(qr: dict, tools: list):  # noqa: ARG001
        _ = tools
        docs = qr.get("docs", [])
        judge_fn = make_containment_judge(qr.get("answer", ""))

        ac_agent = AnswerCacheAgent(
            tools=[],
            embedder=lambda text: embedder.embed(text),
            judge=judge_fn,
            sim_threshold=_cache_sim_threshold(),
            max_iters=15,
        )
        ac_agent._cache = list(seeded_cache)

        class _CACLCAgent:
            def run(self, question: str) -> dict:
                # Read the cache directly rather than calling ac_agent.run().
                # AnswerCacheAgent's miss path runs a tool-less ReAct rollout
                # and a judge call, and we then discard that answer in favour
                # of the long-context backbone below — so it cost a wasted
                # rollout per miss and inflated the measured per-query time.
                # ACRAGAgent (Round 5) already fixed this for the RAG cells.
                t0 = time.monotonic()
                q_emb = ac_agent._embedder(question)
                best_sim, best_idx = _best_similarity(q_emb, ac_agent._cache)
                if best_sim >= ac_agent._sim_threshold:
                    cached = ac_agent._cache[best_idx][1]
                    return {
                        "answer": cached,
                        "trajectory": {},
                        "tokens": 0,
                        "wall_time": time.monotonic() - t0,
                        "n_iters": 0,
                        "cache_hit": True,
                        "path": "cache_hit",
                        "judgment_correct": judge_fn(question, cached),
                    }
                result = backbone.run(question, docs=docs)
                result["judgment_correct"] = judge_fn(question, result.get("answer", ""))
                result["cache_hit"] = False
                result["path"] = "cache_miss"
                result["wall_time"] = time.monotonic() - t0
                return result

        return _CACLCAgent()

    return factory


def build_p4_hgc_lc_factory_financebench(
    embedder: Any,
    p1_db: Path,
    p1_traj_dir: Path,
    p4_db: Path,
    scope_of=None,
    seed: int = 42,
    fraction: float = 0.20,
):
    """C-HGC-LC: Contaminated HGC on Long-Context (mirrors C-HGC-RAG)."""
    from hgc.baselines import AnswerCacheAgent
    from hgc.contaminators.cross_swap import HintContaminator, contaminate_answer_cache_from_p1
    from hgc.gate import CompositeGate, SupportVerifier
    from hgc.hgc_rag import HGCRAGAgent
    from hgc.longctx_backbone import LongCtxBackbone
    from hgc.memory import HintStore

    backbone = LongCtxBackbone(llm=_make_longctx_llm())

    seed_agent = AnswerCacheAgent(
        tools=[],
        embedder=lambda text: embedder.embed(text),
        judge=lambda q, a: False,
        sim_threshold=_cache_sim_threshold(),
        max_iters=15,
    )
    p1_trajs = sorted(p1_traj_dir.glob("trajectory_q*.json")) if p1_traj_dir.exists() else []
    try:
        corrupted_indices = contaminate_answer_cache_from_p1(
            seed_agent, p1_trajs, lambda text: embedder.embed(text), seed=seed, fraction=fraction
        )
        logger.info("C-HGC-LC AC: contaminated %d cache entries", len(corrupted_indices))
    except ValueError as e:
        logger.warning("C-HGC-LC AC contamination skipped: %s", e)
    seeded_cache = list(seed_agent._cache)

    p4_db.parent.mkdir(parents=True, exist_ok=True)
    if p1_db.exists() and not p4_db.exists():
        shutil.copy2(str(p1_db), str(p4_db))
    elif not p1_db.exists():
        logger.warning("C-HGC-LC: P1 store missing at %s; using empty store", p1_db)

    contaminated_store = HintStore(db_path=str(p4_db))
    contaminator = HintContaminator(contaminated_store, seed=seed, fraction=fraction)
    corrupted_ids = contaminator.corrupt()
    logger.info("C-HGC-LC HintStore: contaminated %d hints", len(corrupted_ids))

    verifier = SupportVerifier(
        lm=_gate_lm(),
        max_doc_chars=_verifier_max_doc_chars(),
        containment_min_chars=_verifier_containment_min_chars(),
        predicate=_verifier_predicate(),
    )

    def factory(qr: dict, tools: list):  # noqa: ARG001
        _ = tools
        judge_fn = make_containment_judge(qr.get("answer", ""))
        scope_id = scope_of(qr) if scope_of else f"financebench_doc_{qr.get('doc_name', '?')}"

        ac_agent = AnswerCacheAgent(
            tools=[],
            embedder=lambda text: embedder.embed(text),
            judge=judge_fn,
            sim_threshold=_cache_sim_threshold(),
            max_iters=15,
        )
        ac_agent._cache = list(seeded_cache)

        class _BoundLCBackbone:
            def run(self, question: str, docs: list[dict] | None = None) -> dict:
                return backbone.run(question, docs=docs if docs is not None else qr.get("docs", []))

        return HGCRAGAgent(
            ac_agent=ac_agent,
            rag_backbone=_BoundLCBackbone(),
            store=contaminated_store,
            embedder=lambda text: embedder.embed(text),
            hint_retrieval_params={"scope_id": scope_id},
            gate=CompositeGate(support_verifier=verifier, enabled=_enabled_gate_set()),
            top_k_hints=_top_k_hints(),
            strict_no_hint=_strict_no_hint(),
        )

    return factory


# ---------------------------------------------------------------------------
# Tool factory — FinanceBench
# ---------------------------------------------------------------------------


def build_adaptive_hgc_lc_factory_financebench(
    embedder: Any,
    p1_db: Path,
    p1_traj_dir: Path,
    p4_db: Path,
    scope_of=None,
    seed: int = 42,
    fraction: float = 0.20,
):
    """A-HGC-LC: HGC on Long-Context under *gate-aware* contamination.

    Identical to :func:`build_p4_hgc_lc_factory_financebench` except for how
    the poison is built, so the two phases differ in one variable.

    Two deliberate departures from the cross-swap protocol, both required by
    what the attack is:

    - The poisoned answer is a span lifted from the victim entry's **own**
      source document, so it clears G3's containment fast path without an LLM
      call (see :mod:`hgc.contaminators.gate_aware`).
    - The HintStore is **not** contaminated. Cross-swap corrupts the hint as
      well, which is what lets G2 reject it; an adversary aiming at the gate
      leaves the hint valid on purpose, because a resolvable hint is exactly
      what carries the entry past G1 and G2. Reporting this phase beside
      C-HGC-LC therefore compares two threat models, not two corruption
      rates.
    """
    from hgc.baselines import AnswerCacheAgent
    from hgc.contaminators.gate_aware import contaminate_answer_cache_from_p1
    from hgc.gate import CompositeGate, SupportVerifier
    from hgc.hgc_rag import HGCRAGAgent
    from hgc.longctx_backbone import LongCtxBackbone
    from hgc.memory import HintStore

    backbone = LongCtxBackbone(llm=_make_longctx_llm())

    seed_agent = AnswerCacheAgent(
        tools=[],
        embedder=lambda text: embedder.embed(text),
        judge=lambda q, a: False,
        sim_threshold=_cache_sim_threshold(),
        max_iters=15,
    )
    p1_trajs = sorted(p1_traj_dir.glob("trajectory_q*.json")) if p1_traj_dir.exists() else []
    try:
        corrupted_indices = contaminate_answer_cache_from_p1(
            seed_agent, p1_trajs, lambda text: embedder.embed(text), seed=seed, fraction=fraction
        )
        logger.info(
            "A-HGC-LC AC: gate-aware poison on %d cache entries",
            len(corrupted_indices),
        )
    except ValueError as e:
        logger.warning("A-HGC-LC AC contamination skipped: %s", e)
    seeded_cache = list(seed_agent._cache)

    p4_db.parent.mkdir(parents=True, exist_ok=True)
    if p1_db.exists() and not p4_db.exists():
        shutil.copy2(str(p1_db), str(p4_db))
    elif not p1_db.exists():
        logger.warning("A-HGC-LC: P1 store missing at %s; using empty store", p1_db)

    # Hints are copied from P1 untouched — see the docstring. A valid hint is
    # the attack's vehicle through G1 and G2, not collateral damage.
    contaminated_store = HintStore(db_path=str(p4_db))
    logger.info("A-HGC-LC HintStore: hints left intact (gate-aware threat model)")

    verifier = SupportVerifier(
        lm=_gate_lm(),
        max_doc_chars=_verifier_max_doc_chars(),
        containment_min_chars=_verifier_containment_min_chars(),
        predicate=_verifier_predicate(),
    )

    def factory(qr: dict, tools: list):  # noqa: ARG001
        _ = tools
        judge_fn = make_containment_judge(qr.get("answer", ""))
        scope_id = scope_of(qr) if scope_of else f"financebench_doc_{qr.get('doc_name', '?')}"

        ac_agent = AnswerCacheAgent(
            tools=[],
            embedder=lambda text: embedder.embed(text),
            judge=judge_fn,
            sim_threshold=_cache_sim_threshold(),
            max_iters=15,
        )
        ac_agent._cache = list(seeded_cache)

        class _BoundLCBackbone:
            def run(self, question: str, docs: list[dict] | None = None) -> dict:
                return backbone.run(question, docs=docs if docs is not None else qr.get("docs", []))

        return HGCRAGAgent(
            ac_agent=ac_agent,
            rag_backbone=_BoundLCBackbone(),
            store=contaminated_store,
            embedder=lambda text: embedder.embed(text),
            hint_retrieval_params={"scope_id": scope_id},
            gate=CompositeGate(support_verifier=verifier, enabled=_enabled_gate_set()),
            top_k_hints=_top_k_hints(),
            strict_no_hint=_strict_no_hint(),
        )

    return factory


def _gate_aware_answer_cache(
    phase: str,
    embedder: Any,
    p1_traj_dir: Path,
    contexts_of,
    seed: int,
    fraction: float,
) -> list:
    """Seed an AnswerCache from P1 and poison it with gate-passing spans.

    Shared by the gate-aware phases whose trajectory does not carry the
    document (A-Hybrid on BCP, A-HGC-RAG on QASPER). Zero poisoned entries
    means the span source is wrong, not that the attack failed, so it is an
    error here rather than a warning: the phase would otherwise run a clean
    cache under a contaminated name.
    """
    from hgc.baselines import AnswerCacheAgent
    from hgc.contaminators.gate_aware import contaminate_answer_cache_from_p1

    seed_agent = AnswerCacheAgent(
        tools=[],
        embedder=lambda text: embedder.embed(text),
        judge=lambda q, a: False,
        sim_threshold=_cache_sim_threshold(),
        max_iters=15,
    )
    p1_trajs = sorted(p1_traj_dir.glob("trajectory_q*.json")) if p1_traj_dir.exists() else []
    corrupted_indices = contaminate_answer_cache_from_p1(
        seed_agent,
        p1_trajs,
        lambda text: embedder.embed(text),
        seed=seed,
        fraction=fraction,
        contexts_of=contexts_of,
    )
    n_victims = int(fraction * len(seed_agent._cache))
    logger.info(
        "%s AC: gate-aware poison on %d cache entries (%d victims drawn)",
        phase,
        len(corrupted_indices),
        n_victims,
    )
    if n_victims and not corrupted_indices:
        raise RuntimeError(
            f"{phase}: gate-aware contamination poisoned 0 of {n_victims} victims; "
            "the span source does not reach the documents the gate reads"
        )
    return list(seed_agent._cache)


def build_adaptive_hybrid_factory_bcp(
    embedder: Any,
    p1_db: Path,
    p1_traj_dir: Path,
    extractor: Any,
    a_db: Path,
    queries: list[dict],
    scope_of=None,
    seed: int = 42,
    fraction: float = 0.20,
) -> Callable:
    """A-Hybrid: HGC on BCP under gate-aware contamination.

    :func:`build_p4_hybrid_factory` with the poison built as in
    :func:`build_adaptive_hgc_lc_factory_financebench`: the victim's answer is
    replaced by a sentence from the document its own location hint resolves
    to, and the HintStore is copied from P1 untouched. The ReAct trajectory
    does not record the page it read, so the span comes from the dataset
    record via :func:`hgc.contaminators.gate_aware.hint_document_contexts`.
    """
    from hgc.baselines import AnswerCacheAgent
    from hgc.contaminators.gate_aware import hint_document_contexts
    from hgc.gate import CompositeGate, SupportVerifier
    from hgc.hgc_agent import HGCAgent
    from hgc.hgc_core import HGCCoreAgent
    from hgc.memory import HintStore

    _scope_of = scope_of or (lambda qr: f"bcp_query_{qr['query_id']}")

    a_db.parent.mkdir(parents=True, exist_ok=True)
    if p1_db.exists() and not a_db.exists():
        shutil.copy2(str(p1_db), str(a_db))
    elif not p1_db.exists():
        logger.warning("A-Hybrid: P1 store missing at %s; using empty store", p1_db)
    contaminated_store = HintStore(db_path=str(a_db))
    logger.info("A-Hybrid HintStore: hints left intact (gate-aware threat model)")

    seeded_cache = _gate_aware_answer_cache(
        "A-Hybrid",
        embedder,
        p1_traj_dir,
        hint_document_contexts(
            contaminated_store,
            lambda text: embedder.embed(text),
            queries,
            _scope_of,
            top_k=_top_k_hints(),
        ),
        seed,
        fraction,
    )

    verifier = SupportVerifier(
        lm=_gate_lm(),
        max_doc_chars=_verifier_max_doc_chars(),
        containment_min_chars=_verifier_containment_min_chars(),
        predicate=_verifier_predicate(),
    )

    def factory(qr: dict, tools: list):
        judge_fn = make_containment_judge(qr.get("answer", ""))
        scope_id = _scope_of(qr)

        ac_agent = AnswerCacheAgent(
            tools=[],
            embedder=lambda text: embedder.embed(text),
            judge=judge_fn,
            sim_threshold=_cache_sim_threshold(),
            max_iters=15,
        )
        ac_agent._cache = list(seeded_cache)

        hgc_agent = HGCCoreAgent(
            tools=tools,
            store=contaminated_store,
            embedder=embedder,
            extractor=extractor,
            judge=judge_fn,
            max_iters=15,
            scope_id=scope_id,
        )

        return HGCAgent(
            ac_agent=ac_agent,
            core_agent=hgc_agent,
            store=contaminated_store,
            embedder=lambda text: embedder.embed(text),
            gate=CompositeGate(support_verifier=verifier, enabled=_enabled_gate_set()),
            top_k_hints=_top_k_hints(),
            strict_no_hint=_strict_no_hint(),
        )

    return factory


def build_adaptive_hgc_rag_factory_qasper(
    embedder: Any,
    p1_db: Path,
    p1_traj_dir: Path,
    extractor: Any,
    a_db: Path,
    queries: list[dict],
    scope_of=None,
    seed: int = 42,
    fraction: float = 0.20,
):
    """A-HGC-RAG: HGC on QASPER-RAG under gate-aware contamination.

    :func:`build_p4_hgc_rag_factory_qasper` with the poison built as in
    :func:`build_adaptive_hgc_lc_factory_financebench`. The RAG trajectory's
    ``context`` concatenates five retrieved sections and the gate reads one
    of them, so the span is drawn from the section the entry's own location
    hint resolves to rather than from the concatenation. HintStore copied
    from P1 untouched.
    """
    from langchain_openai import AzureChatOpenAI

    from hgc.baselines import AnswerCacheAgent
    from hgc.contaminators.gate_aware import hint_document_contexts
    from hgc.gate import CompositeGate, SupportVerifier
    from hgc.hgc_rag import HGCRAGAgent
    from hgc.memory import HintStore
    from hgc.rag_backbone import RAGBackbone

    _ = extractor  # signature parity with build_p4_hgc_rag_factory_qasper
    _scope_of = scope_of or (lambda qr: f"qasper_paper_{qr.get('paper_id', qr['query_id'])}")

    deployment = os.environ.get("AZURE_OPENAI_DEPLOYMENT", "gpt-4.1")
    llm = AzureChatOpenAI(
        azure_deployment=deployment,
        azure_endpoint=os.environ.get("AZURE_OPENAI_ENDPOINT", ""),
        api_key=os.environ.get("AZURE_OPENAI_API_KEY", ""),
        api_version=os.environ.get("AZURE_OPENAI_API_VERSION", ""),
        temperature=0,
        max_tokens=1024,
    )
    backbone = RAGBackbone(llm=llm, embedder=_make_langchain_embedder(embedder))

    a_db.parent.mkdir(parents=True, exist_ok=True)
    if p1_db.exists() and not a_db.exists():
        shutil.copy2(str(p1_db), str(a_db))
    elif not p1_db.exists():
        logger.warning("A-HGC-RAG: P1 store missing at %s; using empty store", p1_db)
    contaminated_store = HintStore(db_path=str(a_db))
    logger.info("A-HGC-RAG HintStore: hints left intact (gate-aware threat model)")

    seeded_cache = _gate_aware_answer_cache(
        "A-HGC-RAG",
        embedder,
        p1_traj_dir,
        hint_document_contexts(
            contaminated_store,
            lambda text: embedder.embed(text),
            queries,
            _scope_of,
            top_k=_top_k_hints(),
        ),
        seed,
        fraction,
    )

    verifier = SupportVerifier(
        lm=_gate_lm(),
        max_doc_chars=_verifier_max_doc_chars(),
        containment_min_chars=_verifier_containment_min_chars(),
        predicate=_verifier_predicate(),
    )

    def factory(qr: dict, tools: list):  # noqa: ARG001
        _ = tools
        judge_fn = make_containment_judge(qr.get("answer", ""))
        scope_id = _scope_of(qr)

        ac_agent = AnswerCacheAgent(
            tools=[],
            embedder=lambda text: embedder.embed(text),
            judge=judge_fn,
            sim_threshold=_cache_sim_threshold(),
            max_iters=15,
        )
        ac_agent._cache = list(seeded_cache)

        class _BoundRAGBackbone:
            def run(self, question: str, docs: list[dict] | None = None) -> dict:
                return backbone.run(question, docs=docs)

        return HGCRAGAgent(
            ac_agent=ac_agent,
            rag_backbone=_BoundRAGBackbone(),
            store=contaminated_store,
            embedder=lambda text: embedder.embed(text),
            hint_retrieval_params={"scope_id": scope_id},
            gate=CompositeGate(support_verifier=verifier, enabled=_enabled_gate_set()),
            top_k_hints=_top_k_hints(),
            strict_no_hint=_strict_no_hint(),
        )

    return factory


# ---------------------------------------------------------------------------
# Tool factory — FinanceBench
# ---------------------------------------------------------------------------


def make_tools_financebench(qr: dict) -> list:
    """Build the 4 standard document tools for a FinanceBench query record.

    FinanceBench records carry docs directly under "docs" (same as QASPER).
    """
    docs = qr.get("docs", [])
    doc_by_id = {d["docid"]: d for d in docs}

    def list_document_ids() -> list:
        return [d["docid"] for d in docs]

    def get_document_title(docid: str) -> str:
        d = doc_by_id.get(docid)
        if not d:
            return f"ERROR: docid {docid!r} not found"
        return d.get("text", "").split("\n", 1)[0][:300]

    def get_document_snippet(docid: str, char_start: int = 0, char_len: int = 2000) -> str:
        d = doc_by_id.get(docid)
        if not d:
            return f"ERROR: docid {docid!r} not found"
        return d.get("text", "")[char_start : char_start + char_len]

    def search_documents(keyword: str, max_hits: int = 10) -> list:
        """Substring search; returns list of {docid, snippet}."""
        hits = []
        kl = keyword.lower()
        for d in docs:
            idx = d.get("text", "").lower().find(kl)
            if idx != -1:
                t = d["text"]
                hits.append(
                    {
                        "docid": d["docid"],
                        "snippet": t[max(0, idx - 80) : idx + 120],
                    }
                )
                if len(hits) >= max_hits:
                    break
        return hits

    return [list_document_ids, get_document_title, get_document_snippet, search_documents]


# ---------------------------------------------------------------------------
# Phase factory builders — FinanceBench
# ---------------------------------------------------------------------------


def build_hgc_core_factory_financebench(
    embedder: Any,
    store: Any,
    extractor: Any,
    scope_of=None,
):
    """Return a HGCCoreAgent factory for FinanceBench queries.

    Parameters
    ----------
    scope_of:
        Optional callable ``(qr) -> str`` for the scope_id.
        Defaults to ``f"financebench_doc_{qr['doc_name']}_{qr['doc_period']}"``.
        Doc-level scope is used because multiple questions share the same filing.
    """
    from hgc.hgc_core import HGCCoreAgent

    def factory(qr: dict, tools: list):
        judge_fn = make_containment_judge(qr.get("answer", ""))
        scope_id = (
            scope_of(qr)
            if scope_of
            else f"financebench_doc_{qr.get('doc_name', qr['query_id'])}_{qr.get('doc_period', '')}"
        )
        return HGCCoreAgent(
            tools=tools,
            store=store,
            embedder=embedder,
            extractor=extractor,
            judge=judge_fn,
            max_iters=15,
            scope_id=scope_id,
        )

    return factory


def build_ac_factory_financebench(embedder: Any, p1_traj_dir: Path):
    """AnswerCacheAgent factory seeded from FinanceBench P1 trajectories."""
    from hgc.baselines import AnswerCacheAgent
    from hgc.runner import seed_answer_cache_from_p1

    seed_agent = AnswerCacheAgent(
        tools=[],
        embedder=lambda text: embedder.embed(text),
        judge=lambda q, a: False,
        sim_threshold=_cache_sim_threshold(),
        max_iters=15,
    )
    p1_trajs = sorted(p1_traj_dir.glob("trajectory_q*.json")) if p1_traj_dir.exists() else []
    seed_answer_cache_from_p1(seed_agent, p1_trajs, lambda text: embedder.embed(text))
    seeded_cache = list(seed_agent._cache)

    def factory(qr: dict, tools: list):
        judge_fn = make_containment_judge(qr.get("answer", ""))
        agent = AnswerCacheAgent(
            tools=tools,
            embedder=lambda text: embedder.embed(text),
            judge=judge_fn,
            sim_threshold=_cache_sim_threshold(),
            max_iters=15,
        )
        agent._cache = list(seeded_cache)
        return agent

    return factory


def build_m0_factory_financebench(
    p1_traj_dir: Path, mem0_config: dict | None = None, user_id: str = "p3_m0_financebench"
):
    """Mem0ReActAgent factory seeded from FinanceBench P1."""
    from hgc.baselines import Mem0ReActAgent
    from hgc.runner import seed_mem0_from_p1

    if mem0_config is not None:
        _config = mem0_config
        if isinstance(_config, dict):
            from mem0.configs.base import MemoryConfig as _MemoryConfig

            _config = _MemoryConfig(**_config)
        from mem0 import Memory as _Memory

        shared_memory = _Memory(config=_config)
    else:
        from mem0 import Memory as _Memory

        shared_memory = _Memory()

    p1_trajs = sorted(p1_traj_dir.glob("trajectory_q*.json")) if p1_traj_dir.exists() else []
    seed_mem0_from_p1(shared_memory, p1_trajs, user_id=user_id)

    def factory(qr: dict, tools: list):
        judge_fn = make_containment_judge(qr.get("answer", ""))
        return Mem0ReActAgent(
            tools=tools,
            judge=judge_fn,
            user_id=user_id,
            mem0_memory=shared_memory,
            max_iters=15,
        )

    return factory


def build_p4_factory_financebench(
    embedder: Any,
    p1_db: Path,
    extractor: Any,
    p4_db: Path,
    scope_of=None,
):
    """P4: HGCCoreAgent with contaminated copy of FinanceBench P1's store."""
    from hgc.contaminators.cross_swap import HintContaminator
    from hgc.hgc_core import HGCCoreAgent
    from hgc.memory import HintStore

    p4_db.parent.mkdir(parents=True, exist_ok=True)
    if p1_db.exists() and not p4_db.exists():
        shutil.copy2(str(p1_db), str(p4_db))
    elif not p1_db.exists():
        logger.warning("P4-FinanceBench: P1 store missing at %s; using empty store", p1_db)

    contaminated_store = HintStore(db_path=str(p4_db))
    contaminator = HintContaminator(contaminated_store, seed=42, fraction=0.20)
    corrupted_ids = contaminator.corrupt()
    logger.info("P4-FinanceBench: contaminated %d hints in copied store", len(corrupted_ids))

    def factory(qr: dict, tools: list):
        judge_fn = make_containment_judge(qr.get("answer", ""))
        scope_id = (
            scope_of(qr)
            if scope_of
            else f"financebench_doc_{qr.get('doc_name', qr['query_id'])}_{qr.get('doc_period', '')}"
        )
        return HGCCoreAgent(
            tools=tools,
            store=contaminated_store,
            embedder=embedder,
            extractor=extractor,
            judge=judge_fn,
            max_iters=15,
            scope_id=scope_id,
        )

    return factory


def build_p4_ac_factory_financebench(
    embedder: Any, p1_traj_dir: Path, seed: int = 42, fraction: float = 0.20
):
    """P4-AC: AnswerCacheAgent seeded from FinanceBench P1 then contaminated via 20% cross-swap."""
    from hgc.baselines import AnswerCacheAgent
    from hgc.contaminators.cross_swap import contaminate_answer_cache_from_p1

    seed_agent = AnswerCacheAgent(
        tools=[],
        embedder=lambda text: embedder.embed(text),
        judge=lambda q, a: False,
        sim_threshold=_cache_sim_threshold(),
        max_iters=15,
    )
    p1_trajs = sorted(p1_traj_dir.glob("trajectory_q*.json")) if p1_traj_dir.exists() else []
    corrupted_indices = contaminate_answer_cache_from_p1(
        seed_agent, p1_trajs, lambda text: embedder.embed(text), seed=seed, fraction=fraction
    )
    logger.info("P4-AC-FinanceBench: contaminated %d cache entries", len(corrupted_indices))
    seeded_cache = list(seed_agent._cache)

    def factory(qr: dict, tools: list):
        judge_fn = make_containment_judge(qr.get("answer", ""))
        agent = AnswerCacheAgent(
            tools=tools,
            embedder=lambda text: embedder.embed(text),
            judge=judge_fn,
            sim_threshold=_cache_sim_threshold(),
            max_iters=15,
        )
        agent._cache = list(seeded_cache)
        return agent

    return factory


def build_p4_m0_factory_financebench(
    p1_traj_dir: Path,
    mem0_config: dict | None = None,
    user_id: str = "p4_m0_financebench",
    seed: int = 42,
    fraction: float = 0.20,
):
    """P4-M0: Mem0ReActAgent seeded from FinanceBench P1 then contaminated via 20% cross-swap."""
    from hgc.baselines import Mem0ReActAgent
    from hgc.contaminators.cross_swap import contaminate_mem0_from_p1

    if mem0_config is not None:
        _config = mem0_config
        if isinstance(_config, dict):
            from mem0.configs.base import MemoryConfig as _MemoryConfig

            _config = _MemoryConfig(**_config)
        from mem0 import Memory as _Memory

        shared_memory = _Memory(config=_config)
    else:
        from mem0 import Memory as _Memory

        shared_memory = _Memory()

    p1_trajs = sorted(p1_traj_dir.glob("trajectory_q*.json")) if p1_traj_dir.exists() else []
    corrupted_ids = contaminate_mem0_from_p1(
        shared_memory, p1_trajs, seed=seed, fraction=fraction, user_id=user_id
    )
    logger.info("P4-M0-FinanceBench: contaminated %d mem0 entries", len(corrupted_ids))

    def factory(qr: dict, tools: list):
        judge_fn = make_containment_judge(qr.get("answer", ""))
        return Mem0ReActAgent(
            tools=tools,
            judge=judge_fn,
            user_id=user_id,
            mem0_memory=shared_memory,
            max_iters=15,
        )

    return factory


def build_hybrid_factory_financebench(
    embedder: Any,
    p1_db: Path,
    p1_traj_dir: Path,
    extractor: Any,
    scope_of=None,
):
    """Clean Hybrid (P3-Hybrid) for FinanceBench: AC fast path + hint re-verification."""
    from hgc.baselines import AnswerCacheAgent
    from hgc.hgc_agent import HGCAgent
    from hgc.hgc_core import HGCCoreAgent
    from hgc.memory import HintStore
    from hgc.runner import seed_answer_cache_from_p1

    seed_agent = AnswerCacheAgent(
        tools=[],
        embedder=lambda text: embedder.embed(text),
        judge=lambda q, a: False,
        sim_threshold=_cache_sim_threshold(),
        max_iters=15,
    )
    p1_trajs = sorted(p1_traj_dir.glob("trajectory_q*.json")) if p1_traj_dir.exists() else []
    seed_answer_cache_from_p1(seed_agent, p1_trajs, lambda text: embedder.embed(text))
    seeded_cache = list(seed_agent._cache)
    logger.info(
        "P3-Hybrid-FinanceBench: seeded AC with %d cache entries from P1", len(seeded_cache)
    )

    store = HintStore(db_path=str(p1_db))

    def factory(qr: dict, tools: list):
        judge_fn = make_containment_judge(qr.get("answer", ""))
        scope_id = (
            scope_of(qr)
            if scope_of
            else f"financebench_doc_{qr.get('doc_name', qr['query_id'])}_{qr.get('doc_period', '')}"
        )

        ac_agent = AnswerCacheAgent(
            tools=[],
            embedder=lambda text: embedder.embed(text),
            judge=make_containment_judge(qr.get("answer", "")),
            sim_threshold=_cache_sim_threshold(),
            max_iters=15,
        )
        ac_agent._cache = list(seeded_cache)

        hgc_agent = HGCCoreAgent(
            tools=tools,
            store=store,
            embedder=embedder,
            extractor=extractor,
            judge=judge_fn,
            max_iters=15,
            scope_id=scope_id,
        )

        return HGCAgent(
            ac_agent=ac_agent,
            core_agent=hgc_agent,
            store=store,
            embedder=lambda text: embedder.embed(text),
            top_k_hints=_top_k_hints(),
        )

    return factory


def build_p4_hybrid_factory_financebench(
    embedder: Any,
    p1_db: Path,
    p1_traj_dir: Path,
    extractor: Any,
    p4_db: Path,
    scope_of=None,
    seed: int = 42,
    fraction: float = 0.20,
):
    """Contaminated Hybrid (P4-Hybrid) for FinanceBench."""
    from hgc.baselines import AnswerCacheAgent
    from hgc.contaminators.cross_swap import HintContaminator, contaminate_answer_cache_from_p1
    from hgc.hgc_agent import HGCAgent
    from hgc.hgc_core import HGCCoreAgent
    from hgc.memory import HintStore

    seed_agent = AnswerCacheAgent(
        tools=[],
        embedder=lambda text: embedder.embed(text),
        judge=lambda q, a: False,
        sim_threshold=_cache_sim_threshold(),
        max_iters=15,
    )
    p1_trajs = sorted(p1_traj_dir.glob("trajectory_q*.json")) if p1_traj_dir.exists() else []
    corrupted_indices = contaminate_answer_cache_from_p1(
        seed_agent, p1_trajs, lambda text: embedder.embed(text), seed=seed, fraction=fraction
    )
    logger.info("P4-Hybrid-FinanceBench AC: contaminated %d cache entries", len(corrupted_indices))
    seeded_cache = list(seed_agent._cache)

    p4_db.parent.mkdir(parents=True, exist_ok=True)
    if p1_db.exists() and not p4_db.exists():
        shutil.copy2(str(p1_db), str(p4_db))
    elif not p1_db.exists():
        logger.warning("P4-Hybrid-FinanceBench: P1 store missing at %s; using empty store", p1_db)

    contaminated_store = HintStore(db_path=str(p4_db))
    contaminator = HintContaminator(contaminated_store, seed=seed, fraction=fraction)
    corrupted_ids = contaminator.corrupt()
    logger.info(
        "P4-Hybrid-FinanceBench HintStore: contaminated %d hints in copied store",
        len(corrupted_ids),
    )

    def factory(qr: dict, tools: list):
        judge_fn = make_containment_judge(qr.get("answer", ""))
        scope_id = (
            scope_of(qr)
            if scope_of
            else f"financebench_doc_{qr.get('doc_name', qr['query_id'])}_{qr.get('doc_period', '')}"
        )

        ac_agent = AnswerCacheAgent(
            tools=[],
            embedder=lambda text: embedder.embed(text),
            judge=make_containment_judge(qr.get("answer", "")),
            sim_threshold=_cache_sim_threshold(),
            max_iters=15,
        )
        ac_agent._cache = list(seeded_cache)

        hgc_agent = HGCCoreAgent(
            tools=tools,
            store=contaminated_store,
            embedder=embedder,
            extractor=extractor,
            judge=judge_fn,
            max_iters=15,
            scope_id=scope_id,
        )

        return HGCAgent(
            ac_agent=ac_agent,
            core_agent=hgc_agent,
            store=contaminated_store,
            embedder=lambda text: embedder.embed(text),
            top_k_hints=_top_k_hints(),
        )

    return factory


# ---------------------------------------------------------------------------
# Summary helpers
# ---------------------------------------------------------------------------


def phase_summary(phase_name: str, results: list) -> dict:
    n = len(results)
    if n == 0:
        return {
            "phase": phase_name,
            "n_queries": 0,
            "n_correct_judge": 0,
            "n_correct_containment": 0,
            "avg_tokens": 0,
            "avg_time_s": 0.0,
            "avg_iters": 0.0,
            "total_tokens": 0,
            "estimated_cost_usd": 0.0,
        }
    total_tokens = sum(r.tokens for r in results)
    estimated_cost = total_tokens / 1_000 * COST_PER_1K_TOKENS
    return {
        "phase": phase_name,
        "n_queries": n,
        "n_correct_judge": sum(1 for r in results if r.judgment_correct),
        "n_correct_containment": sum(1 for r in results if r.containment_correct),
        "avg_tokens": round(total_tokens / n),
        "avg_time_s": round(sum(r.time_s for r in results) / n, 2),
        "avg_iters": round(sum(r.n_iters for r in results) / n, 2),
        "total_tokens": total_tokens,
        "estimated_cost_usd": round(estimated_cost, 4),
    }


def print_phase_table(summaries: list[dict]) -> None:
    print(f"\n{'=' * 90}")
    print(
        f"{'Phase':<12} {'N':>3} {'Judge':>8} {'Contain':>8} "
        f"{'AvgTok':>8} {'AvgTime':>8} {'Cost$':>7}"
    )
    print("-" * 90)
    grand_total_tokens = 0
    grand_total_cost = 0.0
    for s in summaries:
        n = s["n_queries"]
        j = s["n_correct_judge"]
        c = s["n_correct_containment"]
        grand_total_tokens += s["total_tokens"]
        grand_total_cost += s["estimated_cost_usd"]
        print(
            f"{s['phase']:<12} {n:>3} {j:>4}/{n:<3} {c:>4}/{n:<3} "
            f"{s['avg_tokens']:>8,} {s['avg_time_s']:>7.1f}s {s['estimated_cost_usd']:>7.4f}"
        )
    print("-" * 90)
    grand_cost = grand_total_tokens / 1_000 * COST_PER_1K_TOKENS
    print(
        f"{'TOTAL':<12} {'':>3} {'':>8} {'':>8} {grand_total_tokens:>8,} {'':>8} {grand_cost:>7.4f}"
    )
    print(f"\nTotal tokens: {grand_total_tokens:,}")
    print(f"Estimated total cost: ${grand_cost:.4f} (at ${COST_PER_1K_TOKENS}/1K tokens avg)")
    print(f"{'=' * 90}")


# ---------------------------------------------------------------------------
# mem0 config builder
# ---------------------------------------------------------------------------


def build_mem0_config(out_dir: Path, collection_name: str, sub_deployment: str) -> dict:
    return {
        "llm": {
            "provider": "azure_openai",
            "config": {
                "model": os.environ.get("AZURE_OPENAI_MEM0_MODEL", sub_deployment),
                "temperature": 0,
                "azure_kwargs": {
                    "api_key": os.environ["AZURE_OPENAI_API_KEY"],
                    "azure_deployment": os.environ.get(
                        "AZURE_OPENAI_MEM0_DEPLOYMENT",
                        os.environ.get("AZURE_OPENAI_SUB_DEPLOYMENT", sub_deployment),
                    ),
                    "azure_endpoint": os.environ["AZURE_OPENAI_ENDPOINT"],
                    "api_version": os.environ["AZURE_OPENAI_API_VERSION"],
                },
            },
        },
        "embedder": {
            "provider": "azure_openai",
            "config": {
                "model": "text-embedding-3-small",
                "azure_kwargs": {
                    "api_key": os.environ["AZURE_OPENAI_API_KEY"],
                    "azure_deployment": os.environ.get(
                        "AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "text-embedding-3-small"
                    ),
                    "azure_endpoint": os.environ["AZURE_OPENAI_ENDPOINT"],
                    "api_version": os.environ["AZURE_OPENAI_API_VERSION"],
                },
            },
        },
        "vector_store": {
            "provider": "qdrant",
            "config": {
                "collection_name": collection_name,
                "embedding_model_dims": 1536,
                "path": str(out_dir / "P3-M0" / "qdrant"),
            },
        },
    }
