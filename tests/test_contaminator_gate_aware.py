"""Unit tests for hgc.contaminators.gate_aware.

The poison has to satisfy two opposing constraints at once: it must be
verbatim present in the victim's own document (so G3's containment fast path
grants support without an LLM call), and it must not contain the gold answer
(so the poisoned entry is genuinely wrong). Tests below pin both, plus the
alignment between cache index and source trajectory — if that drifts, the
poison lands on the wrong document and silently stops being gate-aware.
"""

from __future__ import annotations

import json
import time
import uuid
from pathlib import Path

import numpy as np
import pytest

from hgc.baselines import AnswerCacheAgent
from hgc.contaminators.gate_aware import (
    contaminate_answer_cache_from_p1,
    hint_document_contexts,
    rank_location_hints,
    usable_spans,
)
from hgc.gate.support_verifier import SupportVerifier
from hgc.memory import HintRecord, HintStore

_CONTEXT = (
    "[docid=paper_42_sec_0] Transformers dominate the leaderboard. "
    "We train for twelve epochs on four GPUs. "
    "The reported accuracy is 91.4 percent. "
    "Ok."
)


def _embedder(text: str) -> np.ndarray:
    vec = np.array([1.0, 0.0] if "alpha" in text.lower() else [0.0, 1.0], dtype=np.float32)
    return vec / np.linalg.norm(vec)


def _write_trajectory(
    dirpath: Path, qid: str, question: str, pred: str, gold: str, *, correct: bool = True
) -> None:
    dirpath.mkdir(parents=True, exist_ok=True)
    (dirpath / f"trajectory_q{qid}.json").write_text(
        json.dumps(
            {
                "query_id": qid,
                "question": question,
                "pred": pred,
                "gold": gold,
                "judgment_correct": correct,
                "trajectory": {"context": _CONTEXT, "retrieved_docids": ["paper_42_sec_0"]},
            }
        ),
        encoding="utf-8",
    )


def _make_agent() -> AnswerCacheAgent:
    return AnswerCacheAgent(
        tools=[], embedder=_embedder, judge=lambda q, a: False, sim_threshold=0.85, max_iters=15
    )


# --- span selection -------------------------------------------------------


def test_spans_exclude_the_gold_answer():
    """A span carrying the gold answer would be scored correct, poisoning nothing."""
    spans = usable_spans(_CONTEXT, gold="91.4 percent")
    assert spans
    assert all("91.4 percent" not in s for s in spans)


def test_spans_are_verbatim_present_in_the_document():
    """Every candidate must clear the exact check it is designed to defeat."""
    verifier = SupportVerifier()
    body = _CONTEXT.split("] ", 1)[1]
    for span in usable_spans(_CONTEXT, gold="91.4 percent"):
        assert verifier._containment_pass(span, body) is True


def test_spans_respect_the_containment_floor():
    """ "Ok." is below the floor, so it would be sent to the LLM, not short-circuited."""
    assert "Ok." not in usable_spans(_CONTEXT, gold="91.4 percent", min_chars=5)


def test_spans_drop_the_docid_header():
    assert all(not s.startswith("[docid=") for s in usable_spans(_CONTEXT, gold=""))


# --- cache contamination --------------------------------------------------


def test_poison_lands_on_victims_and_passes_containment(tmp_path):
    p1 = tmp_path / "P1-LC"
    for i in range(10):
        _write_trajectory(p1, str(i), f"alpha question {i}", f"answer {i}", "91.4 percent")

    agent = _make_agent()
    trajs = sorted(p1.glob("trajectory_q*.json"))
    corrupted = contaminate_answer_cache_from_p1(agent, trajs, _embedder, seed=42, fraction=0.20)

    assert len(corrupted) == 2  # floor(0.20 * 10)
    verifier = SupportVerifier()
    body = _CONTEXT.split("] ", 1)[1]
    for i in corrupted:
        poisoned = agent._cache[i][1]
        assert poisoned not in {f"answer {j}" for j in range(10)}
        assert verifier._containment_pass(poisoned, body) is True
        assert "91.4 percent" not in poisoned


def test_untouched_entries_keep_their_original_answer(tmp_path):
    p1 = tmp_path / "P1-LC"
    for i in range(10):
        _write_trajectory(p1, str(i), f"alpha question {i}", f"answer {i}", "91.4 percent")

    agent = _make_agent()
    trajs = sorted(p1.glob("trajectory_q*.json"))
    corrupted = set(
        contaminate_answer_cache_from_p1(agent, trajs, _embedder, seed=42, fraction=0.20)
    )

    clean = [i for i in range(10) if i not in corrupted]
    originals = {t.stem: json.loads(t.read_text())["pred"] for t in trajs}
    for i in clean:
        assert agent._cache[i][1] in originals.values()


def test_selection_is_deterministic_for_a_seed(tmp_path):
    p1 = tmp_path / "P1-LC"
    for i in range(10):
        _write_trajectory(p1, str(i), f"alpha question {i}", f"answer {i}", "91.4 percent")
    trajs = sorted(p1.glob("trajectory_q*.json"))

    runs = []
    for _ in range(2):
        agent = _make_agent()
        idx = contaminate_answer_cache_from_p1(agent, trajs, _embedder, seed=42, fraction=0.20)
        runs.append((idx, [agent._cache[i][1] for i in idx]))
    assert runs[0] == runs[1]


def test_entries_without_usable_spans_are_reported_as_not_corrupted(tmp_path):
    """Attempted coverage must not be mistaken for achieved coverage."""
    p1 = tmp_path / "P1-LC"
    for i in range(10):
        _write_trajectory(p1, str(i), f"alpha question {i}", f"answer {i}", "91.4 percent")
    # Strip the context from every trajectory: no document, no span.
    for t in p1.glob("trajectory_q*.json"):
        data = json.loads(t.read_text())
        data["trajectory"] = {}
        t.write_text(json.dumps(data), encoding="utf-8")

    agent = _make_agent()
    corrupted = contaminate_answer_cache_from_p1(
        agent, sorted(p1.glob("trajectory_q*.json")), _embedder, seed=42, fraction=0.20
    )
    assert corrupted == []


def test_misalignment_between_cache_and_trajectories_is_an_error(tmp_path):
    """The cache index identifies a trajectory; a silent drift would poison the
    wrong document while still looking like a successful attack."""
    p1 = tmp_path / "P1-LC"
    for i in range(4):
        _write_trajectory(p1, str(i), f"alpha question {i}", f"answer {i}", "91.4 percent")

    agent = _make_agent()
    agent._cache.append((_embedder("stray"), "pre-existing entry"))

    with pytest.raises(ValueError, match="misalignment"):
        contaminate_answer_cache_from_p1(
            agent, sorted(p1.glob("trajectory_q*.json")), _embedder, seed=42, fraction=0.20
        )


# --- document source for backbones whose trajectory lacks the document -----
#
# BCP's ReAct trajectory records tool calls, not the page; QASPER-RAG's
# ``context`` concatenates five sections of which the gate reads one. Both
# route through ``hint_document_contexts``: the poison must come from the
# document the entry's own location hint resolves to, because that is the
# document ``CompositeGate`` hands to G3.

_GOLD_DOC = (
    "Alpha Labs was founded in 2003 by a team of four engineers. "
    "The company's first product shipped two years later. "
    "It employs about three hundred people today."
)
_OTHER_DOC = (
    "Beta Corp manufactures industrial pumps in Ohio. "
    "Its headquarters moved twice during the nineties. "
    "Revenue is not disclosed."
)


def _hint(
    scope_id: str,
    content: str,
    question: str,
    *,
    polarity: str = "positive",
    success: int = 1,
    failure: int = 0,
    hint_type: str = "location",
) -> HintRecord:
    now = time.time()
    return HintRecord(
        hint_id=str(uuid.uuid4()),
        hint_type=hint_type,
        polarity=polarity,
        content=content,
        content_meta={},
        query_ctx=question,
        query_ctx_embedding=_embedder(question).tobytes(),
        trajectory_step=1,
        created_at=now,
        last_validated_at=now,
        success_count=success,
        failure_count=failure,
        retrieval_count=0,
        scope_id=scope_id,
    )


def _write_bcp_trajectory(dirpath: Path, qid: str, question: str) -> None:
    """April-cohort shape: no ``trajectory`` key at all."""
    dirpath.mkdir(parents=True, exist_ok=True)
    (dirpath / f"trajectory_q{qid}.json").write_text(
        json.dumps(
            {
                "query_id": qid,
                "question": question,
                "pred": f"answer {qid}",
                "gold": "three hundred",
                "judgment_correct": True,
            }
        ),
        encoding="utf-8",
    )


def _bcp_fixture(tmp_path: Path, n: int = 10):
    p1 = tmp_path / "P1"
    store = HintStore(db_path=str(tmp_path / "memory.db"))
    queries = []
    for i in range(n):
        qid, question = str(i), f"alpha question {i}"
        _write_bcp_trajectory(p1, qid, question)
        queries.append(
            {
                "query_id": qid,
                "query": question,
                "answer": "three hundred",
                "gold_docs": [{"docid": f"{1000 + i}", "text": _GOLD_DOC}],
                "evidence_docs": [],
                "negative_docs": [{"docid": f"{2000 + i}", "text": _OTHER_DOC}],
            }
        )
        store.add(_hint(f"bcp_query_{qid}", f"{1000 + i}", question))
    return p1, store, queries


def test_rank_location_hints_mirrors_store_search(tmp_path):
    """The read-only ranking must agree with the store's own search, which
    is what the agents call at serve time; a drift here would poison the
    document the gate does not read."""
    store = HintStore(db_path=str(tmp_path / "memory.db"))
    q = "alpha question"
    for content, polarity, success, failure in [
        ("11", "positive", 0, 0),
        ("12", "positive", 5, 0),
        ("13", "negative", 3, 3),
        ("14", "positive", 1, 4),
    ]:
        store.add(_hint("s", content, q, polarity=polarity, success=success, failure=failure))
    store.add(_hint("other", "99", q))  # out of scope
    store.add(_hint("s", "search_documents", q, hint_type="strategy"))  # not a location hint

    ours = [h.content for h in rank_location_hints(store, _embedder(q), "s", k=3)]
    theirs = [
        h.content
        for h in store.search(query_embedding=_embedder(q), k=9, scope_id="s")
        if h.hint_type == "location"
    ][:3]
    assert ours == theirs
    assert len(ours) == 3 and "99" not in ours


def test_bcp_shaped_records_are_poisoned_from_the_hint_document(tmp_path):
    p1, store, queries = _bcp_fixture(tmp_path)
    contexts_of = hint_document_contexts(
        store, _embedder, queries, lambda qr: f"bcp_query_{qr['query_id']}"
    )

    agent = _make_agent()
    trajs = sorted(p1.glob("trajectory_q*.json"))
    corrupted = contaminate_answer_cache_from_p1(
        agent, trajs, _embedder, seed=42, fraction=0.20, contexts_of=contexts_of
    )

    assert len(corrupted) == 2  # every victim reached its document
    verifier = SupportVerifier()
    for i in corrupted:
        poisoned = agent._cache[i][1]
        assert verifier._containment_pass(poisoned, _GOLD_DOC) is True
        assert verifier._containment_pass(poisoned, _OTHER_DOC) is False
        assert "three hundred" not in poisoned


def test_bcp_shaped_records_poison_nothing_without_the_hook(tmp_path):
    """The failure mode the hook exists for: the default source finds no
    ``context`` on a ReAct trajectory and silently poisons nothing."""
    p1, _store, _queries = _bcp_fixture(tmp_path)
    agent = _make_agent()
    corrupted = contaminate_answer_cache_from_p1(
        agent, sorted(p1.glob("trajectory_q*.json")), _embedder, seed=42, fraction=0.20
    )
    assert corrupted == []


def test_rag_shaped_records_draw_from_the_hint_section_not_the_concatenation(tmp_path):
    p1 = tmp_path / "P1-RAG"
    store = HintStore(db_path=str(tmp_path / "memory.db"))
    sections = {"paper_7_sec_3": _GOLD_DOC, "paper_7_sec_0": _OTHER_DOC, "paper_7_sec_5": _CONTEXT}
    context = "\n\n".join(f"[docid={d}] {t}" for d, t in sections.items())
    queries = []
    for i in range(10):
        qid, question = f"7_q{i}", f"alpha question {i}"
        p1.mkdir(parents=True, exist_ok=True)
        (p1 / f"trajectory_q{qid}.json").write_text(
            json.dumps(
                {
                    "query_id": qid,
                    "question": question,
                    "pred": f"answer {i}",
                    "gold": "three hundred",
                    "judgment_correct": True,
                    "trajectory": {"retrieved_docids": list(sections), "context": context},
                }
            ),
            encoding="utf-8",
        )
        queries.append(
            {
                "query_id": qid,
                "paper_id": "7",
                "answer": "three hundred",
                "docs": [{"docid": d, "text": t} for d, t in sections.items()],
            }
        )
        # The hint the gate will try first points at sec_3; the others rank below it.
        store.add(_hint("qasper_paper_7", "paper_7_sec_3", question, success=5))
        store.add(_hint("qasper_paper_7", "paper_7_sec_0", question, success=0))

    contexts_of = hint_document_contexts(
        store, _embedder, queries, lambda qr: f"qasper_paper_{qr['paper_id']}"
    )
    agent = _make_agent()
    corrupted = contaminate_answer_cache_from_p1(
        agent,
        sorted(p1.glob("trajectory_q*.json")),
        _embedder,
        seed=42,
        fraction=0.20,
        contexts_of=contexts_of,
    )
    assert len(corrupted) == 2
    verifier = SupportVerifier()
    for i in corrupted:
        poisoned = agent._cache[i][1]
        assert verifier._containment_pass(poisoned, _GOLD_DOC) is True
        assert verifier._containment_pass(poisoned, _OTHER_DOC) is False


def test_hint_document_contexts_follows_gate_order_and_skips_unresolvable(tmp_path):
    store = HintStore(db_path=str(tmp_path / "memory.db"))
    q = "alpha question 0"
    store.add(_hint("bcp_query_0", "2000", q, success=0))  # negative doc, low confidence
    store.add(_hint("bcp_query_0", "1000", q, success=5))  # gold doc, high confidence
    store.add(_hint("bcp_query_0", "31337", q, success=9))  # not in the pool
    queries = [
        {
            "query_id": "0",
            "gold_docs": [{"docid": "1000", "text": _GOLD_DOC}],
            "evidence_docs": [],
            "negative_docs": [{"docid": "2000", "text": _OTHER_DOC}],
        }
    ]
    contexts_of = hint_document_contexts(
        store, _embedder, queries, lambda qr: f"bcp_query_{qr['query_id']}"
    )
    assert contexts_of({"query_id": "0", "question": q}) == [_GOLD_DOC, _OTHER_DOC]
    assert contexts_of({"query_id": "missing", "question": q}) == []


def test_factory_helper_refuses_a_run_that_poisoned_nothing(tmp_path):
    """A gate-aware phase with zero poisoned entries is a clean run under a
    contaminated name. It must not start."""
    from hgc.factories import _gate_aware_answer_cache

    p1, _store, _queries = _bcp_fixture(tmp_path)

    class _Emb:
        def embed(self, text: str):
            return _embedder(text)

    with pytest.raises(RuntimeError, match="poisoned 0"):
        _gate_aware_answer_cache("A-Hybrid", _Emb(), p1, lambda record: [], 42, 0.20)
    cache = _gate_aware_answer_cache("A-Hybrid", _Emb(), p1, lambda record: [_GOLD_DOC], 42, 0.20)
    assert len(cache) == 10
