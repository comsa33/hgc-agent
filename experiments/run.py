#!/usr/bin/env python
"""Unified experiment runner for HGC.

Replaces run_smoke_n10.py / run_smoke_gpt4o.py / run_smoke_hotpot.py.

Examples
--------
# Run a quick BCP smoke
uv run python experiments/run_experiment.py --n=10 --dataset=bcp --model=gpt-4.1

# Continue the same experiment at higher n (incremental accumulation)
uv run python experiments/run_experiment.py --n=20 --dataset=bcp --model=gpt-4.1

# HotpotQA cross-model
uv run python experiments/run_experiment.py --n=10 --dataset=hotpot --model=gpt-4o

# Custom phase subset
uv run python experiments/run_experiment.py --n=10 --phases=P0,P1,P2,P4

# Dry run (no API calls)
uv run python experiments/run_experiment.py --n=10 --dry-run
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

_REPO = Path(__file__).parent.parent
sys.path.insert(0, str(_REPO / "src"))

from hgc.experiment import (  # noqa: E402
    ExperimentConfig,
    capture_run_env,
    detect_git_commit,
    experiment_id,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Phase lists
# ---------------------------------------------------------------------------

ALL_BCP_PHASES = [
    "P0",
    "P1",
    "P3-AC",
    "P3-Hybrid",
    "P4-AC",
    "P4-Hybrid",
    "E4-Hybrid",
    "T4-Hybrid",
    # Gate-aware adversary (paper's E4 experiment). Not "E4-Hybrid": that name
    # is entity-swap. See ALL_QASPER_ORACLE_PHASES for the LC counterpart.
    "A-Hybrid",
]

ALL_HOTPOT_PHASES = ["P0", "P1", "P2", "P3-M0", "P3-RAG", "P4"]

ALL_QASPER_RAG_PHASES = [
    "P1-RAG",
    "V-RAG",
    "AC-RAG",
    "HGC-RAG",
    "C-AC-RAG",
    "C-HGC-RAG",
    "A-HGC-RAG",  # gate-aware adversary; mirrors A-HGC-LC
]
# QASPER-Oracle: mirrors FinanceBench-LC. Uses annotator-marked evidence
# sentences (skipping retrieval), validates contamination resistance only.
ALL_QASPER_ORACLE_PHASES = [
    "P1-LC",
    "V-LC",
    "AC-LC",
    "HGC-LC",
    "C-AC-LC",
    "C-HGC-LC",
    # Gate-aware contamination. Not part of the default sweep — it answers a
    # different question from C-* and is opted into explicitly via --phases.
    "A-HGC-LC",
]
# FinanceBench uses Long-Context (LC) backbone: no retrieval, all evidence
# pages are injected directly into the prompt. Matches Patronus's Oracle
# setting where full relevant context is provided.
ALL_FINANCEBENCH_LC_PHASES = [
    "P1-LC",
    "V-LC",
    "AC-LC",
    "HGC-LC",
    "C-AC-LC",
    "C-HGC-LC",
    # See the note on ALL_QASPER_ORACLE_PHASES.
    "A-HGC-LC",
]

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Unified HGC experiment runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--n",
        type=int,
        default=10,
        help="Number of queries to include (grows incrementally, default: 10)",
    )
    p.add_argument(
        "--dataset",
        choices=["bcp", "hotpot", "qasper-rag", "qasper-oracle", "financebench-lc"],
        default="bcp",
        help="Dataset to use (default: bcp)",
    )
    p.add_argument(
        "--model",
        default="gpt-4.1",
        help="Main LM Azure deployment name (default: gpt-4.1)",
    )
    p.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    p.add_argument(
        "--phases",
        default="all",
        help="Comma-separated phase list or 'all' (default: all)",
    )
    p.add_argument(
        "--out-dir",
        default=None,
        dest="out_dir",
        help="Override output directory (default: results/{experiment_id}/)",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Print plan without running any API calls",
    )
    p.add_argument(
        "--gate-variant",
        choices=["full", "g1g2", "g1g3"],
        default="full",
        help="Gate ablation variant for Hybrid phases: full=G1+G2+G3, "
        "g1g2=G3 disabled, g1g3=G2 disabled (default: full)",
    )
    p.add_argument(
        "--contamination-fraction",
        type=float,
        default=0.20,
        dest="contamination_fraction",
        help="Cache/HintStore corruption fraction for P4/E4/T4 phases (default: 0.20)",
    )
    p.add_argument(
        "--cache-sim-threshold",
        type=float,
        default=None,
        dest="cache_sim_threshold",
        help="Answer-cache similarity threshold tau (default: 0.85). Serve-time "
        "only, so a warm-up cache can be reused across values.",
    )
    p.add_argument(
        "--top-k-hints",
        type=int,
        default=None,
        dest="top_k_hints",
        help="Location hints retrieved per query (default: 3)",
    )
    p.add_argument(
        "--verifier-max-doc-chars",
        type=int,
        default=None,
        dest="verifier_max_doc_chars",
        help="G3 prompt document truncation in characters (default: 2500)",
    )
    p.add_argument(
        "--verifier-containment-min-chars",
        type=int,
        default=None,
        dest="verifier_containment_min_chars",
        help="Minimum answer length for the G3 containment fast path (default: 5)",
    )
    p.add_argument(
        "--verifier-predicate",
        choices=["support", "answerhood", "two_stage", "two_stage_doc"],
        default=None,
        dest="verifier_predicate",
        help="Question the G3 LM answers: grounded-in-document (support, the "
        "default) or responsive-to-the-question (answerhood)",
    )
    return p.parse_args()


def _apply_ablation_knobs(args: argparse.Namespace) -> None:
    """Export ablation knobs to env vars that factories read at build time."""
    os.environ["HGC_GATE_VARIANT"] = args.gate_variant
    os.environ["HGC_CONTAMINATION_FRACTION"] = str(args.contamination_fraction)
    # Sensitivity knobs: only exported when given, so an unset flag leaves the
    # released default in place rather than restating it here (one source of
    # truth, in factories.py).
    for flag, env in (
        ("cache_sim_threshold", "HGC_CACHE_SIM_THRESHOLD"),
        ("top_k_hints", "HGC_TOP_K_HINTS"),
        ("verifier_max_doc_chars", "HGC_VERIFIER_MAX_DOC_CHARS"),
        ("verifier_containment_min_chars", "HGC_VERIFIER_CONTAINMENT_MIN_CHARS"),
        ("verifier_predicate", "HGC_VERIFIER_PREDICATE"),
    ):
        value = getattr(args, flag, None)
        if value is not None:
            os.environ[env] = str(value)


# ---------------------------------------------------------------------------
# Deployment resolution
# ---------------------------------------------------------------------------


def resolve_deployments(model: str) -> tuple[str, str]:
    """Return (sub_deployment, embedding_deployment) for *model*.

    Defaults to gpt-4.1-mini / text-embedding-3-small for unknown models.
    For ollama/* main model we keep the Azure-hosted sub/embedding models
    because those auxiliary tasks (HintExtractor, embeddings) still need a
    reliable commercial LM and our Ollama server may not have them.
    """
    if model.startswith("ollama/"):
        return ("gpt-4.1-mini", "text-embedding-3-small")
    if model.startswith("gpt-4o"):
        return ("gpt-4o-mini", "text-embedding-3-small")
    if model.startswith("gpt-4.1"):
        return ("gpt-4.1-mini", "text-embedding-3-small")
    return ("gpt-4.1-mini", "text-embedding-3-small")


# ---------------------------------------------------------------------------
# Phase set resolver
# ---------------------------------------------------------------------------


def _own_store(out_dir: Path, phase: str, p1_db: Path) -> Path:
    """Give *phase* its own copy of the P1 hint store.

    The ReAct fallback writes hints into whatever store the agent holds. A
    phase that opens ``P1/memory.db`` directly therefore grows it while it
    runs, and every phase that copies P1 later in the same invocation
    inherits those hints: a two-stage P4-Hybrid run after P3-Hybrid started
    from 1,458 location hints instead of 1,257 and lost hits to the no-hint
    path for that reason alone. Copy once, before the phase, and never touch
    P1 again.
    """
    db = out_dir / phase / "memory.db"
    db.parent.mkdir(parents=True, exist_ok=True)
    if p1_db.exists() and not db.exists():
        shutil.copy2(str(p1_db), str(db))
    return db


def phase_set_for(dataset: str, phases_arg: str) -> list[str]:
    """Return the ordered phase list for *dataset* filtered by *phases_arg*.

    Raises ValueError if any phase name is unknown for the dataset.
    Canonical ordering is always preserved regardless of input order.
    """
    if dataset == "bcp":
        full = ALL_BCP_PHASES
    elif dataset == "qasper-rag":
        full = ALL_QASPER_RAG_PHASES
    elif dataset == "qasper-oracle":
        full = ALL_QASPER_ORACLE_PHASES
    elif dataset == "financebench-lc":
        full = ALL_FINANCEBENCH_LC_PHASES
    else:
        full = ALL_HOTPOT_PHASES
    if phases_arg == "all":
        return list(full)
    requested = [p.strip() for p in phases_arg.split(",")]
    unknown = [p for p in requested if p not in full]
    if unknown:
        raise ValueError(f"Unknown phases for dataset={dataset!r}: {unknown}")
    # Preserve canonical ordering
    return [p for p in full if p in requested]


# ---------------------------------------------------------------------------
# Config load/create with incremental support
# ---------------------------------------------------------------------------


def load_or_create_config(out_dir: Path, args: argparse.Namespace) -> tuple[ExperimentConfig, bool]:
    """Return (config, nothing_to_do).

    nothing_to_do is True when the existing config already covers >= n queries.
    In that case callers should skip agent runs and only regenerate the summary.
    """
    sub, emb = resolve_deployments(args.model)
    new_cfg = ExperimentConfig(
        experiment_id=experiment_id(args.dataset, args.model, args.seed),
        dataset=args.dataset,
        model_main=args.model,
        model_sub=sub,
        model_embedding=emb,
        seed=args.seed,
        n_queries=args.n,
        notes="",
        git_commit=detect_git_commit(),
        # Captured here, after _apply_ablation_knobs, so that a second
        # invocation into an existing out-dir is checked against the knobs it
        # actually runs under. Left at the default it is empty at this point
        # (save() fills it later) and every re-invocation fails the check.
        run_env=capture_run_env(),
    )

    config_path = out_dir / "config.json"
    if config_path.exists():
        existing = ExperimentConfig.load(out_dir)
        existing.validate_compatible(new_cfg)  # raises if mismatch

        # "Nothing to do" only applies when the caller wants the full phase set AND
        # n is already covered. When --phases selects a subset, always proceed so
        # newly added phases (e.g. P4-AC) can run against an existing store.
        explicit_subset = args.phases not in ("all", "")
        if args.n <= existing.n_queries and not explicit_subset:
            print(
                f"Config: existing n_queries={existing.n_queries} already >= requested n={args.n}."
            )
            print("Nothing to do; trajectories already cover requested n.")
            return existing, True

        # Grow (or stay at) n_queries; refresh git_commit
        existing.n_queries = max(existing.n_queries, args.n)
        existing.git_commit = detect_git_commit() or existing.git_commit
        return existing, False

    return new_cfg, False


# ---------------------------------------------------------------------------
# LM configuration
# ---------------------------------------------------------------------------


def _require_env(*names: str) -> None:
    """Fail with instructions rather than a KeyError traceback.

    Someone reproducing from the supplementary meets this before anything else,
    and a raw KeyError on AZURE_OPENAI_API_KEY reads like a bug in the code
    rather than a missing credential on their side.
    """
    missing = [n for n in names if not os.environ.get(n)]
    if missing:
        raise SystemExit(
            "Missing required environment variable(s): "
            + ", ".join(missing)
            + "\n\nExperiments call a hosted model, so credentials must be supplied.\n"
            "Copy .env.example to .env and fill it in, or export the variables:\n"
            "  AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_VERSION\n\n"
            "Analysis of the released trajectories needs no credentials -- see\n"
            "analysis/summary_tables.py and experiments/compute_judge_agreement.py."
        )


def configure_lm(deployment: str):
    import dspy

    if deployment.startswith("ollama/"):
        # Ollama (OpenAI-compatible) — DSPy routes via LiteLLM's ollama_chat/.
        model_name = deployment[len("ollama/") :]
        api_base = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
        lm = dspy.LM(
            model=f"ollama_chat/{model_name}",
            api_base=api_base,
            temperature=0,
            max_tokens=4096,
            cache=False,
        )
    else:
        # Azure OpenAI (default)
        # Reasoning-model family (gpt-5.x) requires temperature=1.0 and
        # max_tokens>=16000; anything lower is rejected by the Azure gateway.
        is_reasoning = deployment.startswith("gpt-5")
        _require_env("AZURE_OPENAI_API_KEY", "AZURE_OPENAI_ENDPOINT", "AZURE_OPENAI_API_VERSION")
        lm = dspy.LM(
            model=f"azure/{deployment}",
            api_key=os.environ["AZURE_OPENAI_API_KEY"],
            api_base=os.environ["AZURE_OPENAI_ENDPOINT"].rstrip("/"),
            api_version=os.environ["AZURE_OPENAI_API_VERSION"],
            temperature=1.0 if is_reasoning else 0,
            max_tokens=16000 if is_reasoning else 4096,
            cache=False,
        )
    dspy.configure(lm=lm)
    return lm


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------


def load_dataset_bcp(n: int, seed: int) -> dict:
    """Load or build the BCP dataset sliced to *n* queries."""
    from hgc.datasets.bcp import BCPDataset

    ds = BCPDataset()
    full = ds.load_or_build(n=n, seed=seed, paraphrase=True)
    scope_of = lambda qr: f"bcp_query_{qr['query_id']}"  # noqa: E731
    return {
        "queries": full["queries"][:n],
        "paraphrased": full.get("paraphrased", [])[:n],
        "scope_of": scope_of,
    }


def load_dataset_hotpot(n: int, seed: int) -> dict:
    """Load or build the HotpotQA clustered dataset sliced to *n* queries."""

    hotpot_cache = _REPO / "data" / "hotpot"
    clustered_path = hotpot_cache / "hotpot_clustered.json"
    if not clustered_path.exists():
        raise RuntimeError(
            f"HotpotQA clustered dataset not found at {clustered_path}. "
            "Run: uv run python experiments/build_hotpot_dataset.py"
        )

    ds = HotpotQADataset(cache_dir=str(hotpot_cache))
    result = ds.select_clustered(n_clusters=10, per_cluster=10, seed=seed)
    clusters = result["clusters"]

    flat: list[dict] = []
    for c_idx, cluster in enumerate(clusters):
        for rec in cluster:
            rec_copy = dict(rec)
            rec_copy["cluster_idx"] = c_idx
            rec_copy["query"] = rec_copy.get("question", "")
            rec_copy.setdefault("answer", "")
            rec_copy["gold_docs"] = rec_copy.get("docs", [])
            flat.append(rec_copy)

    scope_of = lambda qr: f"hotpot_cluster_{qr.get('cluster_idx', 0)}"  # noqa: E731
    return {
        "queries": flat[:n],
        "paraphrased": [],
        "scope_of": scope_of,
    }


def load_dataset_qasper(n: int, seed: int, oracle: bool = False) -> dict:
    """Load or build the QASPER dataset sliced to *n* queries.

    When ``oracle=True`` each record's ``docs`` holds only the annotator-
    marked evidence (not the full paper), and paraphrase is enabled so the
    AC/HGC/C-* phases exercise semantic-similarity cache behaviour.
    """
    from hgc.datasets.qasper import QASPERDataset

    ds = QASPERDataset(cache_dir=str(_REPO / "data" / "qasper"))
    full = ds.load_or_build(n=n, seed=seed, paraphrase=oracle, oracle=oracle)
    scope_of = lambda qr: f"qasper_paper_{qr.get('paper_id', qr['query_id'])}"  # noqa: E731
    return {
        "queries": full["queries"][:n],
        "paraphrased": full.get("paraphrased", [])[:n],
        "scope_of": scope_of,
    }


def load_dataset_financebench(n: int, seed: int) -> dict:
    """Load or build the FinanceBench dataset sliced to *n* queries."""
    from hgc.datasets.financebench import FinanceBenchDataset

    ds = FinanceBenchDataset(cache_dir=str(_REPO / "data" / "financebench"))
    # Paraphrase enabled: the AC/HGC/C-* phases run on paraphrased queries so
    # the answer cache is exercised as a semantic-similarity store, not an
    # exact-match lookup. Mirrors the BCP paraphrase regime.
    full = ds.load_or_build(n=n, seed=seed, paraphrase=True)

    def scope_of(qr: dict) -> str:
        doc = qr.get("doc_name", qr["query_id"])
        period = qr.get("doc_period", "")
        return f"financebench_doc_{doc}_{period}"

    return {
        "queries": full["queries"][:n],
        "paraphrased": full.get("paraphrased", [])[:n],
        "scope_of": scope_of,
    }


def load_dataset(dataset: str, n: int, seed: int) -> dict:
    if dataset == "bcp":
        return load_dataset_bcp(n, seed)
    elif dataset == "hotpot":
        return load_dataset_hotpot(n, seed)
    elif dataset in ("qasper", "qasper-rag"):
        return load_dataset_qasper(n, seed, oracle=False)
    elif dataset == "qasper-oracle":
        return load_dataset_qasper(n, seed, oracle=True)
    elif dataset in ("financebench", "financebench-lc"):
        return load_dataset_financebench(n, seed)
    else:
        raise ValueError(f"Unknown dataset: {dataset!r}")


# ---------------------------------------------------------------------------
# Phase factory orchestrators
# ---------------------------------------------------------------------------


def _run_bcp_phases(
    phases: list[str],
    dataset: dict,
    out_dir: Path,
    cfg: ExperimentConfig,
    all_summaries: list[dict],
) -> None:
    from hgc.emb_cache import DocEmbeddingCache
    from hgc.embeddings import Embedder
    from hgc.extraction import HintExtractor
    from hgc.judge import LLMJudge
    from hgc.memory import HintStore
    from hgc.runner import PhaseRunner
    from hgc.smoke_common import (
        build_ac_factory,
        build_adaptive_hybrid_factory_bcp,
        build_e4_hybrid_factory,
        build_hgc_core_factory_bcp,
        build_hybrid_factory_bcp,
        build_m0_factory,
        build_mem0_config,
        build_p4_ac_factory,
        build_p4_hybrid_factory,
        build_rag_factory_bcp,
        build_t4_hybrid_factory,
        make_tools_bcp,
        phase_summary,
    )

    judge = LLMJudge()
    embedder = Embedder()
    extractor = HintExtractor()

    # Contamination fraction for P4/E4/T4 phases — honors HGC_CONTAMINATION_FRACTION
    # env var (set by --contamination-fraction CLI) so A2 rate sweeps work without
    # modifying factory signatures.
    _frac = float(os.environ.get("HGC_CONTAMINATION_FRACTION", "0.20"))

    _bcp_data_dir = _REPO / "data" / "bcp"
    _bcp_data_dir.mkdir(parents=True, exist_ok=True)
    doc_emb_cache = DocEmbeddingCache(path=_bcp_data_dir / "doc_embeddings.npz")

    out_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir = out_dir / "_tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    p1_dir = out_dir / "P1"
    p1_db = p1_dir / "memory.db"
    p1_dir.mkdir(parents=True, exist_ok=True)

    scope_of = dataset["scope_of"]

    runner = PhaseRunner(
        dataset=dataset,
        judge=judge,
        out_dir=out_dir,
        tool_factory=make_tools_bcp,
    )

    def run_phase(phase_name: str, factory, on_paraphrased: bool = False) -> list:
        print(f"\n--- Running {phase_name} (paraphrased={on_paraphrased}) ---", flush=True)
        results = runner.run_phase(phase_name, factory, on_paraphrased=on_paraphrased)
        runner.write_summary(phase_name, results)
        s = phase_summary(phase_name, results)
        all_summaries.append(s)
        print(
            f"    {phase_name}: {s['n_correct_judge']}/{s['n_queries']} correct (judge), "
            f"{s['total_tokens']:,} tokens, ${s['estimated_cost_usd']:.4f}"
        )
        return results

    phases_set = set(phases)
    p1_results = []

    # --- P0: Vanilla original (empty store) ---
    if "P0" in phases_set:
        p0_empty_db = tmp_dir / "p0_empty.db"
        p0_store = HintStore(db_path=str(p0_empty_db))
        run_phase(
            "P0",
            build_hgc_core_factory_bcp(embedder, p0_store, extractor, scope_of),
            on_paraphrased=False,
        )

    # --- P0': Vanilla paraphrased (separate empty store) ---
    if "P0'" in phases_set:
        p0p_empty_db = tmp_dir / "p0p_empty.db"
        p0p_store = HintStore(db_path=str(p0p_empty_db))
        run_phase(
            "P0'",
            build_hgc_core_factory_bcp(embedder, p0p_store, extractor, scope_of),
            on_paraphrased=True,
        )

    # --- P0-RAG: NaiveRAG original ---
    if "P0-RAG" in phases_set:
        run_phase("P0-RAG", build_rag_factory_bcp(embedder, doc_emb_cache), on_paraphrased=False)

    # --- P1: HGC core, cold, sequential, shared store ---
    if "P1" in phases_set:
        p1_store = HintStore(db_path=str(p1_db))
        p1_results = run_phase(
            "P1",
            build_hgc_core_factory_bcp(embedder, p1_store, extractor, scope_of),
            on_paraphrased=False,
        )
        p1_store.close()

    # --- P2: HGC core, warm (copy P1 store) ---
    if "P2" in phases_set:
        p2_db = out_dir / "P2" / "memory.db"
        p2_db.parent.mkdir(parents=True, exist_ok=True)
        if p1_db.exists() and not p2_db.exists():
            shutil.copy2(str(p1_db), str(p2_db))
        p2_store = HintStore(db_path=str(p2_db))
        run_phase(
            "P2",
            build_hgc_core_factory_bcp(embedder, p2_store, extractor, scope_of),
            on_paraphrased=False,
        )
        p2_store.close()

    # --- P3: HGC core, paraphrased (copy P1 store) ---
    if "P3" in phases_set:
        p3_db = out_dir / "P3" / "memory.db"
        p3_db.parent.mkdir(parents=True, exist_ok=True)
        if p1_db.exists() and not p3_db.exists():
            shutil.copy2(str(p1_db), str(p3_db))
        p3_store = HintStore(db_path=str(p3_db))
        run_phase(
            "P3",
            build_hgc_core_factory_bcp(embedder, p3_store, extractor, scope_of),
            on_paraphrased=True,
        )
        p3_store.close()

    # --- P3-AC: AnswerCache seeded from P1, paraphrased ---
    if "P3-AC" in phases_set:
        run_phase("P3-AC", build_ac_factory(embedder, p1_dir), on_paraphrased=True)

    # --- P3-M0: Mem0 seeded from P1, paraphrased ---
    if "P3-M0" in phases_set:
        try:
            mem0_cfg = build_mem0_config(
                out_dir,
                collection_name="mem0_p3_m0_smoke",
                sub_deployment=cfg.model_sub,
            )
            run_phase(
                "P3-M0",
                build_m0_factory(p1_dir, mem0_config=mem0_cfg, user_id="p3_m0_smoke"),
                on_paraphrased=True,
            )
        except Exception as exc:
            logger.warning("P3-M0 failed (mem0ai may not be installed): %s", exc)
            all_summaries.append(
                {
                    "phase": "P3-M0",
                    "n_queries": 0,
                    "n_correct_judge": 0,
                    "n_correct_containment": 0,
                    "avg_tokens": 0,
                    "avg_time_s": 0.0,
                    "avg_iters": 0.0,
                    "total_tokens": 0,
                    "estimated_cost_usd": 0.0,
                    "error": str(exc),
                }
            )

    # --- P3-Hybrid: AC fast path + HGC hint re-verification, paraphrased ---
    if "P3-Hybrid" in phases_set:
        run_phase(
            "P3-Hybrid",
            build_hybrid_factory_bcp(
                embedder,
                _own_store(out_dir, "P3-Hybrid", p1_db),
                p1_dir,
                extractor,
                scope_of=scope_of,
            ),
            on_paraphrased=True,
        )

    # --- P4-AC: AnswerCache seeded from P1, contaminated, paraphrased ---
    if "P4-AC" in phases_set:
        run_phase(
            "P4-AC",
            build_p4_ac_factory(embedder, p1_dir, seed=42, fraction=_frac),
            on_paraphrased=True,
        )

    # --- P4-Hybrid: contaminated AC + contaminated HintStore with Hybrid re-verification ---
    if "P4-Hybrid" in phases_set:
        p4_hybrid_db = out_dir / "P4-Hybrid" / "memory.db"
        p4_hybrid_db.parent.mkdir(parents=True, exist_ok=True)
        run_phase(
            "P4-Hybrid",
            build_p4_hybrid_factory(
                embedder,
                p1_db,
                p1_dir,
                extractor,
                p4_hybrid_db,
                scope_of=scope_of,
                seed=42,
                fraction=_frac,
            ),
            on_paraphrased=True,
        )

    # --- A-Hybrid: gate-aware poison on AC, HintStore copied from P1 untouched ---
    if "A-Hybrid" in phases_set:
        a_hybrid_db = out_dir / "A-Hybrid" / "memory.db"
        a_hybrid_db.parent.mkdir(parents=True, exist_ok=True)
        run_phase(
            "A-Hybrid",
            build_adaptive_hybrid_factory_bcp(
                embedder,
                p1_db,
                p1_dir,
                extractor,
                a_hybrid_db,
                dataset["queries"],
                scope_of=scope_of,
                seed=42,
                fraction=_frac,
            ),
            on_paraphrased=True,
        )

    # --- E4-Hybrid: entity-swap on AC + HintStore ---
    if "E4-Hybrid" in phases_set:
        e4_hybrid_db = out_dir / "E4-Hybrid" / "memory.db"
        e4_hybrid_db.parent.mkdir(parents=True, exist_ok=True)
        run_phase(
            "E4-Hybrid",
            build_e4_hybrid_factory(
                embedder,
                p1_db,
                p1_dir,
                extractor,
                e4_hybrid_db,
                scope_of=scope_of,
                seed=42,
                fraction=0.20,
            ),
            on_paraphrased=True,
        )

    # --- T4-Hybrid: typo-mutation on AC + HintStore ---
    if "T4-Hybrid" in phases_set:
        t4_hybrid_db = out_dir / "T4-Hybrid" / "memory.db"
        t4_hybrid_db.parent.mkdir(parents=True, exist_ok=True)
        run_phase(
            "T4-Hybrid",
            build_t4_hybrid_factory(
                embedder,
                p1_db,
                p1_dir,
                extractor,
                t4_hybrid_db,
                scope_of=scope_of,
                seed=42,
                fraction=0.20,
            ),
            on_paraphrased=True,
        )

    # Print P1 n_added_hints if we ran P1
    if p1_results:
        print("\n--- P1 n_added_hints sequence ---")
        cumulative = 0
        prev_cum = 0
        monotone_ok = True
        for r in p1_results:
            print(f"  q{r.query_id}: n_added_hints={r.n_added_hints}")
            cumulative += r.n_added_hints
            if cumulative < prev_cum:
                monotone_ok = False
            prev_cum = cumulative
        if monotone_ok:
            print(f"  P1 cumulative n_added_hints is non-decreasing (total={cumulative}). OK.")
        else:
            print("  WARNING: P1 cumulative n_added_hints is NOT monotone non-decreasing!")

    # Doc embedding cache stats
    _cache_size_final = doc_emb_cache.size()
    print(f"\nDoc embedding cache: {_cache_size_final} total entries.")
    doc_emb_cache.save()


def _run_qasper_rag_phases(
    phases: list[str],
    dataset: dict,
    out_dir: Path,
    cfg: ExperimentConfig,
    all_summaries: list[dict],
) -> None:
    from hgc.embeddings import Embedder
    from hgc.extraction import HintExtractor
    from hgc.judge import LLMJudge
    from hgc.memory import HintStore
    from hgc.runner import PhaseRunner
    from hgc.smoke_common import (
        build_ac_rag_factory_qasper,
        build_adaptive_hgc_rag_factory_qasper,
        build_hgc_rag_factory_qasper,
        build_p1_rag_factory_qasper,
        build_p4_ac_rag_factory_qasper,
        build_p4_hgc_rag_factory_qasper,
        build_rag_naive_factory_qasper,
        make_tools_qasper,
        phase_summary,
    )

    judge = LLMJudge()
    embedder = Embedder()
    extractor = HintExtractor()

    out_dir.mkdir(parents=True, exist_ok=True)

    # P1-RAG is the RAG-side seed phase (mirrors ReAct's P1): runs vanilla RAG
    # and writes location hints to HintStore. Its trajectories + HintStore feed
    # AC-RAG / HGC-RAG / C-AC-RAG / C-HGC-RAG.
    p1_rag_dir = out_dir / "P1-RAG"
    p1_rag_db = p1_rag_dir / "memory.db"
    p1_rag_dir.mkdir(parents=True, exist_ok=True)

    scope_of = dataset["scope_of"]

    runner = PhaseRunner(
        dataset=dataset,
        judge=judge,
        out_dir=out_dir,
        tool_factory=make_tools_qasper,
    )

    def run_phase(phase_name: str, factory, on_paraphrased: bool = False) -> list:
        print(f"\n--- Running {phase_name} (paraphrased={on_paraphrased}) ---", flush=True)
        results = runner.run_phase(phase_name, factory, on_paraphrased=on_paraphrased)
        runner.write_summary(phase_name, results)
        s = phase_summary(phase_name, results)
        all_summaries.append(s)
        print(
            f"    {phase_name}: {s['n_correct_judge']}/{s['n_queries']} correct (judge), "
            f"{s['total_tokens']:,} tokens, ${s['estimated_cost_usd']:.4f}"
        )
        return results

    phases_set = set(phases)

    # --- P1-RAG: Seed phase — vanilla RAG + HintStore population ---
    if "P1-RAG" in phases_set:
        p1_rag_store = HintStore(db_path=str(p1_rag_db))
        run_phase(
            "P1-RAG",
            build_p1_rag_factory_qasper(embedder, p1_rag_store, scope_of=scope_of, llm_judge=judge),
            on_paraphrased=False,
        )

    # --- V-RAG: Vanilla RAG baseline (no memory, reported separately) ---
    if "V-RAG" in phases_set:
        run_phase("V-RAG", build_rag_naive_factory_qasper(embedder), on_paraphrased=False)

    # --- AC-RAG: AnswerCache seeded from P1-RAG ---
    if "AC-RAG" in phases_set:
        run_phase(
            "AC-RAG",
            build_ac_rag_factory_qasper(embedder, p1_rag_dir),
            on_paraphrased=True,
        )

    # --- HGC-RAG: HGCRAGAgent (AC + RAG fallback + gate) seeded from P1-RAG ---
    if "HGC-RAG" in phases_set:
        run_phase(
            "HGC-RAG",
            build_hgc_rag_factory_qasper(
                embedder, p1_rag_db, p1_rag_dir, extractor, scope_of=scope_of
            ),
            on_paraphrased=True,
        )

    # --- C-AC-RAG: AC contaminated via cross-swap ---
    if "C-AC-RAG" in phases_set:
        run_phase(
            "C-AC-RAG",
            build_p4_ac_rag_factory_qasper(embedder, p1_rag_dir, seed=42, fraction=0.20),
            on_paraphrased=True,
        )

    # --- C-HGC-RAG: Contaminated HGCRAGAgent ---
    if "C-HGC-RAG" in phases_set:
        c_hgc_rag_db = out_dir / "C-HGC-RAG" / "memory.db"
        c_hgc_rag_db.parent.mkdir(parents=True, exist_ok=True)
        run_phase(
            "C-HGC-RAG",
            build_p4_hgc_rag_factory_qasper(
                embedder,
                p1_rag_db,
                p1_rag_dir,
                extractor,
                c_hgc_rag_db,
                scope_of=scope_of,
                seed=42,
                fraction=0.20,
            ),
            on_paraphrased=True,
        )

    # --- A-HGC-RAG: gate-aware poison on AC, HintStore copied from P1 untouched ---
    if "A-HGC-RAG" in phases_set:
        a_hgc_rag_db = out_dir / "A-HGC-RAG" / "memory.db"
        a_hgc_rag_db.parent.mkdir(parents=True, exist_ok=True)
        run_phase(
            "A-HGC-RAG",
            build_adaptive_hgc_rag_factory_qasper(
                embedder,
                p1_rag_db,
                p1_rag_dir,
                extractor,
                a_hgc_rag_db,
                dataset["queries"],
                scope_of=scope_of,
                seed=42,
                fraction=0.20,
            ),
            on_paraphrased=True,
        )


def _run_qasper_oracle_phases(
    phases: list[str],
    dataset: dict,
    out_dir: Path,
    cfg: ExperimentConfig,
    all_summaries: list[dict],
) -> None:
    """QASPER Oracle phases — mirrors financebench-lc but with QASPER scope/tools.

    Reuses the same LongCtxBackbone-driven factories: the only dataset-specific
    bits are ``scope_of`` (provided by the dataset dict) and the ``tool_factory``
    (unused by LC agents but required by PhaseRunner).
    """
    from hgc.embeddings import Embedder
    from hgc.judge import LLMJudge
    from hgc.memory import HintStore
    from hgc.runner import PhaseRunner
    from hgc.smoke_common import (
        build_ac_lc_factory_financebench,
        build_adaptive_hgc_lc_factory_financebench,
        build_hgc_lc_factory_financebench,
        build_lc_naive_factory_financebench,
        build_p1_lc_factory_financebench,
        build_p4_ac_lc_factory_financebench,
        build_p4_hgc_lc_factory_financebench,
        make_tools_qasper,
        phase_summary,
    )

    judge = LLMJudge()
    embedder = Embedder()

    out_dir.mkdir(parents=True, exist_ok=True)

    p1_dir = out_dir / "P1-LC"
    p1_db = p1_dir / "memory.db"
    p1_dir.mkdir(parents=True, exist_ok=True)

    scope_of = dataset["scope_of"]

    runner = PhaseRunner(
        dataset=dataset,
        judge=judge,
        out_dir=out_dir,
        tool_factory=make_tools_qasper,
    )

    def run_phase(phase_name: str, factory, on_paraphrased: bool = False) -> list:
        print(f"\n--- Running {phase_name} (paraphrased={on_paraphrased}) ---", flush=True)
        results = runner.run_phase(phase_name, factory, on_paraphrased=on_paraphrased)
        runner.write_summary(phase_name, results)
        s = phase_summary(phase_name, results)
        all_summaries.append(s)
        print(
            f"    {phase_name}: {s['n_correct_judge']}/{s['n_queries']} correct (judge), "
            f"{s['total_tokens']:,} tokens, ${s['estimated_cost_usd']:.4f}"
        )
        return results

    phases_set = set(phases)
    if "P1-LC" in phases_set:
        store = HintStore(db_path=str(p1_db))
        run_phase(
            "P1-LC",
            build_p1_lc_factory_financebench(embedder, store, scope_of=scope_of, llm_judge=judge),
            on_paraphrased=False,
        )
    if "V-LC" in phases_set:
        run_phase("V-LC", build_lc_naive_factory_financebench(embedder), on_paraphrased=False)
    if "AC-LC" in phases_set:
        run_phase("AC-LC", build_ac_lc_factory_financebench(embedder, p1_dir), on_paraphrased=True)
    if "HGC-LC" in phases_set:
        run_phase(
            "HGC-LC",
            build_hgc_lc_factory_financebench(embedder, p1_db, p1_dir, scope_of=scope_of),
            on_paraphrased=True,
        )
    if "C-AC-LC" in phases_set:
        run_phase(
            "C-AC-LC",
            build_p4_ac_lc_factory_financebench(embedder, p1_dir, seed=42, fraction=0.20),
            on_paraphrased=True,
        )
    if "C-HGC-LC" in phases_set:
        c_db = out_dir / "C-HGC-LC" / "memory.db"
        c_db.parent.mkdir(parents=True, exist_ok=True)
        run_phase(
            "C-HGC-LC",
            build_p4_hgc_lc_factory_financebench(
                embedder, p1_db, p1_dir, c_db, scope_of=scope_of, seed=42, fraction=0.20
            ),
            on_paraphrased=True,
        )
    if "A-HGC-LC" in phases_set:
        a_db = out_dir / "A-HGC-LC" / "memory.db"
        a_db.parent.mkdir(parents=True, exist_ok=True)
        run_phase(
            "A-HGC-LC",
            build_adaptive_hgc_lc_factory_financebench(
                embedder, p1_db, p1_dir, a_db, scope_of=scope_of, seed=42, fraction=0.20
            ),
            on_paraphrased=True,
        )


def _run_financebench_lc_phases(
    phases: list[str],
    dataset: dict,
    out_dir: Path,
    cfg: ExperimentConfig,
    all_summaries: list[dict],
) -> None:
    from hgc.embeddings import Embedder
    from hgc.judge import LLMJudge
    from hgc.memory import HintStore
    from hgc.runner import PhaseRunner
    from hgc.smoke_common import (
        build_ac_lc_factory_financebench,
        build_adaptive_hgc_lc_factory_financebench,
        build_hgc_lc_factory_financebench,
        build_lc_naive_factory_financebench,
        build_p1_lc_factory_financebench,
        build_p4_ac_lc_factory_financebench,
        build_p4_hgc_lc_factory_financebench,
        make_tools_financebench,
        phase_summary,
    )

    judge = LLMJudge()
    embedder = Embedder()

    out_dir.mkdir(parents=True, exist_ok=True)

    p1_dir = out_dir / "P1-LC"
    p1_db = p1_dir / "memory.db"
    p1_dir.mkdir(parents=True, exist_ok=True)

    scope_of = dataset["scope_of"]

    runner = PhaseRunner(
        dataset=dataset,
        judge=judge,
        out_dir=out_dir,
        tool_factory=make_tools_financebench,
    )

    def run_phase(phase_name: str, factory, on_paraphrased: bool = False) -> list:
        print(f"\n--- Running {phase_name} (paraphrased={on_paraphrased}) ---", flush=True)
        results = runner.run_phase(phase_name, factory, on_paraphrased=on_paraphrased)
        runner.write_summary(phase_name, results)
        s = phase_summary(phase_name, results)
        all_summaries.append(s)
        print(
            f"    {phase_name}: {s['n_correct_judge']}/{s['n_queries']} correct (judge), "
            f"{s['total_tokens']:,} tokens, ${s['estimated_cost_usd']:.4f}"
        )
        return results

    phases_set = set(phases)

    if "P1-LC" in phases_set:
        store = HintStore(db_path=str(p1_db))
        run_phase(
            "P1-LC",
            build_p1_lc_factory_financebench(embedder, store, scope_of=scope_of, llm_judge=judge),
            on_paraphrased=False,
        )
    if "V-LC" in phases_set:
        run_phase("V-LC", build_lc_naive_factory_financebench(embedder), on_paraphrased=False)
    if "AC-LC" in phases_set:
        run_phase(
            "AC-LC",
            build_ac_lc_factory_financebench(embedder, p1_dir),
            on_paraphrased=True,
        )
    if "HGC-LC" in phases_set:
        run_phase(
            "HGC-LC",
            build_hgc_lc_factory_financebench(embedder, p1_db, p1_dir, scope_of=scope_of),
            on_paraphrased=True,
        )
    if "C-AC-LC" in phases_set:
        run_phase(
            "C-AC-LC",
            build_p4_ac_lc_factory_financebench(embedder, p1_dir, seed=42, fraction=0.20),
            on_paraphrased=True,
        )
    if "C-HGC-LC" in phases_set:
        c_db = out_dir / "C-HGC-LC" / "memory.db"
        c_db.parent.mkdir(parents=True, exist_ok=True)
        run_phase(
            "C-HGC-LC",
            build_p4_hgc_lc_factory_financebench(
                embedder, p1_db, p1_dir, c_db, scope_of=scope_of, seed=42, fraction=0.20
            ),
            on_paraphrased=True,
        )
    if "A-HGC-LC" in phases_set:
        a_db = out_dir / "A-HGC-LC" / "memory.db"
        a_db.parent.mkdir(parents=True, exist_ok=True)
        run_phase(
            "A-HGC-LC",
            build_adaptive_hgc_lc_factory_financebench(
                embedder, p1_db, p1_dir, a_db, scope_of=scope_of, seed=42, fraction=0.20
            ),
            on_paraphrased=True,
        )


def _run_hotpot_phases(
    phases: list[str],
    dataset: dict,
    out_dir: Path,
    cfg: ExperimentConfig,
    all_summaries: list[dict],
) -> None:
    from hgc.embeddings import Embedder
    from hgc.extraction import HintExtractor
    from hgc.judge import LLMJudge
    from hgc.memory import HintStore
    from hgc.runner import PhaseRunner
    from hgc.smoke_common import (
        build_hgc_core_factory_hotpot,
        build_m0_factory,
        build_mem0_config,
        build_p4_factory,
        build_rag_factory_hotpot,
        make_tools_hotpot,
        phase_summary,
    )

    judge = LLMJudge()
    embedder = Embedder()
    extractor = HintExtractor()

    out_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir = out_dir / "_tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    p1_dir = out_dir / "P1"
    p1_db = p1_dir / "memory.db"
    p1_dir.mkdir(parents=True, exist_ok=True)

    scope_of = dataset["scope_of"]

    runner = PhaseRunner(
        dataset=dataset,
        judge=judge,
        out_dir=out_dir,
        tool_factory=make_tools_hotpot,
    )

    def run_phase(phase_name: str, factory, on_paraphrased: bool = False) -> list:
        print(f"\n--- Running {phase_name} ---", flush=True)
        results = runner.run_phase(phase_name, factory, on_paraphrased=on_paraphrased)
        runner.write_summary(phase_name, results)
        s = phase_summary(phase_name, results)
        all_summaries.append(s)
        print(
            f"    {phase_name}: {s['n_correct_judge']}/{s['n_queries']} correct (judge), "
            f"{s['total_tokens']:,} tokens, ${s['estimated_cost_usd']:.4f}"
        )
        return results

    phases_set = set(phases)
    p1_results = []

    # --- P0: Vanilla (empty store, cluster-scoped) ---
    if "P0" in phases_set:
        p0_db = tmp_dir / "p0_empty.db"
        p0_store = HintStore(db_path=str(p0_db))
        run_phase(
            "P0",
            build_hgc_core_factory_hotpot(embedder, p0_store, extractor, scope_of),
        )

    # --- P1: Cold, sequential, cluster-scoped shared store ---
    if "P1" in phases_set:
        p1_store = HintStore(db_path=str(p1_db))
        p1_results = run_phase(
            "P1",
            build_hgc_core_factory_hotpot(embedder, p1_store, extractor, scope_of),
        )
        p1_store.close()

    # --- P2: Warm re-run (copy P1 store) ---
    if "P2" in phases_set:
        p2_db = out_dir / "P2" / "memory.db"
        p2_db.parent.mkdir(parents=True, exist_ok=True)
        if p1_db.exists() and not p2_db.exists():
            shutil.copy2(str(p1_db), str(p2_db))
        p2_store = HintStore(db_path=str(p2_db))
        run_phase(
            "P2",
            build_hgc_core_factory_hotpot(embedder, p2_store, extractor, scope_of),
        )
        p2_store.close()

    # --- P3-M0: Mem0 baseline seeded from P1 ---
    if "P3-M0" in phases_set:
        try:
            mem0_cfg = build_mem0_config(
                out_dir,
                collection_name="mem0_p3_m0_hotpot_smoke",
                sub_deployment=cfg.model_sub,
            )
            run_phase(
                "P3-M0",
                build_m0_factory(p1_dir, mem0_config=mem0_cfg, user_id="p3_m0_hotpot_smoke"),
            )
        except Exception as exc:
            logger.warning("P3-M0 failed (mem0ai may not be installed): %s", exc)
            all_summaries.append(
                {
                    "phase": "P3-M0",
                    "n_queries": 0,
                    "n_correct_judge": 0,
                    "n_correct_containment": 0,
                    "avg_tokens": 0,
                    "avg_time_s": 0.0,
                    "avg_iters": 0.0,
                    "total_tokens": 0,
                    "estimated_cost_usd": 0.0,
                    "error": str(exc),
                }
            )

    # --- P3-RAG: NaiveRAG baseline ---
    if "P3-RAG" in phases_set:
        run_phase("P3-RAG", build_rag_factory_hotpot(embedder))

    # --- P4: HGC core + contaminated P1 store ---
    if "P4" in phases_set:
        p4_db = out_dir / "P4" / "memory.db"
        run_phase(
            "P4",
            build_p4_factory(embedder, p1_db, extractor, p4_db, scope_of),
        )

    # P1 monotonicity check
    if p1_results:
        print("\n--- P1 n_added_hints (query order) ---")
        cumulative = 0
        prev_cum = 0
        monotone_ok = True
        for r in p1_results:
            print(f"  q{r.query_id}: n_added_hints={r.n_added_hints}")
            cumulative += r.n_added_hints
            if cumulative < prev_cum:
                monotone_ok = False
            prev_cum = cumulative
        if monotone_ok:
            print(f"  P1 cumulative n_added_hints is non-decreasing (total={cumulative}). OK.")
        else:
            print("  WARNING: P1 cumulative n_added_hints is NOT monotone non-decreasing!")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    args = parse_args()
    _apply_ablation_knobs(args)

    eid = experiment_id(args.dataset, args.model, args.seed)
    out_dir = Path(args.out_dir) if args.out_dir else _REPO / "results" / eid

    phases = phase_set_for(args.dataset, args.phases)

    if args.dry_run:
        print(f"DRY RUN: experiment_id={eid}")
        print(f"         out_dir={out_dir}")
        print(f"         n={args.n}, dataset={args.dataset}, model={args.model}, seed={args.seed}")
        print(f"         phases ({len(phases)}): {phases}")
        return 0

    t_start = time.time()
    print("=" * 70)
    print(f"HGC Experiment — {args.dataset.upper()} n={args.n} × {len(phases)} phases")
    print(f"model={args.model}, seed={args.seed}, out_dir={out_dir}")
    print("=" * 70)

    cfg, nothing_to_do = load_or_create_config(out_dir, args)
    if nothing_to_do:
        # Still regenerate smoke_summary.json from existing trajectories if present
        _maybe_regenerate_summary(out_dir, phases, cfg)
        cfg.save(out_dir)
        return 0

    # Save config immediately (creates the out_dir)
    cfg.save(out_dir)

    # Configure LM
    configure_lm(args.model)

    # Load dataset
    print(f"\nLoading dataset ({args.dataset}, n={args.n}, seed={args.seed})...", flush=True)
    dataset = load_dataset(args.dataset, args.n, args.seed)
    print(f"Loaded {len(dataset['queries'])} queries.")

    all_summaries: list[dict] = []

    # Dispatch to dataset-specific phase runner
    if args.dataset == "bcp":
        _run_bcp_phases(phases, dataset, out_dir, cfg, all_summaries)
    elif args.dataset == "qasper-rag":
        _run_qasper_rag_phases(phases, dataset, out_dir, cfg, all_summaries)
    elif args.dataset == "qasper-oracle":
        _run_qasper_oracle_phases(phases, dataset, out_dir, cfg, all_summaries)
    elif args.dataset == "financebench-lc":
        _run_financebench_lc_phases(phases, dataset, out_dir, cfg, all_summaries)
    else:
        _run_hotpot_phases(phases, dataset, out_dir, cfg, all_summaries)

    # Save aggregate smoke_summary.json
    smoke_summary_path = out_dir / "smoke_summary.json"
    smoke_summary_path.write_text(json.dumps(all_summaries, indent=2, ensure_ascii=False))
    print(f"\nSmoke summary saved to: {smoke_summary_path}")

    # Print table
    from hgc.smoke_common import print_phase_table

    print_phase_table(all_summaries)

    # Update config
    cfg.phases_completed = sorted(set(cfg.phases_completed) | set(phases))
    cfg.git_commit = detect_git_commit() or cfg.git_commit
    cfg.save(out_dir)
    print(f"\nConfig saved to {out_dir}/config.json")

    total_time = time.time() - t_start
    print(f"Total wall time: {total_time:.1f}s")

    return 0


def _maybe_regenerate_summary(out_dir: Path, phases: list[str], cfg: ExperimentConfig) -> None:
    import json as _json

    from hgc.runner import PhaseResult
    from hgc.smoke_common import phase_summary

    all_summaries = []
    for phase_name in phases:
        phase_dir = out_dir / phase_name
        if not phase_dir.exists():
            continue
        traj_files = sorted(phase_dir.glob("trajectory_q*.json"))
        results = []
        for tf in traj_files:
            try:
                data = _json.loads(tf.read_text(encoding="utf-8"))
                if data.get("error"):
                    continue
                results.append(
                    PhaseResult(
                        phase=phase_name,
                        query_id=str(data.get("query_id", "")),
                        gold=data.get("gold", ""),
                        pred=data.get("pred", ""),
                        judgment_correct=bool(data.get("judgment_correct", False)),
                        judgment_reasoning=data.get("judgment_reasoning", ""),
                        containment_correct=bool(data.get("containment_correct", False)),
                        tokens=int(data.get("tokens", 0)),
                        time_s=float(data.get("time_s", 0.0)),
                        n_iters=int(data.get("n_iters", 0)),
                        n_retrieved_positive=int(data.get("n_retrieved_positive", 0)),
                        n_retrieved_negative=int(data.get("n_retrieved_negative", 0)),
                        n_added_hints=int(data.get("n_added_hints", 0)),
                        error=data.get("error", ""),
                    )
                )
            except Exception:
                continue
        if results:
            all_summaries.append(phase_summary(phase_name, results))

    if all_summaries:
        smoke_summary_path = out_dir / "smoke_summary.json"
        smoke_summary_path.write_text(_json.dumps(all_summaries, indent=2, ensure_ascii=False))
        print(f"Regenerated smoke_summary.json ({len(all_summaries)} phases)")


if __name__ == "__main__":
    sys.exit(main())
