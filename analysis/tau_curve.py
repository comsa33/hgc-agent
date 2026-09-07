"""Precompute the cache-hit curve over tau without running any experiment.

The answer cache is the only place tau is read (``AnswerCacheAgent.run``
compares the best cosine similarity against ``sim_threshold``), and the
warm-up seeder ignores it entirely. So which queries hit, and which cache
entry each one hits, is decided by embeddings alone — no backbone, no gate,
no judge. That makes the whole tau axis computable offline.

Only the tau values that actually move the hit set are worth paying for in a
sensitivity sweep; the rest reproduce a neighbouring point exactly. This
script reports, per tau, the hit count and whether the hit set differs from
the released tau=0.85 configuration.

Cost is one embedding call per distinct question (~400 per cell at
$0.02/1M tokens). Run from the repo root:

    uv run python analysis/tau_curve.py --cell qasper-oracle
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from hgc.paths import results_root
from dotenv import load_dotenv

load_dotenv()

_REPO = Path(__file__).resolve().parents[1]

# cell -> (results dir, warm-up phase, a serve phase to read paraphrases from)
CELLS = {
    "qasper-oracle": ("base_qasper_oracle", "P1-LC", "HGC-LC"),
    "financebench-lc": ("base_finbench_lc", "P1-LC", "HGC-LC"),
    "qasper-rag": ("base_qasper_rag", "P1-RAG", "HGC-RAG"),
    "bcp": ("bcp_gpt41_seed42", "P1", "P3-Hybrid"),
}

DEFAULT_TAUS = [0.70, 0.75, 0.80, 0.825, 0.85, 0.875, 0.90, 0.925, 0.95]


def _load_phase(results_dir: Path, phase: str) -> list[dict]:
    return [
        json.loads(p.read_text(encoding="utf-8"))
        for p in sorted((results_dir / phase).glob("trajectory_q*.json"))
    ]


def _embed_all(texts: list[str]) -> dict[str, np.ndarray]:
    """Embed each distinct text once, L2-normalised."""
    from hgc.embeddings import Embedder

    embedder = Embedder()
    distinct = sorted(set(texts))
    out: dict[str, np.ndarray] = {}
    for text in distinct:
        vec = np.asarray(embedder.embed(text), dtype=np.float32)
        norm = float(np.linalg.norm(vec))
        out[text] = vec / norm if norm else vec
    return out


def build_curve(cell: str, taus: list[float]) -> dict:
    results_dir_name, warmup_phase, serve_phase = CELLS[cell]
    results_dir = results_root(_REPO) / results_dir_name

    warmup = _load_phase(results_dir, warmup_phase)
    serve = _load_phase(results_dir, serve_phase)
    if not warmup or not serve:
        raise SystemExit(f"{cell}: missing {warmup_phase} or {serve_phase} under {results_dir}")

    # Mirror seed_answer_cache_from_p1 exactly: judge-correct, and both the
    # question and the answer non-empty. Dropping either condition shifts the
    # hit count and the curve stops matching the run it is meant to predict.
    seeded = [
        t["question"]
        for t in warmup
        if t.get("judgment_correct") and t.get("question") and (t.get("pred") or t.get("answer"))
    ]
    queries = [t["question"] for t in serve]

    vectors = _embed_all(seeded + queries)
    cache_matrix = np.stack([vectors[q] for q in seeded])  # (M, dim), normalised
    query_matrix = np.stack([vectors[q] for q in queries])  # (N, dim), normalised

    sims = query_matrix @ cache_matrix.T  # cosine, both sides unit-norm
    best_idx = sims.argmax(axis=1)
    best_sim = sims.max(axis=1)

    rows = []
    baseline_key: tuple | None = None
    for tau in sorted(taus):
        hit = best_sim >= tau
        # Which cache entry each hitting query lands on — two taus that hit the
        # same queries via the same entries produce byte-identical serve runs.
        key = tuple(int(i) if h else -1 for i, h in zip(best_idx, hit, strict=True))
        if abs(tau - 0.85) < 1e-9:
            baseline_key = key
        rows.append({"tau": tau, "hits": int(hit.sum()), "key": key})

    for row in rows:
        row["same_as_released"] = row.pop("key") == baseline_key

    # Cross-check against the released run: at tau=0.85 the predicted hit set
    # must equal the paths actually recorded. A silent mismatch here would
    # send the sweep to the wrong tau values.
    # Anything that is not a miss is a hit. BCP adds a fourth path
    # (cache_hit_no_hint) that the long-context and RAG cells never produce;
    # enumerating hit paths by name silently undercounts it.
    observed_hits = sum(1 for t in serve if t.get("path") and t["path"] != "cache_miss")
    predicted_085 = int((best_sim >= 0.85).sum())
    # The embedding API is not bit-deterministic across calls (~1e-4 wobble),
    # so a query whose best similarity sits within that wobble of tau can flip
    # sides between the recorded run and this recomputation. Such queries are
    # tolerated by the check and reported — and they are themselves a
    # sensitivity finding: tau cuts straight through them.
    boundary = [
        {"query_id": t["query_id"], "best_sim": float(sim)}
        for t, sim in zip(serve, best_sim, strict=True)
        if abs(float(sim) - 0.85) <= 1e-3
    ]

    return {
        "cell": cell,
        "observed_hits_in_run": observed_hits,
        "predicted_hits_at_085": predicted_085,
        "matches_run": abs(observed_hits - predicted_085) <= len(boundary),
        "boundary_queries_at_085": boundary,
        "n_seeded": len(seeded),
        "n_queries": len(queries),
        "best_sim_min": float(best_sim.min()),
        "best_sim_median": float(np.median(best_sim)),
        "best_sim_max": float(best_sim.max()),
        "curve": rows,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cell", choices=sorted(CELLS), required=True)
    ap.add_argument("--taus", type=float, nargs="*", default=DEFAULT_TAUS)
    ap.add_argument("--out", type=Path, default=None, help="write JSON here as well")
    args = ap.parse_args()

    report = build_curve(args.cell, args.taus)

    print(f"cell={report['cell']}  seeded={report['n_seeded']}  queries={report['n_queries']}")
    status = "OK" if report["matches_run"] else "MISMATCH — do not use this curve"
    print(
        f"cross-check vs released run: predicted {report['predicted_hits_at_085']} hits at "
        f"tau=0.85, observed {report['observed_hits_in_run']} — {status}"
    )
    for b in report["boundary_queries_at_085"]:
        print(
            f"  boundary query {b['query_id']} best_sim={b['best_sim']:.6f} — within "
            "embedding-API wobble of tau; may flip between runs"
        )
    print(
        f"best-match similarity: min={report['best_sim_min']:.4f} "
        f"median={report['best_sim_median']:.4f} max={report['best_sim_max']:.4f}"
    )
    print(f"\n{'tau':>7}  {'hits':>5}  {'hit rate':>9}  identical to released run?")
    for row in report["curve"]:
        rate = 100 * row["hits"] / report["n_queries"]
        mark = "yes — no need to run" if row["same_as_released"] else "NO — worth running"
        print(f"{row['tau']:7.3f}  {row['hits']:5d}  {rate:8.1f}%  {mark}")

    if args.out:
        args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
