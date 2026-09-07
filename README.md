# HGC — Hint-Gated Cache

Contamination-resilient answer caching for information-seeking LLM agents, and
the experiments behind the paper's claim that grounding a cached answer is not
the same as checking that it answers the question.

## What it is

```
[query] → [AnswerCache lookup] ──hit──▶ [Hint Gate: scope + docid + LLM verifier]
                │                                │
                └──miss──▶ [Full agentic/RAG fallback] ◀──gate rejects──┘
```

The gate uses compact structural hints (docid references, entity names, search
strategies) to check that a cached answer is supported by the document
substrate available now, not by whatever was cached in the past. Three
backbones: `hgc_agent.py` (DSPy ReAct, BCP), `hgc_rag.py` (LangChain
RetrievalQA, QASPER-RAG), `longctx_backbone.py` (long-context reader,
QASPER-Oracle and FinanceBench-LC).

## Setup

```bash
uv sync --extra dev            # runtime + test dependencies from uv.lock
cp .env.example .env           # then fill in provider keys
.venv/bin/pytest -q            # 372 passed, 2 skipped, no API calls (~15s)
```

Needs Python ≥3.11, [uv](https://docs.astral.sh/uv/), Azure OpenAI deployments
for `gpt-4.1`, `gpt-4.1-mini` and `text-embedding-3-small`, and an Anthropic key
for the abstention-classifier ensemble. A full reproduction of every cell costs
roughly \$400--\$460 in inference. `.env.example` lists every variable the code
reads, including the ablation knobs.

## Data

The benchmarks are not redistributed. QASPER downloads itself from HuggingFace
(`allenai/qasper`, CC-BY-4.0). FinanceBench needs a Patronus AI licence.
BrowseComp-Plus comes from its official distribution and goes in
`data/bcp/queries.jsonl` plus `data/bcp/corpus/`; the runner caches a query
subset and a document-embedding file there on first run.

## Reproducing the tables

The per-query trajectories behind every reported cell are attached to the
GitHub release as `hgc-trajectories.tgz` (86 MB, 55 experiment directories).
Unpack it at the repository root, which creates `trajectories/`; every number
in the paper then comes from the scripts below, none of which calls a paid API.

```bash
tar -xzf hgc-trajectories.tgz          # creates trajectories/
```

```bash
uv run python analysis/summary_tables.py                    # E1-E8 grids
uv run python experiments/analyze_results.py               # accuracy, McNemar, cost, paths
uv run python experiments/compute_judge_agreement.py       # three-rater IAA, Fleiss kappa
uv run python analysis/grounding_checkers/analyze.py       # public-checker comparison
uv run python analysis/tau_curve.py                        # threshold sweep (needs embeddings)
```

Re-running the experiments themselves is optional. Phase names are
case-sensitive: `P1`, `P3-AC`, `P3-Hybrid`, `P4-AC`, `P4-Hybrid`, `E4-Hybrid`,
`T4-Hybrid`, `A-Hybrid` (BCP); `AC-RAG`, `HGC-RAG`, `C-AC-RAG`, `C-HGC-RAG`,
`A-HGC-RAG` (QASPER-RAG); `AC-LC`, `HGC-LC`, `C-AC-LC`, `C-HGC-LC`, `A-HGC-LC`
(QASPER-Oracle, FinanceBench-LC), where the `C-` prefix is random cross-swap
contamination and `A-` is the gate-aware attack. `--verifier-predicate` selects
the G3 predicate: `support` (released), `answerhood`, `two_stage`, `two_stage_doc`.

```bash
# main grid
uv run python experiments/run.py --dataset bcp --seed 42 --n 200 \
    --phases P1,P3-AC,P3-Hybrid,P4-AC,P4-Hybrid
uv run python experiments/run.py --dataset qasper-oracle --n 200
uv run python experiments/run.py --dataset qasper-rag --n 200
uv run python experiments/run.py --dataset financebench-lc --n 150

# gate-aware attack and the three repairs
uv run python experiments/run.py --dataset qasper-oracle --n 200 --phases A-HGC-LC \
    --out-dir trajectories/adaptive_qasper_oracle
uv run python experiments/run.py --dataset qasper-oracle --n 200 --phases A-HGC-LC \
    --verifier-containment-min-chars 100000 \
    --out-dir trajectories/contoff_adaptive_qasper_oracle
uv run python experiments/run.py --dataset qasper-oracle --n 200 --phases A-HGC-LC \
    --verifier-predicate answerhood \
    --out-dir trajectories/pred_adaptive_qasper_oracle
uv run python experiments/run.py --dataset qasper-oracle --n 200 --phases A-HGC-LC \
    --verifier-predicate two_stage \
    --out-dir trajectories/twostage_adaptive_qasper_oracle
```

**Always pass `--out-dir` for anything other than a main-grid run.** Without it
the runner resolves to the main-run directory and rewrites its cross-phase
summary with only this invocation's phases.

Ablations follow the same shape with `--gate-variant {g1g2,g1g3}` and
`--contamination-fraction {0.05,0.50}`; the released directory names under
`trajectories/` say which run produced which table row.

## What is in the trajectory archive

- `trajectories/<experiment>/<phase>/trajectory_q*.json` — per-query records,
  including the full ReAct traces and the retrieved-passage payloads. Those
  payloads are benchmark text and carry whatever the source documents carry,
  including third-party contact details that appear in BCP web pages.
- `.../memory.db` — the SQLite hint store. The per-row `query_ctx_embedding`
  BLOBs are cleared; they are deterministic from `query_ctx` text and can be
  recomputed with `src/hgc/embeddings.py`.
- `.../summary.csv` — per-query verdicts and token counts, enough to redo the
  statistics without touching the trajectories.
- `analysis/judge_validation/` — the 200-row judge-versus-human agreement data
  behind the appendix: three raters, Fleiss $\kappa = 0.813$, judge against the
  human majority on 96.5% of rows.
- `analysis/grounding_checkers/` — the 713 triplets and raw scores behind the
  public-checker comparison.
- `data/<benchmark>/paraphrase_cache.json` — the cache warm-up paraphrases, so
  cache state is deterministic on a rerun.

## Licence

MIT for the code. Each benchmark keeps its own licence.
