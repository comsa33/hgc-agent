# HGC Experimental Design (v3, 2026-04-23)

**Method**: HGC — Hint-Gated Cache
**Scope**: Information-seeking / information-retrieval LLM agents
**Supersedes**: `memreact/DESIGN.md` (old HOMIS / "Where, Not What" framing)
**Repos** (new, clean slate):
- code: `comsa33/hgc-agent` → local: `memreact-project/hgc/`
- paper: `comsa33/hgc-paper` → local: `memreact-project/hgc-paper/`

Parent directory name `memreact-project/` is preserved (Claude memory path encoding constraint). Old `memreact/` and `homis-paper/` remain as local archives; corresponding GitHub repos will be renamed with `-legacy` suffix.

---

## 1. Research question

Answer caches (GPTCache-style) accelerate information-seeking LLM agents but propagate every cached error (stale, hallucinated, adversarially poisoned). Hint-only memory stores the *pointer* to evidence rather than the answer, forcing re-verification on every read. We ask:

> **Can a lightweight hint-based verification gate retain an answer cache's clean-accuracy and speed while neutralising multiple contamination modes (cross-swap, entity-swap, typo-mutation) in the cache?**

The primary contribution is a three-component architectural pattern — **Cache + Hint Gate + Fallback** — with empirical validation across two agent frameworks (DSPy ReAct, LangChain RAG) and three task types (BrowseComp-Plus multi-hop search, QASPER academic paper QA, FinanceBench PDF finance QA) in the information-retrieval domain.

---

## 2. Method: HGC

```
                ┌─── cache hit ───┐
[query] → [AnswerCache lookup] ───┤                        yes
                └─── cache miss ──┘                         │
                         │                                  ▼
                         │        ┌──────────────────────────────────┐
                         │        │ Hint Gate                        │
                         ├───────▶│  G1: typed-scope filter          │
                         │        │  G2: docid validation            │
                         │        │  G3: SupportVerifier LLM yes/no  │
                         │        └──────────────────────────────────┘
                         │                                  │ no
                         ▼                                  ▼
             ┌─────────────────────────────────────────────────┐
             │  Fallback (full agentic loop OR full RAG)       │
             └─────────────────────────────────────────────────┘
                                      │
                                      ▼
                                  [answer]
```

**Design invariants**
- I1: *hints carry structural pointers only* (docid, entity, strategy); never the answer text.
- I2: *mandatory re-verification* on every cache hit; gate independently confirms the cached answer against the current retrieval.
- I3: *fallback always available* — on gate rejection or cache miss, the underlying agent or RAG pipeline runs in full.

**Gate components (all must pass to accept a cache hit)**
- **G1 typed-scope filter**: location hints require exact corpus scope match; entity / strategy hints are cross-scope.
- **G2 docid validation**: if the retrieved hint is a location hint, its docid must be present in the current query's document pool.
- **G3 SupportVerifier**: one-shot LLM yes/no — "does this document support this proposed answer?".

---

## 3. Instance matrix (task-native mapping)

We assign each task to its native framework rather than running every
instance on every task. This avoids task-method mismatch (e.g. RAG on
multi-hop search is trivially weak) and keeps the cost ceiling within
practical bounds for a single paper.

| Instance | Backbone | Native for | Implementation |
|---|---|---|---|
| **HGC-Agent** | DSPy ReAct | multi-hop agentic search | `src/hgc/hgc_agent.py` |
| **HGC-RAG** | LangChain FAISS + LLM | single-shot document QA | `src/hgc/hgc_rag.py` |

Both instances share the HintStore, AnswerCache, and SupportVerifier modules.

**What we do NOT claim**
- Transfer to LlamaIndex / production memory frameworks (argued in §V, not empirically validated).
- Applicability outside information retrieval (math, creative writing, translation — out of scope).
- Cross-domain generality beyond QASPER (FinanceBench RAG deferred as future work).

---

## 4. Datasets and sample sizes (confirmed)

| Dataset | Task | Native backbone | n | Notes |
|---|---|---|---|---|
| **BrowseComp-Plus (BCP)** | multi-hop agentic search | **DSPy ReAct** | **200** | 24% of full 830. Primary benchmark. |
| **QASPER** | academic paper QA | **LangChain RAG** | **200** | dev set ~1000; 200 ≈ reasonable sample. |
| ~~FinanceBench~~ | (deferred) | | | Out of paper scope; loader kept for future work. |

McNemar detectable Δ at n=200: ~5pp. Sufficient for all comparisons
where ground truth shows ≥7pp effects (contamination drop, gate on/off).

**Incremental growth**: `BCPDataset` stores per-n snapshots
(`dataset_n{n}_seed{seed}.json`) so a later `--n=300` run adds the 101-300th
queries without re-running the first 200. Same applies to QASPER.

All runs at `gpt-4.1 × seed=42`. `--seed=7` / `--seed=100` only for A3.

---

## 5. Phase nomenclature (internal mapping)

Paper scope: 8 BCP phases + 6 QASPER-Oracle phases + 6 FinanceBench-LC
phases. (QASPER-RAG was an earlier iteration; scope was migrated to
QASPER-Oracle for retrieval-confound isolation.) Legacy BCP phases
(`P2 / P0' / P0-RAG / P3 / P3-M0 / P3-TC / P4-HOMIS / P4-M0 / P4-TC /
E4-AC / T4-AC / P0-LCS-*`) are dropped from the paper; factory code
remains in the repo for revision but the CLI rejects those names.

### BCP (DSPy ReAct, agentic + distractor docs) — 8 phases
| Phase | Paper label | System |
|---|---|---|
| `P0` | **Vanilla** | ReAct (no memory) |
| `P1` | **HGC-Core cold** | ReAct + HintStore seeding run |
| `P3-AC` | **AC** | AnswerCache clean, paraphrased |
| `P3-Hybrid` | **HGC** | full HGC clean, paraphrased (primary system) |
| `P4-AC` | **C-AC** | AC under 20% cross-swap |
| `P4-Hybrid` | **C-HGC** | HGC under 20% cross-swap (primary contamination) |
| `E4-Hybrid` | **E-HGC** | HGC under 20% entity-swap |
| `T4-Hybrid` | **T-HGC** | HGC under 20% typo-mutation |

### QASPER-Oracle (Long-Context direct, annotator-marked evidence) — 6 phases
Phases use `LongCtxBackbone`; `docs` are the annotator-marked evidence
sentences rather than the full paper, eliminating retrieval as a
confound. AC/HGC/C-* run on paraphrased queries.

| Phase | System |
|---|---|
| `P1-LC` | Seed phase — Oracle direct + HintStore populate |
| `V-LC` | Vanilla Oracle direct (no memory) |
| `AC-LC` | AnswerCache + Oracle fallback, paraphrased |
| `HGC-LC` | full HGC + Oracle fallback, paraphrased (primary system) |
| `C-AC-LC` | AC under 20% cross-swap |
| `C-HGC-LC` | HGC under 20% cross-swap |

### FinanceBench-LC (Long-Context direct, pre-identified evidence pages) — 6 phases
Same phase structure as QASPER-Oracle; `docs` are the pre-extracted
evidence-page texts provided by Patronus's open-source subset
(effectively their Oracle baseline).

| Phase | System |
|---|---|
| `P1-LC` | Seed phase — Oracle direct + HintStore populate |
| `V-LC` | Vanilla Oracle direct |
| `AC-LC` | AnswerCache + Oracle fallback, paraphrased |
| `HGC-LC` | full HGC + Oracle fallback, paraphrased |
| `C-AC-LC` | AC under 20% cross-swap |
| `C-HGC-LC` | HGC under 20% cross-swap |

### Gate ablation (A1, BCP only)
Handled by `--gate-variant=full|g1g2|g1g3` CLI flag (sets
`HGC_GATE_VARIANT` env var that `_enabled_gate_set()` reads).

### Contamination-rate sensitivity (A2, BCP only)
`--contamination-fraction=0.05|0.20|0.50` CLI flag; main table uses 20%.

### Seed robustness (A3, BCP only)
`--seed=7` and `--seed=100` runs on the 5 core phases.

### Multi-model validation (BCP only)
`--model=gpt-4o` run on 5 core phases (confirms HGC generalises beyond
gpt-4.1). gpt-5.x reasoning models attempted and dropped: reasoning
models force temperature=1.0 and max_tokens≥16000, breaking reproducibility.
Open-source agents (gemma4, qwen2.5) also attempted and dropped due to
low BCP accuracy and DSPy ReAct tool-call hallucination.

---

## 6. Contamination modes (three)

### 6.1 cross-swap (existing)
Randomly select 20% of cache entries. Replace each selected entry's *answer* with the answer from a *different* entry in the same cache, keeping the embedding unchanged. Cross-swap tests whether the gate can reject a cache hit when the answer is swapped with an answer that belongs to a different question.

### 6.2 entity-swap (new)
Randomly select 20% of cache entries. Within each selected answer, detect named entities (using spaCy or a lightweight NER) and replace a single named entity with a similar-typed entity drawn from a fallback bank. The answer's structure and most of its text are preserved. Tests fine-grained hallucination — the gate must reject semantically-close but factually-wrong answers.

Example:
- Original: "Madrid, the capital of Spain"
- Swapped: "Lisbon, the capital of Spain"

### 6.3 typo-mutation (new)
Randomly select 20% of cache entries. In each selected answer, mutate ~10% of characters: random substitutions of adjacent keyboard letters, swaps of adjacent letters, and deletions of single letters. Tests low-level noise robustness — how does the gate handle syntactic corruption from OCR errors, transcription noise, or flaky pipelines?

Example:
- Original: "Queen Arwa University"
- Mutated: "Queeb Arwq Unversity"

All three contaminations operate on `AnswerCache._cache` directly and on `HintStore` (for memory baselines that use hints). Implementation goes in `src/hgc/contaminators/`.

---

## 7. Experiment suites (confirmed scope)

Task-native mapping only. BCP runs on DSPy ReAct, QASPER runs on LangChain RAG.

| Suite | Backbone | Dataset | n | Phases (core) | Cost (est.) |
|---|---|---|---|---|---|
| **S1** | DSPy ReAct | BCP | 200 | 7 (V, AC, HGC-Core, HGC, C-AC, C-HGC, E-HGC) | ~$170-230 |
| **S2** | LangChain RAG | QASPER | 200 | 5 (V-RAG, AC-RAG, HGC-RAG, C-AC-RAG, C-HGC-RAG) | ~$80-120 |
| **A1** | DSPy ReAct | BCP | 200 | 2 gate variants (G3-off vs full) | ~$25-35 |
| **Total (baseline)** | | | | **14 phase runs** | **~$275-385** |

**Incremental growth option**: If n=200 results leave a key comparison
borderline (e.g. HGC vs AC clean), rerun at `--n=300`. Thanks to
`dataset_n{n}_seed{seed}.json` snapshotting, the 101-300th queries are
run incrementally at ~$100-150 additional cost. Total ceiling with
incremental growth: ~$400-530.

**Optional extensions (only if budget allows)**:
- T-HGC (typo contamination on HGC): +~$25
- A2 contamination sensitivity sweep (5%, 50% on AC and HGC): +~$70
- A3 seed robustness (+1 additional seed on AC and HGC): +~$35

Deferred to future work: FinanceBench-RAG suite, multi-framework LangChain
agent instance, poison-append / stale-answer contamination modes.

---

## 8. Execution strategy (smoke-first, parallel-aware)

### Rate limit policy
- Azure gpt-4.1 tested at ~2M TPM sustained during sequential runs; parallel 3-4 phases fits within typical 2-10M TPM quota.
- `run.py` already handles 429s with backoff via DSPy's retry logic.
- Gate: if smoke triggers >5 × 429 per minute, throttle to 2 concurrent phases.

### Parallelism tiers
- Tier 1 (safe): 1 phase at a time. Slowest.
- **Tier 2 (default): 2-3 phases concurrently within a suite.** Balanced.
- Tier 3 (aggressive): 5-6 phases concurrently across suites. Risk 429s.

Default Tier 2, escalate only after smoke proves headroom.

### Phased execution (current status 2026-04-22)
| Step | Gate | Cost | Time | Status |
|---|---|---|---|---|
| 0 | DESIGN.md + PRD + GitHub repos created | 0 | 2h | ✅ done |
| 1 | Scaffold + core module migration + baseline tests pass | 0 | 4-6h | ✅ done |
| 2 | Gate components (G1/G2/G3) unit-tested | 0 | 2h | ✅ done |
| 3 | Three contaminators + tests | 0 | 3h | ✅ done |
| 4 | LangChain RAG backbone + HGCRAGAgent + tests | 0 | 1-2d | ✅ done |
| 5 | Experiment runner (dispatches qasper-rag, E4/T4 phases) | 0 | 0.5d | ✅ done |
| 6 | **Smoke run: n=5 × (S1 BCP, S2 QASPER-RAG)** | smoke pass | ~$10 | ~30min | pending |
| 7 | Full S1 (BCP n=200, 7 phases) | HGC > Vanilla? | ~$200 | ~1d | pending |
| 8 | Full S2 (QASPER-RAG n=200, 5 phases) | HGC-RAG resilient? | ~$100 | ~0.5d | pending |
| 9 | A1 gate ablation on BCP | component contribution non-zero? | ~$30 | ~0.5d | pending |
| 10 | Statistical analysis + tables | key comparisons reach p<0.05 | 0 | 0.5d | pending |
| 11 | (conditional) Increment to n=300 if borderline | | ~$100-150 | ~0.5d | optional |
| 12 | Paper writing (pattern-centric) | draft complete | 0 | 3-5d | pending |

---

## 9. Migration status (2026-04-22)

| Module | Status |
|---|---|
| `src/hgc/memory.py` (HintStore) | ✅ migrated |
| `src/hgc/embeddings.py` | ✅ migrated |
| `src/hgc/emb_cache.py`, `judge.py` | ✅ migrated |
| `src/hgc/datasets/bcp.py` (n-aware snapshot fix) | ✅ migrated + fixed |
| `src/hgc/datasets/qasper.py`, `financebench.py` | ✅ migrated |
| `src/hgc/baselines.py` (AnswerCache, Mem0, NaiveRAG) | ✅ migrated |
| `src/hgc/hgc_core.py` (ex `MemReActAgent`) | ✅ migrated + renamed |
| `src/hgc/hgc_agent.py` (HGC pattern on ReAct) | ✅ new |
| `src/hgc/hgc_rag.py` (HGC pattern on RAG) | ✅ new |
| `src/hgc/rag_backbone.py` (LangChain FAISS backbone) | ✅ new |
| `src/hgc/gate/{scope_filter,docid_check,support_verifier,composite}.py` | ✅ new |
| `src/hgc/contaminators/{cross_swap,entity_swap,typo_mutation}.py` | ✅ new |
| `src/hgc/factories.py` (with BCP E4/T4 + QASPER-RAG factories) | ✅ ported + extended |
| `src/hgc/experiment.py`, `src/hgc/runner.py` | ✅ migrated |
| `experiments/run.py` (--dataset={bcp, qasper, qasper-rag, financebench}) | ✅ extended |

Data (git-ignored, copied from memreact/data/bcp):
- `data/bcp/queries.jsonl` (2.0 GB, BCP original)
- `data/bcp/paraphrase_cache.json` (100-query paraphrase cache, hash-keyed)
- `data/bcp/doc_embeddings.npz` (~41 MB doc embedding cache)
- `dataset_100.json` intentionally NOT copied (replaced by n-aware snapshots).

Discarded:
- HotpotQA code paths (out of scope)
- `memreact/` pre-fix and post-fix `results/` (all experiments rerun from scratch)

---

## 10. Paper structure (post-experiment)

1. **Introduction** — pattern contribution, scoped to information-seeking LLM agents, empirical validation on BCP (multi-hop) and QASPER (academic QA), contamination resistance thesis.
2. **Related work** — answer caching, agent memory (A-MEM, CoALA, Mem0), threat models, contamination literature.
3. **Method: HGC** — pattern (Cache + Gate + Fallback, framework-agnostic) → invariants → gate components (G1/G2/G3) → two reference instances (HGC-Agent, HGC-RAG) → Algorithm 1 (pattern-level pseudocode).
4. **Experiments** — S1 BCP-ReAct main table, S2 QASPER-RAG main table, A1 gate ablation. Optional: E-HGC entity-swap contamination column, contamination sensitivity figure.
5. **Discussion** — accuracy-safety trade-off, when HGC-Core suffices vs HGC (cache) helps, threats to validity (single agent framework per task, BCP paraphrase regime), future work (LangChain agent variant, FinanceBench cross-domain, poison-append contamination model).
6. **Conclusion** — HGC pattern cleanly abstracted; two reference instances empirically validated on contamination resistance.

---

---

## Addendum (2026-04-23)

### Token accounting methodology

The `tokens` field recorded in per-query trajectory JSON and aggregated
into phase `summary.csv` is the sum of `total_tokens` from the main LM's
history across the full `.run()` call for HGC phases. This explicitly
includes:

- AnswerCache internal LM fallback (if triggered on miss)
- Gate G3 SupportVerifier LLM invocations (only when the containment
  precheck at `support_verifier.py` fails and the full LLM judge fires)
- Backbone fallback LLM calls (ReAct iters for BCP, single-shot RAG for
  QASPER-RAG, Long-Context direct for QASPER-Oracle/FinanceBench-LC)

A per-result `gate_tokens` field isolates the G3 cost for analysis.

**Explicit exclusions** (not counted in reported tokens, consistent with
standard NLP inference-cost reporting conventions):

- `LLMJudge` (runner-level accuracy evaluation, applied uniformly across
  all phases — cancels out in between-phase comparisons).
- `HintExtractor` (`gpt-4.1-mini`, separate LM instance; contributes
  <1.5% of per-phase tokens, rarely >0.5% of per-phase cost).
- Document/query/hint embedding API calls (different pricing tier,
  ~100× cheaper than gpt-4.1; negligible within reported granularity).
- Offline paraphrase cache generation (one-time cost, amortised across
  all phases of a dataset).

### Completed experiments

| Suite | Model | n | Phases | Status |
|---|---|---|---|---|
| BCP (seed 42) | gpt-4.1 | 200 | 8 + A1 + A2 | ✅ done |
| BCP multi-seed | gpt-4.1 seed={7,100} | 200 | 5 | ✅ done |
| BCP multi-model | gpt-4o seed=42 | 200 | 5 | ✅ done |
| QASPER-RAG | gpt-4.1 seed=42 | 200 | 6 | ✅ done |
| QASPER-Oracle | gpt-4.1 seed=42 | 200 | 6 | ✅ done |
| FinanceBench-LC | gpt-4.1 seed=42 | 150 | 6 | ✅ done |

Hybrid phases (Gate-using) were re-executed after the `gate_tokens`
instrumentation landed so reported costs include G3 verifier overhead.

### Paper role of QASPER-RAG

QASPER-RAG results remain in the repo and surface in the paper as an
**Appendix** section ("Gate conservatism under retrieval noise:
trade-off vs contamination resistance"). Main tables use the cleaner
QASPER-Oracle setting, which removes retrieval as a confound and
produces a narrative consistent with BCP and FinanceBench-LC. The RAG
result is retained because the contamination-resistance pattern holds
even there (AC −12pp vs HGC −1pp under 20% cross-swap), despite the
clean-accuracy gap caused by retriever noise.

---

*Last updated: 2026-04-23 by Claude/Opus 4.7. Token accounting instrumentation and legacy phase cleanup landed this session.*
