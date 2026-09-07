# HGC Final Analysis (post G3 token fix)

Generated from `experiments/analyze_results.py` using released trajectory JSONs in `trajectories/<experiment>/<phase>/` (or `results/...` in the code repository).

## 1. Accuracy table (judge)

Drop columns are computed from raw correct/n counts (so they may differ in the last digit from the displayed-cell difference: e.g., FinanceBench-LC HGC drop $128/150 - 124/150 = -2.67$pp $\to -2.7$pp on a raw-count basis, while the displayed-percent difference $85.3 - 82.7 = -2.6$pp). The paper Table 1 caption uses the same raw-count convention.

| Experiment | Clean AC | Clean HGC | Contam AC | Contam HGC | AC drop | HGC drop |
|---|---|---|---|---|---|---|
| BCP seed=42 (gpt-4.1) | 90.0 | 88.5 | 75.0 | **82.0** | -15.0 | -6.5 |
| BCP seed=7 | 87.0 | 85.0 | 72.0 | **85.0** | -15.0 | +0.0 |
| BCP seed=100 | 82.0 | 80.5 | 72.5 | **77.5** | -9.5 | -3.0 |
| BCP gpt-4o | 76.0 | 75.5 | 62.5 | **72.0** | -13.5 | -3.5 |
| QASPER-RAG | 58.0 | 58.5 | 47.0 | **57.5** | -11.0 | -1.0 |
| QASPER-Oracle | 56.5 | 58.5 | 48.0 | **56.5** | -8.5 | -2.0 |
| FinanceBench-LC | 86.7 | 85.3 | 71.3 | **84.7** | -15.3 | -0.7 |

## 2. McNemar's tests

Pair-level comparison per query, 2x2 contingency on judge correctness.
Significance: *** p<0.001, ** p<0.01, * p<0.05, . p<0.10

### 2.1 Contaminated: C-AC vs C-HGC

| Test | n | a/b/c/d | χ² | p |
|---|---|---|---|---|
| BCP seed=42 (gpt-4.1) (contam) | 200 | a=136 b=14 c=28 d=22 | 4.02 | 0.04486* |
| BCP seed=7 (contam) | 200 | a=135 b=9 c=35 d=21 | 14.20 | 0.000164*** |
| BCP seed=100 (contam) | 200 | a=130 b=15 c=25 d=30 | 2.02 | 0.1547 |
| BCP gpt-4o (contam) | 200 | a=112 b=13 c=32 d=43 | 7.20 | 0.00729** |
| QASPER-RAG (contam) | 200 | a=93 b=1 c=22 d=84 | 17.39 | 3.042e-05*** |
| QASPER-Oracle (contam) | 200 | a=93 b=3 c=20 d=84 | 11.13 | 0.0008492*** |
| FinanceBench-LC (contam) | 150 | a=105 b=2 c=22 d=21 | 15.04 | 0.0001052*** |

### 2.2 Clean: AC vs HGC

| Test | n | a/b/c/d | χ² | p |
|---|---|---|---|---|
| BCP seed=42 (gpt-4.1) (clean) | 200 | a=168 b=12 c=9 d=11 | 0.19 | 0.6625 |
| BCP seed=7 (clean) | 200 | a=162 b=12 c=8 d=18 | 0.45 | 0.5023 |
| BCP seed=100 (clean) | 200 | a=150 b=14 c=11 d=25 | 0.16 | 0.6892 |
| BCP gpt-4o (clean) | 200 | a=139 b=13 c=12 d=36 | 0.00 | 1 |
| QASPER-RAG (clean) | 200 | a=113 b=3 c=4 d=80 | 0.00 | 1 |
| QASPER-Oracle (clean) | 200 | a=110 b=3 c=7 d=80 | 0.90 | 0.3428 |
| FinanceBench-LC (clean) | 150 | a=127 b=3 c=1 d=19 | 0.25 | 0.6171 |

### Bonferroni correction (n=14 tests)

| Test | raw p | Bonferroni-adjusted p | Significant at α=0.05? |
|---|---|---|---|
| BCP seed=42 (gpt-4.1) contam AC vs HGC | 0.04486 | 0.6281 | — |
| BCP seed=7 contam AC vs HGC | 0.000164 | 0.002296 | ✓ |
| BCP seed=100 contam AC vs HGC | 0.1547 | 1 | — |
| BCP gpt-4o contam AC vs HGC | 0.00729 | 0.1021 | — |
| QASPER-RAG contam AC vs HGC | 3.042e-05 | 0.0004259 | ✓ |
| QASPER-Oracle contam AC vs HGC | 0.0008492 | 0.01189 | ✓ |
| FinanceBench-LC contam AC vs HGC | 0.0001052 | 0.001472 | ✓ |
| BCP seed=42 (gpt-4.1) clean AC vs HGC | 0.6625 | 1 | — |
| BCP seed=7 clean AC vs HGC | 0.5023 | 1 | — |
| BCP seed=100 clean AC vs HGC | 0.6892 | 1 | — |
| BCP gpt-4o clean AC vs HGC | 1 | 1 | — |
| QASPER-RAG clean AC vs HGC | 1 | 1 | — |
| QASPER-Oracle clean AC vs HGC | 0.3428 | 1 | — |
| FinanceBench-LC clean AC vs HGC | 0.6171 | 1 | — |

## 3. Cost breakdown

| Experiment | Phase | n | avg_tokens | avg_gate_tokens | gate% | avg_time_s |
|---|---|---|---|---|---|---|
| BCP seed=42 (gpt-4.1) | P3-AC | 200 | 11,406 | 0 | 0.0% | 9.9 |
| BCP seed=42 (gpt-4.1) | P3-Hybrid | 200 | 11,584 | 250 | 2.2% | 12.4 |
| BCP seed=42 (gpt-4.1) | P4-AC | 200 | 11,349 | 0 | 0.0% | 12.2 |
| BCP seed=42 (gpt-4.1) | P4-Hybrid | 200 | 15,912 | 316 | 2.0% | 16.3 |
| BCP seed=7 | P3-AC | 200 | 11,122 | 0 | 0.0% | 11.5 |
| BCP seed=7 | P3-Hybrid | 200 | 12,429 | 233 | 1.9% | 16.0 |
| BCP seed=7 | P4-AC | 200 | 11,537 | 0 | 0.0% | 11.0 |
| BCP seed=7 | P4-Hybrid | 200 | 15,668 | 269 | 1.7% | 16.1 |
| BCP seed=100 | P3-AC | 200 | 14,291 | 0 | 0.0% | 14.3 |
| BCP seed=100 | P3-Hybrid | 200 | 15,638 | 172 | 1.1% | 18.2 |
| BCP seed=100 | P4-AC | 200 | 14,394 | 0 | 0.0% | 14.1 |
| BCP seed=100 | P4-Hybrid | 200 | 17,611 | 182 | 1.0% | 17.5 |
| BCP gpt-4o | P3-AC | 200 | 14,944 | 0 | 0.0% | 11.6 |
| BCP gpt-4o | P3-Hybrid | 200 | 17,677 | 235 | 1.3% | 23.4 |
| BCP gpt-4o | P4-AC | 200 | 14,333 | 0 | 0.0% | 13.1 |
| BCP gpt-4o | P4-Hybrid | 200 | 21,074 | 266 | 1.3% | 27.0 |
| QASPER-RAG | AC-RAG | 200 | 914 | 0 | 0.0% | 4.0 |
| QASPER-RAG | HGC-RAG | 200 | 1,639 | 618 | 37.7% | 5.6 |
| QASPER-RAG | C-AC-RAG | 200 | 912 | 0 | 0.0% | 3.8 |
| QASPER-RAG | C-HGC-RAG | 200 | 1,999 | 637 | 31.9% | 6.0 |
| QASPER-Oracle | AC-LC | 200 | 219 | 0 | 0.0% | 2.2 |
| QASPER-Oracle | HGC-LC | 200 | 415 | 181 | 43.6% | 2.5 |
| QASPER-Oracle | C-AC-LC | 200 | 218 | 0 | 0.0% | 1.9 |
| QASPER-Oracle | C-HGC-LC | 200 | 420 | 155 | 36.9% | 2.4 |
| FinanceBench-LC | AC-LC | 150 | 350 | 0 | 0.0% | 2.4 |
| FinanceBench-LC | HGC-LC | 150 | 1,401 | 888 | 63.4% | 3.1 |
| FinanceBench-LC | C-AC-LC | 150 | 351 | 0 | 0.0% | 1.9 |
| FinanceBench-LC | C-HGC-LC | 150 | 1,500 | 779 | 52.0% | 3.4 |

## 4. HGC path distribution (contaminated setting)

Paths: cache_verified (gate passed), cache_fallback (gate rejected), cache_miss (no AC hit), cache_hit_no_hint (AC hit but no location hints).

| Experiment | cache_verified | cache_fallback | cache_miss | cache_hit_no_hint |
|---|---|---|---|---|
| BCP seed=42 (gpt-4.1) | 72 (36%) | 37 (18%) | 66 (33%) | 24 (12%) |
| BCP seed=7 | 63 (32%) | 43 (22%) | 64 (32%) | 28 (14%) |
| BCP seed=100 | 52 (26%) | 26 (13%) | 81 (40%) | 40 (20%) |
| BCP gpt-4o | 67 (34%) | 32 (16%) | 82 (41%) | 19 (10%) |
| QASPER-RAG | 74 (37%) | 42 (21%) | 84 (42%) | 0 (0%) |
| QASPER-Oracle | 39 (20%) | 33 (16%) | 128 (64%) | 0 (0%) |
| FinanceBench-LC | 59 (39%) | 48 (32%) | 43 (29%) | 0 (0%) |

## 5. Gate reject reasons (cache_fallback breakdown)

| Experiment | no_location_hint_in_scope | no_hint_passed_all_components |
|---|---|---|
| BCP seed=42 (gpt-4.1) | 0 | 37 |
| BCP seed=7 | 0 | 43 |
| BCP seed=100 | 0 | 26 |
| BCP gpt-4o | 0 | 32 |
| QASPER-RAG | 0 | 42 |
| QASPER-Oracle | 0 | 33 |
| FinanceBench-LC | 0 | 48 |

## 6. Contamination-mode comparison (BCP seed=42)

Compare cross-swap / entity-swap / typo-mutation against P4-AC baseline.

| Contamination mode | HGC phase | Accuracy | AC drop baseline | HGC drop |
|---|---|---|---|---|
| cross-swap | P4-Hybrid | 82.0 | AC: -15.0 | HGC: -6.5 |
| entity-swap | E4-Hybrid | 83.0 | AC: -15.0 | HGC: -5.5 |
| typo-mutation | T4-Hybrid | 85.5 | AC: -15.0 | HGC: -3.0 |
