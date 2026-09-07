# Independent grounding checkers

Scores the gate-aware poison against four public checkers, to separate a
property of our verifier from a property of grounding verification. Reported in
the appendix section on independent grounding checkers.

## Files

| | |
|---|---|
| `triplets.jsonl` | 713 rows: one clean, one cross-swap and one gate-aware answer per seeded cache entry, rebuilt offline from the P1 trajectories |
| `scores_hhem.csv`, `scores_lettuce.csv`, `scores_nli.csv`, `scores_msmarco.csv` | raw per-row scores, one row per (triplet, model, question variant) |
| `analyze.py` | reduces those scores to every figure the appendix quotes |

## Reproducing

    uv run python analysis/grounding_checkers/analyze.py     # table from the shipped scores

Rescoring from scratch is optional and needs no API key; all four checkers run
on CPU. The checkers pin incompatible dependency sets, so each gets its own
environment:

    uv run python experiments/grounding_checkers/build_triplets.py

    uv venv .venv-hhem && uv pip install --python .venv-hhem \
        'transformers==4.56.2' torch sentencepiece
    .venv-hhem/bin/python experiments/grounding_checkers/run_hhem.py

    uv run python experiments/grounding_checkers/run_nli_msmarco.py

    uv venv .venv-lettuce && uv pip install --python .venv-lettuce lettucedetect
    .venv-lettuce/bin/python experiments/grounding_checkers/run_lettuce.py

Weights download on first use (438MB, 369MB, 91MB, and 598MB/274MB for the two
LettuceDetect sizes); the sweep takes roughly 80 minutes on a laptop CPU.

## Reading the scores

`regime` is `clean`, `random` (cross-swap) or `targeted` (a sentence copied
verbatim from the victim's own source document). `qvariant` is `orig` or
`shuffled`, the latter being the same row re-scored with an unrelated question
from the same benchmark: the control for whether the question input affects the
verdict. `cand_in_retained` is 0 when the 512-token window cut the candidate out
of the evidence, which makes that row's verdict uninterpretable; the appendix
figures drop those rows. `cand_words` is the candidate length, filtered at seven
words so that short spans cannot be dismissed as trivially unmatchable.
