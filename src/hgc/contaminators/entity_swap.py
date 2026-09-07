"""Entity-swap contamination for AnswerCache and HintStore.

Within each selected entry's answer (or hint content), detect a named-entity
substring and replace it with a similar-typed entity drawn from a fallback
bank. The rest of the text is preserved, so the answer remains semantically
close to the original but factually wrong — a fine-grained hallucination
model distinct from the blunt cross-swap mode.

Light-weight NER: we use a regex-based proper-noun detector keyed on
capitalised runs and known pattern categories (person names, locations,
organisations, dates). This keeps the dependency surface minimal and
deterministic across seeds.
"""

from __future__ import annotations

import math
import random
import re
from collections.abc import Callable
from pathlib import Path as _Path
from typing import Any as _Any

from hgc.memory import HintStore

# A conservative capitalised-run detector: one or more capitalised words,
# optionally chained. This is intentionally loose — for research contamination
# we only need to find *some* swappable token, not perfectly bounded entities.
_CAPWORD_RE = re.compile(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3}\b")

_FALLBACK_ENTITIES: list[str] = [
    "University of Edinburgh",
    "Imperial College London",
    "Rutgers University",
    "Technische Universität Berlin",
    "Université Paris-Saclay",
    "Osaka University",
    "University of Toronto",
    "Carnegie Mellon University",
    "ETH Zürich",
    "Seoul National University",
    "London",
    "Lisbon",
    "Berlin",
    "Tokyo",
    "Helsinki",
    "Vienna",
    "Warsaw",
    "Edinburgh",
    "Brussels",
    "Oslo",
]


def _swap_first_entity(text: str, rng: random.Random, bank: list[str]) -> str | None:
    """Return *text* with its first capitalised run replaced by a bank entry.

    Returns None when *text* contains no capitalised run (caller may then skip).
    """
    m = _CAPWORD_RE.search(text)
    if not m:
        return None
    original = m.group(0)
    candidates = [v for v in bank if v != original]
    if not candidates:
        return None
    replacement = rng.choice(candidates)
    return text[: m.start()] + replacement + text[m.end() :]


# ---------------------------------------------------------------------------
# HintStore contamination
# ---------------------------------------------------------------------------


def corrupt_hint_store_entity_swap(
    store: HintStore,
    *,
    seed: int = 42,
    fraction: float = 0.20,
) -> list[str]:
    """Entity-swap contamination on positive hints in *store*.

    For each selected hint we replace the first capitalised run in its
    ``content`` with a fallback entity. Hints whose content carries no
    swappable entity (e.g. bare-digit location hints) are skipped at
    selection: the contaminator works on hints with parseable capitalised
    substrings.

    Returns the list of corrupted ``hint_id`` values (deterministic for a
    given seed).
    """
    positives = [h for h in store.all() if h.polarity == "positive"]
    swappable = [h for h in positives if _CAPWORD_RE.search(h.content)]
    n_corrupt = math.floor(len(swappable) * fraction)
    if n_corrupt == 0:
        return []

    rng = random.Random(seed)
    selected = rng.sample(swappable, n_corrupt)

    corrupted_ids: list[str] = []
    for hint in selected:
        new_content = _swap_first_entity(hint.content, rng, _FALLBACK_ENTITIES)
        if new_content is None or new_content == hint.content:
            continue
        store.update_content(hint.hint_id, new_content)
        corrupted_ids.append(hint.hint_id)
    return corrupted_ids


# ---------------------------------------------------------------------------
# AnswerCache contamination
# ---------------------------------------------------------------------------


def corrupt_answer_cache_entity_swap(
    ac_agent: _Any,
    p1_trajectories: list[_Path],
    embedder: Callable[[str], _Any],
    *,
    seed: int = 42,
    fraction: float = 0.20,
) -> list[int]:
    """Entity-swap contamination on AnswerCacheAgent._cache.

    Seeds from P1 trajectories first (keeping the embeddings intact) and
    then replaces the first capitalised run in ``floor(fraction * N)``
    victim answers with a bank entry. Embedding → same question, answer →
    semantically close but wrong.
    """
    from hgc.runner import seed_answer_cache_from_p1

    seed_answer_cache_from_p1(ac_agent, p1_trajectories, embedder)

    n = len(ac_agent._cache)
    if n < 2:
        raise ValueError("cannot entity-swap with fewer than 2 cache entries")

    swappable = [
        i for i, (_, ans, *_) in enumerate(ac_agent._cache) if _CAPWORD_RE.search(ans)
    ]
    k = math.floor(len(swappable) * fraction)
    if k == 0:
        return []

    rng = random.Random(seed)
    victim_indices: list[int] = rng.sample(swappable, k)

    for i in victim_indices:
        row = list(ac_agent._cache[i])
        original_answer = row[1]
        new_answer = _swap_first_entity(original_answer, rng, _FALLBACK_ENTITIES)
        if new_answer is None or new_answer == original_answer:
            continue
        row[1] = new_answer
        ac_agent._cache[i] = tuple(row)

    return sorted(victim_indices)
