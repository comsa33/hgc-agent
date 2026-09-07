"""Typo-mutation contamination for AnswerCache and HintStore.

Low-level character noise: for each selected entry's answer (or hint
content), mutate ~10% of characters using three operations:
- substitution with an adjacent QWERTY letter,
- transposition with the next character,
- deletion of a single character.

Tests whether the gate tolerates syntactic noise from OCR / transcription /
flaky-pipeline scenarios. Deterministic given seed + fraction + ratio.
"""

from __future__ import annotations

import math
import random
from collections.abc import Callable
from pathlib import Path as _Path
from typing import Any as _Any

from hgc.memory import HintStore

# QWERTY adjacency map (lower-case). Upper/non-letter chars are pass-through.
_ADJ: dict[str, str] = {
    "q": "wa",
    "w": "qeas",
    "e": "wrds",
    "r": "etdf",
    "t": "ryfg",
    "y": "tugh",
    "u": "yihj",
    "i": "uojk",
    "o": "ipkl",
    "p": "ol",
    "a": "qwsz",
    "s": "awedxz",
    "d": "serfcx",
    "f": "drtgvc",
    "g": "ftyhbv",
    "h": "gyujnb",
    "j": "huiknm",
    "k": "jiolm",
    "l": "kop",
    "z": "asx",
    "x": "zsdc",
    "c": "xdfv",
    "v": "cfgb",
    "b": "vghn",
    "n": "bhjm",
    "m": "njk",
}


def _mutate(text: str, rng: random.Random, rate: float = 0.10) -> str:
    """Return *text* with a fraction *rate* of its characters perturbed."""
    if not text:
        return text
    chars = list(text)
    n = len(chars)
    k = max(1, math.floor(n * rate))
    indices = sorted(rng.sample(range(n), min(k, n)))
    offset = 0
    for idx in indices:
        pos = idx + offset
        if pos < 0 or pos >= len(chars):
            continue
        c = chars[pos]
        c_lower = c.lower()
        op = rng.choice(["sub", "swap", "delete"])
        if op == "sub" and c_lower in _ADJ:
            repl = rng.choice(list(_ADJ[c_lower]))
            chars[pos] = repl.upper() if c.isupper() else repl
        elif op == "swap" and pos + 1 < len(chars):
            chars[pos], chars[pos + 1] = chars[pos + 1], chars[pos]
        elif op == "delete":
            del chars[pos]
            offset -= 1
    return "".join(chars)


# ---------------------------------------------------------------------------
# HintStore contamination
# ---------------------------------------------------------------------------


def corrupt_hint_store_typo(
    store: HintStore,
    *,
    seed: int = 42,
    fraction: float = 0.20,
    char_rate: float = 0.10,
) -> list[str]:
    """Typo-mutation contamination on positive hints.

    Applies character-level noise to ``floor(fraction * N_positive)``
    randomly selected positive hints. Pure-digit content (bare docid) is
    skipped so that location hints remain parseable — typo on docids would
    collapse the docid parser rather than test the LLM verifier, which is
    the intended target of this contamination model. Use cross-swap for
    location-hint corruption.
    """
    positives = [
        h
        for h in store.all()
        if h.polarity == "positive" and not h.content.strip().isdigit()
    ]
    n_corrupt = math.floor(len(positives) * fraction)
    if n_corrupt == 0:
        return []

    rng = random.Random(seed)
    selected = rng.sample(positives, n_corrupt)

    corrupted_ids: list[str] = []
    for hint in selected:
        new_content = _mutate(hint.content, rng, rate=char_rate)
        if new_content == hint.content:
            continue
        store.update_content(hint.hint_id, new_content)
        corrupted_ids.append(hint.hint_id)
    return corrupted_ids


# ---------------------------------------------------------------------------
# AnswerCache contamination
# ---------------------------------------------------------------------------


def corrupt_answer_cache_typo(
    ac_agent: _Any,
    p1_trajectories: list[_Path],
    embedder: Callable[[str], _Any],
    *,
    seed: int = 42,
    fraction: float = 0.20,
    char_rate: float = 0.10,
) -> list[int]:
    """Typo-mutation contamination on AnswerCacheAgent._cache."""
    from hgc.runner import seed_answer_cache_from_p1

    seed_answer_cache_from_p1(ac_agent, p1_trajectories, embedder)

    n = len(ac_agent._cache)
    if n < 2:
        raise ValueError("cannot typo-mutate with fewer than 2 cache entries")

    k = math.floor(n * fraction)
    if k == 0:
        return []

    rng = random.Random(seed)
    victim_indices: list[int] = rng.sample(range(n), k)

    for i in victim_indices:
        row = list(ac_agent._cache[i])
        original_answer = row[1]
        new_answer = _mutate(original_answer, rng, rate=char_rate)
        if new_answer == original_answer:
            continue
        row[1] = new_answer
        ac_agent._cache[i] = tuple(row)

    return sorted(victim_indices)
