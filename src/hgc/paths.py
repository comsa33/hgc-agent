"""Locate the per-query trajectories, in the repo or in the released tree.

The working repository writes runs to ``results/``. The anonymous supplementary
ships the same directories under ``trajectories/``, because that is what they
are to a reader who did not run them. Analysis scripts hard-coded ``results/``
and so returned "not run" for every cell once unpacked from the release --
present data, reported as absent.

Resolving the name once, here, keeps every script working from either layout.
"""

from __future__ import annotations

from pathlib import Path

# Checked in order; the first that exists wins.
_CANDIDATES = ("results", "trajectories")


def results_root(repo_root: Path | str) -> Path:
    """Return the directory holding per-run trajectory folders.

    Falls back to ``results`` when neither exists, so a fresh checkout that has
    not run anything yet still gets the path a run would create.
    """
    root = Path(repo_root)
    for name in _CANDIDATES:
        candidate = root / name
        if candidate.is_dir():
            return candidate
    return root / _CANDIDATES[0]
