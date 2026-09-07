"""The released tree renames results/ to trajectories/; both must resolve.

Analysis scripts used to hard-code ``results/``, so every one of them reported
"not run" for cells that were present in the supplementary, under the other
name. These tests pin the fallback so that regression cannot return silently --
its symptom is empty output, not an error.
"""

from __future__ import annotations

from hgc.paths import results_root


def test_prefers_results_when_present(tmp_path):
    (tmp_path / "results").mkdir()
    assert results_root(tmp_path) == tmp_path / "results"


def test_falls_back_to_trajectories(tmp_path):
    """The layout a reader unpacks from the release."""
    (tmp_path / "trajectories").mkdir()
    assert results_root(tmp_path) == tmp_path / "trajectories"


def test_results_wins_when_both_exist(tmp_path):
    """A working repo that also holds an unpacked release keeps using its own runs."""
    (tmp_path / "results").mkdir()
    (tmp_path / "trajectories").mkdir()
    assert results_root(tmp_path) == tmp_path / "results"


def test_defaults_to_results_when_neither_exists(tmp_path):
    """A fresh checkout gets the path a run would create."""
    assert results_root(tmp_path) == tmp_path / "results"


def test_accepts_a_string_root(tmp_path):
    (tmp_path / "trajectories").mkdir()
    assert results_root(str(tmp_path)) == tmp_path / "trajectories"
