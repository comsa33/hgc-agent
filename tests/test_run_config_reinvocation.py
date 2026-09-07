"""A second ``run.py`` invocation into an existing out-dir must pass the config check.

The chained scripts add phases to an out-dir one invocation at a time (P3-Hybrid
after P4-Hybrid and A-Hybrid, say). ``validate_compatible`` compares ``run_env``
too, and the freshly built config used to leave it empty until ``save()`` filled
it in, so every re-invocation into a directory with a saved config failed with a
``run_env`` mismatch. The check itself is wanted: it is what keeps a two-stage
phase out of a released-gate directory.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import pytest

_RUN = Path(__file__).resolve().parents[1] / "experiments" / "run.py"


@pytest.fixture(scope="module")
def run_module():
    spec = importlib.util.spec_from_file_location("hgc_run_py", _RUN)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["hgc_run_py"] = mod
    spec.loader.exec_module(mod)
    return mod


def _args(**overrides) -> argparse.Namespace:
    base = dict(dataset="bcp", model="gpt-4.1", seed=42, n=200, phases="P3-Hybrid")
    base.update(overrides)
    return argparse.Namespace(**base)


def test_reinvocation_with_the_same_knobs_passes(tmp_path, run_module, monkeypatch):
    monkeypatch.setenv("HGC_GATE_VARIANT", "full")
    monkeypatch.setenv("HGC_CONTAMINATION_FRACTION", "0.2")
    cfg, nothing = run_module.load_or_create_config(tmp_path, _args())
    assert not nothing
    cfg.phases_completed = ["P4-Hybrid"]
    cfg.save(tmp_path)  # fills run_env from the environment, as the real run does

    again, nothing = run_module.load_or_create_config(tmp_path, _args())
    assert not nothing
    assert again.phases_completed == ["P4-Hybrid"]
    assert again.run_env["HGC_GATE_VARIANT"] == "full"


def test_reinvocation_under_different_knobs_is_refused(tmp_path, run_module, monkeypatch):
    monkeypatch.setenv("HGC_GATE_VARIANT", "full")
    cfg, _ = run_module.load_or_create_config(tmp_path, _args())
    cfg.save(tmp_path)

    monkeypatch.setenv("HGC_VERIFIER_PREDICATE", "two_stage")
    with pytest.raises(ValueError, match="run_env"):
        run_module.load_or_create_config(tmp_path, _args())


def test_p3_gets_its_own_store_and_p1_is_left_alone(tmp_path, run_module):
    p1 = tmp_path / "P1" / "memory.db"
    p1.parent.mkdir()
    p1.write_bytes(b"pristine")
    db = run_module._own_store(tmp_path, "P3-Hybrid", p1)
    assert db == tmp_path / "P3-Hybrid" / "memory.db"
    assert db.read_bytes() == b"pristine"
    db.write_bytes(b"grown by the run")
    assert p1.read_bytes() == b"pristine"
    # A second call must not overwrite what the phase has written.
    assert run_module._own_store(tmp_path, "P3-Hybrid", p1).read_bytes() == b"grown by the run"
