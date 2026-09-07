"""ExperimentConfig dataclass + standard results layout helpers (US-024).

Provides a stable, serialisable configuration object that is written to
results/{experiment_id}/config.json on first run and validated on subsequent
runs so that incompatible configurations are rejected early.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _slug(s: str) -> str:
    """Safe directory segment: lowercase, alphanumeric + dashes only."""
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")


def experiment_id(dataset: str, model: str, seed: int) -> str:
    """Return a stable, slug-safe experiment ID.

    Examples
    --------
    >>> experiment_id("bcp", "gpt-4.1", 42)
    'bcp_gpt41_seed42'
    >>> experiment_id("hotpot", "gpt-4o", 7)
    'hotpot_gpt4o_seed7'
    """
    model_short = _slug(model).replace("-", "").replace(".", "")
    return f"{_slug(dataset)}_{model_short}_seed{int(seed)}"


def detect_git_commit() -> str | None:
    """Return the short HEAD commit hash, or None if not in a git repo."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except FileNotFoundError:
        pass
    return None


# ---------------------------------------------------------------------------
# Dataclass
# ---------------------------------------------------------------------------


# Environment variables that change experiment behaviour and therefore belong
# in the run record. Credentials are deliberately absent — this is written to
# config.json, which ships in the supplementary archive.
_TRACKED_ENV = (
    "HGC_GATE_VARIANT",
    "HGC_CONTAMINATION_FRACTION",
    "HGC_GATE_LM",
    "HGC_STRICT_NO_HINT",
    "HGC_CACHE_SIM_THRESHOLD",
    "HGC_TOP_K_HINTS",
    "HGC_VERIFIER_MAX_DOC_CHARS",
    "HGC_VERIFIER_CONTAINMENT_MIN_CHARS",
    # The long-context and RAG backbones read their deployment straight from
    # the environment, so ``model_main`` does not identify them.
    "AZURE_OPENAI_DEPLOYMENT",
    "AZURE_OPENAI_JUDGE_DEPLOYMENT",
    "OLLAMA_CLOUD_BASE_URL",
    "HGC_VERIFIER_PREDICATE",
)


def capture_run_env() -> dict:
    """Return the tracked environment variables that are currently set."""
    return {key: os.environ[key] for key in _TRACKED_ENV if key in os.environ}


@dataclass
class ExperimentConfig:
    """Serialisable configuration for a single experiment run."""

    experiment_id: str
    dataset: str
    model_main: str
    model_sub: str
    model_embedding: str
    seed: int
    n_queries: int
    notes: str = ""
    created_at: str = ""
    updated_at: str = ""
    phases_completed: list = field(default_factory=list)
    git_commit: str | None = None
    # Snapshot of the environment variables that change what an experiment
    # does. Without this there is no record of which gate variant, threshold
    # or deployment a directory was produced under — config.json named the
    # dataset and seed, and nothing else. Treated as immutable so re-entering
    # a directory under different knobs fails loudly instead of resuming into
    # a silent mix of two configurations.
    run_env: dict = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, out_dir: Path) -> None:
        """Atomically write this config to *out_dir*/config.json."""
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        now = datetime.now(UTC).isoformat()
        if not self.created_at:
            self.created_at = now
        self.updated_at = now
        if not self.run_env:
            self.run_env = capture_run_env()

        tmp = out_dir / "config.json.tmp"
        tmp.write_text(json.dumps(asdict(self), indent=2))
        os.replace(tmp, out_dir / "config.json")

    @classmethod
    def load(cls, out_dir: Path) -> ExperimentConfig:
        """Load config from *out_dir*/config.json; raises FileNotFoundError if absent."""
        p = Path(out_dir) / "config.json"
        if not p.exists():
            raise FileNotFoundError(str(p))
        data = json.loads(p.read_text())
        return cls(**data)

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate_compatible(self, other: ExperimentConfig) -> None:
        """Raise ValueError if any immutable field differs between *self* and *other*.

        Mutable fields (allowed to differ): n_queries, phases_completed,
        notes, created_at, updated_at, git_commit.
        """
        _MUTABLE = {
            "n_queries",
            "phases_completed",
            "notes",
            "created_at",
            "updated_at",
            "git_commit",
        }
        for key in asdict(self):
            if key in _MUTABLE:
                continue
            self_val = getattr(self, key)
            other_val = getattr(other, key)
            if self_val != other_val:
                raise ValueError(f"Config field '{key}' mismatch: {self_val!r} vs {other_val!r}")
