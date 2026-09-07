"""smoke_common — re-export alias for hgc.factories.

All factory builders and helper functions live in hgc.factories.
This shim lets the experiment runner use the legacy ``hgc.smoke_common``
import path without code duplication.
"""

# Re-export everything from factories so callers can do:
#   from hgc.smoke_common import build_p4_hybrid_factory, phase_summary, ...
from hgc.factories import *  # noqa: F401, F403
