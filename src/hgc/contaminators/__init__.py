"""HGC contaminators.

Three contamination modes over AnswerCache and HintStore:
- cross_swap : replace answer with a different entry's answer (blunt swap)
- entity_swap: replace one named entity inside an answer (fine hallucination)
- typo_mutation: character-level noise (low-level syntactic corruption)

The modules are kept narrow so that additional modes (poison-append,
stale-answer, etc.) can slot in without touching existing code paths.
"""

from hgc.contaminators.cross_swap import (
    _BARE_DIGIT_RE,
    _DOCID_RE,
    _FALLBACK_DOCIDS,
    HintContaminator,
    _is_docid_content,
)
from hgc.contaminators.cross_swap import (
    contaminate_answer_cache_from_p1 as corrupt_answer_cache_cross_swap,
)
from hgc.contaminators.entity_swap import (
    corrupt_answer_cache_entity_swap,
    corrupt_hint_store_entity_swap,
)
from hgc.contaminators.typo_mutation import (
    corrupt_answer_cache_typo,
    corrupt_hint_store_typo,
)

__all__ = [
    "HintContaminator",
    "corrupt_answer_cache_cross_swap",
    "corrupt_answer_cache_entity_swap",
    "corrupt_answer_cache_typo",
    "corrupt_hint_store_entity_swap",
    "corrupt_hint_store_typo",
    "_is_docid_content",
    "_BARE_DIGIT_RE",
    "_DOCID_RE",
    "_FALLBACK_DOCIDS",
]
