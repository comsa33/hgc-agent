"""G2: docid presence check for a single location hint.

Accepts hint content in either ``"docid=NNN"`` or bare-digit form. Returns
True when either (a) the hint is not a docid hint (let G3 decide), or (b) the
parsed docid is present in the current query's document pool.
"""

from __future__ import annotations

import re

from hgc.memory import HintRecord

_DOCID_INLINE_RE = re.compile(r"docid=(\d+)")


def parse_docid(content: str) -> str | None:
    """Extract a numeric docid from a hint content string.

    Accepts three observed shapes:
    - bare digits, e.g. ``"63970"`` (current production format)
    - ``"docid=63970"`` (legacy / fixture form)
    - any string containing ``docid=NNN`` as a substring
    """
    s = content.strip()
    if s.isdigit():
        return s
    m = _DOCID_INLINE_RE.search(s)
    return m.group(1) if m else None


def resolve_docid(content: str, doc_map: dict[str, str]) -> str | None:
    """Resolve a location hint's content to a key of *doc_map*, else None.

    Two shapes reach this function:
    - string docids minted straight from the doc the backbone consumed
      (``"qasper_1909.00015_q0_oracle"``, ``"AMAZON_2019_10K_page_37"``),
      which are doc_map keys verbatim; and
    - numeric docids extracted by the hint extractor on BCP (``"63970"``).

    Checking membership first means the string shape resolves without going
    through the digit parser, which never matched it.
    """
    s = content.strip()
    if s in doc_map:
        return s
    docid = parse_docid(s)
    return docid if docid is not None and docid in doc_map else None


class DocidCheck:
    """Verify that a location hint's docid is in the current doc pool.

    Returns True for non-location hints. For location hints the content is
    resolved against ``doc_map`` (see :func:`resolve_docid`) and the check
    passes when it resolves.

    When ``doc_map`` is empty the pool is unknown, so the hint is passed
    through to G3 for content verification rather than rejected.

    Note on scope: before the string-docid path was added, only bare-digit
    content could resolve, so G2 was a no-op on every non-BCP cell — 0/200
    QASPER-Oracle, 0/187 FinanceBench and 0/987 QASPER-RAG location hints
    parse as digits, against 1253/1257 on BCP. Contaminated hints there carry
    a ``__CORRUPTED_nnnn`` suffix and so fall out of the pool; catching them
    at G2 is what makes the G3-off ablation meaningful on those cells.
    """

    def __call__(self, hint: HintRecord, doc_map: dict[str, str]) -> bool:
        if hint.hint_type != "location":
            return True
        if not doc_map:
            return True
        return resolve_docid(hint.content, doc_map) is not None
