"""
Split_TEXT.py

Utilities to split a Markdown-ish Portuguese government bulletin into two parts:
- Sumario (summary section)
- Body (rest)

Split rule (as specified):
1) Find the last entity labeled "Sumario".
2) After that, find the first org-like entity (label in {"ORG_LABEL", "ORG_WITH_STAR_LABEL"}).
   Normalize its text to a letters-only key and store it as the boundary key.
3) Continue scanning org-like entities in order. When an org appears whose normalized text
   equals the stored key, that org's start marks (a provisional) beginning of the body.
   If the key appears inside a larger org span, accept it and take the BEGINNING of the inner match.
4) Final boundary adjustment: shift the cut to the **last empty line** before that provisional boundary
   (searching only between the Sumario end and the provisional boundary). If none is found, keep the
   provisional boundary.
5) sumario = text[sumario_end_char : final_boundary]
   body    = text[final_boundary : ]

Edge cases:
- No Sumario entity: (sumario=None, body=original text), meta.reason="no_sumario"
- Sumario found but no org-like after it: sumario=text[sumario_end:], body="", meta.reason="no_org_after_sumario"
- Org-like(s) after Sumario but no repeat encountered: sumario=text[sumario_end:], body="", meta.reason="no_repeat_match"

Normalization policy: letters-only
- NFKD → strip accents → keep only Unicode letters → casefold
- Drops all non-letters (e.g., '*', '_', '`', digits, punctuation, '&', spaces)

Notes:
- Selects the **last** Sumario (right-to-left scan).
- Boundary detection supports exact match and embedded match (letters-only).
- Final cut is aligned to the **last empty line** before the detected boundary when available.
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple, Any
import unicodedata
import re

# Type aliases
SplitResult = Tuple[Optional[str], str, Dict[str, Any]]

ORG_LIKE_LABELS = {"ORG_LABEL", "ORG_WITH_STAR_LABEL"}
SUMARIO_LABEL = "Sumario"


def _normalize_for_match_letters_only(s: str) -> str:
    """Normalize a string for matching org names using letters-only semantics."""
    if s is None:
        return ""
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = "".join(ch for ch in s if ch.isalpha())
    return s.casefold()


def _safe_label(ent) -> Optional[str]:
    try:
        return ent.label_ if hasattr(ent, "label_") else getattr(ent, "label", None)
    except Exception:
        return None


def _is_org_like(ent) -> bool:
    return _safe_label(ent) in ORG_LIKE_LABELS


def _is_sumario(ent) -> bool:
    return _safe_label(ent) == SUMARIO_LABEL


def _letters_only_with_index_map(s: str, base_start_char: int):
    """Build letters-only stream for s and a map from stream index -> original char index."""
    letters = []
    index_map = []
    for rel_i, ch in enumerate(s):
        abs_i = base_start_char + rel_i
        for base in unicodedata.normalize("NFKD", ch):
            if not unicodedata.combining(base):
                if base.isalpha():
                    letters.append(base.casefold())
                    index_map.append(abs_i)
    return "".join(letters), index_map


_EMPTY_LINE_RE = re.compile(r"(?m)^[ \t]*\r?\n[ \t]*\r?\n")


def _last_empty_line_before(text: str, start_limit: int, end_limit: int) -> Optional[int]:
    """
    Return the start index of the last empty line strictly before end_limit,
    but not before start_limit. An "empty line" is a blank line boundary:
    one newline followed by optional whitespace and another newline.
    """
    if end_limit <= start_limit:
        return None

    # We search in the window [start_limit, end_limit)
    window = text[start_limit:end_limit]
    last_start = None
    for m in _EMPTY_LINE_RE.finditer(window):
        # The cut point should be at the start of the empty-line block in the original text
        last_start = start_limit + m.start()
    return last_start


def split_sumario_and_body(doc, text: Optional[str] = None, debug: bool = False) -> SplitResult:
    """Split the original text into (sumario, body) using spaCy entities and the given rule."""
    meta: Dict[str, Any] = {
        "reason": None,
        "sumario_ent_start": None,
        "sumario_ent_end": None,
        "first_org_raw": None,
        "first_org_norm": None,
        "boundary_org_raw": None,
        "boundary_org_norm": None,
        "boundary_org_start": None,
        "boundary_adjusted_to_empty_line": False,
    }

    if text is None:
        text = doc.text

    # 1) Find LAST Sumario entity (right-to-left)
    sumario_ent = None
    for ent in reversed(list(getattr(doc, "ents", ()))):
        if _is_sumario(ent):
            sumario_ent = ent
            break

    if sumario_ent is None:
        meta["reason"] = "no_sumario"
        return None, text, meta

    meta["sumario_ent_start"] = int(sumario_ent.start_char)
    meta["sumario_ent_end"] = int(sumario_ent.end_char)

    # 2) Orgs after Sumario
    seen_sumario_end = sumario_ent.end_char
    orgs_after = [
        ent for ent in getattr(doc, "ents", ())
        if getattr(ent, "start_char", 0) >= seen_sumario_end and _is_org_like(ent)
    ]

    if not orgs_after:
        meta["reason"] = "no_org_after_sumario"
        sumario_text = text[seen_sumario_end:]
        return sumario_text, "", meta

    first_org = orgs_after[0]
    first_org_raw = first_org.text
    first_org_norm = _normalize_for_match_letters_only(first_org_raw)
    meta["first_org_raw"] = first_org_raw
    meta["first_org_norm"] = first_org_norm

    # 3) Scan subsequent org-like entities for the first repeat
    #    - Exact match: boundary at ent.start_char
    #    - Embedded match: boundary at BEGINNING of inner match within the larger span
    boundary_ent = None
    provisional_boundary = None  # before empty-line adjustment

    for ent in orgs_after[1:]:
        raw = ent.text
        cur_norm = _normalize_for_match_letters_only(raw)

        if debug:
            print(f"[ORG SCAN] first={first_org_norm!r} vs cur={cur_norm!r} | raw={raw!r}")

        # Case A) exact match -> boundary at entity start
        if cur_norm == first_org_norm:
            boundary_ent = ent
            provisional_boundary = int(ent.start_char)
            break

        # Case B) key embedded inside this larger org span -> boundary at BEGINNING of inner match
        if first_org_norm and cur_norm and (first_org_norm in cur_norm):
            stream, idx_map = _letters_only_with_index_map(raw, int(ent.start_char))
            pos = stream.find(first_org_norm)
            if pos != -1:
                boundary_ent = ent
                provisional_boundary = int(idx_map[pos])
                break

    if boundary_ent is None:
        meta["reason"] = "no_repeat_match"
        sumario_text = text[seen_sumario_end:]
        return sumario_text, "", meta

    # 4) Adjust boundary to the last empty line before the provisional boundary
    adjusted = _last_empty_line_before(text, seen_sumario_end, provisional_boundary)
    final_boundary = provisional_boundary if adjusted is None else adjusted
    meta["boundary_adjusted_to_empty_line"] = adjusted is not None

    # 5) Slice
    meta["boundary_org_raw"] = boundary_ent.text
    meta["boundary_org_norm"] = _normalize_for_match_letters_only(boundary_ent.text)
    meta["boundary_org_start"] = final_boundary

    sumario_text = text[seen_sumario_end:final_boundary]
    body_text = text[final_boundary:]

    return sumario_text, body_text, meta


__all__ = [
    "split_sumario_and_body",
    "_normalize_for_match_letters_only",
    "ORG_LIKE_LABELS",
    "SUMARIO_LABEL",
]
