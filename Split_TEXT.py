"""
Split_TEXT.py

Utilities to split a Markdown-ish Portuguese government bulletin into two parts:
- Sumario (summary section)
- Body (rest)

Split rule (as specified):
1) Find the first entity labeled "Sumario".
2) After that, find the first org-like entity (label in {"ORG_LABEL", "ORG_WITH_STAR_LABEL"}).
   Normalize its text to a letters-only key and store it as the boundary key.
3) Continue scanning org-like entities in order. When an org appears whose normalized text
   equals the stored key, that org's start marks the beginning of the body.
4) sumario = text[sumario_end_char : boundary_org_start_char]
   body    = text[boundary_org_start_char : ]

Edge cases (strict to the stated rule):
- No Sumario entity: (sumario=None, body=original text), meta.reason="no_sumario"
- Sumario found but no org-like after it: sumario=text[sumario_end:], body="", meta.reason="no_org_after_sumario"
- Org-like(s) after Sumario but no repeat encountered: sumario=text[sumario_end:], body="", meta.reason="no_repeat_match"

Normalization policy: letters-only
- NFKD → strip accents → keep only Unicode letters → casefold
- Drops all non-letters (e.g., '*', '_', '`', digits, punctuation, '&', spaces)

This module is pure-logic; it does not depend on your project layout beyond receiving a spaCy Doc and the original text.
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple, Any
import unicodedata

# Type aliases
SplitResult = Tuple[Optional[str], str, Dict[str, Any]]


ORG_LIKE_LABELS = {"ORG_LABEL", "ORG_WITH_STAR_LABEL"}
SUMARIO_LABEL = "Sumario"


def _normalize_for_match_letters_only(s: str) -> str:
    """Normalize a string for matching org names using letters-only semantics.

    Steps:
      1) Unicode normalize (NFKD)
      2) Drop combining marks (accents)
      3) Keep only Unicode letters
      4) Casefold
    """
    if s is None:
        return ""
    # 1) NFKD
    s = unicodedata.normalize("NFKD", s)
    # 2) Drop accents (combining marks)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    # 3) Keep only letters
    s = "".join(ch for ch in s if ch.isalpha())
    # 4) Casefold
    return s.casefold()


def _is_org_like(ent) -> bool:
    """Return True if the entity label denotes an organization-like unit for our rule."""
    label = getattr(ent, "label_", None)
    return label in ORG_LIKE_LABELS


def _is_sumario(ent) -> bool:
    try:
        label = ent.label_ if hasattr(ent, "label_") else getattr(ent, "label", None)
    except Exception:
        label = None
    return label == SUMARIO_LABEL


def split_sumario_and_body(doc, text: Optional[str] = None, debug: bool = False) -> SplitResult:
    """Split the original text into (sumario, body) using spaCy entities and the given rule.

    Parameters
    ----------
    doc : spacy.tokens.Doc
        The spaCy Doc produced from the original text.
    text : str
        The original raw text used to produce `doc`.

    Returns
    -------
    (sumario, body, meta) : Tuple[Optional[str], str, Dict[str, Any]]
        - sumario: the substring representing the summary section (None if Sumario not found)
        - body: the substring representing the body (may be empty if no boundary found)
        - meta: diagnostic data with keys like:
            {
                "reason": Optional[str],
                "sumario_ent_start": int,
                "sumario_ent_end": int,
                "first_org_raw": Optional[str],
                "first_org_norm": Optional[str],
                "boundary_org_raw": Optional[str],
                "boundary_org_norm": Optional[str],
                "boundary_org_start": Optional[int],
            }
    """
    meta: Dict[str, Any] = {
        "reason": None,
        "sumario_ent_start": None,
        "sumario_ent_end": None,
        "first_org_raw": None,
        "first_org_norm": None,
        "boundary_org_raw": None,
        "boundary_org_norm": None,
        "boundary_org_start": None,
    }

    # If no explicit text was provided, use the Doc's original text
    if text is None:
        text = doc.text


    # 1) Find first Sumario entity
    sumario_ent = None
    for ent in reversed(list(doc.ents)):
        if _is_sumario(ent):
            sumario_ent = ent
            break

    if sumario_ent is None:
        # No Sumario → sumario=None, body=whole text
        meta["reason"] = "no_sumario"
        return None, text, meta

    meta["sumario_ent_start"] = int(sumario_ent.start_char)
    meta["sumario_ent_end"] = int(sumario_ent.end_char)

    # 2) From after Sumario, collect org-like entities in order
    seen_sumario_end = sumario_ent.end_char
    orgs_after = [ent for ent in doc.ents if getattr(ent, 'start_char', 0) >= seen_sumario_end and _is_org_like(ent)]

    if not orgs_after:
        # Sumario but no org-like after it
        meta["reason"] = "no_org_after_sumario"
        sumario_text = text[seen_sumario_end:]
        return sumario_text, "", meta

    first_org = orgs_after[0]
    first_org_raw = first_org.text
    first_org_norm = _normalize_for_match_letters_only(first_org_raw)
    meta["first_org_raw"] = first_org_raw
    meta["first_org_norm"] = first_org_norm

    # 3) Scan subsequent org-like entities for the first repeat of the normalized key
    boundary_ent = None
    for ent in orgs_after[1:]:
        cur_norm = _normalize_for_match_letters_only(ent.text)
        if debug:
            print(f"[ORG SCAN] first={first_org_norm!r} vs cur={cur_norm!r} | raw={ent.text!r}")
        if cur_norm == first_org_norm:
            boundary_ent = ent
            break
    
    if boundary_ent is None:
        for ent in orgs_after[1:]:
            cur_norm = _normalize_for_match_letters_only(ent.text)
            if (cur_norm in first_org_norm) and (len(cur_norm) < len(first_org_norm)):
                boundary_ent = ent
                break

    if boundary_ent is None:
        # No repeat encountered → by strict rule, there's no body yet.
        meta["reason"] = "no_repeat_match"
        sumario_text = text[seen_sumario_end:]
        return sumario_text, "", meta

    # 4) Slice text into sumario and body using char offsets
    boundary_start = int(boundary_ent.start_char)
    meta["boundary_org_raw"] = boundary_ent.text
    meta["boundary_org_norm"] = _normalize_for_match_letters_only(boundary_ent.text)
    meta["boundary_org_start"] = boundary_start

    sumario_text = text[seen_sumario_end:boundary_start]
    body_text = text[boundary_start:]

    return sumario_text, body_text, meta


__all__ = [
    "split_sumario_and_body",
    "_normalize_for_match_letters_only",
    "ORG_LIKE_LABELS",
    "SUMARIO_LABEL",
]
