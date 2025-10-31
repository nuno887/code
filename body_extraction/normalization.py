from __future__ import annotations
import re
import unicodedata

def _strip_accents(s: str) -> str:
    if s is None:
        return ""
    nfkd = unicodedata.normalize("NFD", s)
    return "".join(ch for ch in nfkd if unicodedata.category(ch) != "Mn")

def _strip_markdown_bold(s: str) -> str:
    # Remove surrounding ** that may appear in headers
    return s.replace("**", "").strip()

def _collapse_spaces(s: str) -> str:
    # Collapse multiple whitespace into single spaces; strip ends
    return re.sub(r"\s+", " ", s).strip()

def _join_spaced_caps(s: str) -> str:
    """
    Join artificial spacing often produced by PDF extraction in ALL-CAPS words.
    Example: "D IREÇÃO R EGIONAL" -> "DIREÇÃO REGIONAL".
    We only remove spaces BETWEEN capital letters (incl. Portuguese diacritics).
    """
    caps = "A-ZÁÂÃÀÇÉÊÍÓÔÕÚÜ"
    pattern = rf"([{caps}])\s(?=[{caps}])"
    prev = None
    out = s
    while prev != out:
        prev = out
        out = re.sub(pattern, r"\1", out)
    return out

def normalize_text(s: str) -> str:
    if s is None:
        return ""
    s = _strip_markdown_bold(s)
    s = _collapse_spaces(s)
    s = _join_spaced_caps(s)
    return s

def _org_key(s: str) -> str:
    """
    Aggressive normalization for ORG comparisons:
    - strip markdown bold
    - remove accents
    - replace & with E
    - remove ALL non-alphanumerics
    - uppercase
    """
    s = _strip_markdown_bold(s)
    s = _strip_accents(s)
    s = s.replace("&", "E")
    s = re.sub(r"[^A-Za-z0-9]", "", s)
    return s.upper()

def normalize_doc_title(s: str) -> str:
    """
    Canonicalize DOC headers for matching. We remove the 'nº/n./no./n.º' token
    and keep a single space before the number. Also normalize slash spacing.
    """
    if s is None:
        return ""
    s = normalize_text(s)

    # Remove the Portuguese numbering token before a digit:
    s = re.sub(r"(?i)\b n \s* (?: [\.\-]\s* )? (?: º | o | ° )? \s* (?=\d)", "", s, flags=re.X)

    # Ensure exactly one space before the number
    s = re.sub(r"(?<=\D)\s*(?=\d)", " ", s)

    # Normalize spaces around slashes in numbers like '586 / 2003' -> '586/2003'
    s = re.sub(r"\s*/\s*", "/", s)

    return s.strip()