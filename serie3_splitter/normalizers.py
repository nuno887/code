import re
import unicodedata
from typing import Set


from .config import LETTERS_MIN_RATIO


_DOT_LEADER_RE = re.compile(r"[.\u2022·]{3,}")
_HYPHEN_WRAP_RE = re.compile(r"-\s*(?:\r?\n|\n)\s*")
_INTERLETTER_SEQ_RE = re.compile(r"(?i)\b(?:[A-Za-zÀ-ÿ]\s+){2,}[A-Za-zÀ-ÿ]\b")




def _strip_accents(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")




def _normalize_unicode_spaces(s: str) -> str:
    s = unicodedata.normalize("NFKC", s or "")
    s = s.replace("\u00A0", " ")
    return " ".join(s.split())




def _normalize_title(s: str) -> str:
    s = (s or "").strip()
    if s.startswith("**") and s.endswith("**") and len(s) >= 4:
        s = s[2:-2].strip()
        s = " ".join(s.split())
        if s.endswith(":"):
            s = s[:-1].rstrip()
    return s




def _tighten(s: str) -> str:
    return s.replace(" ", "")




def _letters_only(s: str) -> str:
    s = _strip_accents(s or "").lower()
    return "".join(ch for ch in s if ("a" <= ch <= "z") or ("0" <= ch <= "9"))




def _char_ngrams(s: str, n: int = 3) -> set:
    if len(s) < n:
        return {s} if s else set()
    return {s[i:i + n] for i in range(len(s) - n + 1)}




def _remove_dot_leaders(s: str) -> str:
    return _DOT_LEADER_RE.sub("", s)




def _fix_hyphen_linewraps(s: str) -> str:
    return _HYPHEN_WRAP_RE.sub("", s)




def _collapse_interletter_spacing(s: str) -> str:
    def _join(m):
        return re.sub(r"\s+", "", m.group(0))


    return _INTERLETTER_SEQ_RE.sub(_join, s)




def _ocr_clean(s: str) -> str:
    s = _fix_hyphen_linewraps(s)
    s = _remove_dot_leaders(s)
    s = _collapse_interletter_spacing(s)
    s = _normalize_unicode_spaces(s)
    return s