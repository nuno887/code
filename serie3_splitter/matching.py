from typing import List, Optional, Set, Tuple

from .config import LETTERS_MIN_RATIO, NGRAM_N, NGRAM_JACCARD_MIN, MIN_LEN_FOR_NGRAMS
from .normalizers import (
    _normalize_title_for_match,
    _tighten,
    _letters_only,
    _char_ngrams,
    _ocr_clean,
)


def _best_contained(basis: str, prepared_allowed: List[Tuple[str, str, str, str]], key_idx: int, min_len: int = 4) -> Optional[str]:
    """
    Return the original allowed title whose transformed form (chosen by key_idx)
    is contained in `basis`. Prefers the longest match. Guards against tiny hits.
      basis: normalized/tight/letters-only string to search within
      prepared_allowed: list of (orig, an, at, al)
      key_idx: 1=an (normalized), 2=at (tight), 3=al (letters-only)
      min_len: minimum candidate length to consider (avoid 'ID' in 'RIDER', etc.)
    """
    if not basis:
        return None
    best_orig = None
    best_len = -1
    for orig, an, at, al in prepared_allowed:
        cand = (an, at, al)[key_idx - 1]
        if cand and len(cand) >= min_len and cand in basis:
            if len(cand) > best_len:
                best_orig = orig
                best_len = len(cand)
    return best_orig


def pick_canonical_from_block(block_titles: List[str], allowed_titles: Set[str]) -> Optional[str]:
    """Robustly pick the allowed child title that best matches this header block.
    Tries a block-level match first (after OCR cleaning), then falls back to per-line cascade.
    """
    if not allowed_titles:
        return None

    # Prepare allowed titles once
    prepared_allowed: List[Tuple[str, str, str, str]] = []
    for t in allowed_titles:
        t_clean = _ocr_clean(t)
        t_norm = _normalize_title_for_match(t_clean)
        t_tight = _tighten(t_norm)
        t_letters = _letters_only(t_norm)
        prepared_allowed.append((t, t_norm, t_tight, t_letters))

    # --- Block-level ---
    block_join = _ocr_clean("\n".join(block_titles))
    bj_norm = _normalize_title_for_match(block_join)
    bj_tight = _tighten(bj_norm)
    bj_letters = _letters_only(bj_norm)

    # 1) exact normalized
    for orig, an, at, al in prepared_allowed:
        if bj_norm and bj_norm == an:
            return orig
    # 1b) contains (normalized)
    found = _best_contained(bj_norm, prepared_allowed, key_idx=1, min_len=4)
    if found:
        return found

    # 2) exact tight
    for orig, an, at, al in prepared_allowed:
        if bj_tight and bj_tight == at:
            return orig
    # 2b) contains (tight)
    found = _best_contained(bj_tight, prepared_allowed, key_idx=2, min_len=4)
    if found:
        return found

    # 3) exact letters-only
    for orig, an, at, al in prepared_allowed:
        if bj_letters and bj_letters == al:
            return orig
    # 3b) contains (letters-only)
    found = _best_contained(bj_letters, prepared_allowed, key_idx=3, min_len=4)
    if found:
        return found

    # 4) containment with ratio (letters-only)
    for orig, an, at, al in prepared_allowed:
        if not bj_letters or not al:
            continue
        if bj_letters in al or al in bj_letters:
            shorter = min(len(bj_letters), len(al))
            longer = max(len(bj_letters), len(al))
            if longer and (shorter / longer) >= LETTERS_MIN_RATIO:
                return orig

    # 5) n-gram Jaccard (letters-only)
    if len(bj_letters) >= MIN_LEN_FOR_NGRAMS:
        bj_ngrams = _char_ngrams(bj_letters, NGRAM_N)
        best = (None, 0.0)
        for orig, an, at, al in prepared_allowed:
            if len(al) < MIN_LEN_FOR_NGRAMS:
                continue
            al_ngrams = _char_ngrams(al, NGRAM_N)
            if not al_ngrams:
                continue
            inter = len(bj_ngrams & al_ngrams)
            union = len(bj_ngrams | al_ngrams)
            j = inter / union if union else 0.0
            if j >= NGRAM_JACCARD_MIN and j > best[1]:
                best = (orig, j)
        if best[0] is not None:
            return best[0]

    # --- Per-line fallback ---
    for bt_raw in block_titles:
        bt_norm = _normalize_title_for_match(_ocr_clean(bt_raw))
        bt_tight = _tighten(bt_norm)
        bt_letters = _letters_only(bt_norm)

        # exact normalized
        for orig, an, at, al in prepared_allowed:
            if bt_norm == an:
                return orig
        # contains (normalized)
        found = _best_contained(bt_norm, prepared_allowed, key_idx=1, min_len=4)
        if found:
            return found

        # exact tight
        for orig, an, at, al in prepared_allowed:
            if bt_tight and bt_tight == at:
                return orig
        # contains (tight)
        found = _best_contained(bt_tight, prepared_allowed, key_idx=2, min_len=4)
        if found:
            return found

        # exact letters-only
        for orig, an, at, al in prepared_allowed:
            if bt_letters and bt_letters == al:
                return orig
        # contains (letters-only)
        found = _best_contained(bt_letters, prepared_allowed, key_idx=3, min_len=4)
        if found:
            return found

        # ratio containment (letters-only)
        for orig, an, at, al in prepared_allowed:
            if not bt_letters or not al:
                continue
            if bt_letters in al or al in bt_letters:
                shorter = min(len(bt_letters), len(al))
                longer = max(len(bt_letters), len(al))
                if longer and (shorter / longer) >= LETTERS_MIN_RATIO:
                    return orig

        # n-gram Jaccard (letters-only)
        if len(bt_letters) >= MIN_LEN_FOR_NGRAMS:
            bt_ngrams = _char_ngrams(bt_letters, NGRAM_N)
            best = (None, 0.0)
            for orig, an, at, al in prepared_allowed:
                if len(al) < MIN_LEN_FOR_NGRAMS:
                    continue
                al_ngrams = _char_ngrams(al, NGRAM_N)
                if not al_ngrams:
                    continue
                inter = len(bt_ngrams & al_ngrams)
                union = len(bt_ngrams | al_ngrams)
                j = inter / union if union else 0.0
                if j >= NGRAM_JACCARD_MIN and j > best[1]:
                    best = (orig, j)
            if best[0] is not None:
                return best[0]

    return None
