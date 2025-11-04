from typing import Any, Dict, List, Optional, Tuple, Set
from .utils_text import _normalize_title
from .utils_text import _ocr_clean  

# --- small helpers (local to this module) ---

def _simple_token_set(s: str) -> set:
    return set(t.lower() for t in s.split() if t.strip())

def _jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return float(inter) / float(union) if union else 0.0


def _collect_org_windows_from_ents(doc_body, allowed_orgs: Optional[List[str]] = None) -> List[Dict[str, Any]]:


    def norm(s: str) -> str:
        s = (s or "").strip()
        s = " ".join(s.split())
        if s.startswith("**") and s.endswith("**") and len(s) >= 4:
            s = s[2:-2].strip()
        return s

    def tight(s: str) -> str:
        return norm(s).replace(" ", "").lower()

    def toks(s: str) -> set:
        return set(w for w in norm(s).lower().split() if w)

    allowed_orgs = [norm(o) for o in (allowed_orgs or []) if norm(o)]
    allowed_tight = [tight(o) for o in allowed_orgs]
    allowed_toksets = [toks(o) for o in allowed_orgs]

    
    ACCEPT = {"ORG_LABEL", "ORG_WITH_STAR_LABEL"}

    ents_sorted = sorted(
    [e for e in doc_body.ents if getattr(e, "label_", None) in ACCEPT],
    key=lambda e: e.start_char
)
    

    # Filter to those that actually match an allowed org (tight/overlap)
    kept: List[Tuple[int, int, str]] = []
    MIN_LETTERS = 8 # reject tiny/roman numerals
    MIN_TOKENS =  2 # require at least two tokens
    MIN_SUBSTR = 10 # only allow substring matches when long enough
    for e in ents_sorted:
        txt = e.text
        cand_norm = norm(txt)
        cand_tight = tight(txt)

        # quality gates
        letters_only = "".join(ch for ch in cand_norm if ch.isalpha())
        token_count = len([w for w in cand_norm.split() if w])

        # drop single roman numerals or too-short anchors
        if len(letters_only) < MIN_LETTERS or token_count < MIN_TOKENS:
            continue

        is_allowed = any(
            (cand_tight == at) or (cand_tight in at) or (at in cand_tight)
            for at in allowed_tight
        )
        if is_allowed:
            kept.append((e.start_char, e.end_char, txt))

    windows: List[Dict[str, Any]] = []
    if kept:
        kept_sorted = sorted(kept, key=lambda t: t[0])
        for i, (st, en, txt) in enumerate(kept_sorted):
            start = st
            end = kept_sorted[i + 1][0] if (i + 1) < len(kept_sorted) else len(doc_body.text)
            windows.append({"name": txt, "start": start, "end": end})
    else:
        windows.append({"name": "(global)", "start": 0, "end": len(doc_body.text)})

    return windows



def _match_org_to_window(org_name: str, org_windows: List[Dict[str, Any]]) -> Tuple[Optional[int], str]:
    """
    Prefer strict/substring match between payload org name and window name.
    Fallback to previous Jaccard scoring if strict match fails.
    """
    if not org_windows:
        return None, "org_unanchored"
    if len(org_windows) == 1 and org_windows[0].get("name") == "(global)":
        return 0, "org_anchored"

    # --- strict/substring match first ---
    MIN_SUBSTR = 10 # same threshold used above


    a_norm = _normalize_title(org_name)
    a_tight = a_norm.replace(" ", "").lower()
    for i, w in enumerate(org_windows):
        b_norm = _normalize_title(w["name"])
        b_tight = b_norm.replace(" ", "").lower()
        if a_tight == b_tight:
            return i, "org_anchored"
        if (len(a_tight) >= MIN_SUBSTR and a_tight in b_tight) or \
       (len(b_tight) >= MIN_SUBSTR and b_tight in a_tight):
            return i, "org_anchored"

    # --- fallback: Jaccard scoring (legacy behavior) ---
    best_idx = None
    best_score = -1.0
    a = _simple_token_set(org_name)
    for i, w in enumerate(org_windows):
        b = _simple_token_set(w["name"])
        sc = _jaccard(a, b)
        if sc > best_score:
            best_score = sc
            best_idx = i

    status = "org_anchored" if best_score > 0 else "org_unanchored"
    return best_idx, status

