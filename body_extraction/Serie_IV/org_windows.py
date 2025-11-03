from typing import Any, Dict, List, Optional, Tuple, Set
from .utils_text import _normalize_title
from .utils_text import _ocr_clean  # (not used here, kept symmetrical)
from .debug import DBG

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
    from .debug import DBG

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

    # ✅ Only consider actual org banners as window anchors
    ACCEPT = {"ORG_LABEL", "ORG_WITH_STAR_LABEL"}

    ents_sorted = [e for e in sorted(list(doc_body.ents), key=lambda e: e.start_char)
                   if getattr(e, "label_", None) in ACCEPT]

    # Filter to those that actually match an allowed org (tight/overlap)
    kept: List[Tuple[int, int, str]] = []
    for e in ents_sorted:
        txt = e.text
        cand_tight = tight(txt)
        cand_tokset = toks(txt)
        tight_ok = any((a in cand_tight) or (cand_tight in a) for a in allowed_tight)
        overlap_ok = any(len(cand_tokset & a) >= 2 for a in allowed_toksets)
        if tight_ok or overlap_ok:
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

    # (optional) light debug
    DBG._p(f"WIN simple: kept={len(kept)} windows={len(windows)}")
    return windows



def _match_org_to_window(org_name: str, org_windows: List[Dict[str, Any]]) -> Tuple[Optional[int], str]:
    if not org_windows:
        return None, "org_unanchored"
    if len(org_windows) == 1 and org_windows[0].get("name") == "(global)":
        return 0, "org_anchored"

    best_idx = None
    best_score = -1.0
    a = _simple_token_set(org_name)

    for i, w in enumerate(org_windows):
        b = _simple_token_set(w["name"])
        sc = _jaccard(a, b)
        DBG._p(f"ORG match-cand: idx={i} score={sc:.3f} name={w['name'][:80]!r}")
        if sc > best_score:
            best_score = sc
            best_idx = i

    status = "org_anchored" if best_score > 0 else "org_unanchored"
    DBG._p(f"ORG match-picked: idx={best_idx} status={status} score={best_score:.3f}")
    return best_idx, status
