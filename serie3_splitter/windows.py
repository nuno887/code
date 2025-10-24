from typing import Any, Dict, List, Optional, Tuple

from .config import ORG_HEADER_LABELS, MERGE_MAX_GAP


def simple_token_set(s: str) -> set:
    return set(t.lower() for t in (s or "").split() if t.strip())


def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return float(inter) / float(union) if union else 0.0


def match_org_to_window(org_name: str, org_windows: List[Dict[str, Any]]):
    if not org_windows:
        return None, "org_unanchored"
    if len(org_windows) == 1 and org_windows[0].get("name") == "(global)":
        return 0, "org_anchored"

    best_idx = None
    best_score = -1.0
    a = simple_token_set(org_name)
    for i, w in enumerate(org_windows):
        b = simple_token_set(w["name"])
        sc = jaccard(a, b)
        if sc > best_score:
            best_score = sc
            best_idx = i
    status = "org_anchored" if best_score > 0 else "org_unanchored"
    return best_idx, status


def collect_org_windows_from_ents(doc_body, allowed_orgs: Optional[List[str]] = None) -> List[Dict[str, Any]]:
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

    ents_sorted = sorted(list(doc_body.ents), key=lambda e: e.start_char)
    merged_spans: List[Tuple[int, int, str]] = []

    cur_start = None
    cur_end = None
    for e in ents_sorted:
        lab = getattr(e, "label_", None)
        if lab not in ORG_HEADER_LABELS:
            if cur_start is not None:
                merged_spans.append((cur_start, cur_end, doc_body.text[cur_start:cur_end]))
                cur_start = cur_end = None
            continue
        if cur_start is None:
            cur_start = e.start_char
            cur_end = e.end_char
        else:
            gap = e.start_char - cur_end
            if gap <= MERGE_MAX_GAP:
                cur_end = max(cur_end, e.end_char)
            else:
                merged_spans.append((cur_start, cur_end, doc_body.text[cur_start:cur_end]))
                cur_start = e.start_char
                cur_end = e.end_char

    if cur_start is not None:
        merged_spans.append((cur_start, cur_end, doc_body.text[cur_start:cur_end]))

    kept: List[Tuple[int, int, str]] = []
    for (st, en, txt) in merged_spans:
        cand_text = txt
        cand_tight = tight(cand_text)
        cand_tokset = toks(cand_text)

        tight_ok = any((a in cand_tight) or (cand_tight in a) for a in allowed_tight)
        overlap_ok = any(len(cand_tokset & a) >= 2 for a in allowed_toksets)
        if tight_ok or overlap_ok:
            kept.append((st, en, cand_text))

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
