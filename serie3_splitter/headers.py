from typing import Any, Dict, List, Optional, Tuple

from .normalizers import _normalize_title, _tighten, _letters_only


def doc_type_key(item: Dict[str, Any]) -> Tuple[int, int, str]:
    pid = item.get("paragraph_id")
    org_ids = tuple(item.get("org_ids", []))
    title = _normalize_title((item.get("doc_name") or {}).get("text") or "")
    return (pid if pid is not None else -1, hash(org_ids), title)


def locate_window(pos: int, windows: List[Dict[str, Any]]) -> Optional[int]:
    for i, w in enumerate(windows):
        if w["start"] <= pos < w["end"]:
            return i
    return None


def match_doc_type_headers(doc_body, payload: Dict[str, Any], org_windows: List[Dict[str, Any]]):
    def canon_norm(s: str) -> str:
        return _normalize_title(s).casefold()

    def canon_tight(s: str) -> str:
        return _tighten(_normalize_title(s)).casefold()

    def canon_letters(s: str) -> str:
        return _letters_only(_normalize_title(s))

    payload_titles: List[Dict[str, Any]] = []
    for it in payload.get("items", []):
        title_raw = (it.get("doc_name") or {}).get("text") or ""
        title = _normalize_title(title_raw)
        if not title:
            continue
        payload_titles.append({
            "key": doc_type_key(it),
            "title": title,
            "norm": canon_norm(title),
            "tight": canon_tight(title),
            "letters": canon_letters(title),
            "item": it,
        })

    body_titles: List[Dict[str, Any]] = []
    for e in doc_body.ents:
        if getattr(e, "label_", None) != "DOC_NAME_LABEL":
            continue
        text_norm = _normalize_title(e.text)
        if not text_norm:
            continue
        body_titles.append({
            "ent": e,
            "title": text_norm,
            "norm": canon_norm(text_norm),
            "tight": canon_tight(text_norm),
            "letters": canon_letters(text_norm),
        })

    matches = {pt["key"]: None for pt in payload_titles}
    claimed_positions: set = set()

    def claim(pt_key, ent_obj, confidence: float):
        start, end = ent_obj.start_char, ent_obj.end_char
        claimed_positions.add((start, end))
        win_idx = locate_window(start, org_windows)
        matches[pt_key] = {"start": start, "end": end, "window_index": win_idx, "confidence": confidence}

    # Pass 1
    for pt in payload_titles:
        for bt in body_titles:
            if (bt["ent"].start_char, bt["ent"].end_char) in claimed_positions:
                continue
            if bt["norm"] == pt["norm"] and bt["norm"]:
                claim(pt["key"], bt["ent"], 1.0)
                break

    # Pass 2
    for pt in payload_titles:
        if matches[pt["key"]] is not None:
            continue
        for bt in body_titles:
            if (bt["ent"].start_char, bt["ent"].end_char) in claimed_positions:
                continue
            if bt["tight"] == pt["tight"] and bt["tight"]:
                claim(pt["key"], bt["ent"], 0.95)
                break

    # Pass 3
    for pt in payload_titles:
        if matches[pt["key"]] is not None:
            continue
        for bt in body_titles:
            if (bt["ent"].start_char, bt["ent"].end_char) in claimed_positions:
                continue
            if bt["letters"] and bt["letters"] == pt["letters"]:
                claim(pt["key"], bt["ent"], 0.9)
                break

    # Pass 4
    from .config import LETTERS_MIN_RATIO
    for pt in sorted(payload_titles, key=lambda x: len(x["letters"]), reverse=True):
        if matches[pt["key"]] is not None:
            continue
        best_bt = None
        best_score = 0.0
        a = pt["letters"]
        if not a:
            continue
        for bt in body_titles:
            if (bt["ent"].start_char, bt["ent"].end_char) in claimed_positions:
                continue
            b = bt["letters"]
            if not b:
                continue
            if a in b or b in a:
                shorter = min(len(a), len(b))
                longer = max(len(a), len(b))
                score = (shorter / longer) if longer else 0.0
                if score >= LETTERS_MIN_RATIO and score > best_score:
                    best_score = score
                    best_bt = bt
        if best_bt is not None:
            claim(pt["key"], best_bt["ent"], 0.8 * best_score)

    return matches


def compute_next_bounds_per_window(doc_type_matches, org_windows: List[Dict[str, Any]]):
    starts_by_win = {}
    for mt in doc_type_matches.values():
        if mt is None:
            continue
        win_idx = mt.get("window_index")
        if win_idx is None:
            continue
        starts_by_win.setdefault(win_idx, []).append(mt["start"])

    next_bounds = {}
    for win_idx, starts in starts_by_win.items():
        starts_sorted = sorted(set(starts))
        bounds = {}
        for i, st in enumerate(starts_sorted):
            if i + 1 < len(starts_sorted):
                bounds[st] = starts_sorted[i + 1]
            else:
                bounds[st] = org_windows[win_idx]["end"]
        next_bounds[win_idx] = bounds
    return next_bounds
