from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import spacy
from spacy import displacy
from Entities import setup_entities

nlp = spacy.load("pt_core_news_lg", exclude = "ner")
setup_entities(nlp)

# seg_text lost the spacy ents

# ==========================
# Data models expected upstream
# ==========================

@dataclass
class DocSlice:
    doc_name: str
    text: str
    status: str = "pending"
    confidence: float = 0.0

@dataclass
class OrgResult:
    org: str
    status: str
    docs: List[DocSlice]


# ==========================
# Public entry point
# ==========================

def divide_body_by_org_and_docs_serieIII(doc_body, payload: Dict[str, Any], debug: bool = False) -> Tuple[List[OrgResult], Dict[str, Any]]:
    """
    Refactor step 2 (division by headers):
      - Use ONLY payload + doc_body.ents to anchor *Doc-type* titles (group headers) in the body.
      - Slice the body text into **per-doc-type segments** within each org window:
          start = matched doc-type header
          end   = next matched doc-type header in the same window (or window end)
      - No regex. No hardcoded keywords.
      - Bodies are not allocated yet; we emit doc-type slices as text.
    Returns: (results, summary)
    """
    if not isinstance(payload, dict):
        return [], {"error": "invalid_payload"}

    org_map = _build_org_map(payload)  # {id: name}
    org_windows = _collect_org_windows_from_ents(doc_body)

    # Map payload doc-type titles -> body header entities (if any)
    doc_type_matches = _match_doc_type_headers(doc_body, payload, org_windows, debug=debug)

    # Precompute next-boundaries for matched headers per window
    next_bounds = _compute_next_bounds_per_window(doc_type_matches, org_windows)

    # Group items by org (payload truth)
    items_by_org = _group_items_by_org(payload)

    # Assemble results per org, preserving sumário order
    results: List[OrgResult] = []
    total_slices = 0

    for org_id, org_name in org_map.items():
        win_idx, win_status = _match_org_to_window(org_name, org_windows)
        items = sorted(items_by_org.get(org_id, []), key=lambda it: (it.get("paragraph_id") is None, it.get("paragraph_id")))

        org_result = OrgResult(org=org_name, status=win_status, docs=[])
        for item in items:
            title_raw = (item.get("doc_name") or {}).get("text") or ""
            title = _normalize_title(title_raw)
            key = _doc_type_key(item)

            mt = doc_type_matches.get(key)
            if mt is None:
                # Not anchored in body — emit placeholder
                org_result.docs.append(
                    DocSlice(
                        doc_name=title,
                        text="",
                        status="doc_type_unanchored",
                        confidence=0.0,
                    )
                )
                continue

            # Determine slice [start:end) using next header in the same window
            start = mt["start"]
            # If the matched header is in no window, fall back to global doc end
            win_for_item = mt.get("window_index")
            if win_for_item is not None:
                end = next_bounds.get(win_for_item, {}).get(start)
                if end is None:
                    end = org_windows[win_for_item]["end"]
            else:
                end = len(doc_body.text)
            content_start = mt.get("end", start)
            seg_text = doc_body.text[content_start:end]
            org_result.docs.append(
                DocSlice(
                    doc_name=title,
                    text=seg_text,
                    status="doc_type_segment",
                    confidence=mt.get("confidence", 1.0),
                )
            )
            total_slices += 1

        results.append(org_result)

    summary = {
        "orgs_in_payload": len(org_map),
        "org_windows_found": len(org_windows),
        "doc_type_headers_matched": sum(1 for v in doc_type_matches.values() if v is not None),
        "doc_type_segments": total_slices,
    }
    return results, summary


# ==========================
# Helpers — NO REGEX
# ==========================

def _build_org_map(payload: Dict[str, Any]) -> Dict[int, str]:
    out: Dict[int, str] = {}
    for o in payload.get("orgs", []):
        oid = o.get("id")
        name = (o.get("text") or "").strip()
        if oid is not None:
            out[oid] = name
    if not out:
        # ensure at least one synthetic org so callers don't crash
        out[-1] = "(Sem organização)"
    return out


def _group_items_by_org(payload: Dict[str, Any]) -> Dict[int, List[Dict[str, Any]]]:
    grouped: Dict[int, List[Dict[str, Any]]] = {}
    for it in payload.get("items", []):
        for oid in it.get("org_ids", []):
            grouped.setdefault(oid, []).append(it)
    return grouped


def _normalize_title(s: str) -> str:
    s = s.strip()
    # strip markdown ** around full string, if present
    if s.startswith("**") and s.endswith("**") and len(s) >= 4:
        s = s[2:-2].strip()
    # collapse whitespace runs
    s = " ".join(s.split())
    # drop trailing ':' if present
    if s.endswith(":"):
        s = s[:-1].rstrip()
    return s


def _tighten(s: str) -> str:
    """Remove all spaces to tolerate OCR letter-spacing artifacts."""
    return s.replace(" ", "")


def _collect_org_windows_from_ents(doc_body) -> List[Dict[str, Any]]:
    """
    Build windows from ORG_LABEL entities (ordered). If none found, one global window is created.
    Each window: {"name": str, "start": int, "end": int}
    """
    ents = [e for e in doc_body.ents if getattr(e, "label_", None) == "ORG_LABEL"]
    windows: List[Dict[str, Any]] = []
    for i, e in enumerate(ents):
        start = e.start_char
        end = ents[i + 1].start_char if (i + 1) < len(ents) else len(doc_body.text)
        windows.append({"name": e.text, "start": start, "end": end})
    if not windows:
        windows.append({"name": "(global)", "start": 0, "end": len(doc_body.text)})
    return windows


def _simple_token_set(s: str) -> set:
    """Very light token set for matching (split on whitespace, lowercase)."""
    return set(t.lower() for t in s.split() if t.strip())


def _jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return float(inter) / float(union) if union else 0.0


def _match_org_to_window(org_name: str, org_windows: List[Dict[str, Any]]) -> Tuple[Optional[int], str]:
    if not org_windows:
        return None, "org_unanchored"
    best_idx = None
    best_score = -1.0
    a = _simple_token_set(org_name)

    for i, w in enumerate(org_windows):
        b = _simple_token_set(w["name"])
        sc = _jaccard(a, b)
        if sc > best_score:
            best_score = sc
            best_idx = i

    status = "org_anchored" if best_score > 0 else ("org_unanchored" if org_windows and org_windows[0]["name"] != "(global)" else "org_unanchored")
    return best_idx, status


def _doc_type_key(item: Dict[str, Any]) -> Tuple[int, int, str]:
    """Stable key to reference a payload item regardless of object identity."""
    pid = item.get("paragraph_id")
    org_ids = tuple(item.get("org_ids", []))
    title = _normalize_title((item.get("doc_name") or {}).get("text") or "")
    return (pid if pid is not None else -1, hash(org_ids), title)


def _match_doc_type_headers(doc_body, payload: Dict[str, Any], org_windows: List[Dict[str, Any]], debug: bool = False) -> Dict[Tuple[int, int, str], Optional[Dict[str, Any]]]:
    """
    For every payload doc-type title, try to find a matching DOC_NAME_LABEL entity in the body.
    Matching logic:
      1) exact match on normalized title
      2) tight match (no spaces)
      3) containment tolerance: longest title preference
    Returns a dict: key(item) -> match dict or None
    match dict: {"start": int, "end": int, "window_index": int, "confidence": float}
    """
    # Build payload title variants
    payload_titles: List[Dict[str, Any]] = []
    for it in payload.get("items", []):
        title_raw = (it.get("doc_name") or {}).get("text") or ""
        title = _normalize_title(title_raw)
        if not title:
            continue
        payload_titles.append({
            "key": _doc_type_key(it),
            "title": title,
            "tight": _tighten(title),
            "item": it,
        })

    # Collect body DOC_NAME_LABEL ents
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
            "tight": _tighten(text_norm),
        })

    # Pass 1: exact space-normalized match
    matches: Dict[Tuple[int, int, str], Optional[Dict[str, Any]]] = {pt["key"]: None for pt in payload_titles}
    claimed_positions: set = set()  # (start_char, end_char)

    def claim(pt_key, ent_obj, confidence: float):
        start, end = ent_obj.start_char, ent_obj.end_char
        claimed_positions.add((start, end))
        win_idx = _locate_window(start, org_windows)
        matches[pt_key] = {"start": start, "end": end, "window_index": win_idx, "confidence": confidence}

    # Exact
    for pt in payload_titles:
        for bt in body_titles:
            if bt["title"] == pt["title"]:
                if (bt["ent"].start_char, bt["ent"].end_char) in claimed_positions:
                    continue
                claim(pt["key"], bt["ent"], 1.0)
                break

    # Pass 2: tight match (no spaces)
    for pt in payload_titles:
        if matches[pt["key"]] is not None:
            continue
        for bt in body_titles:
            if (bt["ent"].start_char, bt["ent"].end_char) in claimed_positions:
                continue
            if bt["tight"] == pt["tight"] and bt["title"] and pt["title"]:
                claim(pt["key"], bt["ent"], 0.9)
                break

    # Pass 3: containment tolerance — prefer longest title
    # Allow either side to contain the other (space-normalized)
    for pt in sorted(payload_titles, key=lambda x: len(x["title"]), reverse=True):
        if matches[pt["key"]] is not None:
            continue
        best_bt = None
        best_score = 0.0
        for bt in body_titles:
            if (bt["ent"].start_char, bt["ent"].end_char) in claimed_positions:
                continue
            a = pt["title"]
            b = bt["title"]
            if not a or not b:
                continue
            contains = a.startswith(b) or b.startswith(a) or (a in b) or (b in a)
            if contains:
                shorter = min(len(a), len(b))
                longer = max(len(a), len(b))
                score = shorter / longer if longer else 0.0
                if score > best_score:
                    best_score = score
                    best_bt = bt
        if best_bt is not None and best_score >= 0.5:
            claim(pt["key"], best_bt["ent"], 0.7 * best_score)

    if debug:
        matched = sum(1 for v in matches.values() if v is not None)
        print(f"[dbg] doc-type headers matched: {matched}/{len(matches)}")

    return matches


def _locate_window(pos: int, windows: List[Dict[str, Any]]) -> Optional[int]:
    for i, w in enumerate(windows):
        if w["start"] <= pos < w["end"]:
            return i
    return None

def _compute_next_bounds_per_window(doc_type_matches: Dict[Tuple[int, int, str], Optional[Dict[str, Any]]],
                                     org_windows: List[Dict[str, Any]]) -> Dict[int, Dict[int, int]]:
    """
    For each window index, sort matched header starts and map start -> next_start (or window end).
    Returns: {window_index: {start: end}}
    """
    # Collect starts per window
    starts_by_win: Dict[int, List[int]] = {}
    for mt in doc_type_matches.values():
        if mt is None:
            continue
        win_idx = mt.get("window_index")
        if win_idx is None:
            continue
        starts_by_win.setdefault(win_idx, []).append(mt["start"])

    # Build mapping
    next_bounds: Dict[int, Dict[int, int]] = {}
    for win_idx, starts in starts_by_win.items():
        starts_sorted = sorted(set(starts))
        bounds: Dict[int, int] = {}
        for i, st in enumerate(starts_sorted):
            if i + 1 < len(starts_sorted):
                bounds[st] = starts_sorted[i + 1]
            else:
                # last header in this window ends at window end
                bounds[st] = org_windows[win_idx]["end"]
        next_bounds[win_idx] = bounds
    return next_bounds
