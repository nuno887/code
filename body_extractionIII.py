from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Set

import spacy
from spacy import displacy  # noqa: F401  (kept if you use it elsewhere)
from Entities import setup_entities


# Load Portuguese pipeline and your custom entity setup (keeps your original choice)
nlp = spacy.load("pt_core_news_lg", exclude="ner")
setup_entities(nlp)

# seg_text lost the spacy ents


# ==========================
# Data models expected upstream
# ==========================

@dataclass
class SubSlice:
    """A child subdivision inside a DocSlice, opened by a payload-approved internal header block."""
    title: str             # canonical title (one of the allowed child titles)
    headers: List[str]     # all consecutive DOC_NAME_LABEL lines grouped into this header block (normalized)
    body: str              # text from end of header block up to next approved header (or end)
    start: int             # start offset of the body (relative to the parent seg_text)
    end: int               # end offset of the body (relative to the parent seg_text)

@dataclass
class DocSlice:
    doc_name: str
    text: str
    status: str = "pending"
    confidence: float = 0.0
    # lightweight entity snapshot for this segment (label, text, start, end), offsets relative to seg_text
    ents: List[Tuple[str, str, int, int]] = field(default_factory=list)
    # optional further subdivision inside the segment
    subs: List[SubSlice] = field(default_factory=list)

@dataclass
class OrgResult:
    org: str
    status: str
    docs: List[DocSlice]


# ==========================
# Public entry point
# ==========================

def divide_body_by_org_and_docs_serieIII(
    doc_body,
    payload: Dict[str, Any],
    debug: bool = False,
    *,
    reparse_segments: bool = True,      # run nlp(seg_text) to recover entities
    subdivide_children: bool = True,    # create subs only for payload-approved child titles
) -> Tuple[List[OrgResult], Dict[str, Any]]:
    """
    Refactor step 2 (division by headers):
      - Use ONLY payload + doc_body.ents to anchor *Doc-type* titles (group headers) in the body.
      - Slice the body text into **per-doc-type segments** within each org window:
          start = matched doc-type header
          end   = next matched doc-type header in the same window (or window end)
      - No regex. No hardcoded keywords.
      - Bodies are not allocated yet; we emit doc-type slices as text.

    Second pass (optional):
      - Re-parse each segment with SpaCy to recover entities inside the slice (reparse_segments=True).
      - Further subdivide segments into SubSlices using internal header blocks that match
        ONLY the *children* of the current item from the payload (subdivide_children=True).
      - Collapse consecutive DOC_NAME_LABEL entities into a single header block before matching.

    Returns: (results, summary)
    """
    if not isinstance(payload, dict):
        return [], {"error": "invalid_payload"}
    
    _dbg(debug, "START divide_body_by_org_and_docs_serieIII")
    _dbg(debug, "payload_orgs:", len(payload.get("orgs", [])), "payload_items:", len(payload.get("items", [])))


    org_map = _build_org_map(payload)  # {id: name}
    _dbg(debug, "org_map:", org_map)

    org_windows = _collect_org_windows_from_ents(doc_body)
    _dbg(debug, "org_windows:", [{"name": w["name"], "start": w["start"], "end": w["end"]} for w in org_windows])

    # Map payload doc-type titles -> body header entities (if any)
    doc_type_matches = _match_doc_type_headers(doc_body, payload, org_windows, debug=debug)
    _dbg(debug, "doc_type_headers_matched:",
         sum(1 for v in doc_type_matches.values() if v is not None), "/", len(doc_type_matches))

    # Precompute next-boundaries for matched headers per window
    next_bounds = _compute_next_bounds_per_window(doc_type_matches, org_windows)
    _dbg(debug, "next_bounds keys (per window):", {k: sorted(v.keys()) for k, v in next_bounds.items()})

    # Group items by org (payload truth)
    items_by_org = _group_items_by_org(payload)
    _dbg(debug, "items_by_org keys:", {k: len(v) for k, v in items_by_org.items()})

    # Assemble results per org, preserving sumário order
    results: List[OrgResult] = []
    total_slices = 0

    for org_id, org_name in org_map.items():
        win_idx, win_status = _match_org_to_window(org_name, org_windows)
        items = sorted(items_by_org.get(org_id, []), key=lambda it: (it.get("paragraph_id") is None, it.get("paragraph_id")))
        _dbg(debug, f"ORG {org_id} → '{org_name}'", "win_idx:", win_idx, "status:", win_status, "items:", len(items))


        org_result = OrgResult(org=org_name, status=win_status, docs=[])
        for item in items:
            title_raw = (item.get("doc_name") or {}).get("text") or ""
            title = _normalize_title(title_raw)
            key = _doc_type_key(item)

            mt = doc_type_matches.get(key)
            if mt is None:
                _dbg(debug, "  [ITEM]", title, "→ NOT ANCHORED in body")
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

            content_start = mt.get("end", start)  # exclude the header from the segment body
            seg_text = doc_body.text[content_start:end]

            _dbg(debug, "  [ITEM]", title, "→ anchored:",
                 {"win": win_for_item, "header_start": start, "content_start": content_start, "end": end,
                  "seg_len": len(seg_text)})

            # Build base slice
            ds = DocSlice(
                doc_name=title,
                text=seg_text,
                status="doc_type_segment",
                confidence=mt.get("confidence", 1.0),
            )

            # --- Second pass over seg_text (optional) ---
            if reparse_segments and seg_text.strip():
                ds.ents = _reparse_seg_text(seg_text)
                _dbg(debug, "    ents_in_seg:", len(ds.ents))

                if subdivide_children:
                    # Only children of the *current* item are allowed as internal headers
                    allowed = _allowed_child_titles_for_item(item, debug=debug)
                    _dbg(debug, "    allowed_children:", list(allowed))

                    subs = _subdivide_seg_text_by_allowed_headers(seg_text, allowed, debug=debug)
                    ds.subs = subs
                    _dbg(debug, "    subslices:", len(ds.subs))
                    for idx, sub in enumerate(ds.subs, start=1):
                        _dbg(debug, f"      sub[{idx}] title:", sub.title,
                             "headers:", sub.headers, "body_len:", len(sub.body),
                             "range:", (sub.start, sub.end))

            org_result.docs.append(ds)
            total_slices += 1

        results.append(org_result)

    summary = {
        "orgs_in_payload": len(org_map),
        "org_windows_found": len(org_windows),
        "doc_type_headers_matched": sum(1 for v in doc_type_matches.values() if v is not None),
        "doc_type_segments": total_slices,
        "segment_reparsed": bool(reparse_segments),
        "segments_with_subdivisions": sum(len(d.subs) for r in results for d in r.docs),
    }
    _dbg(debug, "SUMMARY:", summary)
    _dbg(debug, "END divide_body_by_org_and_docs_serieIII")
    return results, summary


# ==========================
# Helpers — NO REGEX
# ==========================
def _dbg(enabled: bool, *parts):
    if enabled:
        print("[serieIII]", *parts)

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
    # remove surrounding ** if the whole string is bolded
    if s.startswith("**") and s.endswith("**") and len(s) >= 4:
        s = s[2:-2].strip()
    # collapse whitespace runs
    s = " ".join(s.split())
    # drop trailing colon
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
        _dbg(True, f"[headers] matched: {matched}/{len(matches)}")

    return matches


def _locate_window(pos: int, windows: List[Dict[str, Any]]) -> Optional[int]:
    for i, w in enumerate(windows):
        if w["start"] <= pos < w["end"]:
            return i
    return None


def _compute_next_bounds_per_window(
    doc_type_matches: Dict[Tuple[int, int, str], Optional[Dict[str, Any]]],
    org_windows: List[Dict[str, Any]]
) -> Dict[int, Dict[int, int]]:
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


# ==========================
# Helpers — second pass (re-parse + subdivision)
# ==========================

def _reparse_seg_text(seg_text: str) -> List[Tuple[str, str, int, int]]:
    """
    Run SpaCy on a segment and return a lightweight snapshot of entities.
    Returns list of (label, text, start_char, end_char) relative to seg_text.
    """
    doc = nlp(seg_text)
    out: List[Tuple[str, str, int, int]] = []
    for e in doc.ents:
        label = getattr(e, "label_", "")
        out.append((label, e.text, e.start_char, e.end_char))
    return out


def _allowed_child_titles_for_item(item: Dict[str, Any], debug: bool = False) -> Set[str]:
    """
    Build the set of *allowed* child titles for this item from the payload.

    Supports:
      - item['allowed_children'] = [ "Title A", ... ]          # strings
      - item['children'] = [
            {"doc_name": {"text": "Title A"}},
            {"text": "Title B"},
            {"child": "<full paragraph possibly with newlines>", "label": "PARAGRAPH"},
        ]
      - item['bodies'][i].doc_name.text                        # future-proof
    Titles are normalized (strip **, collapse ws, drop trailing ':').
    """
    titles: Set[str] = set()

    def _tight_key(s: str) -> str:
        return _normalize_title(s).replace(" ", "").lower()

    # 1) explicit list of strings
    for t in (item.get("allowed_children") or []):
        t_norm = _normalize_title(str(t))
        if t_norm:
            titles.add(t_norm)

    # 2) children objects (various shapes)
    for ch in (item.get("children") or []):
        if not isinstance(ch, dict):
            continue

        txt = ""

        # preferred shape: {"doc_name": {"text": "..."}}
        if isinstance(ch.get("doc_name"), dict) and ch["doc_name"].get("text"):
            txt = ch["doc_name"]["text"]

        # alt shape: {"text": "..."}
        elif "text" in ch and ch.get("text"):
            txt = ch["text"]

        # your new shape: {"child": "<full paragraph>", "label": "PARAGRAPH"}
        elif "child" in ch and ch.get("child"):
            # take the first non-empty line as the title
            raw = str(ch["child"])
            txt = " ".join(raw.split())

        t_norm = _normalize_title(txt or "")
        if t_norm:
            titles.add(t_norm)

    # 3) bodies that might carry titles (future-proof)
    for b in (item.get("bodies") or []):
        if not isinstance(b, dict):
            continue
        if isinstance(b.get("doc_name"), dict) and b["doc_name"].get("text"):
            t_norm = _normalize_title(b["doc_name"]["text"])
            if t_norm:
                titles.add(t_norm)

    # dedupe loosely (OCR spacing)
    dedup: Set[str] = set()
    out: Set[str] = set()
    for t in titles:
        k = _tight_key(t)
        if k in dedup:
            continue
        dedup.add(k)
        out.add(t)

    if debug:
        _dbg(True, "      [_allowed_child_titles_for_item] collected:", out)

    return out

## error problably here

def _subdivide_seg_text_by_allowed_headers(seg_text: str, allowed_titles: Set[str], debug: bool = False) -> List[SubSlice]:
    """
    - Parse seg_text
    - Collapse consecutive DOC_NAME_LABEL into one header block
    - A block starts a *new* sub-slice only if the block canonically matches an *allowed* (child) title
    - Body = from header block end to next approved header block start (exclusive)
    - If no approved headers at all, return a single sub-slice spanning the whole seg_text
      (using the first header block, if present, for context)
    """
    doc = nlp(seg_text)
    ents = sorted(list(doc.ents), key=lambda e: e.start_char)
    _dbg(debug, "      [subdivide] ents:", [(e.label_, e.start_char, e.end_char) for e in ents[:20]],
         "... total:", len(ents))

    # group consecutive DOC_NAME_LABEL into blocks
    header_blocks: List[Dict[str, Any]] = []
    current_block: List[Any] = []
    for e in ents:
        if getattr(e, "label_", None) == "DOC_NAME_LABEL":
            if current_block:
                current_block.append(e)
            else:
                current_block = [e]
        else:
            if current_block:
                start = current_block[0].start_char
                end = current_block[-1].end_char
                header_blocks.append({
                    "headers": current_block[:],
                    "start": start,
                    "end": end,
                    "titles": [_normalize_title(h.text) for h in current_block],
                })
                current_block = []
    if current_block:
        start = current_block[0].start_char
        end = current_block[-1].end_char
        header_blocks.append({
            "headers": current_block[:],
            "start": start,
            "end": end,
            "titles": [_normalize_title(h.text) for h in current_block],
        })

    _dbg(debug, "      [subdivide] header_blocks:",
         [{"titles": hb["titles"], "range": (hb["start"], hb["end"])} for hb in header_blocks])

    # approve blocks by payload titles
    approved: List[Dict[str, Any]] = []
    for hb in header_blocks:
        canon = _pick_canonical_from_block(hb["titles"], allowed_titles)
        if canon is not None:
            approved.append({**hb, "canonical": canon})

    _dbg(debug, "      [subdivide] approved_blocks:",
         [{"title": hb["canonical"], "range": (hb["start"], hb["end"])} for hb in approved])

    subs: List[SubSlice] = []

    if not approved:
        if header_blocks:
            top = header_blocks[0]
            headers_texts = top["titles"]
            body_start = top["end"]
        else:
            headers_texts = []
            body_start = 0
        body_end = len(seg_text)
        body_text = seg_text[body_start:body_end]
        subs.append(SubSlice(
            title=headers_texts[0] if headers_texts else "",
            headers=headers_texts,
            body=body_text,
            start=body_start,
            end=body_end
        ))
        _dbg(debug, "      [subdivide] no approved → single sub from", body_start, "to", body_end)
        return subs

    for i, hb in enumerate(approved):
        header_end = hb["end"]
        next_start = approved[i + 1]["start"] if (i + 1) < len(approved) else len(seg_text)
        body_text = seg_text[header_end:next_start]
        subs.append(SubSlice(
            title=hb["canonical"],
            headers=hb["titles"],
            body=body_text,
            start=header_end,
            end=next_start
        ))
        _dbg(debug, f"      [subdivide] sub[{i+1}] '{hb['canonical']}' body",
             (header_end, next_start), "len:", len(body_text))

    return subs


def _pick_canonical_from_block(block_titles: List[str], allowed_titles: Set[str]) -> Optional[str]:
    """
    Choose the canonical title from a header block that matches one of the allowed titles.
    Matching cascade (increasingly lenient):
      1) exact normalized
      2) tight (no spaces)
      3) prefix match (either side startswith the other, normalized)
      4) containment (substring either way, normalized)
      5) token overlap (Jaccard >= 0.5) on normalized, lowercased tokens
    Return the allowed title (canonical) if matched, else None.
    """
    if not allowed_titles:
        return None

    # Precompute normalized variants for allowed titles
    allowed_norm = [(t, _normalize_title(t), _tighten(_normalize_title(t))) for t in allowed_titles]

    def toks(s: str) -> set:
        return set(w for w in _normalize_title(s).lower().split() if w)

    # Check each line in the header block against allowed titles
    for bt_raw in block_titles:
        bt = _normalize_title(bt_raw)
        bt_tight = _tighten(bt)
        bt_tokens = toks(bt)

        # Pass 1: exact normalized
        for orig, an, at in allowed_norm:
            if bt == an:
                return orig

        # Pass 2: tight (no spaces)
        for orig, an, at in allowed_norm:
            if bt_tight == at:
                return orig

        # Pass 3: prefix match (normalized)
        for orig, an, at in allowed_norm:
            if bt.startswith(an) or an.startswith(bt):
                # keep some minimal ratio so tiny prefixes don't match
                shorter, longer = (len(bt), len(an)) if len(bt) < len(an) else (len(an), len(bt))
                if longer and (shorter / longer) >= 0.5:
                    return orig

        # Pass 4: containment (normalized)
        for orig, an, at in allowed_norm:
            if bt in an or an in bt:
                shorter, longer = (len(bt), len(an)) if len(bt) < len(an) else (len(an), len(bt))
                if longer and (shorter / longer) >= 0.5:
                    return orig

        # Pass 5: token overlap (Jaccard)
        for orig, an, at in allowed_norm:
            an_tokens = set(w for w in an.lower().split() if w)
            if an_tokens:
                inter = len(bt_tokens & an_tokens)
                union = len(bt_tokens | an_tokens)
                if union and (inter / union) >= 0.5:
                    return orig

    return None

