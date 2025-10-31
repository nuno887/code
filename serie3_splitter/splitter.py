from typing import Any, Dict, List, Tuple

from .types import OrgResult, DocSlice
from .windows import collect_org_windows_from_ents, match_org_to_window
from .headers import match_doc_type_headers, compute_next_bounds_per_window, doc_type_key
from .subdivide import reparse_seg_text, subdivide_seg_text_by_allowed_headers, allowed_child_titles_for_item
from .normalizers import _normalize_title_for_match

# ================================================================
DEBUG = True
def dbg(tag: str, **kv):
    if not DEBUG:
        return
    parts = ", ".join(f"{k}={repr(v)}" for k, v in kv.items())
    print(f"[{tag}] {parts}")

# ================================================================


def _build_org_map(payload: Dict[str, Any]) -> Dict[int, str]:
    out: Dict[int, str] = {}
    for o in payload.get("orgs", []):
        oid = o.get("id")
        name = (o.get("text") or "").strip()
        if oid is not None:
            out[oid] = name
    if not out:
        out[-1] = "(Sem organização)"
    return out


def _group_items_by_org(payload: Dict[str, Any]) -> Dict[int, List[Dict[str, Any]]]:
    grouped: Dict[int, List[Dict[str, Any]]] = {}
    for it in payload.get("items", []):
        for oid in it.get("org_ids", []):
            grouped.setdefault(oid, []).append(it)
    return grouped


def divide_body_by_org_and_docs_serieIII(
    doc_body,
    payload: Dict[str, Any],
    *,
    nlp=None,
    reparse_segments: bool = True,
    subdivide_children: bool = True,
) -> Tuple[List[OrgResult], Dict[str, Any]]:
    """Public entry point. Pass the same `nlp` pipeline you used to create `doc_body`.
    If `nlp` is None, we will skip reparsing/subdivision (to avoid relying on doc_body.vocab.nlp).
    """
    if not isinstance(payload, dict):
        return [], {"error": "invalid_payload"}

    org_map = _build_org_map(payload)
    _allowed_orgs = [(o.get("text") or "").strip() for o in payload.get("orgs", [])]
    org_windows = collect_org_windows_from_ents(doc_body, allowed_orgs=_allowed_orgs)

    doc_type_matches = match_doc_type_headers(doc_body, payload, org_windows)
    matched_count = sum(1 for v in doc_type_matches.values() if v is not None)

    next_bounds = compute_next_bounds_per_window(doc_type_matches, org_windows)
    items_by_org = _group_items_by_org(payload)

    results: List[OrgResult] = []
    total_slices = 0

    for org_id, org_name in org_map.items():
        win_idx, win_status = match_org_to_window(org_name, org_windows)
        items = sorted(
            items_by_org.get(org_id, []),
            key=lambda it: (it.get("paragraph_id") is None, it.get("paragraph_id")),
        )

        org_result = OrgResult(org=org_name, status=win_status, docs=[])

        for item in items:
            title_raw = (item.get("doc_name") or {}).get("text") or ""
            title = _normalize_title_for_match(title_raw)
            key = doc_type_key(item)

            mt = doc_type_matches.get(key)
# =========================================================================
            # After: mt = doc_type_matches.get(key)
            if mt is not None:
                matched_raw = (doc_body.text[mt["start"]:mt["end"]] or "")[:200].replace("\n", "\\n")
                matched_norm = _normalize_title_for_match(matched_raw)
                parent_norm = title  # already normalized from above
                allowed_children = allowed_child_titles_for_item(item)
                is_parent_match = (matched_norm == parent_norm)
                is_child_like = (matched_norm in allowed_children)

                dbg("ANCHOR_MATCH",
                    item_title_raw=(item.get("doc_name") or {}).get("text") or "",
                    item_title_norm=parent_norm,
                    matched_preview=matched_raw,
                    matched_norm=matched_norm,
                    is_parent_match=is_parent_match,
                    is_child_like=is_child_like,
                    window_index=mt.get("window_index"),
                    confidence=mt.get("confidence", 1.0),
                )


# =========================================================================

            # Branch: items without doc_name → segment over org window and subdivide children
            if mt is None and not title and win_idx is not None:
                w = org_windows[win_idx]
                seg_text = doc_body.text[w["start"]:w["end"]]

                ds = DocSlice(
                    doc_name="(Preambulo)",
                    text=seg_text,
                    status="doc_children_segment",
                    confidence=0.5,
                )

                if seg_text.strip() and nlp is not None:
                    if reparse_segments:
                        ds.ents = reparse_seg_text(nlp, seg_text)
                    if subdivide_children:
                        allowed = allowed_child_titles_for_item(item)
                        ds.subs = subdivide_seg_text_by_allowed_headers(nlp, seg_text, allowed)

                org_result.docs.append(ds)
                total_slices += 1
                continue

            # No anchor found for titled item → unanchored
            if mt is None:
                org_result.docs.append(
                    DocSlice(doc_name=title, text="", status="doc_type_unanchored", confidence=0.0)
                )
                continue

            # Determine slice [start:end) using next header in same window
            start = mt["start"]
            win_for_item = mt.get("window_index")
            if win_for_item is not None:
                end = next_bounds.get(win_for_item, {}).get(start)
                if end is None:
                    end = org_windows[win_for_item]["end"]
            else:
                end = len(doc_body.text)

            header_end = mt.get("end", start)
            content_start = header_end  # exclude header text from segment
            seg_text = doc_body.text[content_start:end]

# =========================================================================
            # After start/end/header_end/content_start have been computed, just before seg_text
            win_for_item = mt.get("window_index")
            win_start = org_windows[win_for_item]["start"] if win_for_item is not None else None
            win_end = org_windows[win_for_item]["end"] if win_for_item is not None else None

            dbg("SLICE_BOUNDS",
                start=start,
                header_end=mt.get("end", start),
                content_start=header_end,
                end=end,
                window_start=win_start,
                window_end=win_end,
                slice_len=(end - header_end),
            )

# =========================================================================
# =========================================================================
            # After: seg_text = doc_body.text[content_start:end]
            allowed = allowed_child_titles_for_item(item)
            norm_seg = _normalize_title_for_match(seg_text)
            norm_allowed = [t for t in allowed]  # already normalized by allowed_child_titles_for_item
            found_children = [t for t in norm_allowed if t and t in norm_seg]

            dbg("CHILD_SCAN",
                allowed_count=len(norm_allowed),
                found_count=len(found_children),
                found_examples=found_children[:5],
            )

            # Optional guardrail: warn if we expected many children but found none
            if len(norm_allowed) >= 2 and len(found_children) == 0:
                dbg("WIDEN_SUGGEST",
                    reason="No allowed children detected inside seg_text",
                    hint="Anchor may be on a child; consider widening to window or including header text",
                )

# =========================================================================

            ds = DocSlice(
                doc_name=title,
                text=seg_text,
                status="doc_type_segment",
                confidence=mt.get("confidence", 1.0),
            )

            if seg_text.strip() and nlp is not None:
                if reparse_segments:
                    ds.ents = reparse_seg_text(nlp, seg_text)
                if subdivide_children:
                    allowed = allowed_child_titles_for_item(item)
                    ds.subs = subdivide_seg_text_by_allowed_headers(nlp, seg_text, allowed)

            org_result.docs.append(ds)
            total_slices += 1

# =============================================================================
            # Right before subdividing the preamble segment
            allowed = allowed_child_titles_for_item(item)
            norm_seg = _normalize_title_for_match(seg_text)
            found_children = [t for t in allowed if t and t in norm_seg]

            dbg("PREAMBLE_CHILD_SCAN",
                allowed_count=len(allowed),
                found_count=len(found_children),
                found_examples=found_children[:5],
            )


# =============================================================================

        results.append(org_result)

    summary = {
        "orgs_in_payload": len(org_map),
        "org_windows_found": len(org_windows),
        "doc_type_headers_matched": matched_count,
        "doc_type_segments": total_slices,
        "segment_reparsed": bool(reparse_segments and (nlp is not None)),
        "segments_with_subdivisions": sum(len(d.subs) for r in results for d in r.docs),
    }

    return results, summary