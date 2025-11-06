from typing import Any, Dict, List, Tuple

from .nlp_pipeline import nlp
from .models import SubSlice, DocSlice, OrgResult
from .payload_utils import _build_org_map, _group_items_by_org
from .utils_text import _normalize_title
from .org_windows import _collect_org_windows_from_ents, _match_org_to_window
from .doc_type_match import _doc_type_key, _match_doc_type_headers, _compute_next_bounds_per_window
from .subdivision import _reparse_seg_text, _allowed_child_titles_for_item, _subdivide_seg_text_by_allowed_headers


def divide_body_by_org_and_docs_serieIII(
    doc_body,
    payload: Dict[str, Any],
    *,
    reparse_segments: bool = True,
    subdivide_children: bool = True,
) -> Tuple[List[OrgResult], List[Dict[str, Any]], Dict[str, Any]]:
    """
    Serie III splitter:
      1) Anchor top-level items to DOC_NAME_LABEL in body within org windows.
      2) Slice body into per-item segments [content_start:end).
      3) (Optional) Reparse each segment and subdivide by payload-approved child headers.
      4) If an item has no doc_name, create a segment spanning its org window and
         run normal children subdivision on it.
    """

    if not isinstance(payload, dict):
        return [], [], {"error": "invalid_payload"}

    # --- PREP STAGE ----------------------------------------------------------
    org_map = _build_org_map(payload)
    _allowed_orgs = [(o.get("text") or "").strip() for o in payload.get("orgs", [])]
    org_windows = _collect_org_windows_from_ents(doc_body, allowed_orgs=_allowed_orgs)
    doc_type_matches = _match_doc_type_headers(doc_body, payload, org_windows)
    matched_count = sum(1 for v in doc_type_matches.values() if v is not None)
    next_bounds = _compute_next_bounds_per_window(doc_type_matches, org_windows)
    items_by_org = _group_items_by_org(payload)

    results: List[OrgResult] = []
    total_slices = 0

    # --- MAIN LOOP -----------------------------------------------------------
    for org_id, org_name in org_map.items():
        win_idx, win_status = _match_org_to_window(org_name, org_windows)
        items = sorted(
            items_by_org.get(org_id, []),
            key=lambda it: (it.get("paragraph_id") is None, it.get("paragraph_id")),
        )
        org_result = OrgResult(org=org_name, status=win_status, docs=[])

        for item in items:
            title_raw = (item.get("doc_name") or {}).get("text") or ""
            title = _normalize_title(title_raw)
            key = _doc_type_key(item)
            mt = doc_type_matches.get(key)

            # A) items without doc_name → segment over org window and subdivide children
            if mt is None and not title and win_idx is not None:
                w = org_windows[win_idx]
                seg_text = doc_body.text[w["start"]:w["end"]]

                ds = DocSlice(
                    doc_name="(Empty)",
                    text=seg_text,
                    status="doc_children_segment",
                    confidence=0.5,
                )

                if reparse_segments and seg_text.strip():
                    ds.ents = _reparse_seg_text(seg_text)
                    if subdivide_children:
                        allowed = _allowed_child_titles_for_item(item)
                        ds.subs = _subdivide_seg_text_by_allowed_headers(seg_text, allowed)

                org_result.docs.append(ds)
                total_slices += 1
                continue

            # B) titled item but no header anchor
            if mt is None:
                org_result.docs.append(
                    DocSlice(doc_name=title, text="", status="doc_type_unanchored", confidence=0.0)
                )
                continue

            # C) titled item with header match
            start = mt["start"]
            win_for_item = mt.get("window_index")
            if win_for_item is not None:
                end = next_bounds.get(win_for_item, {}).get(start)
                if end is None:
                    end = org_windows[win_for_item]["end"]
            else:
                end = len(doc_body.text)

            header_end = mt.get("end", start)
            content_start = header_end
            seg_text = doc_body.text[content_start:end]

            ds = DocSlice(
                doc_name=title,
                text=seg_text,
                status="doc_type_segment",
                confidence=mt.get("confidence", 1.0),
            )

            if reparse_segments and seg_text.strip():
                ds.ents = _reparse_seg_text(seg_text)
                if subdivide_children:
                    allowed = _allowed_child_titles_for_item(item)
                    ds.subs = _subdivide_seg_text_by_allowed_headers(seg_text, allowed)

            org_result.docs.append(ds)
            total_slices += 1

        results.append(org_result)

    # --- SUMMARY -------------------------------------------------------------
    summary = {
        "orgs_in_payload": len(org_map),
        "org_windows_found": len(org_windows),
        "doc_type_headers_matched": matched_count,
        "doc_type_segments": total_slices,
        "segment_reparsed": bool(reparse_segments),
        "segments_with_subdivisions": sum(len(d.subs) for r in results for d in r.docs),
    }
    # --- MINIMAL OUTPUT ------------------------------------------------------
# Keep only: org, docs, doc_name, subs (title, body)

    results_min: List[Dict[str, Any]] = []

    for r in results:  # r: OrgResult
        org_name = getattr(r, "org", "").replace("\n", " ")
        org_item = {
            "org": org_name,
            "docs": [],
        }

        for d in getattr(r, "docs", []):  # d: DocSlice
            subs = getattr(d, "subs", None) or []
            for s in subs:
                if isinstance(s, dict):
                    title = s.get("title") or s.get("doc_name", "")
                    body = s.get("body", s.get("text", "")) or ""
                else:
                    title = getattr(s, "title", "") or getattr(s, "doc_name", "")
                    body = getattr(s, "body", None) or getattr(s, "text", "") or ""
                
                if not title:
                    continue

                body_with_title = f"{org_name}\n{title}\n\n{body}" if body else title

                org_item["docs"].append({
                    "title": title,
                    "text": body_with_title,
                })

        results_min.append(org_item)



   
    return results_min, summary
