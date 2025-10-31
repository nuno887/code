from typing import Any, Dict, List, Optional, Set, Tuple

from .types import SubSlice
from .normalizers import _normalize_title_for_match
from .matching import pick_canonical_from_block
from .debug_seg_dump import dump_seg_bundle


# -------- Debug helpers --------
DEBUG = False

def dbg(tag: str, **kv: Any):
    if not DEBUG:
        return
    parts = ", ".join(f"{k}={repr(v)}" for k, v in kv.items())
    print(f"[{tag}] {parts}")

def _repr_gap_text(s: str, limit: int = 80) -> str:
    # Make whitespace/newlines visible and truncate
    r = s.replace("\n", "\\n").replace("\t", "\\t")
    return (r[:limit] + "…") if len(r) > limit else r

# --------------------------------

def reparse_seg_text(nlp, seg_text: str) -> List[Tuple[str, str, int, int]]:
    doc = nlp(seg_text)
    

    # AFTER: doc = nlp(seg_text)
    for i, e in enumerate(doc.ents):
        if getattr(e, "label_", "") == "DOC_NAME_LABEL":
            norm = _normalize_title_for_match(e.text)
            dbg("HEADER", i=i, span=(e.start_char, e.end_char), raw=e.text[:100], norm=norm)

    out: List[Tuple[str, str, int, int]] = []
    for e in doc.ents:
        label = getattr(e, "label_", "")
        out.append((label, e.text, e.start_char, e.end_char))
    dbg("REPARSE_DONE", ents=len(out))
    return out


def allowed_child_titles_for_item(item: Dict[str, Any]) -> Set[str]:
    titles: Set[str] = set()

    def _tight_key(s: str) -> str:
        return _normalize_title_for_match(s).replace(" ", "").lower()

    for t in (item.get("allowed_children") or []):
        t_norm = _normalize_title_for_match(str(t))
        if t_norm:
            titles.add(t_norm)

    for ch in (item.get("children") or []):
        if not isinstance(ch, dict):
            continue
        txt = ""
        if isinstance(ch.get("doc_name"), dict) and ch["doc_name"].get("text"):
            txt = ch["doc_name"]["text"]
        elif "text" in ch and ch.get("text"):
            txt = ch["text"]
        elif "child" in ch and ch.get("child"):
            raw = str(ch["child"])
            txt = " ".join(raw.split())
        t_norm = _normalize_title_for_match(txt or "")
        if t_norm:
            titles.add(t_norm)

    for b in (item.get("bodies") or []):
        if not isinstance(b, dict):
            continue
        if isinstance(b.get("doc_name"), dict) and b["doc_name"].get("text"):
            t_norm = _normalize_title_for_match(b["doc_name"]["text"])
            if t_norm:
                titles.add(t_norm)

    dedup: Set[str] = set()
    out: Set[str] = set()
    for t in titles:
        k = _tight_key(t)
        if k in dedup:
            continue
        dedup.add(k)
        out.add(t)

    dbg("ALLOWED_TITLES", count=len(out), examples=list(out)[:8])
    return out


def subdivide_seg_text_by_allowed_headers(nlp, seg_text: str, allowed_titles: Set[str]) -> List[SubSlice]:
    HEADER_LABEL = "DOC_NAME_LABEL"
    BOUNDARY_LABELS = {"PARAGRAPH", "DOC_TEXT"}  # you can add more if needed

    dbg("SUBDIVIDE_START", seg_len=len(seg_text), allowed_count=len(allowed_titles))

    doc = nlp(seg_text)
    dump_seg_bundle(seg_text=seg_text, doc=doc, out_dir="debug_out", tag="subdivide", allowed_titles=allowed_titles)
    ents = sorted(list(doc.ents), key=lambda e: e.start_char)

    # Precompute a simple list for gap scans
    ent_index = [
        (e.start_char, e.end_char, getattr(e, "label_", None), e.text)
        for e in ents
    ]
    dbg("ENTS_TOTAL", count=len(ent_index))

    def ents_in_gap(g0: int, g1: int) -> List[Tuple[int, int, str, str]]:
        if g1 <= g0:
            return []
        out = []
        for s, e, lab, txt in ent_index:
            if s >= g1:
                break
            if s >= g0 and e <= g1:
                out.append((s, e, lab, txt))
        return out

    header_blocks: List[Dict[str, Any]] = []
    current_block: List[Any] = []

    def _flush_block(reason: str):
        nonlocal current_block
        if current_block:
            start = current_block[0].start_char
            end = current_block[-1].end_char
            titles_norm = [_normalize_title_for_match(h.text) for h in current_block]
            header_blocks.append({
                "headers": current_block[:],
                "start": start,
                "end": end,
                "titles": titles_norm,
            })
            dbg("FLUSH_BLOCK",
                reason=reason,
                block_index=len(header_blocks)-1,
                start=start,
                end=end,
                headers_raw=[h.text for h in current_block],
                titles=titles_norm
            )
            current_block = []

    for e in ents:
        lab = getattr(e, "label_", None)
        # Log first few entities for context
        if DEBUG:
            dbg("ENT", label=lab, span=(e.start_char, e.end_char), text=e.text[:80])

        if lab == HEADER_LABEL:
            if current_block:
                # analyze the gap between last header in block and this header
                gap_start = current_block[-1].end_char
                gap_end = e.start_char
                gap_text = seg_text[gap_start:gap_end] if gap_end > gap_start else ""
                gap_struct = ents_in_gap(gap_start, gap_end)
                gap_labels = [g[2] for g in gap_struct]
                has_boundary_ent = any(gl in BOUNDARY_LABELS for gl in gap_labels)
                has_newline = ("\n" in gap_text)
                has_non_ws = bool(gap_text.strip())

                dbg("GAP_CHECK",
                    prev_end=gap_start,
                    next_start=gap_end,
                    gap_len=(gap_end - gap_start),
                    gap_preview=_repr_gap_text(gap_text),
                    gap_labels=gap_labels,
                    has_boundary_ent=has_boundary_ent,
                    has_newline=has_newline,
                    has_non_ws=has_non_ws
                )

                # Decision priority:
                # 1) structural boundary entity in gap → new block
                # 2) newline in gap → new block
                # 3) else: same header block (multiline header)
                if has_boundary_ent:
                    _flush_block(reason="boundary_entity_in_gap")
                    current_block = [e]
                elif has_newline:
                    _flush_block(reason="newline_in_gap")
                    current_block = [e]
                else:
                    current_block.append(e)
                    dbg("BLOCK_APPEND", header_text=e.text)
            else:
                current_block = [e]
                dbg("BLOCK_START", header_text=e.text, start=e.start_char)
        else:
            # Any non-header entity can delimit a header block as in original behavior
            if current_block:
                dbg("NON_HEADER_FLUSH", seen_label=lab, text=e.text[:60])
            _flush_block(reason="non_header_seen")
        
    # =====================================================================================
    # AFTER: ents and ent_index are computed (right after building `ents`)
    for i, e in enumerate(doc.ents):
        if getattr(e, "label_", "") == "DOC_NAME_LABEL":
            norm = _normalize_title_for_match(e.text)
            allowed = norm in allowed_titles
            dbg("HEADER", i=i, span=(e.start_char, e.end_char), raw=e.text[:100], norm=norm, allowed=allowed)

    # OPTIONAL tiny summary
    hdr_count = sum(1 for e in doc.ents if getattr(e, "label_", "") == "DOC_NAME_LABEL")
    hdr_match = sum(
        1 for e in doc.ents
        if getattr(e, "label_", "") == "DOC_NAME_LABEL" and _normalize_title_for_match(e.text) in allowed_titles
    )
    dbg("HEADER_SUMMARY", total=hdr_count, matched=hdr_match)


    
    # =====================================================================================

    _flush_block(reason="end_of_ents")

    dbg("HEADER_BLOCKS_SUMMARY", count=len(header_blocks))
    for i, hb in enumerate(header_blocks):
        dbg("HEADER_BLOCK",
            i=i,
            start=hb["start"],
            end=hb["end"],
            headers_raw=[h.text for h in hb["headers"]],
            titles=hb["titles"]
        )

    # Approval
    approved: List[Dict[str, Any]] = []
    for i, hb in enumerate(header_blocks):
        canon = pick_canonical_from_block(hb["titles"], allowed_titles)
        dbg("APPROVAL", block=i, titles=hb["titles"], canonical=canon)
        if canon is not None:
            approved.append({**hb, "canonical": canon})

    subs: List[SubSlice] = []
    if not approved:
        dbg("NO_APPROVED_BLOCKS")
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
        dbg("SUBSLICE", index=0, title=(headers_texts[0] if headers_texts else ""),
            span=(body_start, body_end), body_len=len(body_text))
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
        dbg("SUBSLICE",
            index=i,
            title=hb["canonical"],
            span=(header_end, next_start),
            body_len=len(body_text),
            preview=_repr_gap_text(body_text[:120])
        )

    dbg("SUBDIVIDE_DONE", subslices=len(subs))
    return subs
