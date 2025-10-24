from typing import Any, Dict, List, Optional, Set, Tuple

from .types import SubSlice
from .normalizers import _normalize_title
from .matching import pick_canonical_from_block


def reparse_seg_text(nlp, seg_text: str) -> List[Tuple[str, str, int, int]]:
    doc = nlp(seg_text)
    out: List[Tuple[str, str, int, int]] = []
    for e in doc.ents:
        label = getattr(e, "label_", "")
        out.append((label, e.text, e.start_char, e.end_char))
    return out


def allowed_child_titles_for_item(item: Dict[str, Any]) -> Set[str]:
    titles: Set[str] = set()

    def _tight_key(s: str) -> str:
        return _normalize_title(s).replace(" ", "").lower()

    for t in (item.get("allowed_children") or []):
        t_norm = _normalize_title(str(t))
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
        t_norm = _normalize_title(txt or "")
        if t_norm:
            titles.add(t_norm)

    for b in (item.get("bodies") or []):
        if not isinstance(b, dict):
            continue
        if isinstance(b.get("doc_name"), dict) and b["doc_name"].get("text"):
            t_norm = _normalize_title(b["doc_name"]["text"])
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
    return out


def subdivide_seg_text_by_allowed_headers(nlp, seg_text: str, allowed_titles: Set[str]) -> List[SubSlice]:
    doc = nlp(seg_text)
    ents = sorted(list(doc.ents), key=lambda e: e.start_char)

    header_blocks: List[Dict[str, Any]] = []
    current_block: List[Any] = []

    def _flush_block():
        nonlocal current_block
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

    last_header_end: Optional[int] = None

    for e in ents:
        if getattr(e, "label_", None) == "DOC_NAME_LABEL":
            if current_block:
                # If there is non-whitespace text between the last header and this one,
                # the previous block should be closed and a new one started.
                gap_has_text = False
                gap_start = current_block[-1].end_char
                gap_end = e.start_char
                if gap_end > gap_start:
                    gap_has_text = bool(seg_text[gap_start:gap_end].strip())

                if gap_has_text:
                    _flush_block()
                    current_block = [e]
                else:
                    # Still in the same logical header block (e.g., multiline header)
                    current_block.append(e)
            else:
                current_block = [e]
            last_header_end = e.end_char
        else:
            # Seeing a non-header entity can also delimit the header block.
            _flush_block()

    _flush_block()
    # --- the rest of your function stays the same ---
    approved: List[Dict[str, Any]] = []
    for hb in header_blocks:
        canon = pick_canonical_from_block(hb["titles"], allowed_titles)
        if canon is not None:
            approved.append({**hb, "canonical": canon})

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
    return subs

