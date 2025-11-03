from typing import Any, Dict, List, Optional, Set, Tuple
from .nlp_pipeline import nlp
from .models import SubSlice
from .utils_text import (
    _normalize_title, _tighten, _letters_only,
    _ocr_clean, _char_ngrams,
    LETTERS_MIN_RATIO, NGRAM_N, NGRAM_JACCARD_MIN, MIN_LEN_FOR_NGRAMS
)



def _reparse_seg_text(seg_text: str) -> List[Tuple[str, str, int, int]]:
    doc = nlp(seg_text)

    out: List[Tuple[str, str, int, int]] = []
    for e in doc.ents:
        label = getattr(e, "label_", "")
        out.append((label, e.text, e.start_char, e.end_char))
    return out


def _allowed_child_titles_for_item(item: Dict[str, Any]) -> Set[str]:
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


def _pick_canonical_from_block(block_titles: List[str], allowed_titles: Set[str]) -> Optional[str]:
    if not allowed_titles:
        return None

    prepared_allowed = []
    for t in allowed_titles:
        t_clean = _ocr_clean(t)
        t_norm  = _normalize_title(t_clean)
        t_tight = _tighten(t_norm)
        t_letters = _letters_only(t_norm)
        prepared_allowed.append((t, t_norm, t_tight, t_letters))

    block_join_raw = "\n".join(block_titles)
    block_join = _ocr_clean(block_join_raw)
    bj_norm   = _normalize_title(block_join)
    bj_tight  = _tighten(bj_norm)
    bj_letters= _letters_only(bj_norm)

    for orig, an, at, al in prepared_allowed:
        if bj_norm and bj_norm == an:
            return orig
    for orig, an, at, al in prepared_allowed:
        if bj_tight and bj_tight == at:
            return orig
    for orig, an, at, al in prepared_allowed:
        if bj_letters and bj_letters == al:
            return orig
    for orig, an, at, al in prepared_allowed:
        if not bj_letters or not al:
            continue
        if bj_letters in al or al in bj_letters:
            shorter = min(len(bj_letters), len(al))
            longer  = max(len(bj_letters), len(al))
            if longer and (shorter / longer) >= LETTERS_MIN_RATIO:
                return orig

    if len(bj_letters) >= MIN_LEN_FOR_NGRAMS:
        bj_ngrams = _char_ngrams(bj_letters, NGRAM_N)
        best = (None, 0.0)
        for orig, an, at, al in prepared_allowed:
            if len(al) < MIN_LEN_FOR_NGRAMS:
                continue
            al_ngrams = _char_ngrams(al, NGRAM_N)
            if not al_ngrams:
                continue
            inter = len(bj_ngrams & al_ngrams)
            union = len(bj_ngrams | al_ngrams)
            j = inter / union if union else 0.0
            if j >= NGRAM_JACCARD_MIN and j > best[1]:
                best = (orig, j)
        if best[0] is not None:
            return best[0]

    for bt_raw in block_titles:
        bt_norm    = _normalize_title(_ocr_clean(bt_raw))
        bt_tight   = _tighten(bt_norm)
        bt_letters = _letters_only(bt_norm)

        for orig, an, at, al in prepared_allowed:
            if bt_norm == an:
                return orig
        for orig, an, at, al in prepared_allowed:
            if bt_tight and bt_tight == at:
                return orig
        for orig, an, at, al in prepared_allowed:
            if bt_letters and bt_letters == al:
                return orig
        for orig, an, at, al in prepared_allowed:
            if not bt_letters or not al:
                continue
            if bt_letters in al or al in bt_letters:
                shorter = min(len(bt_letters), len(al))
                longer  = max(len(bt_letters), len(al))
                if longer and (shorter / longer) >= LETTERS_MIN_RATIO:
                    return orig

        if len(bt_letters) >= MIN_LEN_FOR_NGRAMS:
            bt_ngrams = _char_ngrams(bt_letters, NGRAM_N)
            best = (None, 0.0)
            for orig, an, at, al in prepared_allowed:
                if len(al) < MIN_LEN_FOR_NGRAMS:
                    continue
                al_ngrams = _char_ngrams(al, NGRAM_N)
                if not al_ngrams:
                    continue
                inter = len(bt_ngrams & al_ngrams)
                union = len(bt_ngrams | al_ngrams)
                j = inter / union if union else 0.0
                if j >= NGRAM_JACCARD_MIN and j > best[1]:
                    best = (orig, j)
            if best[0] is not None:
                return best[0]

    return None


def _subdivide_seg_text_by_allowed_headers(seg_text: str, allowed_titles: Set[str]) -> List[SubSlice]:
    doc = nlp(seg_text)
    ents = sorted(list(doc.ents), key=lambda e: e.start_char)


    header_blocks: List[Dict[str, Any]] = []
    current_block: List[Any] = []
    for e in ents:
        if getattr(e, "label_", None) == "DOC_NAME_LABEL":
            if current_block:
                prev_e = current_block[-1]
                gap = e.start_char - prev_e.end_char
                current_block.append(e)

            else:
                current_block = [e]

        else:
            if current_block:
                start = current_block[0].start_char
                end = current_block[-1].end_char
                hb = {
                    "headers": current_block[:],
                    "start": start,
                    "end": end,
                    "titles": [_normalize_title(h.text) for h in current_block],
                }
                header_blocks.append(hb)
                current_block = []
    if current_block:
        start = current_block[0].start_char
        end = current_block[-1].end_char
        hb = {
            "headers": current_block[:],
            "start": start,
            "end": end,
            "titles": [_normalize_title(h.text) for h in current_block],
        }
        header_blocks.append(hb)

    approved: List[Dict[str, Any]] = []
    for hb in header_blocks:
        canon = _pick_canonical_from_block(hb["titles"], allowed_titles)
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
