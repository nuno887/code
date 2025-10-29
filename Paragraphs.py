import re
from spacy.language import Language
from spacy.util import filter_spans
import unicodedata

TEXT_LABEL = "DOC_TEXT"
PARAGRAPH_LABEL = "PARAGRAPH"

# Treat ., !, ?, ellipsis, or long ... as a terminator,
# AND allow optional spaces + page number (e.g., "................ 10") before end-of-line.
_term_rx = re.compile(r"(?:[.!?]|…+|\.{3,})(?:\s*\d+[A-Za-z]?)?\s*$")  # strong terminators incl. leaders+page at end

def _starts_with_upper(s: str) -> bool:
    # Skip leading spaces and opening punctuation/symbols (quotes, dashes, brackets…)
    for ch in s.lstrip():
        if ch.isalpha():
            return ch == ch.upper()
        if ch.isdigit():
            return True  # allow numeric-start paragraphs if needed
        cat = unicodedata.category(ch)
        if ch.isspace():
            continue
        # Unicode categories: P* = punctuation, S* = symbol
        if cat.startswith(("P", "S")):
            # keep skipping leading punctuation/symbols like “ ‘ ( [ { — -
            continue
        # Any other leading character → not a valid paragraph start
        return False
    return False

def _ends_with_terminator(s: str) -> bool:
    return bool(_term_rx.search(s.strip()))

def _leading_alpha_case_or_none(s: str):
    """
    After skipping spaces and opening punctuation/symbols, return:
      - 'lower' if first significant alpha is lowercase
      - 'upper' if uppercase
      - None   if first significant char is non-alpha or string is empty
    """
    for ch in s.lstrip():
        if ch.isalpha():
            return 'upper' if ch == ch.upper() else 'lower'
        cat = unicodedata.category(ch)
        if ch.isspace():
            continue
        if cat.startswith(("P", "S")):  # opening punctuation/symbol
            continue
        return None
    return None

_list_start_rx = re.compile(
    r"""
    ^\s*
    (?:[•—–]            # dash/bullet
     |\d+\s*[\)\.]      # 1) or 1.
    )
    """,
    re.VERBOSE,
)

# put near the other small helpers
_ellipsis_eol_rx = re.compile(r"(?:…+|\.{3,})\s*$")
def _ends_with_ellipsis(s: str) -> bool:
    return bool(_ellipsis_eol_rx.search(s.strip()))

def _looks_like_list_start(s: str) -> bool:
    """Detect simple list/bullet starts to avoid false merges."""
    return bool(_list_start_rx.search(s))

# ---- leader + page split support --------------------------------------------

# Accept 3+ dots (with optional spaces), repeated ellipses, or middle-dots as a "leader" run.
_LEADER_RUN = r"(?:(?:\.\s*){3,}|…+|(?:·\s*){3,})"

# A leader run followed by spaces + a page number (optionally one trailing letter like 10A)
_leader_page_break_rx = re.compile(rf"{_LEADER_RUN}\s*(\d+[A-Za-z]?)")

def _first_leader_page_break_index(s: str):
    """
    If there's a leader run + page number and there's more non-space text after it,
    return the index (in s) right AFTER the page number (i.e., where we should split).
    Otherwise return None.
    """
    m = _leader_page_break_rx.search(s)
    if not m:
        return None
    end_num = m.end(1)  # end of the page number group
    # Only split if there's more text after the page number (same physical line/entity)
    if s[end_num:].strip():
        return end_num
    return None

# -----------------------------------------------------------------------------

@Language.component("paragraph_entity")
def paragraph_entity(doc):
    text = doc.text
    ents = sorted(doc.ents, key=lambda e: e.start_char)

    # --- PROTECTION: never overlap DOC_NAME_LABEL or SERIE_III ----------------
    PROTECTED_LABELS = {"DOC_NAME_LABEL", "SERIE_III"}
    protected_spans = sorted(
        [(e.start_char, e.end_char) for e in doc.ents if e.label_ in PROTECTED_LABELS]
    )

    def first_overlap(s: int, e: int):
        """Return (a,b) of the first protected span that overlaps [s,e), else None."""
        for a, b in protected_spans:
            if not (e <= a or s >= b):
                return (a, b)
        return None

    def gap_has_protected(left_e: int, right_s: int) -> bool:
        """Any protected span touching the gap [left_e, right_s]?"""
        for a, b in protected_spans:
            if a < right_s and b > left_e:
                return True
        return False

    def clip_to_before_protected(s: int, e: int):
        """
        If [s,e) overlaps a protected span (a,b), return (s, min(e,a), True).
        Else, return (s, e, False).
        """
        ov = first_overlap(s, e)
        if ov is None:
            return s, e, False
        a, _b = ov
        if s < a:
            return s, min(e, a), True
        # protected starts at or before s -> do not emit anything
        return s, s, True
    # --------------------------------------------------------------------------

    spans = []
    i = 0
    n = len(ents)

    while i < n:
        ent = ents[i]
        if ent.label_ == TEXT_LABEL and _starts_with_upper(text[ent.start_char:ent.end_char]):
            start = ent.start_char
            end = ent.end_char
            last_piece = text[start:end]

            # --- Intra-entity TOC "leader + page" splits ----------------------
            local_start = start
            local_slice = text[local_start:end]

            while True:
                cut = _first_leader_page_break_index(local_slice)
                if cut is None:
                    break

                cut_abs = local_start + cut
                s_emit, e_emit, clipped = clip_to_before_protected(local_start, cut_abs)
                if e_emit > s_emit:
                    span = doc.char_span(s_emit, e_emit, label=PARAGRAPH_LABEL, alignment_mode="contract")
                    if span is not None:
                        spans.append(span)

                # Advance to the next non-space char after the cut
                local_start = cut_abs
                while local_start < end and text[local_start].isspace():
                    local_start += 1
                local_slice = text[local_start:end]

                # If we clipped due to protection starting exactly at or before local_start,
                # stop splitting here; remaining content will be handled by the normal loop.
                if clipped and local_start >= end:
                    break

            # If we produced at least one intra-entity paragraph and consumed the whole slice, skip normal merge.
            if local_start > start:
                if local_start < end:
                    start = local_start
                    last_piece = text[start:end]
                else:
                    i += 1
                    continue
            # --- END intra-entity splits --------------------------------------

            j = i
            # Concatenate TEXT ents; allow continuation if next line starts lowercase.
            while True:
                k = j + 1
                if k >= n:
                    break
                nxt = ents[k]
                if nxt.label_ != TEXT_LABEL:
                    break

                nxt_slice = text[nxt.start_char:nxt.end_char]

                # do not merge into lists/bullets
                if _looks_like_list_start(nxt_slice):
                    break

                ends_like_sentence = _ends_with_terminator(last_piece)
                nxt_lead = _leading_alpha_case_or_none(nxt_slice)

                # If current ends with . ! ? (or leader+page) but next starts lowercase, treat as wrapped continuation.
                if ends_like_sentence and nxt_lead == 'lower' and not _ends_with_ellipsis(last_piece):
                    ends_like_sentence = False

                if ends_like_sentence:
                    break

                # STOP merging if we'd cross into a protected span in the gap or by extending
                if gap_has_protected(end, nxt.start_char):
                    break
                if first_overlap(start, nxt.end_char) is not None:
                    break

                # extend paragraph to include next TEXT line
                end = nxt.end_char
                last_piece = nxt_slice
                j = k

            # Before emitting final paragraph, clip to avoid overlap with protected
            s_emit, e_emit, clipped = clip_to_before_protected(start, end)
            if e_emit > s_emit:
                span = doc.char_span(s_emit, e_emit, label=PARAGRAPH_LABEL, alignment_mode="contract")
                if span is not None:
                    spans.append(span)

            i = j + 1
        else:
            i += 1

    if spans:
        doc.ents = filter_spans(list(doc.ents) + spans)
    return doc
