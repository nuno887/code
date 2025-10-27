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
     |\d+\s*[\)\.]       # 1) or 1.
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


# ---- NEW: leader + page split support ---------------------------------------

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

    spans = []
    i = 0
    n = len(ents)

    while i < n:
        ent = ents[i]
        if ent.label_ == TEXT_LABEL and _starts_with_upper(text[ent.start_char:ent.end_char]):
            start = ent.start_char
            end = ent.end_char
            last_piece = text[start:end]

            # --- NEW: handle TOC-style "leader + page" breaks inside this same TEXT entity ---
            local_start = start
            local_slice = text[local_start:end]

            while True:
                cut = _first_leader_page_break_index(local_slice)
                if cut is None:
                    break

                # Emit a paragraph up to the end of the page number
                cut_abs = local_start + cut
                span = doc.char_span(local_start, cut_abs, label=PARAGRAPH_LABEL, alignment_mode="contract")
                if span is not None:
                    spans.append(span)

                # Advance to the next non-space char after the cut (the next TOC item)
                local_start = cut_abs
                while local_start < end and text[local_start].isspace():
                    local_start += 1
                local_slice = text[local_start:end]

            # If we produced at least one intra-entity paragraph and consumed the whole slice, skip normal merge.
            if local_start > start:
                if local_start < end:
                    # There is remaining text in this entity; continue with normal merging from here
                    start = local_start
                    last_piece = text[start:end]
                else:
                    # Entire slice was consumed by intra-entity splits; move on to next entity
                    i += 1
                    continue
            # --- END NEW ---

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

                # Re-enable guard: do not merge into lists/bullets
                if _looks_like_list_start(nxt_slice):
                    break

                ends_like_sentence = _ends_with_terminator(last_piece)
                nxt_lead = _leading_alpha_case_or_none(nxt_slice)

                # If current ends with . ! ? (or leader+page) but next starts lowercase, treat as wrapped continuation.
                # NOTE: Leader+page considered a HARD stop at end-of-line by _ends_with_terminator;
                # this lowercase exception should NOT override a forced split that already happened inside the same entity.
                if ends_like_sentence and nxt_lead == 'lower' and not _ends_with_ellipsis(last_piece):
                    ends_like_sentence = False

                if ends_like_sentence:
                    break

                # extend paragraph to include next TEXT line
                end = nxt.end_char
                last_piece = nxt_slice
                j = k

            span = doc.char_span(start, end, label=PARAGRAPH_LABEL, alignment_mode="contract")
            if span is not None:
                spans.append(span)
            i = j + 1
        else:
            i += 1

    if spans:
        doc.ents = filter_spans(list(doc.ents) + spans)
    return doc
