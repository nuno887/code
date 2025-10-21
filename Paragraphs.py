import re
from spacy.language import Language
from spacy.util import filter_spans
import unicodedata

TEXT_LABEL = "DOC_TEXT"
PARAGRAPH_LABEL = "PARAGRAPH"

_term_rx = re.compile(r"[.!?]\s*$")  # strong terminators at end


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
    (?:[-•—–]            # dash/bullet
     |\d+\s*[\)\.]       # 1) or 1.
    )
    """,
    re.VERBOSE,
)

def _looks_like_list_start(s: str) -> bool:
    """Detect simple list/bullet starts to avoid false merges."""
    return bool(_list_start_rx.search(s))


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
            last_piece = text[ent.start_char:ent.end_char]

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
                #if _looks_like_list_start(nxt_slice):
                 #   break  # do not merge into lists/bullets

                ends_like_sentence = _ends_with_terminator(last_piece)
                nxt_lead = _leading_alpha_case_or_none(nxt_slice)

                # If current ends with . ! ? but next starts lowercase, treat as wrapped continuation.
                if ends_like_sentence and nxt_lead == 'lower':
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
