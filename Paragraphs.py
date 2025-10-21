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
            # Concatenate TEXT ents until we hit a strong terminator or a non-TEXT entity
            while not _ends_with_terminator(last_piece):
                k = j + 1
                if k >= n:
                    break
                nxt = ents[k]
                if nxt.label_ != TEXT_LABEL:
                    break
                # extend paragraph to include next TEXT line
                end = nxt.end_char
                last_piece = text[nxt.start_char:nxt.end_char]
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

