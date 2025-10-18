# DocText.py
from spacy.language import Language
from spacy.util import filter_spans

TEXT_LABEL = "DOC_TEXT"

def _line_has_entity_overlap(doc, start, end) -> bool:
    # True if any existing entity overlaps the [start, end) range
    for ent in doc.ents:
        if ent.start_char < end and ent.end_char > start:
            return True
    return False

@Language.component("doc_text_entity")
def text_line_entity(doc):
    text = doc.text
    lines = text.splitlines(keepends=True)

    spans = []
    pos = 0
    for ln in lines:
        line_end = pos + len(ln)
        # strip trailing newline from the visual extent
        end_idx = line_end - (1 if ln.endswith("\n") else 0)

        # compute start at first non-space, compute end without trailing spaces
        content = ln[:-1] if ln.endswith("\n") else ln
        stripped = content.strip()
        if stripped:  # non-empty line
            leading = len(content) - len(content.lstrip())
            trailing = len(content) - len(content.rstrip())
            start_idx = pos + leading
            end_idx = line_end - (1 if ln.endswith("\n") else 0) - trailing

            if start_idx < end_idx and not _line_has_entity_overlap(doc, start_idx, end_idx):
                span = doc.char_span(start_idx, end_idx, label=TEXT_LABEL, alignment_mode="contract")
                if span is not None:
                    spans.append(span)

        pos = line_end

    if spans:
        doc.ents = filter_spans(list(doc.ents) + spans)
    return doc


