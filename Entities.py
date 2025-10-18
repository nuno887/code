from spacy.pipeline import EntityRuler
import re
from spacy.language import Language
from spacy.util import filter_spans


OPTIONS = {"colors": {
    "Sumario": "#ffd166",
    "ORG_LABEL": "#6e77b8",
    "ORG_WITH_STAR_LABEL": "#6fffff",
    "DOC_NAME_LABEL": "#b23bbd"
    
    }}


RULER_PATTERNS = [

{"label": "Sumario", "pattern": "### **Sumário**"}

]

_pattern_allcaps = re.compile(r'[A-ZÁÂÃÀÉÊÍÓÔÕÚÜÇ][A-ZÁÂÃÀÉÊÍÓÔÕÚÜÇ0-9 ,.\'&\-\n]{5,}')


def _docname_line_is_eligible(line: str) -> bool:
    s = line.strip()
    return (
        s != ""
        and "*" in s
        and any(ch.isalpha() for ch in s)     # has letters
        and any(ch.isalpha() and ch.islower() for ch in s)  # has lowercase
    )


def _line_has_lowercase(line: str) -> bool:
    return any(ch.isalpha() and ch.islower() for ch in line)

def _eligible_line(line: str) -> bool:
    # non-empty, contains letters, at least two words, and NO lowercase letters
    stripped = line.strip()
    return (
        stripped != ""
        and any(ch.isalpha() for ch in stripped)
        and " " in stripped
        and not _line_has_lowercase(line)
    )

# --- REPLACE your allcaps_entity component with this ---
@Language.component("allcaps_entity")
def allcaps_entity(doc):
    text = doc.text
    spans = []

    lines = text.splitlines(keepends=True)
    pos = 0

    # Current run state
    run_label = None         # ORG_LABEL / ORG_WITH_STAR_LABEL / None
    run_start = None         # char start of current run (first non-space)
    run_end = None           # char end (exclusive), trimmed of trailing \n

    def flush_run():
        nonlocal run_label, run_start, run_end
        if run_label is not None and run_start is not None and run_end is not None and run_end > run_start:
            # Trim trailing newline if present
            end_idx = run_end - 1 if text[run_end - 1:run_end] == "\n" else run_end
            span = doc.char_span(run_start, end_idx, label=run_label, alignment_mode="contract")
            if span is not None:
                spans.append(span)
        run_label = None
        run_start = None
        run_end = None

    for ln in lines:
        line_end = pos + len(ln)
        content = ln[:-1] if ln.endswith("\n") else ln
        stripped = content.strip()

        if _eligible_line(stripped):
            # Determine this line's type
            this_label = "ORG_WITH_STAR_LABEL" if "*" in stripped else "ORG_LABEL"

            # Compute start index at first non-space on this line
            leading_spaces = len(content) - len(content.lstrip())
            line_start_idx = pos + leading_spaces
            line_end_idx = line_end - (1 if ln.endswith("\n") else 0)

            if run_label is None:
                # start a new run
                run_label = this_label
                run_start = line_start_idx
                run_end = line_end_idx
            elif this_label == run_label:
                # extend current run
                run_end = line_end_idx
            else:
                # label changed → flush previous, start new
                flush_run()
                run_label = this_label
                run_start = line_start_idx
                run_end = line_end_idx
        else:
            # not eligible → break any open run
            flush_run()

        pos = line_end

    # close final run
    flush_run()

    if spans:
        from spacy.util import filter_spans
        doc.ents = filter_spans(list(doc.ents) + spans)
    return doc


@Language.component("docname_entity")
def docname_entity(doc):
    text = doc.text
    spans = []

    lines = text.splitlines(keepends=True)
    pos = 0

    run_start = None  # char start of current DOC_NAME block
    run_end = None    # char end (exclusive)

    def flush_run():
        nonlocal run_start, run_end
        if run_start is not None and run_end is not None and run_end > run_start:
            end_idx = run_end - 1 if text[run_end - 1:run_end] == "\n" else run_end
            span = doc.char_span(run_start, end_idx, label="DOC_NAME_LABEL", alignment_mode="contract")
            if span is not None:
                spans.append(span)
        run_start = None
        run_end = None

    for ln in lines:
        line_end = pos + len(ln)
        content = ln[:-1] if ln.endswith("\n") else ln
        if _docname_line_is_eligible(content):
            leading_spaces = len(content) - len(content.lstrip())
            line_start_idx = pos + leading_spaces
            line_end_idx = line_end - (1 if ln.endswith("\n") else 0)
            if run_start is None:
                run_start = line_start_idx
            run_end = line_end_idx
        else:
            flush_run()
        pos = line_end

    flush_run()

    if spans:
        from spacy.util import filter_spans
        doc.ents = filter_spans(list(doc.ents) + spans)
    return doc



def setup_entities(nlp):

    ruler = nlp.add_pipe("entity_ruler", first = True)
    ruler.add_patterns(RULER_PATTERNS)
    nlp.add_pipe("allcaps_entity")
    nlp.add_pipe("docname_entity") 


