from spacy.pipeline import EntityRuler
import re, unicodedata
from spacy.language import Language
from spacy.util import filter_spans
import DocText
import Paragraphs
import entity_bold

OPTIONS = {"colors": {
    "Sumario": "#ffd166",
    "ORG_LABEL": "#6e77b8",
    "ORG_WITH_STAR_LABEL": "#6fffff",
    "DOC_NAME_LABEL": "#b23bbd",
    "DOC_TEXT": "#47965e",
    "PARAGRAPH": "#14b840",
    "JUNK_LABEL": "#e11111",
    "SERIE_III": "#D1B1B1"
    }}


RULER_PATTERNS = [

{"label": "Sumario", "pattern": "**Sumário**"},
{"label": "Sumario", "pattern": "**Sumario**"},
{"label": "Sumario",
 "pattern": [
   {"LOWER": {"IN": ["sumário", "sumario"]}},
   {"ORTH": ":", "OP": "!"}
 ]},

{"label": "SERIE_III", "pattern": "**Regulamentação do Trabalho**"},

{"label": "SERIE_III", "pattern": "Direção Regional do Trabalho"},
{"label": "SERIE_III", "pattern": "Direcção Regional do Trabalho"},
{"label": "JUNK_LABEL", "pattern": "SECRETARI"},

{"label": "DOC_NAME_LABEL", "pattern": "Convenções Coletivas de Trabalho:"},
{"label": "DOC_NAME_LABEL", "pattern": "Portarias de Extensão:"},
{"label": "DOC_NAME_LABEL", "pattern": "Despachos:"},
{"label": "DOC_NAME_LABEL", "pattern": "Portarias de Condições de Trabalho:"},
{"label": "DOC_NAME_LABEL", "pattern": "Eleição de Representantes:"},
{"label": "DOC_NAME_LABEL", "pattern": "Avisos de Cessação da Vigência de Convenções Colectivas de Trabalho:"},
{"label": "DOC_NAME_LABEL", "pattern": "Direção:"},
{"label": "DOC_NAME_LABEL", "pattern": "Convocatórias:"},
{"label": "DOC_NAME_LABEL", "pattern": "Eleições:"},
{"label": "DOC_NAME_LABEL", "pattern": "Estatutos:"},
{"label": "DOC_NAME_LABEL", "pattern": "Corpos Gerentes / Alterações:"},
{"label": "DOC_NAME_LABEL", "pattern": "Corpos Gerentes/Alterações:"},
{"label": "DOC_NAME_LABEL", "pattern": "Corpos Gerentes:"},
{"label": "DOC_NAME_LABEL", "pattern": "Alterações:"},
{"label": "DOC_NAME_LABEL", "pattern": "Associações Sindicais/Corpos Gerentes:"},
{"label": "DOC_NAME_LABEL", "pattern": "Associações de Empregadores/Direcção:"},
{"label": "DOC_NAME_LABEL", "pattern": "Associações de Empregadores/Direção:"},

]

KNOWN_DOC_NAMES = [
    "Regulamentação do Trabalho",
    "Portarias de Extensão:",
    "Convenções Colectivas de Trabalho:",
    "Despachos:",
    "Regulamentos de Condições Mínimas:",
    "Estatutos/Alterações:",
]

def _normalize_for_match(s: str) -> str:
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))  # strip accents
    s = re.sub(r"\s+", "", s)  # drop ALL whitespace (handles "Tr a b a l h o")
    return s.casefold()

KNOWN_DOC_NAMES_NORM = { _normalize_for_match(x) for x in KNOWN_DOC_NAMES }

_pattern_allcaps = re.compile(r'[A-ZÁÂÃÀÉÊÍÓÔÕÚÜÇ][A-ZÁÂÃÀÉÊÍÓÔÕÚÜÇ0-9 ,.\'&\-\n]{5,}')
_junk_rx = re.compile(r"^[\d\s\-\–—\.\,;:·•*'\"`´\+\=\(\)\[\]\{\}/\\<>~^_|]{1,20}$")

def _is_junk_line(s: str) -> bool:
    s = s.strip()
    if not s:
        return False
    if '|' in s:          
        return False
    if any(ch.isalpha() for ch in s):
        return False
    return bool(_junk_rx.match(s))



def _docname_line_is_eligible(line: str) -> bool:
    s = line.strip()
    return (
        s != ""
        and "*" in s
        and any(ch.isalpha() for ch in s)     # has letters
        and any(ch.isalpha() and ch.islower() for ch in s)  # has lowercase
    )


def _line_has_lowercase(line: str) -> bool:
    # count only characters in the Unicode "Ll" (Letter, lowercase) category
    return any(ch.isalpha() and unicodedata.category(ch) == 'Ll' for ch in line)

def _eligible_line(line: str) -> bool:
    # non-empty, contains letters, at least two words, and NO lowercase letters
    stripped = line.strip()
    return (
        any(ch.isalpha() for ch in stripped)
        and not _line_has_lowercase(line)
    )


@Language.component("strip_junk_ents")
def strip_junk_ents(doc):
    doc.ents = tuple(e for e in doc.ents if e.label_ != "JUNK_LABEL")
    return doc

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
            # not eligible
            # If the line is blank/whitespace-only, keep the current run open
            # so headings separated by empty lines are merged.
            if stripped == "":
                if run_label is not None:
                    # extend the run to include this newline/whitespace
                    run_end = line_end
            else:
                # real content that breaks the run → flush
                flush_run()

        pos = line_end

    # close final run
    flush_run()

    if spans:
        from spacy.util import filter_spans
        doc.ents = filter_spans(list(doc.ents) + spans)
    return doc




from spacy.language import Language
from spacy.util import filter_spans

def _iter_bold_pairs_no_merge_III(text: str):
    """
    Yield primitive bold pairs without merging across whitespace.
    Returns (outer_start, inner_start, inner_end, outer_end) for each **...**.
    """
    n = len(text)
    i = 0
    while i < n:
        open_idx = text.find("**", i)
        if open_idx == -1:
            break
        inner_start = open_idx + 2
        close_idx = text.find("**", inner_start)
        if close_idx == -1:
            break
        yield open_idx, inner_start, close_idx, close_idx + 2
        i = close_idx + 2

STOP_PUNCT = set(".:;?!…")  # add more here later

def _ends_with_stop(inner: str, stops: set[str] = STOP_PUNCT) -> bool:
    i = len(inner) - 1
    while i >= 0 and inner[i].isspace():
        i -= 1
    return i >= 0 and inner[i] in stops


@Language.component("docname_entity_III")
def docname_entity(doc):
    text = doc.text

    # Collect SERIE_III spans from existing ents (EntityRuler ran first)
    serie3_spans = [(e.start_char, e.end_char) for e in doc.ents if e.label_ == "SERIE_III"]

    def overlaps_serie3(s, e):
        for a, b in serie3_spans:
            if not (e <= a or s >= b):
                return True
        return False

    def serie3_in_gap(left_e, right_s):
        for a, b in serie3_spans:
            if a < right_s and b > left_e:
                return True
        return False

    # Known SERIE_III titles (normalized)
    KNOWN_SERIE3_TITLES = [
        "Direção Regional do Trabalho",
        "Direcção Regional do Trabalho",
        "Regulamentação do Trabalho",
    ]
    KNOWN_SERIE3_TITLES_NORM = {_normalize_for_match(x) for x in KNOWN_SERIE3_TITLES}

    # 1) Collect primitive bold pairs that look like doc names
    prim = []
    for os, is_, ie, oe in _iter_bold_pairs_no_merge_III(text):
        inner = text[is_:ie]
        inner_norm = _normalize_for_match(inner)

        # skip if overlaps SERIE_III or is a known SERIE_III heading
        if overlaps_serie3(os, oe):
            continue
        if inner_norm in KNOWN_SERIE3_TITLES_NORM:
            continue

        looks_like_docname = (
            inner_norm in KNOWN_DOC_NAMES_NORM
            or any(ch.isalpha() and ch.islower() for ch in inner)
        )
        if looks_like_docname:
            # keep full (OUTER) bounds so ** are included
            prim.append((os, is_, ie, oe))

    if not prim:
        return doc

    prim.sort(key=lambda t: t[0])  # by outer_start

    # 2) Merge adjacent primitives when only whitespace is between,
    #    but stop on colon or trailing dot, and never cross SERIE_III.
    merged = []
    cur_os, cur_is, cur_ie, cur_oe = prim[0]
    for os, is_, ie, oe in prim[1:]:
        between = text[cur_oe:os]
        left_inner = text[cur_is:cur_ie]

        hard_stop = left_inner.rstrip().endswith(":") or _ends_with_stop(left_inner)

        if (between.strip() == "") and (not hard_stop) and (not serie3_in_gap(cur_oe, os)):
            # extend current group to include the next segment
            cur_oe = oe
            cur_ie = ie
        else:
            # store OUTER bounds so ** markers are part of the entity
            merged.append((cur_os, cur_oe))
            cur_os, cur_is, cur_ie, cur_oe = os, is_, ie, oe
    merged.append((cur_os, cur_oe))

    # 3) Create spans using OUTER bounds to include the ** markers
    spans = []
    for s, e in merged:
        if s < e:
            span = doc.char_span(s, e, label="DOC_NAME_LABEL", alignment_mode="expand")
            if span is not None:
                spans.append(span)

    if spans:
        doc.ents = filter_spans(list(doc.ents) + spans)
    return doc


# NEW: iterate bold blocks, merging adjacent **...** chunks separated only by whitespace,
# and handling lines that are just "**" as open/close markers across lines.
def _iter_bold_blocks(text: str):
    """
    Yield Markdown bold blocks **...**; merge adjacent pairs on the SAME line.
    Returns (outer_start, inner_start, inner_end, outer_end).
    """
    n = len(text)
    i = 0
    while i < n:
        open_idx = text.find("**", i)
        if open_idx == -1:
            break
        inner_start = open_idx + 2
        close_idx = text.find("**", inner_start)
        if close_idx == -1:
            break

        # initial block bounds (including **)
        block_start = open_idx
        block_end = close_idx + 2

        # try to merge following **...** pairs if only spaces (no newline) lie between
        j = block_end
        while j < n:
            # stop merging if we see a newline between pairs
            k = j
            saw_newline = False
            while k < n and text[k].isspace():
                if text[k] == "\n":
                    saw_newline = True
                    break
                k += 1
            if saw_newline:
                break
            # next pair must start immediately after spaces
            if k + 1 < n and text[k] == "*" and text[k + 1] == "*":
                next_open = k
                next_inner_start = next_open + 2
                next_close = text.find("**", next_inner_start)
                if next_close == -1:
                    break
                # extend current block to include this adjacent pair
                block_end = next_close + 2
                j = block_end
            else:
                break

        yield block_start, inner_start, block_end - 2, block_end
        i = block_end



@Language.component("docname_entity")
def docname_entity(doc):
    text = doc.text
    spans = []
    for outer_start, inner_start, inner_end, outer_end in _iter_bold_blocks(text):
        inner = text[inner_start:inner_end]
        # Match if (1) normalized text is in your known list OR (2) it contains lowercase letters
        if (_normalize_for_match(inner) in KNOWN_DOC_NAMES_NORM) or any(ch.isalpha() and ch.islower() for ch in inner):
            span = doc.char_span(outer_start, outer_end, label="DOC_NAME_LABEL", alignment_mode="contract")
            if span is not None:
                spans.append(span)
    if spans:
        doc.ents = filter_spans(list(doc.ents) + spans)
    return doc

@Language.component("junk_entity")  # NEW
def junk_entity(doc):
    text = doc.text
    lines = text.splitlines(keepends=True)

    spans = []
    pos = 0
    for ln in lines:
        line_end = pos + len(ln)
        content = ln[:-1] if ln.endswith("\n") else ln
        stripped = content.strip()
        if stripped:
            leading = len(content) - len(content.lstrip())
            trailing = len(content) - len(content.rstrip())
            start_idx = pos + leading
            end_idx = (line_end - (1 if ln.endswith("\n") else 0)) - trailing

            if start_idx < end_idx and _is_junk_line(stripped):
                # skip if overlaps an existing ent
                if not any(e.start_char < end_idx and e.end_char > start_idx for e in doc.ents):
                    span = doc.char_span(start_idx, end_idx, label="JUNK_LABEL", alignment_mode="contract")
                    if span is not None:
                        spans.append(span)
        pos = line_end

    if spans:
        doc.ents = filter_spans(list(doc.ents) + spans)
    return doc

import re
from spacy.language import Language
from spacy.util import filter_spans

_BOLD_PAIR_RE = re.compile(r"\*\*.+?\*\*", re.DOTALL)

def _overlaps(a_s, a_e, b_s, b_e):
    return (a_s < b_e) and (b_s < a_e)

@Language.component("paragraph_to_org_star")
def paragraph_to_org_star(doc):
    text = doc.text
    new_spans = []

    for e in doc.ents:
        if e.label_ != "PARAGRAPH":
            continue

        segment = text[e.start_char:e.end_char]

        # 🔸 Skip paragraphs that have any lowercase (Unicode-aware)
        # This ensures we only process ALL-CAPS paragraphs.
        if any(ch.islower() for ch in segment):
            continue

        # Find all **…** pairs inside this ALL-CAPS paragraph
        for m in _BOLD_PAIR_RE.finditer(segment):
            inner = segment[m.start()+2 : m.end()-2]
            if _eligible_line(inner):
                os = e.start_char + m.start()
                oe = e.start_char + m.end()
                span = doc.char_span(os, oe, label="ORG_WITH_STAR_LABEL", alignment_mode="expand")
                if span is not None:
                    new_spans.append(span)

    if not new_spans:
        return doc

    kept = []
    for e in doc.ents:
        if e.label_ != "PARAGRAPH":
            kept.append(e)
            continue
        if any(_overlaps(e.start_char, e.end_char, s.start_char, s.end_char) for s in new_spans):
            continue
        kept.append(e)

    doc.ents = filter_spans(kept + new_spans)
    return doc




def setup_entities(nlp, SerieIII: bool):

    ruler = nlp.add_pipe("entity_ruler", first = True)
    ruler.add_patterns(RULER_PATTERNS)
    nlp.add_pipe("allcaps_entity_02")
    if SerieIII:
        nlp.add_pipe("docname_entity_III")
    else:
        nlp.add_pipe("docname_entity")
    # nlp.add_pipe("junk_entity")
    nlp.add_pipe("doc_text_entity")
    # nlp.add_pipe("strip_junk_ents")
    nlp.add_pipe("paragraph_entity")
    nlp.add_pipe("paragraph_to_org_star")
    
        


