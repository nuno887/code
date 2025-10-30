# entity_bold.py
import re
from typing import List, Tuple, Optional
from spacy.language import Language
from spacy.tokens import Doc, Span
from spacy.util import filter_spans

# ------------------------- config -------------------------

# Prepositions/articles/connectors that imply continuation
_CONNECTORS = {"DO", "DA", "DE", "DOS", "DAS", "NO", "NA", "NOS", "NAS", "E"}

# Corporate suffixes that can appear as a *pure* bold chunk (e.g., **S.A.**)
_PURE_SUFFIXES = {
    "LDA", "LDA.", "S.A.", "SA", "SGPS", "E.P.E.", "EPE", "CRL", "E.M.", "EM",
    "UNIPESSOAL LDA", "UNIPESSOAL LDA."
}

# Matches **...** (non-greedy)
_BOLD_RE = re.compile(r"\*\*(.+?)\*\*")

# ------------------------- helpers -------------------------

def _compact_uc(s: str) -> str:
    """Uppercase and remove internal whitespace (so 'D O' == 'DO')."""
    return re.sub(r"\s+", "", s.upper())

def _last_uc_word(txt: str) -> str:
    parts = re.findall(r"[A-ZÀ-ÖØ-Ý]+", txt.upper())
    return parts[-1] if parts else ""

def _first_uc_word(txt: str) -> str:
    parts = re.findall(r"[A-ZÀ-ÖØ-Ý]+", txt.upper())
    return parts[0] if parts else ""

def _ends_with_connector(txt: str) -> bool:
    t = txt.rstrip()
    if t.endswith("-"):
        return True
    last = _compact_uc(_last_uc_word(t))
    return last in _CONNECTORS

def _starts_with_connector_or_punct(txt: str) -> bool:
    t = txt.lstrip()
    if t.startswith(("-", ",", "/")):
        return True
    first = _compact_uc(_first_uc_word(t))
    return first in _CONNECTORS

def _is_pure_suffix(txt: str) -> bool:
    """Return True if the text is *just* a suffix block like 'S.A.' or 'LDA.'."""
    t = txt.strip().strip(" ,;/")
    norm_spaced = " ".join(t.upper().split())
    norm_compact = _compact_uc(norm_spaced)
    return (
        norm_spaced in _PURE_SUFFIXES
        or norm_compact in {_compact_uc(s) for s in _PURE_SUFFIXES}
    )

def _line_has_lowercase(s: str) -> bool:
    return any(ch.isalpha() and ch.islower() for ch in s)

def _eligible_line(line: str) -> bool:
    # non-empty, has letters, and NO lowercase letters
    stripped = line.strip()
    return any(ch.isalpha() for ch in stripped) and not _line_has_lowercase(line)

def _strip_newline_end(text: str, end_idx: int) -> int:
    return end_idx - 1 if text[end_idx - 1: end_idx] == "\n" else end_idx

def _extract_bold_chunks_on_line(abs_line_start: int, line_text: str):
    """
    Yield bold chunks on this line as tuples:
      (outer_start, outer_end, inner_start, inner_end, inner_text)
    outer_* include the ** markers; inner_* exclude them.
    """
    chunks = []
    i = 0
    n = len(line_text)
    while i < n:
        open_idx = line_text.find("**", i)
        if open_idx == -1:
            break
        inner_start = open_idx + 2
        close_idx = line_text.find("**", inner_start)
        if close_idx == -1:
            break
        outer_start = abs_line_start + open_idx
        outer_end   = abs_line_start + close_idx + 2
        inner_s_abs = abs_line_start + inner_start
        inner_e_abs = abs_line_start + close_idx
        inner_text  = line_text[inner_start:close_idx]
        chunks.append((outer_start, outer_end, inner_s_abs, inner_e_abs, inner_text))
        i = close_idx + 2
    return chunks

def _join_or_split_chunks(chunks):
    """
    Group bold chunks on the SAME line with connector rules.

    chunks: list of (outer_s, outer_e, inner_s, inner_e, inner_txt)
    returns: list of (outer_s, outer_e, inner_txt_joined)
    """
    if len(chunks) <= 1:
        if not chunks:
            return []
        os, oe, _, __, it = chunks[0]
        return [(os, oe, it)]

    groups = []
    cur_os, cur_oe, _, __, cur_it = chunks[0]

    for os, oe, _, __, it in chunks[1:]:
        join = (
            _ends_with_connector(cur_it)
            or _starts_with_connector_or_punct(it)
            or _is_pure_suffix(it)  # join only when the second chunk is a PURE suffix block
        )
        if join:
            cur_oe = oe
            cur_it = f"{cur_it.rstrip()} {it.lstrip()}"
        else:
            groups.append((cur_os, cur_oe, cur_it))
            cur_os, cur_oe, cur_it = os, oe, it

    groups.append((cur_os, cur_oe, cur_it))
    return groups

def _overlaps(a_start: int, a_end: int, b_start: int, b_end: int) -> bool:
    return (a_start < b_end) and (b_start < a_end)

# ------------------------- component -------------------------

@Language.component("allcaps_entity_02")
def allcaps_entity_02(doc: Doc) -> Doc:
    text = doc.text
    spans: List[Span] = []

    lines = text.splitlines(keepends=True)
    pos = 0

    # Pending starred group for cross-line decision: (outer_s, outer_e, inner_text)
    pending_star: Optional[Tuple[int, int, str]] = None

    # Run state for non-starred ALL-CAPS lines (ORG_LABEL legacy merging)
    run_label = None       # only "ORG_LABEL"
    run_start = None
    run_end = None

    def flush_run():
        nonlocal run_label, run_start, run_end
        if run_label and run_start is not None and run_end is not None and run_end > run_start:
            end_idx = _strip_newline_end(text, run_end)
            span = doc.char_span(run_start, end_idx, label=run_label, alignment_mode="contract")
            if span is not None:
                spans.append(span)
        run_label = None
        run_start = None
        run_end = None

    def flush_pending_star():
        nonlocal pending_star
        if pending_star is not None:
            os, oe, _ = pending_star
            span = doc.char_span(os, oe, label="ORG_WITH_STAR_LABEL", alignment_mode="expand")
            if span is not None:
                spans.append(span)
        pending_star = None

    for ln in lines:
        line_end = pos + len(ln)
        content = ln[:-1] if ln.endswith("\n") else ln
        stripped = content.strip()

        # --- Bold parsing on this line ---
        raw_chunks = _extract_bold_chunks_on_line(pos, content)
        # Eligible starred chunks (all-caps inside)
        eligible_bold = [(os, oe, is_, ie, it) for (os, oe, is_, ie, it) in raw_chunks if _eligible_line(it)]

        if len(eligible_bold) >= 1:
            # Group within this line first
            groups = _join_or_split_chunks(eligible_bold)

            if len(groups) == 1:
                # Single starred group → cross-line logic via pending_star
                g_os, g_oe, g_it = groups[0]
                if pending_star is None:
                    pending_star = (g_os, g_oe, g_it)
                else:
                    p_os, p_oe, p_it = pending_star
                    join = (
                        _ends_with_connector(p_it)
                        or _starts_with_connector_or_punct(g_it)
                        or _is_pure_suffix(g_it)
                    )
                    if join:
                        # Extend across newline
                        pending_star = (p_os, g_oe, f"{p_it.rstrip()} {g_it.lstrip()}")
                    else:
                        # Split: flush previous, start new pending
                        flush_pending_star()
                        pending_star = (g_os, g_oe, g_it)
                # starred found → don't mix with non-star run
                flush_run()

            else:
                # Multiple starred groups on this line
                # If we had a pending previous starred, resolve it with the first group
                if pending_star is not None:
                    g0_os, g0_oe, g0_it = groups[0]
                    p_os, p_oe, p_it = pending_star
                    join = (
                        _ends_with_connector(p_it)
                        or _starts_with_connector_or_punct(g0_it)
                        or _is_pure_suffix(g0_it)
                    )
                    if join:
                        groups[0] = (p_os, g0_oe, f"{p_it.rstrip()} {g0_it.lstrip()}")
                    else:
                        flush_pending_star()

                # Emit all groups (starred) with OUTER bounds, including ** markers
                for os, oe, _ in groups:
                    span = doc.char_span(os, oe, label="ORG_WITH_STAR_LABEL", alignment_mode="expand")
                    if span is not None:
                        spans.append(span)

                pending_star = None
                flush_run()

        else:
            # No starred chunks on this line
            if stripped == "":
                # Blank line: allow pending_star to remain (wrapping)
                pass
            else:
                # Real content breaks pending starred continuation
                flush_pending_star()

            # Handle non-starred ALL-CAPS lines as ORG_LABEL (legacy merging)
            if _eligible_line(stripped) and "*" not in stripped:
                leading_spaces = len(content) - len(content.lstrip())
                line_start_idx = pos + leading_spaces
                line_end_idx = line_end - (1 if ln.endswith("\n") else 0)
                if run_label is None:
                    run_label = "ORG_LABEL"
                    run_start = line_start_idx
                    run_end = line_end_idx
                else:
                    run_end = line_end_idx
            else:
                flush_run()

        pos = line_end

    # Close out
    flush_pending_star()
    flush_run()

    if spans:
        # Prefer starred orgs over PARAGRAPH by dropping overlapping paragraph spans
        prior = []
        for e in doc.ents:
            if e.label_ != "PARAGRAPH":
                prior.append(e)
                continue
            # drop PARAGRAPH if it overlaps any new starred/label spans we just made
            drop = any(_overlaps(e.start_char, e.end_char, s.start_char, s.end_char) for s in spans)
            if not drop:
                prior.append(e)

        doc.ents = filter_spans(prior + spans)

    return doc
