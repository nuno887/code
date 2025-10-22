from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import json
import re
import unicodedata

# --------------------------------------------------------------------------------------
# Normalization helpers (NO regex-based searching in the body; these are only for
# text cleanup/equality checks across JSON <-> labeled spans.)
# --------------------------------------------------------------------------------------


def _strip_accents(s: str) -> str:
    if s is None:
        return ""
    nfkd = unicodedata.normalize("NFD", s)
    return "".join(ch for ch in nfkd if unicodedata.category(ch) != "Mn")

def _org_key(s: str) -> str:
    """
    Aggressive normalization for ORG comparisons:
    - strip markdown bold
    - remove accents
    - remove ALL non-alphanumerics (spaces, punctuation)
    - uppercase
    """
    s = _strip_markdown_bold(s)
    s = _strip_accents(s)
    s = re.sub(r"[^A-Za-z0-9]", "", s)
    return s.upper()


def _strip_markdown_bold(s: str) -> str:
    # Remove surrounding ** that may appear in headers
    return s.replace("**", "").strip()


def _collapse_spaces(s: str) -> str:
    # Collapse multiple whitespace into single spaces; strip ends
    return re.sub(r"\s+", " ", s).strip()


def _join_spaced_caps(s: str) -> str:
    """
    Join artificial spacing often produced by PDF extraction in ALL-CAPS words.
    Example: "D IREÇÃO R EGIONAL" -> "DIREÇÃO REGIONAL".
    We only remove spaces BETWEEN capital letters (incl. Portuguese diacritics).
    """
    # Replace sequences like "A B C" -> "ABC" (only when both sides are caps)
    caps = "A-ZÁÂÃÀÇÉÊÍÓÔÕÚÜ"
    pattern = rf"([ {caps}])\s(?=[{caps}])"
    # Run repeatedly to fully collapse chains
    prev = None
    out = s
    while prev != out:
        prev = out
        out = re.sub(pattern, r"\1", out)
    return out


def normalize_text(s: str) -> str:
    if s is None:
        return ""
    s = _strip_markdown_bold(s)
    s = _collapse_spaces(s)
    s = _join_spaced_caps(s)
    return s

# Add this helper (or inline the logic)
def _build_org_blocks_filtered(doc_text: str, spans: List[SpanInfo], valid_org_norms: set[str]) -> List[Tuple[SpanInfo, int, int]]:
    # Keep only ORG spans that are present in JSON (after normalization)
    orgs = [s for s in spans if s.label in ORG_LABELS and normalize_text(s.text) in valid_org_norms]
    blocks: List[Tuple[SpanInfo, int, int]] = []
    if not orgs:
        return blocks
    for i, org in enumerate(orgs):
        start = org.start_char
        end = orgs[i + 1].start_char if i + 1 < len(orgs) else len(doc_text)
        blocks.append((org, start, end))
    return blocks


# --------------------------------------------------------------------------------------
# Data structures
# --------------------------------------------------------------------------------------

@dataclass
class SpanInfo:
    label: str
    text: str
    start_char: int
    end_char: int

@dataclass
class DocSlice:
    doc_name: str
    text: str

@dataclass
class OrgBlockResult:
    org: str
    org_block_text: str
    docs: List[DocSlice] = field(default_factory=list)
    status: str = "ok"  # ok | partial | doc_missing | org_missing

# --------------------------------------------------------------------------------------
# Core indexing over spaCy doc_body
# --------------------------------------------------------------------------------------

ORG_LABELS = {"ORG_WITH_STAR_LABEL", "ORG_LABEL"}
DOC_NAME_LABEL = "DOC_NAME_LABEL"
CONTENT_LABELS = {"PARAGRAPH", "DOC_TEXT", DOC_NAME_LABEL}
IGNORE_LABELS = {"JUNK_LABEL"}


def _collect_spans(doc) -> List[SpanInfo]:
    spans: List[SpanInfo] = []
    for ent in doc.ents:
        label = ent.label_ if hasattr(ent, "label_") else str(ent.label)
        if label in IGNORE_LABELS:
            continue
        spans.append(SpanInfo(label=label, text=str(ent.text), start_char=ent.start_char, end_char=ent.end_char))
    # Ensure sorted by position
    spans.sort(key=lambda s: s.start_char)
    return spans


def _org_anchors(spans: List[SpanInfo]) -> List[SpanInfo]:
    return [s for s in spans if s.label in ORG_LABELS]


def _docname_anchors(spans: List[SpanInfo]) -> List[SpanInfo]:
    return [s for s in spans if s.label == DOC_NAME_LABEL]


# --------------------------------------------------------------------------------------
# Build ORG blocks from the labeled spans
# --------------------------------------------------------------------------------------

def _build_org_blocks(doc_text: str, spans: List[SpanInfo]) -> List[Tuple[SpanInfo, int, int]]:
    """
    Returns a list of tuples (org_anchor_span, block_start, block_end), where the block
    covers from this org anchor to the next org anchor (or EOF).
    """
    orgs = _org_anchors(spans)
    blocks: List[Tuple[SpanInfo, int, int]] = []
    if not orgs:
        return blocks
    for i, org in enumerate(orgs):
        start = org.start_char
        if i + 1 < len(orgs):
            end = orgs[i + 1].start_char
        else:
            end = len(doc_text)
        blocks.append((org, start, end))
    return blocks


def _spans_within(spans: List[SpanInfo], start: int, end: int, labels: Optional[set] = None) -> List[SpanInfo]:
    out = []
    for s in spans:
        if s.start_char >= start and s.start_char < end:
            if labels is None or s.label in labels:
                out.append(s)
    return out

def _build_org_blocks_coalesced_to_json(
    doc_text: str,
    spans: List[SpanInfo],
    valid_org_keys: set[str],
    *,
    max_merge: int = 3,
) -> List[Tuple[SpanInfo, int, int]]:
    orgs = [s for s in spans if s.label in ORG_LABELS]
    orgs.sort(key=lambda s: s.start_char)

    anchors: List[SpanInfo] = []
    i = 0
    while i < len(orgs):
        best_j = None
        end_char = orgs[i].end_char
        concatenated_raw = orgs[i].text  # start with raw to keep original casing/accents

        for j in range(i, min(i + max_merge, len(orgs))):
            if j > i:
                gap = doc_text[end_char:orgs[j].start_char]
                if not re.match(r"^[\s•\-–,.;:]*$", gap):
                    break
                concatenated_raw = concatenated_raw + " " + orgs[j].text  # keep a space between parts
            # check the aggressive key
            if _org_key(concatenated_raw) in valid_org_keys:
                best_j = j
            end_char = orgs[j].end_char

        if best_j is not None:
            start_char = orgs[i].start_char
            end_char = orgs[best_j].end_char
            text_slice = doc_text[start_char:end_char]
            anchors.append(SpanInfo(label=orgs[i].label, text=text_slice, start_char=start_char, end_char=end_char))
            i = best_j + 1
        else:
            if _org_key(orgs[i].text) in valid_org_keys:
                anchors.append(orgs[i])
            i += 1

    blocks: List[Tuple[SpanInfo, int, int]] = []
    for k, a in enumerate(anchors):
        start = a.start_char
        end = anchors[k + 1].start_char if k + 1 < len(anchors) else len(doc_text)
        blocks.append((a, start, end))
    return blocks



# --------------------------------------------------------------------------------------
# JSON loading
# --------------------------------------------------------------------------------------

def load_serieIII_minimal(path: Path | str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def coerce_items_payload(serieIII_json_or_path: Any) -> Dict[str, Any]:
    """Accept either a dict payload or a path-like, and return the JSON dict."""
    if isinstance(serieIII_json_or_path, dict):
        return serieIII_json_or_path
    # treat as a path
    return load_serieIII_minimal(serieIII_json_or_path)

# --------------------------------------------------------------------------------------
# Main function: divide by ORG first, then by DOC_NAME within each ORG block
# --------------------------------------------------------------------------------------

def _build_org_blocks_filtered(
    doc_text: str,
    spans: List[SpanInfo],
    valid_org_norms: set[str],
) -> List[Tuple[SpanInfo, int, int]]:
    """Like _build_org_blocks, but only uses ORG anchors whose normalized text is in the JSON."""
    orgs = [s for s in spans if s.label in ORG_LABELS and normalize_text(s.text) in valid_org_norms]
    orgs.sort(key=lambda s: s.start_char)
    blocks: List[Tuple[SpanInfo, int, int]] = []
    for i, org in enumerate(orgs):
        start = org.start_char
        end = orgs[i + 1].start_char if i + 1 < len(orgs) else len(doc_text)
        blocks.append((org, start, end))
    return blocks
def _collect_suborg_anchors_coalesced(
    doc_text: str,
    spans: List[SpanInfo],
    bstart: int,
    bend: int,
    valid_suborg_keys: set[str],
    *,
    max_merge: int = 3,
) -> List[SpanInfo]:
    """
    Within [bstart,bend), coalesce adjacent ORG spans and keep those whose _org_key matches
    valid_suborg_keys. IMPORTANT: returns ALL matching occurrences in order (no dedup).
    """
    block_orgs = [s for s in spans if s.label in ORG_LABELS and bstart <= s.start_char < bend]
    block_orgs.sort(key=lambda s: s.start_char)

    anchors: List[SpanInfo] = []
    i = 0
    while i < len(block_orgs):
        best_j = None
        end_char = block_orgs[i].end_char
        concatenated_raw = block_orgs[i].text

        for j in range(i, min(i + max_merge, len(block_orgs))):
            if j > i:
                gap = doc_text[end_char:block_orgs[j].start_char]
                if not re.match(r"^[\s•\-–,.;:]*$", gap):
                    break
                concatenated_raw = concatenated_raw + " " + block_orgs[j].text
            if _org_key(concatenated_raw) in valid_suborg_keys:
                best_j = j
            end_char = block_orgs[j].end_char

        if best_j is not None:
            start_char = block_orgs[i].start_char
            end_char = block_orgs[best_j].end_char
            text_slice = doc_text[start_char:end_char]
            anchors.append(SpanInfo(label=block_orgs[i].label, text=text_slice, start_char=start_char, end_char=end_char))
            i = best_j + 1
        else:
            if _org_key(block_orgs[i].text) in valid_suborg_keys:
                anchors.append(block_orgs[i])
            i += 1

    anchors.sort(key=lambda s: s.start_char)
    return anchors


def _build_org_blocks_coalesced_to_json(
    doc_text: str,
    spans: List[SpanInfo],
    valid_org_keys: set[str],
    *,
    max_merge: int = 3,
) -> List[Tuple[SpanInfo, int, int]]:
    orgs = [s for s in spans if s.label in ORG_LABELS]
    orgs.sort(key=lambda s: s.start_char)

    anchors: List[SpanInfo] = []
    i = 0
    while i < len(orgs):
        best_j = None
        end_char = orgs[i].end_char
        concatenated_raw = orgs[i].text

        for j in range(i, min(i + max_merge, len(orgs))):
            if j > i:
                gap = doc_text[end_char:orgs[j].start_char]
                if not re.match(r"^[\s•\-–,.;:]*$", gap):
                    break
                concatenated_raw = concatenated_raw + " " + orgs[j].text
            if _org_key(concatenated_raw) in valid_org_keys:
                best_j = j
            end_char = orgs[j].end_char

        if best_j is not None:
            start_char = orgs[i].start_char
            end_char = orgs[best_j].end_char
            text_slice = doc_text[start_char:end_char]
            anchors.append(SpanInfo(label=orgs[i].label, text=text_slice, start_char=start_char, end_char=end_char))
            i = best_j + 1
        else:
            if _org_key(orgs[i].text) in valid_org_keys:
                anchors.append(orgs[i])
            i += 1

    blocks: List[Tuple[SpanInfo, int, int]] = []
    for k, a in enumerate(anchors):
        start = a.start_char
        end = anchors[k + 1].start_char if k + 1 < len(anchors) else len(doc_text)
        blocks.append((a, start, end))
    return blocks


def divide_body_by_org_and_docs(
    doc_body,
    serieIII_json_or_path: Any,
    *,
    write_org_files: bool = False,
    write_doc_files: bool = False,
    out_dir: Path | str = "output_docs",
    file_prefix: str = "serieIII",
    verbose: bool = False,
) -> Tuple[List[OrgBlockResult], Dict[str, Any]]:
    """
    Supports two JSON schemas:

    Flat:
      items[].org + docs  -> slice by DOC_NAME_LABEL (with fallbacks).

    Hierarchical:
      items[].top_org + sub_orgs[].org + sub_orgs[].docs
        -> inside each top_org block, cut **ORG→next ORG** for **every occurrence**
           of each sub_org header (no dedup), then map these slices sequentially
           to the sub_org's JSON docs. Extras => partial; shortages => partial.

    Flat-mode fallbacks (unchanged):
      - If headers exist but no names match, slice all headers present.
      - If zero headers and exactly 1 JSON doc, slice the whole block as that doc (ok).
    """
    data = coerce_items_payload(serieIII_json_or_path)
    items = data.get("items", [])

    spans = _collect_spans(doc_body)  # JUNK skipped
    doc_text = doc_body.text
    out_dir = Path(out_dir)
    if write_org_files or write_doc_files:
        out_dir.mkdir(parents=True, exist_ok=True)

    # Decide schema
    hierarchical = any("top_org" in it for it in items)

    results: List[OrgBlockResult] = []
    org_ok = org_partial = org_doc_missing = org_missing = 0
    total_orgs = 0
    total_docs_expected = 0
    total_docs_matched = 0

    # ---- helper for FLAT mode: slice docs inside [bstart,bend) ----
    def _slice_docs_in_block(
        bstart: int,
        bend: int,
        json_docs: List[Dict[str, Any]],
        org_label_text_raw: str,
    ) -> Tuple[List[DocSlice], List[str], str, int]:
        # Ordered DOC_NAME anchors inside this block
        docname_spans_sorted = sorted(
            _spans_within(spans, bstart, bend, labels={DOC_NAME_LABEL}),
            key=lambda s: s.start_char
        )

        # Zero-header fallback: exactly 1 expected doc -> whole block (ok)
        if not docname_spans_sorted and len(json_docs) == 1:
            jdoc_raw = json_docs[0].get("text", "")
            if verbose:
                print(f"[INFO] Zero-header fallback: whole-block slice for ORG {org_label_text_raw!r}")
            return (
                [DocSlice(doc_name=_strip_markdown_bold(jdoc_raw).strip(),
                          text=doc_text[bstart:bend])],
                [],
                "ok",
                1
            )

        matched_slices: List[DocSlice] = []
        unmatched_docs: List[str] = []

        def _slice_end(start_char: int) -> int:
            for s in docname_spans_sorted:
                if s.start_char > start_char:
                    return s.start_char
            return bend

        # Sequential name matching; if zero matches but headers exist, slice all headers
        json_doc_names = [normalize_text(d.get("text", "")) for d in json_docs]
        i_ptr = 0
        for jdoc_raw, jdoc_norm in zip([d.get("text", "") for d in json_docs], json_doc_names):
            while i_ptr < len(docname_spans_sorted) and normalize_text(docname_spans_sorted[i_ptr].text) != jdoc_norm:
                i_ptr += 1
            if i_ptr < len(docname_spans_sorted):
                head_span = docname_spans_sorted[i_ptr]
                start = head_span.start_char
                end = _slice_end(start)
                matched_slices.append(DocSlice(
                    doc_name=_strip_markdown_bold(jdoc_raw).strip(),
                    text=doc_text[start:end]
                ))
                i_ptr += 1
            else:
                unmatched_docs.append(jdoc_raw)

        # Brutal fallback: slice all headers if no name-matches but headers exist
        if not matched_slices and docname_spans_sorted:
            if verbose:
                print(f"[INFO] Fallback: slicing {len(docname_spans_sorted)} docs by body headers for ORG {org_label_text_raw!r}")
            for h in docname_spans_sorted:
                start = h.start_char
                end = _slice_end(start)
                matched_slices.append(DocSlice(
                    doc_name=_strip_markdown_bold(h.text).strip(),
                    text=doc_text[start:end]
                ))
            unmatched_docs = [d.get("text", "") for d in json_docs]

        # Status
        if len(json_docs) == 0:
            status = "ok"
        elif matched_slices and unmatched_docs:
            status = "partial"
        elif not matched_slices and json_docs:
            status = "doc_missing"
        else:
            status = "ok"

        return matched_slices, unmatched_docs, status, len(matched_slices)

    if not hierarchical:
        # -------- FLAT MODE --------
        json_org_keys = {_org_key(item.get("org", {}).get("text", "")) for item in items}
        org_blocks = _build_org_blocks_coalesced_to_json(doc_text, spans, json_org_keys)

        body_org_lookup: Dict[str, Tuple[SpanInfo, int, int]] = {}
        for org_span, bstart, bend in org_blocks:
            body_org_lookup.setdefault(_org_key(org_span.text), (org_span, bstart, bend))

        total_orgs = len(items)

        for idx, item in enumerate(items, start=1):
            json_org_text_raw = item.get("org", {}).get("text", "")
            json_org_key = _org_key(json_org_text_raw)
            json_docs = item.get("docs", [])
            total_docs_expected += len(json_docs)

            if json_org_key in body_org_lookup:
                org_span, bstart, bend = body_org_lookup[json_org_key]
                block_text = doc_text[bstart:bend]

                matched_slices, unmatched_docs, status, matched_count = _slice_docs_in_block(
                    bstart, bend, json_docs, json_org_text_raw
                )
                total_docs_matched += matched_count

                if status == "ok":
                    org_ok += 1
                elif status == "partial":
                    org_partial += 1
                elif status == "doc_missing":
                    org_doc_missing += 1

                results.append(OrgBlockResult(
                    org=_strip_markdown_bold(json_org_text_raw).strip(),
                    org_block_text=block_text,
                    docs=matched_slices,
                    status=status,
                ))

                if write_org_files:
                    org_slug = re.sub(r"[^\w.-]+", "_", normalize_text(json_org_text_raw))[:120]
                    path = out_dir / f"{file_prefix}_ORG_{idx:03d}_{org_slug}.txt"
                    with open(path, "w", encoding="utf-8") as f:
                        f.write("DOC_BEGIN\n")
                        f.write(f"ORG: {_strip_markdown_bold(json_org_text_raw).strip()}\n")
                        f.write(block_text)
                        f.write("\nDOC_END\n")
                if write_doc_files and matched_slices:
                    for k, ds in enumerate(matched_slices, start=1):
                        doc_slug = re.sub(r"[^\w.-]+", "_", normalize_text(ds.doc_name))[:120]
                        path = out_dir / f"{file_prefix}_ORG_{idx:03d}_DOC_{k:03d}_{doc_slug}.txt"
                        with open(path, "w", encoding="utf-8") as f:
                            f.write("DOC_BEGIN\n")
                            f.write(f"ORG: {_strip_markdown_bold(json_org_text_raw).strip()}\n")
                            f.write(f"DOC: {ds.doc_name}\n")
                            f.write(ds.text)
                            f.write("\nDOC_END\n")
            else:
                org_missing += 1
                if verbose:
                    print(f"[WARN] ORG from JSON not found in body: {json_org_text_raw!r}")
                results.append(OrgBlockResult(
                    org=_strip_markdown_bold(json_org_text_raw).strip(),
                    org_block_text="",
                    docs=[],
                    status="org_missing",
                ))

    else:
        # -------- HIERARCHICAL MODE (ORG→ORG per occurrence) --------
        # 1) Build top_org blocks
        top_org_keys = {_org_key(item.get("top_org", {}).get("text", "")) for item in items}
        top_blocks = _build_org_blocks_coalesced_to_json(doc_text, spans, top_org_keys)

        top_lookup: Dict[str, Tuple[SpanInfo, int, int]] = {}
        for top_span, tstart, tend in top_blocks:
            top_lookup.setdefault(_org_key(top_span.text), (top_span, tstart, tend))

        # Count total sub_orgs as "orgs_total"
        total_orgs = sum(len(item.get("sub_orgs", [])) for item in items)

        # 2) For each top_org, cut sub_org occurrences ORG→next ORG and map to docs
        for item in items:
            top_org_raw = item.get("top_org", {}).get("text", "")
            top_key = _org_key(top_org_raw)
            sub_orgs = item.get("sub_orgs", [])

            if top_key not in top_lookup:
                # Top missing: mark every sub_org under it as org_missing
                for sub in sub_orgs:
                    sub_raw = sub.get("org", {}).get("text", "")
                    org_missing += 1
                    if verbose:
                        print(f"[WARN] top_org missing; marking sub_org missing: {sub_raw!r}")
                    results.append(OrgBlockResult(
                        org=_strip_markdown_bold(sub_raw).strip(),
                        org_block_text="",
                        docs=[],
                        status="org_missing",
                    ))
                continue

            _, tstart, tend = top_lookup[top_key]

            # Build valid sub-org keys for this item (from JSON)
            sub_keys = {_org_key(sub.get("org", {}).get("text", "")) for sub in sub_orgs}

            # Collect ALL occurrences of sub-org anchors within the top block
            sub_anchors = _collect_suborg_anchors_coalesced(doc_text, spans, tstart, tend, sub_keys)
            sub_anchors_sorted = sorted(sub_anchors, key=lambda s: s.start_char)

            # Precompute end boundary for each anchor: next sub-org anchor or end of top block
            end_by_anchor_id: Dict[int, int] = {}
            for idx_a, a in enumerate(sub_anchors_sorted):
                end_by_anchor_id[id(a)] = sub_anchors_sorted[idx_a + 1].start_char if idx_a + 1 < len(sub_anchors_sorted) else tend

            # Group anchors by sub-org key (PRESERVE ORDER, KEEP ALL OCCURRENCES)
            anchors_by_key: Dict[str, List[SpanInfo]] = {}
            for a in sub_anchors_sorted:
                anchors_by_key.setdefault(_org_key(a.text), []).append(a)

            # For each JSON sub_org: create one slice per occurrence and map sequentially to docs
            for sub in sub_orgs:
                sub_raw = sub.get("org", {}).get("text", "")
                sub_key = _org_key(sub_raw)
                json_docs = sub.get("docs", [])
                total_docs_expected += len(json_docs)

                occurs = anchors_by_key.get(sub_key, [])
                if not occurs:
                    org_missing += 1
                    if verbose:
                        print(f"[WARN] sub_org not found in body: {sub_raw!r}")
                    results.append(OrgBlockResult(
                        org=_strip_markdown_bold(sub_raw).strip(),
                        org_block_text="",
                        docs=[],
                        status="org_missing",
                    ))
                    continue

                # ORG→ORG slices for each occurrence
                slices: List[DocSlice] = []
                for occ in occurs:
                    s = occ.start_char
                    e = end_by_anchor_id[id(occ)]
                    slices.append(DocSlice(
                        doc_name=_strip_markdown_bold(occ.text).strip(),  # temp name; may be replaced by JSON doc title
                        text=doc_text[s:e]
                    ))

                # Map sequentially to JSON docs; rename first N slices with JSON titles
                matched_count = min(len(slices), len(json_docs))
                matched_slices: List[DocSlice] = []
                for i_pair in range(matched_count):
                    jdoc_raw = json_docs[i_pair].get("text", "")
                    matched_slices.append(DocSlice(
                        doc_name=_strip_markdown_bold(jdoc_raw).strip(),
                        text=slices[i_pair].text
                    ))
                # Append any extra body slices beyond JSON docs (keep body header name)
                if len(slices) > matched_count:
                    matched_slices.extend(slices[matched_count:])

                # Determine status
                if len(slices) == len(json_docs) == matched_count:
                    status = "ok"
                else:
                    status = "partial"

                total_docs_matched += len(matched_slices)

                results.append(OrgBlockResult(
                    org=_strip_markdown_bold(sub_raw).strip(),
                    org_block_text=doc_text[occurs[0].start_char : end_by_anchor_id[id(occurs[-1])]],
                    docs=matched_slices,
                    status=status,
                ))

                if write_doc_files and matched_slices:
                    for k, ds in enumerate(matched_slices, start=1):
                        doc_slug = re.sub(r"[^\w.-]+", "_", normalize_text(ds.doc_name))[:120]
                        path = out_dir / f"{file_prefix}_SUBORG_DOC_{k:03d}_{doc_slug}.txt"
                        with open(path, "w", encoding="utf-8") as f:
                            f.write("DOC_BEGIN\n")
                            f.write(f"ORG: {_strip_markdown_bold(sub_raw).strip()}\n")
                            f.write(f"DOC: {ds.doc_name}\n")
                            f.write(ds.text)
                            f.write("\nDOC_END\n")

    summary = {
        "orgs_total": total_orgs,
        "org_ok": org_ok,
        "org_partial": org_partial,
        "org_doc_missing": org_doc_missing,
        "org_missing": org_missing,
        "docs_expected": total_docs_expected,
        "docs_matched": total_docs_matched,
    }
    return results, summary





# --------------------------------------------------------------------------------------
# Convenience: pretty print a short summary to stdout (optional)
# --------------------------------------------------------------------------------------

def print_summary(summary: Dict[str, Any]) -> None:
    print(
        "ORGs: {org_ok} ok, {org_partial} partial, {org_doc_missing} doc-missing, {org_missing} missing / {orgs_total}".format(
            **summary
        )
    )
    print("Docs: {docs_matched} matched / {docs_expected}".format(**summary))
