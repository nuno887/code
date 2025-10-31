from __future__ import annotations
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import json
import re

from .types import SpanInfo, DocSlice, OrgBlockResult
from .normalization import normalize_text, normalize_doc_title, _org_key, _strip_markdown_bold

# --------------------------------------------------------------------------------------
# Labels
# --------------------------------------------------------------------------------------

ORG_LABELS = {"ORG_WITH_STAR_LABEL", "ORG_LABEL"}
DOC_NAME_LABEL = "DOC_NAME_LABEL"
CONTENT_LABELS = {"PARAGRAPH", "DOC_TEXT", DOC_NAME_LABEL}
IGNORE_LABELS = {"JUNK_LABEL"}

# --------------------------------------------------------------------------------------
# Span helpers
# --------------------------------------------------------------------------------------

def _collect_spans(doc) -> List[SpanInfo]:
    spans: List[SpanInfo] = []
    for ent in getattr(doc, "ents", []):
        label = getattr(ent, "label_", str(getattr(ent, "label", "")))
        if label in IGNORE_LABELS:
            continue
        spans.append(SpanInfo(label=label, text=str(ent.text), start_char=int(ent.start_char), end_char=int(ent.end_char)))
    spans.sort(key=lambda s: s.start_char)
    return spans

def _spans_within(spans: List[SpanInfo], start: int, end: int, labels: Optional[set] = None) -> List[SpanInfo]:
    out = []
    for s in spans:
        if start <= s.start_char < end:
            if labels is None or s.label in labels:
                out.append(s)
    return out

# --------------------------------------------------------------------------------------
# ORG block building
# --------------------------------------------------------------------------------------

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

def _collect_suborg_anchors_coalesced(
    doc_text: str,
    spans: List[SpanInfo],
    bstart: int,
    bend: int,
    valid_suborg_keys: set[str],
    *,
    max_merge: int = 3,
) -> List[SpanInfo]:
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

# --------------------------------------------------------------------------------------
# JSON I/O helpers
# --------------------------------------------------------------------------------------

def load_serieIII_minimal(path: Path | str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def coerce_items_payload(serieIII_json_or_path: Any) -> Dict[str, Any]:
    if isinstance(serieIII_json_or_path, dict):
        return serieIII_json_or_path
    return load_serieIII_minimal(serieIII_json_or_path)

# --------------------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------------------

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
    Slice a spaCy-parsed document body by organizations and document names using
    a JSON payload describing expected orgs/docs.

    Supports two JSON schemas:

    Flat:
      items[].org + docs  -> slice by DOC_NAME_LABEL (with fallbacks).

    Hierarchical:
      items[].top_org + sub_orgs[].org + sub_orgs[].docs
        -> inside each top_org block, cut ORG→next ORG for every occurrence
           of each sub_org header (no dedup), then map these slices sequentially
           to the sub_org's JSON docs.
    """
    data = coerce_items_payload(serieIII_json_or_path)
    items = data.get("items", [])

    spans = _collect_spans(doc_body)  # JUNK skipped
    doc_text = doc_body.text
    out_path = Path(out_dir)
    if write_org_files or write_doc_files:
        out_path.mkdir(parents=True, exist_ok=True)

    hierarchical = any("top_org" in it for it in items)

    results: List[OrgBlockResult] = []
    org_ok = org_partial = org_doc_missing = org_missing = 0
    total_orgs = 0
    total_docs_expected = 0
    total_docs_matched = 0

    def _slice_docs_in_block(
        bstart: int,
        bend: int,
        json_docs: List[Dict[str, Any]],
        org_label_text_raw: str,
    ) -> Tuple[List[DocSlice], List[str], str, int]:
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
                if s.start_char > start_char and normalize_doc_title(s.text) in json_name_set:
                    return s.start_char
            return bend

        json_doc_names = [normalize_doc_title(d.get("text", "")) for d in json_docs]
        json_name_set = set(json_doc_names)
        i_ptr = 0
        for jdoc_raw, jdoc_norm in zip([d.get("text", "") for d in json_docs], json_doc_names):
            while i_ptr < len(docname_spans_sorted) and normalize_doc_title(docname_spans_sorted[i_ptr].text) != jdoc_norm:
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

        # Fallback: prefer only headers that appear in JSON
        if not matched_slices and docname_spans_sorted:
            matching_headers = [h for h in docname_spans_sorted if normalize_doc_title(h.text) in json_name_set]
            if matching_headers:
                if verbose:
                    print(f"[INFO] Fallback: slicing {len(matching_headers)} matching headers for ORG {org_label_text_raw!r}")
                for h in matching_headers:
                    start = h.start_char
                    end = _slice_end(start)
                    matched_slices.append(DocSlice(
                        doc_name=_strip_markdown_bold(h.text).strip(),
                        text=doc_text[start:end]
                    ))
                matched_norms = {normalize_doc_title(h.text) for h in matching_headers}
                unmatched_docs = [d.get("text", "") for d in json_docs if normalize_doc_title(d.get("text", "")) not in matched_norms]
            else:
                if len(json_docs) == 1:
                    jdoc_raw = json_docs[0].get("text", "")
                    matched_slices = [DocSlice(
                        doc_name=_strip_markdown_bold(jdoc_raw).strip(),
                        text=doc_text[bstart:bend]
                    )]
                    unmatched_docs = []
                else:
                    if verbose:
                        print(f"[INFO] Fallback: slicing all {len(docname_spans_sorted)} headers for ORG {org_label_text_raw!r}")
                    for h in docname_spans_sorted:
                        start = h.start_char
                        end = _slice_end(start)
                        matched_slices.append(DocSlice(
                            doc_name=_strip_markdown_bold(h.text).strip(),
                            text=doc_text[start:end]
                        ))
                    unmatched_docs = [d.get("text", "") for d in json_docs]

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
        # FLAT MODE
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
                    path = out_path / f"{file_prefix}_ORG_{idx:03d}_{org_slug}.txt"
                    with open(path, "w", encoding="utf-8") as f:
                        f.write("DOC_BEGIN\\n")
                        f.write(f"ORG: {_strip_markdown_bold(json_org_text_raw).strip()}\\n")
                        f.write(block_text)
                        f.write("\\nDOC_END\\n")
                if write_doc_files and matched_slices:
                    for k, ds in enumerate(matched_slices, start=1):
                        doc_slug = re.sub(r"[^\w.-]+", "_", normalize_text(ds.doc_name))[:120]
                        path = out_path / f"{file_prefix}_ORG_{idx:03d}_DOC_{k:03d}_{doc_slug}.txt"
                        with open(path, "w", encoding="utf-8") as f:
                            f.write("DOC_BEGIN\\n")
                            f.write(f"ORG: {_strip_markdown_bold(json_org_text_raw).strip()}\\n")
                            f.write(f"DOC: {ds.doc_name}\\n")
                            f.write(ds.text)
                            f.write("\\nDOC_END\\n")
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
        # HIERARCHICAL MODE
        top_org_keys = {_org_key(item.get("top_org", {}).get("text", "")) for item in items}
        top_blocks = _build_org_blocks_coalesced_to_json(doc_text, spans, top_org_keys)

        top_lookup: Dict[str, Tuple[SpanInfo, int, int]] = {}
        for top_span, tstart, tend in top_blocks:
            top_lookup.setdefault(_org_key(top_span.text), (top_span, tstart, tend))

        total_orgs = sum(len(item.get("sub_orgs", [])) for item in items)

        for item in items:
            top_org_raw = item.get("top_org", {}).get("text", "")
            top_key = _org_key(top_org_raw)
            sub_orgs = item.get("sub_orgs", [])

            if top_key not in top_lookup:
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

            sub_keys = {_org_key(sub.get("org", {}).get("text", "")) for sub in sub_orgs}
            sub_anchors = _collect_suborg_anchors_coalesced(doc_text, spans, tstart, tend, sub_keys)
            sub_anchors_sorted = sorted(sub_anchors, key=lambda s: s.start_char)

            end_by_anchor_id: Dict[int, int] = {}
            for idx_a, a in enumerate(sub_anchors_sorted):
                end_by_anchor_id[id(a)] = sub_anchors_sorted[idx_a + 1].start_char if idx_a + 1 < len(sub_anchors_sorted) else tend

            anchors_by_key: Dict[str, List[SpanInfo]] = {}
            for a in sub_anchors_sorted:
                anchors_by_key.setdefault(_org_key(a.text), []).append(a)

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

                slices: List[DocSlice] = []
                for occ in occurs:
                    s = occ.start_char
                    e = end_by_anchor_id[id(occ)]
                    slices.append(DocSlice(
                        doc_name=_strip_markdown_bold(occ.text).strip(),
                        text=doc_text[s:e]
                    ))

                matched_count = min(len(slices), len(json_docs))
                matched_slices: List[DocSlice] = []
                for i_pair in range(matched_count):
                    jdoc_raw = json_docs[i_pair].get("text", "")
                    matched_slices.append(DocSlice(
                        doc_name=_strip_markdown_bold(jdoc_raw).strip(),
                        text=slices[i_pair].text
                    ))
                if len(slices) > matched_count:
                    matched_slices.extend(slices[matched_count:])

                status = "ok" if len(slices) == len(json_docs) == matched_count else "partial"
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
                        path = out_path / f"{file_prefix}_SUBORG_DOC_{k:03d}_{doc_slug}.txt"
                        with open(path, "w", encoding="utf-8") as f:
                            f.write("DOC_BEGIN\\n")
                            f.write(f"ORG: {_strip_markdown_bold(sub_raw).strip()}\\n")
                            f.write(f"DOC: {ds.doc_name}\\n")
                            f.write(ds.text)
                            f.write("\\nDOC_END\\n")

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

def print_summary(summary: Dict[str, Any]) -> None:
    print(
        "ORGs: {org_ok} ok, {org_partial} partial, {org_doc_missing} doc-missing, {org_missing} missing / {orgs_total}".format(
            **summary
        )
    )
    print("Docs: {docs_matched} matched / {docs_expected}".format(**summary))