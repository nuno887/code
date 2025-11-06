# compare.py
from __future__ import annotations

import json
import re
import unicodedata
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Set, Tuple


# ----------------------------
# Normalization helpers
# ----------------------------

def _nfkc(s: str) -> str:
    return unicodedata.normalize("NFKC", s)


_DASHES_RE = re.compile(r"[\u2012\u2013\u2014\u2015\u2212]+")
_ELLIPSIS_RE = re.compile(r"[\u2026\.]{3,}")
_WS_RE = re.compile(r"\s+")
_QUOTES_RE = re.compile(r"^[\"'«»“”‘’\(\)\[\]\{\}]+|[\"'«»“”‘’\(\)\[\]\{\}]+$")
_TRAILING_PUNCT_RE = re.compile(r"^[\s,.;:]+|[\s,.;:]+$")
# Map No., Nº, n.º, etc. to canonical "n.º"
_N_DOT_REPLACEMENTS = [
    (re.compile(r"(?i)\bno\.\b"), "n.º"),
    (re.compile(r"(?i)\bnº\b"), "n.º"),
    (re.compile(r"(?i)\bn\.º\b"), "n.º"),
    (re.compile(r"(?i)\bn°\b"), "n.º"),
    (re.compile(r"(?i)\bn\.\s*o\b"), "n.º"),
    (re.compile(r"(?i)\bn\s*o\b"), "n.º"),
]
# Normalize "6 / 2025" -> "6/2025"
_SLASH_WS_RE = re.compile(r"\s*/\s*")


def _canon(s: str) -> str:
    """General canonicalizer (kept for reference; not used when letters-only is active)."""
    if not s:
        return ""
    s = _nfkc(s)
    s = _DASHES_RE.sub("-", s)
    s = _ELLIPSIS_RE.sub(".", s)
    for pat, repl in _N_DOT_REPLACEMENTS:
        s = pat.sub(repl, s)
    s = _SLASH_WS_RE.sub("/", s)
    s = _WS_RE.sub(" ", s).strip()
    s = s.strip()
    s = _TRAILING_PUNCT_RE.sub("", s)
    s = _WS_RE.sub(" ", s).strip().lower()
    s = _QUOTES_RE.sub("", s).strip()
    return s


def _letters_key(s: str) -> str:
    """
    Keep only Unicode letters (A–Z incl. accents, Cyrillic, etc.), lowercase.
    Removes spaces, punctuation, digits, symbols. NFKC first for consistency.
    """
    if not s:
        return ""
    s = unicodedata.normalize("NFKC", s).lower()
    return "".join(ch for ch in s if ch.isalpha())


def norm_org(s: str) -> str:
    # Letters-only comparison key for organizations
    return _letters_key(s)


def norm_doc(s: str) -> str:
    # Letters-only comparison key for document names
    return _letters_key(s)


# ----------------------------
# Data structures
# ----------------------------

@dataclass(frozen=True)
class Pair:
    org: str
    doc: str


@dataclass
class CoverageResult:
    mode_name: str
    payload_pairs: Set[Pair]
    slim_pairs: Set[Pair]
    missing_in_slim: Set[Pair]
    extra_in_slim: Set[Pair]
    intersection: Set[Pair]
    coverage_pct: float
    # Per-org detail
    payload_docs_by_org: Dict[str, Set[str]]
    slim_docs_by_org: Dict[str, Set[str]]
    docless_payload_count: int
    payload_docs_extracted_count: int = 0
    payload_children_count: int = 0
    payload_children_without_doc_count: int = 0


# ----------------------------
# Extraction from SLIM
# ----------------------------

def extract_pairs_from_slim(
    slim: dict,
) -> Tuple[Set[Pair], Set[Pair], Dict[str, Set[str]], Dict[str, Set[str]]]:
    """
    Returns:
      strict_pairs, relaxed_pairs, strict_docs_by_org, relaxed_docs_by_org
    strict: org_key = "org :: sub_org" if sub_org available else "org"
    relaxed: org_key = "org" only
    """
    strict_pairs: Set[Pair] = set()
    relaxed_pairs: Set[Pair] = set()
    strict_by_org: Dict[str, Set[str]] = defaultdict(set)
    relaxed_by_org: Dict[str, Set[str]] = defaultdict(set)

    for d in (slim.get("docs") or []):
        org = norm_org(d.get("org") or "")
        sub_org_raw = d.get("sub_org") or ""
        sub_org = norm_org(sub_org_raw) if sub_org_raw else ""
        doc_name_raw = d.get("doc_name") or ""
        doc = norm_doc(doc_name_raw)
        if not org or not doc:
            continue

        org_key_relaxed = org
        org_key_strict = f"{org} :: {sub_org}" if sub_org else org

        p_strict = Pair(org_key_strict, doc)
        p_relaxed = Pair(org_key_relaxed, doc)

        strict_pairs.add(p_strict)
        relaxed_pairs.add(p_relaxed)

        strict_by_org[org_key_strict].add(doc)
        relaxed_by_org[org_key_relaxed].add(doc)

    return strict_pairs, relaxed_pairs, strict_by_org, relaxed_by_org


# ----------------------------
# Extraction from PAYLOAD
# ----------------------------

def _is_serie_iii_payload(payload: dict) -> bool:
    # Heuristic: Série III has "orgs" (list of {id,text,label}) and items with "org_ids"
    return isinstance(payload.get("orgs"), list) and any(
        isinstance(it, dict) and "org_ids" in it for it in (payload.get("items") or [])
    )


def extract_pairs_from_payload(payload: dict) -> Tuple[Set[Pair], Dict[str, Set[str]], int, int, int, int]:
    """
    Returns:
      payload_pairs, payload_docs_by_org, docless_count, doc_extracted_count,
      children_total, children_under_docless
    Note: payload pairs are always "relaxed" (org only) because payload has no sub_org key.
    """
    pairs: Set[Pair] = set()
    by_org: Dict[str, Set[str]] = defaultdict(set)
    docless = 0
    doc_extracted = 0
    children_total = 0
    children_under_docless = 0

    items = payload.get("items") or []

    if _is_serie_iii_payload(payload):
        # Build org id -> text
        org_lookup = {int(o["id"]): o.get("text", "") for o in (payload.get("orgs") or []) if "id" in o}
        for it in items:
            # Count Série III children kept by exporter (already cleaned)
            ch_list = it.get("children") or []
            kept = sum(1 for ch in ch_list if (ch.get("child") or "").strip())
            children_total += kept

            org_ids = it.get("org_ids") or []
            doc_name = it.get("doc_name")
            if doc_name is None:
                docless += 1
                children_under_docless += kept
                continue
            doc_text = norm_doc((doc_name.get("text") or ""))
            if not doc_text:
                docless += 1
                children_under_docless += kept
                continue
            doc_extracted += 1
            for oid in org_ids:
                org_text = norm_org(org_lookup.get(int(oid), ""))
                if not org_text:
                    continue
                p = Pair(org_text, doc_text)
                pairs.add(p)
                by_org[org_text].add(doc_text)
    else:
        # Others payload
        for it in items:
            if "top_org" in it:
                # STAR paragraph: docs live under sub_orgs[].docs[]
                for so in (it.get("sub_orgs") or []):
                    org_text = norm_org(((so.get("org") or {}).get("text") or ""))
                    if not org_text:
                        continue
                    for d in (so.get("docs") or []):
                        doc_text = norm_doc(d.get("text") or "")
                        if not doc_text:
                            continue
                        doc_extracted += 1
                        p = Pair(org_text, doc_text)
                        pairs.add(p)
                        by_org[org_text].add(doc_text)
            elif "org" in it:
                org_text = norm_org(((it.get("org") or {}).get("text") or ""))
                if not org_text:
                    continue
                for d in (it.get("docs") or []):
                    doc_text = norm_doc(d.get("text") or "")
                    if not doc_text:
                        continue
                    doc_extracted += 1
                    p = Pair(org_text, doc_text)
                    pairs.add(p)
                    by_org[org_text].add(doc_text)
            else:
                # No recognizable structure; skip
                continue

    return pairs, by_org, docless, doc_extracted, children_total, children_under_docless


# ----------------------------
# Core comparison
# ----------------------------

@dataclass
class CoverageResult:
    mode_name: str
    payload_pairs: Set[Pair]
    slim_pairs: Set[Pair]
    missing_in_slim: Set[Pair]
    extra_in_slim: Set[Pair]
    intersection: Set[Pair]
    coverage_pct: float
    payload_docs_by_org: Dict[str, Set[str]]
    slim_docs_by_org: Dict[str, Set[str]]
    docless_payload_count: int
    payload_docs_extracted_count: int = 0
    payload_children_count: int = 0
    payload_children_without_doc_count: int = 0


def _coverage(payload_pairs: Set[Pair], slim_pairs: Set[Pair]) -> CoverageResult:
    inter = payload_pairs & slim_pairs
    missing = payload_pairs - slim_pairs
    extra = slim_pairs - payload_pairs
    pct = (len(inter) / len(payload_pairs) * 100.0) if payload_pairs else 100.0
    # Placeholder; overridden by caller with per-org maps and counters
    return CoverageResult(
        mode_name="",
        payload_pairs=payload_pairs,
        slim_pairs=slim_pairs,
        missing_in_slim=missing,
        extra_in_slim=extra,
        intersection=inter,
        coverage_pct=pct,
        payload_docs_by_org={},
        slim_docs_by_org={},
        docless_payload_count=0,
    )


def compare_slim_vs_payload(slim: dict, payload: dict) -> Dict[str, CoverageResult]:
    """
    Returns a dict with results for:
      - strict (org :: sub_org)
      - relaxed (org)
    """
    # From slim
    slim_strict, slim_relaxed, slim_docs_by_org_strict, slim_docs_by_org_relaxed = extract_pairs_from_slim(slim)

    # From payload (always org-only)
    payload_pairs_relaxed, payload_docs_by_org, docless, doc_extracted, ch_total, ch_docless = extract_pairs_from_payload(payload)

    # Strict comparison (payload org-only vs slim strict key)
    r_strict = _coverage(payload_pairs_relaxed, slim_strict)
    r_strict.mode_name = "strict (org :: sub_org)"
    r_strict.payload_docs_by_org = payload_docs_by_org
    r_strict.slim_docs_by_org = slim_docs_by_org_strict
    r_strict.docless_payload_count = docless
    r_strict.payload_docs_extracted_count = doc_extracted
    r_strict.payload_children_count = ch_total
    r_strict.payload_children_without_doc_count = ch_docless

    # Relaxed comparison (payload org-only vs slim relaxed key)
    r_relaxed = _coverage(payload_pairs_relaxed, slim_relaxed)
    r_relaxed.mode_name = "relaxed (org only)"
    r_relaxed.payload_docs_by_org = payload_docs_by_org
    r_relaxed.slim_docs_by_org = slim_docs_by_org_relaxed
    r_relaxed.docless_payload_count = docless
    r_relaxed.payload_docs_extracted_count = doc_extracted
    r_relaxed.payload_children_count = ch_total
    r_relaxed.payload_children_without_doc_count = ch_docless

    return {"strict": r_strict, "relaxed": r_relaxed}


# ----------------------------
# Formatting
# ----------------------------

def _format_pairs(pairs: Iterable[Pair], limit: int = 20) -> List[str]:
    pairs_list = list(pairs)
    out: List[str] = []
    for i, p in enumerate(pairs_list):
        if i >= limit:
            out.append(f"... (+{len(pairs_list) - limit} more)")
            break
        out.append(f"[{p.org}] — {p.doc}")
    return out


def _format_per_org_diff(
    payload_docs_by_org: Dict[str, Set[str]],
    slim_docs_by_org: Dict[str, Set[str]],
    limit_orgs: int = 20,
    limit_docs: int = 10,
) -> List[str]:
    lines: List[str] = []
    orgs = sorted(set(payload_docs_by_org.keys()) | set(slim_docs_by_org.keys()))
    shown = 0
    for org in orgs:
        payload_docs = payload_docs_by_org.get(org, set())
        slim_docs = slim_docs_by_org.get(org, set())
        missing = sorted(payload_docs - slim_docs)
        extra = sorted(slim_docs - payload_docs)
        if not missing and not extra:
            continue
        if shown >= limit_orgs:
            lines.append(f"... (+{len(orgs) - limit_orgs} orgs with differences)")
            break
        lines.append(f"  • {org}")
        if missing:
            mlist = ", ".join(missing[:limit_docs]) + (f", +{len(missing)-limit_docs} more" if len(missing) > limit_docs else "")
            lines.append(f"     - missing in slim: {mlist}")
        if extra:
            elist = ", ".join(extra[:limit_docs]) + (f", +{len(extra)-limit_docs} more" if len(extra) > limit_docs else "")
            lines.append(f"     - extra in slim:   {elist}")
        shown += 1
    return lines


def render_report(results: Dict[str, CoverageResult], *, show_per_org: bool = True) -> str:
    r_strict = results["strict"]
    r_relaxed = results["relaxed"]

    def block(r: CoverageResult) -> List[str]:
        lines = []
        lines.append(f"Mode: {r.mode_name}")
        lines.append(f"  Payload pairs: {len(r.payload_pairs)}")
        lines.append(f"  Slim pairs:    {len(r.slim_pairs)}")
        lines.append(f"  Coverage:      {r.coverage_pct:.1f}%")
        lines.append(f"  Missing in slim: {len(r.missing_in_slim)}")
        lines.append(f"  Extra in slim:   {len(r.extra_in_slim)}")
        lines.append(f"  Docless items in payload: {r.docless_payload_count}")
        lines.append(f"  Docs extracted in payload: {r.payload_docs_extracted_count}")
        lines.append(f"  Children extracted in payload (Série III): {r.payload_children_count}")
        lines.append(f"  Children under docless items: {r.payload_children_without_doc_count}")
        if r.missing_in_slim:
            lines.append("  Top missing:")
            lines += [f"    {s}" for s in _format_pairs(sorted(r.missing_in_slim, key=lambda x: (x.org, x.doc)))]
        if r.extra_in_slim:
            lines.append("  Top extra:")
            lines += [f"    {s}" for s in _format_pairs(sorted(r.extra_in_slim, key=lambda x: (x.org, x.doc)))]
        if show_per_org:
            lines.append("  Per-org differences:")
            lines += _format_per_org_diff(r.payload_docs_by_org, r.slim_docs_by_org)
        return lines

    out: List[str] = []
    out += block(r_strict)
    out.append("")
    out += block(r_relaxed)
    return "\n".join(out)


# ----------------------------
# CLI
# ----------------------------

def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def main():
    import argparse

    ap = argparse.ArgumentParser(description="Compare slim vs payload coverage (org, doc_name).")
    ap.add_argument("--slim", required=True, type=Path, help="Path to slim JSON (build_slim_payload output)")
    ap.add_argument("--payload", required=True, type=Path, help="Path to payload JSON (Serie III or Others)")
    ap.add_argument("--json-out", type=Path, help="Optional: write machine-readable diff JSON here")
    ap.add_argument("--no-per-org", action="store_true", help="Do not print per-org diff block")
    args = ap.parse_args()

    slim = _load_json(args.slim)
    payload = _load_json(args.payload)

    results = compare_slim_vs_payload(slim, payload)
    report = render_report(results, show_per_org=not args.no_per_org)
    print(report)

    if args.json_out:
        out = {}
        for k, r in results.items():
            out[k] = {
                "mode": r.mode_name,
                "coverage_pct": r.coverage_pct,
                "payload_pairs": sorted([[p.org, p.doc] for p in r.payload_pairs]),
                "slim_pairs": sorted([[p.org, p.doc] for p in r.slim_pairs]),
                "missing_in_slim": sorted([[p.org, p.doc] for p in r.missing_in_slim]),
                "extra_in_slim": sorted([[p.org, p.doc] for p in r.extra_in_slim]),
                "docless_payload_count": r.docless_payload_count,
                "docs_extracted_in_payload": r.payload_docs_extracted_count,
                "children_extracted_in_payload": r.payload_children_count,
                "children_under_docless_items": r.payload_children_without_doc_count,
            }
        args.json_out.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
