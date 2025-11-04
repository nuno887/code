
"""
result_normalizer.py

Converts the two splitter outputs into a single, stable schema for FastAPI.

Unified schema:
{
  "api_version": "1.0",
  "mode": "serie_iii" | "serie_std",
  "input_meta": {...},
  "summary": {
    "orgs_total": int,
    "docs_expected": int,
    "docs_matched": int,
    "org_ok": int,
    "org_partial": int,
    "org_doc_missing": int,
    "org_missing": int,
    "serie_iii": { ... } | absent,
    "serie_std": { ... } | absent
  },
  "results": [
    {
      "org": str,
      "status": str,
      "org_block_text": str | "",
      "docs": [
        {
          "doc_name": str,
          "text": str,
          "status": "segment" | "unanchored" | "children_segment",
          "confidence": float?,
          "ents": [{"label": str, "text": str, "start": int, "end": int}]?,
          "subs": [{"title": str, "headers"?: [str], "body": str}]?,
          "extras": dict?
        }
      ],
      "extras": dict?
    }
  ]
}
"""
from __future__ import annotations
from typing import Any, Dict, List, Tuple
import json
from pathlib import Path
import re

API_VERSION = "1.0"

# --- Public API -------------------------------------------------------------

def normalize_results(
    mode: str,
    raw_results: List[Any],
    raw_summary: Dict[str, Any],
    input_meta: Dict[str, Any] | None = None,
    *,
    serie_std_mode: str | None = None,  # "flat" | "hierarchical" (optional hint)
) -> Dict[str, Any]:
    """Normalize outputs from either Serie III or Serie I/II/IV into a unified dict.

    Parameters
    ----------
    mode: "serie_iii" or "serie_std" (I/II/IV)
    raw_results: list from your splitter
    raw_summary: dict from your splitter
    input_meta: optional metadata (filename, text_length, etc.)
    serie_std_mode: optional hint for std payload mode ("flat" or "hierarchical")
    """
    if mode not in {"serie_iii", "serie_std"}:
        raise ValueError("mode must be 'serie_iii' or 'serie_std'")

    if mode == "serie_iii":
        results = [_map_org_serie_iii(orgres) for orgres in raw_results]
        summary = _summary_serie_iii(raw_summary)
    else:
        results = [_map_org_serie_std(orgres) for orgres in raw_results]
        summary = _summary_serie_std(raw_summary, serie_std_mode)

    return {
        "api_version": API_VERSION,
        "mode": mode,
        "input_meta": input_meta or {},
        "summary": summary,
        "results": results,
    }

# --- Serie III mapping ------------------------------------------------------

def _map_org_serie_iii(orgres: Any) -> Dict[str, Any]:
    """Map an OrgResult (Serie III) into the unified OrgResult dict."""
    org = getattr(orgres, "org", None) or orgres.get("org")
    status = getattr(orgres, "status", None) or orgres.get("status")
    docs = getattr(orgres, "docs", None) or orgres.get("docs", [])

    return {
        "org": org or "",
        "status": status or "no_window",
        # Serie III does not expose a whole-block org text; leave empty string for uniformity
        "org_block_text": "",
        "docs": [_map_doc_serie_iii(d) for d in (docs or [])],
        # Room for future window info, if present in your OrgResult type
        # "extras": {"window_index": ..., "window_bounds": {"start":..., "end":...}}
    }


def _map_doc_serie_iii(ds: Any) -> Dict[str, Any]:
    name = getattr(ds, "doc_name", None) or ds.get("doc_name", "")
    text = getattr(ds, "text", None) or ds.get("text", "")
    status_raw = getattr(ds, "status", None) or ds.get("status") or "doc_type_segment"
    confidence = getattr(ds, "confidence", None) or ds.get("confidence")
    ents = getattr(ds, "ents", None) or ds.get("ents")
    subs = getattr(ds, "subs", None) or ds.get("subs")

    status = _map_doc_status_serie_iii(status_raw)

    out: Dict[str, Any] = {
        "doc_name": name,
        "text": text,
        "status": status,
    }
    if confidence is not None:
        out["confidence"] = float(confidence)
    if ents:
        out["ents"] = [_tuple_ent_to_dict(e) for e in ents]
    if subs:
        out["subs"] = [_map_subslice(s) for s in subs]
    return out


def _map_doc_status_serie_iii(status_raw: str) -> str:
    # Map Serie III labels to unified enums
    mapping = {
        "doc_type_segment": "segment",
        "doc_children_segment": "children_segment",
        "doc_type_unanchored": "unanchored",
    }
    return mapping.get(status_raw, "segment")


def _tuple_ent_to_dict(e: Any) -> Dict[str, Any]:
    # Accept either tuple-like (lbl, txt, st, en) or dict-like
    if isinstance(e, dict):
        return {
            "label": e.get("label", ""),
            "text": e.get("text", ""),
            "start": int(e.get("start", 0)),
            "end": int(e.get("end", 0)),
        }
    try:
        lbl, txt, st, en = e  # type: ignore[misc]
        return {"label": str(lbl), "text": str(txt), "start": int(st), "end": int(en)}
    except Exception:
        return {"label": "", "text": str(e), "start": 0, "end": 0}


def _map_subslice(s: Any) -> Dict[str, Any]:
    # Case 0: object with attributes (e.g., a dataclass or custom SubSlice)
    # Handle this BEFORE other cases so real objects don't fall through to str(s).
    if hasattr(s, "title") or hasattr(s, "body"):
        out: Dict[str, Any] = {
            "title": getattr(s, "title", "") or "",
            "body": getattr(s, "body", "") or "",
        }
        headers = getattr(s, "headers", None)
        if headers:
            out["headers"] = list(headers)
        start = getattr(s, "start", None)
        end = getattr(s, "end", None)
        if isinstance(start, int):
            out["start"] = start
        if isinstance(end, int):
            out["end"] = end
        return out

    # Case 1: already dict-like
    if isinstance(s, dict):
        out = {"title": s.get("title", ""), "body": s.get("body", "")}
        if s.get("headers"):
            out["headers"] = list(s["headers"])
        if "start" in s:
            try: out["start"] = int(s["start"])
            except Exception: pass
        if "end" in s:
            try: out["end"] = int(s["end"])
            except Exception: pass
        return out

    # Case 2: string repr like "SubSlice(title='...', headers=[...], body='...', start=0, end=7)"
    if isinstance(s, str) and s.lstrip().startswith("SubSlice("):
        def _m(pat: str, text: str, group: int = 1):
            m = re.search(pat, text, flags=re.S)
            return m.group(group) if m else None

        title = _m(r"title=([\"'])(.*?)\1", s, 2) or ""
        body = _m(r"body=([\"'])(.*?)\1", s, 2) or ""
        headers_raw = _m(r"headers=\[(.*?)\]", s, 1)
        start = _m(r"\bstart=(\d+)", s, 1)
        end = _m(r"\bend=(\d+)", s, 1)

        out: Dict[str, Any] = {"title": title, "body": body}
        if headers_raw:
            out["headers"] = [m.group(2) for m in re.finditer(r"([\"'])(.*?)\1", headers_raw)]
        if start is not None:
            try: out["start"] = int(start)
            except Exception: pass
        if end is not None:
            try: out["end"] = int(end)
            except Exception: pass
        return out

    # Case 3: tuple-like fallback (title, headers?, body?, start?, end?)
    try:
        title = s[0]
        body = s[2] if len(s) > 2 else ""
        out = {"title": str(title), "body": str(body)}
        if len(s) > 1 and s[1]:
            out["headers"] = list(s[1])
        if len(s) > 3 and isinstance(s[3], int): out["start"] = s[3]
        if len(s) > 4 and isinstance(s[4], int): out["end"] = s[4]
        return out
    except Exception:
        # Final safety: keep something useful instead of a noisy repr
        return {"title": "", "body": str(s)}





def _summary_serie_iii(s: Dict[str, Any]) -> Dict[str, Any]:
    # Compose the unified summary with a namespaced block for Serie III
    return {
        "orgs_total": int(s.get("orgs_in_payload", 0)),
        "docs_expected": int(s.get("doc_type_headers_matched", 0)),  # best proxy across modes
        "docs_matched": int(s.get("doc_type_segments", 0)),
        "org_ok": 0,
        "org_partial": 0,
        "org_doc_missing": 0,
        "org_missing": int(s.get("orgs_in_payload", 0)) - int(s.get("org_windows_found", 0)),
        "serie_iii": {
            "orgs_in_payload": int(s.get("orgs_in_payload", 0)),
            "org_windows_found": int(s.get("org_windows_found", 0)),
            "doc_type_headers_matched": int(s.get("doc_type_headers_matched", 0)),
            "doc_type_segments": int(s.get("doc_type_segments", 0)),
            "segment_reparsed": bool(s.get("segment_reparsed", False)),
            "segments_with_subdivisions": int(s.get("segments_with_subdivisions", 0)),
        },
    }

# --- Serie I/II/IV (std) mapping -------------------------------------------

def _map_org_serie_std(orgres: Any) -> Dict[str, Any]:
    org = getattr(orgres, "org", None) or orgres.get("org")
    status = getattr(orgres, "status", None) or orgres.get("status")
    docs = getattr(orgres, "docs", None) or orgres.get("docs", [])
    org_block_text = getattr(orgres, "org_block_text", None) or orgres.get("org_block_text", "")

    return {
        "org": org or "",
        "status": status or "org_missing",
        "org_block_text": org_block_text or "",
        "docs": [_map_doc_serie_std(d) for d in (docs or [])],
    }


def _map_doc_serie_std(ds: Any) -> Dict[str, Any]:
    name = getattr(ds, "doc_name", None) or ds.get("doc_name", "")
    text = getattr(ds, "text", None) or ds.get("text", "")
    return {
        "doc_name": name,
        "text": text,
        "status": "segment",  # std path does not differentiate per-slice statuses
    }


def _summary_serie_std(s: Dict[str, Any], mode_hint: str | None) -> Dict[str, Any]:
    return {
        "orgs_total": int(s.get("orgs_total", 0)),
        "docs_expected": int(s.get("docs_expected", 0)),
        "docs_matched": int(s.get("docs_matched", 0)),
        "org_ok": int(s.get("org_ok", 0)),
        "org_partial": int(s.get("org_partial", 0)),
        "org_doc_missing": int(s.get("org_doc_missing", 0)),
        "org_missing": int(s.get("org_missing", 0)),
        "serie_std": {"mode": str(mode_hint) if mode_hint else _infer_std_mode(s)},
    }


def _infer_std_mode(s: Dict[str, Any]) -> str:
    # If you want, you can pass the explicit mode via `serie_std_mode`.
    # This heuristic just returns "flat" by default.
    return "flat"



def _maybe_prepend_doc_name(
    text: str,
    doc_name: str,
    *,
    bold: bool = True,
    separator: str = "\n\n",
    skip_if_already_present: bool = True,
) -> str:
    """Return text with doc_name prepended, guarding against duplication."""
    if not isinstance(text, str) or not doc_name:
        return text

    lead = text.lstrip()
    # Strip an initial markdown **...** to check duplication
    lead_no_bold = re.sub(r"^\*{2}(.*?)\*{2}", r"\1", lead).lstrip()
    if skip_if_already_present and lead_no_bold.startswith(doc_name):
        return text

    prefix = f"**{doc_name}**" if bold else doc_name
    return f"{prefix}{separator}{text}"


# --- Slim payload builder ---------------------------------------------------

def build_slim_payload(
    filename: str,
    unified: Dict[str, Any],
    *,
    include_text: bool = True,
    split_by_subs: bool = True,  # When True, emit one entry per subdivision if available
    include_headers: bool = False,  # If True, include subdivision headers array when present
) -> Dict[str, Any]:
    """Create a slimmed-down payload for downstream consumers.

    Output shape:
    {
      "file": str,
      "docs": [
        { "org": str, "doc_name": str, "text"?: str }
      ]
    }

    Notes:
    - No truncation is performed.
    - If `split_by_subs` is True and a doc slice has subdivisions, the output will
      contain **one entry per subdivision** (using the subdivision's body as text).
      In that case the top-level doc text is **not** included, to avoid duplicates.
    - If a doc slice has no subdivisions or `split_by_subs` is False, a single entry
      is emitted per doc slice (with its full text when `include_text` is True).
    - If `include_headers` is True, subdivision headers (if any) are included.
    """
    docs_out: List[Dict[str, Any]] = []
    for orgres in unified.get("results", []):
        org_name = orgres.get("org", "")
        for ds in orgres.get("docs", []):
            subs = ds.get("subs") or []
            if split_by_subs and subs:
                for sub in subs:
                    entry = {
                        "org": org_name,
                        "doc_name": ds.get("doc_name", ""),
                        "section_title": sub.get("title", ""),
                    }
                    if include_text:
                        entry["text"] = sub.get("body", "")
                    if include_headers and sub.get("headers"):
                        entry["headers"] = sub.get("headers")
                    docs_out.append(entry)
            else:
                entry = {
                    "org": org_name,
                    "doc_name": ds.get("doc_name", ""),
                }
                if include_text:
                    entry["text"] = ds.get("text", "")
                docs_out.append(entry)

    return {"file": filename, "docs": docs_out}


# --- File output helpers ----------------------------------------------------

def build_output_paths(filename: str, out_dir: str | Path = "output_json") -> Dict[str, Path]:
    """Return default paths for unified and slim JSON files for a given PDF filename."""
    out_dir = Path(out_dir)
    stem = Path(filename).stem
    return {
        "dir": out_dir,
        "unified": out_dir / f"{stem}__unified.json",
        "slim": out_dir / f"{stem}__slim.json",
    }


def save_json(data: Dict[str, Any], path: str | Path) -> None:
    """Write JSON to disk with UTF-8 and pretty formatting (no ASCII escaping)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
