# processing.py
from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from serie3_splitter.nlp import get_nlp
from serie3_splitter import divide_body_by_org_and_docs_serieIII
from pdf_markup import extract_pdf_to_markdown
from Split_TEXT import split_sumario_and_body
from relations_extractor import RelationExtractor, export_relations_items_minimal_json
from relations_extractor_serieIII import (
    RelationExtractorSerieIII,
    export_serieIII_items_minimal_json,
)

try:
    from body_extraction import divide_body_by_org_and_docs as divide_body_I_II
except Exception:
    divide_body_I_II = None  # If you only want Série III, leave this as None.

# ---- Singleton NLP ----
_nlp = None
def _nlp_instance():
    global _nlp
    if _nlp is None:
        _nlp = get_nlp(disable_ner=True)
    return _nlp

def _is_serie_iii_from_name(filename: str) -> bool:
    return "iiiserie" in (filename or "").lower()

def _build_docs(full_text: str):
    nlp = _nlp_instance()
    nlp.max_length = max(nlp.max_length, len(full_text) + 1)
    doc = nlp(full_text)
    sumario_text, body_text, meta = split_sumario_and_body(doc, None)
    doc_sumario = nlp(sumario_text)
    doc_body = nlp(body_text)
    return doc, doc_sumario, doc_body, sumario_text, body_text, meta

def _extract_relations_and_payload(doc_sumario, serie_iii: bool):
    if serie_iii:
        rex = RelationExtractorSerieIII(debug=True)
        rels = rex.extract(doc_sumario)
        payload = export_serieIII_items_minimal_json(rels)
    else:
        rex = RelationExtractor(debug=True)
        rels = rex.extract(doc_sumario)
        payload = export_relations_items_minimal_json(rels, path=None)
    return rels, payload

def _split_body(doc_body, payload, serie_iii: bool):
    nlp = _nlp_instance()
    if serie_iii:
        return divide_body_by_org_and_docs_serieIII(doc_body, payload, nlp=nlp)
    if divide_body_I_II is None:
        raise RuntimeError("Série I/II splitter not available in this build.")
    return divide_body_I_II(
        doc_body,
        payload,
        write_org_files=False,
        write_doc_files=False,
    )

def _serialize_results(results) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for orgres in results:
        item: Dict[str, Any] = {
            "org": getattr(orgres, "org", None),
            "status": getattr(orgres, "status", None),
            "docs": [],
        }
        for ds in getattr(orgres, "docs", []) or []:
            d: Dict[str, Any] = {
                "doc_name": getattr(ds, "doc_name", None),
                "text": getattr(ds, "text", None),
            }
            ents = getattr(ds, "ents", None)
            if ents:
                d["ents"] = [
                    {"label": e[0], "text": e[1], "start": e[2], "end": e[3]}
                    for e in ents
                ]
            subs = getattr(ds, "subs", None)
            if subs:
                d["subs"] = [
                    {
                        "title": getattr(sub, "title", None),
                        "headers": list(getattr(sub, "headers", []) or []),
                        "body": getattr(sub, "body", None),
                    }
                    for sub in subs
                ]
            item["docs"].append(d)
        out.append(item)
    return out

def process_pdf(path: Path) -> Dict[str, Any]:
    """
    Path-only pipeline. No file writes. Returns JSON-serializable dict.
    """
    p = Path(path)
    if not p.exists() or not p.is_file():
        raise RuntimeError(f"File not found: {p}")
    # optional: basic guard
    if p.suffix.lower() != ".pdf":
        raise RuntimeError("Input must be a .pdf file.")

    serie_iii = _is_serie_iii_from_name(p.name)

    try:
        text = extract_pdf_to_markdown(p)
        _doc, doc_sumario, doc_body, _sum, _body, _meta = _build_docs(text)
        _rels, payload = _extract_relations_and_payload(doc_sumario, serie_iii)
        results, summary = _split_body(doc_body, payload, serie_iii)
        return {
            "serie_iii": bool(serie_iii),
            "summary": summary,
            "payload": payload,
            "results": _serialize_results(results),
        }
    except Exception as e:
        raise RuntimeError(f"Processing failed: {e}") from e
