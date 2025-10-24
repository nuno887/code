# main.py
from __future__ import annotations

from pathlib import Path
import argparse
import html as html_lib

from spacy import displacy

from Entities import setup_entities, OPTIONS  # keep your OPTIONS for displacy
from Split_TEXT import split_sumario_and_body
from relations_extractor import (
    RelationExtractor,
    export_relations_items_minimal_json,
)
from relations_extractor_serieIII import (
    RelationExtractorSerieIII,
    export_serieIII_items_minimal_json,
)

from pdf_markup import extract_pdf_to_markdown

# NEW: use our single factory + splitter
from serie3_splitter.nlp import get_nlp
from serie3_splitter import divide_body_by_org_and_docs_serieIII



# -----------------------
# Config knobs
# -----------------------
DEFAULT_INPUT_DIR = Path("input_pdfs")
CROP_TOP_RATIO = 0.10
SKIP_LAST_PAGE = True


def is_serie_iii(filename: str) -> bool:
    """Decide if this is Serie III by filename heuristic."""
    return "iiiserie" in filename.lower()


def load_text_from_pdf(pdf_path: Path) -> str:
    return extract_pdf_to_markdown(
        pdf_path
    )


def build_docs(nlp, full_text: str):
    """
    Returns (doc, doc_sumario, doc_body, sumario_text, body_text, meta)
    """
    doc = nlp(full_text)
    sumario_text, body_text, meta = split_sumario_and_body(doc, None)
    doc_sumario = nlp(sumario_text)
    doc_body = nlp(body_text)
    return doc, doc_sumario, doc_body, sumario_text, body_text, meta


def extract_relations_and_payload(doc_sumario, serie_iii: bool):
    if serie_iii:
        rex = RelationExtractorSerieIII(debug=True)
        rels = rex.extract(doc_sumario)
        payload = export_serieIII_items_minimal_json(rels)
    else:
        rex = RelationExtractor(debug=True)
        rels = rex.extract(doc_sumario)
        payload = export_relations_items_minimal_json(rels, path=None)
    return rels, payload


def split_body(doc_body, payload, serie_iii: bool, nlp):
    if serie_iii:
        # IMPORTANT: pass the same pipeline used to build doc_body
        results, summary = divide_body_by_org_and_docs_serieIII(
            doc_body,
            payload,
            nlp=nlp,
        )
    else:
        # Keep your Serie I/II path as-is if you still use it elsewhere
        from body_extraction import divide_body_by_org_and_docs

        results, summary = divide_body_by_org_and_docs(
            doc_body,
            payload,
            write_org_files=False,
            write_doc_files=False,
        )
    return results, summary


def render_entities_html(doc_body, out_path: Path):
    html = displacy.render(doc_body, style="ent", jupyter=False, options=OPTIONS)
    full_html = f"""<!doctype html>
<html lang="pt">
<head>
  <meta charset="utf-8">
  <title>Entidades</title>
  <meta name="viewport" content="width=device-width, initial-scale=1">
</head>
<body>
  <span class="tex2jax_ignore">{html}</span>
</body>
</html>"""
    out_path.write_text(full_html, encoding="utf-8")


def render_results_html(results, summary, out_path: Path):
    lines = [
        "<!doctype html>",
        "<meta charset='utf-8'>",
        "<title>Results</title>",
        "<body style='font-family:system-ui,Segoe UI,Roboto,Arial,sans-serif;margin:24px'>",
        "<h1>Resultados</h1>",
        f"<p><strong>Resumo:</strong> {html_lib.escape(str(summary))}</p>",
    ]

    for i, orgres in enumerate(results, start=1):
        lines.append(
            f"<h2>{i:02d}. {html_lib.escape(orgres.org)} "
            f"<small>[{orgres.status}]</small></h2>"
        )
        if not orgres.docs:
            lines.append("<p><em>Sem documentos neste bloco.</em></p>")
            continue

        for j, ds in enumerate(orgres.docs, start=1):
            lines.append(f"<h3>{i:02d}.{j:02d} — {html_lib.escape(ds.doc_name)}</h3>")

            # Full segment
            lines.append("<details open>")
            lines.append("<summary>Segmento completo</summary>")
            lines.append(f"<pre style='white-space:pre-wrap'>{html_lib.escape(ds.text)}</pre>")
            lines.append("</details>")

            # Entities (second pass)
            if getattr(ds, "ents", None):
                lines.append("<details>")
                lines.append("<summary>Entidades no segmento (segunda passagem)</summary>")
                lines.append("<ul>")
                for (lbl, txt, st, en) in ds.ents:
                    lines.append(
                        f"<li><code>{html_lib.escape(lbl)}</code> — "
                        f"{html_lib.escape(txt)} "
                        f"<small>[{st}:{en}]</small></li>"
                    )
                lines.append("</ul>")
                lines.append("</details>")

            # Sub-slices
            if getattr(ds, "subs", None):
                lines.append("<h4>Subdivisões</h4>")
                for k, sub in enumerate(ds.subs, start=1):
                    lines.append(f"<h5>{i:02d}.{j:02d}.{k:02d} — {html_lib.escape(sub.title)}</h5>")
                    if getattr(sub, "headers", None):
                        lines.append("<p><strong>Headers colapsados:</strong></p><ul>")
                        for h in sub.headers:
                            lines.append(f"<li>{html_lib.escape(h)}</li>")
                        lines.append("</ul>")
                    lines.append(f"<pre style='white-space:pre-wrap'>{html_lib.escape(sub.body)}</pre>")
            else:
                lines.append("<p><em>Sem subdivisões para este segmento.</em></p>")

    lines.append("</body>")
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Process a Serie I/II/III/IV PDF.")
    parser.add_argument(
        "pdf",
        nargs="?",
        default="IIISerie-03-2012-02-02.pdf",
        help="PDF filename inside input_pdfs/ (default: %(default)s)",
    )
    args = parser.parse_args()

    pdf_path = (DEFAULT_INPUT_DIR / args.pdf).resolve()
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    serie_iii = is_serie_iii(pdf_path.name)

    # 1) Load model (single factory) + custom entities
    nlp = get_nlp(disable_ner=True)
    # setup_entities(nlp)  # already called by get_nlp()

    # 2) Extract text and build docs
    text = load_text_from_pdf(pdf_path)

    nlp.max_length = max(nlp.max_length, len(text)+1)


    doc, doc_sumario, doc_body, sumario_text, body_text, _meta = build_docs(nlp, text)

    # 3) Extract relations + payload
    rels, payload = extract_relations_and_payload(doc_sumario, serie_iii)

    # 4) Split body (Serie III uses our new splitter)
    results, summary = split_body(doc_body, payload, serie_iii, nlp)

    # 5) Dump HTML artifacts
    render_entities_html(doc_body, Path("ents.html"))
    render_results_html(results, summary, Path("results.html"))

    print("Saved entity visualization to ents.html")
    print("Wrote results.html")


if __name__ == "__main__":
    main()
