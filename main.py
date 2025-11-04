# main.py
from __future__ import annotations

from pathlib import Path
import argparse
import html as html_lib

from spacy import displacy

from split_text import split_sumario_and_body

from relation_extractor import RelationExtractor, RelationExtractorSerieIII, export_relations_items_minimal_json, export_serieIII_items_minimal_json

from pdf_markup import extract_pdf_to_markdown

from spacy_modulo import get_nlp, setup_entities, OPTIONS

# from serie3_splitter import divide_body_by_org_and_docs_serieIII

from body_extraction import divide_body_by_org_and_docs, divide_body_by_org_and_docs_serieIII

from resultNormalizer import normalize_results, build_slim_payload, build_output_paths, save_json

from collections import defaultdict




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


def split_body(doc_body, payload, serie_iii: bool):
    if serie_iii:
        # IMPORTANT: pass the same pipeline used to build doc_body
        results, summary = divide_body_by_org_and_docs_serieIII(
            doc_body,
            payload,

        )
    else:
        # Keep your Serie I/II path as-is if you still use it elsewhere
        

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


    from collections import defaultdict
import html as html_lib

def render_slim_html(slim: dict, out_path: Path) -> None:
    """
    Render a clean HTML view for the slim payload:
    {
      "file": str,
      "docs": [
        { "org": str, "doc_name": str, "section_title"?: str, "headers"?: [str], "text"?: str }
      ]
    }
    """
    file_name = html_lib.escape(str(slim.get("file", "")))
    docs = slim.get("docs", [])

    # Group by org -> doc_name
    grouped = defaultdict(lambda: defaultdict(list))
    for d in docs:
        org = d.get("org", "") or ""
        doc_name = d.get("doc_name", "") or ""
        grouped[org][doc_name].append(d)

    lines = []
    lines.append("<!doctype html>")
    lines.append("<meta charset='utf-8'>")
    lines.append(f"<title>Slim Results — {file_name}</title>")
    lines.append("""
<style>
  body { font-family: system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif; margin: 24px; line-height: 1.45; }
  h1 { margin-top: 0; }
  .meta { color: #666; margin-bottom: 16px; }
  .org { margin: 24px 0 8px; padding-bottom: 4px; border-bottom: 1px solid #e5e5e5; }
  details { margin: 8px 0 16px; }
  summary { cursor: pointer; font-weight: 600; }
  .doc { margin: 12px 0; padding: 12px; border: 1px solid #eee; border-radius: 8px; background: #fafafa; }
  .badge { display: inline-block; padding: 2px 8px; border: 1px solid #ddd; border-radius: 999px; font-size: 12px; color: #444; background: #fff; }
  .headers { margin: 6px 0; padding-left: 18px; }
  pre { white-space: pre-wrap; background: #fff; padding: 10px; border: 1px solid #eee; border-radius: 6px; }
  .search { margin: 12px 0 24px; }
  .muted { color: #777; }
</style>
""")
    lines.append("<body>")
    lines.append(f"<h1>Slim Results</h1>")
    lines.append(f"<div class='meta'><strong>File:</strong> {file_name} &middot; "
                 f"<strong>Total entries:</strong> {len(docs)}</div>")

    # Simple client-side filter (org/doc/section/text)
    lines.append("""
<div class="search">
  <input id="q" type="search" placeholder="Filter (org, doc, section, text)..." style="width: min(720px, 100%); padding: 8px; font-size: 14px;">
</div>
<script>
  const q = document.getElementById('q');
  q?.addEventListener('input', () => {
    const needle = q.value.toLowerCase();
    document.querySelectorAll('[data-entry]').forEach(el => {
      const hay = el.getAttribute('data-entry') || '';
      el.style.display = hay.includes(needle) ? '' : 'none';
    });
  });
</script>
""")

    if not docs:
        lines.append("<p class='muted'><em>No docs in slim payload.</em></p>")

    org_idx = 0
    for org, by_doc in grouped.items():
        org_idx += 1
        org_esc = html_lib.escape(org or "(sem entidade)")
        lines.append(f"<h2 class='org'>{org_idx:02d}. {org_esc}</h2>")

        doc_idx = 0
        for doc_name, entries in by_doc.items():
            doc_idx += 1
            doc_name_esc = html_lib.escape(doc_name or "(sem nome de documento)")
            lines.append(f"<h3>{org_idx:02d}.{doc_idx:02d} — {doc_name_esc} "
                         f"<span class='badge'>{len(entries)} entr{ 'y' if len(entries)==1 else 'ies' }</span></h3>")

            for k, e in enumerate(entries, start=1):
                section_title = e.get("section_title", "") or ""
                headers = e.get("headers") or []
                text = e.get("text", "") or ""

                # Build a searchable string for the filter
                haystack = " ".join([
                    org or "", doc_name or "", section_title or "",
                    " ".join(headers) if isinstance(headers, list) else "",
                    text or ""
                ]).lower()

                section_esc = html_lib.escape(section_title)
                text_esc = html_lib.escape(text)

                lines.append(f"<div class='doc' data-entry='{html_lib.escape(haystack)}'>")
                # Collapsible block per entry
                summary_title = section_esc if section_title else "(sem título de secção)"
                lines.append(f"<details>")
                lines.append(f"<summary>{org_idx:02d}.{doc_idx:02d}.{k:02d} — {summary_title}</summary>")
                if headers:
                    lines.append("<div class='headers'><strong>Headers:</strong><ul>")
                    for h in headers:
                        lines.append(f"<li>{html_lib.escape(str(h))}</li>")
                    lines.append("</ul></div>")
                lines.append(f"<pre>{text_esc}</pre>")
                lines.append("</details>")
                lines.append("</div>")

    lines.append("</body>")

    out_path.write_text("\n".join(lines), encoding="utf-8")





def main():
    parser = argparse.ArgumentParser(description="Process a Serie I/II/III/IV PDF.")
    parser.add_argument(
        "pdf",
        nargs="?",
        default="IISerie-249-2005-12-30Supl051.pdf",
        help="PDF filename inside input_pdfs/ (default: %(default)s)",
    )
    args = parser.parse_args()

    pdf_path = (DEFAULT_INPUT_DIR / args.pdf).resolve()
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    serie_iii = is_serie_iii(pdf_path.name)

    # 1) Load model (single factory) + custom entities
    nlp = get_nlp(disable_ner=True, SerieIII= serie_iii)
    # setup_entities(nlp)  # already called by get_nlp()

    # 2) Extract text and build docs
    text = load_text_from_pdf(pdf_path)

    nlp.max_length = max(nlp.max_length, len(text)+1)


    doc, doc_sumario, doc_body, sumario_text, body_text, _meta = build_docs(nlp, text)

    # 3) Extract relations + payload
    rels, payload = extract_relations_and_payload(doc_sumario, serie_iii)

   
 
    # 4) Split body (Serie III uses our new splitter)
    results, summary = split_body(doc_body, payload, serie_iii)

    # ---- Normalize & save outputs (unified + slim) ----
    mode = "serie_iii" if serie_iii else "serie_std"
    input_meta = {"filename": pdf_path.name, "text_length": len(text)}

    unified = normalize_results(
        mode=mode,
        raw_results=results,
        raw_summary=summary,
        input_meta=input_meta,
        # For Serie I/II/IV you can hint the payload type:
        # serie_std_mode="flat" or "hierarchical"
    )

    slim = build_slim_payload(
        filename=pdf_path.name,
        unified=unified,
        include_text=True,   # Full text; no truncation
    )


    paths = build_output_paths(pdf_path.name, out_dir="output_json")
    save_json(slim, paths["slim"])

    print(f"Saved slim JSON to {paths['slim']}")
    slim = build_slim_payload(
    filename=pdf_path.name,
    unified=unified,
    include_text=True,
)
    render_slim_html(slim, Path("slim.html"))
    print("Wrote slim.html")









    print("============== MAIN =====================")

    print(f"payload", payload)
    print("===================================")

    # 5) Dump HTML artifacts
    render_entities_html(doc_sumario, Path("ents.html"))
    render_results_html(results, summary, Path("results.html"))

    print("Saved entity visualization to ents.html")
    print("Wrote results.html")


    # print(f"sumario_text",sumario_text)
    # print(f"_meta", _meta)
    print (f"results",results)




if __name__ == "__main__":
    main()

