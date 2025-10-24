from pathlib import Path
import spacy
from spacy import displacy
from Entities import setup_entities, OPTIONS

from Split_TEXT import split_sumario_and_body

from relations_extractor import RelationExtractor, export_relations_items_minimal_json
from relations_extractor_serieIII import RelationExtractorSerieIII, export_serieIII_items_minimal_json

from body_extraction import divide_body_by_org_and_docs
from body_extractionIII import divide_body_by_org_and_docs_serieIII
import html as html_lib


from pdf_markup import extract_pdf_to_markdown

PDF_NAME = "IIIserie-03-2006-02-01.pdf"
pdf_path = Path("input_pdfs")/ PDF_NAME

is_serieIII = "iiiserie" in PDF_NAME.lower()

text = extract_pdf_to_markdown(pdf_path, crop_top_ratio=0.10, skip_last_page=True)

# 1) Load model
nlp = spacy.load("pt_core_news_lg", exclude = "ner")
setup_entities(nlp)


doc = nlp(text)



sumario_text, body_text, _meta = split_sumario_and_body(doc, None)


doc_body = nlp(body_text)
doc_sumario = nlp(sumario_text)
if is_serieIII:
    rex = RelationExtractorSerieIII(debug=True)
else:
    rex = RelationExtractor(debug=True)

rels = rex.extract(doc_sumario)


if is_serieIII:
    payload = export_serieIII_items_minimal_json(rels)

    print(f"payload:", payload)

    results, summary = divide_body_by_org_and_docs_serieIII(doc_body, payload)

else:  
    payload = export_relations_items_minimal_json(rels, path = None)

    results, summary = divide_body_by_org_and_docs(
        doc_body,
        payload,
        write_org_files = False,
        write_doc_files = False
    )



#print(f"Payload:", payload)
#print(f"Summary", summary)
#print()
#print(f"Results:", results)
html = displacy.render(doc_body, style="ent", jupyter=False, options= OPTIONS)


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

Path("ents.html").write_text(full_html, encoding="utf-8")


print("Saved entity visualization to ents.html")

# NEW: minimal HTML dump of `results`
lines = [
    "<!doctype html>",
    "<meta charset='utf-8'>",
    "<title>Results</title>",
    "<body style='font-family:system-ui,Segoe UI,Roboto,Arial,sans-serif;margin:24px'>",
    "<h1>Resultados</h1>",
    f"<p><strong>Resumo:</strong> {html_lib.escape(str(summary))}</p>",
]
for i, orgres in enumerate(results, start=1):
    lines.append(f"<h2>{i:02d}. {html_lib.escape(orgres.org)} <small>[{orgres.status}]</small></h2>")
    if not orgres.docs:
        lines.append("<p><em>Sem documentos neste bloco.</em></p>")
        continue

    for j, ds in enumerate(orgres.docs, start=1):
        lines.append(f"<h3>{i:02d}.{j:02d} — {html_lib.escape(ds.doc_name)}</h3>")

        # Show the full segment as before
        lines.append("<details open>")
        lines.append("<summary>Segmento completo</summary>")
        lines.append(f"<pre style='white-space:pre-wrap'>{html_lib.escape(ds.text)}</pre>")
        lines.append("</details>")

        # If you want to inspect entities recovered in seg_text:
        if getattr(ds, 'ents', None):
            lines.append("<details>")
            lines.append("<summary>Entidades no segmento (segunda passagem)</summary>")
            lines.append("<ul>")
            for (lbl, txt, st, en) in ds.ents:
                lines.append(f"<li><code>{html_lib.escape(lbl)}</code> — {html_lib.escape(txt)} "
                             f"<small>[{st}:{en}]</small></li>")
            lines.append("</ul>")
            lines.append("</details>")

        # Render sub-slices (what we care about now)
        if getattr(ds, 'subs', None):
            lines.append("<h4>Subdivisões</h4>")
            for k, sub in enumerate(ds.subs, start=1):
                lines.append(f"<h5>{i:02d}.{j:02d}.{k:02d} — {html_lib.escape(sub.title)}</h5>")
                if getattr(sub, 'headers', None):
                    lines.append("<p><strong>Headers colapsados:</strong></p><ul>")
                    for h in sub.headers:
                        lines.append(f"<li>{html_lib.escape(h)}</li>")
                    lines.append("</ul>")
                lines.append(f"<pre style='white-space:pre-wrap'>{html_lib.escape(sub.body)}</pre>")
        else:
            lines.append("<p><em>Sem subdivisões para este segmento.</em></p>")

lines.append("</body>")
Path("results.html").write_text("\n".join(lines), encoding="utf-8")
print("Wrote results.html")