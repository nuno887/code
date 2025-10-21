from pathlib import Path
import spacy
from spacy import displacy
from Entities import setup_entities, OPTIONS
from Split_TEXT import split_sumario_and_body

from relations_extractor import RelationExtractor, export_relations_items_minimal_json
from relations_extractor_serieIII import RelationExtractorSerieIII, export_serieIII_items_minimal_json


from pdf_markup import extract_pdf_to_markdown

PDF_NAME = "IISerie-128-2005-07-06.pdf"
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
    export_serieIII_items_minimal_json(rels, "relations.items.minimal.json")
else:  
    export_relations_items_minimal_json(rels, "relations.items.minimal.json")





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
