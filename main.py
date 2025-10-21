from pathlib import Path
import spacy
from spacy import displacy
from Entities import setup_entities, OPTIONS
from Split_TEXT import split_sumario_and_body

from relations_extractor import RelationExtractor, export_relations_items_minimal_json
from relations_extractor_serieIII import RelationExtractorSerieIII, export_serieIII_items_minimal_json




# 1) Load model
nlp = spacy.load("pt_core_news_lg", exclude = "ner")
setup_entities(nlp)

# 2) Paths (edit FILE_NAME to switch files)
FILES_DIR = Path("files")
FILE_NAME = "IIIserie-23-2014-12-02.md"
#FILE_NAME = "IISerie-247-2003-12-30Supl9.md"

is_serieIII = "iiiserie" in FILE_NAME.lower()

# 3) Read, process, render
text = (FILES_DIR / FILE_NAME).read_text(encoding="utf-8")
doc = nlp(text)



sumario_text, body_text, _meta = split_sumario_and_body(doc, None)



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





html = displacy.render(doc_sumario, style="ent", jupyter=False, options= OPTIONS)


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
