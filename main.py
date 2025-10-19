from pathlib import Path
import spacy
from spacy import displacy
from Entities import setup_entities, OPTIONS
from Split_TEXT import split_sumario_and_body




# 1) Load model
nlp = spacy.load("pt_core_news_lg", exclude = "ner")
setup_entities(nlp)

# 2) Paths (edit FILE_NAME to switch files)
FILES_DIR = Path("files")
FILE_NAME = "IIISerie-14-2014-07-18.md"

# 3) Read, process, render
text = (FILES_DIR / FILE_NAME).read_text(encoding="utf-8")
doc = nlp(text)

sumario_text, body_text, _meta = split_sumario_and_body(doc, None)


print("=== SUMARIO ===")
print(sumario_text)
print("\n=== BODY ===")
#print(body_text)




html = displacy.render(doc, style="ent", jupyter=False, options= OPTIONS)


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
