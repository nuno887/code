from pathlib import Path
import spacy
from spacy import displacy
from Entities import setup_entities, OPTIONS




# 1) Load model
nlp = spacy.load("pt_core_news_lg", exclude = "ner")
setup_entities(nlp)

# 2) Paths (edit FILE_NAME to switch files)
FILES_DIR = Path("files")
FILE_NAME = "IISerie-007-2025-01-10Supl2.md"

# 3) Read, process, render
text = (FILES_DIR / FILE_NAME).read_text(encoding="utf-8")
doc = nlp(text)

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
