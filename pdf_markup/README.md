# pdf-markup

Batch-convert PDFs to Markdown using PyMuPDF + pymupdf4llm with smart heuristics:
- Top cropping per page
- Table-safe merging of bold-only lines (ALL-CAPS for all PDFs; any-bold for files with "IIISerie" in the name)

## Install (editable)

```bash
pip install -U pip
pip install -e .
