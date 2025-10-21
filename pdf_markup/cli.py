from pdf_markup import get_settings, extract_pdf_to_markdown

s = get_settings()  # shared settings
# s.input_dir, s.output_dir, s.crop_top

# Example: ensure an .md exists for a matching PDF before your NLP pipeline:
from pathlib import Path

file_name = "IISerie-165-2025-09-12.md"
pdf_path = s.input_dir / file_name.replace(".md", ".pdf")
md_path = s.output_dir / file_name

if pdf_path.exists() and not md_path.exists():
    s.output_dir.mkdir(parents=True, exist_ok=True)
    md = extract_pdf_to_markdown(pdf_path, crop_top_ratio=s.crop_top, skip_last_page=True)
    md_path.write_text(md, encoding="utf-8")
    print(f"✓ {pdf_path.name} -> {md_path.name}")
