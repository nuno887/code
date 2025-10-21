from pathlib import Path
import fitz
import pymupdf4llm as pdfllm
from .heuristics import (
    crop_top,
    merge_bold_runs_table_safe,          # existing: merge any bold-only (IIISerie)
    merge_bold_runs_table_safe_allcaps,   # new: merge only ALL-CAPS bold-only (all PDFs)
)

def page_to_markdown(
    pdf_path: Path,
    page_index: int,
    crop_top_ratio: float = 0.10,
    table_strategy: str | None = "lines_strict",
) -> str:
    doc = fitz.open(str(pdf_path))
    page = doc[page_index]
    if crop_top_ratio:
        crop_top(page, crop_top_ratio)
    md = pdfllm.to_markdown(doc, pages=[page_index], table_strategy=table_strategy or "lines_strict")
    return merge_bold_runs_table_safe(md)

def extract_pdf_to_markdown(
    pdf_path: Path,
    crop_top_ratio: float = 0.10,
    skip_last_page: bool = True,
) -> str:
    doc = fitz.open(str(pdf_path))
    parts: list[str] = []
    total_pages = len(doc) - 1 if skip_last_page else len(doc)

    for i in range(total_pages):
        page = doc[i]
        if crop_top_ratio:
            crop_top(page, crop_top_ratio)
        md = pdfllm.to_markdown(
            doc,
            pages=[i],
            table_strategy="lines_strict",
        )
        parts.append(md)

    full_md = "\n\n---\n\n".join(parts)

    # 1) Always: merge ALL-CAPS bold-only runs (handles CONSERVATÓRIA ... DO FUNCHAL)
    full_md = merge_bold_runs_table_safe_allcaps(full_md)

    # 2) Additionally for IIISerie: merge ANY bold-only runs
    if "IIISerie" in pdf_path.name:
        full_md = merge_bold_runs_table_safe(full_md)

    return full_md
