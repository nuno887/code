from __future__ import annotations

from pathlib import Path
import argparse
import html as html_lib

from spacy import displacy

from split_text import split_sumario_and_body

from relation_extractor import RelationExtractor, RelationExtractorSerieIII, export_relations_items_minimal_json, export_serieIII_items_minimal_json

from pdf_markup import extract_pdf_to_markdown

from spacy_modulo import get_nlp, setup_entities, OPTIONS

from body_extraction import divide_body_by_org_and_docs, divide_body_by_org_and_docs_serieIII


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
        

        results, results_reduzed,summary = divide_body_by_org_and_docs(
            doc_body,
            payload,
            write_org_files=False,
            write_doc_files=False,
        )
    return results,results_reduzed,summary


def main():
    parser = argparse.ArgumentParser(description="Process a Serie I/II/III/IV PDF.")
    parser.add_argument(
        "pdf",
        nargs="?",
        default="IISerie-099-2005-05-23Supl.pdf",
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
    results, results_reduzed,summary = split_body(doc_body, payload, serie_iii)

    # print(f"results:", results)
    with open("results.txt", "w", encoding="utf-8") as f:
        f.write(f"results: {results_reduzed}\n")






  
   

 




if __name__ == "__main__":
    main()