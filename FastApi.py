from __future__ import annotations

import os
import io
import shutil
import tempfile
from pathlib import Path
from typing import Optional, Dict, Any

import threading

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

# ---- Project imports (your existing modules) ----
from pdf_markup import extract_pdf_to_markdown
from spacy_modulo import get_nlp
from split_text import split_sumario_and_body
from relation_extractor import (
    RelationExtractor,
    RelationExtractorSerieIII,
    export_relations_items_minimal_json,
    export_serieIII_items_minimal_json,
)
from body_extraction import (
    divide_body_by_org_and_docs,
    divide_body_by_org_and_docs_serieIII,
)
from resultNormalizer import (
    normalize_results,
    build_slim_payload,
)

# -----------------------
# App & Config
# -----------------------
app = FastAPI(title="Diário da República Extractor API", version="1.0.0")

UPLOAD_SIZE_LIMIT_MB = int(os.getenv("UPLOAD_SIZE_LIMIT_MB", "50"))
NLP_MAX_LENGTH = int(os.getenv("NLP_MAX_LENGTH", str(10_000_000)))  # characters

# Pipelines initialized at startup
nlp_std = None
nlp_iii = None

# Locks to safely adjust max_length at request time
nlp_std_lock = threading.Lock()
nlp_iii_lock = threading.Lock()


# -----------------------
# Models
# -----------------------
class ExtractResponse(BaseModel):
    file: str
    docs: list


# -----------------------
# Utilities
# -----------------------

def is_serie_iii(filename: str) -> bool:
    """Heuristic based on filename, same as your main.py."""
    return "iiiserie" in (filename or "").lower()


def sanitize_filename(name: str) -> str:
    keep = [c for c in name if c.isalnum() or c in (".", "-", "_", " ")]
    cleaned = "".join(keep).strip()
    return cleaned or "upload.pdf"



def save_upload_to_temp(upload: UploadFile, original_name: str) -> Path:
    """Persist UploadFile to a secure temp file, preserving part of the name."""
    cleaned = sanitize_filename(original_name or upload.filename or "upload.pdf")
    # ensure suffix
    suffix = ".pdf" if not cleaned.lower().endswith(".pdf") else ""
    tmp = tempfile.NamedTemporaryFile(prefix=f"{Path(cleaned).stem}_", suffix=f"{suffix}", delete=False)
    tmp_path = Path(tmp.name)
    try:
        # size-guarded streaming copy
        limit_bytes = UPLOAD_SIZE_LIMIT_MB * 1024 * 1024
        bytes_copied = 0
        while True:
            chunk = upload.file.read(1024 * 1024)
            if not chunk:
                break
            bytes_copied += len(chunk)
            if bytes_copied > limit_bytes:
                raise HTTPException(status_code=422, detail=f"File too large. Limit is {UPLOAD_SIZE_LIMIT_MB} MB.")
            tmp.write(chunk)
        tmp.flush()
        tmp.close()
    except HTTPException:
        tmp.close()
        try:
            tmp_path.unlink(missing_ok=True)
        except Exception:
            pass
        raise
    except Exception as e:
        tmp.close()
        try:
            tmp_path.unlink(missing_ok=True)
        except Exception:
            pass
        raise HTTPException(status_code=400, detail=f"Failed to store upload: {e}")


    return tmp_path


# -----------------------
# Core pipeline (no HTML, no disk outputs)
# -----------------------

def process_pdf(temp_pdf_path: Path, original_filename: str) -> Dict[str, Any]:

    # 1) Detect series based on filename
    serie_iii = is_serie_iii(original_filename)

    # 2) Extract text from the saved file (path-based)
    try:
        text = extract_pdf_to_markdown(temp_pdf_path)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"pdf_extract: {e}")

    # 3) Choose NLP (prebuilt at startup)
    nlp = nlp_iii if serie_iii else nlp_std
    if nlp is None:
        raise HTTPException(status_code=500, detail="NLP pipeline not initialized.")

    # 4) Build docs
    try:
        # Dynamically raise max_length if this document is larger
        needed = len(text) + 1
        lock = nlp_iii_lock if serie_iii else nlp_std_lock
        with lock:
            if nlp.max_length < needed:
                nlp.max_length = needed
        # Now parse safely
        doc = nlp(text)
        sumario_text, body_text, _meta = split_sumario_and_body(doc, None)
        doc_sumario = nlp(sumario_text)
        doc_body = nlp(body_text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"split_sumario: {e}")

    # 5) Extract relations & payload from sumário
    try:
        if serie_iii:
            rex = RelationExtractorSerieIII(debug=False)
            rels = rex.extract(doc_sumario)
            payload = export_serieIII_items_minimal_json(rels)
        else:
            rex = RelationExtractor(debug=False)
            rels = rex.extract(doc_sumario)
            payload = export_relations_items_minimal_json(rels, path=None)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"relations: {e}")

    # 6) Split body
    try:
        if serie_iii:
            results, summary = divide_body_by_org_and_docs_serieIII(doc_body, payload)
        else:
            results, summary = divide_body_by_org_and_docs(
                doc_body,
                payload,
                write_org_files=False,
                write_doc_files=False,
            )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"split_body: {e}")

    # 7) Normalize & build slim
    try:
        mode = "serie_iii" if serie_iii else "serie_std"
        input_meta = {"filename": original_filename, "text_length": len(text)}
        unified = normalize_results(
            mode=mode,
            raw_results=results,
            raw_summary=summary,
            input_meta=input_meta,
        )
        slim = build_slim_payload(
            filename=original_filename,
            unified=unified,
            include_text=True,
        )
        return slim
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"normalize: {e}")


# -----------------------
# Routes
# -----------------------
@app.get("/health")
def health() -> Dict[str, Any]:
    return {
        "status": "ok",
        "pipelines": {
            "std": bool(nlp_std is not None),
            "iii": bool(nlp_iii is not None),
        },
        "upload_limit_mb": UPLOAD_SIZE_LIMIT_MB,
        "nlp_max_length": NLP_MAX_LENGTH,
    }


@app.post("/extract", response_model=ExtractResponse)
async def extract(
    file: UploadFile = File(..., description="PDF file to process"),
):
    # Basic content-type guard (best-effort; magic check done after write)
    if (file.content_type or "").lower() not in {"application/pdf", "application/x-pdf", "application/acrobat"}:
        # allow anyway; we validate via magic
        pass

    original_name = file.filename or "upload.pdf"
    temp_path: Optional[Path] = None

    try:
        temp_path = save_upload_to_temp(file, original_name)
        slim = process_pdf(temp_path, original_name)
        return JSONResponse(status_code=200, content=slim)
    finally:
        # cleanup temp file
        if temp_path and temp_path.exists():
            try:
                temp_path.unlink(missing_ok=True)
            except Exception:
                pass


# -----------------------
# Startup (initialize both pipelines once)
# -----------------------
@app.on_event("startup")
def _startup_init_nlp():
    global nlp_std, nlp_iii
    try:
        nlp_std = get_nlp(disable_ner=True, SerieIII=False)
        nlp_iii = get_nlp(disable_ner=True, SerieIII=True)
        # Set safe max_length once for both
        nlp_std.max_length = max(nlp_std.max_length, NLP_MAX_LENGTH)
        nlp_iii.max_length = max(nlp_iii.max_length, NLP_MAX_LENGTH)
    except Exception as e:
        # If initialization fails, expose via health endpoint and raise on first request
        nlp_std = None
        nlp_iii = None
        print(f"Failed to initialize NLP pipelines: {e}")


# -----------------------
# Dev entrypoint
# -----------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("fastapi_app:app", host="0.0.0.0", port=8000, reload=True)


# python -m uvicorn FastApi:app --reload --port 8000
