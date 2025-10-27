# api.py
from __future__ import annotations
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from processing import process_pdf

app = FastAPI(title="DR Processor API (Path Only)", version="1.0.0")


class ProcessRequest(BaseModel):
    path: str

@app.post("/process")
def process(req: ProcessRequest):
    try:
        result = process_pdf(Path(req.path))
        return JSONResponse(content=result)
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {e}")



# uvicorn api:app --host 0.0.0.0 --port 8000 --reload