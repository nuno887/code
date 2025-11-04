# call_api.py
import json
from pathlib import Path
import requests

API_URL = "http://localhost:8000/extract"     # FastAPI endpoint
FILE_PATH = "input_pdfs/IIISerie-004-2025-02-27.pdf"             # <-- change to your PDF
OUTPUT_PATH = "last_extract.txt"              # always overwrite this file

def main():
    pdf_path = Path(FILE_PATH)
    if not pdf_path.exists():
        raise SystemExit(f"File not found: {pdf_path}")

    with pdf_path.open("rb") as f:
        files = {"file": (pdf_path.name, f, "application/pdf")}
        try:
            resp = requests.post(API_URL, files=files, timeout=300)
        except requests.RequestException as e:
            raise SystemExit(f"Request failed: {e}")

    print(f"HTTP {resp.status_code}")

    # Try to pretty-print JSON; fall back to raw text
    try:
        content = json.dumps(resp.json(), indent=2, ensure_ascii=False)
    except ValueError:
        content = resp.text

    Path(OUTPUT_PATH).write_text(content, encoding="utf-8")
    print(f"Wrote response to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
