from __future__ import annotations
import os
import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

# Project root = folder containing your main.py and/or appsettings.json
PROJECT_ROOT = Path(__file__).resolve().parents[2]  # .../pdf-markup/ -> .../<project root>

@dataclass(frozen=True)
class Settings:
    input_dir: Path
    output_dir: Path
    crop_top: float

def _read_json(path: Path) -> dict[str, Any]:
    try:
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        pass
    return {}

def _deep_get(d: dict[str, Any], path: list[str], default: Any = None) -> Any:
    cur = d
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur

@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """
    Resolution order (highest precedence first):
      1) Environment variables:
         - PDF_MARKUP_INPUT
         - PDF_MARKUP_OUTPUT
         - PDF_MARKUP_CROP   (float, e.g. 0.10)
      2) appsettings.Development.json (if present)
      3) appsettings.json
      4) Built-in defaults: input/, output/, crop_top=0.10
    """
    # Built-in defaults
    defaults = {
        "PdfMarkup": {
            "InputDir": "input",
            "OutputDir": "output",
            "CropTop": 0.10,
        }
    }

    # Load files
    base = _read_json(PROJECT_ROOT / "appsettings.json")
    dev  = _read_json(PROJECT_ROOT / "appsettings.Development.json")

    # Merge precedence: dev overrides base, base overrides defaults
    def merged(path, default):
        # path like ["PdfMarkup", "InputDir"]
        return (
            _deep_get(dev, path, None)
            or _deep_get(base, path, None)
            or _deep_get(defaults, path, default)
        )

    # Start with file/default values
    input_dir  = Path(merged(["PdfMarkup", "InputDir"], "input"))
    output_dir = Path(merged(["PdfMarkup", "OutputDir"], "output"))
    crop_top   = float(merged(["PdfMarkup", "CropTop"], 0.10))

    # Environment overrides
    env_input = os.getenv("PDF_MARKUP_INPUT")
    env_output = os.getenv("PDF_MARKUP_OUTPUT")
    env_crop = os.getenv("PDF_MARKUP_CROP")

    if env_input:
        input_dir = Path(env_input)
    if env_output:
        output_dir = Path(env_output)
    if env_crop:
        try:
            crop_top = float(env_crop)
        except ValueError:
            pass  # ignore bad env value; keep previous crop_top

    # Normalize to absolute paths relative to project root if given as relative
    if not input_dir.is_absolute():
        input_dir = (PROJECT_ROOT / input_dir).resolve()
    if not output_dir.is_absolute():
        output_dir = (PROJECT_ROOT / output_dir).resolve()

    return Settings(input_dir=input_dir, output_dir=output_dir, crop_top=crop_top)
