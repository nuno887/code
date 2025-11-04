from __future__ import annotations
import os
import datetime
from typing import Iterable, Optional

# Toggle this to disable all writes at once
DEBUG_DUMPS = True

def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def _ts() -> str:
    # Short timestamp for unique filenames
    return datetime.datetime.now().strftime("%Y%m%d-%H%M%S-%f")

def _write(path: str, text: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)

def dump_seg_bundle(
    *,
    seg_text: str,
    doc,  # spaCy Doc (or compatible) already computed
    out_dir: str = "debug_out",
    tag: str = "seg",
    allowed_titles: Optional[Iterable[str]] = None,
) -> None:
    """
    Writes four files into out_dir:
      - <tag>.<ts>.raw.txt          : exact seg_text
      - <tag>.<ts>.visible.txt      : with \n and \t made visible
      - <tag>.<ts>.annotated.txt    : inline markup for entities
      - <tag>.<ts>.headers.txt      : only DOC_NAME_LABELs (with norm + allowed flag)

    Safe to call frequently; filenames include a timestamp.
    """
    if not DEBUG_DUMPS:
        return

    _ensure_dir(out_dir)
    stamp = _ts()
    base = os.path.join(out_dir, f"{tag}.{stamp}")

    # 1) Raw
    _write(base + ".raw.txt", seg_text)

    # 2) Visible whitespace
    visible = seg_text.replace("\n", "\\n\n").replace("\t", "\\t")
    _write(base + ".visible.txt", visible)

    # 3) Annotated entities
    #    Wrap each entity inline: [<LABEL>|...|]</LABEL>
    annotated_parts = []
    cursor = 0
    for ent in doc.ents:
        s, e = ent.start_char, ent.end_char
        if s > cursor:
            annotated_parts.append(seg_text[cursor:s])
        label = getattr(ent, "label_", "")
        annotated_parts.append(f"[<{label}>|{seg_text[s:e]}|]</{label}>")
        cursor = e
    if cursor < len(seg_text):
        annotated_parts.append(seg_text[cursor:])
    _write(base + ".annotated.txt", "".join(annotated_parts))

    # 4) Headers list (just DOC_NAME_LABELs)
    #    Show index, span, raw preview, normalized, and allowed? (if provided)
    def _normalize_title(s: str) -> str:
        # Local lightweight normalizer to avoid importing internals:
        # strip markdown bold, trim, collapse spaces.
        t = s.replace("**", "").strip()
        t = " ".join(t.split())
        return t

    allowed_set = set(a for a in (allowed_titles or []) if a)
    lines = []
    headers = [ent for ent in doc.ents if getattr(ent, "label_", "") == "DOC_NAME_LABEL"]
    for i, ent in enumerate(headers):
        norm = _normalize_title(ent.text)
        allowed = (norm in allowed_set) if allowed_set else None
        preview = ent.text[:120].replace("\n", "\\n")
        if allowed is None:
            lines.append(
                f"i={i} span=({ent.start_char},{ent.end_char}) raw='{preview}' norm='{norm}'"
            )
        else:
            lines.append(
                f"i={i} span=({ent.start_char},{ent.end_char}) raw='{preview}' norm='{norm}' allowed={allowed}"
            )

    lines.append(f"\nTOTAL_HEADERS={len(headers)}")
    if allowed_set:
        matched = sum(1 for ent in headers if _normalize_title(ent.text) in allowed_set)
        lines.append(f"MATCHED={matched} of {len(headers)}")

    _write(base + ".headers.txt", "\n".join(lines))
