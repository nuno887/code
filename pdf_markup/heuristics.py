import re
import fitz

_TABLE_ALIGN_RE = re.compile(r"[-:]{3,}")

def crop_top(page: fitz.Page, ratio: float) -> None:
    r = page.rect
    page.set_cropbox(fitz.Rect(r.x0, r.y0 + r.height * ratio, r.x1, r.y1))

def is_table_row(line: str) -> bool:
    l = line.strip()
    if not l or "|" not in l:
        return False
    c = l.count("|")
    starts = l.startswith("|")
    row_like = starts and (l.endswith("|") or c >= 3)
    align_like = starts and (_TABLE_ALIGN_RE.search(l) is not None)
    return row_like or align_like

def merge_bold_runs_table_safe(md: str) -> str:
    """(Your existing IIISerie merge) Merge ANY consecutive bold-only lines into one block."""
    out, buf, in_table = [], [], False

    def flush():
        nonlocal buf
        if buf:
            out.append("**" + "\n".join(buf) + "**")
            buf = []

    for line in md.splitlines():
        if is_table_row(line):
            flush()
            in_table = True
            out.append(line)
            continue
        if in_table and not is_table_row(line):
            in_table = False

        m = re.match(r'^\s*\*\*(.+)\*\*\s*$', line)
        if m:
            buf.append(m.group(1))
        else:
            flush()
            out.append(line)

    flush()
    return "\n".join(out)

def _fix_glued_bold_boundaries(line: str) -> str:
    """Insert missing spaces around bold markers when words touch '**'."""
    line = re.sub(r'([A-Za-zÀ-ÖØ-öø-ÿ0-9])\*\*', r'\1 **', line)
    line = re.sub(r'\*\*([A-Za-zÀ-ÖØ-öø-ÿ0-9])', r'** \1', line)
    return line

def consolidate_inline_bold_on_line(line: str) -> str:
    """
    Collapse all bold spans in a single line into one **bold** block and fix glued words.
    Example:
      '**SECRETARIA** DOS**Humanos**' -> '**SECRETARIA DOS Humanos**'
    """
    if is_table_row(line) or "**" not in line:
        return line

    stripped = line.strip()

    # Already a single bold block? leave it
    if stripped.startswith("**") and stripped.endswith("**") and stripped.count("**") == 2:
        return line

    # Fix glued boundaries like 'DOS**Humanos**'
    line = _fix_glued_bold_boundaries(line)

    # Remove all bold markers and rewrap
    content = line.replace("**", "")
    content = re.sub(r"\s+", " ", content).strip()
    if not content:
        return line
    return f"**{content}**"

def clean_inline_bold_everywhere(md: str) -> str:
    """Apply inline bold cleaning to all lines in the document."""
    return "\n".join(consolidate_inline_bold_on_line(line) for line in md.splitlines())

_CAPS_LETTERS = r"A-Za-zÀ-ÖØ-öø-ÿ"

def _is_all_caps_text(text: str) -> bool:
    """
    True if the alphabetic letters in text are all uppercase.
    Non-letters (spaces, punctuation, digits) are ignored.
    Requires at least 2 letters to avoid matching '**A**' etc.
    """
    letters = re.findall(fr"[{_CAPS_LETTERS}]", text)
    if len(letters) < 2:
        return False
    return "".join(letters).upper() == "".join(letters)

def merge_bold_runs_table_safe_allcaps(md: str) -> str:
    """
    Merge consecutive bold-only lines into a single bold block
    BUT ONLY if each of those lines is ALL-CAPS. Table-safe.
    """
    out, buf, in_table = [], [], False

    def flush():
        nonlocal buf
        if buf:
            out.append("**" + "\n".join(buf) + "**")
            buf = []

    for line in md.splitlines():
        if is_table_row(line):
            flush()
            in_table = True
            out.append(line)
            continue
        if in_table and not is_table_row(line):
            in_table = False

        m = re.match(r'^\s*\*\*(.+)\*\*\s*$', line)
        if m and _is_all_caps_text(m.group(1)):
            # Accumulate only ALL-CAPS bold-only lines
            buf.append(m.group(1))
        else:
            flush()
            out.append(line)

    flush()
    return "\n".join(out)
