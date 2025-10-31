from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

@dataclass
class SubSlice:
    """A child subdivision inside a DocSlice, opened by a payload-approved internal header block."""
    title: str             # canonical title (one of the allowed child titles)
    headers: List[str]     # all consecutive DOC_NAME_LABEL lines grouped into this header block (normalized)
    body: str              # text from end of header block up to next approved header (or end)
    start: int             # start offset of the body (relative to the parent seg_text)
    end: int               # end offset of the body (relative to the parent seg_text)

@dataclass
class DocSlice:
    doc_name: str
    text: str
    status: str = "pending"
    confidence: float = 0.0
    # lightweight entity snapshot for this segment (label, text, start, end), offsets relative to seg_text
    ents: List[Tuple[str, str, int, int]] = field(default_factory=list)
    # optional further subdivision inside the segment
    subs: List[SubSlice] = field(default_factory=list)

@dataclass
class OrgResult:
    org: str
    status: str
    docs: List[DocSlice]
