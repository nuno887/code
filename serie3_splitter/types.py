from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class SubSlice:
    """A child subdivision inside a DocSlice, opened by a payload-approved internal header block."""
    title: str
    headers: List[str]
    body: str
    start: int
    end: int


@dataclass
class DocSlice:
    doc_name: str
    text: str
    status: str = "pending"
    confidence: float = 0.0
    ents: List[Tuple[str, str, int, int]] = field(default_factory=list)
    subs: List[SubSlice] = field(default_factory=list)


@dataclass
class OrgResult:
    org: str
    status: str
    docs: List[DocSlice]