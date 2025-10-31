from __future__ import annotations
from dataclasses import dataclass, field
from typing import List

@dataclass
class SpanInfo:
    label: str
    text: str
    start_char: int
    end_char: int

@dataclass
class DocSlice:
    doc_name: str
    text: str

@dataclass
class OrgBlockResult:
    org: str
    org_block_text: str
    docs: List[DocSlice] = field(default_factory=list)
    status: str = "ok"  # ok | partial | doc_missing | org_missing