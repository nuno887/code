from .Serie_I_II_IV.types import SpanInfo, DocSlice, OrgBlockResult
from .Serie_I_II_IV.extract import divide_body_by_org_and_docs, print_summary, normalize_doc_title

# ======================= Serie III ==========================
from .Serie_IV.segmenter import divide_body_by_org_and_docs_serieIII
from .Serie_IV.models import SubSlice, DocSlice, OrgResult
# from .Serie_IV.debug import  DBG

# ============================================================
__all__ = [
    "SpanInfo",
    "DocSlice",
    "OrgBlockResult",
    "divide_body_by_org_and_docs",
    "print_summary",
    "normalize_doc_title",
    "divide_body_by_org_and_docs_serieIII",
    "SubSlice",
    "DocSlice",
    "OrgResult",
]

