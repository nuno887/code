from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Literal, Iterable
from collections import defaultdict, OrderedDict
import json
import csv
from spacy.tokens import Doc, Span
import re

# ---- Labels used -------------------------------------------------------------
Label = Literal[
    "ORG_LABEL",
    "ORG_WITH_STAR_LABEL",
    "DOC_NAME_LABEL",
    "DOC_TEXT",
    "PARAGRAPH",
    "SERIE_III",  # kept for backward compatibility; ignored in logic
]

RelKind = Literal[
    "ORG→DOC_NAME",
    "ORG*→DOC_NAME",
    "DOC_NAME→DOC_TEXT",
    "DOC_NAME→PARAGRAPH",
    "ORG→DOC_TEXT",
    "ORG→PARAGRAPH",
]

# ---- Data classes ------------------------------------------------------------
@dataclass(frozen=True)
class EntitySpan:
    text: str
    label: Label
    start: int
    end: int
    paragraph_id: Optional[int]
    sent_id: Optional[int]

    @staticmethod
    def from_span(sp: Span, paragraph_id: Optional[int], sent_id: Optional[int]) -> "EntitySpan":
        return EntitySpan(
            text=sp.text,
            label=sp.label_,  # type: ignore[assignment]
            start=sp.start_char,
            end=sp.end_char,
            paragraph_id=paragraph_id,
            sent_id=sent_id,
        )

@dataclass(frozen=True)
class Relation:
    head: EntitySpan
    tail: EntitySpan
    kind: RelKind
    paragraph_id: Optional[int]
    sent_id: Optional[int]
    evidence_text: str

# ---- Extractor ---------------------------------------------------------------
class RelationExtractorSerieIII:
    """
    III Série extractor (refactored, no Mode A/B)

    Behavior:
      • Items frequently start with DOC_NAME_LABEL (ORG may be missing or come later).
      • Always distinguish DOC_TEXT vs PARAGRAPH.
      • DOC_NAME links to ALL following bodies (DOC_TEXT/PARAGRAPH) in the paragraph.
      • If a paragraph has no DOC_NAME, link ORG(s) → every body in that paragraph.
      • SERIE_III markers are ignored (kept in labels for compatibility).
    """

    def __init__(
        self,
        *,
        debug: bool = False,
        valid_labels: Tuple[str, ...] = (
            "ORG_LABEL",
            "ORG_WITH_STAR_LABEL",
            "DOC_NAME_LABEL",
            "DOC_TEXT",
            "PARAGRAPH",
            "SERIE_III",  # inert
        ),
    ):
        self.debug = debug
        self.valid_labels = valid_labels
        self.serieIII = True  # keep flag for downstream expectations

    # --- public API -----------------------------------------------------------
    def extract(self, doc_sumario: Doc) -> List[Relation]:
        ents = self._collect_entities(doc_sumario)

        # Group by paragraph
        by_para: Dict[Optional[int], List[EntitySpan]] = {}
        for e in ents:
            by_para.setdefault(e.paragraph_id, []).append(e)

        relations: List[Relation] = []
        for pid, seq in by_para.items():
            relations.extend(self._extract_block(doc_sumario, seq, pid, sent_id=None))

        return relations

    # --- internals ------------------------------------------------------------
    def _collect_entities(self, doc: Doc) -> List[EntitySpan]:
        """
        Paragraph detection (no Mode B):
        - Track the most recent ORG.
        - Ignore SERIE_III markers entirely.
        - Every DOC_NAME starts a new paragraph; propagate the current ORG (if any).
        - Other labels (DOC_TEXT, PARAGRAPH) belong to the current paragraph.
        """
        collected: List[EntitySpan] = []
        current_pid = -1
        started = False

        last_org_span: Optional[Span] = None

        for e in doc.ents:  # keep spaCy's order
            if e.label_ not in self.valid_labels:
                continue

            if e.label_ in ("ORG_LABEL", "ORG_WITH_STAR_LABEL"):
                # remember current ORG
                last_org_span = e

                # ensure we have a paragraph to attach to (prior to first DOC_NAME)
                if not started:
                    current_pid += 1
                    started = True

                pid = current_pid if current_pid >= 0 else None
                collected.append(EntitySpan.from_span(e, paragraph_id=pid, sent_id=None))
                continue

            if e.label_ == "SERIE_III":
                # legacy artifact: completely ignore
                continue

            if e.label_ == "DOC_NAME_LABEL":
                # every DOC_NAME starts a new paragraph
                current_pid += 1
                started = True

                # propagate the current ORG (if any)
                if last_org_span is not None:
                    collected.append(EntitySpan.from_span(last_org_span, paragraph_id=current_pid, sent_id=None))

                # add the DOC_NAME itself
                collected.append(EntitySpan.from_span(e, paragraph_id=current_pid, sent_id=None))
                continue

            # default: other labels (DOC_TEXT, PARAGRAPH, etc.)
            pid = current_pid if current_pid >= 0 else None
            collected.append(EntitySpan.from_span(e, paragraph_id=pid, sent_id=None))

        return collected

    def _pair_kind(self, head_label: str, tail_label: str) -> Optional[RelKind]:
        # ORG → DOC_NAME
        if head_label == "ORG_LABEL" and tail_label == "DOC_NAME_LABEL":
            return "ORG→DOC_NAME"
        if head_label == "ORG_WITH_STAR_LABEL" and tail_label == "DOC_NAME_LABEL":
            return "ORG*→DOC_NAME"

        # DOC_NAME → DOC_TEXT / PARAGRAPH
        if head_label == "DOC_NAME_LABEL":
            if tail_label == "DOC_TEXT":
                return "DOC_NAME→DOC_TEXT"
            if tail_label == "PARAGRAPH":
                return "DOC_NAME→PARAGRAPH"

        # ORG → DOC_TEXT/PARAGRAPH handled in fallback inside _extract_block
        return None

    def _extract_block(
    self,
    doc: Doc,
    seq: List[EntitySpan],
    paragraph_id: Optional[int],
    sent_id: Optional[int],
) -> List[Relation]:
        """
        Build relations for a single paragraph:

        - If the paragraph has NO DOC_NAME:
            pick ONLY the FIRST ORG in the paragraph and link it to each body
            (DOC_TEXT / PARAGRAPH). This avoids multiplying children by the number
            of ORGs and keeps org_ids to a single org.

        - If the paragraph HAS DOC_NAME:
            standard left-to-right scan:
            * ORG(_WITH_STAR) → DOC_NAME for any org that precedes a doc name
            * DOC_NAME → (DOC_TEXT | PARAGRAPH) for all subsequent bodies
        """
        out: List[Relation] = []

        # ---- Fallback: no DOC_NAME in this paragraph → link a single ORG (the first) → each body
        has_docname = any(e.label == "DOC_NAME_LABEL" for e in seq)
        if not has_docname:
            orgs = [e for e in seq if e.label in ("ORG_LABEL", "ORG_WITH_STAR_LABEL")]
            bodies = [e for e in seq if e.label in ("DOC_TEXT", "PARAGRAPH")]

            # Choose only the FIRST ORG (do nothing if none).
            effective_org = orgs[0] if orgs else None

            if effective_org is not None:
                for b in bodies:
                    kind: RelKind = "ORG→DOC_TEXT" if b.label == "DOC_TEXT" else "ORG→PARAGRAPH"
                    out.append(Relation(
                        head=effective_org,
                        tail=b,
                        kind=kind,
                        paragraph_id=paragraph_id,
                        sent_id=sent_id,
                        evidence_text=doc.text[effective_org.end:b.start].strip(),
                    ))
            return out

        # ---- Standard left-to-right scan (DOC_NAME links to ALL following bodies)
        n = len(seq)
        for i in range(n):
            head = seq[i]
            if head.label not in ("ORG_LABEL", "ORG_WITH_STAR_LABEL", "DOC_NAME_LABEL"):
                continue

            for j in range(i + 1, n):
                tail = seq[j]
                kind = self._pair_kind(head.label, tail.label)
                if kind is None:
                    continue

                out.append(Relation(
                    head=head,
                    tail=tail,
                    kind=kind,
                    paragraph_id=paragraph_id,
                    sent_id=sent_id,
                    evidence_text=doc.text[head.end:tail.start].strip(),
                ))

        return out




def export_serieIII_items_minimal_json(
    relations: Iterable["Relation"],
    render: Literal["none", "markdown"] = "none",
) -> dict:
    """
    Compact III Série builder (returns a Python dict):
      - Top-level "orgs": [{id, text, label}]
      - Items: { paragraph_id, org_ids: [id...], doc_name (or None), children: [...] }
      - Supports both:
          * DOC_NAME path: DOC_NAME→(DOC_TEXT|PARAGRAPH)  (multiple children)
          * Fallback: ORG→(DOC_TEXT|PARAGRAPH) when no DOC_NAME

    Child cleaning rules:
      - Keep if it has any letter; or matches "n.º ..." style references.
      - Delete if it's just numbers/symbols (e.g., "9 10 12", "4/2025", "---"), unless it’s "n.º ...".
      - Normalize dot leaders "......" → ".", collapse dash runs.

    If render == "markdown", the returned dict includes an extra key:
      - "markdown": a deduplicated, display-ready Markdown string
    """
    from collections import OrderedDict
    import re

    # Accept things like: "n.º 6/2025", "N.º12", "No. 3/2024", "nº 12"
    _N_DOT_NUM_RE = re.compile(
        r"""(?ix)
        \b
        n
        \s*
        (?:[.\u00BA\u00B0o]\s*){0,2}   # allow ".", "º", "°", "o" in any order, up to two (covers "n.º")
        \d+
        (?:\s*/\s*\d+)?                # optional /year
        \b
        """
    )

    # Sequences of ≥3 dots/ellipsis → single period
    _DOT_LEADER_RE = re.compile(r"[.\u2026·]{3,}")
    # Sequences of dashes/long dashes
    _DASH_RUN_RE = re.compile(r"[-\u2013\u2014]{2,}")

    # Standalone numeric-like tokens (digits optionally with dot/comma or slash),
    # possibly wrapped with light punctuation like parentheses — but NOT if followed by º/° (e.g., "22.º").
    _NUM_TOKEN_RE = re.compile(
        r"""(?x)
        (?<!\w)
        \(?
        \s*
        \d+(?:[.,]\d+)?(?:/\d+)?   # 22 | 3.14 | 4/2025 | 12/2024
        \s*
        \)?
        (?!\s*[º°])                # don't match when followed by º/°
        (?!\w)
        """
    )
    _LEADING_LIST_PREFIX_RE = re.compile(r"""(?x)
    ^\s*
    (?:\(?\d+\)?(?:\s+\(?\d+\)?){0,3})   # up to 4 numbers like: 3 or 3 4 or (3) (4)
    \s*[-–—]\s*                          # followed by a dash
""")

    

    def _clean_child_text(text: str) -> str | None:
        if text is None:
            return None
        t = text.strip()
        if not t:
            return None
        
        # 1) ALWAYS remove leading list prefixes like "3 4 - "
        t = _LEADING_LIST_PREFIX_RE.sub("", t)

        # If it already matches an allowed "n.º …" pattern, keep with light normalization
        if _N_DOT_NUM_RE.search(t):
            t = _DOT_LEADER_RE.sub(".", t)
            t = _DASH_RUN_RE.sub("-", t)
            t = " ".join(t.split())
            return t or None

        # Normalize leaders / dashes
        t = _DOT_LEADER_RE.sub(".", t)
        t = _DASH_RUN_RE.sub("-", t)

        # Remove standalone numeric-like tokens (e.g., "89", "(7)", "4/2025", "3.14")
        t = _NUM_TOKEN_RE.sub("", t)

        # Collapse whitespace
        t = " ".join(t.split())

        # If nothing meaningful remains, or there are no letters and no "n.º …", drop it
        if not t:
            return None
        if not any(ch.isalpha() for ch in t) and not _N_DOT_NUM_RE.search(t):
            return None

        return t

    # 1) Bucket by paragraph
    by_pid = OrderedDict()
    for r in relations:
        by_pid.setdefault(r.paragraph_id, []).append(r)

    # 2) Collect unique ORGs across the whole doc and assign ids
    org_to_id = {}
    orgs_out = []

    def get_org_id(text: str, label: str) -> int:
        key = (text, label)
        if key not in org_to_id:
            org_to_id[key] = len(org_to_id) + 1
            orgs_out.append({"id": org_to_id[key], "text": text, "label": label})
        return org_to_id[key]

    items = []

    for pid, rels in by_pid.items():
        # ORGs for this paragraph (from both ORG→DOC_NAME and ORG→BODY fallback)
        org_heads = [
            r.head for r in rels
            if r.head.label in ("ORG_LABEL", "ORG_WITH_STAR_LABEL")
            and r.kind in ("ORG→DOC_NAME", "ORG*→DOC_NAME", "ORG→DOC_TEXT", "ORG→PARAGRAPH")
        ]
        org_ids = []
        seen_local = set()
        for h in org_heads:
            oid = get_org_id(h.text, h.label)
            if oid not in seen_local:
                seen_local.add(oid)
                org_ids.append(oid)

        # Primary DOC_NAME if present (prefer tail of ORG→DOC_NAME; else any DOC_NAME head)
        docname_tails = [r.tail for r in rels if r.kind in ("ORG→DOC_NAME", "ORG*→DOC_NAME")]
        if docname_tails:
            doc_span = docname_tails[0]
        else:
            heads_doc = [r.head for r in rels if r.head.label == "DOC_NAME_LABEL"]
            doc_span = heads_doc[0] if heads_doc else None

        # Bodies path A: from DOC_NAME→...
        bodies_docname = [r.tail for r in rels if r.kind in ("DOC_NAME→DOC_TEXT", "DOC_NAME→PARAGRAPH")]
        # Bodies path B: from ORG→... (fallback)
        bodies_org = [r.tail for r in rels if r.kind in ("ORG→DOC_TEXT", "ORG→PARAGRAPH")]

        item = {"paragraph_id": pid, "org_ids": org_ids}

        if doc_span is not None:
            item["doc_name"] = {"text": doc_span.text, "label": doc_span.label}
            children = []
            for b in bodies_docname:
                cleaned = _clean_child_text(b.text)
                if cleaned is not None:
                    children.append({"child": cleaned, "label": b.label})
            item["children"] = children
        else:
            item["doc_name"] = None
            children = []
            for b in bodies_org:
                cleaned = _clean_child_text(b.text)
                if cleaned is not None:
                    children.append({"child": cleaned, "label": b.label})
            item["children"] = children

        items.append(item)

    payload = {"orgs": orgs_out, "items": items}

    # --- Optional: attach Markdown rendering (deduped, display-ready)
    if render == "markdown":
        org_lookup = {o["id"]: o for o in payload.get("orgs", [])}
        lines = []

        for item in payload.get("items", []):
            pid = item.get("paragraph_id")
            org_ids = item.get("org_ids", [])
            orgs = [org_lookup[oid]["text"] for oid in org_ids if oid in org_lookup]

            # Header
            lines.append(f"### Parágrafo {pid}" if pid is not None else "### Parágrafo (sem id)")

            # ORGs (unique, inline)
            if orgs:
                uniq_orgs = []
                seen = set()
                for o in orgs:
                    if o not in seen:
                        seen.add(o)
                        uniq_orgs.append(o)
                lines.append(f"**Entidade(s):** {', '.join(uniq_orgs)}")

            # DOC_NAME
            doc_name = item.get("doc_name")
            if doc_name is not None:
                lines.append(f"**Documento:** {doc_name.get('text','').strip()}")
            else:
                lines.append("**Documento:** (não identificado)")

            # Children
            children = item.get("children", [])
            if children:
                lines.append("**Conteúdo:**")
                for ch in children:
                    child_txt = ch.get("child", "").strip()
                    if child_txt:
                        lines.append(f"- {child_txt}")
            else:
                lines.append("_Sem conteúdo associado_")

            lines.append("")  # blank line between paragraphs

        payload["markdown"] = "\n".join(lines)

    return payload
