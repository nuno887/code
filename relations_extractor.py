from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Literal, Iterable, Union, Any
from collections import defaultdict, OrderedDict
import json
import re

from spacy.tokens import Doc, Span

# ---- Labels we care about ----------------------------------------------------
Label = Literal[
    "ORG_LABEL",
    "ORG_WITH_STAR_LABEL",
    "DOC_NAME_LABEL",
    "DOC_TEXT",
    "PARAGRAPH",
]

# Relation kinds are derived from (head.label, tail.label)
RelKind = Literal[
    "ORG→DOC_NAME",
    "ORG→ORG*",
    "ORG*→ORG",         # starred org has sub-org(s)
    "ORG*→DOC_NAME",
    "DOC_NAME→DOC_TEXT",
    "DOC_NAME→PARAGRAPH",
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
    def from_span(
        sp: Span,
        paragraph_id: Optional[int],
        sent_id: Optional[int],
    ) -> "EntitySpan":
        return EntitySpan(
            text=sp.text,
            label=sp.label_,  # type: ignore[assignment]
            start=sp.start_char,
            end=sp.end_char,
            paragraph_id=paragraph_id,
            sent_id=sent_id,
        )

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "label": self.label,
            "start": self.start,
            "end": self.end,
            "paragraph_id": self.paragraph_id,
            "sent_id": self.sent_id,
        }


@dataclass(frozen=True)
class Relation:
    head: EntitySpan
    tail: EntitySpan
    kind: RelKind
    # convenience/meta
    paragraph_id: Optional[int]
    sent_id: Optional[int]
    evidence_text: str  # substring between head and tail (trimmed)

    def to_dict(self) -> dict:
        return {
            "kind": self.kind,
            "head": self.head.to_dict(),
            "tail": self.tail.to_dict(),
            "paragraph_id": self.paragraph_id,
            "sent_id": self.sent_id,
            "evidence_text": self.evidence_text,
        }

# ---- Extractor ---------------------------------------------------------------
class RelationExtractor:
    """
    Deterministic, label-driven extraction:

      • Ignore sentences entirely. Use spaCy entity order and offsets.
      • Paragraph (Sumário item) starts when we see:
          - ORG_WITH_STAR_LABEL  (starred block; subsequent ORG_LABELs are sub-orgs)
          - ORG_LABEL            (only if not currently in a starred block)
      • For each entity, link to the nearest-right entity that forms a valid pair.
      • Valid relation kinds (directional):
          ORG_LABEL           → DOC_NAME_LABEL
          ORG_LABEL           → ORG_WITH_STAR_LABEL
          ORG_WITH_STAR_LABEL → ORG_LABEL        (sub-org)
          ORG_WITH_STAR_LABEL → DOC_NAME_LABEL
          DOC_NAME_LABEL      → DOC_TEXT         (or PARAGRAPH)
        The last one currently collapses PARAGRAPH into DOC_TEXT.
    """

    def __init__(
        self,
        *,
        serieIII: bool = True,
        valid_labels: Tuple[str, ...] = (
            "ORG_LABEL",
            "ORG_WITH_STAR_LABEL",
            "DOC_NAME_LABEL",
            "DOC_TEXT",
            "PARAGRAPH",
        ),
        debug: bool = False,  # kept but unused here
    ):
        self.valid_labels = valid_labels
        self.debug = debug
        self.serieIII = serieIII

    # ----- public API ---------------------------------------------------------
    def extract(self, doc_sumario: Doc) -> List[Relation]:
        ents = self._collect_entities(doc_sumario)

        # Group by paragraph while preserving insertion order
        by_para: Dict[Optional[int], List[EntitySpan]] = {}
        for e in ents:
            by_para.setdefault(e.paragraph_id, []).append(e)

        relations: List[Relation] = []
        for para_id, seq in by_para.items():
            is_star_block = bool(seq and seq[0].label == "ORG_WITH_STAR_LABEL")
            rels_here = self._extract_in_sequence(
                doc_sumario, seq, para_id, sent_id=None, is_star_block=is_star_block
            )
            relations.extend(rels_here)

        return relations

    # ----- internals ----------------------------------------------------------
    def _collect_entities(self, doc: Doc) -> List[EntitySpan]:
        """Collect EntitySpan in spaCy's native order.
        Paragraph boundaries:
          - ORG_WITH_STAR_LABEL: always starts a new paragraph; enter star block.
          - ORG_LABEL: starts new paragraph only if not currently in a star block.
        Star block ends when another ORG_WITH_STAR_LABEL appears.
        """
        collected: List[EntitySpan] = []
        current_pid = -1  # before first item
        in_star_block = False

        for e in doc.ents:  # preserve original order
            if e.label_ not in self.valid_labels:
                continue

            if e.label_ == "ORG_WITH_STAR_LABEL":
                current_pid += 1
                in_star_block = True
            elif e.label_ == "ORG_LABEL":
                if not in_star_block:
                    current_pid += 1

            pid = current_pid if current_pid >= 0 else None

            collected.append(
                EntitySpan.from_span(
                    e,
                    paragraph_id=pid,
                    sent_id=None,  # sentences not used
                )
            )

        return collected

    def _pair_kind(self, head_label: str, tail_label: str) -> Optional[RelKind]:
        # Common pairs (unchanged by serieIII)
        if head_label == "ORG_LABEL" and tail_label == "DOC_NAME_LABEL":
            return "ORG→DOC_NAME"
        if head_label == "ORG_LABEL" and tail_label == "ORG_WITH_STAR_LABEL":
            return "ORG→ORG*"
        if head_label == "ORG_WITH_STAR_LABEL" and tail_label == "ORG_LABEL":
            return "ORG*→ORG"
        if head_label == "ORG_WITH_STAR_LABEL" and tail_label == "DOC_NAME_LABEL":
            return "ORG*→DOC_NAME"

        # DOC_NAME -> content: treat DOC_TEXT and PARAGRAPH as equivalent
        if head_label == "DOC_NAME_LABEL" and tail_label in ("DOC_TEXT", "PARAGRAPH"):
            return "DOC_NAME→DOC_TEXT"

        return None

    def _extract_in_sequence(
        self,
        doc: Doc,
        seq: List[EntitySpan],
        paragraph_id: Optional[int],
        sent_id: Optional[int],
        *,
        is_star_block: bool = False,
    ) -> List[Relation]:
        out: List[Relation] = []

        # ---------- STAR BLOCK PATH ----------
        if is_star_block:
            if not seq or seq[0].label != "ORG_WITH_STAR_LABEL":
                is_star_block = False
            else:
                star = seq[0]

                # 1) Add ORG*→ORG links (star → each sub-org)
                for ent in seq[1:]:
                    if ent.label == "ORG_LABEL":
                        out.append(Relation(
                            head=star,
                            tail=ent,
                            kind="ORG*→ORG",
                            paragraph_id=paragraph_id,
                            sent_id=sent_id,
                            evidence_text=doc.text[star.end:ent.start].strip()
                        ))

                # 2) Build sub-blocks per sub-org: [ORG_LABEL ... up to next ORG/STAR)
                block_starts: List[int] = []
                for idx in range(1, len(seq)):
                    if seq[idx].label == "ORG_LABEL":
                        block_starts.append(idx)

                block_bounds: List[tuple[int, int]] = []
                for start_idx in block_starts:
                    end_idx = next(
                        (k for k in range(start_idx + 1, len(seq))
                         if seq[k].label in ("ORG_LABEL", "ORG_WITH_STAR_LABEL")),
                        len(seq)
                    )
                    block_bounds.append((start_idx, end_idx))

                # 3) For each sub-block, run the standard pairing within the block
                for (b_start, b_end) in block_bounds:
                    block = seq[b_start:b_end]
                    out.extend(self._extract_block(doc, block, paragraph_id, sent_id))

                # 4) prune redundant ORG*→DOC_NAME vs ORG→DOC_NAME in this paragraph
                has_star_suborg = any(r.kind == "ORG*→ORG" for r in out)
                if has_star_suborg:
                    docs_from_org = {r.tail.text for r in out if r.kind == "ORG→DOC_NAME"}
                    out = [
                        r for r in out
                        if not (r.kind == "ORG*→DOC_NAME" and r.tail.text in docs_from_org)
                    ]

                return out  # done with star-block path

        # ---------- NON-STAR PATH ----------
        out.extend(self._extract_block(doc, seq, paragraph_id, sent_id))

        # pruning is harmless here too
        has_star_suborg = any(r.kind == "ORG*→ORG" for r in out)
        if has_star_suborg:
            docs_from_org = {r.tail.text for r in out if r.kind == "ORG→DOC_NAME"}
            out = [
                r for r in out
                if not (r.kind == "ORG*→DOC_NAME" and r.tail.text in docs_from_org)
            ]
        return out

    def _extract_block(
        self,
        doc: Doc,
        seq: List[EntitySpan],
        paragraph_id: Optional[int],
        sent_id: Optional[int],
    ) -> List[Relation]:
        out: List[Relation] = []
        linked_tail_labels_by_head: Dict[int, set[str]] = {}

        n = len(seq)
        for i in range(n):
            head = seq[i]
            if head.label not in ("ORG_LABEL", "ORG_WITH_STAR_LABEL", "DOC_NAME_LABEL"):
                continue

            already_for_head = linked_tail_labels_by_head.setdefault(head.start, set())

            for j in range(i + 1, n):
                tail = seq[j]
                kind = self._pair_kind(head.label, tail.label)
                if kind is None:
                    continue

                tail_type = tail.label
                allow_multi = (
                    (head.label == "ORG_WITH_STAR_LABEL" and tail_type == "ORG_LABEL")
                    or (tail_type == "DOC_NAME_LABEL")
                )

                if not allow_multi and tail_type in already_for_head:
                    continue

                out.append(Relation(
                    head=head,
                    tail=tail,
                    kind=kind,
                    paragraph_id=paragraph_id,
                    sent_id=sent_id,
                    evidence_text=doc.text[head.end:tail.start].strip(),
                ))

                if not allow_multi:
                    already_for_head.add(tail_type)

        return out

# ---- Normalization helper for export-time de-duplication ---------------------

def _norm_org(s: str) -> str:
    """
    Normalize organization strings for keying/dedup:
      - Collapse whitespace
      - Normalize hyphen/commas spacing
      - Uppercase
    Used for keys only; display uses original text.
    """
    s = re.sub(r"\s+", " ", s.strip())
    s = re.sub(r"\s*-\s*", "-", s)
    s = re.sub(r"\s*,\s*", ", ", s)  # <-- add the missing third arg: s
    return s.upper()


def _norm_doc(s: str) -> str:
    """
    Normalize document name strings for keying/dedup:
      - Collapse whitespace
      - Normalize hyphen/commas spacing
      - Uppercase
    """
    s = re.sub(r"\s+", " ", s.strip())
    s = re.sub(r"\s*-\s*", "-", s)
    s = re.sub(r"\s*,\s*", ", ", s)
    return s.upper()



# ---- Minimal items exporter with per-paragraph de-dup ------------------------

def export_relations_items_minimal_json(
    relations: Iterable[Relation],
    path: Optional[str] = None,
) -> Union[Dict[str, Any], str]:
    # Group relations by paragraph_id preserving insertion order
    by_pid: "OrderedDict[Optional[int], List[Relation]]" = OrderedDict()
    for r in relations:
        if r.paragraph_id not in by_pid:
            by_pid[r.paragraph_id] = []
        by_pid[r.paragraph_id].append(r)

    items: List[dict] = []

    for pid, rels in by_pid.items():
        # -------- STAR PARAGRAPH --------
        star_links = [r for r in rels if r.kind == "ORG*→ORG"]
        if star_links:
            top_org_head = star_links[0].head
            top_org = {"text": top_org_head.text, "label": top_org_head.label}

            # sub-org order by normalized key
            sub_org_order: "OrderedDict[str, str]" = OrderedDict()
            for r in star_links:
                norm_key = _norm_org(r.tail.text)
                if norm_key not in sub_org_order:
                    sub_org_order[norm_key] = r.tail.text

            # Collect DOC_NAMEs per sub-org, with de-dup (DEDUP)
            docs_by_norm_org: Dict[str, List[dict]] = {}
            seen_docs_by_norm_org: Dict[str, set[str]] = {}

            for r in rels:
                if r.kind == "ORG→DOC_NAME" and r.head.label == "ORG_LABEL":
                    org_key = _norm_org(r.head.text)
                    doc_key = _norm_doc(r.tail.text)

                    if org_key not in seen_docs_by_norm_org:
                        seen_docs_by_norm_org[org_key] = set()
                        docs_by_norm_org[org_key] = []

                    if doc_key not in seen_docs_by_norm_org[org_key]:
                        docs_by_norm_org[org_key].append(
                            {"text": r.tail.text, "label": r.tail.label}
                        )
                        seen_docs_by_norm_org[org_key].add(doc_key)

            sub_orgs: List[dict] = []
            for norm_key, original_text in sub_org_order.items():
                sub_orgs.append({
                    "org": {"text": original_text, "label": "ORG_LABEL"},
                    "docs": docs_by_norm_org.get(norm_key, [])
                })

            items.append({
                "paragraph_id": pid,
                "top_org": top_org,
                "sub_orgs": sub_orgs
            })
            continue

        # -------- NON-STAR PARAGRAPH --------
        org_doc_rels = [
            r for r in rels
            if r.kind == "ORG→DOC_NAME" and r.head.label == "ORG_LABEL"
        ]
        if org_doc_rels:
            primary_head = org_doc_rels[0].head
            primary_key = _norm_org(primary_head.text)

            # Gather docs for all relations with same normalized org key, with de-dup (DEDUP)
            docs: List[dict] = []
            seen_doc_keys: set[str] = set()

            for r in org_doc_rels:
                if _norm_org(r.head.text) == primary_key:
                    doc_key = _norm_doc(r.tail.text)
                    if doc_key not in seen_doc_keys:
                        docs.append({"text": r.tail.text, "label": r.tail.label})
                        seen_doc_keys.add(doc_key)

            items.append({
                "paragraph_id": pid,
                "org": {"text": primary_head.text, "label": primary_head.label},
                "docs": docs
            })
            continue

        # Fallback: no ORG→DOC_NAME found; skip

    payload = {"items": items}

    if path is None:
        return payload

    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return path
