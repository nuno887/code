# relations_extractor_serieIII.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Literal, Iterable
from collections import defaultdict, OrderedDict
import json
import csv
from spacy.tokens import Doc, Span

# ---- Labels used -------------------------------------------------------------
Label = Literal[
    "ORG_LABEL",
    "ORG_WITH_STAR_LABEL",
    "DOC_NAME_LABEL",
    "DOC_TEXT",
    "PARAGRAPH",
    "SERIE_III",
]

RelKind = Literal[
    "ORG→DOC_NAME",
    "ORG*→DOC_NAME",
    "DOC_NAME→DOC_TEXT",
    "DOC_NAME→PARAGRAPH",
]

# ---- Data classes (standalone; you could also import from your base module) --
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
    III Série extractor (skeleton)

    Key differences vs. I/II:
      • Items frequently start with DOC_NAME_LABEL (ORG may be missing or come later).
      • We ALWAYS distinguish DOC_TEXT vs PARAGRAPH (serieIII=True).
      • Starred orgs can still appear, but typically fewer 'sub-org' trees in III Série.

    Relation kinds (minimal):
      ORG_LABEL          → DOC_NAME_LABEL
      ORG_WITH_STAR_LABEL→ DOC_NAME_LABEL
      DOC_NAME_LABEL     → DOC_TEXT
      DOC_NAME_LABEL     → PARAGRAPH
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
            "SERIE_III",
        ),
    ):
        self.debug = debug
        self.valid_labels = valid_labels
        self.serieIII = True  # fixed: III Série distinguishes DOC_TEXT vs PARAGRAPH

    # --- public API -----------------------------------------------------------
    def extract(self, doc_sumario: Doc) -> List[Relation]:
        ents = self._collect_entities(doc_sumario)

        by_para: Dict[Optional[int], List[EntitySpan]] = {}
        for e in ents:
            by_para.setdefault(e.paragraph_id, []).append(e)

        relations: List[Relation] = []
        for pid, seq in by_para.items():
            has_org = any(e.label in ("ORG_LABEL", "ORG_WITH_STAR_LABEL") for e in seq)
            has_marker = any(e.label == "SERIE_III" for e in seq)
            mode: Literal["A", "B"] = "B" if (has_org and has_marker) else "A"

            relations.extend(self._extract_block(doc_sumario, seq, pid, sent_id=None, mode=mode))
        return relations


    # --- internals ------------------------------------------------------------
    def _dbg(self, *args):
        if self.debug:
            print("[serieIII]", *args)

    def _collect_entities(self, doc: Doc) -> List[EntitySpan]:
        collected: List[EntitySpan] = []
        current_pid = -1
        started = False

        # >>> ADDED: track Mode B activation and last seen ORG / marker spans
        mode_b_active = False
        last_org_span: Optional[Span] = None
        last_marker_span: Optional[Span] = None
        seen_org = False
        seen_marker = False
        # <<<

        for e in doc.ents:
            if e.label_ not in self.valid_labels:
                continue

            # >>> ADDED: remember ORG and SERIE_III, and (lazily) activate Mode B when both seen
            if e.label_ in ("ORG_LABEL", "ORG_WITH_STAR_LABEL"):
                seen_org = True
                last_org_span = e
            elif e.label_ == "SERIE_III":
                seen_marker = True
                last_marker_span = e

            if not mode_b_active and seen_org and seen_marker:
                mode_b_active = True
            # <<<

            if e.label_ in ("DOC_NAME_LABEL", "ORG_WITH_STAR_LABEL"):
                # CHANGED: in Mode B, every DOC_NAME starts a new paragraph
                if e.label_ == "DOC_NAME_LABEL" and mode_b_active:
                    current_pid += 1
                    started = True

                    # >>> ADDED: propagate ORG and SERIE_III into this new paragraph
                    if last_org_span is not None:
                        collected.append(
                            EntitySpan.from_span(last_org_span, paragraph_id=current_pid, sent_id=None)
                        )
                    if last_marker_span is not None:
                        collected.append(
                            EntitySpan.from_span(last_marker_span, paragraph_id=current_pid, sent_id=None)
                        )
                    # <<<

                else:
                    current_pid += 1
                    started = True

            elif e.label_ == "ORG_LABEL" and not started:
                current_pid += 1
                started = True

            pid = current_pid if current_pid >= 0 else None
            collected.append(EntitySpan.from_span(e, paragraph_id=pid, sent_id=None))

        if self.debug:
            self._dbg(f"Collected ents: {len(collected)} across {current_pid + 1 if current_pid >= 0 else 0} paragraphs")
        return collected


    def _pair_kind(self, head_label: str, tail_label: str) -> Optional[RelKind]:
        # ORG → DOC_NAME
        if head_label == "ORG_LABEL" and tail_label == "DOC_NAME_LABEL":
            return "ORG→DOC_NAME"
        if head_label == "ORG_WITH_STAR_LABEL" and tail_label == "DOC_NAME_LABEL":
            return "ORG*→DOC_NAME"

        # DOC_NAME → DOC_TEXT / PARAGRAPH (III Série distinguishes these)
        if head_label == "DOC_NAME_LABEL":
            if tail_label == "DOC_TEXT":
                return "DOC_NAME→DOC_TEXT"
            if tail_label == "PARAGRAPH":
                return "DOC_NAME→PARAGRAPH"

        return None

    def _extract_block(
        self,
        doc: Doc,
        seq: List[EntitySpan],
        paragraph_id: Optional[int],
        sent_id: Optional[int],
        mode: Literal["A", "B"] = "A",   # (already added previously)
    ) -> List[Relation]:
        out: List[Relation] = []
        linked_tail_labels_by_head: Dict[int, set[str]] = {}

        n = len(seq)
        for i in range(n):
            head = seq[i]
            if head.label not in ("ORG_LABEL", "ORG_WITH_STAR_LABEL", "DOC_NAME_LABEL"):
                continue

            already = linked_tail_labels_by_head.setdefault(head.start, set())

            for j in range(i + 1, n):
                tail = seq[j]
                kind = self._pair_kind(head.label, tail.label)
                if kind is None:
                    continue

                # >>> CHANGED: in Mode B, allow multiple bodies (DOC_TEXT / PARAGRAPH) per DOC_NAME
                if mode == "B" and head.label == "DOC_NAME_LABEL" and tail.label in ("DOC_TEXT", "PARAGRAPH"):
                    allow_multi = True
                else:
                    allow_multi = (tail.label == "DOC_NAME_LABEL")
                # <<<

                if not allow_multi and tail.label in already:
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
                    already.add(tail.label)

                # >>> CHANGED: in Mode B, do NOT stop after the first body; in Mode A, keep old behavior
                if head.label == "DOC_NAME_LABEL" and mode != "B":
                    break
                # <<<
        return out


# ---- Export helpers (III Série tailored) -------------------------------------

def export_serieIII_csv_compact(relations: Iterable[Relation], path: str) -> None:
    """
    Compact CSV (III Série): paragraph_id, kind, head(label,text), tail(label,text)
    """
    fields = ["paragraph_id", "kind", "head_label", "head_text", "tail_label", "tail_text"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in relations:
            w.writerow({
                "paragraph_id": r.paragraph_id,
                "kind": r.kind,
                "head_label": r.head.label, "head_text": r.head.text,
                "tail_label": r.tail.label, "tail_text": r.tail.text,
            })

def export_serieIII_json_grouped(relations: Iterable[Relation], path: str) -> None:
    """
    Grouped JSON (simple): retains only kind + head/tail text+label, grouped by paragraph.
    """
    buckets: Dict[Optional[int], List[dict]] = defaultdict(list)
    order: List[Optional[int]] = []
    for r in relations:
        pid = r.paragraph_id
        if pid not in buckets:
            order.append(pid)
        buckets[pid].append({
            "kind": r.kind,
            "head": {"text": r.head.text, "label": r.head.label},
            "tail": {"text": r.tail.text, "label": r.tail.label},
        })

    payload = {
        "paragraphs": [
            {"paragraph_id": pid, "relations": buckets[pid]}
            for pid in order
        ]
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

def export_serieIII_items_minimal_json(relations: Iterable[Relation], path: str) -> None:
    """
    Variant of the III Série items exporter that avoids repeating ORGs on every item.
    - Top-level "orgs": [{id, text, label}]
    - Each item references orgs via "org_ids": [int, ...]
    - Items still include: paragraph_id, doc_name, bodies (list)
    """
    from collections import OrderedDict

    # 1) Bucket relations by paragraph
    by_pid: "OrderedDict[Optional[int], List[Relation]]" = OrderedDict()
    for r in relations:
        by_pid.setdefault(r.paragraph_id, []).append(r)

    # 2) Collect unique orgs across the whole document; assign stable ids
    org_to_id: dict[tuple[str, str], int] = {}
    orgs_out: list[dict] = []
    def get_org_id(text: str, label: str) -> int:
        key = (text, label)
        if key not in org_to_id:
            org_to_id[key] = len(org_to_id) + 1
            orgs_out.append({"id": org_to_id[key], "text": text, "label": label})
        return org_to_id[key]

    items: List[dict] = []

    for pid, rels in by_pid.items():
        # ORGs (unique per item, but referenced by id)
        org_heads = [r.head for r in rels if r.kind in ("ORG→DOC_NAME", "ORG*→DOC_NAME")]
        org_ids: list[int] = []
        seen_local: set[int] = set()
        for h in org_heads:
            oid = get_org_id(h.text, h.label)
            if oid not in seen_local:
                seen_local.add(oid)
                org_ids.append(oid)

        # DOC_NAME (first encountered, with fallback if only DOC_NAME→* exists)
        doc_names = [r.tail for r in rels if r.kind in ("ORG→DOC_NAME", "ORG*→DOC_NAME")]
        if not doc_names:
            # fallback: take the head of any DOC_NAME→* relation
            heads_docname = [r.head for r in rels if r.head.label == "DOC_NAME_LABEL"]
            doc_span = heads_docname[0] if heads_docname else None
        else:
            doc_span = doc_names[0]

        # Bodies: all DOC_TEXT/PARAGRAPH linked from doc_name in this paragraph
        bodies_all = [r.tail for r in rels if r.kind in ("DOC_NAME→DOC_TEXT", "DOC_NAME→PARAGRAPH")]

        item: dict = {"paragraph_id": pid, "org_ids": org_ids}
        if doc_span is not None:
            item["doc_name"] = {"text": doc_span.text, "label": doc_span.label}
        if bodies_all:
            item["bodies"] = [{"text": b.text, "label": b.label} for b in bodies_all]
        else:
            item["bodies"] = []  # empty list if no body for this DOC_NAME

        # Keep the item (even if it only has doc_name without bodies)
        if any(k in item for k in ("doc_name", "bodies", "org_ids")):
            items.append(item)

    payload = {"orgs": orgs_out, "items": items}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)



