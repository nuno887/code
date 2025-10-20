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
        ),
    ):
        self.debug = debug
        self.valid_labels = valid_labels
        self.serieIII = True  # fixed: III Série distinguishes DOC_TEXT vs PARAGRAPH

    # --- public API -----------------------------------------------------------
    def extract(self, doc_sumario: Doc) -> List[Relation]:
        ents = self._collect_entities(doc_sumario)  # preserves original order

        # Group by paragraph (III Série item)
        by_para: Dict[Optional[int], List[EntitySpan]] = {}
        for e in ents:
            by_para.setdefault(e.paragraph_id, []).append(e)

        relations: List[Relation] = []
        for pid, seq in by_para.items():
            # In III Série we usually don't have star blocks with sub-org lists,
            # so a single block pass often suffices. Keep hook if needed later.
            relations.extend(self._extract_block(doc_sumario, seq, pid, sent_id=None))

        return relations

    # --- internals ------------------------------------------------------------
    def _dbg(self, *args):
        if self.debug:
            print("[serieIII]", *args)

    def _collect_entities(self, doc: Doc) -> List[EntitySpan]:
        """
        III Série paragraph detection (skeleton):
          • Start a NEW paragraph when we see a DOC_NAME_LABEL.
          • If an ORG_LABEL appears and no paragraph has started yet, start a paragraph.
          • If an ORG_WITH_STAR_LABEL appears, also start a paragraph (rare but supported).

        TODO: If your corpus uses a custom 'Sumario' header label or line-break markers
        to delimit items, add that logic here.
        """
        collected: List[EntitySpan] = []
        current_pid = -1
        started = False  # whether we've started the first paragraph

        for e in doc.ents:  # keep spaCy's order
            if e.label_ not in self.valid_labels:
                continue

            if e.label_ in ("DOC_NAME_LABEL", "ORG_WITH_STAR_LABEL"):
                current_pid += 1
                started = True
            elif e.label_ == "ORG_LABEL" and not started:
                # If an ORG appears before any DOC_NAME in the document, treat it as a start.
                current_pid += 1
                started = True
            # else: remain in current paragraph

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
    ) -> List[Relation]:
        """
        Minimal left→right scan:
          - Link ORG/ORG* → nearest-right DOC_NAME
          - Link DOC_NAME → nearest-right DOC_TEXT or PARAGRAPH (whichever appears first)
        No cross-paragraph links. Deterministic. No dependency parsing.
        """
        out: List[Relation] = []
        # track per-head tail label types to avoid duplicate *types* per head,
        # BUT allow multiple DOC_NAMEs for one ORG (common in III Série? usually 1, but safe).
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

                # allow multiple *→DOC_NAME; dedupe other tail types
                allow_multi = (tail.label == "DOC_NAME_LABEL")
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

                # For DOC_NAME → (DOC_TEXT | PARAGRAPH), stop after the FIRST content we found
                if head.label == "DOC_NAME_LABEL":
                    break

            # OPTIONAL: if you want only the first DOC_NAME per ORG in III Série, uncomment:
            # if head.label in ("ORG_LABEL", "ORG_WITH_STAR_LABEL"):
            #     break

        # No special pruning step here; if you later add star blocks that also link directly to docs,
        # you can add a similar pruning as in the base module.
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
    Minimal 'items' JSON tailored to III Série.

    Heuristic mapping (skeleton):
      - Each paragraph_id becomes one item.
      - Primary DOC_NAME is the first DOC_NAME encountered in that paragraph.
      - 'body' prefers DOC_TEXT; if absent, falls back to PARAGRAPH (kept separate if you want both).

    TODO: If an item contains multiple DOC_NAMEs (rare in III Série), either:
          (a) split into multiple items here, or
          (b) keep an array of doc_names and arrays of bodies. Current skeleton keeps the first.
    """
    by_pid: "OrderedDict[Optional[int], List[Relation]]" = OrderedDict()
    for r in relations:
        by_pid.setdefault(r.paragraph_id, []).append(r)

    items: List[dict] = []
    for pid, rels in by_pid.items():
        # collect parts
        doc_names = [r.tail for r in rels if r.kind in ("ORG→DOC_NAME", "ORG*→DOC_NAME")]
        body_texts = [r.tail for r in rels if r.kind == "DOC_NAME→DOC_TEXT"]
        paras = [r.tail for r in rels if r.kind == "DOC_NAME→PARAGRAPH"]
        orgs = [r.head for r in rels if r.kind in ("ORG→DOC_NAME", "ORG*→DOC_NAME")]

        item: dict = {"paragraph_id": pid}

        if orgs:
            # keep distinct orgs in order of appearance
            seen = set()
            item["orgs"] = []
            for h in orgs:
                if h.text not in seen:
                    seen.add(h.text)
                    item["orgs"].append({"text": h.text, "label": h.label})

        if doc_names:
            # choose the first as the primary name (skeleton)
            item["doc_name"] = {"text": doc_names[0].text, "label": doc_names[0].label}
            # TODO: if you want all doc names, add: item["doc_names"] = [...]
        # body: prefer DOC_TEXT; else PARAGRAPH
        if body_texts:
            item["body"] = {"text": body_texts[0].text, "label": body_texts[0].label}
        elif paras:
            item["body"] = {"text": paras[0].text, "label": paras[0].label}

        # Only append non-empty items
        if any(k in item for k in ("orgs", "doc_name", "body")):
            items.append(item)

    payload = {"items": items}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
