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
    "ORG→DOC_TEXT",
    "ORG→PARAGRAPH",
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
        # No cutoff logic anymore — per-ORG Mode B scoping happens in _collect_entities
        ents = self._collect_entities(doc_sumario)

        # Group by paragraph (III Série item)
        by_para: Dict[Optional[int], List[EntitySpan]] = {}
        for e in ents:
            by_para.setdefault(e.paragraph_id, []).append(e)

        relations: List[Relation] = []
        for pid, seq in by_para.items():
            has_org = any(e.label in ("ORG_LABEL", "ORG_WITH_STAR_LABEL") for e in seq)
            has_marker = any(e.label == "SERIE_III" for e in seq)  # injected only for Mode B paragraphs
            mode: Literal["A", "B"] = "B" if (has_org and has_marker) else "A"

            if self.debug:
                num_docname = sum(1 for e in seq if e.label == "DOC_NAME_LABEL")
                num_bodies = sum(1 for e in seq if e.label in ("DOC_TEXT", "PARAGRAPH"))
                self._dbg(
                    f"pid={pid} mode={mode} ents={len(seq)} org={has_org} marker={has_marker} "
                    f"doc_names={num_docname} bodies={num_bodies}"
                )

            relations.extend(self._extract_block(doc_sumario, seq, pid, sent_id=None, mode=mode))

        if self.debug:
            self._dbg(f"TOTAL relations: {len(relations)}")
        return relations




    # --- internals ------------------------------------------------------------
    def _dbg(self, *args):
        if self.debug:
            print("[serieIII]", *args)

    def _collect_entities(self, doc: Doc) -> List[EntitySpan]:
        """
        Paragraph detection with per-ORG Mode B scoping:
        - Start in Mode A.
        - On ORG_*: set current ORG, reset marker -> back to Mode A for this ORG.
        - On SERIE_III: DO NOT append as a normal entity; just remember it and
            flag Mode B as active for this current ORG.
        - On DOC_NAME_LABEL: start a new paragraph; always propagate the current ORG;
            if Mode B is active for this ORG, also propagate the SERIE_III marker.
        """
        collected: List[EntitySpan] = []
        current_pid = -1
        started = False

        # per-ORG state
        last_org_span: Optional[Span] = None
        last_marker_span: Optional[Span] = None
        mode_b_for_current_org = False

        for e in doc.ents:  # keep spaCy's order
            if e.label_ not in self.valid_labels:
                continue

            if e.label_ in ("ORG_LABEL", "ORG_WITH_STAR_LABEL"):
                # new ORG scope: reset marker → back to Mode A
                last_org_span = e
                last_marker_span = None
                mode_b_for_current_org = False

                if not started:
                    current_pid += 1
                    started = True

                # record the ORG itself in the current paragraph
                pid = current_pid if current_pid >= 0 else None
                collected.append(EntitySpan.from_span(e, paragraph_id=pid, sent_id=None))
                continue

            if e.label_ == "SERIE_III":
                # do NOT append the marker as a normal entity; just remember it
                if last_org_span is not None:
                    last_marker_span = e
                    mode_b_for_current_org = True
                # skip adding to collected
                continue

            if e.label_ == "DOC_NAME_LABEL":
                # every DOC_NAME starts a new paragraph
                current_pid += 1
                started = True

                # propagate the current ORG into this new paragraph (A & B)
                if last_org_span is not None:
                    collected.append(EntitySpan.from_span(last_org_span, paragraph_id=current_pid, sent_id=None))
                # propagate SERIE_III ONLY if Mode B is active for this ORG
                if mode_b_for_current_org and last_marker_span is not None:
                    collected.append(EntitySpan.from_span(last_marker_span, paragraph_id=current_pid, sent_id=None))

                # then add the DOC_NAME itself
                collected.append(EntitySpan.from_span(e, paragraph_id=current_pid, sent_id=None))
                continue

            # default: other labels (DOC_TEXT, PARAGRAPH, etc.)
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
        # NOTE: ORG → DOC_TEXT/PARAGRAPH is handled only in Mode A fallback inside _extract_block
        return None

    def _extract_block(
        self,
        doc: Doc,
        seq: List[EntitySpan],
        paragraph_id: Optional[int],
        sent_id: Optional[int],
        mode: Literal["A", "B"] = "A",
    ) -> List[Relation]:
        out: List[Relation] = []

        # ---- Mode A fallback: no DOC_NAME in this paragraph → link ORG → each body
        if mode == "A":
            has_docname = any(e.label == "DOC_NAME_LABEL" for e in seq)
            if not has_docname:
                org = next((e for e in seq if e.label in ("ORG_LABEL", "ORG_WITH_STAR_LABEL")), None)
                bodies = [e for e in seq if e.label in ("DOC_TEXT", "PARAGRAPH")]

                if self.debug:
                    self._dbg(f"pid={paragraph_id} ModeA Fallback: org={'yes' if org else 'no'}, bodies={len(bodies)}")

                if org is not None and bodies:
                    for b in bodies:
                        kind: RelKind = "ORG→DOC_TEXT" if b.label == "DOC_TEXT" else "ORG→PARAGRAPH"
                        out.append(Relation(
                            head=org,
                            tail=b,
                            kind=kind,
                            paragraph_id=paragraph_id,
                            sent_id=sent_id,
                            evidence_text=doc.text[org.end:b.start].strip(),
                        ))
                    if self.debug:
                        self._dbg(f"pid={paragraph_id} ModeA Fallback: created {len(out)} ORG→BODY relations")
                    return out  # done for this paragraph

        # ---- Standard left-to-right scan (both modes)
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

                if mode == "B" and head.label == "DOC_NAME_LABEL" and tail.label in ("DOC_TEXT", "PARAGRAPH"):
                    allow_multi = True
                else:
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

                if head.label == "DOC_NAME_LABEL" and mode != "B":
                    break

        if self.debug:
            # quick count by kind to see what we actually created
            from collections import Counter
            c = Counter(r.kind for r in out)
            self._dbg(f"pid={paragraph_id} standard-scan relations: total={len(out)} kinds={dict(c)}")
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

from typing import Iterable, List, Optional
from collections import OrderedDict
import json  # only needed if you later want a JSON string

def export_serieIII_items_minimal_json(relations: Iterable["Relation"]) -> dict:
    """
    Compact III Série builder (returns a Python dict):
      - Top-level "orgs": [{id, text, label}]
      - Items: { paragraph_id, org_ids: [id...], doc_name (or None), bodies: [...] }
      - Supports both:
          * DOC_NAME path: DOC_NAME→(DOC_TEXT|PARAGRAPH)
          * Mode A fallback: ORG→(DOC_TEXT|PARAGRAPH) when no DOC_NAME
    """
    # 1) Bucket by paragraph
    by_pid: "OrderedDict[Optional[int], List[Relation]]" = OrderedDict()
    for r in relations:
        by_pid.setdefault(r.paragraph_id, []).append(r)

    # 2) Collect unique ORGs across the whole doc and assign ids
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
        # ORGs for this paragraph (from both ORG→DOC_NAME and ORG→BODY fallback)
        org_heads = [
            r.head for r in rels
            if r.head.label in ("ORG_LABEL", "ORG_WITH_STAR_LABEL")
            and r.kind in ("ORG→DOC_NAME", "ORG*→DOC_NAME", "ORG→DOC_TEXT", "ORG→PARAGRAPH")
        ]
        org_ids: list[int] = []
        seen_local: set[int] = set()
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
        # Bodies path B: from ORG→... (Mode A fallback)
        bodies_org = [r.tail for r in rels if r.kind in ("ORG→DOC_TEXT", "ORG→PARAGRAPH")]

        item: dict = {"paragraph_id": pid, "org_ids": org_ids}
        if doc_span is not None:
            item["doc_name"] = {"text": doc_span.text, "label": doc_span.label}
            item["children"] = [{"child": b.text, "label": b.label} for b in bodies_docname]
        else:
            item["doc_name"] = None
            item["children"] = [{"child": b.text, "label": b.label} for b in bodies_org]

        items.append(item)

    payload = {"orgs": orgs_out, "items": items}
    return payload








