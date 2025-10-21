
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Literal, Iterable
from collections import defaultdict, OrderedDict
import json
import csv
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
        The last one depends on 'serieIII':
          - If serieIII=True: DOC_TEXT and PARAGRAPH are distinct kinds.
          - If serieIII=False: both are treated as DOC_TEXT (collapsed).
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
        debug: bool = False,
    ):
        self.valid_labels = valid_labels
        self.debug = debug

    # ----- public API ---------------------------------------------------------
    def extract(self, doc_sumario: Doc) -> List[Relation]:
        ents = self._collect_entities(doc_sumario)

        # Group by paragraph while preserving insertion order
        by_para: Dict[Optional[int], List[EntitySpan]] = {}
        for e in ents:
            by_para.setdefault(e.paragraph_id, []).append(e)

        relations: List[Relation] = []
        for para_id, seq in by_para.items():
            # seq is already in original order
            is_star_block = bool(seq and seq[0].label == "ORG_WITH_STAR_LABEL")
            rels_here = self._extract_in_sequence(
                doc_sumario, seq, para_id, sent_id=None, is_star_block=is_star_block
            )
            relations.extend(rels_here)

        return relations

    # ----- internals ----------------------------------------------------------
    def _dbg(self, *args):
        if self.debug:
            print("[relations]", *args)

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
                # start a new paragraph and enter star block
                current_pid += 1
                in_star_block = True

            elif e.label_ == "ORG_LABEL":
                # new paragraph only if NOT inside a starred block
                if not in_star_block:
                    current_pid += 1

            # We don't flip in_star_block off explicitly; a new starred org will start a new block.

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

        # DOC_NAME -> content: in I/II Série we treat DOC_TEXT and PARAGRAPH as equivalent
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
        """
        Left→right scan within a paragraph.
        If is_star_block: star header + multiple sub-orgs; we hard-scope each sub-org's block
        so its DOC_NAMEs cannot bleed into the next org.
        """
        out: List[Relation] = []

        if self.debug:
            self._dbg(f"_extract_in_sequence: {len(seq)} ents in paragraph {paragraph_id}, star={is_star_block}")

        # ---------- STAR BLOCK PATH ----------
        if is_star_block:
            if not seq or seq[0].label != "ORG_WITH_STAR_LABEL":
                # safety: fall back to non-star path
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
        """
        Standard left→right scan applied to a slice (block).
        Allows:
          • multiple *→DOC_NAME (heads can have many doc names)
          • multiple ORG*→ORG (sub-orgs) if a star head is present (usually not in sub-blocks)
        """
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

# ---- Export helpers ----------------------------------------------------------
def export_relations_ndjson(relations: Iterable[Relation], path: str) -> None:
    """Write one JSON object per line (NDJSON)."""
    with open(path, "w", encoding="utf-8") as f:
        for r in relations:
            f.write(json.dumps(r.to_dict(), ensure_ascii=False))
            f.write("\n")

def export_relations_grouped_json(relations: Iterable[Relation], path: str) -> None:
    """Group relations by paragraph_id into a single JSON file (full fields)."""
    buckets: Dict[Optional[int], List[dict]] = defaultdict(list)
    order: List[Optional[int]] = []
    for r in relations:
        pid = r.paragraph_id
        if pid not in buckets:
            order.append(pid)
        buckets[pid].append(r.to_dict())
    payload = {
        "paragraphs": [
            {"paragraph_id": pid, "relations": buckets[pid]}
            for pid in order
        ]
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

def export_relations_grouped_json_compact(relations: Iterable[Relation], path: str) -> None:
    """Compact grouped JSON with only kind, head{text,label}, tail{text,label}."""
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

def export_relations_grouped_json_by_head(relations: Iterable[Relation], path: str) -> None:
    """
    Group by paragraph -> head (text,label) -> kind -> [tails...]
    to avoid repeating the same head for multiple tails.
    """
    paragraphs = OrderedDict()  # pid -> list[Relation]
    for r in relations:
        pid = r.paragraph_id
        paragraphs.setdefault(pid, []).append(r)

    payload = {"paragraphs": []}
    kind_order = ["ORG→DOC_NAME", "ORG→ORG*", "ORG*→ORG", "ORG*→DOC_NAME", "DOC_NAME→DOC_TEXT", "DOC_NAME→PARAGRAPH"]

    for pid, rels in paragraphs.items():
        # group by head (text+label) preserving insertion order
        heads = OrderedDict()  # (head_text, head_label) -> dict(kind -> [tails])
        for r in rels:
            key = (r.head.text, r.head.label)
            if key not in heads:
                heads[key] = defaultdict(list)
            heads[key][r.kind].append({"text": r.tail.text, "label": r.tail.label})

        heads_list = []
        for (h_text, h_label), relmap in heads.items():
            relations_obj = {k: relmap[k] for k in kind_order if k in relmap}
            heads_list.append({
                "head": {"text": h_text, "label": h_label},
                "relations": relations_obj
            })

        payload["paragraphs"].append({
            "paragraph_id": pid,
            "heads": heads_list
        })

    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

def export_relations_csv(relations: Iterable[Relation], path: str) -> None:
    """
    Compact CSV: only paragraph_id, kind, head(label,text), tail(label,text).
    """
    fields = [
        "paragraph_id",
        "kind",
        "head_label", "head_text",
        "tail_label", "tail_text",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in relations:
            w.writerow({
                "paragraph_id": r.paragraph_id,
                "kind": r.kind,
                "head_label": r.head.label,
                "head_text": r.head.text,
                "tail_label": r.tail.label,
                "tail_text": r.tail.text,
            })


def export_relations_items_minimal_json(relations: Iterable[Relation], path: str) -> None:
    """
    Export as minimal hierarchical 'items' (Option D):
      - Non-star paragraph -> { paragraph_id, org, docs[] }
      - Star paragraph     -> { paragraph_id, top_org, sub_orgs[ {org, docs[]} ] }
    Fields keep only {text, label} to stay compact.

    Example:
    {
      "items": [
        {
          "paragraph_id": 0,
          "org":  {"text": "...", "label": "ORG_LABEL"},
          "docs": [{"text":"...","label":"DOC_NAME_LABEL"}, ...]
        },
        {
          "paragraph_id": 5,
          "top_org": {"text":"...","label":"ORG_WITH_STAR_LABEL"},
          "sub_orgs": [
            {
              "org":  {"text":"...","label":"ORG_LABEL"},
              "docs": [{"text":"...","label":"DOC_NAME_LABEL"}, ...]
            }
          ]
        }
      ]
    }
    """
    from collections import OrderedDict

    # Group relations by paragraph_id preserving insertion order
    by_pid: "OrderedDict[Optional[int], List[Relation]]" = OrderedDict()
    for r in relations:
        if r.paragraph_id not in by_pid:
            by_pid[r.paragraph_id] = []
        by_pid[r.paragraph_id].append(r)

    items: List[dict] = []

    for pid, rels in by_pid.items():
        # Is this a star block? (top org with sub-orgs)
        star_links = [r for r in rels if r.kind == "ORG*→ORG"]
        if star_links:
            # pick the first starred head as the top_org (they should all share the same head)
            top_org_head = star_links[0].head
            top_org = {"text": top_org_head.text, "label": top_org_head.label}

            # Ordered list of sub-org names (preserve text order as they appear)
            sub_org_order: "OrderedDict[str, None]" = OrderedDict(
                (r.tail.text, None) for r in star_links
            )

            # Collect docs per sub-org (ORG→DOC_NAME)
            docs_by_org: Dict[str, List[dict]] = {}
            for r in rels:
                if r.kind == "ORG→DOC_NAME" and r.head.label == "ORG_LABEL":
                    docs_by_org.setdefault(r.head.text, []).append(
                        {"text": r.tail.text, "label": r.tail.label}
                    )

            sub_orgs: List[dict] = []
            for org_text in sub_org_order.keys():
                sub_orgs.append({
                    "org": {"text": org_text, "label": "ORG_LABEL"},
                    "docs": docs_by_org.get(org_text, [])
                })

            items.append({
                "paragraph_id": pid,
                "top_org": top_org,
                "sub_orgs": sub_orgs
            })
            continue  # next paragraph

        # Non-star paragraph: pick the first ORG→DOC_NAME head as the org
        org_doc_rels = [r for r in rels if r.kind == "ORG→DOC_NAME" and r.head.label == "ORG_LABEL"]
        if org_doc_rels:
            primary_head = org_doc_rels[0].head
            docs = [
                {"text": r.tail.text, "label": r.tail.label}
                for r in org_doc_rels
                if r.head.text == primary_head.text
            ]
            items.append({
                "paragraph_id": pid,
                "org": {"text": primary_head.text, "label": primary_head.label},
                "docs": docs
            })
            continue

        # Fallback: no ORG→DOC_NAME found (rare). Skip or capture empty shell.
        # Here we skip to keep the output clean.

    payload = {"items": items}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
