# debug.py
from __future__ import annotations
import sys, re, textwrap
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Set

# Fallbacks if utils_text isn’t available yet
try:
    from .utils_text import _normalize_title, _tighten, _letters_only
except Exception:
    def _normalize_title(s: str) -> str: return " ".join(str(s).split())
    def _tighten(s: str) -> str: return str(s).replace(" ", "")
    def _letters_only(s: str) -> str: return "".join(ch for ch in str(s) if ch.isalpha())

def _preview(s: str, n: int = 80) -> str:
    s = s.replace("\n", " ⏎ ")
    return (s[: n - 1] + "…") if len(s) > n else s

@dataclass
class _Color:
    dim: str = "\x1b[2m"; bold: str = "\x1b[1m"
    blue: str = "\x1b[34m"; green: str = "\x1b[32m"
    yellow: str = "\x1b[33m"; red: str = "\x1b[31m"
    reset: str = "\x1b[0m"
C = _Color()

class _Debug:
    def __init__(self):
        self._enabled = False
        self._indent = 0
        # output controls
        self._stream = sys.stderr
        self._open_file = None
        self._color = True
        self._ansi_re = re.compile(r"\x1b\[[0-9;]*m")

    # ---- output controls ----
    def set_output_file(self, path: str, append: bool = False) -> None:
        if self._open_file:
            try: self._open_file.close()
            except Exception: pass
        mode = "a" if append else "w"
        f = open(path, mode, encoding="utf-8")
        self._open_file, self._stream = f, f

    def set_stream(self, stream) -> None:
        self._stream = stream

    def use_color(self, on: bool = True) -> None:
        self._color = bool(on)

    def close(self) -> None:
        if self._open_file:
            try: self._open_file.close()
            finally:
                self._open_file = None
                self._stream = sys.stderr

    # ---- toggles ----
    def enable(self, on: bool = True) -> None:
        self._enabled = bool(on)

    def is_enabled(self) -> bool:
        return self._enabled

    # ---- printing ----
    def _p(self, msg: str = "") -> None:
        if not self._enabled: return
        indent = "  " * self._indent
        out = indent + msg
        if not self._color:
            out = self._ansi_re.sub("", out)
        print(out, file=self._stream)

    @contextmanager
    def section(self, title: str):
        if not self._enabled:
            yield; return
        self._p(f"{C.bold}▶ {title}{C.reset}")
        self._indent += 1
        try:
            yield
        finally:
            self._indent = max(0, self._indent - 1)
            self._p(f"{C.dim}▲ end {title}{C.reset}")

    # ---- high-level overviews ----
    def payload_overview(self, payload: dict) -> None:
        if not self._enabled: return
        orgs = payload.get("orgs", []) or []
        items = payload.get("items", []) or []
        per_org = {}
        for it in items:
            for oid in (it.get("org_ids") or []):
                per_org[oid] = per_org.get(oid, 0) + 1
        self._p(f"Payload: orgs={len(orgs)} items={len(items)} items_by_org={per_org}")

    def org_windows(self, org_windows: Sequence[dict], doc_text: str, max_preview: int = 60) -> None:
        if not self._enabled: return
        self._p(f"Org windows found: {len(org_windows)}")
        for i, w in enumerate(org_windows):
            seg = _preview(doc_text[w["start"]: w["end"]], max_preview)
            self._p(f"  [{i}] {w['start']}–{w['end']} :: {seg}")

    def doc_type_matches(self, matches: dict) -> None:
        if not self._enabled: return
        self._p("Doc type header matches:")
        for k, mt in matches.items():
            if mt is None:
                self._p(f"  {k}: None")
            else:
                self._p(f"  {k}: start={mt.get('start')} end={mt.get('end')} "
                        f"win={mt.get('window_index')} conf={mt.get('confidence')}")

    def item_anchor(self, org: str, item: dict, status: str, mt: Optional[dict]) -> None:
        if not self._enabled: return
        t = (item.get("doc_name") or {}).get("text") or ""
        pid = item.get("paragraph_id")
        key = f"pid={pid} title={_preview(_normalize_title(t), 60)}"
        if mt is None:
            self._p(f"Item anchor {C.red}UNANCHORED{C.reset}: {key} (status={status})")
        else:
            self._p(f"Item anchor OK: {key} → start={mt.get('start')} end={mt.get('end')} "
                    f"win={mt.get('window_index')} conf={mt.get('confidence')}")

    def slice_bounds(self, header_end: int, content_start: int, content_end: int, seg_text: str) -> None:
        if not self._enabled: return
        self._p(f"Slice: header_end={header_end} content=[{content_start}:{content_end}] "
                f"len={len(seg_text)} preview=\"{_preview(seg_text, 80)}\"")

    # ---- entity-level ----
    def entities_stream(self, doc, prefix: str = "") -> None:
        if not self._enabled: return
        ents = list(getattr(doc, "ents", []) or [])
        self._p(f"{prefix} ents: {len(ents)} total")
        counts = {}
        for e in ents:
            label = getattr(e, "label_", "?")
            counts[label] = counts.get(label, 0) + 1
        self._p(f"label counts: {counts}")
        for e in ents:
            label = getattr(e, "label_", "?")
            txt = _preview(e.text, 120)
            self._p(f"  {label:<18} [{e.start_char}:{e.end_char}] :: {txt}")

    # ---- allowed titles ----
    def allowed_titles(self, allowed: Set[str]) -> None:
        if not self._enabled: return
        vals = sorted(allowed)
        self._p(f"Allowed titles ({len(vals)}):")
        for t in vals:
            n = _normalize_title(t)
            self._p(f"  - orig=\"{_preview(t, 80)}\" | norm=\"{_preview(n, 80)}\" "
                    f"| tight=\"{_tighten(n)}\" | letters=\"{_letters_only(n)}\"")

    # ---- header block lifecycle ----
    def header_block_start(self, e) -> None:
        if not self._enabled: return
        self._p(f"HB start @ {e.start_char}: {C.blue}{_preview(e.text, 120)}{C.reset}")

    def header_block_append(self, prev_e, e, gap: int) -> None:
        if not self._enabled: return
        warn = "" if gap <= 10 else f" {C.yellow}WARN gap={gap}{C.reset}"
        self._p(f"HB append @ {e.start_char}: +DOC_NAME_LABEL (gap={gap}).{warn}")

    def header_block_close(self, start: int, end: int, titles: Sequence[str]) -> None:
        if not self._enabled: return
        joined = ", ".join(_preview(_normalize_title(t), 50) for t in titles)
        self._p(f"HB close [{start}:{end}] titles=[{joined}]")

    def approved_block(self, idx: int, titles: Sequence[str], canonical: Optional[str]) -> None:
        if not self._enabled: return
        joined = ", ".join(_preview(_normalize_title(t), 50) for t in titles)
        canon = _preview(str(canonical), 60) if canonical else None
        self._p(f"APPROVED[{idx}]: titles=[{joined}] → canonical={canon}")

    # ---- subslices ----
    def subslice(self, idx: int, header_end: int, next_start: int, seg_text: str) -> None:
        if not self._enabled: return
        body = seg_text[header_end: next_start]
        self._p(f"SubSlice[{idx}]: body=[{header_end}:{next_start}] len={len(body)} "
                f"preview=\"{_preview(body, 100)}\"")

    def subslice_fallback(self, body_start: int, body_end: int, seg_text: str, headers_texts: Sequence[str]) -> None:
        if not self._enabled: return
        body = seg_text[body_start: body_end]
        self._p(f"FALLBACK SubSlice: headers={list(headers_texts)} body=[{body_start}:{body_end}] "
                f"len={len(body)} preview=\"{_preview(body, 100)}\"")

# Singleton
DBG = _Debug()
__all__ = ["DBG"]
