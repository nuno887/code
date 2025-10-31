from typing import Any, Dict, List

def _build_org_map(payload: Dict[str, Any]) -> Dict[int, str]:
    out: Dict[int, str] = {}
    for o in payload.get("orgs", []):
        oid = o.get("id")
        name = (o.get("text") or "").strip()
        if oid is not None:
            out[oid] = name
    if not out:
        out[-1] = "(Sem organização)"
    return out

def _group_items_by_org(payload: Dict[str, Any]) -> Dict[int, List[Dict[str, Any]]]:
    grouped: Dict[int, List[Dict[str, Any]]] = {}
    for it in payload.get("items", []):
        for oid in it.get("org_ids", []):
            grouped.setdefault(oid, []).append(it)
    return grouped
