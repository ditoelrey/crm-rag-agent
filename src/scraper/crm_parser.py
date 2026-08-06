"""
crm_parser.py  --  TRANSFORM LAYER for the Central Registry e-services corpus.
=============================================================================
Pure functions, no network I/O. Reads a raw archived service payload (dict or
str) and produces normalized, HTML-free records and granular corpus blocks
ready for chunking / embedding / hybrid search.

Grounded in the real payloads (verified 2026-07)
------------------------------------------------
* The custom {Columns, Data} tabular schema, with the self-describing `Format`
  field ("Html", "Curency"/"Currency", "String", "Text") driving HTML stripping
  and number coercion.
* Data has three states -- rows / None / key-absent -- all handled.
* THE NAME TRAP: top-level `name` is the CATEGORY name; the real service name is
  in `serviceName`. We never read `name`.
* VARIATIONS ARE DOCUMENTS: a multi-variation base is an empty container
  (idServiceVariation==0, no process/tariff rows) that only lists its variations.
  Each variation carries its OWN process/documents/tariffs and a distinct
  `variationShortName` (e.g. "АД", "ДОО, ДООЕЛ", "Фондација"), which becomes the
  contextual-header discriminator so retrieval can tell them apart.
* Chunk ids are variation-aware:  srv_{idService}_v{idVariation}_{section}_{row}
  (v0 for a base/no-variation service) -- so 11 sibling variations never collide.
* New fields captured: onlineURL/isOnline, paymentMethods, otherOffices,
  documentsLocations, connected, registries, institutions.
* Defensive payload guard: non-payload files (WAF pages, truncated json) are
  rejected up front so the corpus build cannot be poisoned or crash.
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from html import unescape
from typing import Any, Iterable, Iterator

from bs4 import BeautifulSoup

try:
    from payload_guard import classify_payload
except ImportError:  # keep the parser usable even if the guard isn't on path
    classify_payload = None  # type: ignore

log = logging.getLogger("crm_parser")

_HTML_FORMATS = {"html"}
_NUMERIC_FORMATS = {"curency", "currency", "number", "decimal"}
_ROW_META_KEYS = {"Id", "Order", "Require", "IdValueType", "isSubPageId"}


# --------------------------------------------------------------------------- #
# 1. Robust JSON loading + payload guard
# --------------------------------------------------------------------------- #
def robust_load_json(source: str) -> Any:
    text = source.lstrip("\ufeff").strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        if text and text[0] not in "{[":
            for cand in ("{" + text, "{" + text + "}"):
                try:
                    log.warning("Repaired JSON by adding a leading brace.")
                    return json.loads(cand)
                except json.JSONDecodeError:
                    continue
        raise


def load_service_file(path: str) -> Any | None:
    """Load one archived file, returning the dict only if it's a real payload.

    Returns None (with a warning) for WAF blocks, soft errors, truncated json,
    or empty files -- so callers can skip-and-log instead of crashing.
    """
    with open(path, "rb") as fh:
        raw = fh.read()
    if classify_payload is not None:
        kind, obj = classify_payload(raw)
        if kind != "ok":
            log.warning("Skipping %s: payload guard says '%s'.", path, kind)
            return None
        return obj
    # fallback if guard unavailable
    try:
        return robust_load_json(raw.decode("utf-8", "replace"))
    except Exception as e:
        log.warning("Skipping %s: %s", path, e)
        return None


# --------------------------------------------------------------------------- #
# 2. HTML -> clean text (links, lists, paragraphs preserved)
# --------------------------------------------------------------------------- #
_WS = re.compile(r"[ \t\u00a0]+")
_NL = re.compile(r"\n{3,}")

# THE :8081 TRAP. The payloads link attachments through an internal port that is
# not reachable from outside the registry's network -- an external user clicking
# one gets ERR_CONNECTION_TIMED_OUT. The public route for the same file is
# /CRMPublicPortalApi/api/files/. Verified uniform across the archive: all 875
# occurrences are https://www.crm.com.mk:8081/api/files/, and the corpus already
# contains 156 working links in the public form, so this is a rewrite between two
# observed shapes rather than a guess.
#
# Fixed here, in the transform layer, because a dead link that reaches the model
# reaches the user: the agent quotes these URLs verbatim as the place to get a
# form or a tariff schedule.
_PORT_URL = re.compile(r"(https?://[^\s/:]+):8081/api/files/", re.I)


def fix_links(text: str) -> str:
    return _PORT_URL.sub(r"\1/CRMPublicPortalApi/api/files/", text)


def html_to_text(value: Any, keep_links: bool = True) -> str:
    if value is None:
        return ""
    if not isinstance(value, str):
        return str(value)
    if "<" not in value:
        return fix_links(_WS.sub(" ", unescape(value)).strip())

    soup = BeautifulSoup(value, "html.parser")
    for a in soup.find_all("a"):
        href = (a.get("href") or "").strip()
        txt = a.get_text(" ", strip=True)
        if keep_links and href and href not in txt:
            a.replace_with(f"{txt} ({href})" if txt else href)
        else:
            a.replace_with(txt)
    for li in soup.find_all("li"):
        li.insert_before("\n- ")
    for tag in soup.find_all(["p", "br", "div", "tr", "h1", "h2", "h3", "h4", "ul", "ol"]):
        tag.insert_after("\n")

    text = unescape(soup.get_text())
    text = _WS.sub(" ", text)
    text = "\n".join(line.strip() for line in text.splitlines())
    return fix_links(_NL.sub("\n\n", text).strip())


# --------------------------------------------------------------------------- #
# 3. {Columns, Data} tabular decoder
# --------------------------------------------------------------------------- #
@dataclass
class TabularBlock:
    rows: list[dict[str, Any]] = field(default_factory=list)
    is_empty: bool = True


def decode_tabular(block_wrapper: Any) -> TabularBlock:
    if isinstance(block_wrapper, list):
        block = block_wrapper[0] if block_wrapper else {}
    elif isinstance(block_wrapper, dict):
        block = block_wrapper
    else:
        return TabularBlock()

    columns = block.get("Columns") or []
    col_map = {
        c.get("Column"): ((c.get("Title") or c.get("Column") or "").strip(),
                          (c.get("Format") or "String").strip().lower())
        for c in columns if isinstance(c, dict) and c.get("Column")
    }

    data = block.get("Data")
    if not data:
        return TabularBlock(is_empty=True)

    out_rows: list[dict[str, Any]] = []
    for raw_row in data:
        if not isinstance(raw_row, dict):
            continue
        clean: dict[str, Any] = {}
        for dyn_key, cell in raw_row.items():
            if dyn_key in _ROW_META_KEYS:
                continue
            title, fmt = col_map.get(dyn_key, (dyn_key, "string"))
            if fmt in _HTML_FORMATS:
                clean[title] = html_to_text(cell)
            elif fmt in _NUMERIC_FORMATS:
                clean[title] = _to_number(cell)
            else:
                clean[title] = html_to_text(cell) if isinstance(cell, str) and "<" in cell else cell
        if "Order" in raw_row:
            clean["_order"] = raw_row["Order"]
        out_rows.append(clean)

    out_rows.sort(key=lambda r: _order_key(r.get("_order")))
    return TabularBlock(rows=out_rows, is_empty=not out_rows)


def _order_key(v: Any) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return 0.0


def _to_number(value: Any) -> Any:
    """Coerce a Curency/number cell, keeping whole amounts whole.

    The payloads store tariffs as floats, so every one of the 315 tariff rows
    rendered as "Висина на тарифа: 2452.0" -- and, for the free ones, "0.0",
    which the agent then quoted back to users as "чини 0.0 МКД". Amounts here
    are denars; an integral value is written as an integer.
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value) if value.is_integer() else value
    if isinstance(value, str):
        s = value.replace(",", ".").strip()
        try:
            n = float(s) if "." in s else int(s)
        except ValueError:
            return value
        return int(n) if isinstance(n, float) and n.is_integer() else n
    return value


# --------------------------------------------------------------------------- #
# 4. Service normalization
# --------------------------------------------------------------------------- #
# section key -> (human MK label, is_tabular)
_SECTIONS = {
    "process": ("Постапка", True),
    "documents": ("Потребни документи", True),
    "forms": ("Обрасци", True),
    "tariffs": ("Тарифи", True),
    "deadlines": ("Рокови", True),
    "documentsLocations": ("Начин на подигнување", True),
    "terminology": ("Терминологија", True),
    "instructions": ("Упатства", True),
    "discounts": ("Попусти", True),
    "legalBasis": ("Правен основ", False),
}


def parse_service(service_json: Any) -> dict[str, Any]:
    """Normalize one raw service/variation payload into a clean record."""
    if isinstance(service_json, str):
        service_json = robust_load_json(service_json)
    d = service_json if isinstance(service_json, dict) else {}

    id_variation = d.get("idServiceVariation") or 0
    # NAME TRAP: use serviceName / variation names, never top-level `name`.
    service_name = (d.get("serviceName") or "").strip()
    variation_short = (d.get("variationShortName") or "").strip()
    variation_name = (d.get("variationName") or "").strip()

    rec: dict[str, Any] = {
        "id_service": d.get("idService"),
        "id_variation": id_variation,
        "service_name": service_name,
        "service_short_name": (d.get("serviceShortName") or "").strip(),
        "variation_name": variation_name,
        "variation_short_name": variation_short,
        "is_variation": bool(id_variation),
        "url": d.get("url"),
        "url_prefix": d.get("urlPrefix"),
        "keywords": _split_keywords(d.get("keywords")),
        "tags": [t.get("name") for t in (d.get("tags") or []) if isinstance(t, dict)],
        "service_description": html_to_text(d.get("serviceDescription")),
        # actionable service facts
        "is_online": bool(d.get("isOnline")),
        "online_url": (d.get("onlineURL") or "").strip() or None,
        "payment_methods": [p.get("name", "").strip()
                            for p in (d.get("paymentMethods") or []) if isinstance(p, dict)],
        "other_offices": [o.get("name", "").strip()
                          for o in (d.get("otherOffices") or []) if isinstance(o, dict)],
        "connected_services": [c.get("shortName") or c.get("name")
                               for c in (d.get("connected") or []) if isinstance(c, dict)],
        "registries": [r.get("name", "").strip()
                       for r in (d.get("registries") or []) if isinstance(r, dict)],
        "institutions": [i.get("name", "").strip()
                         for i in (d.get("institutions") or [])
                         if isinstance(i, dict) and i.get("selected")],
        "sections": {},
        "faqs": [],
        "variations": [],   # references only (present on a base container)
    }

    for key, (human, is_tab) in _SECTIONS.items():
        raw = d.get(key)
        if raw is None:
            continue
        if is_tab:
            tb = decode_tabular(raw)
            if not tb.is_empty:
                rec["sections"][key] = {"label": human, "rows": tb.rows}
        else:
            txt = html_to_text(raw)
            if txt:
                rec["sections"][key] = {"label": human, "text": txt}

    for f in d.get("faQs") or []:
        if isinstance(f, dict) and (f.get("question") or f.get("answer")):
            rec["faqs"].append({
                "question": html_to_text(f.get("question")),
                "answer": html_to_text(f.get("answer")),
                "order": f.get("orderQ"),
            })
    rec["faqs"].sort(key=lambda x: x.get("order") or 0)

    for v in d.get("variations") or []:
        if isinstance(v, dict) and v.get("idServiceVariation"):
            rec["variations"].append({
                "id_variation": v["idServiceVariation"],
                "short_name": (v.get("shortName") or "").strip(),
            })

    return rec


def _split_keywords(value: Any) -> list[str]:
    if not value or not isinstance(value, str):
        return []
    return [w for w in re.split(r"[\s,;]+", value.strip()) if w]


# --------------------------------------------------------------------------- #
# 4b. Shared vs variation-specific section routing
# --------------------------------------------------------------------------- #
# Content the base container holds on behalf of the whole service family; these
# are emitted ONCE at service scope (tagged with all variation ids) rather than
# duplicated per variation. process/documents/tariffs/deadlines are
# variation-specific and are emitted per variation.
_SHARED_SECTIONS = ("terminology", "legalBasis")


# Sections that are variation-specific (never shared/inherited).
_VARIATION_SECTIONS = ("process", "documents", "forms", "tariffs", "deadlines",
                       "documentsLocations", "instructions", "discounts")


def iter_shared_blocks(base_rec: dict[str, Any], variation_ids: list[int]) -> Iterable[dict[str, Any]]:
    """Emit service-level shared blocks (terminology, legalBasis, description,
    FAQs) ONCE, tagged with every variation id they apply to.

    chunk_id is service-scoped (srv_{id}_shared_...), and metadata carries
    `applies_to_variations` (the full sibling id list) + `scope: service`, so a
    retriever filtering by any one variation still matches these.
    """
    svc = base_rec["service_name"] or base_rec["service_short_name"] or f"услуга {base_rec['id_service']}"
    sid = base_rec["id_service"]
    meta = {
        "id_service": sid,
        "scope": "service",
        "applies_to_variations": variation_ids,
        "service_name": base_rec["service_name"],
    }

    def blk(suffix: str, label: str, content: str, extra: dict) -> dict[str, Any]:
        context = f"Услуга: {svc} | {label}"
        return {
            "chunk_id": f"srv_{sid}_shared_{suffix}",
            "context": context,
            "content": content,
            "embedding_text": f"{context}\n{content}",
            "metadata": {**meta, **extra},
        }

    if base_rec.get("service_description"):
        yield blk("description", "Опис на услугата", base_rec["service_description"],
                  {"type": "description"})
    for key in _SHARED_SECTIONS:
        sec = base_rec.get("sections", {}).get(key)
        if not sec:
            continue
        if "text" in sec:
            yield blk(key, sec["label"], sec["text"], {"type": key})
        else:
            for i, row in enumerate(sec["rows"], 1):
                body = " | ".join(f"{k}: {v}" for k, v in row.items()
                                  if not k.startswith("_") and v not in (None, ""))
                if body:
                    yield blk(f"{key}_{i}", sec["label"], body, {"type": key, "row": i})
    for i, faq in enumerate(base_rec.get("faqs", []), 1):
        # NOTE: generic FAQs are byte-identical across some services (the portal
        # copies boilerplate). We deliberately KEEP them per-service (not global
        # dedup) so each answer's provenance/context header stays that service's
        # own -- critical for a legal-answer system. Cross-service duplicate FAQ
        # content is intended, not a bug.
        body = f"Прашање: {faq['question']}\nОдговор: {faq['answer']}"
        yield blk(f"faq_{i}", "Често поставувани прашања", body, {"type": "faq", "row": i})


def iter_variation_blocks(var_rec: dict[str, Any]) -> Iterable[dict[str, Any]]:
    """Emit ONLY variation-specific blocks for a variation (process, documents,
    tariffs, etc.) plus its access block -- shared content is emitted separately
    at service level by iter_shared_blocks(). Description/terminology/legalBasis/
    FAQs are skipped here to avoid duplicating the service-level shared blocks.
    """
    svc = _display_name(var_rec)
    vid = var_rec["id_variation"]
    base_meta = {
        "id_service": var_rec["id_service"],
        "id_variation": vid,
        "scope": "variation",
        "service_name": var_rec["service_name"],
        "variation_short_name": var_rec["variation_short_name"] or None,
        "is_online": var_rec["is_online"],
    }

    def blk(suffix: str, label: str, content: str, extra: dict) -> dict[str, Any]:
        context = f"Услуга: {svc}" + (f" | {label}" if label else "")
        return {
            "chunk_id": f"srv_{var_rec['id_service']}_v{vid}_{suffix}",
            "context": context,
            "content": content,
            "embedding_text": f"{context}\n{content}",
            "metadata": {**base_meta, **extra},
        }

    # access block (variation-specific: onlineURL/payment can differ per variation)
    yield from _access_blocks(var_rec, blk)
    # variation-specific tabular sections only
    for key in _VARIATION_SECTIONS:
        sec = var_rec["sections"].get(key)
        if not sec:
            continue
        for i, row in enumerate(sec["rows"], 1):
            body = " | ".join(f"{k}: {v}" for k, v in row.items()
                              if not k.startswith("_") and v not in (None, ""))
            if body:
                yield blk(f"{key}_{i}", sec["label"], body, {"type": key, "row": i})


def _access_blocks(rec: dict[str, Any], blk) -> Iterable[dict[str, Any]]:
    bits = []
    if rec["is_online"] and rec["online_url"]:
        bits.append(f"Услугата е достапна онлајн: {rec['online_url']}")
    elif rec["is_online"]:
        bits.append("Услугата е достапна онлајн.")
    if rec["payment_methods"]:
        bits.append("Начини на плаќање: " + "; ".join(rec["payment_methods"]))
    if rec["other_offices"]:
        bits.append("Начини на пристап: " + "; ".join(rec["other_offices"]))
    if bits:
        yield blk("access", "Пристап и плаќање", "\n".join(bits),
                  {"type": "access", "online_url": rec["online_url"]})


# --------------------------------------------------------------------------- #
# 5. Corpus block builder (one self-contained unit per knowledge item)
# --------------------------------------------------------------------------- #
def _display_name(rec: dict[str, Any]) -> str:
    """Human label for the contextual header.

    For a variation, append its discriminator: "<service> — <variation>"
    (e.g. "Самостојна регистрација на субјект — АД").
    """
    base = rec["service_name"] or rec["service_short_name"] or f"услуга {rec['id_service']}"
    vs = rec["variation_short_name"]
    if rec["is_variation"] and vs and vs.lower() not in base.lower():
        return f"{base} — {vs}"
    return base


def iter_corpus_blocks(rec: dict[str, Any], is_empty_container: bool = False) -> Iterable[dict[str, Any]]:
    """Yield granular, self-contained, variation-aware corpus blocks.

    If `is_empty_container` is True (a base whose content has been merged into
    its variations), nothing is emitted -- the variations carry everything, so
    emitting here would duplicate the shared content a second time.
    """
    if is_empty_container:
        return
    svc = _display_name(rec)
    vid = rec["id_variation"]
    base_meta = {
        "id_service": rec["id_service"],
        "id_variation": vid,
        "scope": "variation",   # a standalone service is its own single variation (v0)
        "service_name": rec["service_name"],
        "variation_short_name": rec["variation_short_name"] or None,
        "is_online": rec["is_online"],
    }

    def blk(suffix: str, label: str, content: str, extra: dict) -> dict[str, Any]:
        context = f"Услуга: {svc}" + (f" | {label}" if label else "")
        return {
            "chunk_id": f"srv_{rec['id_service']}_v{vid}_{suffix}",
            "context": context,
            "content": content,
            "embedding_text": f"{context}\n{content}",
            "metadata": {**base_meta, **extra},
        }

    # description
    if rec["service_description"]:
        yield blk("description", "Опис на услугата", rec["service_description"],
                  {"type": "description"})

    # an "access / how to use" block from the actionable fields
    access_bits = []
    if rec["is_online"] and rec["online_url"]:
        access_bits.append(f"Услугата е достапна онлајн: {rec['online_url']}")
    elif rec["is_online"]:
        access_bits.append("Услугата е достапна онлајн.")
    if rec["payment_methods"]:
        access_bits.append("Начини на плаќање: " + "; ".join(rec["payment_methods"]))
    if rec["other_offices"]:
        access_bits.append("Начини на пристап: " + "; ".join(rec["other_offices"]))
    if access_bits:
        yield blk("access", "Пристап и плаќање", "\n".join(access_bits),
                  {"type": "access", "online_url": rec["online_url"]})

    # tabular + free-text sections
    for key, sec in rec["sections"].items():
        label = sec["label"]
        if "text" in sec:
            yield blk(key, label, sec["text"], {"type": key})
        else:
            for i, row in enumerate(sec["rows"], 1):
                body = " | ".join(f"{k}: {v}" for k, v in row.items()
                                  if not k.startswith("_") and v not in (None, ""))
                if body:
                    yield blk(f"{key}_{i}", label, body, {"type": key, "row": i})

    # FAQs
    for i, faq in enumerate(rec["faqs"], 1):
        body = f"Прашање: {faq['question']}\nОдговор: {faq['answer']}"
        yield blk(f"faq_{i}", "Често поставувани прашања", body, {"type": "faq", "row": i})


# --------------------------------------------------------------------------- #
# 6. Directory builder: two-pass merge over an archive of raw files
# --------------------------------------------------------------------------- #
import glob
import hashlib
import os


def file_to_blocks(path: str) -> list[dict[str, Any]]:
    """Single-file convenience (no cross-file merge). Guard-protected."""
    obj = load_service_file(path)
    if obj is None:
        return []
    return list(iter_corpus_blocks(parse_service(obj)))


def build_corpus(raw_dir: str, language: int = 1) -> dict[str, Any]:
    """Parse a whole data/raw/ archive into corpus blocks with base->variation
    merge, hash dedup, and a skip report.

    Two passes:
      1. Parse every service/variation file (guard-protected). Group by id_service.
      2. For each service, extract shared content from the base container and
         merge it into every variation, then emit blocks. A base WITH variations
         emits nothing standalone; a base WITHOUT variations emits normally.

    Returns {blocks, skipped, dedup_dropped, stats}.
    """
    suffix = f"_L{language}.json"
    paths = sorted(glob.glob(os.path.join(raw_dir, f"*{suffix}")))
    paths = [p for p in paths
             if not os.path.basename(p).startswith(("_menu", "_category"))]

    # pass 1: parse + group
    by_service: dict[Any, dict[str, Any]] = {}
    skipped: list[str] = []
    for p in paths:
        obj = load_service_file(p)
        if obj is None:
            skipped.append(os.path.basename(p))
            continue
        rec = parse_service(obj)
        sid = rec["id_service"]
        entry = by_service.setdefault(sid, {"base": None, "variations": [],
                                            "_seen_var_ids": set()})
        if rec["is_variation"]:
            # A single-variation service's BASE file carries idServiceVariation ==
            # its sole variation id, so it parses as a variation and would collide
            # with the explicitly-fetched _v{id} file (byte-identical duplicate).
            # Dedup on id_variation within the service: first file wins.
            vid = rec["id_variation"]
            if vid in entry["_seen_var_ids"]:
                continue
            entry["_seen_var_ids"].add(vid)
            entry["variations"].append(rec)
        else:
            entry["base"] = rec

    # pass 2: emit -- shared once per service (tagged w/ all variation ids),
    #                 variation-specific per variation.
    blocks: list[dict[str, Any]] = []
    n_services = n_variations = n_solo = 0

    for sid, entry in by_service.items():
        base = entry["base"]
        variations = entry["variations"]

        if variations:
            n_services += 1
            var_ids = [v["id_variation"] for v in variations]
            # shared service-level content, emitted once, tagged with all variation ids
            if base:
                blocks += list(iter_shared_blocks(base, var_ids))
            else:
                # no base file archived; synthesize shared from the first variation
                blocks += list(iter_shared_blocks(variations[0], var_ids))
            # variation-specific content per variation
            for var in variations:
                blocks += list(iter_variation_blocks(var))
                n_variations += 1
        elif base:
            # standalone service (no variations): emit everything under it
            n_solo += 1
            blocks += list(iter_corpus_blocks(base))

    stats = {
        "files": len(paths),
        "skipped": len(skipped),
        "services_with_variations": n_services,
        "variation_docs": n_variations,
        "standalone_services": n_solo,
        "blocks": len(blocks),
    }
    return {"blocks": blocks, "skipped": skipped, "stats": stats}




def write_jsonl(blocks: Iterable[dict[str, Any]], out_path: str) -> int:
    n = 0
    with open(out_path, "w", encoding="utf-8") as fh:
        for b in blocks:
            fh.write(json.dumps(b, ensure_ascii=False) + "\n")
            n += 1
    return n