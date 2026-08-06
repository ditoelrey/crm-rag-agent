"""
fetch_services.py  --  RAW EXTRACTION LAYER for the CRM e-services corpus.
=========================================================================
Archives verbatim API responses to data/raw/. NO parsing or cleaning happens
here -- that is the transform layer's job (crm_parser.py). The only structural
inspection done is the minimum needed to expand the crawl frontier.

CONFIRMED API CHAIN (verified against the live portal, 2026-07)
---------------------------------------------------------------
  Hop 1  GET /api/init/menus?ln={lang}
           -> navigation tree. Service-catalog CATEGORY pages are the leaves
              under /uslugi/ with idTmpl == 3. The page's `idSiteMap` IS the
              service-category id.

  Hop 2  GET /api/service/category?idC={idSiteMap}&idL={lang}
           -> {idServiceCategory, name, ..., services: [{idService, name, ...}]}
              This is the enumeration step: category -> its services.

  Hop 3  GET /api/service/{idService}?idSC={category}&idV={variation}&idL={lang}&idr=
           -> the full service payload. `idV` is EMPTY for the base service and
              carries an idServiceVariation to fetch a variation. Variations are
              therefore query params on the parent, NOT standalone services.

NOTE ON HEADERS: the API rejects requests that do not look like they came from
the SPA (returns a generic "not available / ticket ID" error). Browser-like
headers (Referer/Origin/X-Requested-With) are required, not optional.

Dependencies: httpx.   Run:  python fetch_services.py --help
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import logging
import random
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("fetch")

_RETRY_STATUS = {429, 500, 502, 503, 504}

# The portal returns HTTP 200 with an error body when it dislikes a request.
# Detect it so we don't archive garbage as if it were a real payload.
_SOFT_ERROR_MARKERS = ("ticket ID", "not available at the moment", "contact support")


@dataclass
class Config:
    base_url: str = "https://www.crm.com.mk/CRMPublicPortalApi"
    menus_path: str = "/api/init/menus"
    category_path: str = "/api/service/category"      # hop 2
    service_path: str = "/api/service/{id}"           # hop 3
    language: int = 1                                 # 1=MK, 2=SQ, 3=EN                               # 1=MK, 2=SQ, 3=EN

    out_dir: Path = Path("data/raw")
    concurrency: int = 4                # modest: government host
    min_interval_s: float = 0.25        # global spacing between request starts
    timeout_s: float = 30.0
    max_retries: int = 4
    backoff_base_s: float = 1.5
    limit: int | None = None            # cap #services (smoke test)
    skip_variations: bool = False

    @property
    def headers(self) -> dict[str, str]:
        return {
            "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                           "(KHTML, like Gecko) Chrome/125.0 Safari/537.36"),
            "Accept": "application/json, text/plain, */*",
            "Accept-Language": "mk,en;q=0.9",
            "X-Requested-With": "XMLHttpRequest",
            "Referer": "https://www.crm.com.mk/",
            "Origin": "https://www.crm.com.mk",
        }


# --------------------------------------------------------------------------- #
# URL builders (param order mirrors the live SPA exactly)
# --------------------------------------------------------------------------- #
def url_menus(cfg: Config) -> str:
    return f"{cfg.base_url.rstrip('/')}{cfg.menus_path}?ln={cfg.language}"


def url_category(cfg: Config, id_category: int | str) -> str:
    return f"{cfg.base_url.rstrip('/')}{cfg.category_path}?idC={id_category}&idL={cfg.language}"


def url_service(cfg: Config, id_service: int | str, id_category: int | str,
                id_variation: int | str | None = None) -> str:
    idv = "" if id_variation in (None, 0, "0", "") else str(id_variation)
    path = cfg.service_path.format(id=id_service)
    return (f"{cfg.base_url.rstrip('/')}{path}"
            f"?idSC={id_category}&idV={idv}&idL={cfg.language}&idr=")


# --------------------------------------------------------------------------- #
# Tolerant JSON load (survives BOM / missing leading brace)
# --------------------------------------------------------------------------- #
def loads_tolerant(text: str) -> Any:
    text = text.lstrip("\ufeff").strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        if text and text[0] not in "{[":
            for cand in ("{" + text, "{" + text + "}"):
                try:
                    return json.loads(cand)
                except json.JSONDecodeError:
                    continue
        raise


def is_soft_error(body: bytes) -> bool:
    """Portal sometimes returns 200 with an error message body."""
    if len(body) > 2000:
        return False
    try:
        text = body.decode("utf-8", "replace")
    except Exception:
        return False
    return any(m.lower() in text.lower() for m in _SOFT_ERROR_MARKERS)


# --------------------------------------------------------------------------- #
# HOP 1 -- menu -> service-catalog category pages
# --------------------------------------------------------------------------- #
def harvest_menu_categories(menu_obj: Any) -> list[dict[str, Any]]:
    """Leaves under /uslugi/ with idTmpl == 3 are the service-category pages.
    The leaf's idSiteMap is the category id used as idC (hop 2) and idSC (hop 3).
    """
    out: list[dict[str, Any]] = []
    seen: set[Any] = set()

    def walk(node: Any, path: tuple = ()) -> None:
        if isinstance(node, list):
            for x in node:
                walk(x, path)
            return
        if not isinstance(node, dict):
            return
        title = node.get("title")
        here = path + ((title,) if title else ())
        kids = node.get("children") or node.get("menuItems")
        if kids:
            for c in kids:
                walk(c, here)
            return
        url = node.get("fullUrl") or ""
        if "/uslugi/" in url and node.get("idTmpl") == 3:
            cid = node.get("idSiteMap")
            if cid is not None and cid not in seen:
                seen.add(cid)
                out.append({"id_category": cid, "full_url": url,
                            "title": title, "path": " > ".join(here)})

    walk(menu_obj.get("menus", menu_obj) if isinstance(menu_obj, dict) else menu_obj)
    return out


# --------------------------------------------------------------------------- #
# HOP 2 -- category payload -> its services
# --------------------------------------------------------------------------- #
def harvest_category_services(cat_obj: Any) -> list[dict[str, Any]]:
    """Extract [{id_service, name}] from a /api/service/category response."""
    if not isinstance(cat_obj, dict):
        return []
    out = []
    for s in cat_obj.get("services") or []:
        if isinstance(s, dict) and s.get("idService"):
            out.append({"id_service": s["idService"],
                        "name": (s.get("shortName") or s.get("name") or "").strip()})
    return out


# --------------------------------------------------------------------------- #
# HOP 3 -- service payload -> its variation ids
# --------------------------------------------------------------------------- #
def harvest_variation_ids(service_obj: Any) -> list[int]:
    if not isinstance(service_obj, dict):
        return []
    out = []
    for v in service_obj.get("variations") or []:
        if isinstance(v, dict) and v.get("idServiceVariation"):
            out.append(v["idServiceVariation"])
    return out


# --------------------------------------------------------------------------- #
# Manifest -> resumability
# --------------------------------------------------------------------------- #
class Manifest:
    """Append-only log of every fetch. Enables crash-resume + idempotency."""

    def __init__(self, path: Path):
        self.path = path
        self.done: set[str] = set()          # keys already fetched OK
        self._fh = None

    def load(self) -> None:
        if not self.path.exists():
            return
        with self.path.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if rec.get("status") == "ok" and rec.get("key"):
                    self.done.add(rec["key"])
        log.info("Manifest: %d artifacts already fetched OK.", len(self.done))

    def open(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self.path.open("a", encoding="utf-8")

    def write(self, rec: dict[str, Any]) -> None:
        rec.setdefault("fetched_at", datetime.now(timezone.utc).isoformat())
        self._fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        self._fh.flush()

    def close(self) -> None:
        if self._fh:
            self._fh.close()


# --------------------------------------------------------------------------- #
# HTTP with throttle + backoff
# --------------------------------------------------------------------------- #
class Throttle:
    def __init__(self, min_interval_s: float):
        self.min = min_interval_s
        self._lock = asyncio.Lock()
        self._last = 0.0

    async def wait(self) -> None:
        async with self._lock:
            delay = self.min - (time.monotonic() - self._last)
            if delay > 0:
                await asyncio.sleep(delay)
            self._last = time.monotonic()


async def fetch_raw(client: httpx.AsyncClient, url: str, cfg: Config,
                    throttle: Throttle) -> tuple[int, bytes]:
    attempt = 0
    while True:
        attempt += 1
        await throttle.wait()
        try:
            resp = await client.get(url, timeout=cfg.timeout_s)
        except (httpx.TransportError, httpx.TimeoutException) as e:
            if attempt > cfg.max_retries:
                raise
            await _backoff(cfg, attempt, str(e))
            continue

        if resp.status_code in _RETRY_STATUS and attempt <= cfg.max_retries:
            ra = resp.headers.get("Retry-After")
            if resp.status_code == 429 and ra and ra.isdigit():
                await asyncio.sleep(float(ra))
            else:
                await _backoff(cfg, attempt, f"HTTP {resp.status_code}")
            continue

        # 200-with-error-body: back off too, it often means soft throttling
        if resp.status_code == 200 and is_soft_error(resp.content) and attempt <= cfg.max_retries:
            await _backoff(cfg, attempt, "soft error body")
            continue

        return resp.status_code, resp.content


async def _backoff(cfg: Config, attempt: int, reason: str) -> None:
    delay = cfg.backoff_base_s * (2 ** (attempt - 1))
    delay += random.uniform(0, delay * 0.25)
    log.warning("retry %d in %.1fs (%s)", attempt, delay, reason)
    await asyncio.sleep(delay)


# --------------------------------------------------------------------------- #
# Archival
# --------------------------------------------------------------------------- #
def archive(out_dir: Path, key: str, body: bytes) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{key}.json").write_bytes(body)   # VERBATIM


def sha(body: bytes) -> str:
    return hashlib.sha256(body).hexdigest()


# --------------------------------------------------------------------------- #
# The crawl
# --------------------------------------------------------------------------- #
async def crawl(cfg: Config) -> None:
    manifest = Manifest(cfg.out_dir.parent / "_manifest.jsonl")
    manifest.load()
    manifest.open()
    throttle = Throttle(cfg.min_interval_s)
    stats = {"categories": 0, "services": 0, "variations": 0, "failed": 0}

    async with httpx.AsyncClient(headers=cfg.headers, follow_redirects=True) as client:
        # ---------------- HOP 1: menu ----------------
        log.info("HOP 1: fetching menu ...")
        status, body = await fetch_raw(client, url_menus(cfg), cfg, throttle)
        archive(cfg.out_dir, f"_menu_L{cfg.language}", body)
        manifest.write({"kind": "menu", "key": f"_menu_L{cfg.language}",
                        "status": "ok" if status == 200 else "http_error",
                        "http_status": status, "bytes": len(body), "sha256": sha(body)})
        categories = harvest_menu_categories(loads_tolerant(body.decode("utf-8", "replace")))
        log.info("HOP 1: %d service-catalog categories found.", len(categories))
        if not categories:
            log.error("No categories harvested -- aborting (menu shape changed?).")
            manifest.close()
            return

        # ---------------- HOP 2: category -> services ----------------
        log.info("HOP 2: enumerating services per category ...")
        service_refs: list[dict[str, Any]] = []
        sem = asyncio.Semaphore(cfg.concurrency)

        async def do_category(cat: dict[str, Any]) -> None:
            key = f"_category_{cat['id_category']}_L{cfg.language}"
            async with sem:
                try:
                    st, bd = await fetch_raw(client, url_category(cfg, cat["id_category"]),
                                             cfg, throttle)
                except Exception as e:
                    log.error("category %s failed: %s", cat["id_category"], e)
                    stats["failed"] += 1
                    manifest.write({"kind": "category", "key": key,
                                    "status": "failed", "error": str(e)})
                    return
            archive(cfg.out_dir, key, bd)
            svcs = []
            if st == 200 and not is_soft_error(bd):
                try:
                    svcs = harvest_category_services(loads_tolerant(bd.decode("utf-8", "replace")))
                except Exception as e:
                    log.warning("category %s: parse failed (%s)", cat["id_category"], e)
            manifest.write({"kind": "category", "key": key, "id_category": cat["id_category"],
                            "status": "ok" if st == 200 else "http_error", "http_status": st,
                            "bytes": len(bd), "sha256": sha(bd), "n_services": len(svcs)})
            for s in svcs:
                service_refs.append({**s, "id_category": cat["id_category"]})
            stats["categories"] += 1
            log.info("  category %-5s %-45s -> %d services",
                     cat["id_category"], (cat["title"] or "")[:45], len(svcs))

        await asyncio.gather(*(do_category(c) for c in categories))

        # dedupe: a service can appear under more than one category
        uniq: dict[Any, dict[str, Any]] = {}
        for ref in service_refs:
            uniq.setdefault(ref["id_service"], ref)
        service_refs = list(uniq.values())
        log.info("HOP 2: %d unique services across %d categories.",
                 len(service_refs), stats["categories"])

        if cfg.limit:
            service_refs = service_refs[: cfg.limit]
            log.info("--limit active: fetching only %d services.", len(service_refs))

        # ---------------- HOP 3: service (+ variations) ----------------
        log.info("HOP 3: fetching service payloads ...")

        async def do_service(ref: dict[str, Any]) -> None:
            sid, cid = ref["id_service"], ref["id_category"]
            key = f"{sid}_L{cfg.language}"
            if key in manifest.done:
                return
            async with sem:
                try:
                    st, bd = await fetch_raw(client, url_service(cfg, sid, cid), cfg, throttle)
                except Exception as e:
                    log.error("service %s failed: %s", sid, e)
                    stats["failed"] += 1
                    manifest.write({"kind": "service", "key": key, "id_service": sid,
                                    "status": "failed", "error": str(e)})
                    return
            archive(cfg.out_dir, key, bd)
            variations: list[int] = []
            if st == 200 and not is_soft_error(bd):
                try:
                    variations = harvest_variation_ids(loads_tolerant(bd.decode("utf-8", "replace")))
                except Exception as e:
                    log.warning("service %s: variation harvest failed (%s)", sid, e)
            manifest.write({"kind": "service", "key": key, "id_service": sid,
                            "id_category": cid,
                            "status": "ok" if st == 200 and not is_soft_error(bd) else "error",
                            "http_status": st, "bytes": len(bd), "sha256": sha(bd),
                            "variations": variations})
            stats["services"] += 1
            log.info("  service %-6s %-40s (%d bytes, %d variations)",
                     sid, (ref.get("name") or "")[:40], len(bd), len(variations))

            if cfg.skip_variations:
                return
            # variations are query params on THIS service, not separate services
            for vid in variations:
                vkey = f"{sid}_v{vid}_L{cfg.language}"
                if vkey in manifest.done:
                    continue
                async with sem:
                    try:
                        vst, vbd = await fetch_raw(client, url_service(cfg, sid, cid, vid),
                                                   cfg, throttle)
                    except Exception as e:
                        log.error("variation %s/%s failed: %s", sid, vid, e)
                        stats["failed"] += 1
                        manifest.write({"kind": "variation", "key": vkey, "id_service": sid,
                                        "id_variation": vid, "status": "failed", "error": str(e)})
                        continue
                archive(cfg.out_dir, vkey, vbd)
                manifest.write({"kind": "variation", "key": vkey, "id_service": sid,
                                "id_variation": vid, "id_category": cid,
                                "status": "ok" if vst == 200 and not is_soft_error(vbd) else "error",
                                "http_status": vst, "bytes": len(vbd), "sha256": sha(vbd)})
                stats["variations"] += 1
                log.info("    variation %s/%s (%d bytes)", sid, vid, len(vbd))

        await asyncio.gather(*(do_service(r) for r in service_refs))

    manifest.close()
    log.info("DONE. categories=%d services=%d variations=%d failed=%d",
             stats["categories"], stats["services"], stats["variations"], stats["failed"])


# --------------------------------------------------------------------------- #
def parse_args() -> Config:
    p = argparse.ArgumentParser(description="CRM e-services raw extraction crawler.")
    p.add_argument("--base-url", default=Config.base_url)
    p.add_argument("--out", type=Path, default=Config.out_dir, dest="out_dir")
    p.add_argument("--concurrency", type=int, default=Config.concurrency)
    p.add_argument("--language", type=int, default=Config.language, help="1=MK 2=SQ 3=EN")
    p.add_argument("--limit", type=int, default=None, help="cap #services (smoke test)")
    p.add_argument("--min-interval", type=float, default=Config.min_interval_s,
                   dest="min_interval_s")
    p.add_argument("--skip-variations", action="store_true")
    a = p.parse_args()
    return Config(base_url=a.base_url, out_dir=a.out_dir, concurrency=a.concurrency,
                  language=a.language, limit=a.limit, min_interval_s=a.min_interval_s,
                  skip_variations=a.skip_variations)


if __name__ == "__main__":
    asyncio.run(crawl(parse_args()))