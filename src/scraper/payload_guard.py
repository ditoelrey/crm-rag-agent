"""
payload_guard.py  --  shared gate that decides whether an archived response is a
real service payload or a block/error page. Imported by BOTH the extractor
(fetch_services.py, to mark blocked fetches as retryable) and the transform
layer (crm_parser / build_jsonl, to skip poisoned files instead of crashing).

Classifications
---------------
  ok         -> a genuine service payload (dict with idService); returns the obj
  waf        -> firewall block page ("Access Denied", Signature ID, HTML body)
  soft_error -> portal "ticket ID / contact support" 200-with-error-body
  bad_json   -> not parseable, or JSON without idService (e.g. truncated)
  empty      -> zero-length / whitespace-only file
Anything other than `ok` should NOT be embedded, and (in the extractor) should be
treated as a transient failure worth retrying, not a success.
"""
from __future__ import annotations

import json
from typing import Any

# Firewall / block-page fingerprints. `<` catch covers any HTML error page.
_WAF_MARKERS = ("access denied", "signature id", "request rejected",
                "request unsuccessful", "blocked", "<html")
# Portal application-layer soft error (returned with HTTP 200).
_SOFT_MARKERS = ("ticket id", "not available at the moment", "contact support")


def classify_payload(raw: bytes) -> tuple[str, Any]:
    """Return (kind, obj). obj is the parsed dict only when kind == 'ok'."""
    if not raw or not raw.strip():
        return ("empty", None)
    text = raw.decode("utf-8", "replace").lstrip("\ufeff").strip()
    head = text[:600].lower()

    # HTML / firewall block pages
    if head.startswith("<") or any(m in head for m in _WAF_MARKERS):
        return ("waf", None)
    # portal soft-error message (short, non-JSON)
    if len(text) < 2000 and any(m in head for m in _SOFT_MARKERS):
        return ("soft_error", None)

    # tolerant parse (handles the leading-brace / BOM quirk from hand-saved files)
    try:
        obj = json.loads(text if text[:1] in "{[" else "{" + text)
    except json.JSONDecodeError:
        return ("bad_json", None)

    if isinstance(obj, dict) and obj.get("idService"):
        return ("ok", obj)
    return ("bad_json", None)


def is_blocked(raw: bytes) -> bool:
    """True if the response is a firewall/soft-error block (retryable)."""
    return classify_payload(raw)[0] in ("waf", "soft_error")