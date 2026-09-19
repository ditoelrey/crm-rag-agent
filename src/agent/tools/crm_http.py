"""
crm_http.py  --  shared HTTP session for the ЦРРСМ public WebForms.
===================================================================
The portal is ASP.NET WebForms, which makes the request shape non-obvious: every
POST must echo back the __VIEWSTATE / __VIEWSTATEGENERATOR / __EVENTVALIDATION
tokens issued by the GET that preceded it, on the same cookie jar. A bare field
POST does not error -- it returns the page unchanged, with an empty result span.
That failure is indistinguishable from "entity not found" unless you look for
it, so the handshake is not an optimisation and its absence fails silently.

Forms 2-4 talk to the same stack and will reuse this.

Politeness is deliberate and matches the corpus scraper: this is a government
host, one request at a time, browser-like headers because the portal rejects
anything that does not look like it came from its own SPA.
"""
from __future__ import annotations

import httpx
from bs4 import BeautifulSoup

TIMEOUT_S = 20.0
HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                   "(KHTML, like Gecko) Chrome/125.0 Safari/537.36"),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "mk,en;q=0.9",
    "Origin": "https://e-submit.crm.com.mk",
    "Content-Type": "application/x-www-form-urlencoded",
}

# Echoed back verbatim when present. __EVENTTARGET/__EVENTARGUMENT are usually
# empty strings but the page still expects the keys.
_HIDDEN = ("__VIEWSTATE", "__VIEWSTATEGENERATOR", "__EVENTVALIDATION",
           "__EVENTTARGET", "__EVENTARGUMENT", "__LASTFOCUS")


class WebFormsError(RuntimeError):
    """The portal did not behave like the form we expect.

    Raised rather than returning an empty result, so a markup change or a
    rejected handshake can never be mistaken downstream for a legitimate
    "no such entity" answer.
    """


def hidden_fields(page: BeautifulSoup) -> dict[str, str]:
    """The WebForms state tokens carried by a rendered page."""
    out: dict[str, str] = {}
    for name in _HIDDEN:
        tag = page.find("input", {"name": name})
        if tag is not None:
            out[name] = tag.get("value", "")
    return out


def post_webform(url: str, fields: dict[str, str], *,
                 client: httpx.Client | None = None) -> BeautifulSoup:
    """GET the form, replay its hidden tokens, POST `fields`, return the result.

    One client across both requests so the ASP.NET session cookie survives.
    `client` is injectable so tests can drive this from a transport stub without
    touching the network.
    """
    own = client is None
    client = client or httpx.Client(timeout=TIMEOUT_S,
                                    headers={**HEADERS, "Referer": url},
                                    follow_redirects=True)
    try:
        first = client.get(url)
        first.raise_for_status()
        page = BeautifulSoup(first.text, "html.parser")

        payload = hidden_fields(page)
        if "__VIEWSTATE" not in payload:
            raise WebFormsError(
                f"no __VIEWSTATE at {url} -- the page served is not the WebForm "
                f"this tool was written against")

        payload.update(fields)
        result = client.post(url, data=payload)
        result.raise_for_status()
        return BeautifulSoup(result.text, "html.parser")
    finally:
        if own:
            client.close()
