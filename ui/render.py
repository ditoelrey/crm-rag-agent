"""
render.py  --  turning an Answer into what the screen shows.
============================================================
Pure functions over the Answer dataclass: no Chainlit import, no I/O, so this
module can be exercised from a plain Python prompt while the server is not
running -- which is how you check the citation rendering without paying for a
model call.

WHY CITATIONS GET THIS MUCH ATTENTION
-------------------------------------
Every factual sentence the agent produces carries a chunk_id, and an answer that
cites nothing is replaced outright by the abstention guard. The ids are not
decoration, they are the claim's evidence, so the UI shows what each one IS
rather than printing an opaque token like srv_2135_v11116_documents_2.
"""
from __future__ import annotations

import re

# A bracketed token in the answer: [srv_2135_v11116_documents_2] or
# [tool:entity_size:07696876]. Non-greedy and bracket-free inside, so two
# adjacent citations "[a] [b]" are matched separately.
_CITE_RE = re.compile(r"\[([^\[\]]+)\]")

# A markdown heading line, and the fence that means "leave this alone".
_HEADING_RE = re.compile(r"^\s{0,3}(#{1,6})\s+(.*?)\s*#*\s*$")
_FENCE_RE = re.compile(r"^\s*```")

# The agent's own labels for section types, in the user's language.
_TYPE_LABELS = {
    "process": "постапка",
    "documents": "документи",
    "documentsLocations": "каде се поднесува",
    "tariffs": "тарифа",
    "deadlines": "рокови",
    "forms": "обрасци",
    "faq": "често поставувано прашање",
    "terminology": "поими",
    "legalBasis": "правен основ",
    "access": "пристап",
    "instructions": "упатство",
    "agents": "регистрациони агенти",
    "description": "опис",
    "live_lookup": "проверка во живо",
}


def is_live(chunk_id: str) -> bool:
    """A synthetic id minted by a tool, rather than a corpus block."""
    return chunk_id.startswith("tool:")


def source_label(doc) -> str:
    """A human title for one cited document.

    Live results say so explicitly: a user should never have to work out that
    one line of an answer came from a lookup made seconds ago while the rest
    came from documentation.
    """
    block = doc.block
    kind = _TYPE_LABELS.get(block.type, block.type)
    if is_live(doc.chunk_id):
        return f"🔎 {block.service_name} · {kind}"
    variant = block.variation_short_name
    tail = f" · {variant}" if variant else ""
    return f"📄 {block.service_name} · {kind}{tail}"


def source_body(doc) -> str:
    """The panel shown when a citation is clicked."""
    block = doc.block
    lines = [f"**{source_label(doc)}**", ""]
    if block.context:
        lines += [f"_{block.context}_", ""]
    lines += [block.content, "", "---", f"`{doc.chunk_id}`"]
    if doc.also_in:
        lines.append(f"\nИстиот текст постои и кај: {', '.join(doc.also_in[:5])}")
    return "\n".join(lines)


def demote_headings(text: str) -> str:
    """Turn markdown headings into bold lines.

    The model occasionally closes an answer with "## Ова не е комплетна листа
    ...", which a chat bubble renders at banner size -- the quietest sentence
    in the answer shouting the loudest. Nothing the agent writes is a document
    heading; it is all one message, so every level is demoted rather than only
    the trailing one. Chat bubbles have no use for h1.

    Fenced code is skipped, so a `# comment` inside a block stays a comment.
    """
    out, in_fence = [], False
    for line in text.splitlines():
        if _FENCE_RE.match(line):
            in_fence = not in_fence
        elif not in_fence:
            heading = _HEADING_RE.match(line)
            if heading and heading.group(2).strip():
                line = f"**{heading.group(2).strip()}**"
        out.append(line)
    return "\n".join(out)


def citation_labels(answer) -> dict[str, str]:
    """chunk_id -> the short marker shown in the text, numbered as they appear.

    The raw ids are precise and unreadable: an answer carrying nine of them
    reads like a stack trace. They are replaced in the prose by [Извор 1] and
    kept verbatim in the source list and the side panel, so nothing about the
    provenance is lost -- the id is still one click or one glance away.

    The brackets are PART of the label, not decoration. Chainlit turns an
    element into a link by finding its name as a substring of the message, and
    a bare "Извор 1" is a substring of "Извор 10" -- with ten or more sources
    the first would swallow the tenth. "[Извор 1]" cannot prefix "[Извор 10]".
    """
    cited = {doc.chunk_id for doc in answer.cited_docs}
    order: list[str] = []
    for match in _CITE_RE.finditer(answer.text):
        cid = match.group(1).strip()
        if cid in cited and cid not in order:
            order.append(cid)
    # Cited but never bracketed in the prose: still deserves a number.
    for doc in answer.cited_docs:
        if doc.chunk_id not in order:
            order.append(doc.chunk_id)
    return {cid: f"[Извор {n}]" for n, cid in enumerate(order, start=1)}


def linkify(text: str, labels: dict[str, str]) -> str:
    """Swap raw ids for their markers, leaving anything else alone.

    An unknown token stays exactly as written. That matters: a citation the
    agent invented is reported as invalid elsewhere, and quietly renumbering it
    would disguise the one failure this system most needs to stay visible.
    """
    return _CITE_RE.sub(
        lambda m: labels.get(m.group(1).strip(), m.group(0)), text)


def elements(answer, labels: dict[str, str]) -> list[tuple[str, str]]:
    """(name, body) per cited document, for app.py to turn into cl.Text.

    Returned as plain tuples so this module stays Chainlit-free and testable.
    """
    by_id = {doc.chunk_id: doc for doc in answer.cited_docs}
    # In label order. `labels` is built in numbering order, and iterating
    # cited_docs instead would list the sources 1, 2, 5, 6 ... 4, 3.
    return [(label, source_body(by_id[cid]))
            for cid, label in labels.items() if cid in by_id]


def prepare(answer) -> tuple[str, list[tuple[str, str]]]:
    """The message body and its side-panel elements, numbered consistently.

    No source list under the answer: every [Извор N] in the prose is itself the
    link to that source's panel, and repeating all of them underneath buried
    the answer. The raw chunk_id still appears inside each panel, so the
    provenance is one click away rather than printed twice.
    """
    labels = citation_labels(answer)
    body = (linkify(demote_headings(answer.text), labels)
            + warnings_section(answer) + footer(answer))
    return body, elements(answer, labels)


def warnings_section(answer) -> str:
    """Stale or invalid citations, shown rather than hidden.

    The agent already distinguishes them; surfacing them is what makes the
    citation discipline visible instead of a claim in a thesis chapter.
    """
    out = []
    if answer.stale_citations:
        out.append("⚠️ Цитати од претходен чекор: "
                   + ", ".join(f"`{c}`" for c in answer.stale_citations))
    if answer.invalid_citations:
        out.append("⛔ Непостоечки цитати: "
                   + ", ".join(f"`{c}`" for c in answer.invalid_citations))
    return ("\n\n" + "\n\n".join(out)) if out else ""


def footer(answer) -> str:
    """One dim line of provenance: what it cost and where it went.

    Worth showing in a defence. A clarification turn reports 0 tokens and $0.00
    because the ambiguity detector answers it structurally, without the model --
    that is visible here and nowhere else.
    """
    bits = [f"{answer.model}"]
    if answer.tool_calls:
        bits.append("алатки: " + " → ".join(answer.tool_calls))
    bits.append(f"{answer.prompt_tokens + answer.completion_tokens} токени")
    bits.append(f"${answer.cost_usd:.4f}")
    bits.append(f"пребарување {answer.retrieval_ms:.0f} ms")
    if answer.generation_ms:
        bits.append(f"генерирање {answer.generation_ms / 1000:.1f} s")
    return "\n\n`" + " · ".join(bits) + "`"


def compose(answer) -> str:
    """The full message body for one answer, without the element pairing."""
    return prepare(answer)[0]
