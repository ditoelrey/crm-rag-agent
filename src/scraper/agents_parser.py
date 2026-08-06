"""
agents_parser.py  --  the registry's authorised-agent lists (.xlsx) -> corpus blocks.
=====================================================================================
Two attachments the portal publishes alongside the registration services:

  doo_tp      Овластени регистрациони агенти за упис на основање на ДОО, ДООЕЛ и ТП
  advocates   Овластени регистрациони агенти (адвокати) за упис на основање,
              промени и бришења

Both are flat tables: Општина | Назив | Адреса | Место | Телефон.
They are kept as SEPARATE block families on purpose -- the two lists carry
different scopes of authority (founding only vs founding, changes and deletions),
and merging them into one "agents near you" list would blur a real legal
distinction.

ONE BLOCK PER MUNICIPALITY, NOT PER AGENT
-----------------------------------------
3,010 one-agent blocks would grow the corpus by 51% with near-identical
boilerplate, dilute retrieval for every service question, and -- worst --
recreate the completeness failure this project has already been bitten by twice:
"агенти во Прилеп" would retrieve ten of seventy-five and the model would present
them as the list. Grouping by municipality makes each block complete by
construction.

The exception is size. Центар has 342 agents in one list and 381 in the other;
at roughly 90 characters each that is ~31 KB, well past the 8,191-token
embedding limit. Oversized municipalities are split into numbered parts that
state their own range and the true total, so a partial answer is visibly partial.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Any, Iterable, Iterator

MAX_AGENTS_PER_BLOCK = 40

LISTS = {
    "doo_tp": {
        "title": "Овластени регистрациони агенти за упис на основање на ДОО, ДООЕЛ и ТП",
        "short": "ДОО, ДООЕЛ и ТП",
        "scope_note": ("Овие агенти се овластени за упис на основање на ДОО, "
                       "ДООЕЛ и ТП."),
    },
    "advocates": {
        "title": ("Овластени регистрациони агенти (адвокати) за упис на основање, "
                  "промени и бришења"),
        "short": "адвокати",
        "scope_note": ("Овие агенти се адвокати овластени за упис на основање, "
                       "промени и бришења."),
    },
}

# NOTE: the Skopje municipality group ("агенти во Скопје" must reach all ten)
# lives in index/aliases.py as MUNICIPALITY_GROUPS -- it is a query-time concern,
# not a parsing one, and belongs next to the other vocabulary bridges.

# Two Општина cells in the advocates sheet contain leaked spreadsheet ranges
# ("ТЕТОВО+A1403:E1404"). Strip them rather than creating phantom municipalities.
_FORMULA_JUNK = re.compile(r"\s*\+\s*\$?[A-Z]+\$?\d+\s*:\s*\$?[A-Z]+\$?\d+\s*$")
_WS = re.compile(r"\s+")

# Latin letters typed inside a Cyrillic word. The advocates sheet contains
# "АЕРОДРОM" ending in Latin M (U+004D) instead of Cyrillic М (U+041C) -- to a
# reader they are the same word, to a dict key they are two municipalities, and
# a directory lookup would silently return half the agents in Аеродром. Only
# the unambiguous lookalikes are mapped.
_HOMOGLYPHS = str.maketrans({
    "A": "А", "B": "В", "C": "С", "E": "Е", "H": "Н", "J": "Ј", "K": "К",
    "M": "М", "O": "О", "P": "Р", "T": "Т", "X": "Х", "Y": "У",
})

_TRANSLIT = {
    "а": "a", "б": "b", "в": "v", "г": "g", "д": "d", "ѓ": "gj", "е": "e",
    "ж": "zh", "з": "z", "ѕ": "dz", "и": "i", "ј": "j", "к": "k", "л": "l",
    "љ": "lj", "м": "m", "н": "n", "њ": "nj", "о": "o", "п": "p", "р": "r",
    "с": "s", "т": "t", "ќ": "kj", "у": "u", "ф": "f", "х": "h", "ц": "c",
    "ч": "ch", "џ": "dj", "ш": "sh",
}


def slug(text: str) -> str:
    """Cyrillic municipality -> ascii slug, so chunk_ids stay readable in a
    citation (`agents_doo_tp_gostivar_p1`)."""
    out = []
    for ch in text.strip().lower():
        if ch in _TRANSLIT:
            out.append(_TRANSLIT[ch])
        elif ch.isalnum():
            out.append(ch)
        elif out and out[-1] != "_":
            out.append("_")
    return "".join(out).strip("_") or "unknown"


@dataclass(frozen=True)
class Agent:
    municipality: str
    name: str
    address: str
    place: str
    phone: str

    def line(self) -> str:
        bits = [self.name]
        if self.address:
            bits.append(f"адреса: {self.address}")
        if self.place and self.place.upper() != self.municipality.upper():
            bits.append(f"место: {self.place}")
        bits.append(f"телефон: {self.phone}" if self.phone
                    else "телефон: не е наведен")
        return " | ".join(bits)


def _clean(value: Any) -> str:
    return _WS.sub(" ", str(value).strip()) if value is not None else ""


def read_agents(path: str) -> list[Agent]:
    """Parse one .xlsx into cleaned rows. Sheets are padded to 256 columns and
    carry a title row above the header, so the header is located rather than
    assumed."""
    try:
        import openpyxl
    except ImportError as e:  # pragma: no cover
        raise RuntimeError("pip install openpyxl") from e

    wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
    try:
        ws = wb[wb.sheetnames[0]]
        rows = [[_clean(v) for v in r[:5]] for r in ws.iter_rows(values_only=True)]
    finally:
        wb.close()

    rows = [r for r in rows if any(r)]
    header = next((i for i, r in enumerate(rows)
                   if r and r[0].lower().startswith("општина")), None)
    if header is None:
        raise ValueError(f"{path}: no 'Општина' header row found")

    agents: list[Agent] = []
    for r in rows[header + 1:]:
        r = (r + [""] * 5)[:5]
        municipality = _FORMULA_JUNK.sub("", r[0]).strip().upper().translate(_HOMOGLYPHS)
        if not municipality or not r[1]:
            continue
        agents.append(Agent(municipality=municipality, name=r[1], address=r[2],
                            place=r[3], phone=r[4]))
    return agents


def updated_from_filename(path: str) -> str | None:
    """The lists carry their revision date in the filename
    (`..._16.7.2026.xlsx`). Worth keeping: an agent directory goes stale, and the
    answer should be able to say as of when it is true."""
    m = re.search(r"(\d{1,2})\.(\d{1,2})\.(\d{4})", os.path.basename(path))
    if not m:
        return None
    day, month, year = (int(x) for x in m.groups())
    return f"{year:04d}-{month:02d}-{day:02d}"


def _chunk(items: list[Agent], size: int) -> Iterator[list[Agent]]:
    for i in range(0, len(items), size):
        yield items[i:i + size]


def build_blocks(path: str, list_key: str, *,
                 max_agents: int = MAX_AGENTS_PER_BLOCK) -> list[dict[str, Any]]:
    """One block per (municipality, part) for a single list file."""
    if list_key not in LISTS:
        raise ValueError(f"unknown list {list_key!r}")
    meta = LISTS[list_key]
    agents = read_agents(path)
    updated = updated_from_filename(path)
    source = os.path.basename(path)

    by_municipality: dict[str, list[Agent]] = {}
    for a in agents:
        by_municipality.setdefault(a.municipality, []).append(a)

    blocks: list[dict[str, Any]] = []
    for municipality in sorted(by_municipality):
        group = sorted(by_municipality[municipality], key=lambda a: a.name)
        parts = list(_chunk(group, max_agents))
        for part_no, part in enumerate(parts, 1):
            first = (part_no - 1) * max_agents + 1
            last = first + len(part) - 1
            context = (f"{meta['title']} | Општина {municipality}")
            head = (f"Во општина {municipality} има {len(group)} овластени "
                    f"регистрациони агенти ({meta['short']}).")
            if len(parts) > 1:
                head += (f" Ова е дел {part_no} од {len(parts)} "
                         f"(агенти {first}-{last} од вкупно {len(group)}).")
            head += " " + meta["scope_note"]
            if updated:
                head += f" Списокот е ажуриран на {updated}."
            body = "\n".join(f"{first + i}. {a.line()}" for i, a in enumerate(part))
            content = f"{head}\n{body}"
            blocks.append({
                "chunk_id": f"agents_{list_key}_{slug(municipality)}_p{part_no}",
                "context": context,
                "content": content,
                "embedding_text": f"{context}\n{content}",
                "metadata": {
                    "scope": "directory",
                    "type": "agents",
                    "service_name": meta["title"],
                    "list": list_key,
                    "municipality": municipality,
                    "n_agents": len(group),
                    "part": part_no,
                    "n_parts": len(parts),
                    "source_file": source,
                    "updated": updated,
                },
            })
    return blocks


def build_all(attachments_dir: str) -> list[dict[str, Any]]:
    """Every agent list found in the attachments directory."""
    patterns = {"doo_tp": "agents_doo_tp", "advocates": "agents_advocates"}
    blocks: list[dict[str, Any]] = []
    for list_key, prefix in patterns.items():
        matches = sorted(f for f in os.listdir(attachments_dir)
                         if f.startswith(prefix) and f.endswith(".xlsx"))
        if not matches:
            continue
        # Newest revision wins if several are present.
        blocks += build_blocks(os.path.join(attachments_dir, matches[-1]), list_key)
    return blocks
