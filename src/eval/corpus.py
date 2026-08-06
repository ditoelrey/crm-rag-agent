"""
corpus.py  --  read-only view over data/jsonl/corpus.jsonl for evaluation.
==========================================================================
Loads the validated corpus into typed records plus the lookup tables the gold
set and the diagnostics need:

  by_id            chunk_id  -> Block
  by_service       id_service -> [Block]
  variations_of    id_service -> [id_variation]   (sorted, excludes shared 0)
  shared_of        id_service -> [Block]          (scope == "service")

The file's sha256 is recorded on the Corpus and stamped into every eval report:
a score is only comparable to another score computed over the same bytes.
"""
from __future__ import annotations

import hashlib
import json
import os
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Iterator

# src/scraper/data/jsonl/corpus.jsonl, resolved from this file so the CLI works
# from any working directory.
_SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_CORPUS = os.path.join(_SRC_DIR, "scraper", "data", "jsonl", "corpus.jsonl")


@dataclass(frozen=True)
class Block:
    chunk_id: str
    context: str
    content: str
    embedding_text: str
    scope: str                      # "variation" | "service" | "directory"
    type: str                       # process | documents | tariffs | faq | agents
    id_service: int | None          # None for directory (agent-list) blocks
    service_name: str
    id_variation: int | None = None            # variation scope only
    variation_short_name: str | None = None    # variation scope only
    is_online: bool | None = None              # variation scope only
    applies_to_variations: tuple[int, ...] = ()  # service scope only
    row: int | None = None
    # directory blocks only
    municipality: str | None = None
    list_key: str | None = None
    n_agents: int | None = None
    part: int | None = None
    n_parts: int | None = None
    updated: str | None = None

    @property
    def is_shared(self) -> bool:
        return self.scope == "service"

    @property
    def is_directory(self) -> bool:
        return self.scope == "directory"


@dataclass
class Corpus:
    path: str
    sha256: str
    blocks: list[Block]
    by_id: dict[str, Block] = field(default_factory=dict)
    by_service: dict[int, list[Block]] = field(default_factory=lambda: defaultdict(list))
    variations_of: dict[int, list[int]] = field(default_factory=dict)
    shared_of: dict[int, list[Block]] = field(default_factory=lambda: defaultdict(list))
    service_name: dict[int, str] = field(default_factory=dict)
    variation_name: dict[tuple[int, int], str] = field(default_factory=dict)
    by_municipality: dict[str, list[Block]] = field(
        default_factory=lambda: defaultdict(list))
    _content_index: dict[str, list[str]] | None = None

    def __len__(self) -> int:
        return len(self.blocks)

    def __iter__(self) -> Iterator[Block]:
        return iter(self.blocks)

    def identical_to(self, chunk_id: str) -> list[str]:
        """Every chunk (including this one) whose `content` is byte-identical.

        The portal copies boilerplate -- one FAQ answer and one glossary entry
        appear verbatim under 14 and 16 different services respectively. The
        parser keeps them per-service on purpose (provenance), but for grading a
        question that names no service, every copy is an equally correct answer:
        gold that pins one copy would punish a retriever for being right.
        """
        if self._content_index is None:
            idx: dict[str, list[str]] = defaultdict(list)
            for b in self.blocks:
                idx[b.content].append(b.chunk_id)
            self._content_index = idx
        return list(self._content_index.get(self.by_id[chunk_id].content, [chunk_id]))

    def of_type(self, id_service: int, type_: str, id_variation: int | None = None) -> list[Block]:
        """Blocks of one section type for a service, optionally one variation."""
        out = []
        for b in self.by_service.get(id_service, ()):
            if b.type != type_:
                continue
            if id_variation is not None and b.id_variation != id_variation:
                continue
            out.append(b)
        return out


def load(path: str = DEFAULT_CORPUS) -> Corpus:
    with open(path, "rb") as fh:
        raw = fh.read()
    digest = hashlib.sha256(raw).hexdigest()

    blocks: list[Block] = []
    for lineno, line in enumerate(raw.decode("utf-8").splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            row: dict[str, Any] = json.loads(line)
            meta = row["metadata"]
            blocks.append(Block(
                chunk_id=row["chunk_id"],
                context=row["context"],
                content=row["content"],
                embedding_text=row["embedding_text"],
                scope=meta["scope"],
                type=meta["type"],
                id_service=meta.get("id_service"),
                service_name=meta["service_name"],
                id_variation=meta.get("id_variation"),
                variation_short_name=meta.get("variation_short_name"),
                is_online=meta.get("is_online"),
                applies_to_variations=tuple(meta.get("applies_to_variations") or ()),
                row=meta.get("row"),
                municipality=meta.get("municipality"),
                list_key=meta.get("list"),
                n_agents=meta.get("n_agents"),
                part=meta.get("part"),
                n_parts=meta.get("n_parts"),
                updated=meta.get("updated"),
            ))
        except (KeyError, json.JSONDecodeError) as e:
            raise ValueError(f"{path}:{lineno}: unreadable corpus row ({e}). "
                             f"Run validate_corpus.py first.") from e

    c = Corpus(path=path, sha256=digest, blocks=blocks)
    var_ids: dict[int, set[int]] = defaultdict(set)
    for b in blocks:
        if b.chunk_id in c.by_id:
            raise ValueError(f"{path}: duplicate chunk_id {b.chunk_id!r}")
        c.by_id[b.chunk_id] = b
        if b.is_directory:
            c.by_municipality[b.municipality].append(b)
            continue
        c.by_service[b.id_service].append(b)
        c.service_name.setdefault(b.id_service, b.service_name)
        if b.is_shared:
            c.shared_of[b.id_service].append(b)
            var_ids[b.id_service].update(b.applies_to_variations)
        elif b.id_variation:
            var_ids[b.id_service].add(b.id_variation)
            if b.variation_short_name:
                c.variation_name.setdefault((b.id_service, b.id_variation),
                                            b.variation_short_name)
    c.variations_of = {sid: sorted(v) for sid, v in var_ids.items()}
    return c
