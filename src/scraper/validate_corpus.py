"""
validate_corpus.py  --  STRICT validation gate for the CRM corpus JSONL.
========================================================================
Validates every block against a Pydantic schema BEFORE it reaches the vector DB.
Fail-fast: the first invalid row aborts the whole build with a precise error, so
a malformed chunk can never silently become a wrong legal answer.

The metadata schema is a tagged union on `scope`:
  * scope == "variation": a variation-specific block (process, tariffs, ...).
      carries id_variation, variation_short_name, is_online.
  * scope == "service":   a service-level shared block (terminology, legalBasis,
      description, faq). carries applies_to_variations (list of sibling ids).

Run:  python validate_corpus.py data/jsonl/corpus.jsonl
Exit code 0 = all rows valid; 1 = validation failed (message identifies the row).
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from typing import Literal, Union

from pydantic import BaseModel, Field, ValidationError, field_validator, model_validator

# The section/content types the emitter can produce.
_ALLOWED_TYPES = {
    "description", "terminology", "legalBasis", "faq", "access",
    "process", "documents", "forms", "tariffs", "deadlines",
    "documentsLocations", "instructions", "discounts",
    "agents",          # scope == "directory"
}


class VariationMeta(BaseModel, extra="forbid"):
    scope: Literal["variation"]
    id_service: int
    id_variation: int
    type: str
    service_name: str = Field(min_length=1)
    variation_short_name: str | None = None
    is_online: bool
    row: int | None = None
    online_url: str | None = None

    @field_validator("type")
    @classmethod
    def _known_type(cls, v: str) -> str:
        if v not in _ALLOWED_TYPES:
            raise ValueError(f"unknown block type {v!r}")
        return v


class ServiceMeta(BaseModel, extra="forbid"):
    scope: Literal["service"]
    id_service: int
    type: str
    service_name: str = Field(min_length=1)
    applies_to_variations: list[int] = Field(default_factory=list)
    row: int | None = None

    @field_validator("type")
    @classmethod
    def _known_type(cls, v: str) -> str:
        if v not in _ALLOWED_TYPES:
            raise ValueError(f"unknown block type {v!r}")
        return v


class DirectoryMeta(BaseModel, extra="forbid"):
    """Reference data attached to the registry rather than to one service: the
    authorised-agent lists, keyed by municipality.

    Deliberately NOT modelled as a pseudo-service. These rows answer "who can
    file this for me near me", which is a directory lookup on `municipality`,
    and forcing them into the service schema would discard the one field that
    makes the lookup possible.

    `service_name` carries the list title so every downstream consumer -- the
    context renderer, the source list, the eval Block -- keeps working without
    a special case.
    """
    scope: Literal["directory"]
    type: str
    service_name: str = Field(min_length=1)
    list: str = Field(min_length=1)
    municipality: str = Field(min_length=1)
    n_agents: int = Field(ge=1)
    part: int = Field(ge=1)
    n_parts: int = Field(ge=1)
    source_file: str = Field(min_length=1)
    updated: str | None = None

    @field_validator("type")
    @classmethod
    def _known_type(cls, v: str) -> str:
        if v not in _ALLOWED_TYPES:
            raise ValueError(f"unknown block type {v!r}")
        return v

    @model_validator(mode="after")
    def _parts(self) -> "DirectoryMeta":
        if self.part > self.n_parts:
            raise ValueError(f"part {self.part} of {self.n_parts}")
        return self


class Block(BaseModel, extra="forbid"):
    chunk_id: str = Field(min_length=1)
    context: str = Field(min_length=1)
    content: str = Field(min_length=1)
    embedding_text: str = Field(min_length=1)
    metadata: Union[VariationMeta, ServiceMeta, DirectoryMeta] = Field(
        discriminator="scope")

    @model_validator(mode="after")
    def _consistency(self) -> "Block":
        # chunk_id must encode the right service id (directory blocks have none)
        sid = getattr(self.metadata, "id_service", None)
        if sid is not None and f"srv_{sid}_" not in self.chunk_id:
            raise ValueError(
                f"chunk_id {self.chunk_id!r} does not match id_service {sid}")
        # embedding_text must actually contain the content (guards a builder bug)
        if self.content not in self.embedding_text:
            raise ValueError("embedding_text does not contain content")
        return self


def validate_file(path: str) -> dict:
    """Validate every line. Raises SystemExit(1) on the first invalid row."""
    seen_ids: set[str] = set()
    type_counts: Counter[str] = Counter()
    scope_counts: Counter[str] = Counter()
    identical_names = 0
    n = 0

    with open(path, encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            n += 1
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as e:
                _abort(lineno, f"invalid JSON: {e}")
            try:
                blk = Block.model_validate(raw)
            except ValidationError as e:
                _abort(lineno, f"schema error in chunk_id={raw.get('chunk_id')!r}:\n{e}")

            if blk.chunk_id in seen_ids:
                _abort(lineno, f"duplicate chunk_id {blk.chunk_id!r}")
            seen_ids.add(blk.chunk_id)

            type_counts[blk.metadata.type] += 1
            scope_counts[blk.metadata.scope] += 1
            m = blk.metadata
            if isinstance(m, VariationMeta) and m.variation_short_name and \
                    m.variation_short_name.strip().lower() == m.service_name.strip().lower():
                identical_names += 1

    return {
        "rows": n,
        "unique_chunk_ids": len(seen_ids),
        "by_scope": dict(scope_counts),
        "by_type": dict(type_counts.most_common()),
        "single_form_variations": identical_names,  # data-quality signal, not an error
    }


def _abort(lineno: int, msg: str) -> None:
    print(f"\n VALIDATION FAILED at line {lineno}:\n  {msg}\n", file=sys.stderr)
    print("Corpus NOT written to the index. Fix the transform and re-run.", file=sys.stderr)
    sys.exit(1)


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "data/jsonl/corpus.jsonl"
    report = validate_file(path)
    print("VALIDATION PASSED")
    for k, v in report.items():
        print(f"  {k}: {v}")