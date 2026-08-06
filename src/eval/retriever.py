"""
retriever.py  --  the one interface the harness knows about.
============================================================
Any retrieval stack -- BM25, dense vectors, hybrid + reranker, an HTTP call to a
hosted index -- is evaluable here the moment it can answer:

    search(query: str, k: int) -> ranked chunk_ids

`search` may return plain chunk_id strings or `Hit(chunk_id, score)`; the harness
normalizes either. Scores are recorded but never scored on -- only the order
matters -- so a retriever with no meaningful score can return zeros.

Contract
--------
* Return at most k items, best first, no duplicate chunk_ids.
* Every returned chunk_id must exist in the corpus being evaluated (the harness
  checks this and fails loudly: a retriever that returns ids from a stale index
  would otherwise just look bad rather than broken).
* `name` should identify the *configuration*, not the class -- "bm25(k1=1.2,
  b=0.75,stem)" -- because it is what report files are keyed and compared by.
"""
from __future__ import annotations

from typing import Iterable, NamedTuple, Protocol, Sequence, runtime_checkable


class Hit(NamedTuple):
    chunk_id: str
    score: float = 0.0


@runtime_checkable
class Retriever(Protocol):
    name: str

    def search(self, query: str, k: int) -> Sequence[Hit | str]:
        ...


def normalize_hits(raw: Iterable[Hit | str | tuple], k: int) -> list[Hit]:
    """Coerce a retriever's output to deduplicated Hits, truncated to k."""
    out: list[Hit] = []
    seen: set[str] = set()
    for item in raw:
        if isinstance(item, Hit):
            hit = item
        elif isinstance(item, str):
            hit = Hit(item, 0.0)
        elif isinstance(item, tuple) and len(item) == 2:
            hit = Hit(str(item[0]), float(item[1]))
        else:
            raise TypeError(f"retriever returned unsupported item {item!r}")
        if hit.chunk_id in seen:
            continue
        seen.add(hit.chunk_id)
        out.append(hit)
        if len(out) >= k:
            break
    return out
