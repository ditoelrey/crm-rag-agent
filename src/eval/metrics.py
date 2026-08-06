"""
metrics.py  --  ranking metrics with graded relevance.
======================================================
Pure functions over (ranked chunk_ids, gold {chunk_id: grade}). No corpus, no
retriever, no I/O -- so they can be unit-checked against hand-computed values
(see selftest.py). A silent metric bug invalidates every number the harness
ever prints, so this module is the one place that gets known-answer tests.

Grades
------
  2  primary   -- the block that actually answers the query
  1  acceptable-- same service/variation, useful context (shared terminology,
                  legal basis, a sibling row of the same section)
  0  irrelevant

recall / hit / MRR count only grade-2 items; nDCG uses the full grade scale with
the standard gain 2^g - 1.
"""
from __future__ import annotations

from math import log2
from typing import Mapping, Sequence

PRIMARY = 2
ACCEPTABLE = 1


def _primary(gold: Mapping[str, int]) -> set[str]:
    return {cid for cid, g in gold.items() if g >= PRIMARY}


def hit_at_k(ranked: Sequence[str], gold: Mapping[str, int], k: int) -> float:
    """1.0 if any primary gold item appears in the top k."""
    prim = _primary(gold)
    return 1.0 if prim and any(c in prim for c in ranked[:k]) else 0.0


def recall_at_k(ranked: Sequence[str], gold: Mapping[str, int], k: int) -> float:
    prim = _primary(gold)
    if not prim:
        return 0.0
    return len(prim.intersection(ranked[:k])) / len(prim)


def precision_at_k(ranked: Sequence[str], gold: Mapping[str, int], k: int) -> float:
    if k <= 0 or not ranked:
        return 0.0
    top = ranked[:k]
    prim = _primary(gold)
    return sum(1 for c in top if c in prim) / len(top)


def coverage_at_k(ranked: Sequence[str], gold: Mapping[str, int], k: int) -> float:
    """Recall normalized by what is *reachable* in k slots: |hit| / min(|gold|, k).

    Plain recall@5 is capped at 0.42 for a query whose gold set is 12 document
    rows, which makes an aggregate over mixed-size gold sets unreadable.
    coverage@k answers "of the slots you had, how many did you spend well?".
    """
    prim = _primary(gold)
    if not prim or k <= 0:
        return 0.0
    return len(prim.intersection(ranked[:k])) / min(len(prim), k)


def rr_at_k(ranked: Sequence[str], gold: Mapping[str, int], k: int) -> float:
    """Reciprocal rank of the first primary gold item (0 if none in top k)."""
    prim = _primary(gold)
    for i, cid in enumerate(ranked[:k], 1):
        if cid in prim:
            return 1.0 / i
    return 0.0


def ndcg_at_k(ranked: Sequence[str], gold: Mapping[str, int], k: int) -> float:
    """Standard nDCG with gain = 2^grade - 1 and log2(rank+1) discount."""
    if not gold:
        return 0.0
    dcg = sum((2 ** gold.get(cid, 0) - 1) / log2(i + 1)
              for i, cid in enumerate(ranked[:k], 1))
    ideal = sorted(gold.values(), reverse=True)[:k]
    idcg = sum((2 ** g - 1) / log2(i + 1) for i, g in enumerate(ideal, 1))
    return dcg / idcg if idcg else 0.0


def mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0
