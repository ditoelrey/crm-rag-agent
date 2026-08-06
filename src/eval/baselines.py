"""
baselines.py  --  reference retrievers, so the harness has a floor, a ceiling
=============================================================================
and a real system to score on day one.

  RandomRetriever  floor. If a real retriever is not far above this, the metric
                   or the gold set is wrong -- check that before the retriever.
  OracleRetriever  ceiling. Returns the gold itself; must score 1.0 on hit/MRR
                   and coverage. Any oracle score below 1.0 is a harness bug,
                   which is exactly what it is here to catch.
  BM25Retriever    a genuine lexical baseline in pure stdlib. Also the intended
                   lexical arm of the hybrid index, so improving it is not
                   throwaway work.

BM25 notes
----------
Okapi BM25 over `embedding_text` (the contextual header + content), which is the
same string the embedder will see -- keeping lexical and dense arms on identical
input makes their scores fusable later. k1=1.2 / b=0.75 are the standard values;
b matters here because block lengths span 133..7906 chars.
"""
from __future__ import annotations

import math
import random
from collections import defaultdict
from typing import Sequence

from .corpus import Corpus
from .retriever import Hit
from .text import tokenize


class BM25Retriever:
    """Okapi BM25 with an in-memory inverted index. Builds over 5.9k blocks in
    well under a second; no dependencies."""

    def __init__(self, corpus: Corpus, *, k1: float = 1.2, b: float = 0.75,
                 field: str = "embedding_text", do_stem: bool = True):
        self.k1, self.b, self.field, self.do_stem = k1, b, field, do_stem
        self.name = f"bm25(k1={k1},b={b},{'stem' if do_stem else 'raw'},{field})"
        self.chunk_ids: list[str] = []
        self.doc_len: list[int] = []
        self.postings: dict[str, list[tuple[int, int]]] = defaultdict(list)  # term -> [(doc, tf)]

        for doc_id, block in enumerate(corpus):
            toks = tokenize(getattr(block, field), do_stem=do_stem)
            self.chunk_ids.append(block.chunk_id)
            self.doc_len.append(len(toks))
            tf: dict[str, int] = defaultdict(int)
            for t in toks:
                tf[t] += 1
            for term, count in tf.items():
                self.postings[term].append((doc_id, count))

        self.n_docs = len(self.chunk_ids)
        self.avgdl = (sum(self.doc_len) / self.n_docs) if self.n_docs else 0.0
        # Robertson/Sparck-Jones idf with the +1 smoothing that keeps it positive
        # for terms appearing in more than half the collection.
        self.idf = {
            term: math.log(1 + (self.n_docs - len(p) + 0.5) / (len(p) + 0.5))
            for term, p in self.postings.items()
        }
        # length normalization is query-independent -- precompute it once
        self._norm = [
            self.k1 * (1 - self.b + self.b * (dl / self.avgdl if self.avgdl else 0.0))
            for dl in self.doc_len
        ]

    def search(self, query: str, k: int) -> Sequence[Hit]:
        scores: dict[int, float] = defaultdict(float)
        for term in tokenize(query, do_stem=self.do_stem):
            postings = self.postings.get(term)
            if not postings:
                continue
            idf = self.idf[term]
            for doc_id, tf in postings:
                scores[doc_id] += idf * (tf * (self.k1 + 1)) / (tf + self._norm[doc_id])
        top = sorted(scores.items(), key=lambda kv: (-kv[1], self.chunk_ids[kv[0]]))[:k]
        return [Hit(self.chunk_ids[doc_id], score) for doc_id, score in top]


class RandomRetriever:
    """Seeded random ranking -- the score floor."""

    def __init__(self, corpus: Corpus, *, seed: int = 0):
        self.name = f"random(seed={seed})"
        self._ids = [b.chunk_id for b in corpus]
        self._seed = seed

    def search(self, query: str, k: int) -> Sequence[Hit]:
        rng = random.Random(f"{self._seed}:{query}")
        return [Hit(cid, 0.0) for cid in rng.sample(self._ids, min(k, len(self._ids)))]


class OracleRetriever:
    """Returns each case's own gold set -- the harness's self-check.

    Needs the cases up front, so it is wired in by the CLI rather than being a
    general retriever.
    """

    def __init__(self, cases, *, noise: int = 0, corpus: Corpus | None = None):
        self.name = f"oracle(noise={noise})"
        self._gold = {c.query: sorted(c.gold, key=lambda cid: -c.gold[cid]) for c in cases}
        self._noise = noise
        self._pool = [b.chunk_id for b in corpus] if corpus is not None else []

    def search(self, query: str, k: int) -> Sequence[Hit]:
        gold = list(self._gold.get(query, ()))
        if self._noise and self._pool:
            rng = random.Random(f"noise:{query}")
            for _ in range(self._noise):
                gold.insert(rng.randrange(len(gold) + 1), rng.choice(self._pool))
        return [Hit(cid, 1.0) for cid in gold[:k]]
