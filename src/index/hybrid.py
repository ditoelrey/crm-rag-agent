"""
hybrid.py  --  BM25 + dense fusion by Reciprocal Rank Fusion.
=============================================================
The baseline showed the two arms fail differently on this corpus: BM25 nails
service routing (`service_acc@1` 0.95-1.00) because the service name sits in
every contextual header, but ignores the *intent* of the question
(`type_precision@5` 0.05-0.35). Dense retrieval is the opposite bet -- it should
know that "колку чини" means a tariff row, while being weaker on the long,
near-identical service names this registry uses.

RRF is the right fusion here because the two scores are not comparable: BM25
returns unbounded term-weight sums, Qdrant returns cosine similarity. Ranks are.

    score(d) = w_lex / (K + rank_lex(d)) + w_vec / (K + rank_vec(d))

K=60 is the standard damping constant. Each arm is queried for `depth` (default
50) candidates so a document ranked ~30 by one arm can still be rescued by the
other; with depth == k, fusion degenerates to an intersection.

    python -m eval.cli run --retriever index.hybrid:build --tag hybrid \\
           --baseline eval/reports/baseline_bm25.json
"""
from __future__ import annotations

from typing import Any, Sequence

from eval.baselines import BM25Retriever
from eval.corpus import Corpus
from eval.corpus import load as load_corpus
from eval.retriever import Hit

from .qdrant_retriever import QdrantRetriever

RRF_K = 60


class HybridRetriever:
    def __init__(self, corpus: Corpus, *, dense: QdrantRetriever | None = None,
                 depth: int = 50, w_lex: float = 1.0, w_vec: float = 1.0,
                 rrf_k: int = RRF_K):
        self.corpus = corpus
        self.lexical = BM25Retriever(corpus)
        self.dense = dense or QdrantRetriever(corpus=corpus)
        self.depth, self.w_lex, self.w_vec, self.rrf_k = depth, w_lex, w_vec, rrf_k
        self.name = (f"hybrid-rrf(bm25 x {self.dense.name},depth={depth},"
                     f"w={w_lex}/{w_vec},K={rrf_k})")

    def search(self, query: str, k: int = 10, *,
               filters: dict[str, Any] | None = None) -> list[Hit]:
        lex = self.lexical.search(query, self.depth)
        vec = self.dense.search(query, self.depth, filters=filters)

        if filters:
            # BM25 has no payload filter, so apply the same predicate to its arm
            # rather than letting unfiltered lexical hits leak into the fusion.
            lex = [h for h in lex if _matches(self.corpus.by_id[h.chunk_id], filters)]

        scores: dict[str, float] = {}
        for arm, weight in ((lex, self.w_lex), (vec, self.w_vec)):
            for rank, hit in enumerate(arm, 1):
                scores[hit.chunk_id] = scores.get(hit.chunk_id, 0.0) + weight / (self.rrf_k + rank)
        ranked = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
        return [Hit(cid, score) for cid, score in ranked[:k]]

    def close(self) -> None:
        self.dense.close()


def _matches(block, filters: dict[str, Any]) -> bool:
    for field, want in filters.items():
        if field == "variation_scope":
            have: Any = (list(block.applies_to_variations) if block.is_shared
                         else ([block.id_variation] if block.id_variation else []))
        else:
            have = getattr(block, field, None)
        wanted = want if isinstance(want, (list, tuple, set)) else [want]
        if isinstance(have, list):
            if not set(have) & set(wanted):
                return False
        elif have not in wanted:
            return False
    return True


def build(corpus: Corpus | None = None) -> HybridRetriever:
    """Factory for `eval.cli run --retriever index.hybrid:build`."""
    return HybridRetriever(corpus or load_corpus())


def build_aliased(corpus: Corpus | None = None):
    """Hybrid retrieval with user-vocabulary expansion (see index/aliases.py).

    Kept as a separate factory so the alias layer is an A/B switch measured
    against the same gold set, not a silent change to the retriever.
    """
    from .aliases import AliasExpandingRetriever
    c = corpus or load_corpus()
    return AliasExpandingRetriever(HybridRetriever(c), c)
