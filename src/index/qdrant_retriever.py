"""
qdrant_retriever.py  --  dense retrieval over the Qdrant collection.
====================================================================
Implements the eval harness's retriever contract (`.name` + `.search(query, k)`),
so it is scored by exactly the same gold set and metrics as the BM25 baseline:

    python -m eval.cli run --retriever index.qdrant_retriever:build --tag dense \\
           --baseline eval/reports/baseline_bm25.json

Manifest check
--------------
On construction the retriever compares the collection's manifest against the
corpus it is being evaluated on and against its own embedding model. A query
embedded with a different model than the index is not a degraded search, it is
a meaningless one, and scoring it would produce a number that looks like a bad
retriever instead of a broken setup. Mismatches raise; a corpus-hash difference
warns (the index may legitimately be a superset during an incremental rebuild).

Filters
-------
`search(..., filters=...)` accepts the payload fields the indexer indexes:

    filters={"type": "tariffs"}                  one section type
    filters={"type": ["tariffs", "discounts"]}   any of
    filters={"id_service": 2135, "variation_scope": 11116}

`variation_scope` is the useful one: it matches a variation's own blocks *and*
the service-level shared blocks that apply to it (see qdrant_indexer).
"""
from __future__ import annotations

import sys
from typing import Any, Sequence

from qdrant_client import QdrantClient, models

from eval.corpus import Corpus
from eval.retriever import Hit

from . import config
from .embedder import HashEmbedder, OpenAIEmbedder
from .qdrant_indexer import open_client, read_manifest


def _to_condition(field: str, value: Any) -> models.FieldCondition:
    if isinstance(value, (list, tuple, set)):
        values = list(value)
        if values and isinstance(values[0], int):
            return models.FieldCondition(key=field, match=models.MatchAny(any=values))
        return models.FieldCondition(key=field,
                                     match=models.MatchAny(any=[str(v) for v in values]))
    return models.FieldCondition(key=field, match=models.MatchValue(value=value))


# Filter names the agent speaks -> payload fields the collection actually
# indexes. "service_scope" is the agent's word for "the service is already
# decided"; on this side it is just id_service. Without the mapping the dense
# arm would filter on a key no point carries and quietly return nothing.
_FIELD_ALIASES = {"service_scope": "id_service"}


def to_filter(filters: dict[str, Any] | None) -> models.Filter | None:
    if not filters:
        return None
    filters = {_FIELD_ALIASES.get(f, f): v for f, v in filters.items()}
    return models.Filter(must=[_to_condition(f, v) for f, v in filters.items()])


class QdrantRetriever:
    """Dense search. `search()` returns eval `Hit`s; `search_payloads()` returns
    the full payloads, which is what an answer layer wants."""

    def __init__(self, *, client: QdrantClient | None = None,
                 collection: str = config.COLLECTION,
                 qdrant_path: str = config.QDRANT_PATH,
                 embedder=None, corpus: Corpus | None = None,
                 strict: bool = True):
        self.client = client or open_client(qdrant_path)
        self.collection = collection
        self.manifest = read_manifest(qdrant_path, collection)

        if self.manifest is None:
            raise RuntimeError(
                f"no manifest for collection {collection!r} in {qdrant_path}. "
                f"Run:  python -m index.cli build")
        if not self.client.collection_exists(collection):
            raise RuntimeError(f"collection {collection!r} does not exist. "
                               f"Run:  python -m index.cli build")

        model = self.manifest["model"]
        dims = int(self.manifest["dims"])
        if embedder is None:
            embedder = (HashEmbedder(dims=dims) if model.startswith("hash-bow")
                        else OpenAIEmbedder(model=model, dims=dims, quiet=True))
        if embedder.model != model:
            raise RuntimeError(
                f"index was built with {model!r} but queries would be embedded "
                f"with {embedder.model!r}. Rebuild, or pass the matching embedder.")
        self.embedder = embedder
        self.dims = dims

        if corpus is not None and strict and corpus.sha256 != self.manifest.get("corpus_sha256"):
            print(f"warning: index was built from corpus "
                  f"{str(self.manifest.get('corpus_sha256'))[:12]} but you are "
                  f"evaluating {corpus.sha256[:12]}. Re-run `index.cli build`.",
                  file=sys.stderr)

        self.name = f"qdrant-dense({model},d={dims})"

    # -- retrieval --------------------------------------------------------- #
    def search_payloads(self, query: str, k: int = 10, *,
                        filters: dict[str, Any] | None = None,
                        score_threshold: float | None = None) -> list[dict[str, Any]]:
        vector = self.embedder.embed_query(query)
        res = self.client.query_points(
            self.collection, query=vector, limit=k,
            query_filter=to_filter(filters), with_payload=True,
            score_threshold=score_threshold)
        return [{**(p.payload or {}), "score": p.score} for p in res.points]

    def search(self, query: str, k: int = 10, *,
               filters: dict[str, Any] | None = None) -> list[Hit]:
        return [Hit(p["chunk_id"], float(p["score"]))
                for p in self.search_payloads(query, k, filters=filters)
                if p.get("chunk_id")]

    def close(self) -> None:
        self.client.close()


def build(corpus: Corpus | None = None) -> QdrantRetriever:
    """Factory for `eval.cli run --retriever index.qdrant_retriever:build`."""
    return QdrantRetriever(corpus=corpus)
