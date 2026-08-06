"""
qdrant_indexer.py  --  embed the corpus and upsert it into a local Qdrant.
==========================================================================
Reads the validated corpus, embeds `embedding_text` (contextual header + content
-- the same string the BM25 arm indexes, so the two are fusable), and stores one
point per block with the full metadata as payload.

Point identity
--------------
`uuid5(NAMESPACE_URL, chunk_id)` -- deterministic, so re-running upserts in place
instead of duplicating. Combined with the `text_sha256` payload field the build
is incremental: unchanged blocks are neither re-embedded nor re-upserted, and an
interrupted run resumes.

The `variation_scope` payload field
-----------------------------------
A service-scoped shared block (terminology, legalBasis, description, FAQ) is
relevant to *every* sibling variation; a variation-scoped block only to its own.
Filtering by `id_variation` alone would therefore silently drop the shared
content that answers half the questions. `variation_scope` flattens both cases
into one list -- `[id_variation]` for variation blocks, `applies_to_variations`
for shared ones -- so "everything relevant to variation 11116" is a single
MatchAny, which is what makes variation-filtered retrieval cheap and correct.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import shutil
import time
import uuid
import warnings
from dataclasses import dataclass
from typing import Any, Sequence

from qdrant_client import QdrantClient, models

from eval.corpus import Block, Corpus
from eval.corpus import load as load_corpus

from . import config

# Fixed namespace: point ids must stay stable across rebuilds and machines.
_NAMESPACE = uuid.UUID("6f1a5f6e-2f4e-5c33-9a1a-9b7c1d0e4f22")

# Payload fields that get a Qdrant index (everything we filter or facet on).
_INDEXED_FIELDS: dict[str, Any] = {
    "id_service": models.PayloadSchemaType.INTEGER,
    "id_variation": models.PayloadSchemaType.INTEGER,
    "variation_scope": models.PayloadSchemaType.INTEGER,
    "scope": models.PayloadSchemaType.KEYWORD,
    "type": models.PayloadSchemaType.KEYWORD,
    "chunk_id": models.PayloadSchemaType.KEYWORD,
    "is_online": models.PayloadSchemaType.BOOL,
    "municipality": models.PayloadSchemaType.KEYWORD,   # agent-directory lookup
    "list": models.PayloadSchemaType.KEYWORD,
}


def point_id(chunk_id: str) -> str:
    return str(uuid.uuid5(_NAMESPACE, chunk_id))


def payload_of(block: Block, text_sha: str) -> dict[str, Any]:
    """Full metadata as payload. `content`/`context` are stored so the answer
    layer can build a grounded prompt straight from a search result, without a
    second lookup into the JSONL."""
    if block.is_shared:
        variation_scope = list(block.applies_to_variations)
    else:
        variation_scope = [block.id_variation] if block.id_variation else []
    return {
        "chunk_id": block.chunk_id,
        "scope": block.scope,
        "type": block.type,
        "id_service": block.id_service,
        "service_name": block.service_name,
        "id_variation": block.id_variation,
        "variation_short_name": block.variation_short_name,
        "is_online": block.is_online,
        "applies_to_variations": list(block.applies_to_variations),
        "variation_scope": variation_scope,
        "row": block.row,
        "municipality": block.municipality,
        "list": block.list_key,
        "n_agents": block.n_agents,
        "part": block.part,
        "n_parts": block.n_parts,
        "updated": block.updated,
        "context": block.context,
        "content": block.content,
        "text_sha256": text_sha,
    }


@dataclass
class BuildStats:
    blocks: int = 0
    embedded: int = 0
    upserted: int = 0
    unchanged: int = 0
    deleted: int = 0
    api_calls: int = 0
    api_tokens: int = 0
    cost_usd: float = 0.0
    seconds: float = 0.0


def open_client(qdrant_path: str = config.QDRANT_PATH) -> QdrantClient:
    os.makedirs(qdrant_path, exist_ok=True)
    return QdrantClient(path=qdrant_path)


def reset_storage(qdrant_path: str = config.QDRANT_PATH, *, quiet: bool = False) -> bool:
    """Delete the local storage directory. NO client may be open on it.

    Works around a qdrant-client local-mode bug: `delete_collection()` drops the
    collection from its in-memory registry without closing that collection's
    sqlite handle, so the on-disk vector storage survives and a subsequent
    `create_collection()` with a different vector size fails with
    "could not broadcast input array from shape (N,) into shape (M,)".
    Reproducible in six lines against qdrant-client 1.18.0, independent of this
    code. Purging the directory before any client is opened side-steps it.

    Safe because everything here is derived: the embedding cache lives outside
    QDRANT_PATH (config.CACHE_PATH), so a reset re-upserts but never re-embeds.
    """
    if not os.path.isdir(qdrant_path):
        return False
    shutil.rmtree(qdrant_path)
    if not quiet:
        print(f"purged local storage {qdrant_path}")
    os.makedirs(qdrant_path, exist_ok=True)
    return True


def ensure_collection(client: QdrantClient, collection: str, dims: int,
                      *, recreate: bool = False, quiet: bool = False) -> None:
    exists = client.collection_exists(collection)
    if exists and recreate:
        client.delete_collection(collection)
        exists = False
    if not exists:
        client.create_collection(
            collection,
            vectors_config=models.VectorParams(size=dims,
                                               distance=models.Distance.COSINE),
        )
        if not quiet:
            print(f"created collection {collection!r} (dim={dims}, cosine)")
    else:
        info = client.get_collection(collection)
        actual = info.config.params.vectors.size
        if actual != dims:
            raise ValueError(
                f"collection {collection!r} stores {actual}-dim vectors but this "
                f"build produces {dims}-dim. Re-run with --recreate.")

    # Local Qdrant filters by scanning the payload and warns that indexes are a
    # no-op; we still declare them so the same build script produces a properly
    # indexed collection when this is pointed at a Qdrant server.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*[Pp]ayload indexes.*")
        for field, schema in _INDEXED_FIELDS.items():
            try:
                client.create_payload_index(collection, field_name=field,
                                            field_schema=schema, wait=True)
            except Exception:
                pass   # a server rejects duplicates; not fatal either way


def existing_hashes(client: QdrantClient, collection: str) -> dict[str, str]:
    """chunk_id -> text_sha256 for everything already stored."""
    out: dict[str, str] = {}
    offset = None
    while True:
        points, offset = client.scroll(
            collection, limit=2048, offset=offset,
            with_payload=["chunk_id", "text_sha256"], with_vectors=False)
        for p in points:
            pl = p.payload or {}
            if pl.get("chunk_id"):
                out[pl["chunk_id"]] = pl.get("text_sha256", "")
        if offset is None:
            break
    return out


def build(corpus: Corpus, embedder, *, collection: str = config.COLLECTION,
          qdrant_path: str = config.QDRANT_PATH, recreate: bool = False,
          prune: bool = True, quiet: bool = False,
          upsert_batch: int = 256) -> BuildStats:
    """Embed and upsert the whole corpus. Incremental unless `recreate`.

    Owns the Qdrant client for the duration: local mode takes an exclusive lock
    on the storage directory, and a reset has to happen with nothing open.
    """
    t0 = time.perf_counter()
    stats = BuildStats(blocks=len(corpus))
    dims = embedder.dims if hasattr(embedder, "dims") else config.DEFAULT_DIMS

    # Vectors from two different models share no geometry, so a model or
    # dimension change invalidates every stored point. Rebuild rather than
    # mixing two vector spaces in one collection.
    prior = read_manifest(qdrant_path, collection)
    if prior and (prior.get("model") != embedder.model or int(prior.get("dims", 0)) != dims):
        if not quiet:
            print(f"embedding changed ({prior.get('model')}/{prior.get('dims')} -> "
                  f"{embedder.model}/{dims}); rebuilding from scratch")
        recreate = True
    if recreate:
        reset_storage(qdrant_path, quiet=quiet)

    client = open_client(qdrant_path)
    try:
        return _build_with_client(client, corpus, embedder, dims, stats, t0,
                                  collection=collection, qdrant_path=qdrant_path,
                                  recreate=recreate, prune=prune, quiet=quiet,
                                  upsert_batch=upsert_batch)
    finally:
        client.close()


def _build_with_client(client: QdrantClient, corpus: Corpus, embedder, dims: int,
                       stats: BuildStats, t0: float, *, collection: str,
                       qdrant_path: str, recreate: bool, prune: bool,
                       quiet: bool, upsert_batch: int) -> BuildStats:
    ensure_collection(client, collection, dims, recreate=False, quiet=quiet)

    texts = [getattr(b, config.EMBED_FIELD) for b in corpus]
    shas = [hashlib.sha256(f"{embedder.model}|{dims}|{t}".encode()).hexdigest()
            for t in texts]

    known = {} if recreate else existing_hashes(client, collection)
    todo = [i for i, b in enumerate(corpus) if known.get(b.chunk_id) != shas[i]]
    stats.unchanged = len(corpus) - len(todo)
    if not quiet:
        print(f"{len(corpus)} blocks: {len(todo)} to (re)index, "
              f"{stats.unchanged} unchanged")

    if todo:
        vectors = embedder.embed([texts[i] for i in todo], progress=not quiet)
        stats.embedded = len(todo)
        blocks = list(corpus)
        for start in range(0, len(todo), upsert_batch):
            chunk = todo[start:start + upsert_batch]
            client.upsert(collection, wait=True, points=[
                models.PointStruct(
                    id=point_id(blocks[i].chunk_id),
                    vector=vectors[start + j],
                    payload=payload_of(blocks[i], shas[i]),
                )
                for j, i in enumerate(chunk)
            ])
            stats.upserted += len(chunk)
            if not quiet:
                print(f"  upserted {stats.upserted}/{len(todo)}",
                      end="\r", file=sys.stderr, flush=True)
        if not quiet:
            print(file=sys.stderr)

    if prune and not recreate:
        live = {b.chunk_id for b in corpus}
        stale = [cid for cid in known if cid not in live]
        if stale:
            client.delete(collection, wait=True, points_selector=models.PointIdsList(
                points=[point_id(cid) for cid in stale]))
            stats.deleted = len(stale)
            if not quiet:
                print(f"pruned {len(stale)} block(s) no longer in the corpus")

    stats.api_calls = getattr(embedder, "api_calls", 0)
    stats.api_tokens = getattr(embedder, "api_tokens", 0)
    stats.cost_usd = getattr(embedder, "cost_usd", 0.0)
    stats.seconds = time.perf_counter() - t0
    write_manifest(corpus, embedder, dims, collection, stats, qdrant_path)
    return stats


def write_manifest(corpus: Corpus, embedder, dims: int, collection: str,
                   stats: BuildStats, qdrant_path: str = config.QDRANT_PATH) -> str:
    """Record what the vectors *mean*, so a retriever can refuse a mismatch."""
    path = config.manifest_path(qdrant_path, collection)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump({
            "collection": collection,
            "model": embedder.model,
            "dims": dims,
            "embed_field": config.EMBED_FIELD,
            "corpus_path": os.path.basename(corpus.path),
            "corpus_sha256": corpus.sha256,
            "n_blocks": len(corpus),
            "built_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "build": {"embedded": stats.embedded, "unchanged": stats.unchanged,
                      "deleted": stats.deleted, "api_tokens": stats.api_tokens,
                      "cost_usd": round(stats.cost_usd, 6),
                      "seconds": round(stats.seconds, 1)},
        }, fh, ensure_ascii=False, indent=2)
    return path


def read_manifest(qdrant_path: str = config.QDRANT_PATH,
                  collection: str = config.COLLECTION) -> dict[str, Any] | None:
    path = config.manifest_path(qdrant_path, collection)
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)
