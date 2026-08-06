"""
cli.py  --  build and query the vector index.   Run from src/.
==============================================================

  python -m index.cli build                    # embed + upsert (incremental)
  python -m index.cli build --dry-run          # offline, no API key, no cost
  python -m index.cli build --estimate         # what would this cost?
  python -m index.cli build --recreate         # wipe and rebuild
  python -m index.cli info
  python -m index.cli search "Колку чини потврда за тековна состојба?" --type tariffs

Then score it with the eval harness:

  python -m eval.cli run --retriever index.qdrant_retriever:build --tag dense \\
         --baseline eval/reports/baseline_bm25.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys

from eval.corpus import DEFAULT_CORPUS
from eval.corpus import load as load_corpus

from . import config
from .embedder import EmbeddingCache, HashEmbedder, MissingAPIKey, OpenAIEmbedder
from .qdrant_indexer import build as build_index
from .qdrant_indexer import open_client, read_manifest


def _estimate(corpus) -> tuple[int, float]:
    chars = sum(len(getattr(b, config.EMBED_FIELD)) for b in corpus)
    tokens = int(chars / config.CHARS_PER_TOKEN)
    return tokens, tokens / 1_000_000 * config.PRICE_PER_1M_TOKENS


def cmd_build(args) -> int:
    corpus = load_corpus(args.corpus)
    tokens, cost = _estimate(corpus)
    print(f"corpus  : {os.path.basename(corpus.path)}  "
          f"{len(corpus)} blocks  sha={corpus.sha256[:12]}")
    print(f"estimate: ~{tokens:,} tokens  ~${cost:.4f} at "
          f"${config.PRICE_PER_1M_TOKENS}/1M  (cached blocks are free)")
    if args.estimate:
        return 0

    if args.dry_run:
        embedder = HashEmbedder(dims=args.dims if args.dims != config.DEFAULT_DIMS else 256)
        print(f"embedder: {embedder.name}  [DRY RUN -- offline, not the real index]")
    else:
        try:
            embedder = OpenAIEmbedder(model=args.model, dims=args.dims,
                                      cache=EmbeddingCache(args.cache))
        except MissingAPIKey as e:
            print(f"\n{e}", file=sys.stderr)
            return 2
        print(f"embedder: {embedder.name}  cache={os.path.relpath(args.cache)}")

    stats = build_index(corpus, embedder, collection=args.collection,
                        qdrant_path=args.qdrant_path, recreate=args.recreate,
                        prune=not args.no_prune)

    print(f"\nblocks     : {stats.blocks}")
    print(f"embedded   : {stats.embedded}   unchanged: {stats.unchanged}   "
          f"pruned: {stats.deleted}")
    print(f"api        : {stats.api_calls} call(s), {stats.api_tokens:,} tokens, "
          f"${stats.cost_usd:.4f}")
    print(f"elapsed    : {stats.seconds:.1f}s")
    print(f"manifest   : {os.path.relpath(config.manifest_path(args.qdrant_path, args.collection))}")
    return 0


def cmd_info(args) -> int:
    manifest = read_manifest(args.qdrant_path, args.collection)
    if manifest is None:
        print(f"no manifest in {args.qdrant_path} for {args.collection!r}; "
              f"run `python -m index.cli build`")
        return 1
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    client = open_client(args.qdrant_path)
    try:
        if client.collection_exists(args.collection):
            print(f"\npoints in collection: "
                  f"{client.count(args.collection, exact=True).count}")
        else:
            print(f"\ncollection {args.collection!r} is MISSING (manifest is stale)")
    finally:
        client.close()
    return 0


def cmd_search(args) -> int:
    from .hybrid import HybridRetriever
    from .qdrant_retriever import QdrantRetriever

    corpus = load_corpus(args.corpus) if args.mode == "hybrid" else None
    filters: dict = {}
    if args.type:
        filters["type"] = args.type
    if args.service:
        filters["id_service"] = args.service
    if args.variation:
        filters["variation_scope"] = args.variation

    client = open_client(args.qdrant_path)
    try:
        dense = QdrantRetriever(client=client, collection=args.collection,
                                qdrant_path=args.qdrant_path, corpus=corpus)
        retriever = (HybridRetriever(corpus, dense=dense) if args.mode == "hybrid"
                     else dense)
        print(f"{retriever.name}\nquery: {args.query}"
              f"{('  filters: ' + json.dumps(filters, ensure_ascii=False)) if filters else ''}\n")
        hits = retriever.search(args.query, args.k, filters=filters or None)
        by_id = {b.chunk_id: b for b in load_corpus(args.corpus)} if corpus is None else corpus.by_id
        for rank, hit in enumerate(hits, 1):
            b = by_id[hit.chunk_id]
            label = f"{b.service_name[:58]}" + (f" — {b.variation_short_name}"
                                                if b.variation_short_name else "")
            print(f"{rank:>2}. {hit.score:.4f}  [{b.type}] {label}")
            print(f"    {b.chunk_id}")
            print(f"    {b.content[:160].replace(chr(10), ' ')}")
    finally:
        client.close()
    return 0


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

    p = argparse.ArgumentParser(prog="index.cli", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--corpus", default=DEFAULT_CORPUS)
    p.add_argument("--qdrant-path", default=config.QDRANT_PATH)
    p.add_argument("--collection", default=config.COLLECTION)
    sub = p.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build", help="embed the corpus and upsert into Qdrant")
    b.add_argument("--model", default=config.DEFAULT_MODEL)
    b.add_argument("--dims", type=int, default=config.DEFAULT_DIMS)
    b.add_argument("--cache", default=config.CACHE_PATH)
    b.add_argument("--recreate", action="store_true", help="drop the collection first")
    b.add_argument("--no-prune", action="store_true",
                   help="keep points whose blocks left the corpus")
    b.add_argument("--dry-run", action="store_true",
                   help="offline HashEmbedder: verifies the pipeline, costs nothing")
    b.add_argument("--estimate", action="store_true", help="print cost and exit")
    b.set_defaults(func=cmd_build)

    i = sub.add_parser("info", help="show the collection manifest")
    i.set_defaults(func=cmd_info)

    s = sub.add_parser("search", help="query the index")
    s.add_argument("query")
    s.add_argument("-k", type=int, default=10)
    s.add_argument("--mode", choices=["dense", "hybrid"], default="dense")
    s.add_argument("--type", nargs="*", help="filter by block type(s)")
    s.add_argument("--service", type=int, help="filter by id_service")
    s.add_argument("--variation", type=int,
                   help="filter by variation (matches its shared blocks too)")
    s.set_defaults(func=cmd_search)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
