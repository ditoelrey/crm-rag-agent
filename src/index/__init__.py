"""Vector index layer: OpenAI embeddings -> local Qdrant -> eval-compatible retrievers.

  python -m index.cli build            # embed + upsert
  python -m index.cli search "Колку чини потврда за тековна состојба?"
  python -m eval.cli run --retriever index.qdrant_retriever:build --tag dense
"""
from __future__ import annotations

__all__ = ["config", "embedder", "qdrant_indexer", "qdrant_retriever", "hybrid"]
