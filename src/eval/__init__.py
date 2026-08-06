"""Retrieval evaluation harness for the CRM e-services corpus.

Retriever-agnostic by design: anything that implements `search(query, k)` can be
scored, so the harness exists *before* the index does and outlives any single
indexing choice (lexical / dense / hybrid / reranked).

Run:  python -m eval.cli --help      (from src/)
"""
from __future__ import annotations

__all__ = ["corpus", "text", "metrics", "goldset", "retriever", "baselines", "harness"]
