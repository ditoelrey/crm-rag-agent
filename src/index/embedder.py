"""
embedder.py  --  OpenAI embeddings with caching, batching and backoff.
======================================================================
Three things this does that a bare `client.embeddings.create()` loop does not:

  CACHE     Every vector is persisted in a sqlite file keyed by
            sha256(model|dims|text). Re-running the indexer after a corpus
            rebuild only pays for the blocks whose text actually changed, and a
            crashed run resumes for free. This matters more than it sounds:
            without it, every experiment costs a full re-embed.

  BATCHING  Requests are shaped by BOTH input count and total characters, and a
            400 from an over-long batch is recovered by splitting the batch in
            half rather than failing the build.

  BACKOFF   Rate limits and 5xx are retried with exponential backoff + jitter,
            on top of the SDK's own retries.

`HashEmbedder` is a deterministic offline stand-in (random-projection
bag-of-words over the eval tokenizer). It needs no API key and no network, so
the whole index -> retrieve -> eval pipeline can be verified end-to-end before
spending anything. Its scores are weak but real, which is the point: a plumbing
bug shows up as zero, not as "the model is bad".
"""
from __future__ import annotations

import hashlib
import math
import os
import random
import sqlite3
import struct
import sys
import time
from typing import Iterable, Iterator, Sequence

from . import config


# --------------------------------------------------------------------------- #
# cache
# --------------------------------------------------------------------------- #
class EmbeddingCache:
    """sqlite-backed vector cache. float32 blobs; ~24 MB for the full corpus."""

    def __init__(self, path: str = config.CACHE_PATH):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        self.path = path
        self.conn = sqlite3.connect(path)
        self.conn.execute(
            "CREATE TABLE IF NOT EXISTS vectors ("
            "  key TEXT PRIMARY KEY, dims INTEGER NOT NULL, vec BLOB NOT NULL)")
        self.conn.commit()
        self.hits = 0
        self.misses = 0

    @staticmethod
    def key(model: str, dims: int, text: str) -> str:
        return hashlib.sha256(f"{model}|{dims}|{text}".encode("utf-8")).hexdigest()

    def get(self, key: str) -> list[float] | None:
        row = self.conn.execute("SELECT dims, vec FROM vectors WHERE key = ?",
                                (key,)).fetchone()
        if row is None:
            self.misses += 1
            return None
        self.hits += 1
        dims, blob = row
        return list(struct.unpack(f"<{dims}f", blob))

    def put_many(self, items: Iterable[tuple[str, Sequence[float]]]) -> None:
        rows = [(k, len(v), struct.pack(f"<{len(v)}f", *v)) for k, v in items]
        self.conn.executemany(
            "INSERT OR REPLACE INTO vectors (key, dims, vec) VALUES (?, ?, ?)", rows)
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()


# --------------------------------------------------------------------------- #
# OpenAI
# --------------------------------------------------------------------------- #
class MissingAPIKey(RuntimeError):
    pass


class OpenAIEmbedder:
    def __init__(self, *, model: str = config.DEFAULT_MODEL,
                 dims: int = config.DEFAULT_DIMS, api_key: str | None = None,
                 cache: EmbeddingCache | None = None, quiet: bool = False):
        key = api_key or os.environ.get("OPENAI_API_KEY")
        if not key:
            raise MissingAPIKey(
                "OPENAI_API_KEY is not set.\n"
                "  PowerShell (this session):  $env:OPENAI_API_KEY = '<your key>'\n"
                "  PowerShell (persistent)  :  setx OPENAI_API_KEY '<your key>'\n"
                "Or run with --dry-run to exercise the pipeline offline.")
        try:
            from openai import OpenAI
        except ImportError as e:  # pragma: no cover
            raise RuntimeError("pip install openai") from e

        self.model, self.dims, self.quiet = model, dims, quiet
        self.client = OpenAI(api_key=key, max_retries=2)
        self.cache = cache if cache is not None else EmbeddingCache()
        self.name = f"openai:{model}:{dims}"
        self.api_tokens = 0      # billed tokens this process (cache hits are free)
        self.api_calls = 0

    # -- public ------------------------------------------------------------ #
    def embed_query(self, text: str) -> list[float]:
        return self.embed([text])[0]

    def embed(self, texts: Sequence[str], *, progress: bool = False) -> list[list[float]]:
        """Cached, batched embedding. Order of the input is preserved."""
        prepared = [self._truncate(t) for t in texts]
        keys = [EmbeddingCache.key(self.model, self.dims, t) for t in prepared]

        out: list[list[float] | None] = [None] * len(prepared)
        todo: list[int] = []
        for i, k in enumerate(keys):
            cached = self.cache.get(k)
            if cached is None:
                todo.append(i)
            else:
                out[i] = cached

        if todo and progress and not self.quiet:
            print(f"  {len(prepared) - len(todo)} cached, {len(todo)} to embed",
                  file=sys.stderr)

        done = 0
        for batch in _batches(todo, prepared):
            vectors = self._embed_batch([prepared[i] for i in batch])
            self.cache.put_many((keys[i], v) for i, v in zip(batch, vectors))
            for i, v in zip(batch, vectors):
                out[i] = v
            done += len(batch)
            if progress and not self.quiet:
                pct = 100 * done / max(1, len(todo))
                print(f"  embedded {done}/{len(todo)} ({pct:.0f}%)  "
                      f"tokens={self.api_tokens:,}  ~${self.cost_usd:.4f}",
                      end="\r", file=sys.stderr, flush=True)
        if todo and progress and not self.quiet:
            print(file=sys.stderr)

        missing = [i for i, v in enumerate(out) if v is None]
        if missing:
            raise RuntimeError(f"embedding failed for {len(missing)} input(s)")
        return out  # type: ignore[return-value]

    @property
    def cost_usd(self) -> float:
        return self.api_tokens / 1_000_000 * config.PRICE_PER_1M_TOKENS

    # -- internals --------------------------------------------------------- #
    def _truncate(self, text: str) -> str:
        text = text.strip() or " "     # the API rejects empty strings
        if len(text) > config.MAX_INPUT_CHARS:
            if not self.quiet:
                print(f"warning: truncating a {len(text)}-char block to "
                      f"{config.MAX_INPUT_CHARS}", file=sys.stderr)
            return text[: config.MAX_INPUT_CHARS]
        return text

    def _embed_batch(self, batch: Sequence[str], depth: int = 0) -> list[list[float]]:
        import openai

        for attempt in range(config.MAX_RETRIES):
            try:
                resp = self.client.embeddings.create(
                    model=self.model, input=list(batch), dimensions=self.dims)
                self.api_calls += 1
                self.api_tokens += getattr(resp.usage, "total_tokens", 0) or 0
                return [d.embedding for d in sorted(resp.data, key=lambda d: d.index)]

            except openai.BadRequestError:
                # Almost always "too many tokens in the batch" -- halve and retry.
                # A single input that still fails is a real error and propagates.
                if len(batch) == 1 or depth > 8:
                    raise
                mid = len(batch) // 2
                return (self._embed_batch(batch[:mid], depth + 1) +
                        self._embed_batch(batch[mid:], depth + 1))

            except (openai.RateLimitError, openai.APITimeoutError,
                    openai.APIConnectionError, openai.InternalServerError) as e:
                if attempt == config.MAX_RETRIES - 1:
                    raise
                delay = min(60.0, 2 ** attempt) * (0.75 + random.random() * 0.5)
                if not self.quiet:
                    print(f"\n  {type(e).__name__}; retrying in {delay:.1f}s "
                          f"({attempt + 1}/{config.MAX_RETRIES})", file=sys.stderr)
                time.sleep(delay)
        raise RuntimeError("unreachable")


def _batches(indices: Sequence[int], texts: Sequence[str]) -> Iterator[list[int]]:
    """Yield index batches bounded by both count and total characters."""
    batch: list[int] = []
    chars = 0
    for i in indices:
        n = len(texts[i])
        if batch and (len(batch) >= config.BATCH_SIZE or
                      chars + n > config.MAX_BATCH_CHARS):
            yield batch
            batch, chars = [], 0
        batch.append(i)
        chars += n
    if batch:
        yield batch


# --------------------------------------------------------------------------- #
# offline stand-in
# --------------------------------------------------------------------------- #
class HashEmbedder:
    """Deterministic, network-free embeddings: a random projection of the token
    bag, using the same MK tokenizer as the lexical index.

    Not a quality baseline -- a *plumbing* baseline. It proves the collection,
    payload, filters, retriever adapter and eval wiring all work before a single
    token is billed.
    """

    def __init__(self, *, dims: int = 256, seed: int = 11):
        self.dims, self.seed = dims, seed
        self.model = f"hash-bow-{dims}"
        self.name = f"hash:{dims}"
        self.api_tokens = 0
        self.api_calls = 0
        self.cost_usd = 0.0
        self._vecs: dict[str, list[float]] = {}

    def _term_vector(self, term: str) -> list[float]:
        v = self._vecs.get(term)
        if v is None:
            rng = random.Random(f"{self.seed}:{term}")
            v = [rng.gauss(0.0, 1.0) for _ in range(self.dims)]
            self._vecs[term] = v
        return v

    def embed_query(self, text: str) -> list[float]:
        return self.embed([text])[0]

    def embed(self, texts: Sequence[str], *, progress: bool = False) -> list[list[float]]:
        from eval.text import tokenize

        out = []
        for text in texts:
            acc = [0.0] * self.dims
            for term in tokenize(text):
                tv = self._term_vector(term)
                for j in range(self.dims):
                    acc[j] += tv[j]
            norm = math.sqrt(sum(x * x for x in acc)) or 1.0
            out.append([x / norm for x in acc])
        return out
