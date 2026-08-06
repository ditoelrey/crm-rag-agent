"""
config.py  --  one place for every knob the index layer has.
============================================================
Anything that changes the *meaning* of a stored vector (model, dimensions,
which field is embedded) is recorded in the collection manifest at build time
and checked at query time -- a query embedded with a different model than the
index is not a worse search, it is a meaningless one.
"""
from __future__ import annotations

import os

_SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJECT_ROOT = os.path.dirname(_SRC_DIR)

# --- storage ---------------------------------------------------------------
QDRANT_PATH = os.path.join(PROJECT_ROOT, "qdrant_data")
COLLECTION = "crm_services"

# Deliberately NOT inside QDRANT_PATH: a --recreate purges the whole storage
# directory (see qdrant_indexer.reset_storage), and the cache is the one
# artifact here that costs money to regenerate. The index is disposable; the
# vectors are not.
CACHE_PATH = os.path.join(PROJECT_ROOT, ".cache", "embeddings.sqlite3")

# --- embeddings ------------------------------------------------------------
DEFAULT_MODEL = "text-embedding-3-small"
DEFAULT_DIMS = 1536          # text-embedding-3-small native; may be shortened
EMBED_FIELD = "embedding_text"   # context header + content, as the corpus builds it

# Request shaping. OpenAI allows 2048 inputs / 300k tokens per embeddings call;
# these sit well under both so a burst of retries cannot trip a hard limit.
BATCH_SIZE = 128
MAX_BATCH_CHARS = 200_000
MAX_INPUT_CHARS = 24_000     # ~8k tokens of Cyrillic; longest block is 7,906
MAX_RETRIES = 6

# Cyrillic costs roughly 2 chars/token on cl100k -- used only for batch shaping
# and the cost estimate, never for correctness.
CHARS_PER_TOKEN = 2.0
PRICE_PER_1M_TOKENS = 0.02   # text-embedding-3-small, USD


def manifest_path(qdrant_path: str = QDRANT_PATH, collection: str = COLLECTION) -> str:
    return os.path.join(qdrant_path, f"{collection}.manifest.json")
