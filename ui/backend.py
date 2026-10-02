"""
backend.py  --  process-wide singletons and the per-session agent factory.
==========================================================================
WHAT IS SHARED AND WHAT IS NOT
------------------------------
Shared, built once at import ON THE BACKEND THREAD (see EXECUTOR -- the index
and its sqlite vector cache are thread-affine, so whichever thread builds them
must also be the thread that uses them):

    CORPUS      7,759 blocks parsed from jsonl
    RETRIEVER   BM25 index + Qdrant client + the alias and structured layers
    CLIENT      one OpenAI client, so sessions do not each open a socket pool

Per session:

    CRMAgent    because `history`, `seen_docs` and `total_cost_usd` are
                per-conversation state. Sharing one agent between two users
                would splice their conversations together.

Building a retriever per session would re-read the corpus and re-open Qdrant on
every browser tab, which is slow and, for the local Qdrant store, not even
possible twice in one process.

THE STACK MUST MATCH THE AGENT'S OWN
------------------------------------
CRMAgent builds `SectionFetch(AliasExpanding(Hybrid(corpus)))` when no retriever
is passed -- and passing one SKIPS that wrapping entirely. index.structured.build
is exactly that composition, so the UI gets the same retrieval the evals measured
rather than a bare hybrid that quietly drops two layers.
"""
from __future__ import annotations

import asyncio
import contextvars
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

# src/ is the import root for this project; app.py lives one level above it.
SRC = Path(__file__).resolve().parent.parent / "src"


def ensure_src_path() -> None:
    """Put src/ on sys.path, and APPEND -- never insert at 0.

    Chainlit loads the app module like this (chainlit/config.py load_module):

        sys.path.insert(0, target_dir)   # the project root
        ...exec app.py...                # which imports this module
        sys.path.pop(0)                  # assumed to be target_dir

    The pop is positional. An insert(0) here during exec puts OUR entry at index
    0, so the pop removes src/ instead of the project root -- and because the
    module-level imports below have already succeeded and are cached in
    sys.modules, everything looks fine until the first DEFERRED import runs and
    dies with "No module named 'agent'". Appending leaves index 0 alone.

    Called again at each entry point, because idempotent cheap insurance beats
    depending on another library's list arithmetic.
    """
    if str(SRC) not in sys.path:
        sys.path.append(str(SRC))


ensure_src_path()

# Offline demo mode: replay recorded tool responses instead of calling the
# registry. OFF by default -- the UI must not quietly serve test data, for the
# same reason production does not import fixtures.py. Set CRM_UI_FIXTURES=1
# when the demo room's network cannot be trusted.
USE_FIXTURES = os.environ.get("CRM_UI_FIXTURES") == "1"

# ONE thread owns the backend: it builds the index and runs every turn.
#
# Two separate reasons, and one thread settles both:
#
#   sqlite  index/embedder.py opens its vector cache with sqlite3's default
#           check_same_thread=True. Built on the import thread and used from
#           anyio's worker pool, it raises ProgrammingError on the first query
#           -- the connection is bound to the thread that made it. Building it
#           HERE means the creating thread and the using thread are the same
#           one, so the check passes instead of being switched off.
#
#   qdrant  the local store is single-process and not written for concurrent
#           readers. max_workers=1 serialises turns for free, which is what the
#           old TURN_LOCK did by hand.
#
# The alternative was passing check_same_thread=False into src/. That is a
# change to the backend for the convenience of a UI, and it trades a loud error
# for a silent data race if anything ever runs two turns at once. This keeps
# src/ agnostic, which was the point of the whole layout.
EXECUTOR = ThreadPoolExecutor(max_workers=1, thread_name_prefix="crm-backend")


def _build():
    """Load the corpus and the full retrieval stack. Runs on the backend thread."""
    ensure_src_path()
    from eval.corpus import load as load_corpus
    from index.structured import build as build_retriever

    corpus = load_corpus()
    # CRMAgent builds SectionFetch(AliasExpanding(Hybrid(corpus))) when given no
    # retriever, and passing one SKIPS that wrapping. This is that composition.
    return corpus, build_retriever(corpus)


# Blocking on purpose: a failure here should stop the server at startup, not on
# someone's first question.
CORPUS, RETRIEVER = EXECUTOR.submit(_build).result()

_CLIENT = None


async def run_on_backend(fn, *args):
    """Await a blocking backend call, on the thread that owns the index.

    contextvars are copied across, which is how Chainlit's step context reaches
    the tool wrappers -- run_in_executor does not propagate them by itself, and
    without this the steps would silently stop appearing.
    """
    loop = asyncio.get_running_loop()
    ctx = contextvars.copy_context()
    return await loop.run_in_executor(EXECUTOR, lambda: ctx.run(fn, *args))


def openai_client():
    """One shared client. Thread-safe, and avoids a pool per session."""
    global _CLIENT
    if _CLIENT is None:
        from openai import OpenAI

        key = os.environ.get("OPENAI_API_KEY")
        if not key:
            raise RuntimeError(
                "OPENAI_API_KEY is not set.\n"
                "  PowerShell (persistent): setx OPENAI_API_KEY '<your key>'")
        _CLIENT = OpenAI(api_key=key, max_retries=3, timeout=60.0)
    return _CLIENT


def dispatch_table() -> dict:
    """A fresh copy of the tool table, so wrapping it cannot mutate the real one."""
    ensure_src_path()
    if USE_FIXTURES:
        from agent.tools.fixtures import DISPATCH
    else:
        from agent.tools import DISPATCH
    return dict(DISPATCH)


def new_agent(*, retriever=None, dispatch=None):
    """One conversation's agent, over the shared corpus and index.

    `retriever` and `dispatch` are the injection points the eval harness already
    uses; the UI passes traced wrappers through the same door. src/ is untouched.
    """
    ensure_src_path()
    from agent.agent import CRMAgent

    return CRMAgent(
        corpus=CORPUS,
        retriever=retriever if retriever is not None else RETRIEVER,
        dispatch=dispatch if dispatch is not None else dispatch_table(),
        client=openai_client(),
    )


def _banner() -> None:
    """To the terminal, not the chat: the welcome screen now belongs to the
    starters, and a message there would hide them."""
    print(f"[crm] {describe()}", file=sys.stderr)


def describe() -> str:
    """One line for the startup banner and the UI header."""
    mode = "фикстури (офлајн)" if USE_FIXTURES else "во живо"
    return (f"{len(CORPUS)} блока · {RETRIEVER.name.split('(')[0]} · "
            f"алатки: {mode}")


_banner()
