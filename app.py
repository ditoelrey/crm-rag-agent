"""
app.py  --  Chainlit entry point for the ЦРРСМ agent.
=====================================================
Handlers only. Everything with substance lives in ui/, and nothing at all lives
in src/ -- the backend does not know this file exists.

    chainlit run app.py -w

Two things this file is careful about:

  * ask() is synchronous and takes ~9s on average, ~35s at p95, so it is never
    awaited on the event loop -- one slow turn would freeze every other
    session, including during a live demo. It runs on ui.backend's own thread
    rather than through cl.make_async, because the retrieval stack is
    thread-affine: sqlite binds its connection to the thread that opened it.
  * one CRMAgent per chat session, because history and seen_docs are
    per-conversation, over the single corpus and index loaded in ui.backend.
"""
from __future__ import annotations

from functools import partial

import chainlit as cl

from ui import backend, render
from ui.tracing import traced_dispatch, traced_retriever

@cl.set_starters
async def starters():
    """The empty-screen suggestions.

    One per capability, so a first-time visitor sees the range without being
    told: documentation retrieval, a procedure, and a live lookup that has to
    resolve a company NAME to its ЕМБС before it can fetch anything.
    """
    # label == message on purpose: the button says exactly what it will send,
    # so nothing changes under the user between clicking and reading.
    #
    # All-Cyrillic, deliberately. The second question arrived with a LATIN "K"
    # in "Kоја" -- indistinguishable on screen, but this string is the query,
    # and the tokenizer would index "kоја" as a word the corpus does not
    # contain. The third drops "ДОО Скопје": partial names resolve against the
    # full legal title, so the shorter form still finds ЕМБС 7696876.
    questions = [
        "Кои документи ми се потребни за основање на здружение?",
        "Која е постапката за регистрирање на ТП?",
        "Дај ми ги основните податоци за компанијата ЛОРА КОМПАНИ 2023.",
    ]
    return [cl.Starter(label=q, message=q) for q in questions]


@cl.on_chat_start
async def start():
    """One agent per conversation, wired to the shared corpus and index.

    Deliberately sends NO welcome message: Chainlit shows the starters only
    while the thread is empty, so greeting the user would hide the three
    buttons that greet them better. The introduction lives in chainlit.md.
    """
    def build():
        return backend.new_agent(
            retriever=traced_retriever(backend.RETRIEVER),
            dispatch=traced_dispatch(backend.dispatch_table()),
        )

    cl.user_session.set("agent", await backend.run_on_backend(build))


@cl.on_message
async def on_message(message: cl.Message):
    agent = cl.user_session.get("agent")
    if agent is None:      # on_chat_start failed; say so instead of AttributeError
        await cl.Message(content="⛔ Агентот не е иницијализиран. "
                                 "Погледнете го логот на серверот.").send()
        return

    try:
        # One worker thread owns the index, so turns are serialised and every
        # sqlite handle is touched by the thread that opened it.
        answer = await backend.run_on_backend(partial(agent.ask, message.content))
    except Exception as exc:
        # A dead integration must be visible, not narrated around -- the same
        # rule _run_tool applies inside the agent.
        await cl.Message(
            content=f"⛔ Барањето не успеа: `{type(exc).__name__}: {exc}`"
        ).send()
        raise

    # The raw chunk_ids in the prose are replaced by [Извор 1], [Извор 2] ...
    # and each one becomes a side panel. Chainlit links an element wherever its
    # NAME appears in the message, and the name is the marker itself, so the
    # inline markers are the links. The ids stay visible under Извори.
    body, sources = render.prepare(answer)
    elements = [cl.Text(name=name, content=content, display="side")
                for name, content in sources]

    if answer.asked_for_clarification:
        # Worth its own marker: this turn never reached the model. The
        # ambiguity detector answered it structurally, which is why the footer
        # reports zero tokens.
        body = "❓ **Потребно е појаснување**\n\n" + body

    await cl.Message(content=body, elements=elements).send()
