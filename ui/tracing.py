"""
tracing.py  --  making the agent's steps visible, without the agent knowing.
============================================================================
CRMAgent.ask() is one blocking call that returns a finished Answer, so a chat UI
wrapped around it shows a spinner for up to ~35 seconds and then a wall of text.
Nothing in the Answer arrives early enough to narrate the turn.

The way in is the injection the eval harness already uses. `retriever` and
`dispatch` are constructor arguments, so the UI can hand the agent wrapped
versions that report to Chainlit on the way past:

    CRMAgent(retriever=traced_retriever(R), dispatch=traced_dispatch(D))

src/ stays untouched, and the same seam that lets the gates replay fixtures
lets the UI draw the execution.

RUNNING FROM A WORKER THREAD
----------------------------
ask() is synchronous and is called through cl.make_async, so these wrappers run
in a thread, not on the event loop. Chainlit copies its context into that thread
and `with cl.Step(...)` works there. If it ever does not -- a version change, a
call from some other thread -- every step here degrades to a no-op rather than
taking the turn down with it. A tracing bug must never cost the user an answer.
"""
from __future__ import annotations

from contextlib import contextmanager

import chainlit as cl

# What each tool is doing, in the user's language. A tool absent from this map
# still gets a step, under its own name.
_TOOL_LABELS = {
    "check_entity_size": "Проверка на големина на субјект",
    "get_entity_profile": "Читање основен профил",
    "get_registration_decision": "Читање решение за упис",
    "search_announcements": "Пребарување објави на уписи",
    "search_entity_profile": "Пребарување субјект по назив",
    "get_status_info": "Проверка на статус на предмет",
}


# Shown the moment a tool starts, replaced by the result when it finishes.
# Reading a profile or a decision means a vision call: 3-4 seconds on its own,
# and a question that chains a search into a fetch can run half a minute. A
# spinner with no words reads as a hung app long before that.
WAIT_NOTICE = ("⏳ Се процесира, ве молиме почекајте — ова може да потрае "
               "(читање на документ од Централниот регистар).")


def _progress(step, text: str) -> None:
    """Push an interim output to a step that is still running.

    Step.update() is async and we are on the backend worker thread, so it goes
    through cl.run_sync. Failing silently is deliberate: a progress notice is
    never worth losing an answer over.
    """
    try:
        step.output = text
        cl.run_sync(step.update())
    except Exception:
        pass


@contextmanager
def _step(name: str, *, type_: str = "tool", input_: str = ""):
    """A Chainlit step that cannot break the turn it is describing."""
    try:
        with cl.Step(name=name, type=type_) as step:
            if input_:
                step.input = input_
            yield step
            return
    except Exception:
        pass
    # Chainlit unavailable or no context in this thread: run the body anyway.
    class _Silent:
        input = output = ""
    yield _Silent()


def _summarise(name: str, result) -> str:
    """One line describing what a tool returned, without dumping the payload."""
    if result is None:
        return "нема резултат"
    hits = getattr(result, "hits", None)
    if hits is not None:
        if not hits:
            return "нема пронајдено записи во офлајн архивата"
        found = [getattr(h, "deloveden_broj", None) or getattr(h, "embs", "?")
                 for h in hits]
        return f"{len(hits)} пронајдено: " + ", ".join(str(f) for f in found[:5])
    if getattr(result, "available", True) is False:
        return "не е достапно офлајн"
    for attr in ("size", "status"):
        value = getattr(result, attr, None)
        if value:
            return f"{attr}: {value}"
    info = getattr(result, "info", None) or getattr(result, "profile", None)
    if info is not None:
        return "податоците се прочитани"
    decision = getattr(result, "decision", None)
    if decision is not None:
        return f"решение {decision.deloveden_broj} прочитано"
    return "готово"


def traced_dispatch(table: dict) -> dict:
    """The tool table, each entry wrapped in a Chainlit step.

    A failing tool re-raises after the step records the error: the agent's own
    rule is that a broken lookup must not be narrated around, and that rule
    belongs to the agent, not to the UI.
    """
    def wrap(name, fn):
        def traced(**kwargs):
            shown = ", ".join(f"{k}={v}" for k, v in kwargs.items() if v)
            label = _TOOL_LABELS.get(name, name)
            with _step(label, input_=shown or name) as step:
                _progress(step, WAIT_NOTICE)
                try:
                    result = fn(**kwargs)
                except Exception as exc:
                    step.output = f"⛔ {type(exc).__name__}: {exc}"
                    raise
                step.output = _summarise(name, result)
                return result
        return traced

    return {name: wrap(name, fn) for name, fn in table.items()}


class TracedRetriever:
    """The retriever, reporting each search as a step.

    Delegates everything it does not override, so the structured-fetch and alias
    layers keep whatever attributes the agent reads off them.
    """

    def __init__(self, inner):
        self._inner = inner

    def __getattr__(self, item):
        return getattr(self._inner, item)

    def search(self, query, k=10, *, filters=None):
        detail = query if not filters else f"{query}   ⟨{filters}⟩"
        with _step("Пребарување низ документацијата", type_="retrieval",
                   input_=detail) as step:
            hits = self._inner.search(query, k, filters=filters)
            step.output = f"{len(hits)} блока"
            return hits


def traced_retriever(inner) -> TracedRetriever:
    return TracedRetriever(inner)
