"""
tools  --  live lookups against the ЦРРСМ public forms.
=======================================================
One module per form. Each exposes the same three things:

    <Args>       a Pydantic model, validated before any request leaves
    fetch_*()    deterministic I/O + parsing, no LLM anywhere in it
    TOOL_SPEC    the OpenAI function definition -- the only part the model sees

REGISTRY is what the agent passes as `tools=`, and DISPATCH maps a tool name to
its callable. Adding a form means adding a module and two lines here; nothing in
the agent loop needs to know which forms exist.

Form 2 is registered but runs OFFLINE: its endpoint sits behind reCAPTCHA v3,
so it reads profiles from PNGs an operator saved out of the portal and reports
a clean "not available" for anything it does not hold. The tool is registered
anyway because a partial answer the agent can give today beats a complete one
it cannot -- and because the source seam (entity_profile.ProfileSource) means
the day an official feed arrives, default_source() changes and nothing else does.

Form 3 (Објави на уписи -- a published Решение) runs offline for the same reason
and in the same way, but is keyed by the 14-digit деловоден број rather than the
ЕМБС: one entity accumulates many decisions, and the endpoint addresses a filing.
search_announcements resolves an entity to those numbers, so the pair is used in
sequence -- search, then fetch -- which is what MAX_TOOL_ROUNDS already allows.

Form 4 (Статус на предмет) is here now, offline like the rest. Its statusInfo
endpoint looked unprotected in the capture; asked directly, it answers
412 "Recaptcha token missing", so the live GET stays dormant and the tool reads
statuses an operator saved. It is the one form with no vision step: the view is
text, so what the agent reports is the saved bytes rather than a read of a
picture.
"""
from __future__ import annotations

from . import (announcement_search, entity_profile, entity_search, entity_size,
               registration_decision, status_info)

REGISTRY = [entity_size.TOOL_SPEC, entity_profile.TOOL_SPEC,
            registration_decision.TOOL_SPEC, announcement_search.TOOL_SPEC,
            entity_search.TOOL_SPEC, status_info.TOOL_SPEC]

DISPATCH = {
    "check_entity_size": lambda **kw: entity_size.fetch_entity_size(**kw),
    "get_entity_profile": lambda **kw: entity_profile.fetch_profile(**kw),
    "get_registration_decision":
        lambda **kw: registration_decision.fetch_decision(**kw),
    "search_announcements":
        lambda **kw: announcement_search.search_announcements(**kw),
    "search_entity_profile":
        lambda **kw: entity_search.search_entity_profile(**kw),
    "get_status_info": lambda **kw: status_info.fetch_status(**kw),
}

__all__ = ["REGISTRY", "DISPATCH", "announcement_search", "entity_profile",
           "entity_search", "entity_size", "registration_decision",
           "status_info"]
