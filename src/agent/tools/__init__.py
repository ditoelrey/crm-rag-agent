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

Form 4 (Статус на обработка) is deliberately absent: it sits behind reCAPTCHA,
and working around bot protection is not on the table. If the registry provides
an API for it, or the user completes the challenge themselves, it can be added
here like any other.
"""
from __future__ import annotations

from . import entity_profile, entity_size

REGISTRY = [entity_size.TOOL_SPEC, entity_profile.TOOL_SPEC]

DISPATCH = {
    "check_entity_size": lambda **kw: entity_size.fetch_entity_size(**kw),
    "get_entity_profile": lambda **kw: entity_profile.fetch_profile(**kw),
}

__all__ = ["REGISTRY", "DISPATCH", "entity_profile", "entity_size"]
