"""
schema.py  --  Pydantic -> OpenAI schemas, for tools and for vision extraction.
==============================================================================
Two things model_json_schema() does that OpenAI's strict modes reject or punish:

  * it copies the CLASS docstring into `description`, shipping our internal
    commentary to the model on every call;
  * it omits `additionalProperties: false`, which both strict tool definitions
    and structured outputs require -- the API rejects the request without it.

Field descriptions are kept: those are written FOR the model.
"""
from __future__ import annotations

from pydantic import BaseModel


def strict_schema(model: type[BaseModel]) -> dict:
    """A schema OpenAI accepts in strict mode. See module docstring."""
    schema = model.model_json_schema()
    schema.pop("description", None)
    schema.pop("title", None)
    for prop in schema.get("properties", {}).values():
        prop.pop("title", None)
        # Structured outputs reject `anyOf` containing null for optional fields
        # unless every branch is explicit; a nullable string is simpler as a
        # plain string the extractor fills with "" when a row is absent.
        prop.pop("default", None)
    schema["additionalProperties"] = False
    # Strict mode requires EVERY property in `required`, optional or not.
    schema["required"] = list(schema.get("properties", {}))
    return schema
