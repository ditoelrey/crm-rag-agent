"""
vision.py  --  the shared step for tools whose only source is an image.
=====================================================================
Forms 2 and 3 both render as PNGs behind reCAPTCHA, so both are read offline
from images an operator saved and both go through the same three steps:

    read_png()             bytes off disk, refusing anything that is not a PNG
    extract_structured()   gpt-4o vision -> a Pydantic model, strict JSON schema
    ImageCache             one vision call per file version, not per question

One implementation on purpose. Two copies of the vision call is two places for
the prompt plumbing, the size cap and the strict-schema wiring to drift apart,
and the drift would surface as one form reading images worse than the other for
no reason anyone could find.

What does NOT live here: each form's schema, prompt, self-check and result
types. Those are the parts that differ, and they are where the correctness
argument for each form is made.
"""
from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Callable, TypeVar

from pydantic import BaseModel

from .schema import strict_schema

VISION_MODEL = "gpt-4o"
# gpt-4o prices images as input tokens; one registry table is ~1-1.5k. Cheap per
# call, but not free per eval run -- which is why the gates replay fixtures.
MAX_IMAGE_BYTES = 6_000_000
PNG_MAGIC = b"\x89PNG\r\n\x1a\n"

T = TypeVar("T", bound=BaseModel)


def read_png(path: str | Path) -> bytes:
    """Read an image an operator saved from the portal."""
    data = Path(path).read_bytes()
    if not data.startswith(PNG_MAGIC):
        raise ValueError(f"{path} is not a PNG")
    return data


def extract_structured(image_bytes: bytes, schema: type[T], prompt: str, *,
                       name: str, client=None, model: str = VISION_MODEL) -> T:
    """Read an image into `schema`. The one non-deterministic step.

    temperature=0 and strict structured outputs narrow what the model can
    return, but do not make the read correct -- that is what each form's own
    self-check is for.
    """
    if len(image_bytes) > MAX_IMAGE_BYTES:
        raise ValueError(f"image is {len(image_bytes)} bytes, over the cap")
    if client is None:
        import os

        from openai import OpenAI
        client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    b64 = base64.b64encode(image_bytes).decode("ascii")
    resp = client.chat.completions.create(
        model=model,
        temperature=0,
        messages=[{"role": "user", "content": [
            {"type": "text", "text": prompt},
            {"type": "image_url",
             "image_url": {"url": f"data:image/png;base64,{b64}"}},
        ]}],
        response_format={"type": "json_schema", "json_schema": {
            "name": name,
            "schema": strict_schema(schema),
            "strict": True,
        }},
    )
    return schema(**json.loads(resp.choices[0].message.content))


class ImageCache:
    """Extraction results, one per version of each file.

    Vision costs a call and ~3s, and the same document asked about twice in a
    session should not pay twice. The key includes mtime and size, so replacing
    a PNG invalidates its entry instead of serving an old read of a file that
    has changed underneath it.
    """

    def __init__(self) -> None:
        self._store: dict[tuple[str, int, int], BaseModel] = {}

    def get(self, path: Path, extract: Callable[[bytes], T]) -> T:
        stat = path.stat()
        key = (str(path), stat.st_mtime_ns, stat.st_size)
        if key not in self._store:
            self._store[key] = extract(read_png(path))
        return self._store[key]  # type: ignore[return-value]
