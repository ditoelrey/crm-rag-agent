"""
serve.py  --  run the UI without Chainlit's CLI.
================================================
USE THIS INSTEAD OF `chainlit run app.py`.

Chainlit's command line entry point calls nest_asyncio.apply() at import
(chainlit/cli/__init__.py:11). On this stack -- Python 3.14, anyio 4.15,
starlette 1.7 -- that patch breaks anyio's event-loop detection, so every call
to anyio.to_thread.run_sync raises

    NoEventLoopError: Not currently running on any asynchronous event loop

Starlette's StaticFiles uses exactly that call to stat a file, so Chainlit's own
JavaScript bundle returns HTTP 500 and the page renders blank. It happens to any
Chainlit app on this stack, with or without our code. Reproduced in three lines:

    nest_asyncio.apply(); asyncio.run(anyio.to_thread.run_sync(os.stat, "x"))

mount_chainlit() gives the same UI without importing chainlit.cli, so
nest_asyncio is never applied and static files serve normally.

    python serve.py

Reload-on-edit is what `-w` gave us and this drops; pass an import string to
uvicorn.run with reload=True if you want it back, at the cost of restarting the
6-second corpus load on every save.
"""
from __future__ import annotations

import os

import uvicorn
from chainlit.utils import mount_chainlit
from fastapi import FastAPI

app = FastAPI()
mount_chainlit(app=app, target="app.py", path="")

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1",
                port=int(os.environ.get("PORT", "8000")), log_level="info")
