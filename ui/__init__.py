"""
ui  --  the Chainlit presentation layer.
========================================
Nothing under src/ imports anything from here, and nothing here is imported by
the agent, the eval harness or the gates. The dependency runs one way only:

    app.py -> ui/ -> src/

Three modules, split by what they depend on:

    backend.py   process-wide singletons and the agent factory. No Chainlit.
    render.py    pure formatting of an Answer. No Chainlit, so it is testable.
    tracing.py   the wrappers that emit Chainlit steps. Chainlit-only.

The split matters for one practical reason: `render` and `backend` can be
imported and exercised without a Chainlit server running, which keeps the part
that touches the agent honest.
"""
