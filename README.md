# termin-server

The Termin reference hosting layer. Provides:

- The FastAPI app factory (`create_termin_app`) that turns a compiled
  `.termin.pkg` IR into a running web app.
- IO-bound builtin providers — SQLite storage, Anthropic LLM/agent
  compute, Tailwind SSR presentation, channel webhook/email/messaging
  stubs.
- Static client assets (`termin.js`, bootstrap shim) and Jinja2
  templates the SSR renderer consumes.
- A FastAPI WebSocket adapter that wraps `fastapi.WebSocket` as a
  `termin_core.routing.TerminWebSocket`.

`termin-server` is one possible host for an application compiled by
`termin-compiler`. Any other runtime that imports `termin-core` and
supplies its own framework adapter is also a conforming Termin
runtime — `termin-server` is the reference, not the contract.

## Layering

```
termin-compiler (.termin → IR + .termin.pkg)
       │
       ▼
termin-core (Provider Protocols, IR types, routing dispatch,
             expression eval, confidentiality, errors)
       │
       ▼
termin-server (FastAPI app, builtin providers, static, CLI)
```

Conformance:

```
termin-conformance (spec + tests; runs against any termin-core-conforming
                    adapter — `reference` adapter binds to termin-server)
```

## Status

**v0.9.0 — released 2026-04-30.** Phase 7 of the v0.9 milestone
extracted this package out of `termin-compiler/termin_runtime/`
(slice 7.3) and dropped the back-compat shim layer in
`termin-compiler` (slice 7.5a). `termin-compiler` now imports
`from termin_server import create_termin_app` directly; alternate
runtimes built on `termin-core` do not depend on `termin-server`.

## Quick start

```bash
pip install termin-server termin-compiler

# Compile a .termin source to a .termin.pkg
termin compile examples/warehouse.termin

# Serve the compiled package
termin serve warehouse.termin.pkg --port 8000
```

Or programmatically:

```python
from termin_server import create_termin_app

with open("warehouse.termin.pkg", "rb") as f:
    # ... unpack the IR JSON from the .termin.pkg ZIP ...
    app = create_termin_app(ir_json)

# `app` is a FastAPI instance — serve via uvicorn or any ASGI server.
```

## Tests

```bash
pip install -e ".[test]"
pytest tests/
```

24 own-repo tests across smoke, storage unit, and HTTP integration
tiers — the layer below the [conformance suite](https://github.com/jamieleigh3d/termin-conformance).

## License

Apache-2.0. See LICENSE.
