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

**v0.9.2 — released 2026-05-05.** The conversation-field runtime
release. Implements the runtime side of the v0.9.2 IR additions
landed in `termin-compiler` and `termin-core`: SQL storage for the
new `structured` and `conversation` base types, the
`POST <resource>/{id}/<field>:append` REST handler + WebSocket
parity frame, the `content.<source>.<field>.appended` event
channel, conversation materialization to Anthropic with
auto-write-back per §11.5, token streaming for conversation-mode
agents, the chat presentation hydrator, and the
`system_refuse(reason)` → terminate-loop contract enforcement. Also
renames the canonical kind `assistant` → `agent` (with
`assistant` accepted as legacy back-compat) and rebrands the chat
UI label to "AI Agent." See [CHANGELOG.md](CHANGELOG.md) for the
full list. 98 tests passing.

### v0.9 release arc

- **v0.9.0** (2026-04-30) — opening release. Phase 7 of the v0.9
  Termin milestone extracted this package out of
  `termin-compiler/termin_runtime/` (slice 7.3); the legacy shim
  was dropped in slice 7.5a. `termin-compiler` now imports
  `from termin_server import create_termin_app` directly;
  alternate runtimes built on `termin-core` do not depend on
  `termin-server`.
- **v0.9.1** (2026-05-01) — correctness + hygiene patch. Closed
  audit-trail gaps surfaced by the Phase 3 conformance pack
  (manual-trigger CEL audit, anonymous-principal stamping),
  fixed the stale-action hydrator on warehouse, and made
  `surface-as-error` channel failure mode behave deterministically
  per the contract. 24 tests.
- **v0.9.2** (2026-05-05) — conversation-field runtime; see above.

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
