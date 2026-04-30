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

`v0.9` (Phase 7 in flight). Slice 7.3 of Phase 7 (2026-04-30) extracted
this package out of `termin-compiler/termin_runtime/`. The Python
package import is `termin_server`; the legacy `termin_runtime` module
in `termin-compiler` re-exports from this package as a back-compat
shim and drops in slice 7.5.

## License

Apache-2.0. See LICENSE.
