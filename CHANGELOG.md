# Changelog

All notable changes to `termin-server` will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.9.0] — 2026-04-30

The opening release of `termin-server`. Phase 7 of the v0.9 Termin
milestone (slice 7.3, 2026-04-30) extracted the FastAPI hosting
layer + IO-bound builtin providers out of `termin-compiler/termin_runtime/`
into this sibling repo. `termin-compiler` now depends on
`termin-server>=0.9.0`; the legacy `termin_runtime/` shim layer that
carried the transition was deleted in slice 7.5a.

The package surface is one factory function — `create_termin_app(ir_json)`
— that turns a compiled `.termin.pkg` IR into a FastAPI app. Internally
it wires up:

- **Storage** — SQLite + aiosqlite, schema migrations, generic CRUD.
- **Identity** — cookie-based role resolution, anonymous fallback, the
  v0.9 `the user` shape on the CEL surface (BRD #3 §4.2).
- **Routing** — generated CRUD routes per content, page rendering,
  `/api/v1/_runtime/*` reflection endpoints, WebSocket dispatcher,
  `/_transition/<table>/<machine>/<id>/<state>` state-machine writes.
- **Presentation** — Tailwind SSR via the built-in
  `tailwind-default` provider (registered through the same
  `termin.providers` entry-point group external providers like
  `termin-spectrum-provider` use).
- **Compute** — Anthropic LLM/agent compute provider, channel
  webhook/email/messaging stubs.
- **Errors** — TerminAtor router that classifies errors into the
  framework-agnostic envelope and lets per-app rules redirect.

### Added — own-repo test suite

Slice 7.3's release-day audit surfaced a gap: `termin-server` had no
own-repo tests, only transitive coverage through `termin-compiler`'s
suite and `termin-conformance` against the reference adapter. The
v0.9.0 release adds 24 tests in `tests/`:

- **7 smoke tests** (`tests/test_smoke.py`) — package surface, the
  no-`termin_runtime`-import guard locking in slice 7.5a's deletion,
  `create_termin_app` returning a `FastAPI` app, route registration,
  invalid-IR fail-fast.
- **8 storage unit tests** (`tests/test_storage_unit.py`) — identifier
  safety guards (the SQL-injection rejection layer), CRUD round-trip
  against an ephemeral SQLite, the `get_record` 404-on-miss contract.
- **9 integration tests** (`tests/test_integration.py`) — runtime
  registry + bootstrap endpoints, full POST/GET/PUT/DELETE round-trip
  on the warehouse fixture's `products` table, page rendering with
  `data-termin-*` marker presence, anonymous-request smoke.

These are the layer below conformance: a regression in the storage
layer or a missing `data-termin-*` marker lights up here in 4
seconds without needing the full conformance run. They depend on
two `.termin.pkg` fixtures (`hello`, `warehouse`) checked into
`tests/fixtures/`.

### Compatibility

Requires Python 3.10+. Depends on `termin-core>=0.9.0,<0.10`,
FastAPI, uvicorn, aiosqlite, Jinja2, Anthropic.
