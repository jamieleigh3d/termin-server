# Changelog

All notable changes to `termin-server` will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.9.1] — 2026-05-01

Correctness + hygiene patch on top of v0.9.0. Closes the audit-trail
gaps surfaced by the Phase 3 conformance pack and one stale-action
hydrator bug surfaced by JL on the warehouse demo.

### Fixed

- **Manual-trigger CEL audit gap** (compute-contract §5.2). v0.9.0's
  ``execute_compute`` only routed to LLM and ai-agent handlers; for
  ``default-CEL`` it printed a warning and silently dropped the call,
  so ``POST /api/v1/compute/<name>/trigger`` for a CEL compute
  returned ``status: completed`` but never wrote an audit row. v0.9.1
  adds ``_execute_cel_compute`` (CEL evaluation + audit write) and
  routes ``provider in (None, "", "cel", "default-CEL")`` to it
  explicitly. The synchronous endpoint at
  ``/api/v1/compute/<name>/`` retains its full pre/postcondition
  + transaction path; ``_execute_cel_compute`` is the slim
  audit-only path for the manual-trigger / event-handler routes.

- **Anonymous principal stamping** (compute-contract §7.1). Audit
  rows stamped ``invoked_by_principal_id`` and
  ``on_behalf_of_principal_id`` as empty strings for anonymous
  callers. v0.9.1 wires anonymous-principal synthesis through the
  core identity layer: ``identity._resolve_principal_and_scopes``
  now calls ``make_anonymous_principal(cookie_name)`` (in
  ``termin-core``) when the resolved role is anonymous, producing
  a Principal with id ``anonymous:<sanitized-marker>``. The
  ``write_audit_trace`` synthesis remains as defense-in-depth for
  any code path that constructs a bare ``ANONYMOUS_PRINCIPAL``
  sentinel directly. Operators filter audit logs with
  ``invoked_by_principal_id LIKE 'anonymous:%'`` to find
  anonymous-caller activity.

- **Tailwind row-actions stale after state transition**
  (`static/termin.js`). v0.9.0's hydrator had a hardcoded
  ``data.status`` precondition gate from the v0.8 single-state-
  machine era. v0.9 multi-state-machine renamed the column
  per machine (``product_lifecycle``, ``approval_status``, etc.),
  so the legacy check always evaluated false and action buttons
  stayed stale until page refresh. v0.9.1 ``updateActionButtons``
  reads each transition span's ``data-machine-name``, looks up
  the row's matching cell, recomputes against the v0.9 nested
  ``transitions[source][machine][from|to]`` shape, and posts to
  the v0.9 4-segment ``/_transition/<source>/<machine>/<id>/<target>``
  URL. Click "Activate" → state cell flips AND action buttons
  re-render with "Discontinue" replacing "Activate" without
  refresh.

- **`surface-as-error` channel failure mode** (channel-contract
  §5.3, BRD §6.4.5). v0.9.0 accepted the grammar but the
  dispatcher unconditionally swallowed exceptions (log-and-drop
  fallback). v0.9.1 reads ``failure_mode`` from each channel
  spec; on ``surface-as-error`` the dispatcher re-raises a
  ``ChannelError(...)`` to the caller with the original
  exception chained via ``__cause__``. Source authors can now
  fail-loud on channels where silent swallowing isn't
  acceptable.

### Changed

- Renamed ``queue-and-retry-forever`` → ``queue-and-retry`` in
  the dispatcher comment block. Grammar accepts the new spelling
  via the ``termin-compiler`` analyzer; v0.9.x falls back to
  log-and-drop with a logged-warning posture distinguishing the
  placeholder from the genuine default. Full retry-worker
  implementation (exponential backoff + dead-letter table after
  configurable max-retry-hours, 24h cap) lands v0.10.

- ``identity.py`` module docstring refreshed to reflect the v0.9
  ``the user`` CEL surface (legacy ``User.PascalCase`` was
  retired in slice 7.5b — TERMIN-S014).

- ``datetime.utcnow()`` (deprecated in Python 3.12, removed in
  3.13) → ``datetime.now(timezone.utc)`` across 16 sites in
  ``compute_runner.py`` (audit timing), ``scheduler.py``,
  ``transaction.py``, and ``pages.py``. Wire format preserved
  byte-for-byte via ``.replace("+00:00", "Z")`` so audit columns,
  CEL ``now`` bindings, and any external consumers see identical
  strings.

### Suite

24 tests passing on Windows (unchanged; the migrations are
wire-format-preserving).

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
