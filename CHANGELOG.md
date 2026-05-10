# Changelog

All notable changes to `termin-server` will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **`termin_server.__version__`** is now declared in
  ``termin_server/__init__.py`` (was missing pre-v0.9.4). All
  in-package callers that need the package version
  (``runtime_version`` reflection in ``routes.py``, the
  ``version=`` kwarg on every built-in provider's
  ``ProviderRecord`` registration, test assertions on
  ``runtime_version`` / ``provider.version``) now import this
  single canonical value instead of hardcoding the literal. Per
  ``termin-compiler/docs/version-policy.md`` §2.1: the package
  version has exactly one source of truth and everything else
  imports.

### Changed

- **`runtime_version` reflection** (``routes.py:943``) reads
  from ``termin_server.__version__`` instead of the hardcoded
  ``"0.9.2"`` literal. v0.9.3 shipped with the literal lagging the
  package version (the live server reported ``0.9.2`` while the
  installed package was ``0.9.3``); the import-from-canonical
  pattern makes the value automatically track the package on
  every release.
- **All eleven built-in providers** under
  ``termin_server/providers/builtins/`` (``compute_default_cel``,
  ``compute_llm_anthropic`` / ``_stub``, ``compute_agent_anthropic``
  / ``_stub``, ``identity_stub``, ``channel_webhook_stub`` /
  ``_messaging_stub`` / ``_email_stub``, ``presentation_tailwind_default``,
  ``storage_sqlite``) read ``version=__version__`` from
  ``termin_server`` instead of hardcoding the literal. Same
  drift-correction motivation as ``runtime_version`` above.
- **Test pin in ``test_integration.py::test_runtime_registry_returns_json``**
  now asserts against the imported ``__version__`` rather than a
  literal so the test moves with the package without requiring a
  release-time bump.

### Compatibility

- Wire-shape unchanged. Pre-fix: live server reported
  ``runtime_version="0.9.2"`` and registered providers with
  ``version="0.9.2"``. Post-fix: both report ``"0.9.3"`` (the
  current package version). Consumers that pinned to ``"0.9.2"``
  on the wire will need to update their expectations — this is
  the *correction* of the v0.9.3 ship drift, not a new
  contract change.
- New companion doc at
  ``termin-compiler/docs/version-policy.md`` enumerates the
  source-of-truth conventions for both version tracks (package
  vs IR). ``release.py`` now matches the policy.

### Fixed

- **SQLite is now the per-runtime serialization boundary for
  list/dict-typed fields (issue #5).** Storage Protocol callers in
  framework-free code (the v0.9.3 ``termin_core.routing.append``
  helper, CRUD handlers, channel handlers) pass *native* Python
  objects per the Provider contract — each storage implementation
  owns its own serialization. Pre-fix, ``aiosqlite`` rejected
  list/dict parameter bindings outright with
  ``sqlite3.ProgrammingError: Error binding parameter N: type
  'list' is not supported``, blocking the v0.9.3 ``append_to_field``
  from being adoptable end-to-end. Added a small
  ``_serialize_for_sqlite`` helper at the parameter-binding
  boundary in ``termin_server.storage`` and applied it to
  ``create_record``, ``update_record``, ``update_fields``, and
  ``insert_raw``. Native list/dict patch values are now JSON-encoded
  here on the way to ``aiosqlite``; primitives pass through
  unchanged. The SQLite ``StorageProvider`` (which delegates to
  these helpers) inherits the fix automatically.
- **Legacy ``_do_append`` in ``termin_server.routes`` brought into
  lock-step with ``termin_core.routing.append.append_to_field``
  (issue #5).** The dual-implementation path that
  ``termin-server`` keeps for the SQLite event-suppression hook
  also had the same SQLite-specific ``json.dumps`` / ``json.loads``
  leak. Both implementations now share the same
  storage-Protocol-shaped contract: read accepts native list /
  JSON text / None / empty / malformed-degraded-to-empty; write
  passes a native list. The SQLite serialization happens at the
  ``_serialize_for_sqlite`` boundary above.
- 4 new unit tests in
  ``tests/test_storage_unit.py::TestStructuredFieldSerialization``
  pin the SQLite-side serialization contract:
  ``update_record`` / ``update_fields`` / ``insert_raw`` accept
  native lists and dicts; primitives still round-trip unchanged.

### Compatibility

- Backwards-compatible. SQLite-stored TEXT columns continue to
  hold the same JSON-encoded shape they did pre-fix; the
  difference is that the SQLite layer now does the encoding
  instead of leaving it to the caller. No data migration needed.
- Pair-fix in ``termin-core`` v0.9.3-Unreleased makes
  ``append_to_field`` storage-Protocol agnostic. The two changes
  ship as one logical fix split across the package boundary.

## [0.9.3] — 2026-05-07

The runtime extraction release. Internal API surface only — no IR
change, no DSL change. Per `RELEASE_PROCESS.md` §2 (in the
compiler repo), this is a patch release: additive Python API, no
removal of public surface. Tech design at
`termin-compiler/docs/termin-v0.9.3-runtime-extraction-tech-design.md`.

`termin-server` loses ~3500 lines of framework-free orchestration
code to `termin-core` and drops 16 re-export shim files (the
slice 7.1 shim layer that landed in v0.9.0). Server-internal
imports + `termin-spectrum-provider` + `termin-conformance` updated
to import from `termin_core.X` directly.

### Removed (the no-shims sweep)

- `termin_server.errors`, `.state`, `.validation`, `.expression`,
  `.confidentiality`, `.cel_predicate` — all six were
  `from termin_core.X import *` shims. Deleted.
- `termin_server.providers.binding`, `.contracts`,
  `.deploy_config`, `.registry`, `.storage_contract`,
  `.identity_contract`, `.channel_contract`, `.compute_contract`,
  `.presentation_contract` — all ten were re-exports of
  `termin_core.providers.X`. Deleted. The `providers/` package
  itself stays as a marker; concrete IO providers
  (`builtins/storage_sqlite.py`, `builtins/compute_*.py`,
  `builtins/channel_*_stub.py`, etc.) continue to live there.

### Moved to `termin-core`

- `events.py`, `scheduler.py`, `transaction.py`, `reflection.py`
  — runtime infrastructure.
- `boundaries.py`, `colorblind.py`, `markdown_sanitizer.py` —
  security + accessibility primitives.
- `migrations/` — IR migration package.
- `channels.py`, `channel_config.py`, `channel_ws.py` — channel
  dispatch.
- `pages.build_compute_js` — extracted to
  `termin_core.expression.compute_js`.
- `pages.extract_page_reqs` — extracted to
  `termin_core.presentation.compose`. The `build_*template`
  Jinja-binding functions stay here; they return Jinja2
  `Template` objects and use the Jinja-bound `render_component`
  dispatch table.
- `ai_provider.py` SDK-agnostic helpers (`materialize_to_anthropic`,
  `entry_role`, `build_content_blocks`,
  `build_invokable_compute_tools`, `truncate_purpose`,
  `purpose_property`, `add_purpose_to_tool`, plus the canonical
  kind sets and `ConversationMaterializationError`) — extracted to
  `termin_core.compute.materialize`. The Anthropic SDK call site
  stays here under `AIProvider`. `build_agent_tools` and
  `build_output_tool` keep their richer server-local versions
  (with `state_transition`, `system_refuse`, per-content schema
  elaboration); core ships scaffold versions for alt runtimes to
  extend.

### Kept (with rationale)

- `_do_append` in `routes.py` — server-local parallel
  implementation. Uses `aiosqlite`-direct `update_record(...,
  event_bus=None)` to suppress the standard `_updated` event so it
  doesn't double-fire alongside the field-specific `appended`
  event. Core's `append_to_field` uses `ctx.storage.update(...)`
  via the StorageProvider Protocol, which doesn't expose the
  event-suppression hook today. v0.10 cleanup either teaches
  `StorageProvider.update` to accept the flag or refactors the
  reference runtime to fire one event from the append path.
- `presentation.py` Jinja2 SSR renderer — bound to Jinja2 by
  design.
- `storage.py` (aiosqlite), `app.py`, `bootstrap.py`, `routes.py`,
  `pages.py` (FastAPI handlers), `identity.py`, `transitions.py`,
  `websocket_manager.py`, `presentation_bundles.py`,
  `fastapi_adapter.py` — framework-bound modules stay where they
  are.
- `providers/builtins/` — concrete IO providers (SQLite storage,
  Anthropic LLM/agent, Tailwind SSR renderer, channel stubs).

### Test count

- 98 passing.

## [0.9.2] — 2026-05-05

Conversation-field runtime release. Implements the runtime side of
the v0.9.2 IR additions landed in `termin-compiler` and `termin-core`:
SQL storage for the new `structured` and `conversation` base types,
the `POST <resource>/{id}/<field>:append` REST handler + WebSocket
parity frame, the `content.<source>.<field>.appended` event channel,
conversation materialization to Anthropic with auto-write-back per
§11.5, token streaming for conversation-mode agents, and the chat
presentation hydrator. Also lands the `system_refuse(reason)` →
hard-stop contract enforcement, the `assistant` → `agent` kind
rename with UI rebrand to "AI Agent", and a clutch of chat-surface
fixes from JL's manual testing.

### Added

- **L1 — `structured` SQL type** (`storage.py::_SQL_TYPES`). Maps
  the new IR base type to TEXT storage; serialized/deserialized
  via JSON at the boundary.
- **L2 — `conversation` SQL type** (`storage.py::_SQL_TYPES`).
  Same TEXT storage as `structured`; the runtime treats the JSON
  shape as a typed list of entries with the §3.2 closed kind enum.
- **L3 — append handler** (`routes.py::_do_append`).
  `POST /api/v1/<resource>/{id}/<field>:append` reads the request
  body as a partial entry envelope, validates against the
  conversation-entry shape (kind enum, body shape, optional
  parent_id / tool_call_id / purpose linkage), assigns id +
  timestamp, appends to the parent record's conversation field,
  writes a `Verb.APPEND` audit row, and returns the materialized
  `appended_entry`.
- **L4 — WebSocket append frame parity** (`conn_manager.py`).
  Clients can append via the same WS that drives subscriptions:
  `{type: "append", resource, id, field, payload}`. Same gating
  as the REST path, same audit trail. The chat hydrator uses this
  frame so the user→assistant turnaround happens on one socket
  with no extra HTTP round-trip.
- **L5 — `content.<source>.<field>.appended` event publish**
  (`routes.py::_do_append`). Every successful append publishes
  the appended entry on the `content.<source>.<field>.appended`
  event-bus channel. Author-declared When rules with
  `where appended_entry.kind == "..."` predicates fire from
  this; the chat hydrator's WS subscription consumes it for
  live UI updates.
- **L7.1+L7.2+L7.3 — conversation materialization to Anthropic**
  (`ai_provider.py::materialize_to_anthropic`). Walks the
  conversation entries and emits an Anthropic `messages` list
  with adjacent-role merging, `tool_use` / `tool_result`
  pairing validation, and orphan-tool-call dropping (legacy
  data hygiene). Honors the closed kind enum: `user` → `user`,
  `agent` / `assistant` → `assistant`, `tool_call` → `assistant`
  with `tool_use` block, `tool_result` → `user` with
  `tool_result` block, `system_event` → skipped (consumed by the
  audit layer, not the agent context).
- **§11.5 auto-write-back** (`compute_runner.py::_on_writeback`).
  After the agent loop terminates, the runtime appends the
  agent's text reply to the conversation field as a fresh
  `kind="agent"` entry — the author-declared compute does not
  have to write the reply back manually. Works for both
  blocking and streaming providers.
- **L7.5 — refusal as `kind="agent"` `type="refusal"`**
  (`compute_runner.py`). `system_refuse(reason)` now records the
  refusal as a conversation entry instead of writing to the
  retired `compute_refusals` sidecar. The conversation timeline
  carries the refusal next to the user message that triggered
  it; audit-trail coverage is unchanged.
- **L8 — When-rule action dispatch + Append-action executor**
  (`compute_runner.py`). When rules with action lists land each
  action through the verb-specific dispatcher; Append-actions
  go through the same `_do_append` path as the REST/WS surface.
- **L9 — Tailwind chat hydrator for the v0.9.2 conversation
  binding** (`presentation.py::_render_chat_conversation`,
  `static/termin.js::hydrateConversationFieldChat`). SSR shell
  carries `data-termin-source` + `data-termin-conversation-field`;
  the hydrator subscribes to the L5 event channel, fetches the
  active record's conversation, renders one bubble per entry,
  and binds the input form to the L4 WS append frame.
- **L11 — agent_chatbot end-to-end coverage**
  (`tests/test_l11_agent_chatbot.py`). Tests append authorization,
  append validation, append → trigger → agent_loop → write-back
  round-trip, streaming delta delivery, thread switching, history
  load, refusal termination.
- **`purpose` field on tool_call entries** (close-out task).
  Optional 6-word-or-less display string for tool-call entries
  in the conversation timeline. Hard-truncated at 12 words with
  ellipses. Tool-binding wires Anthropic's tool-use blocks to
  display the `purpose` instead of the full tool input when
  rendering bubbles.
- **`Invokes "<compute>"` runtime wiring** (close-out task).
  When an `ai-agent` Compute declares `Invokes "X"`, the
  runtime registers `X` as a tool on the agent's tool list with
  the gating from `X`'s access rules. Tool-call entries carry
  the matching `purpose` description.
- **Token streaming for conversation-mode agents**
  (`ai_provider.py::_anthropic_agent_loop_with_conversation`).
  Switches the Anthropic call to `messages.stream` and emits
  `on_text_delta` / `on_text_end` callbacks; `compute_runner`
  publishes them to the `content.<source>.<field>.streaming`
  channel; the chat hydrator subscribes and renders into a
  pending bubble until `on_text_end` fires, then replaces with
  the persisted entry from the L5 channel.

### Changed

- **`assistant` → `agent` kind rename + UI rebrand to "AI
  Agent"** (close-out task). Canonical kind for AI-agent
  entries is now `agent`. Schema validator and materializer
  accept both `agent` and `assistant` for back-compat; new
  writes are `agent`. UI labels updated: bubble attribution
  reads "AI Agent" (was "Assistant"), refusal banner reads
  "AI Agent refused" (was "Refused"). JL: "we should not call
  the AI 'Assistant' in our system."
- **`system_refuse(reason)` is hard-terminating per §6.1**
  (`compute_runner.py`, `ai_provider.py`). The agent loop
  short-circuits on the first `system_refuse` tool call: the
  refusal entry lands in the conversation, the response stream
  hard-stops, no further tool calls or text deltas are emitted
  for the same invocation. Three-layer enforcement (provider
  `should_halt`, runtime `_on_writeback` short-circuit,
  `_execute_tool` defensive gate) — belt-and-suspenders for
  the trust boundary. Tenet 2 (enforcement over vigilance):
  authors should not have to remember to short-circuit on
  refuse; the platform enforces it.

### Fixed

- **Chat surface — bootstrap-window form submit race** (`static/
  termin.js`, `presentation.py`). Generic AJAX form interceptor
  used to fire on the chat input form before the chat-specific
  hydrator bound its `preventDefault` listener, racing with the
  send-via-WS path. Two layered defenses: chat form opts out via
  `data-termin-no-default-submit`; `onsubmit="return false"`
  blocks any natural default submit if the opt-out is missed.
- **Chat surface — tool_result orphan pairing**
  (`hydrateConversationFieldChat::_attachToolResultToCall`).
  v0.9.2 conversation timeline pairs `tool_call` with its
  `tool_result` by `tool_call_id`; the hydrator now attaches
  the result inline under the call rather than rendering a
  standalone "Tool result (unknown, orphan)" bubble.
- **Chat surface — history load on page refresh.**
  Conversation field returned by SQLite GET arrives as a JSON
  string; `_coerceEntries` parses it before render so the
  history shows up on first paint instead of staying empty
  until the next append fires.
- **Chat surface — thread switching UI.** Clickable thread rows
  in the data_table picker (event delegation on the table
  element so WS-pushed new rows pick up the click handler
  without re-hydration). Header carries a "+ New chat" button
  that creates a fresh parent record and switches the chat to
  it. NB: the table → chat coupling is implicit (table is a
  picker if its `source` matches the chat's `source` and they
  co-occur on the page); see the v0.10 backlog item "Explicit
  picker binding for chat-driving tables" for the planned
  DSL-level fix.
- **Chat surface — WS push envelope.** v0.9.2 broadcast
  unwrapped to `event.get("record")`, dropping the
  `appended_entry` envelope the hydrator needs. Wraps the
  publish in a `data:` key so the conn_manager unwrap picks the
  full envelope.
- **`system_refuse` orphan tool_call breaking next turn.** The
  refusal tool_call entry was being persisted to the
  conversation, then `_on_writeback` short-circuited so no
  matching `tool_result` ever appeared. The next agent turn
  failed Anthropic 400 ("`tool_use` ids found without
  `tool_result` blocks immediately after"). Three fixes:
  (1) skip `_on_writeback` entirely for `system_refuse` calls;
  (2) drop orphan `tool_call` entries in
  `materialize_to_anthropic` for legacy data;
  (3) longer drain in `_close_bg_loop_cleanly` so aiosqlite
  worker callbacks land before the bg loop closes.
- **aiosqlite race with bg-loop close**
  (`app.py::_close_bg_loop_cleanly`). Drains pending tasks +
  calls `shutdown_asyncgens` + `asyncio.sleep(0.1)` before
  closing the bg event loop so aiosqlite worker callbacks
  don't land on a closed loop. Replaces the bare
  `bg_loop.close()` that was raising "Event loop is closed"
  tracebacks during refusals.

### Suite

98 tests passing on Windows (was 24; +74 from L1–L11 across
storage, append handler, WS frame, materialization, write-back,
streaming, refusal termination, kind rename back-compat, chat
hydrator e2e, and the close-out hotfixes). End-to-end manual
verification by JL on `agent_chatbot`: streaming visible,
`system_refuse` terminates cleanly, recovery turn after refuse
works, history loads on refresh, thread switching works, "AI
Agent" label correct.

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
