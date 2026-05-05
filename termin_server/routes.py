# Copyright 2026 Jamie-Leigh Blake and Termin project contributors
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0

"""CRUD route registration, reflection endpoints, channel endpoints, webhooks.

Auto-CRUD from IR RouteSpec (D-11). Reflection API. Channel action/send
endpoints. Inbound webhook handlers. SSE streams.
"""

import json
import sqlite3

from fastapi import Depends, HTTPException, Query, Request, Response
from fastapi.responses import StreamingResponse
from pathlib import Path

from .context import RuntimeContext
from .storage import (
    get_db, create_record, get_record, update_record, delete_record,
    list_records, find_by_field,
)
from .providers import (
    Eq, And, OrderBy, QueryOptions, CascadeMode,
)
from .state import do_state_transition
from .confidentiality import redact_record, redact_records, check_write_access
from .boundaries import check_boundary_identity
from .validation import (
    validate_dependent_values, validate_enum_constraints,
    validate_min_max_constraints, evaluate_field_defaults, strip_unknown_fields,
)
from .compute_runner import redact_audit_traces
from .preferences import (
    InvalidThemeValueError,
    VALID_THEMES,
    ensure_preferences_table,
    get_theme_preference,
    set_theme_preference,
)
from .presentation_bundles import (
    register_presentation_bundle_endpoint,
    register_provider_bundle_route,
)
from .bootstrap import (
    register_page_data_endpoint,
    register_shell_endpoint,
)


# v0.9 Phase 2: cross-cutting helpers that wrap ctx.storage with the
# event-publishing + error-routing concerns the legacy storage.py
# entrypoints used to bundle in. The provider stays pure (BRD §6.2
# "Provider's job is small"); the runtime owns the workflow.

async def _publish_content_event(ctx, kind: str, content_name: str, record: dict):
    """Publish a {created|updated|deleted} event for a content row."""
    if ctx.event_bus is None:
        return
    payload = {
        "type": f"{content_name}_{kind}",
        "channel_id": f"content.{content_name}.{kind}",
        "content_name": content_name,
    }
    if kind == "deleted":
        payload["record_id"] = record.get("id")
    else:
        payload["data"] = record
    await ctx.event_bus.publish(payload)


def _route_terminator(ctx, content_name: str, exc: Exception) -> None:
    """Route a storage exception through TerminAtor as a validation
    error. No-op if no terminator is configured. Exception is
    re-raised by the caller — TerminAtor records, doesn't intercept."""
    if ctx.terminator is None:
        return
    from .errors import TerminError
    ctx.terminator.route(TerminError(
        source=content_name, kind="validation", message=str(exc),
    ))


def _seed_state_columns(body: dict, sm_info, *, strip_existing: bool = False) -> dict:
    """Apply state-machine initial values to a creation body.

    If strip_existing is True (CREATE-route gate), state columns are
    removed entirely so the SQL DEFAULT applies — preserves the
    transition-rules-only-via-transitions invariant.
    Otherwise (e.g. internal seeding), a missing/empty state column
    is filled with the machine's initial state.
    """
    if not sm_info:
        return body
    out = dict(body)
    if isinstance(sm_info, list):
        state_cols = {sm["machine_name"] for sm in sm_info if sm.get("machine_name")}
        if strip_existing:
            return {k: v for k, v in out.items() if k not in state_cols}
        for sm in sm_info:
            col = sm.get("machine_name", "")
            if col and not out.get(col):
                out[col] = sm.get("initial", "")
    return out


def register_crud_routes(app, ctx: RuntimeContext):
    """Register all CRUD routes from the IR route specs."""

    # Build a per-content ownership-field lookup so CREATE routes can
    # stamp the ownership field with `the user.id` before insert
    # (Phase 6a.5 / BRD #3 §3.5). The IR's ContentSchema.ownership.field
    # is the snake-case column name; None when no ownership declared.
    ownership_field_for_content: dict[str, str | None] = {}
    for cs in ctx.ir.get("content", []):
        own = cs.get("ownership")
        if own and own.get("field"):
            ownership_field_for_content[cs.get("name", {}).get("snake", "")] = own["field"]

    for route in ctx.ir.get("routes", []):
        content_ref = route.get("content_ref", "")
        method = route.get("method", "GET")
        path = route.get("path", "")
        kind = route.get("kind", "LIST")
        scope = route.get("scope") or route.get("required_scope")
        lookup_col = route.get("lookup_column", "id")
        target_state = route.get("target_state")
        machine_name = route.get("machine_name")
        # v0.9 Phase 6a.5: row_filter from RouteSpec drives ownership-
        # restricted routes. Shape: {"kind": "ownership", "field":
        # "<snake>"} or None.
        row_filter = route.get("row_filter")
        owner_field_for_create = ownership_field_for_content.get(content_ref)

        if kind == "LIST":
            _make_list_route(app, ctx, path, content_ref, scope, row_filter)
        elif kind == "CREATE":
            _make_create_route(app, ctx, path, content_ref, scope,
                               ctx.sm_lookup.get(content_ref, []),
                               owner_field_for_create)
        elif kind == "GET_ONE":
            _make_get_route(app, ctx, path, content_ref, scope, lookup_col, row_filter)
        elif kind == "UPDATE":
            _make_update_route(app, ctx, path, content_ref, scope, lookup_col, row_filter)
        elif kind == "DELETE":
            _make_delete_route(app, ctx, path, content_ref, scope, lookup_col, row_filter)
        elif kind == "TRANSITION":
            _make_transition_route(app, ctx, path, content_ref, scope,
                                   lookup_col, target_state, machine_name)
        elif kind == "APPEND":
            # v0.9.2 L3: field-targeted append on conversation fields.
            field_name = route.get("field_name")
            _make_append_route(app, ctx, path, content_ref, scope,
                               field_name, row_filter)


def _make_list_route(app, ctx, path, cr, sc, row_filter=None):
    """Slice 7.2.e of Phase 7 (2026-04-30): list-content handler
    extracted to ``termin_core.routing.crud.list_content_handler``.
    The FastAPI route here is now a thin bridge that wraps the
    request, delegates to the pure handler, and unwraps the
    response.

    Boundary identity check, redaction, and audit-trace redaction
    are runtime-internal concerns; the handler reads them off ctx
    via thin shims. Slice 7.5 may move boundary checks into core
    too, at which point the ctx hooks here become unnecessary.
    """
    from termin_core.routing import list_content_handler
    from .fastapi_adapter import (
        make_auth_context,
        to_fastapi_response,
        to_termin_request,
    )
    from .compute_runner import redact_audit_traces as _redact_audit_traces

    deps = [Depends(ctx.require_scope(sc))] if sc else []

    # Stash row_filter and ctx-side runtime concerns on ctx itself
    # so the pure handler can read them. Closure-style binding —
    # the handler doesn't see the route registration's scope.
    if not hasattr(ctx, "_row_filter_for_content"):
        ctx._row_filter_for_content = {}
    if row_filter:
        ctx._row_filter_for_content[cr] = row_filter

    if not hasattr(ctx, "row_filter_for"):
        ctx.row_filter_for = lambda cn: ctx._row_filter_for_content.get(cn)
    if not hasattr(ctx, "_check_boundary_identity"):
        ctx._check_boundary_identity = lambda cn, scopes: check_boundary_identity(
            ctx.boundary_identity_scopes, ctx.boundary_for_content,
            cn, scopes,
        )
    if not hasattr(ctx, "redact_audit_traces"):
        async def _redact(records, content_ref, scopes):
            return await _redact_audit_traces(ctx, records, content_ref, scopes)
        ctx.redact_audit_traces = _redact

    @app.get(path, dependencies=deps)
    async def list_route(request: Request, _cr=cr):
        user = ctx.get_current_user(request)
        auth = make_auth_context(user)
        termin_req = await to_termin_request(
            request,
            path_params={"content": _cr},
            auth=auth,
        )
        response = await list_content_handler(termin_req, ctx)
        return to_fastapi_response(response)


def _make_create_route(app, ctx, path, cr, sc, sm_info, owner_field=None):
    """Slice 7.2.e of Phase 7 (2026-04-30): create handler extracted
    to ``termin_core.routing.crud.create_content_handler``. The
    FastAPI route is now a thin bridge that delegates to the pure
    handler. State-machine seeding, event publishing, and IR event
    handlers are stashed on ctx as runtime-internal hooks the pure
    handler reads.
    """
    from termin_core.routing import create_content_handler
    from .fastapi_adapter import (
        make_auth_context,
        to_fastapi_response,
        to_termin_request,
    )

    deps = [Depends(ctx.require_scope(sc))] if sc else []

    # Per-content-type registration: state-machine info, owner field.
    if not hasattr(ctx, "_state_machine_info_for_content"):
        ctx._state_machine_info_for_content = {}
    ctx._state_machine_info_for_content[cr] = sm_info
    if not hasattr(ctx, "state_machine_info_for"):
        ctx.state_machine_info_for = lambda cn: ctx._state_machine_info_for_content.get(cn)

    if not hasattr(ctx, "_owner_field_for_content"):
        ctx._owner_field_for_content = {}
    if owner_field:
        ctx._owner_field_for_content[cr] = owner_field
    if not hasattr(ctx, "owner_field_for"):
        ctx.owner_field_for = lambda cn: ctx._owner_field_for_content.get(cn)

    # Pure-rule helpers that haven't moved to termin-core yet —
    # exposed via ctx so the handler can call them. Slice 7.5 may
    # move state-column seeding into core proper.
    if not hasattr(ctx, "seed_state_columns"):
        ctx.seed_state_columns = _seed_state_columns

    if not hasattr(ctx, "publish_content_event"):
        async def _publish(kind, content_name, record):
            await _publish_content_event(ctx, kind, content_name, record)
        ctx.publish_content_event = _publish

    if not hasattr(ctx, "route_terminator_validation"):
        ctx.route_terminator_validation = lambda cn, exc: _route_terminator(ctx, cn, exc)

    if not hasattr(ctx, "run_event_handlers_for_content"):
        async def _run_evt(content_name, kind, record):
            db = await get_db(ctx.db_path)
            try:
                await ctx.run_event_handlers(db, content_name, kind, record)
            finally:
                await db.close()
        ctx.run_event_handlers_for_content = _run_evt

    @app.post(path, status_code=201, dependencies=deps)
    async def create_route(request: Request, _cr=cr):
        user = ctx.get_current_user(request)
        auth = make_auth_context(user)
        termin_req = await to_termin_request(
            request,
            path_params={"content": _cr},
            auth=auth,
        )
        response = await create_content_handler(termin_req, ctx)
        return to_fastapi_response(response)


def _make_get_route(app, ctx, path, cr, sc, lc, row_filter=None):
    """Slice 7.2.e of Phase 7 (2026-04-30): get-by-id handler
    extracted to ``termin_core.routing.crud.get_content_handler``.
    """
    from termin_core.routing import get_content_handler
    from .fastapi_adapter import (
        make_auth_context,
        to_fastapi_response,
        to_termin_request,
    )

    deps = [Depends(ctx.require_scope(sc))] if sc else []

    # Stash lookup-column + row_filter on ctx so the pure handler
    # can read them without per-route closure capture.
    if not hasattr(ctx, "_lookup_column_for_content"):
        ctx._lookup_column_for_content = {}
    ctx._lookup_column_for_content[cr] = lc
    if not hasattr(ctx, "lookup_column_for"):
        ctx.lookup_column_for = lambda cn: ctx._lookup_column_for_content.get(cn, "id")
    if not hasattr(ctx, "_row_filter_for_content"):
        ctx._row_filter_for_content = {}
    if row_filter:
        ctx._row_filter_for_content[cr] = row_filter
    if not hasattr(ctx, "row_filter_for"):
        ctx.row_filter_for = lambda cn: ctx._row_filter_for_content.get(cn)

    @app.get(path, dependencies=deps)
    async def get_route(request: Request, _cr=cr, _lc=lc):
        # FastAPI extracts the lookup-key path param under whatever
        # name the route declared — typically {id} or {sku}. The
        # core handler reads it under "key" so the bridge name-maps.
        key_val = list(request.path_params.values())[0] if request.path_params else None
        user = ctx.get_current_user(request)
        auth = make_auth_context(user)
        termin_req = await to_termin_request(
            request,
            path_params={"content": _cr, "key": key_val},
            auth=auth,
        )
        response = await get_content_handler(termin_req, ctx)
        return to_fastapi_response(response)


def _make_update_route(app, ctx, path, cr, sc, lc, row_filter=None):
    """Slice 7.2.e of Phase 7 (2026-04-30): update handler extracted
    to ``termin_core.routing.crud.update_content_handler``. The
    FastAPI route is a thin bridge.
    """
    from termin_core.routing import update_content_handler
    from .fastapi_adapter import (
        make_auth_context,
        to_fastapi_response,
        to_termin_request,
    )

    deps = [Depends(ctx.require_scope(sc))] if sc else []

    # Per-content-type: lookup column + row_filter already stashed
    # by _make_get_route on the same content_ref. Be defensive in
    # case create runs in isolation.
    if not hasattr(ctx, "_lookup_column_for_content"):
        ctx._lookup_column_for_content = {}
    ctx._lookup_column_for_content[cr] = lc
    if not hasattr(ctx, "lookup_column_for"):
        ctx.lookup_column_for = lambda cn: ctx._lookup_column_for_content.get(cn, "id")
    if not hasattr(ctx, "_row_filter_for_content"):
        ctx._row_filter_for_content = {}
    if row_filter:
        ctx._row_filter_for_content[cr] = row_filter
    if not hasattr(ctx, "row_filter_for"):
        ctx.row_filter_for = lambda cn: ctx._row_filter_for_content.get(cn)

    @app.put(path, dependencies=deps)
    async def update_route(request: Request, _cr=cr):
        key_val = list(request.path_params.values())[0] if request.path_params else None
        user = ctx.get_current_user(request)
        auth = make_auth_context(user)
        termin_req = await to_termin_request(
            request,
            path_params={"content": _cr, "key": key_val},
            auth=auth,
        )
        response = await update_content_handler(termin_req, ctx)
        return to_fastapi_response(response)


def _make_delete_route(app, ctx, path, cr, sc, lc, row_filter=None):
    """Slice 7.2.e of Phase 7 (2026-04-30): delete handler extracted
    to ``termin_core.routing.crud.delete_content_handler``. The
    FastAPI route is a thin bridge.
    """
    from termin_core.routing import delete_content_handler
    from .fastapi_adapter import (
        make_auth_context,
        to_fastapi_response,
        to_termin_request,
    )

    deps = [Depends(ctx.require_scope(sc))] if sc else []

    if not hasattr(ctx, "_lookup_column_for_content"):
        ctx._lookup_column_for_content = {}
    ctx._lookup_column_for_content[cr] = lc
    if not hasattr(ctx, "lookup_column_for"):
        ctx.lookup_column_for = lambda cn: ctx._lookup_column_for_content.get(cn, "id")
    if not hasattr(ctx, "_row_filter_for_content"):
        ctx._row_filter_for_content = {}
    if row_filter:
        ctx._row_filter_for_content[cr] = row_filter
    if not hasattr(ctx, "row_filter_for"):
        ctx.row_filter_for = lambda cn: ctx._row_filter_for_content.get(cn)

    @app.delete(path, dependencies=deps)
    async def delete_route(request: Request, _cr=cr):
        key_val = list(request.path_params.values())[0] if request.path_params else None
        user = ctx.get_current_user(request)
        auth = make_auth_context(user)
        termin_req = await to_termin_request(
            request,
            path_params={"content": _cr, "key": key_val},
            auth=auth,
        )
        response = await delete_content_handler(termin_req, ctx)
        return to_fastapi_response(response)


def _make_transition_route(app, ctx, path, cr, sc, lc, ts, mn=None):
    """Slice 7.2.e of Phase 7 (2026-04-30): per-machine, per-target
    state-transition route. Body extracted to
    ``termin_core.routing.crud.transition_content_handler``.

    ``mn`` is the machine_name (snake_case) the route drives.
    Required in v0.9 — every transition route addresses one machine
    on one content. Callers from ``register_crud_routes`` always
    pass it; older internal callers (none currently) would not, in
    which case the core handler falls back to the first state
    machine on the content for backward compatibility.
    """
    from termin_core.routing import transition_content_handler
    from .fastapi_adapter import (
        make_auth_context,
        to_fastapi_response,
        to_termin_request,
    )

    deps = [Depends(ctx.require_scope(sc))] if sc else []

    if not hasattr(ctx, "_lookup_column_for_content"):
        ctx._lookup_column_for_content = {}
    ctx._lookup_column_for_content[cr] = lc
    if not hasattr(ctx, "lookup_column_for"):
        ctx.lookup_column_for = lambda cn: ctx._lookup_column_for_content.get(cn, "id")

    @app.post(path, dependencies=deps)
    async def transition_route(request: Request, _cr=cr, _ts=ts, _mn=mn):
        key_val = list(request.path_params.values())[0] if request.path_params else None
        user = ctx.get_current_user(request)
        auth = make_auth_context(user)
        termin_req = await to_termin_request(
            request,
            path_params={
                "content": _cr,
                "key": key_val,
                "machine": _mn,
                "target": _ts,
            },
            auth=auth,
        )
        response = await transition_content_handler(termin_req, ctx)
        return to_fastapi_response(response)


# v0.9.2 L3: UUID v7 generator. UUIDv7 is time-ordered (millisecond
# Unix timestamp prefix) so entry ids sort by creation order — useful
# for audit citations and chronological reads. We roll our own rather
# than add a dep; the format follows RFC 9562 §5.7.
def _uuid7_str() -> str:
    import os
    import time
    import uuid
    ts_ms = int(time.time() * 1000) & ((1 << 48) - 1)
    rand_a = int.from_bytes(os.urandom(2), "big") & 0x0FFF  # 12 random bits
    rand_b = int.from_bytes(os.urandom(8), "big") & ((1 << 62) - 1)
    high = (ts_ms << 16) | (0x7 << 12) | rand_a   # version=7 in high nibble of time-mid
    low = (0b10 << 62) | rand_b                    # variant=10 in top two bits
    return str(uuid.UUID(int=(high << 64) | low))


# v0.9.2 L3: canonical conversation entry kinds (per tech-design §7.2).
# Validated at append time so storage never holds entries the runtime
# doesn't recognize.
_CANONICAL_KINDS = frozenset({
    "user", "assistant", "tool_call", "tool_result", "system_event",
})


# v0.9.2 L3/L4: structured exception types so the WS frame handler
# can map validation/permission failures to error frames without
# reaching for FastAPI's HTTPException class. The REST wrapper
# translates these into HTTP status codes; the WS dispatcher
# translates them into structured error frames. Same helper, two
# transports — neither one depends on the other's framing.
class AppendValidationError(Exception):
    """Body shape problem (invalid kind, missing body, malformed JSON)."""

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


class AppendNotFoundError(Exception):
    """Parent record doesn't exist or row-filter excludes it."""

    def __init__(self, message: str = "Not found") -> None:
        super().__init__(message)
        self.message = message


async def _do_append(
    ctx,
    *,
    content_ref: str,
    key_val,
    field_name: str,
    payload: dict,
    user: dict | None,
    row_filter: dict | None = None,
) -> dict:
    """Shared append logic — used by both the REST endpoint and the
    WebSocket frame handler.

    Validates the payload, loads the parent record (404 if absent or
    if a row_filter excludes it), reads the existing JSON column,
    builds the new entry with canonical metadata, and writes the
    updated entry list back via ``update_record``.

    Returns the new entry on success. Raises:

    * :class:`AppendValidationError` — payload shape problem (caller
      maps to HTTP 400 / WS error frame with code "validation_error").
    * :class:`AppendNotFoundError` — record absent or row-filter
      rejected (caller maps to HTTP 404 / WS "not_found").

    Event firing on append (``<content>.<field>.appended``) is L5's
    job; this helper passes ``event_bus=None`` to ``update_record``
    on the column write so the L5 hook can wire up cleanly.

    The shape is `kwargs-only` after ``ctx`` to keep the helper's
    call sites readable when they pass mostly-static metadata.
    """
    from datetime import datetime, timezone

    if not key_val:
        raise AppendValidationError("Missing record id")
    if not isinstance(payload, dict):
        raise AppendValidationError("Body must be a JSON object")

    kind = payload.get("kind", "")
    if kind not in _CANONICAL_KINDS:
        raise AppendValidationError(
            f"Invalid kind '{kind}'. Must be one of: {sorted(_CANONICAL_KINDS)}"
        )
    body_text = payload.get("body")
    if body_text is None or body_text == "":
        raise AppendValidationError("body is required")

    db = await get_db(ctx.db_path)
    db.row_factory = sqlite3.Row
    try:
        record = await get_record(db, content_ref, key_val)
    except HTTPException as e:
        # Storage layer raises HTTPException(404) on missing — translate
        # to our transport-neutral exception so the caller maps it
        # back to its native shape (HTTP 404 or WS error frame).
        if e.status_code == 404:
            raise AppendNotFoundError("Not found") from None
        raise

    # Row filter: their_own ownership check on the parent record.
    # Mirrors what the runtime does for view/update/delete on owned
    # content (BRD #3 §3.7). Same 404 surface as a missing record so
    # ownership doesn't leak existence.
    if row_filter and row_filter.get("kind") == "ownership":
        user_id = (user or {}).get("id") if user else None
        # The user dict is the legacy shape — `the_user.id` carries
        # the principal id under the v0.9 layout. Fall back to the
        # top-level `id` for unit-test friendliness.
        if not user_id and isinstance(user, dict):
            the_user = user.get("the_user") or {}
            user_id = the_user.get("id")
        owner_field = row_filter.get("field")
        if owner_field and record.get(owner_field) != user_id:
            raise AppendNotFoundError("Not found")

    # Read existing entries (TEXT column holding a JSON array).
    raw = record.get(field_name)
    if raw in (None, ""):
        entries = []
    else:
        try:
            entries = json.loads(raw)
            if not isinstance(entries, list):
                entries = []
        except (TypeError, ValueError):
            entries = []

    # Build the new entry with canonical metadata. Optional
    # caller-supplied fields pass through unchanged; runtime owns
    # id, created_at, appended_by_principal_id.
    user_dict = user or {}
    appender_id = user_dict.get("id", "")
    if not appender_id and isinstance(user_dict, dict):
        the_user = user_dict.get("the_user") or {}
        appender_id = the_user.get("id", "")
    entry = {
        "id": _uuid7_str(),
        "kind": kind,
        "body": body_text,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "appended_by_principal_id": appender_id,
    }
    # v0.9.2 L7.4 (per JL Wave 3 §7.2 update): `type` is an optional
    # per-kind sub-discriminator (free-form text). v0.9.2 documents
    # `assistant.type == "refusal"` for system.refuse-driven entries;
    # other kinds reserve the field for later. No validation — the
    # field passes through to storage as whatever the caller supplied.
    for k in ("type", "source", "tool_call_id", "parent_id", "tool_name",
              "tool_args", "attachments", "purpose"):
        if k in payload:
            entry[k] = payload[k]

    entries.append(entry)
    updated_record = await update_record(
        db, content_ref, key_val,
        {field_name: json.dumps(entries)},
        terminator=ctx.terminator,
        event_bus=None,   # the standard _updated event is suppressed —
                          # L5 publishes the field-specific .appended event
                          # below instead, so subscribers get one signal,
                          # not two.
    )

    # v0.9.2 L5: publish `content.<name>.<field>.appended` so listener
    # computes (Trigger on event "X.Y.appended" where ...) and any WS
    # subscribers receive the new entry. Channel is field-specific so
    # subscribers can react to conversation activity on one field
    # without false positives from other column updates. Lives in the
    # shared helper (not the REST route) so the WS frame handler (L4)
    # also fires the event without any duplication.
    if ctx.event_bus is not None:
        # The event envelope. We duplicate it under `data` so the WS
        # forwarder's payload-unwrap (conn_manager.broadcast_to_subscribers
        # — `event.get("data") or event.get("record") or event`) ends up
        # forwarding the full envelope (with `appended_entry`) to JS
        # subscribers, NOT just the `record` (which would lose the
        # appended_entry the chat hydrator needs to render the new bubble).
        # Other fields are kept at the top level for any in-process
        # subscriber that walks the dict directly.
        envelope = {
            "type": f"{content_ref}_{field_name}_appended",
            "channel_id": f"content.{content_ref}.{field_name}.appended",
            "content_name": content_ref,
            "field_name": field_name,
            "record_id": key_val,
            "record": updated_record,
            "appended_entry": entry,
            "triggered_at": entry["created_at"],
            "invoked_by_principal_id": entry["appended_by_principal_id"],
            "trigger_kind": "crud-append",
        }
        envelope["data"] = dict(envelope)
        await ctx.event_bus.publish(envelope)

    # v0.9.2 L5: dispatch listener computes that triggered on this
    # event. Mirrors the per-CRUD-verb dispatch path used by
    # create/update/delete — `run_event_handlers` is the existing
    # entrypoint that walks `ir.events` (When-rules) and
    # `ir.computes` with `Trigger on event "..."`. The trigger
    # string for an append is `<content>.<field>.appended`.
    #
    # v0.9.2 L8: pass `invoked_by_principal_id` so When-rule Append
    # actions can attribute their synthetic entries to the upstream
    # caller (the audit chain stays cohesive — the When-rule's append
    # shows the same principal as the user message that triggered it).
    if hasattr(ctx, "run_event_handlers"):
        await ctx.run_event_handlers(
            db, content_ref, f"{field_name}.appended", updated_record,
            appended_entry=entry,
            invoked_by_principal_id=entry.get("appended_by_principal_id"),
        )

    return entry


def _make_append_route(app, ctx, path, cr, sc, field_name, row_filter=None):
    """v0.9.2 L3: register a POST /<resource>/{id}/<field>:append handler.

    Thin wrapper around :func:`_do_append`. The shared helper carries
    the validation, row-filter, RMW, and entry-construction logic so
    the REST endpoint and the WebSocket frame handler (L4) cannot
    drift apart.

    Side effect: stashes ``(content_ref, field_name) →
    {scope, row_filter}`` on ``ctx._append_targets`` so the WS frame
    handler can resolve the same metadata when an inbound frame
    addresses an arbitrary content+field pair without a separate
    route registration.
    """
    deps = [Depends(ctx.require_scope(sc))] if sc else []

    # Per-(content, field) registration so the WS append-frame handler
    # can look up the same scope/row_filter the REST endpoint applies.
    # Keyed by (content_snake, field_snake); value is a dict of the
    # route-level metadata. Initialized lazily so multiple register-
    # passes (e.g., test harnesses that boot the app twice) don't
    # error.
    if not hasattr(ctx, "_append_targets"):
        ctx._append_targets = {}
    ctx._append_targets[(cr, field_name)] = {
        "scope": sc,
        "row_filter": row_filter,
    }

    @app.post(path, status_code=201, dependencies=deps)
    async def append_route(request: Request, _cr=cr, _fn=field_name, _rf=row_filter):
        # Path-param extraction: the {id} is the only path param on
        # the canonical APPEND path. If a future variant adds more,
        # take the first.
        key_val = list(request.path_params.values())[0] if request.path_params else None

        try:
            payload = await request.json()
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid JSON body")

        user = ctx.get_current_user(request)
        try:
            entry = await _do_append(
                ctx,
                content_ref=_cr,
                key_val=key_val,
                field_name=_fn,
                payload=payload,
                user=user,
                row_filter=_rf,
            )
        except AppendValidationError as e:
            raise HTTPException(status_code=400, detail=e.message)
        except AppendNotFoundError as e:
            raise HTTPException(status_code=404, detail=e.message)

        return entry


def register_reflection_routes(app, ctx: RuntimeContext):
    """Register reflection, error, and event API endpoints."""

    @app.get("/api/reflect")
    async def api_reflect():
        return json.loads(ctx.ir_json)

    @app.get("/api/reflect/content")
    async def api_reflect_content():
        return ctx.reflection.content_schemas()

    @app.get("/api/reflect/compute")
    async def api_reflect_compute():
        return ctx.reflection.compute_functions()

    @app.get("/api/reflect/roles")
    async def api_reflect_roles():
        return ctx.reflection.roles()

    @app.get("/api/reflect/roles/{role_name}")
    async def api_reflect_role(role_name: str):
        role = ctx.reflection.role(role_name)
        if not role:
            raise HTTPException(status_code=404, detail=f"Role '{role_name}' not found")
        return role

    @app.get("/api/reflect/channels")
    async def api_reflect_channels():
        return ctx.channel_dispatcher.get_full_status()

    @app.get("/api/reflect/channels/{channel_name}")
    async def api_reflect_channel(channel_name: str):
        spec = ctx.channel_dispatcher.get_spec(channel_name)
        if not spec:
            raise HTTPException(status_code=404, detail=f"Channel '{channel_name}' not found")
        display = spec["name"]["display"]
        config = ctx.channel_dispatcher.get_config(channel_name)
        return {
            "name": display,
            "direction": spec.get("direction", ""),
            "delivery": spec.get("delivery", ""),
            "carries": spec.get("carries_content", ""),
            "actions": [a["name"]["display"] for a in spec.get("actions", [])],
            "configured": ctx.channel_dispatcher.is_configured(channel_name),
            "state": ctx.channel_dispatcher.get_connection_state(channel_name),
            "protocol": config.protocol if config else "none",
            "metrics": ctx.channel_dispatcher.get_metrics(channel_name),
        }

    @app.get("/api/errors")
    async def api_errors():
        return ctx.terminator.get_error_log()

    @app.get("/api/events")
    async def api_events(level: str = Query(default=None)):
        log = ctx.event_bus.get_event_log()
        if level:
            order = {"TRACE": 0, "DEBUG": 1, "INFO": 2, "WARN": 3, "ERROR": 4}
            min_l = order.get(level.upper(), 0)
            log = [e for e in log if order.get(e.get("log_level", "INFO"), 2) >= min_l]
        return log


def register_channel_routes(app, ctx: RuntimeContext):
    """Register channel action/send endpoints and inbound webhook handlers."""

    from termin_core.routing import (
        channel_send_handler,
        invoke_channel_action_handler,
        webhook_receive_handler,
    )
    from .fastapi_adapter import (
        make_auth_context,
        to_fastapi_response,
        to_termin_request,
    )

    @app.post("/api/v1/channels/{channel_name}/actions/{action_name}")
    async def invoke_channel_action(channel_name: str, action_name: str, request: Request):
        user = ctx.get_current_user(request)
        auth = make_auth_context(user)
        termin_req = await to_termin_request(
            request,
            path_params={"channel_name": channel_name, "action_name": action_name},
            auth=auth,
        )
        response = await invoke_channel_action_handler(termin_req, ctx)
        return to_fastapi_response(response)

    @app.post("/api/v1/channels/{channel_name}/send")
    async def channel_send_endpoint(channel_name: str, request: Request):
        user = ctx.get_current_user(request)
        auth = make_auth_context(user)
        termin_req = await to_termin_request(
            request,
            path_params={"channel_name": channel_name},
            auth=auth,
        )
        response = await channel_send_handler(termin_req, ctx)
        return to_fastapi_response(response)

    # Inbound webhook handlers
    for ch in ctx.ir.get("channels", []):
        ch_direction = ch.get("direction", "")
        if ch_direction not in ("INBOUND", "BIDIRECTIONAL"):
            continue
        ch_display = ch["name"]["display"]
        ch_snake = ch["name"]["snake"]
        ch_carries = ch.get("carries_content", "")
        if not ch_carries:
            continue

        webhook_path = f"/webhooks/{ch_snake}"

        def _make_webhook(ch_name=ch_display, ch_content=ch_carries, ch_spec=ch, ch_snake_local=ch_snake):
            @app.post(webhook_path, name=f"webhook_{ch_snake_local}")
            async def webhook_receive(request: Request):
                user = ctx.get_current_user(request)
                auth = make_auth_context(user)
                termin_req = await to_termin_request(
                    request,
                    path_params={"channel_snake": ch_snake_local},
                    auth=auth,
                )
                response = await webhook_receive_handler(
                    termin_req, ctx, channel_spec=ch_spec,
                )
                # Webhook successes log the same line the runtime did
                # before — visible in dev-loop console for debugging.
                if response.json_body and response.json_body.get("ok"):
                    rec_id = response.json_body.get("id", "?")
                    print(f"[Termin] Webhook '{ch_name}': created {ch_content} record (id={rec_id})")
                return to_fastapi_response(response)

        _make_webhook()
        print(f"[Termin] Registered webhook: POST {webhook_path} -> {ch_carries}")


def register_sse_routes(app, ctx: RuntimeContext):
    """Register SSE stream endpoints."""
    for stream in ctx.ir.get("streams", []):
        def make_sse(p):
            @app.get(p)
            async def sse_stream(request: Request, _p=p):
                async def generate():
                    q = ctx.event_bus.subscribe()
                    try:
                        while True:
                            event = await q.get()
                            yield f"data: {json.dumps(event)}\n\n"
                    except Exception:
                        ctx.event_bus.unsubscribe(q)
                return StreamingResponse(generate(), media_type="text/event-stream")
        make_sse(stream["path"])


def register_runtime_endpoints(app, ctx: RuntimeContext):
    """Register runtime infrastructure endpoints (registry, bootstrap, termin.js)."""

    @app.get("/runtime/registry")
    async def runtime_registry(request: Request):
        host = request.headers.get("host", "localhost:8000")
        scheme = "wss" if request.url.scheme == "https" else "ws"
        http_scheme = request.url.scheme or "http"
        boundaries = {}
        for bnd in ctx.ir.get("boundaries", []):
            name = bnd.get("name", {}).get("snake", "unknown")
            boundaries[name] = {
                "location": "local",
                "channels": {
                    "realtime": f"{scheme}://{host}/runtime/ws",
                    "reliable": f"{http_scheme}://{host}/runtime/api",
                },
            }
        boundaries["presentation"] = {
            "location": "client",
            "channels": {
                "realtime": f"{scheme}://{host}/runtime/ws",
                "reliable": f"{http_scheme}://{host}/runtime/api",
            },
        }
        return {
            "runtime_version": "0.9.1",
            "application": ctx.ir.get("name", "Termin App"),
            "boundaries": boundaries,
            "protocols": {"realtime": "websocket", "reliable": "rest"},
        }

    @app.get("/runtime/bootstrap")
    async def runtime_bootstrap(request: Request):
        user = ctx.get_current_user(request)
        role = user["role"]
        user_pages = [p for p in ctx.ir.get("pages", [])
                      if p["role"] == role or p["role"].lower() == role.lower()]
        client_computes = []
        for comp in ctx.ir.get("computes", []):
            if comp.get("body_lines"):
                client_computes.append({
                    "name": comp["name"],
                    "input_params": comp.get("input_params", []),
                    "body_lines": comp.get("body_lines", []),
                })
        content_names = [cs["name"]["snake"] for cs in ctx.ir.get("content", [])]
        # v0.9 multi-SM: emit one transition map per machine, keyed by
        # content_ref → machine_name → "from|to" → scope. External clients
        # see every machine on every content; legacy single-SM clients
        # that read transitions[content] directly need to update.
        transitions = {}
        for content_ref, sm_list in ctx.sm_lookup.items():
            transitions[content_ref] = {}
            for sm in sm_list:
                transitions[content_ref][sm["machine_name"]] = {
                    f"{from_s}|{to_s}": scope
                    for (from_s, to_s), scope in sm["transitions"].items()
                }
        return {
            "identity": {"role": role, "scopes": user["scopes"], "profile": user["profile"]},
            "pages": user_pages,
            "computes": client_computes,
            "schemas": ctx.ir.get("content", []),
            "content_names": content_names,
            "transitions": transitions,
        }

    @app.get("/runtime/termin.js")
    async def serve_termin_js():
        js_path = Path(__file__).parent / "static" / "termin.js"
        if js_path.exists():
            return Response(content=js_path.read_text(encoding="utf-8"),
                            media_type="application/javascript",
                            headers={"Cache-Control": "no-cache"})
        return Response(content="// termin.js not found",
                        media_type="application/javascript", status_code=404)

    # v0.9 Phase 5a.3: theme preference endpoints. BRD #2 §6.2 +
    # presentation-provider-design.md §3.4. Authenticated principals
    # get a row in `_termin_principal_preferences`; anonymous
    # principals get a session-scoped cookie. Both paths apply
    # `theme_locked` resolution at read time.
    _ANON_THEME_COOKIE = "termin_theme_pref"

    def _resolve_theme_for_request(request: Request) -> str:
        user = ctx.get_current_user(request)
        principal = user.get("Principal")
        theme_default = ctx.theme_default
        theme_locked = ctx.theme_locked
        if principal is not None and not principal.is_anonymous:
            conn = sqlite3.connect(ctx.db_path)
            try:
                return get_theme_preference(
                    conn,
                    principal.id,
                    theme_default=theme_default,
                    theme_locked=theme_locked,
                )
            finally:
                conn.close()
        # Anonymous: cookie-scoped storage, with theme_locked still
        # winning. Cookie cleared on session end (no Max-Age).
        if theme_locked is not None:
            return theme_locked
        cookie_val = request.cookies.get(_ANON_THEME_COOKIE)
        if cookie_val and cookie_val in VALID_THEMES:
            return cookie_val
        return theme_default or "auto"

    @app.get("/_termin/preferences/theme")
    async def get_theme_preference_endpoint(request: Request):
        return {"value": _resolve_theme_for_request(request)}

    @app.post("/_termin/preferences/theme")
    async def set_theme_preference_endpoint(request: Request):
        body = await request.json()
        if not isinstance(body, dict) or "value" not in body:
            raise HTTPException(
                status_code=422,
                detail="Body must be an object with a 'value' key.",
            )
        value = body["value"]
        if value not in VALID_THEMES:
            raise HTTPException(
                status_code=422,
                detail=(
                    f"value must be one of {list(VALID_THEMES)!r}; "
                    f"got {value!r}"
                ),
            )
        user = ctx.get_current_user(request)
        principal = user.get("Principal")
        # Per BRD §6.2: write succeeds even under theme_locked. The
        # lock check applies only at read time so the user's stored
        # preference survives lock removal.
        if principal is not None and not principal.is_anonymous:
            conn = sqlite3.connect(ctx.db_path)
            try:
                set_theme_preference(conn, principal.id, value)
                conn.commit()
            except InvalidThemeValueError as e:
                # Should not happen — value already validated above —
                # but kept defensively in case the validator and the
                # endpoint diverge.
                raise HTTPException(status_code=422, detail=str(e))
            finally:
                conn.close()
            effective = (
                ctx.theme_locked
                if ctx.theme_locked is not None
                else value
            )
            return {"value": effective}
        # Anonymous: cookie-scoped store. Set-Cookie with no Max-Age
        # → session cookie, cleared when the browser closes.
        effective = (
            ctx.theme_locked
            if ctx.theme_locked is not None
            else value
        )
        response = Response(
            content=json.dumps({"value": effective}),
            media_type="application/json",
        )
        response.set_cookie(
            key=_ANON_THEME_COOKIE,
            value=value,
            httponly=True,
            samesite="lax",
        )
        return response

    @app.get("/runtime/termin.css")
    async def serve_termin_css():
        css_path = Path(__file__).parent / "static" / "termin.css"
        if css_path.exists():
            return Response(content=css_path.read_text(encoding="utf-8"),
                            media_type="text/css",
                            headers={"Cache-Control": "no-cache"})
        return Response(content="/* termin.css not found */",
                        media_type="text/css", status_code=404)

    # v0.9 Phase 5b.4 platform: CSR bundle discovery for presentation
    # providers. termin.js fetches this at boot to load registered
    # provider bundles and bind their per-contract render functions.
    register_presentation_bundle_endpoint(app, ctx)

    # v0.9 Phase 5b.4 B' loop: serve provider bundle files from the
    # provider package's `static/bundle.js`. Pairs with the discovery
    # endpoint above — the discovery list points at this URL by
    # default; CDN-overrides bypass it.
    register_provider_bundle_route(app, ctx)

    # v0.9 Phase 5b.4 B' plumbing: page-data endpoint for SPA
    # navigation. Per the Spectrum-provider design Q2 (B' = server-
    # authoritative + JS-as-renderer), the client fetches each page's
    # bootstrap JSON via this endpoint instead of doing a full HTML
    # round-trip. Auth is identical to a regular page request.
    register_page_data_endpoint(app, ctx)

    # No action endpoint — `Termin.action(payload)` in termin.js
    # dispatches client-side to the existing CRUD / transition /
    # compute REST surface that BRD #2 §11 already standardizes.
    # See docs/spectrum-provider-design.md "Q-extra (action API
    # surface)" for the rationale.

    # v0.9 Phase 5b.4 B' plumbing: HTML shell endpoint. Returns
    # the minimal SPA shell with embedded bootstrap JSON for a
    # given path. Used for dev / provider-validation today;
    # flipping the production page routes to this in place of
    # SSR-composited HTML is the follow-on slice.
    register_shell_endpoint(app, ctx)
