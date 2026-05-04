# Copyright 2026 Jamie-Leigh Blake and Termin project contributors
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""FastAPI bridge for the framework-agnostic WebSocket dispatcher.

Slice 7.2.f of Phase 7 (2026-04-30) extracted ``ConnectionManager``,
the ownership-cascade gate, and the per-frame multiplexer loop into
``termin_core.routing.{connection_manager,channel_dispatch}``.

What stays here:

* :class:`ConnectionManager` — re-exported from core for back-compat
  (drops in slice 7.5 once nothing imports from
  ``termin_runtime.websocket_manager`` directly).
* :func:`register_websocket_routes` — the FastAPI route shell. It
  authenticates the incoming connection, accepts the socket, wraps
  the ``fastapi.WebSocket`` as a :class:`TerminWebSocket` via
  :class:`FastAPIWebSocketAdapter`, then hands control to
  :func:`dispatch_websocket_session`. On disconnect the route
  shell cleans up the registry entry the dispatcher's ``connect``
  call created.

v0.9.2 L4 (this file): a thin
:class:`AppendFrameInterceptor` wraps the adapter so inbound frames
of shape ``{"type": "append", ...}`` (per the v0.9.2 conversation
field type tech design §8.3) are handled inline before the core
dispatcher sees them. The dispatcher only ever observes the legacy
``{v, ch, op, ref}``-shaped frames it already knows. Append frames
fire the standard ``<content>.<field>.appended`` event (L5's job)
which propagates back to the originator via its existing record
subscription — there is no separate "append response" frame.
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import WebSocket, WebSocketDisconnect

from termin_core.routing import (  # noqa: F401  (back-compat re-exports)
    ConnectionManager,
    dispatch_websocket_session,
    filter_owned_rows as _filter_owned_rows,
)

from .context import RuntimeContext
from .fastapi_adapter import FastAPIWebSocketAdapter

logger = logging.getLogger("termin.ws")


class AppendFrameInterceptor:
    """Wrap a :class:`TerminWebSocket` adapter so append frames are
    handled inline before the core dispatcher receives them.

    The wrapper proxies every method through to the underlying
    adapter except :meth:`receive_json`, which loops until it sees
    a frame the core dispatcher should handle. Append frames are
    dispatched via :func:`_handle_append_frame` and, on failure,
    surfaced as a structured ``op == "error"`` frame back to the
    originator. On success the originator picks up the new entry
    via the standard ``<content>.<field>.appended`` event (L5),
    so this handler never echoes the entry directly.

    Per §8.3 (v0.9.2 conversation field type tech design): there is
    no separate "append response" frame. The originator sees its own
    append as an event delivery, same as any other client's append.
    """

    def __init__(self, inner: Any, ctx: RuntimeContext, user: Any) -> None:
        self._inner = inner
        self._ctx = ctx
        self._user = user

    # ── Pure proxy methods (six of seven from the Protocol) ──

    @property
    def principal(self):
        return self._inner.principal

    @principal.setter
    def principal(self, value):
        self._inner.principal = value

    async def accept(self) -> None:
        await self._inner.accept()

    async def send_json(self, data: Any) -> None:
        await self._inner.send_json(data)

    async def send_bytes(self, data: bytes) -> None:
        await self._inner.send_bytes(data)

    async def receive_text(self) -> str:
        return await self._inner.receive_text()

    async def close(self, code: int = 1000) -> None:
        await self._inner.close(code=code)

    # ── Intercepting receive_json ──

    async def receive_json(self) -> Any:
        """Read frames until one isn't an append frame; dispatch
        appends inline.

        The core dispatcher is none the wiser — from its POV every
        ``receive_json`` returns one of its three known op-shaped
        frames, just possibly after some milliseconds of latency
        while the interceptor handled an append.
        """
        while True:
            frame = await self._inner.receive_json()
            if not isinstance(frame, dict) or frame.get("type") != "append":
                return frame
            await _handle_append_frame(self._inner, self._ctx, self._user, frame)


async def _send_error(
    ws: Any, *, code: str, message: str, ref: Any = None,
) -> None:
    """Send a structured error frame on the per-connection channel.

    Mirrors the existing ``op == "error"`` shape the core dispatcher
    uses for failed subscribes / requests so client-side error
    handling has one envelope to parse. ``ch`` is fixed at the
    "runtime.append" topic — distinct from any subscription topic
    the client may have, so error frames don't get conflated with
    push deliveries on a shared channel.
    """
    try:
        await ws.send_json({
            "v": 1,
            "ch": "runtime.append",
            "op": "error",
            "ref": ref,
            "payload": {"code": code, "message": message},
        })
    except Exception:
        # Best-effort surface — a closed socket while we're trying to
        # tell the client about an error isn't worth raising over.
        logger.debug("WS append: failed to send error frame", exc_info=True)


async def _handle_append_frame(
    ws: Any, ctx: RuntimeContext, user: Any, frame: dict,
) -> None:
    """v0.9.2 L4: dispatch a single inbound append frame.

    Frame shape (§8.3):

        {"type": "append",
         "resource": "<content snake>",
         "id": "<record id>",
         "field": "<conversation field name>",
         "payload": { ...same as REST body... }}

    Resolves the registered route metadata (scope, row_filter) from
    ``ctx._append_targets`` (stashed by ``_make_append_route``),
    enforces the same scope gate the REST endpoint enforces, calls
    the shared :func:`termin_server.routes._do_append` helper, and
    on failure surfaces a structured error frame. On success the
    L5 event-publish path delivers the entry to subscribers
    (including the originator via its existing record subscription).
    """
    # Local import to avoid circular reference at module import time
    # (routes.py imports from this module's siblings).
    from .routes import (
        AppendNotFoundError,
        AppendValidationError,
        _do_append,
    )

    ref = frame.get("ref")
    resource = frame.get("resource")
    record_id = frame.get("id")
    field_name = frame.get("field")
    payload = frame.get("payload")

    if not resource or not field_name:
        await _send_error(
            ws, code="invalid_frame",
            message="append frame requires 'resource' and 'field'",
            ref=ref,
        )
        return
    if not isinstance(payload, dict):
        await _send_error(
            ws, code="invalid_frame",
            message="append frame 'payload' must be a JSON object",
            ref=ref,
        )
        return

    targets = getattr(ctx, "_append_targets", {}) or {}
    target = targets.get((resource, field_name))
    if target is None:
        # No append route registered for this content+field pair —
        # the source either doesn't grant `can append to X.Y` or the
        # client typo'd. Same 404-ish surface as the REST side
        # would give for an unmapped path.
        await _send_error(
            ws, code="not_found",
            message=(
                f"No append target registered for "
                f"'{resource}.{field_name}'"
            ),
            ref=ref,
        )
        return

    required_scope = target.get("scope")
    row_filter = target.get("row_filter")

    # Permission check: same scope gate the REST endpoint enforces.
    # The user dict carries the resolved scope set on the WS
    # connection via the standard cookie-auth path.
    if required_scope:
        scopes = (user or {}).get("scopes", []) if isinstance(user, dict) else []
        if required_scope not in scopes:
            await _send_error(
                ws, code="forbidden",
                message=f"Requires scope: {required_scope}",
                ref=ref,
            )
            return

    try:
        await _do_append(
            ctx,
            content_ref=resource,
            key_val=record_id,
            field_name=field_name,
            payload=payload,
            user=user,
            row_filter=row_filter,
        )
    except AppendValidationError as e:
        await _send_error(
            ws, code="validation_error", message=e.message, ref=ref,
        )
    except AppendNotFoundError as e:
        await _send_error(
            ws, code="not_found", message=e.message, ref=ref,
        )
    except Exception as e:
        # Storage / runtime errors — log + surface a generic frame so
        # the client knows the append didn't land. Don't leak
        # internal exception text in production deployments; for
        # now (parity with the REST side, which raises 500) we
        # include the message.
        logger.error("WS append failed", exc_info=True)
        await _send_error(
            ws, code="internal_error", message=str(e), ref=ref,
        )

    # On success: no echo frame. The L5 event-publish path
    # delivers the new entry to every subscriber of
    # `content.<resource>.appended` (or the corresponding event
    # channel L5 settles on), including the originator via its
    # existing subscription. See §8.3 of the v0.9.2 conversation
    # field type tech design.


def register_websocket_routes(app, ctx: RuntimeContext):
    """Register the WebSocket multiplexer endpoint on ``app``.

    The shell is intentionally tiny — accept, wrap, dispatch, clean
    up. All decisions about frame shape, ownership cascade, and
    initial-data load live in ``termin-core``. v0.9.2 L4 inserts an
    :class:`AppendFrameInterceptor` between the FastAPI adapter and
    the core dispatcher to handle inbound append frames per §8.3.
    """

    @app.websocket("/runtime/ws")
    async def runtime_ws(websocket: WebSocket):
        user = ctx.get_user_from_ws(websocket)
        await websocket.accept()
        adapter = FastAPIWebSocketAdapter(websocket)
        # The user dict carries scopes/role today; principal will move
        # off the dict in slice 7.5 when AuthContext flows through the
        # WS path the same way it does for HTTP.
        adapter.principal = None
        # v0.9.2 L4: wrap the adapter so append frames are handled
        # before the core dispatcher sees them. The dispatcher only
        # observes legacy op-shaped frames; append frames are
        # transparently consumed by the interceptor's receive_json.
        intercepted = AppendFrameInterceptor(adapter, ctx, user)
        # ``connect`` registers the conn before the dispatcher's loop
        # starts, so any in-flight broadcast can find it. We replicate
        # that order here and let the dispatcher reuse the same conn
        # registration via its ``connect`` call — the registry is
        # idempotent per-conn-id.
        try:
            await dispatch_websocket_session(intercepted, ctx, user)
        except WebSocketDisconnect:
            pass
        except Exception:
            # Same conservative cleanup the v0.9 implementation had:
            # any error tears down the conn so a client retry gets
            # fresh state.
            pass
        finally:
            # The dispatcher's connect minted a conn_id we don't have
            # a handle to from out here. Walk the registry and drop
            # any entry whose ws is this adapter (or its wrapper) —
            # single connection per request, so this is exact.
            for cid, entry in list(ctx.conn_manager.active.items()):
                if entry.get("ws") is intercepted or entry.get("ws") is adapter:
                    ctx.conn_manager.disconnect(cid)
                    break


__all__ = [
    "AppendFrameInterceptor",
    "ConnectionManager",
    "register_websocket_routes",
]
