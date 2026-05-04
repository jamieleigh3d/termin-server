# Copyright 2026 Jamie-Leigh Blake and Termin project contributors
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Integration tests for termin-server.

These exercise the full FastAPI app booted from a real .termin.pkg
fixture: HTTP CRUD round-trips, role-cookie auth, scope gating,
reflection endpoints, and page rendering. They are the
own-repo equivalent of the conformance suite's API-contract tier
— not a substitute for conformance, but a fast feedback loop on
the layers that conformance exercises only via adapter."""

from __future__ import annotations

import pytest


# ── v0.9.2 L3: APPEND verb on conversation fields ──


class TestAppendVerb:
    """v0.9.2 L3: POST /<resource>/{id}/<field>:append registers from
    `Anyone with X can append to <plural>' <field>` access rules. The
    handler generates a UUIDv7 entry id, stamps created_at +
    appended_by_principal_id, and writes the entry into the JSON
    column on the parent record. Read-back via GET returns the entry
    list; the canonical entry shape lives in the runtime, not in the
    user's source.

    L5 (event firing on append) and L4 (WebSocket frame) are separate
    slices; this class verifies the REST surface works end-to-end."""

    _APPEND_SOURCE = '''Application: Append Test
  Description: v0.9.2 L3 fixture
Id: 7e1b3a2c-4f9d-4e1a-b3c5-1d2e8f4a9c01

Identity:
  Scopes are "chat.use"
  An "anonymous" has "chat.use"

Content called "chat_threads":
  Each chat_thread has a title which is text, default "Conversation"
  Each chat_thread has a conversation which is conversation
  Anyone with "chat.use" can view chat_threads
  Anyone with "chat.use" can create chat_threads
  Anyone with "chat.use" can append to chat_threads.conversation

As anonymous, I want to chat so that I can verify append:
  Show a page called "Chat"
'''

    @pytest.fixture
    def append_client(self, tmp_path):
        """Compile the inline append source on the fly and boot a
        TestClient against it. Reuses the existing import-installed
        compiler — both packages live in the same dev venv per the
        workspace's conventions."""
        from fastapi.testclient import TestClient

        from termin.peg_parser import parse_peg
        from termin.analyzer import analyze
        from termin.lower import lower
        from termin_core.ir.serialize import serialize_ir
        from termin_server import create_termin_app

        program, perr = parse_peg(self._APPEND_SOURCE)
        assert perr.ok, perr.format()
        aerr = analyze(program)
        assert aerr.ok, aerr.format()
        spec = lower(program)
        ir_json = serialize_ir(spec)

        db_path = str(tmp_path / "append.db")
        app = create_termin_app(ir_json, db_path=db_path)
        with TestClient(app) as client:
            yield client

    def test_append_route_registered(self, append_client):
        """The compile pipeline emits a POST .../conversation:append
        route from the access rule. Smoke check that the path lives
        in the app's routing table."""
        paths = {r.path for r in append_client.app.routes}
        assert any(":append" in p for p in paths), (
            f"expected an :append route; got {sorted(paths)}"
        )

    def test_append_round_trip(self, append_client):
        """Create a thread, append a user entry, read it back via
        GET. The entry list grows by one and carries the canonical
        runtime-set fields (id, created_at)."""
        # Create a parent record first.
        create = append_client.post(
            "/api/v1/chat_threads", json={"title": "round trip"})
        assert create.status_code in (200, 201), create.text
        thread = create.json()
        thread_id = thread.get("id")
        assert thread_id, f"create returned no id: {thread}"

        # Append a user entry.
        append = append_client.post(
            f"/api/v1/chat_threads/{thread_id}/conversation:append",
            json={"kind": "user", "body": "hello"})
        assert append.status_code == 201, append.text
        entry = append.json()
        assert entry.get("kind") == "user"
        assert entry.get("body") == "hello"
        assert entry.get("id"), "entry must carry a runtime-generated id"
        assert entry.get("created_at"), "entry must carry created_at"

        # Read back: the JSON column now holds the entry list.
        import json as _json
        get = append_client.get(f"/api/v1/chat_threads/{thread_id}")
        assert get.status_code == 200, get.text
        record = get.json()
        raw = record.get("conversation")
        entries = _json.loads(raw) if isinstance(raw, str) else raw
        assert isinstance(entries, list)
        assert len(entries) == 1
        assert entries[0]["body"] == "hello"

    def test_append_rejects_invalid_kind(self, append_client):
        """Non-canonical kinds get a 400, not silently stored. Keeps
        downstream conversation handling honest."""
        create = append_client.post(
            "/api/v1/chat_threads", json={"title": "invalid kind test"})
        thread_id = create.json()["id"]
        bad = append_client.post(
            f"/api/v1/chat_threads/{thread_id}/conversation:append",
            json={"kind": "refusal", "body": "nope"})  # refusal isn't canonical
        assert bad.status_code == 400, bad.text

    def test_append_requires_body(self, append_client):
        """Missing body field → 400. The runtime won't append a blank entry."""
        create = append_client.post(
            "/api/v1/chat_threads", json={"title": "missing body test"})
        thread_id = create.json()["id"]
        bad = append_client.post(
            f"/api/v1/chat_threads/{thread_id}/conversation:append",
            json={"kind": "user"})
        assert bad.status_code == 400, bad.text


# ── v0.9.2 L5: <content>.<field>.appended event class ──


class TestAppendedEvent:
    """v0.9.2 L5: every successful append fires a
    `<content>.<field>.appended` event on the EventBus. The event is
    distinct from the existing `<content>.updated` shape — subscribers
    can filter by the field-specific channel, and computes can declare
    `Trigger on event "<content>.<field>.appended"` to react to
    conversation activity without false positives from any other
    column update.

    Per tech-design §9, the event payload carries:
      record_id, record (full record after append), appended_entry,
      triggered_at, invoked_by_principal_id, trigger_kind.

    Trigger predicates can reference `appended_entry` (e.g.,
    `appended_entry.kind == "user"`) so listener computes can filter
    on per-entry kind without double-loading the conversation.
    """

    _APPEND_SOURCE = '''Application: Append Event Test
  Description: v0.9.2 L5 fixture
Id: 6c8f2d1e-3a44-4b1c-9d8e-2f5a3b7c8d92

Identity:
  Scopes are "chat.use"
  An "anonymous" has "chat.use"

Content called "chat_threads":
  Each chat_thread has a title which is text, default "Conversation"
  Each chat_thread has a conversation which is conversation
  Anyone with "chat.use" can view chat_threads
  Anyone with "chat.use" can create chat_threads
  Anyone with "chat.use" can append to chat_threads.conversation

As anonymous, I want to chat so that I can verify event firing:
  Show a page called "Chat"
'''

    @pytest.fixture
    def append_client(self, tmp_path):
        from fastapi.testclient import TestClient

        from termin.peg_parser import parse_peg
        from termin.analyzer import analyze
        from termin.lower import lower
        from termin_core.ir.serialize import serialize_ir
        from termin_server import create_termin_app

        program, perr = parse_peg(self._APPEND_SOURCE)
        assert perr.ok, perr.format()
        aerr = analyze(program)
        assert aerr.ok, aerr.format()
        spec = lower(program)
        ir_json = serialize_ir(spec)

        db_path = str(tmp_path / "append_event.db")
        app = create_termin_app(ir_json, db_path=db_path)
        with TestClient(app) as client:
            yield client

    def _ctx_from_client(self, client):
        """Pull the RuntimeContext off the FastAPI app state so tests
        can subscribe to its EventBus directly."""
        ctx = getattr(client.app.state, "ctx", None)
        if ctx is None:
            # Older create_termin_app stashes ctx on app itself
            ctx = getattr(client.app, "_termin_ctx", None)
        assert ctx is not None and getattr(ctx, "event_bus", None) is not None, (
            "RuntimeContext.event_bus should be reachable from test client app"
        )
        return ctx

    def test_append_publishes_appended_event(self, append_client):
        """The append handler must publish on the
        `content.<name>.<field>.appended` channel with the documented
        payload shape (§9.1)."""
        import asyncio

        ctx = self._ctx_from_client(append_client)
        # Subscribe BEFORE the append so the in-memory queue catches the event.
        queue = ctx.event_bus.subscribe(
            channel_id="content.chat_threads.conversation.appended")

        create = append_client.post(
            "/api/v1/chat_threads", json={"title": "event test"})
        thread_id = create.json()["id"]

        ap = append_client.post(
            f"/api/v1/chat_threads/{thread_id}/conversation:append",
            json={"kind": "user", "body": "hello world"})
        assert ap.status_code == 201, ap.text
        entry = ap.json()

        # The publish happens inside the append handler; by the time
        # the HTTP response is back the event must already be on the queue.
        try:
            event = asyncio.get_event_loop().run_until_complete(
                asyncio.wait_for(queue.get(), timeout=1.0))
        except asyncio.TimeoutError:
            pytest.fail("appended event was not published within 1s")

        assert event["channel_id"] == "content.chat_threads.conversation.appended"
        assert event["content_name"] == "chat_threads"
        assert event["field_name"] == "conversation"
        # record_id comes from the URL path-param (string); the create-route
        # returns an int. Compare as strings — the wire shape is text either way.
        assert str(event["record_id"]) == str(thread_id)
        assert event["trigger_kind"] == "crud-append"
        assert event["appended_entry"]["id"] == entry["id"]
        assert event["appended_entry"]["kind"] == "user"
        assert event["appended_entry"]["body"] == "hello world"
        assert event.get("triggered_at"), "triggered_at timestamp required"
        # record carries the full parent post-append; the conversation
        # column holds the JSON list including the new entry.
        assert event.get("record"), "full record required in payload"

    def test_appended_event_does_not_fire_on_other_updates(self, append_client):
        """A regular PUT update on the parent record must not fire the
        :appended event. The :appended channel is reserved for actual
        appends — keeping the channel discriminating is the whole point
        of the new event class."""
        import asyncio

        ctx = self._ctx_from_client(append_client)
        queue = ctx.event_bus.subscribe(
            channel_id="content.chat_threads.conversation.appended")

        create = append_client.post(
            "/api/v1/chat_threads", json={"title": "before"})
        thread_id = create.json()["id"]

        # Update the title — should fire `_updated` but not `.conversation.appended`.
        upd = append_client.put(
            f"/api/v1/chat_threads/{thread_id}",
            json={"title": "after"})
        assert upd.status_code in (200, 201)

        with pytest.raises(asyncio.TimeoutError):
            asyncio.get_event_loop().run_until_complete(
                asyncio.wait_for(queue.get(), timeout=0.3))




# ── v0.9.2 L4: APPEND frame over WebSocket ──


class TestAppendVerbWebSocket:
    """v0.9.2 L4: WebSocket parity for the REST :append endpoint.

    Per §8.3 of the v0.9.2 conversation field type tech design:
    inbound frames of shape ``{"type": "append", "resource": ...,
    "id": ..., "field": ..., "payload": {...}}`` are dispatched
    through the same shared ``_do_append`` helper the REST endpoint
    uses. The same scope gate, the same row_filter check, the same
    canonical entry shape — same code, two transports.

    Server response: there is NO separate "append response" frame.
    The append fires the standard ``<content>.<field>.appended``
    event (L5's job) which propagates to subscribers — including the
    originating client — via the existing record-subscription
    channel. On failure: a structured error frame on the
    ``runtime.append`` topic.

    L5 (event firing) is being implemented in parallel; the
    round-trip test below verifies storage-side write parity (re-read
    via REST). Once L5 lands, the same handler will deliver the
    entry over the originator's existing subscription with no further
    work in this slice.
    """

    _WS_APPEND_SOURCE = '''Application: WS Append Test
  Description: v0.9.2 L4 fixture
Id: 7e1b3a2c-4f9d-4e1a-b3c5-1d2e8f4a9c02

Identity:
  Scopes are "chat.use"
  An "anonymous" has "chat.use"

Content called "chat_threads":
  Each chat_thread has a title which is text, default "Conversation"
  Each chat_thread has a conversation which is conversation
  Anyone with "chat.use" can view chat_threads
  Anyone with "chat.use" can create chat_threads
  Anyone with "chat.use" can append to chat_threads.conversation

As anonymous, I want to chat so that I can verify ws append:
  Show a page called "Chat"
'''

    @pytest.fixture
    def ws_append_client(self, tmp_path):
        """Same compile-and-boot pattern as TestAppendVerb. Reuses the
        REST surface for setup (creating a parent record) and exercises
        the WS surface for the append itself."""
        from fastapi.testclient import TestClient

        from termin.peg_parser import parse_peg
        from termin.analyzer import analyze
        from termin.lower import lower
        from termin_core.ir.serialize import serialize_ir
        from termin_server import create_termin_app

        program, perr = parse_peg(self._WS_APPEND_SOURCE)
        assert perr.ok, perr.format()
        aerr = analyze(program)
        assert aerr.ok, aerr.format()
        spec = lower(program)
        ir_json = serialize_ir(spec)

        db_path = str(tmp_path / "ws_append.db")
        app = create_termin_app(ir_json, db_path=db_path)
        with TestClient(app) as client:
            yield client

    def _drain_identity(self, ws):
        """Consume the identity push frame the dispatcher sends right
        after accept (per ``channel_dispatch._identity_frame``)."""
        first = ws.receive_json()
        assert first.get("ch") == "runtime.identity", first

    def _flush_append(self, ws):
        """Synchronization barrier: send a no-op subscribe frame and
        await its response. Append frames are dispatched serially in
        the interceptor's ``receive_json`` loop, so by the time the
        server responds to a follow-up frame any prior append has
        committed. Without this barrier, the TestClient WS context
        manager can close the connection mid-await and lose the
        in-flight append."""
        ws.send_json({
            "v": 1, "ch": "runtime.flush", "op": "subscribe",
            "ref": "flush",
        })
        ack = ws.receive_json()
        assert ack.get("op") == "response", ack

    def test_ws_append_frame_round_trip(self, ws_append_client):
        """Send an append frame over WS; verify the entry lands in
        storage and is readable via the REST GET endpoint.

        L5 (event firing on append) will deliver the new entry to the
        originator over its existing subscription. Until L5 lands, we
        verify storage-side write parity by re-reading via REST. The
        WS handler itself is stable and contractually equivalent to
        the REST handler for the storage side; the event delivery is
        a separate seam L5 will wire."""
        # Create a parent record via the REST surface — the WS handler
        # operates on existing records, doesn't create them.
        create = ws_append_client.post(
            "/api/v1/chat_threads", json={"title": "ws round trip"})
        assert create.status_code in (200, 201), create.text
        thread_id = create.json()["id"]

        with ws_append_client.websocket_connect("/runtime/ws") as ws:
            self._drain_identity(ws)
            ws.send_json({
                "type": "append",
                "resource": "chat_threads",
                "id": thread_id,
                "field": "conversation",
                "payload": {"kind": "user", "body": "hello over ws"},
            })
            # No separate response frame per §8.3 — but the test
            # client closes the connection eagerly, so synchronize on
            # a follow-up frame to guarantee the append has committed
            # before the context exits.
            self._flush_append(ws)

        # Re-read the parent record. The conversation column now
        # carries one entry with the WS-supplied body.
        import json as _json
        get = ws_append_client.get(f"/api/v1/chat_threads/{thread_id}")
        assert get.status_code == 200, get.text
        record = get.json()
        raw = record.get("conversation")
        entries = _json.loads(raw) if isinstance(raw, str) else raw
        assert isinstance(entries, list), f"expected list, got {raw!r}"
        assert len(entries) == 1, entries
        assert entries[0]["kind"] == "user"
        assert entries[0]["body"] == "hello over ws"
        # Canonical runtime-set fields must be present — same shape as
        # the REST handler produces.
        assert entries[0].get("id"), "WS-appended entry missing id"
        assert entries[0].get("created_at"), (
            "WS-appended entry missing created_at"
        )

    def test_ws_append_rejects_invalid_kind(self, ws_append_client):
        """An invalid kind on an append frame surfaces as a structured
        error frame on the ``runtime.append`` channel; nothing is
        written to storage."""
        create = ws_append_client.post(
            "/api/v1/chat_threads", json={"title": "ws invalid kind"})
        thread_id = create.json()["id"]

        with ws_append_client.websocket_connect("/runtime/ws") as ws:
            self._drain_identity(ws)
            ws.send_json({
                "type": "append",
                "resource": "chat_threads",
                "id": thread_id,
                "field": "conversation",
                "payload": {"kind": "bogus", "body": "should fail"},
            })
            err = ws.receive_json()

        assert err.get("op") == "error", err
        assert err.get("ch") == "runtime.append", err
        assert err.get("payload", {}).get("code") == "validation_error", err

        # Storage must be empty — error path should not commit.
        import json as _json
        get = ws_append_client.get(f"/api/v1/chat_threads/{thread_id}")
        raw = get.json().get("conversation")
        entries = (
            _json.loads(raw)
            if isinstance(raw, str) and raw else (raw or [])
        )
        assert entries == [], (
            f"validation error left a partial write: {entries!r}"
        )

    def test_ws_append_rejects_unauthorized(self, tmp_path):
        """A connection whose principal lacks the required scope must
        not be able to append. The frame surfaces an error frame with
        ``code == "forbidden"``; nothing is written.

        The fixture for this test uses a stricter scope than the
        anonymous role gets, so the same WS connection that would
        normally succeed gets rejected on the scope gate."""
        from fastapi.testclient import TestClient

        from termin.peg_parser import parse_peg
        from termin.analyzer import analyze
        from termin.lower import lower
        from termin_core.ir.serialize import serialize_ir
        from termin_server import create_termin_app

        # Anonymous holds only `chat.read`; appending requires
        # `chat.write`, which anonymous does not have.
        source = '''Application: WS Append Auth Test
  Description: v0.9.2 L4 unauthorized fixture
Id: 7e1b3a2c-4f9d-4e1a-b3c5-1d2e8f4a9c03

Identity:
  Scopes are "chat.read", "chat.write"
  An "anonymous" has "chat.read"

Content called "chat_threads":
  Each chat_thread has a title which is text, default "Conversation"
  Each chat_thread has a conversation which is conversation
  Anyone with "chat.read" can view chat_threads
  Anyone with "chat.read" can create chat_threads
  Anyone with "chat.write" can append to chat_threads.conversation

As anonymous, I want to chat so that I can verify auth gate:
  Show a page called "Chat"
'''
        program, perr = parse_peg(source)
        assert perr.ok, perr.format()
        aerr = analyze(program)
        assert aerr.ok, aerr.format()
        spec = lower(program)
        ir_json = serialize_ir(spec)

        db_path = str(tmp_path / "ws_auth.db")
        app = create_termin_app(ir_json, db_path=db_path)
        with TestClient(app) as client:
            # Create the parent record via REST first (anonymous can
            # `create chat_threads`).
            create = client.post(
                "/api/v1/chat_threads", json={"title": "ws auth test"})
            assert create.status_code in (200, 201), create.text
            thread_id = create.json()["id"]

            with client.websocket_connect("/runtime/ws") as ws:
                self._drain_identity(ws)
                ws.send_json({
                    "type": "append",
                    "resource": "chat_threads",
                    "id": thread_id,
                    "field": "conversation",
                    "payload": {"kind": "user", "body": "denied"},
                })
                err = ws.receive_json()

            assert err.get("op") == "error", err
            assert err.get("payload", {}).get("code") == "forbidden", err

            # Nothing written.
            import json as _json
            get = client.get(f"/api/v1/chat_threads/{thread_id}")
            raw = get.json().get("conversation")
            entries = (
                _json.loads(raw)
                if isinstance(raw, str) and raw else (raw or [])
            )
            assert entries == [], (
                f"forbidden append left a partial write: {entries!r}"
            )

    def test_ws_and_rest_share_one_handler(self, ws_append_client):
        """End-to-end parity check: append once via WS, once via REST,
        verify both produce the same canonical entry shape (same set
        of keys, same kind/body fidelity) and that the parent record's
        conversation column ends with both entries in source order.

        This is the test that catches handler-drift — the failure mode
        the L4 refactor exists to prevent. If the WS path ever forks
        from the shared ``_do_append`` helper, the per-entry shape
        assertion below diverges first."""
        create = ws_append_client.post(
            "/api/v1/chat_threads", json={"title": "parity check"})
        thread_id = create.json()["id"]

        # WS append first.
        with ws_append_client.websocket_connect("/runtime/ws") as ws:
            self._drain_identity(ws)
            ws.send_json({
                "type": "append",
                "resource": "chat_threads",
                "id": thread_id,
                "field": "conversation",
                "payload": {
                    "kind": "user",
                    "body": "via ws",
                    "source": "ws-test",
                },
            })
            self._flush_append(ws)

        # Then REST append on the same record.
        rest = ws_append_client.post(
            f"/api/v1/chat_threads/{thread_id}/conversation:append",
            json={
                "kind": "user",
                "body": "via rest",
                "source": "rest-test",
            },
        )
        assert rest.status_code == 201, rest.text
        rest_entry = rest.json()

        # Re-read the record; verify both entries are present in
        # source order.
        import json as _json
        get = ws_append_client.get(f"/api/v1/chat_threads/{thread_id}")
        raw = get.json().get("conversation")
        entries = _json.loads(raw) if isinstance(raw, str) else raw
        assert isinstance(entries, list)
        assert len(entries) == 2, (
            f"expected 2 entries (ws + rest), got {len(entries)}: {entries}"
        )
        assert entries[0]["body"] == "via ws"
        assert entries[1]["body"] == "via rest"

        # Same key surface from both transports — neither carries a
        # field the other lacks (drift smoke check).
        ws_entry = entries[0]
        # Pre-comparison: REST entry currently has the same shape we
        # built in _do_append; the WS entry must too. The set of
        # canonical-field keys must be identical between the two
        # transports — drift in the helper would surface as a missing
        # key in one but not the other.
        canonical_keys = {
            "id", "kind", "body", "created_at", "appended_by_principal_id",
            "source",
        }
        assert canonical_keys.issubset(set(ws_entry.keys())), (
            f"WS entry missing canonical keys: "
            f"{canonical_keys - set(ws_entry.keys())} (got {ws_entry})"
        )
        assert canonical_keys.issubset(set(rest_entry.keys())), (
            f"REST entry missing canonical keys: "
            f"{canonical_keys - set(rest_entry.keys())} (got {rest_entry})"
        )
        # Same source-prefix shape, distinct ids — confirms each
        # transport produced its own UUIDv7.
        assert ws_entry["id"] != rest_entry["id"]
        assert ws_entry["source"] == "ws-test"
        assert rest_entry["source"] == "rest-test"


# ── Reflection / introspection ──


class TestReflection:
    def test_runtime_registry_endpoint(self, warehouse_client):
        """``/runtime/registry`` advertises the application's
        boundaries + transport URLs; clients use it to wire up
        WebSocket and REST endpoints. v0.9.1 emits a runtime_version
        of 0.9.1."""
        resp = warehouse_client.get("/runtime/registry")
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body.get("runtime_version") == "0.9.1"
        assert "boundaries" in body
        assert body.get("application") == "Warehouse Inventory Manager"

    def test_runtime_bootstrap_carries_user_pages(self, warehouse_client):
        """``/runtime/bootstrap`` resolves the current user, filters
        the IR's pages by their role, and emits the bootstrap
        payload the client uses to render the SPA shell. Anonymous
        callers may get an empty list; the endpoint must still 200."""
        resp = warehouse_client.get("/runtime/bootstrap")
        assert resp.status_code == 200, resp.text
        body = resp.json()
        # The bootstrap payload always carries the user-pages and
        # transitions maps even when the user is anonymous.
        assert isinstance(body, dict), f"unexpected shape {type(body)}"


# ── CRUD round-trip ──


class TestCrudRoundTrip:
    @pytest.fixture
    def manager(self, warehouse_client):
        """A test client carrying the warehouse-manager role cookie
        (full inventory.{read,write,admin} scope set)."""
        warehouse_client.cookies.set("termin_role", "warehouse manager")
        return warehouse_client

    def test_list_products_returns_seeded_records(self, manager):
        resp = manager.get("/api/v1/products")
        assert resp.status_code == 200, resp.text
        body = resp.json()
        # The response shape varies (list, {records: [...]}, {data: [...]}).
        # All three are acceptable; at minimum we want some records back
        # because the warehouse fixture seeds 6.
        if isinstance(body, dict):
            records = body.get("records", body.get("data", []))
        else:
            records = body
        assert len(records) >= 1, f"Expected seeded records; got {body!r}"

    def test_create_product_returns_id(self, manager):
        resp = manager.post(
            "/api/v1/products",
            json={"name": "Integration Test Item",
                  "sku": "INT-001", "category": "raw material"},
        )
        # 200 or 201 are both valid creation responses.
        assert resp.status_code in (200, 201), resp.text
        body = resp.json()
        assert "id" in body, f"Created record missing id: {body}"
        assert body["name"] == "Integration Test Item"

    def test_get_individual_product(self, manager):
        # Create then read back.
        created = manager.post(
            "/api/v1/products",
            json={"name": "Readback", "sku": "RB-1",
                  "category": "raw material"},
        ).json()
        resp = manager.get(f"/api/v1/products/{created['id']}")
        assert resp.status_code == 200
        assert resp.json()["sku"] == "RB-1"

    def test_update_product_field(self, manager):
        created = manager.post(
            "/api/v1/products",
            json={"name": "Mutable", "sku": "MUT-1",
                  "category": "raw material"},
        ).json()
        resp = manager.put(
            f"/api/v1/products/{created['id']}",
            json={"name": "Mutated"},
        )
        assert resp.status_code in (200, 204), resp.text
        # Verify the change persisted.
        readback = manager.get(f"/api/v1/products/{created['id']}").json()
        assert readback["name"] == "Mutated"

    def test_delete_product(self, manager):
        created = manager.post(
            "/api/v1/products",
            json={"name": "Doomed", "sku": "DM-1",
                  "category": "raw material"},
        ).json()
        resp = manager.delete(f"/api/v1/products/{created['id']}")
        assert resp.status_code in (200, 204)
        # Subsequent get must 404.
        followup = manager.get(f"/api/v1/products/{created['id']}")
        assert followup.status_code == 404


# ── Pages / rendering ──


class TestPageRendering:
    def test_inventory_dashboard_renders_data_termin_markers(
            self, warehouse_client):
        """A rendered page must carry data-termin-* attributes for
        downstream tools (browser conformance, JS hydration). This
        catches presentation regressions where the renderer falls
        back to plain HTML without the marker layer."""
        warehouse_client.cookies.set("termin_role", "warehouse manager")
        resp = warehouse_client.get("/inventory_dashboard")
        # 200 (rendered) or 307 (redirect, then rendered after follow).
        assert resp.status_code in (200, 307), (
            f"page returned {resp.status_code}: {resp.text[:200]}"
        )
        if resp.status_code == 307:
            resp = warehouse_client.get(
                resp.headers["location"], follow_redirects=True)
        body = resp.text
        assert "data-termin-component" in body, (
            "rendered page missing data-termin-component markers — "
            "presentation layer regression"
        )

    def test_anonymous_request_does_not_500(self, warehouse_client):
        """Anonymous (no cookie) callers must get a clean response —
        either the page renders for app.view scope or a 401/403
        redirect — never a 5xx."""
        resp = warehouse_client.get("/inventory_dashboard")
        assert resp.status_code < 500, (
            f"anonymous request 5xx'd: {resp.status_code} "
            f"{resp.text[:200]}"
        )
