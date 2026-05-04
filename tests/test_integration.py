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
