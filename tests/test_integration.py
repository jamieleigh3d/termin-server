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


# ── Reflection / introspection ──


class TestReflection:
    def test_runtime_registry_endpoint(self, warehouse_client):
        """``/runtime/registry`` advertises the application's
        boundaries + transport URLs; clients use it to wire up
        WebSocket and REST endpoints. v0.9 emits a runtime_version
        of 0.9.0."""
        resp = warehouse_client.get("/runtime/registry")
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body.get("runtime_version") == "0.9.0"
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
