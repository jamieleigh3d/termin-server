# Copyright 2026 Jamie-Leigh Blake and Termin project contributors
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Smoke tests for termin-server.

These verify the package's public surface — that ``termin_server``
imports cleanly, that ``create_termin_app`` returns a usable FastAPI
app, and that booting against minimal IR produces a working HTTP
server. They are the first line of defense against import-graph
breakage like the slice-7.5 ``termin_runtime`` cleanup that left a
dangling import in the compiler CLI.
"""

from __future__ import annotations

import inspect
import json

import pytest
from fastapi import FastAPI


class TestPackageSurface:
    def test_create_termin_app_is_exported(self):
        import termin_server
        assert hasattr(termin_server, "create_termin_app")
        assert callable(termin_server.create_termin_app)

    def test_no_termin_runtime_import(self):
        """Slice 7.5a deleted the ``termin_runtime`` shim layer.
        termin-server must import cleanly without falling back to it,
        otherwise alternate runtimes that ``pip install termin-server``
        without termin-compiler will fail at import time."""
        import sys
        # Triggering a fresh import path; re-importing termin_server
        # should not bring termin_runtime along.
        import termin_server  # noqa: F401
        assert "termin_runtime" not in sys.modules, (
            "termin-server must not import termin_runtime — that "
            "package was deleted in slice 7.5a"
        )

    def test_create_termin_app_signature(self):
        from termin_server import create_termin_app
        sig = inspect.signature(create_termin_app)
        params = sig.parameters
        # The factory must accept an IR JSON string as its first
        # positional argument; everything else is optional.
        assert "ir_json" in params
        assert params["ir_json"].kind in (
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.POSITIONAL_ONLY,
        )


class TestAppBoot:
    def test_create_termin_app_returns_fastapi(self, hello_ir, tmp_path):
        from termin_server import create_termin_app
        app = create_termin_app(
            hello_ir, db_path=str(tmp_path / "hello.db"))
        assert isinstance(app, FastAPI)

    def test_app_has_routes(self, hello_ir, tmp_path):
        """The IR's pages and reflection endpoint must lower into
        registered routes."""
        from termin_server import create_termin_app
        app = create_termin_app(
            hello_ir, db_path=str(tmp_path / "hello.db"))
        paths = {route.path for route in app.routes}
        # Reflection always emits /api/v1/_runtime/* endpoints when
        # reflection_enabled is true (which the IR sets).
        assert any(
            "_runtime" in p or "/hello" in p for p in paths
        ), f"Expected reflection or page routes; got {sorted(paths)[:10]}"

    def test_app_serves_root_after_boot(self, hello_client):
        """A booted app must respond to HTTP. The status code is
        permissive — 200 (rendered page), 307 (redirect to default
        page), or 404 (no root route registered) are all acceptable
        evidence the app is alive; only a 5xx or connection error
        would fail this smoke test."""
        resp = hello_client.get("/")
        assert resp.status_code < 500, (
            f"Root request returned 5xx — app is broken at boot: "
            f"{resp.status_code} {resp.text[:200]}"
        )

    def test_invalid_ir_json_raises_clear_error(self, tmp_path):
        """Malformed IR should raise at create_termin_app, not at
        first request — fail-fast on operator misconfig."""
        from termin_server import create_termin_app
        with pytest.raises(Exception):
            create_termin_app(
                "{not valid json", db_path=str(tmp_path / "x.db"))
