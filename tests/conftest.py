# Copyright 2026 Jamie-Leigh Blake and Termin project contributors
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Shared fixtures for the termin-server own-test suite.

The fixtures here build apps from .termin.pkg files copied into
tests/fixtures/ — hello (zero-table page-only) for smoke + page
tests, warehouse (multi-content + state machines + scopes) for
integration tests that need a real CRUD surface.

These are not meant to replace conformance — they exercise
termin-server's own moving parts (storage, identity, presentation,
errors) directly so a regression in this repo lights up here
without needing the full conformance run."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from termin_server import create_termin_app


FIXTURES = Path(__file__).parent / "fixtures"


def _ir_from_pkg(pkg_name: str) -> str:
    pkg_path = FIXTURES / pkg_name
    with zipfile.ZipFile(pkg_path) as zf:
        manifest = json.loads(zf.read("manifest.json"))
        return zf.read(manifest["ir"]["entry"]).decode("utf-8")


def _seed_from_pkg(pkg_name: str) -> dict | None:
    pkg_path = FIXTURES / pkg_name
    with zipfile.ZipFile(pkg_path) as zf:
        manifest = json.loads(zf.read("manifest.json"))
        seed_entry = manifest.get("seed")
        if not seed_entry:
            return None
        try:
            return json.loads(zf.read(seed_entry).decode("utf-8"))
        except (KeyError, json.JSONDecodeError):
            return None


@pytest.fixture
def hello_ir() -> str:
    """Hello-world IR: zero content tables, one anonymous page. Used
    for the boot-and-render smoke tests where a CRUD surface would
    add irrelevant complexity."""
    return _ir_from_pkg("hello.termin.pkg")


@pytest.fixture
def warehouse_ir() -> str:
    """Warehouse IR: multiple content tables, scopes, state machine.
    Used for integration tests that need a real CRUD surface."""
    return _ir_from_pkg("warehouse.termin.pkg")


@pytest.fixture
def hello_client(hello_ir, tmp_path) -> TestClient:
    db_path = str(tmp_path / "hello.db")
    app = create_termin_app(hello_ir, db_path=db_path)
    with TestClient(app) as client:
        yield client


@pytest.fixture
def warehouse_client(warehouse_ir, tmp_path) -> TestClient:
    db_path = str(tmp_path / "warehouse.db")
    seed_data = _seed_from_pkg("warehouse.termin.pkg")
    app = create_termin_app(
        warehouse_ir, db_path=db_path, seed_data=seed_data)
    with TestClient(app) as client:
        yield client
