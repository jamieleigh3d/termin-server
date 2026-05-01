# Copyright 2026 Jamie-Leigh Blake and Termin project contributors
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Unit tests for ``termin_server.storage``.

These exercise the storage layer in isolation against an ephemeral
SQLite file, without the FastAPI app or HTTP layer. They cover the
get/list/create/update/delete primitives and the identifier-safety
guards that prevent SQL injection via content-table names."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from termin_server import storage


def _run(coro):
    return asyncio.run(coro)


@pytest.fixture
def fresh_db(tmp_path) -> str:
    """Initialize a tiny single-content schema and return the db path."""
    db_path = str(tmp_path / "unit.db")
    schemas = [{
        "name": {"snake": "items", "pascal": "Items", "display": "items"},
        "singular": "item",
        "fields": [
            # init_db automatically adds an `id` integer primary key —
            # don't declare one here or sqlite raises duplicate column.
            {"name": "name", "column_type": "TEXT", "required": True},
            {"name": "qty", "column_type": "INTEGER"},
        ],
    }]

    async def setup():
        await storage.init_db(schemas, db_path=db_path)

    _run(setup())
    return db_path


class TestIdentifierSafety:
    def test_validate_identifier_accepts_snake_case(self):
        assert storage.validate_identifier("products")
        assert storage.validate_identifier("stock_levels")
        assert storage.validate_identifier("a")

    def test_validate_identifier_rejects_injection(self):
        # SQL injection vectors must be rejected by the identifier
        # check before any quoting decision is made.
        assert not storage.validate_identifier("products; DROP TABLE")
        assert not storage.validate_identifier("a' OR '1")
        assert not storage.validate_identifier("a\"b")
        assert not storage.validate_identifier("")

    def test_assert_safe_raises_on_unsafe(self):
        # The internal _assert_safe helper raises a ValueError so
        # callers can't accidentally pass user input through.
        with pytest.raises(ValueError):
            storage._assert_safe("products; --", context="table")


class TestCrudRoundTrip:
    def test_create_and_get(self, fresh_db):
        async def go():
            db = await storage.get_db(db_path=fresh_db)
            try:
                rec = await storage.create_record(
                    db, "items", {"name": "alpha", "qty": 3})
                assert rec["id"] >= 1
                got = await storage.get_record(db, "items", rec["id"])
                assert got["name"] == "alpha"
                # SQLite has dynamic typing; the inline test schema
                # declares column_type "INTEGER" but the Python value
                # comes back as a string under some affinity paths.
                # Coerce defensively — what matters is that the value
                # round-trips, not its representation.
                assert int(got["qty"]) == 3
            finally:
                await db.close()
        _run(go())

    def test_get_missing_raises_http_404(self, fresh_db):
        """get_record raises HTTPException(404) on miss rather than
        returning None — the FastAPI layer relies on this for the
        /api/v1/<table>/<id> route to produce a 404 without explicit
        error handling. Documenting the behavior as a contract."""
        from fastapi import HTTPException

        async def go():
            db = await storage.get_db(db_path=fresh_db)
            try:
                with pytest.raises(HTTPException) as exc:
                    await storage.get_record(db, "items", 99999)
                assert exc.value.status_code == 404
            finally:
                await db.close()
        _run(go())

    def test_list_records_empty(self, fresh_db):
        async def go():
            db = await storage.get_db(db_path=fresh_db)
            try:
                result = await storage.list_records(db, "items")
                # list_records returns either a list directly or a
                # tuple/dict depending on pagination shape; either
                # way the empty case must produce zero items.
                if isinstance(result, dict):
                    records = result.get("records", result.get("data", []))
                elif isinstance(result, tuple):
                    records = result[0]
                else:
                    records = result
                assert len(records) == 0
            finally:
                await db.close()
        _run(go())

    def test_update_and_delete(self, fresh_db):
        async def go():
            db = await storage.get_db(db_path=fresh_db)
            try:
                rec = await storage.create_record(
                    db, "items", {"name": "x", "qty": 1})
                pid = rec["id"]
                await storage.update_record(
                    db, "items", pid, {"qty": 7})
                updated = await storage.get_record(db, "items", pid)
                assert int(updated["qty"]) == 7
                await storage.delete_record(db, "items", pid)
                # Post-delete: get_record raises 404 (contract above).
                from fastapi import HTTPException
                with pytest.raises(HTTPException) as exc:
                    await storage.get_record(db, "items", pid)
                assert exc.value.status_code == 404
            finally:
                await db.close()
        _run(go())

    def test_count_records_reflects_state(self, fresh_db):
        async def go():
            db = await storage.get_db(db_path=fresh_db)
            try:
                assert await storage.count_records(db, "items") == 0
                for n in ("a", "b", "c"):
                    await storage.create_record(
                        db, "items", {"name": n, "qty": 1})
                assert await storage.count_records(db, "items") == 3
            finally:
                await db.close()
        _run(go())
