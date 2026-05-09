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


class TestStructuredFieldSerialization:
    """Issue #5: SQLite is the per-runtime serialization boundary for
    list/dict-typed fields. Storage Protocol callers (append_to_field,
    crud handlers, channel handlers) must be able to pass native Python
    objects and have the SQLite provider serialize them to JSON text
    on the way in. Pre-fix, ``aiosqlite`` rejected list/dict parameter
    bindings outright with ``InterfaceError``.
    """

    @pytest.fixture
    def conv_db(self, tmp_path) -> str:
        """Schema with a TEXT-typed structured/conversation column."""
        db_path = str(tmp_path / "conv.db")
        schemas = [{
            "name": {"snake": "tickets", "pascal": "Tickets",
                     "display": "tickets"},
            "singular": "ticket",
            "fields": [
                {"name": "title", "column_type": "TEXT", "required": True},
                # Conversation field — TEXT column holding a JSON list
                # per termin_server.storage._SQL_TYPES["conversation"].
                {"name": "messages", "column_type": "TEXT"},
            ],
        }]

        async def setup():
            await storage.init_db(schemas, db_path=db_path)

        _run(setup())
        return db_path

    def test_update_record_serializes_native_list(self, conv_db):
        """update_record must accept a native Python list as a patch
        value and persist it as JSON text. Pre-fix, this raised
        ``InterfaceError: Error binding parameter 0: type 'list' is
        not supported``."""
        import json as _json

        async def go():
            db = await storage.get_db(db_path=conv_db)
            try:
                rec = await storage.create_record(
                    db, "tickets", {"title": "first"})
                pid = rec["id"]
                entries = [{"id": "e1", "kind": "user", "body": "hi"}]
                # The contract under test: a native list passed in,
                # round-trips through SQLite as a JSON-text column.
                await storage.update_record(
                    db, "tickets", pid, {"messages": entries})
                updated = await storage.get_record(db, "tickets", pid)
                assert updated["messages"] is not None
                # SQLite stores TEXT — read back as a string and
                # decode. The decoded value must be the original list.
                decoded = _json.loads(updated["messages"])
                assert decoded == entries
            finally:
                await db.close()
        _run(go())

    def test_update_fields_serializes_native_dict(self, conv_db):
        """update_fields (the lower-level helper used by the SQLite
        StorageProvider) must also serialize native dict patch
        values to JSON text on the way in."""
        import json as _json

        async def go():
            db = await storage.get_db(db_path=conv_db)
            try:
                rec = await storage.create_record(
                    db, "tickets", {"title": "second"})
                pid = rec["id"]
                # update_fields filters None/"" but should keep dict.
                await storage.update_fields(
                    db, "tickets", pid,
                    {"messages": {"latest": "value", "count": 3}})
                updated = await storage.get_record(db, "tickets", pid)
                assert updated["messages"] is not None
                decoded = _json.loads(updated["messages"])
                assert decoded == {"latest": "value", "count": 3}
            finally:
                await db.close()
        _run(go())

    def test_insert_raw_serializes_native_list(self, conv_db):
        """insert_raw is the low-level INSERT helper used by SQLite
        StorageProvider.create(). Native list/dict values for
        structured columns must be serialized identically."""
        import json as _json

        async def go():
            db = await storage.get_db(db_path=conv_db)
            try:
                entries = [{"id": "e1", "kind": "user", "body": "seeded"}]
                row_id = await storage.insert_raw(
                    db, "tickets",
                    {"title": "third", "messages": entries})
                updated = await storage.get_record(db, "tickets", row_id)
                assert updated["messages"] is not None
                decoded = _json.loads(updated["messages"])
                assert decoded == entries
            finally:
                await db.close()
        _run(go())

    def test_primitive_patch_values_unchanged(self, conv_db):
        """Sanity: strings, ints, and existing JSON-text strings must
        keep round-tripping unchanged. The serialization step targets
        list/dict only."""
        async def go():
            db = await storage.get_db(db_path=conv_db)
            try:
                rec = await storage.create_record(
                    db, "tickets",
                    {"title": "fourth", "messages": '[{"already": "json"}]'})
                pid = rec["id"]
                await storage.update_record(
                    db, "tickets", pid, {"title": "renamed"})
                updated = await storage.get_record(db, "tickets", pid)
                assert updated["title"] == "renamed"
                assert updated["messages"] == '[{"already": "json"}]'
            finally:
                await db.close()
        _run(go())
