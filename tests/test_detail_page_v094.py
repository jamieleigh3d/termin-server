# Copyright 2026 Jamie-Leigh Blake and Termin project contributors
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Runtime tests for the v0.9.4 Phase 2 detail-page primitive.

`Show a detail page for <plural> called "<name>"` lowers to a
PageEntry with `record_binding=<plural>`. The runtime registers
this page at `/<slug>/{id}` instead of `/<slug>` and fetches the
record server-side before rendering — returning 404 cleanly for
missing or non-owned records (the latter using the same surface
to avoid leaking existence).

The React component on the page reads the `{id}` from the URL
and re-fetches via /api/v1/<plural>/<id> to get the live record;
the server-side fetch in this slice exists for routing + auth
gating, not for client data delivery (that's a v0.10 bound_data
pass).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

COMPILER_ROOT = Path(__file__).parent.parent.parent / "termin-compiler"
sys.path.insert(0, str(COMPILER_ROOT))

from termin.peg_parser import parse_peg  # noqa: E402
from termin.analyzer import analyze  # noqa: E402
from termin.lower import lower  # noqa: E402
from termin_core.ir.serialize import serialize_ir  # noqa: E402

from termin_server import create_termin_app  # noqa: E402


_SOURCE = """
Application: DetailPageTest
Description: Phase 2 runtime smoke for detail-page primitive.

Identity:
  Scopes are "play"
  A "player" has "play"

Content called "notes":
  Each note has a player_principal which is principal, required
  Each note is owned by player_principal
  Each note has a title which is text, required
  Each note has a body which is text
  Anyone with "play" can view their own notes
  Anyone with "play" can create notes
  Anyone with "play" can update their own notes

As a player, I want to see my notes so that I can pick one:
  Show a page called "Notes"
  Display a table of notes

As a player, I want to see one note in detail so that I can read it:
  Show a detail page for notes called "Note Detail"
"""


def _compile_to_ir_json() -> str:
    program, errors = parse_peg(_SOURCE)
    assert errors.ok, f"Parse errors: {errors.format()}"
    result = analyze(program)
    assert result.ok, f"Analyzer errors: {result.format()}"
    spec = lower(program)
    return serialize_ir(spec)


@pytest.fixture
def detail_page_client(tmp_path) -> TestClient:
    ir_json = _compile_to_ir_json()
    db_path = str(tmp_path / "detail.db")
    app = create_termin_app(ir_json, db_path=db_path)
    with TestClient(app) as client:
        client.cookies.set("termin_role", "player")
        client.cookies.set("termin_user_name", "alice")
        yield client


class TestDetailPageRoute:
    def test_detail_route_registered_at_slug_id(self, detail_page_client):
        """The /note_detail/{id} route should exist (200 for a real id,
        404 for nonexistent) — proving the page lowered to the right
        path shape. The list page /notes still exists separately."""
        client = detail_page_client
        # Create a note so we have an id to address.
        r = client.post(
            "/api/v1/notes",
            json={"title": "Hello", "body": "World"},
        )
        assert r.status_code == 201, r.text
        note_id = r.json()["id"]
        # Detail route returns 200 for the real id.
        r = client.get(f"/note_detail/{note_id}")
        assert r.status_code == 200, r.text
        # The list-page slug also still works — separate route.
        r = client.get("/notes")
        assert r.status_code == 200

    def test_detail_route_returns_404_for_missing_id(self, detail_page_client):
        """Missing ids must 404 cleanly server-side rather than
        rendering an empty shell that hydrates and then fails."""
        client = detail_page_client
        r = client.get("/note_detail/nonexistent-id")
        assert r.status_code == 404

    def test_detail_route_returns_404_for_other_principals_record(
        self, detail_page_client,
    ):
        """Ownership-filtered content: a record owned by another
        principal must surface as 404, NOT 403 — so ownership doesn't
        leak existence (per BRD §3.7 / matches the append route
        and the auto-CRUD GET handler patterns)."""
        client = detail_page_client
        # Create a note as alice.
        r = client.post(
            "/api/v1/notes",
            json={"title": "alices-note", "body": "private"},
        )
        assert r.status_code == 201
        alice_note_id = r.json()["id"]
        # Switch to bob.
        client.cookies.set("termin_user_name", "bob")
        # bob's GET on alice's note must 404, not 403.
        r = client.get(f"/note_detail/{alice_note_id}")
        assert r.status_code == 404

    def test_regular_pages_unaffected(self, detail_page_client):
        """Smoke check: regular non-detail pages still serve at
        /<slug> after the dispatcher learned about detail pages.
        The /notes list page should still work."""
        client = detail_page_client
        r = client.get("/notes")
        assert r.status_code == 200
