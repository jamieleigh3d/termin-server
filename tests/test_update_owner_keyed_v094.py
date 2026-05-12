# Copyright 2026 Jamie-Leigh Blake and Termin project contributors
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Runtime tests for the v0.9.4 cross-content-updates slice B5 —
owner-keyed Update action with upsert + bare-anonymous skip.

The compiler-side slices (B1b/B2/B3) lower
`Update the user's <singular>: <field> = `<cel>`` to an
`EventActionSpec` carrying `update_target_kind="owner-keyed"` and
`update_target_owner=<ownership-field-snake>`. This test exercises
the runtime side: when a When-rule fires and one of its actions
carries the owner-keyed discriminator, the runtime must:

  1. Resolve the target by querying `update_content` for the row
     whose `update_target_owner` field equals the event's
     principal id.
  2. If a target exists, apply the patch via `storage.update`.
  3. If no target exists AND the ownership field declares
     `unique`, upsert: build a default-valued record, apply
     the patch, insert via `storage.create`.
  4. If the principal is bare-anonymous (id == "anonymous"),
     skip the action with a log warning (no shared singleton
     mutations).

Test app shape:

    Content called "profiles":
      Each profile has a player_principal which is principal,
        required, unique
      Each profile is owned by player_principal
      Each profile has games_played which is a whole number,
        defaults to 0

    Content called "rounds":
      Each round has a player_principal which is principal, required
      Each round is owned by player_principal
      Each round has a status which is state:
        status starts as in_progress
        status can also be done
        in_progress can become done if the user has "play"

    When round status enters done:
      Update the user's profile:
        games_played = `profile.games_played + 1`
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
Application: Test
Description: B5 runtime smoke for owner-keyed Update action.

Identity:
  Scopes are "play"
  A "player" has "play"

Content called "profiles":
  Each profile has a player_principal which is principal, required, unique
  Each profile is owned by player_principal
  Each profile has games_played which is a whole number, defaults to 0
  Each profile has best_score which is a whole number, defaults to 0
  Anyone with "play" can view their own profiles
  Anyone with "play" can update their own profiles
  Anyone with "play" can create profiles

Content called "rounds":
  Each round has a player_principal which is principal, required
  Each round is owned by player_principal
  Each round has points which is a whole number, defaults to 0
  Each round has a status which is state:
    status starts as in_progress
    status can also be done
    in_progress can become done if the user has "play"
  Anyone with "play" can view rounds
  Anyone with "play" can create rounds
  Anyone with "play" can update rounds

When round status enters done:
  Update the user's profile: games_played = `profile.games_played + 1`
"""


def _compile_to_ir_json() -> str:
    program, errors = parse_peg(_SOURCE)
    assert errors.ok, f"Parse errors: {errors.format()}"
    result = analyze(program)
    assert result.ok, f"Analyzer errors: {result.format()}"
    spec = lower(program)
    return serialize_ir(spec)


@pytest.fixture
def owner_keyed_client(tmp_path) -> TestClient:
    ir_json = _compile_to_ir_json()
    db_path = str(tmp_path / "owner_keyed.db")
    app = create_termin_app(ir_json, db_path=db_path)
    with TestClient(app) as client:
        client.cookies.set("termin_role", "player")
        client.cookies.set("termin_user_name", "alice")
        yield client


class TestOwnerKeyedUpdateRuntime:

    def test_upsert_creates_profile_on_first_play(self, owner_keyed_client):
        """Player has no profile yet — finishing a round upserts one
        with games_played=1."""
        client = owner_keyed_client
        # Before any rounds: no profile.
        r = client.get("/api/v1/profiles")
        assert r.status_code == 200
        assert r.json() == [], f"Expected no profiles initially, got {r.json()}"

        # Create + finish a round.
        r = client.post("/api/v1/rounds", json={"points": 5})
        assert r.status_code == 201
        round_id = r.json()["id"]
        r = client.post(f"/_transition/rounds/status/{round_id}/done")
        assert r.status_code == 200, r.text

        # The owner-keyed Update should have upserted alice's profile.
        r = client.get("/api/v1/profiles")
        assert r.status_code == 200
        profiles = r.json()
        assert len(profiles) == 1, (
            f"Expected one profile after first round, got {len(profiles)}: "
            f"{profiles}"
        )
        assert profiles[0]["games_played"] == 1
        # Default for best_score is 0 (the upsert built it from defaults).
        assert profiles[0]["best_score"] == 0

    def test_update_increments_existing_profile(self, owner_keyed_client):
        """Second round increments the existing profile, doesn't
        create a duplicate."""
        client = owner_keyed_client
        # First round.
        r = client.post("/api/v1/rounds", json={"points": 5})
        rid = r.json()["id"]
        client.post(f"/_transition/rounds/status/{rid}/done")
        # Second round.
        r = client.post("/api/v1/rounds", json={"points": 7})
        rid = r.json()["id"]
        r = client.post(f"/_transition/rounds/status/{rid}/done")
        assert r.status_code == 200

        r = client.get("/api/v1/profiles")
        profiles = r.json()
        assert len(profiles) == 1, (
            f"Owner-keyed lookup must update the existing profile, "
            f"not create a second. Got: {profiles}"
        )
        assert profiles[0]["games_played"] == 2

    def test_two_players_independent_profiles(self, tmp_path):
        """Each session-bearing anonymous principal owns its own
        profile — owner-scoped lookup never crosses streams."""
        ir_json = _compile_to_ir_json()
        db_path = str(tmp_path / "owner_keyed_2.db")
        app = create_termin_app(ir_json, db_path=db_path)

        with TestClient(app) as client_a, TestClient(app) as client_b:
            client_a.cookies.set("termin_role", "player")
            client_a.cookies.set("termin_user_name", "alice")
            client_b.cookies.set("termin_role", "player")
            client_b.cookies.set("termin_user_name", "bob")

            # Alice plays a round.
            r = client_a.post("/api/v1/rounds", json={"points": 5})
            rid = r.json()["id"]
            client_a.post(f"/_transition/rounds/status/{rid}/done")

            # Bob plays a round.
            r = client_b.post("/api/v1/rounds", json={"points": 9})
            rid = r.json()["id"]
            client_b.post(f"/_transition/rounds/status/{rid}/done")

            # Alice sees only her profile.
            ra = client_a.get("/api/v1/profiles")
            assert len(ra.json()) == 1
            assert ra.json()[0]["games_played"] == 1

            # Bob sees only his profile.
            rb = client_b.get("/api/v1/profiles")
            assert len(rb.json()) == 1
            assert rb.json()[0]["games_played"] == 1

    def test_target_record_bound_in_cel_scope(self, owner_keyed_client):
        """The Update CEL `profile.games_played + 1` must resolve to
        the existing target's value (or the default-0 on upsert).
        Two rounds in a row should produce games_played=2 — proves
        the second iteration sees the post-upsert value, not the
        default."""
        client = owner_keyed_client
        for _ in range(2):
            r = client.post("/api/v1/rounds", json={"points": 1})
            rid = r.json()["id"]
            client.post(f"/_transition/rounds/status/{rid}/done")
        r = client.get("/api/v1/profiles")
        assert r.json()[0]["games_played"] == 2
