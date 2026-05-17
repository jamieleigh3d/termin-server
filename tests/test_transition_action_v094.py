# Copyright 2026 Jamie-Leigh Blake and Termin project contributors
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Runtime test for v0.9.4 Phase 3 C1 — Transition action verb.

When a state-entered When-rule's body contains a Transition action,
the runtime fires the named state-machine transition through the
same do_state_transition path the HTTP /_transition/<plural>/
<field>/{id}/<target> route uses. The cascade is intentional:
chained transitions are the C2 airlock use case (hatch_state
enters unlocked → lifecycle transitions scenario → scoring →
scores_state enters ready → lifecycle transitions to complete).

Stale-fire suppression: if the target state is unreachable from
the current state (or scope is missing), the transition fails
silently per the dispatcher contract.

Test app shape:

    Content called "rounds":
      Each round has a status which is state:
        status starts as in_progress
        status can also be done
        in_progress can become done if the user has "play"
      Each round has a archive_status which is state:
        archive_status starts as live
        archive_status can also be archived
        live can become archived if the user has "play"

    When round status enters done:
      Transition rounds archive_status to archived

POSTing a transition to status=done should fire the When-rule,
which transitions archive_status to archived.
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
Application: TransitionActionRuntime
Description: C1 runtime smoke for Transition action verb.

Identity:
  Scopes are "play"
  A "player" has "play"

Content called "rounds":
  Each round has a player_principal which is principal, required
  Each round is owned by player_principal
  Each round has a points which is a whole number, defaults to 0
  Each round has a status which is state:
    status starts as in_progress
    status can also be done
    in_progress can become done if the user has "play"
  Each round has a archive_status which is state:
    archive_status starts as live
    archive_status can also be archived
    live can become archived if the user has "play"
  Anyone with "play" can view rounds
  Anyone with "play" can create rounds
  Anyone with "play" can update rounds

When round status enters done:
  Transition rounds archive_status to archived
"""


def _compile_to_ir_json() -> str:
    program, errors = parse_peg(_SOURCE)
    assert errors.ok, f"Parse errors: {errors.format()}"
    result = analyze(program)
    assert result.ok, f"Analyzer errors: {result.format()}"
    spec = lower(program)
    return serialize_ir(spec)


@pytest.fixture
def transition_client(tmp_path) -> TestClient:
    ir_json = _compile_to_ir_json()
    db_path = str(tmp_path / "transition.db")
    app = create_termin_app(ir_json, db_path=db_path)
    with TestClient(app) as client:
        client.cookies.set("termin_role", "player")
        client.cookies.set("termin_user_name", "alice")
        yield client


class TestTransitionActionRuntime:
    def test_when_rule_transition_action_fires_cascading_transition(
        self, transition_client,
    ):
        """POSTing status → done should fire the When-rule whose body
        is `Transition rounds archive_status to archived`. After the
        cascade, archive_status should be 'archived' on the round."""
        client = transition_client
        r = client.post(
            "/api/v1/rounds",
            json={"points": 5},
        )
        assert r.status_code == 201, r.text
        round_id = r.json()["id"]
        # Initial state: status=in_progress, archive_status=live.
        assert r.json()["status"] == "in_progress"
        assert r.json()["archive_status"] == "live"

        # Transition status -> done. This should trigger the
        # When-rule, which transitions archive_status -> archived.
        r = client.post(
            f"/_transition/rounds/status/{round_id}/done",
        )
        assert r.status_code == 200, r.text

        # Verify both state writes happened.
        r = client.get(f"/api/v1/rounds/{round_id}")
        assert r.status_code == 200
        record = r.json()
        assert record["status"] == "done"
        assert record["archive_status"] == "archived", (
            f"Expected archive_status to be 'archived' after "
            f"cascading transition. Record: {record}"
        )

    def test_transition_does_not_fire_on_irrelevant_state(
        self, transition_client,
    ):
        """The When-rule subscribes to `enters done`. Creating a
        round (initial state in_progress) must not fire the rule —
        archive_status stays 'live'."""
        client = transition_client
        r = client.post(
            "/api/v1/rounds",
            json={"points": 5},
        )
        assert r.status_code == 201
        round_id = r.json()["id"]
        # Don't transition. Verify no spurious cascade.
        r = client.get(f"/api/v1/rounds/{round_id}")
        assert r.json()["archive_status"] == "live"
