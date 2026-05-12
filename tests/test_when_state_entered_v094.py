# Copyright 2026 Jamie-Leigh Blake and Termin project contributors
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Runtime test for the v0.9.4 cross-content-updates slice B4 —
state-entered When-rule trigger.

The compiler-side slices (B1a/B2/B3) lower
`When <singular> <field> enters <state>:` to an EventSpec carrying
`trigger_state_field` + `trigger_state_value`. This test exercises
the runtime side: when the state machine fires its
`<plural>.<field>.<state>.entered` event, every When-rule
subscribing to that channel must run its body (here, a same-record
A3a Update — the cross-content owner-keyed Update lands in B5).

Test app shape:

    Content called "rounds":
      Each round has a points which is a whole number, defaults to 0
      Each round has a status which is state:
        status starts as in_progress
        status can also be done
        in_progress can become done if the user has "play"
      Each round has a finalized which is yes or no, defaults to "no"

    When round status enters done:
      Update rounds: finalized = `"yes"`

After POSTing a transition to `done`, the round's `finalized`
field should flip to `"yes"` — proving the When-rule fired.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

# The compiler isn't a runtime dependency, but tests in this repo
# are allowed to use it for inline .termin → IR compilation. The
# editable install setup in workspace CLAUDE.md guarantees both
# repos are importable from any session.
COMPILER_ROOT = Path(__file__).parent.parent.parent / "termin-compiler"
sys.path.insert(0, str(COMPILER_ROOT))

from termin.peg_parser import parse_peg  # noqa: E402
from termin.analyzer import analyze  # noqa: E402
from termin.lower import lower  # noqa: E402
from termin_core.ir.serialize import serialize_ir  # noqa: E402

from termin_server import create_termin_app  # noqa: E402


_SOURCE = """
Application: Test
Description: B4 runtime smoke for state-entered When-rule trigger.

Identity:
  Scopes are "play"
  A "player" has "play"

Content called "rounds":
  Each round has a player_principal which is principal, required
  Each round is owned by player_principal
  Each round has points which is a whole number, defaults to 0
  Each round has a finalized which is yes or no, defaults to "no"
  Each round has a status which is state:
    status starts as in_progress
    status can also be done
    in_progress can become done if the user has "play"
  Anyone with "play" can view rounds
  Anyone with "play" can create rounds
  Anyone with "play" can update rounds

When round status enters done:
  Update rounds: finalized = `"yes"`
"""


def _compile_to_ir_json() -> str:
    program, errors = parse_peg(_SOURCE)
    assert errors.ok, f"Parse errors: {errors.format()}"
    result = analyze(program)
    assert result.ok, f"Analyzer errors: {result.format()}"
    spec = lower(program)
    return serialize_ir(spec)


@pytest.fixture
def state_entered_client(tmp_path) -> TestClient:
    ir_json = _compile_to_ir_json()
    db_path = str(tmp_path / "state_entered.db")
    app = create_termin_app(ir_json, db_path=db_path)
    with TestClient(app) as client:
        client.cookies.set("termin_role", "player")
        client.cookies.set("termin_user_name", "alice")
        yield client


class TestStateEnteredWhenRuleRuntime:
    def test_when_rule_fires_on_state_transition(self, state_entered_client):
        """POSTing a transition to `done` should fire the When-rule
        that flips `finalized` to "yes"."""
        client = state_entered_client
        # Create a round.
        r = client.post(
            "/api/v1/rounds",
            json={"points": 5},
        )
        assert r.status_code == 201, r.text
        round_id = r.json()["id"]
        assert r.json()["status"] == "in_progress"
        assert r.json()["finalized"] == "no"

        # Transition to done.
        r = client.post(
            f"/_transition/rounds/status/{round_id}/done",
        )
        assert r.status_code == 200, r.text

        # Re-fetch — the When-rule should have flipped `finalized`
        # to "yes".
        r = client.get(f"/api/v1/rounds/{round_id}")
        assert r.status_code == 200
        record = r.json()
        assert record["status"] == "done"
        assert record["finalized"] == "yes", (
            f"Expected finalized to flip to 'yes' after transition. "
            f"Record: {record}"
        )

    def test_when_rule_does_not_fire_on_irrelevant_transitions(
        self, state_entered_client,
    ):
        """The When-rule subscribes specifically to `enters done`.
        A round in in_progress (the initial state, never transitioned
        TO) should NOT trigger the When-rule — finalized stays "no"."""
        client = state_entered_client
        r = client.post(
            "/api/v1/rounds",
            json={"points": 5},
        )
        assert r.status_code == 201
        round_id = r.json()["id"]
        # Don't transition. Just re-fetch and verify no spurious fire.
        r = client.get(f"/api/v1/rounds/{round_id}")
        assert r.json()["finalized"] == "no"
