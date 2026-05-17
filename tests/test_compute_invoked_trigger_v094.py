# Copyright 2026 Jamie-Leigh Blake and Termin project contributors
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Runtime tests for v0.9.4 Phase 3 C3d — compute-invoked trigger.

The compiler-side slices C3a/b/c (grammar / analyzer / lower) emit
EventSpecs carrying ``trigger="compute-invoked"`` +
``trigger_compute=<name>`` + optional ``trigger_compute_filter``
(CEL). This slice exercises the runtime side: after a compute
completes successfully, the runtime emits a synthetic
``<compute>.invoked`` event and a dispatcher executes every
When-rule whose ``trigger_compute`` matches the compute name —
applying the same Update / Append / Transition body the
state-entered handler already runs (B4 + C1).

Failure isolation: a per-rule failure must NOT fail the compute
itself (the compute already returned its result to its caller —
ARIA, an HTTP trigger, an event subscriber — and they should not
be told the compute failed because a downstream rule's CEL was
broken). Log + skip per the state-entered precedent.

Test app shape (matches the C3 design doc §5.6 conformance
fixture)::

    Content called "rounds":
      Each round has a player_principal which is principal, required
      Each round is owned by player_principal
      Each round has triggered_flag which is yes or no, defaults "no"
      Each round has filtered_flag which is yes or no, defaults "no"
      Anyone with "play" can create/view/update rounds

    Compute called "test_tool":
      Transform: takes a round, produces a round
      `{"ok": true, "marker": args.marker}`
      Anyone with "play" can execute this

    When test_tool called:                                  # unfiltered
      Update rounds: triggered_flag = `"yes"`

    When test_tool called with `args.marker == "filter_me"`:  # filtered
      Update rounds: filtered_flag = `"yes"`

Three behavioural surfaces under test:

1. **Unfiltered fire**: invoking ``test_tool`` with any marker flips
   ``triggered_flag`` to ``"yes"``.
2. **Filter match**: invoking with ``marker="filter_me"`` flips
   BOTH flags.
3. **Filter miss**: invoking with ``marker="other"`` flips only
   ``triggered_flag``.
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
Application: ComputeInvokedRuntimeTest
Description: C3d runtime smoke for compute-invoked event trigger.

Identity:
  Scopes are "play"
  A "player" has "play"

Content called "rounds":
  Each round has a player_principal which is principal, required
  Each round is owned by player_principal
  Each round has triggered_flag which is yes or no, defaults to "no"
  Each round has filtered_flag which is yes or no, defaults to "no"
  Anyone with "play" can view rounds
  Anyone with "play" can create rounds
  Anyone with "play" can update rounds

Compute called "test_tool":
  Transform: takes a round, produces a round
  `{"ok": true, "marker": args.marker}`
  Anyone with "play" can execute this

When test_tool called:
  Update rounds: triggered_flag = `"yes"`

When test_tool called with `args.marker == "filter_me"`:
  Update rounds: filtered_flag = `"yes"`
"""


def _compile_to_ir_json() -> str:
    program, errors = parse_peg(_SOURCE)
    assert errors.ok, f"Parse errors: {errors.format()}"
    result = analyze(program)
    assert result.ok, f"Analyzer errors: {result.format()}"
    spec = lower(program)
    return serialize_ir(spec)


@pytest.fixture
def compute_invoked_client(tmp_path) -> TestClient:
    ir_json = _compile_to_ir_json()
    db_path = str(tmp_path / "compute_invoked.db")
    app = create_termin_app(ir_json, db_path=db_path)
    with TestClient(app) as client:
        client.cookies.set("termin_role", "player")
        client.cookies.set("termin_user_name", "alice")
        yield client


def _create_round(client: TestClient) -> dict:
    """Create a round and return the freshly-created record dict."""
    r = client.post("/api/v1/rounds", json={})
    assert r.status_code == 201, r.text
    rec = r.json()
    # Sanity: both flags start "no".
    assert rec["triggered_flag"] == "no", rec
    assert rec["filtered_flag"] == "no", rec
    return rec


def _trigger_compute(client: TestClient, record: dict, marker: str):
    """Hit the manual trigger endpoint with `marker` riding on the
    record dict so `args.marker` resolves in the When-rule filter."""
    payload = {
        "record": dict(record, marker=marker),
        "content_name": "rounds",
    }
    r = client.post("/api/v1/compute/test_tool/trigger", json=payload)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body.get("status") == "completed", body
    return body


def _fetch_round(client: TestClient, round_id: int) -> dict:
    r = client.get(f"/api/v1/rounds/{round_id}")
    assert r.status_code == 200, r.text
    return r.json()


# ── Behavioural tests ─────────────────────────────────────────────


class TestComputeInvokedUnfilteredRule:
    """The unfiltered ``When test_tool called`` rule should fire on
    every successful invocation — the filter is empty, so the
    dispatcher takes the take-everything path."""

    def test_fires_with_arbitrary_marker(self, compute_invoked_client):
        client = compute_invoked_client
        rec = _create_round(client)
        _trigger_compute(client, rec, marker="anything")
        post = _fetch_round(client, rec["id"])
        assert post["triggered_flag"] == "yes", (
            f"Unfiltered When-rule did not fire — triggered_flag should "
            f"be 'yes' after invocation. Round: {post}"
        )


class TestComputeInvokedFilteredRule:
    """The filtered ``When test_tool called with `args.marker == 'filter_me'`:``
    rule fires only when the CEL filter evaluates True against the
    event context. The event context binds ``args`` to the input
    record so ``args.marker`` resolves to the caller-supplied
    marker value."""

    def test_fires_when_filter_matches(self, compute_invoked_client):
        client = compute_invoked_client
        rec = _create_round(client)
        _trigger_compute(client, rec, marker="filter_me")
        post = _fetch_round(client, rec["id"])
        # Both rules should have fired.
        assert post["triggered_flag"] == "yes", post
        assert post["filtered_flag"] == "yes", (
            f"Filtered When-rule did not fire on a matching marker — "
            f"filtered_flag should be 'yes'. Round: {post}"
        )

    def test_does_not_fire_when_filter_misses(self, compute_invoked_client):
        client = compute_invoked_client
        rec = _create_round(client)
        _trigger_compute(client, rec, marker="something_else")
        post = _fetch_round(client, rec["id"])
        # Unfiltered rule still fires.
        assert post["triggered_flag"] == "yes", post
        # Filtered rule should NOT have fired — marker didn't match.
        assert post["filtered_flag"] == "no", (
            f"Filtered When-rule fired on a non-matching marker — "
            f"filtered_flag should still be 'no'. Round: {post}"
        )
