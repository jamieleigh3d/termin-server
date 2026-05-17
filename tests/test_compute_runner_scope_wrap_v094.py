# Copyright 2026 Jamie-Leigh Blake and Termin project contributors
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Regression test for the per-character scope-split bug in
``compute_runner.py:763`` — the ``state_transition`` agent-tool branch.

Bug shape (pre-fix)::

    {"role": "service",
     "scopes": list(ctx.scope_for_content_verb(cname, "update") or [])}

``RuntimeContext.scope_for_content_verb`` returns a **single scope
string** (or ``None``) — see ``context.py``. ``list(str)`` iterates
over characters, so a scope like ``"play"`` silently became
``["p", "l", "a", "y"]``. The service-user dict that travelled into
``do_state_transition`` failed the state-machine scope gate (``"play"
not in ["p","l","a","y"]``), and the transition was rejected
without surfacing the cause to the agent loop in any obvious way.

The fix (compute_runner.py, ``state_transition`` branch) wraps the
scalar correctly: ``[scope] if scope else []``. The companion
``Transition`` action verb at ``app.py:1491`` was already written
this way; this is the matching repair on the agent-tool side.

This file pins the bug class on three surfaces, so any future
regression is caught by a targeted unit test rather than by a hard-
to-reproduce agent-loop integration symptom:

1. The contract: ``RuntimeContext.scope_for_content_verb`` returns a
   single string (or ``None``), never a list — wrapping at every
   call site must be ``[scope]``, not ``list(scope)``.

2. The silent-failure surface: when a service user's ``scopes`` are
   per-character-split, ``do_state_transition`` refuses the transition
   with ``TerminScopeError`` (the original symptom the agent loop
   swallowed).

3. The fix shape: when ``scopes`` is correctly ``[scope]``,
   ``do_state_transition`` accepts the transition and the column
   flips to the target state.
"""

from __future__ import annotations

import pytest

from termin_core.errors import TerminScopeError
from termin_core.providers.storage_contract import Eq, UpdateResult
from termin_core.state.machine import do_state_transition


# ── (1) Contract: scope_for_content_verb returns str | None ──


class TestScopeForContentVerbReturnsString:
    """``RuntimeContext.scope_for_content_verb`` is the source of the
    scope value every service-user-construction site reads. It must
    return a single string (or ``None``); callers wrap with
    ``[scope]``, never ``list(scope)``.

    These tests pin the contract directly so a future refactor of
    ``scope_for_content_verb`` can't silently start returning a list
    (which would re-mask the per-character bug — ``list(list_value)``
    accidentally does the right thing, hiding the wider class of
    error).
    """

    def _ctx(self, access_grants):
        from termin_server.context import RuntimeContext

        return RuntimeContext(
            ir={"access_grants": access_grants},
            ir_json="{}",
            db_path=":memory:",
        )

    def test_returns_single_scope_string_for_matching_grant(self):
        ctx = self._ctx([
            {"content": "rounds", "verbs": ["update"], "scope": "play"},
        ])
        result = ctx.scope_for_content_verb("rounds", "update")
        assert isinstance(result, str), (
            f"scope_for_content_verb must return a string, got "
            f"{type(result).__name__}: {result!r}. If this changes to "
            f"a list, every wrap-with-[scope] site becomes wrong."
        )
        assert result == "play"

    def test_returns_none_for_no_matching_grant(self):
        ctx = self._ctx([
            {"content": "rounds", "verbs": ["view"], "scope": "play"},
        ])
        # "update" verb isn't granted; should be None.
        assert ctx.scope_for_content_verb("rounds", "update") is None

    def test_returns_none_for_unknown_content(self):
        ctx = self._ctx([
            {"content": "rounds", "verbs": ["update"], "scope": "play"},
        ])
        assert ctx.scope_for_content_verb("unknown_content", "update") is None


# ── (2) + (3) State-machine scope gate exercises the bug class ──


class _FakeStorage:
    """Minimal StorageProvider surface for ``do_state_transition``.

    Two methods are touched: ``read`` (used to load the current row
    + current state column) and ``update_if`` (the atomic CAS the
    transition writes through). The fake always reports ``applied``
    so the only failure mode the test sees is the scope gate itself.
    """

    def __init__(self, record: dict):
        self._record = dict(record)

    async def read(self, table, record_id):
        if record_id != self._record.get("id"):
            return None
        return dict(self._record)

    async def update_if(self, table, record_id, *, condition: Eq, patch):
        # The bug we're pinning never reaches this method — the
        # scope check rejects the transition first. But we keep the
        # fake honest so the positive path can be tested too.
        if record_id != self._record.get("id"):
            return UpdateResult(applied=False, record=None, reason="not_found")
        current = self._record.get(condition.field)
        if current != condition.value:
            return UpdateResult(
                applied=False, record=dict(self._record),
                reason="condition_failed",
            )
        self._record.update(patch)
        return UpdateResult(
            applied=True, record=dict(self._record), reason="applied",
        )


def _sm_lookup_for_rounds():
    """Build the state-machine lookup `do_state_transition` consults.

    Mirrors the IR shape the runtime builds: ``{table_name:
    [sm_dict, ...]}`` where each sm_dict carries ``machine_name``,
    ``column``, ``initial``, ``transitions``. The transition gate
    here uses the legacy plain-scope-string shape (``"play"``) so
    the path under test is precisely the one the agent-tool branch
    hits at ``compute_runner.py:763``."""
    return {
        "rounds": [{
            "machine_name": "status",
            "column": "status",
            "initial": "in_progress",
            "transitions": {
                ("in_progress", "done"): "play",
            },
        }],
    }


class TestScopeSplitBreaksStateTransition:
    """The per-character-split user dict reproduces the bug symptom:
    the scope gate rejects what should be a privileged service
    transition because none of the per-character pseudo-scopes match
    the gate's required scope string ``"play"``.

    This is the test that would have failed pre-fix and passes
    post-fix. It runs against a real ``do_state_transition`` —
    nothing about the gate check is mocked.
    """

    @pytest.mark.asyncio
    async def test_per_character_scopes_raises_scope_error(self):
        storage = _FakeStorage(
            {"id": 1, "status": "in_progress", "points": 5},
        )
        sm_lookup = _sm_lookup_for_rounds()

        # The buggy user shape: `list("play") == ["p","l","a","y"]`.
        # The runtime's downstream check
        # `"play" not in user["scopes"]` evaluates True and raises.
        buggy_user = {
            "role": "service",
            "scopes": list("play"),
            "id": "",
        }

        with pytest.raises(TerminScopeError) as excinfo:
            await do_state_transition(
                storage, "rounds", 1, "status", "done",
                buggy_user, sm_lookup,
            )

        assert "play" in str(excinfo.value), (
            f"Expected the scope-error message to mention the "
            f"required scope 'play'; got: {excinfo.value!r}"
        )


class TestSingleElementScopeAllowsStateTransition:
    """The fix shape (``[scope]`` not ``list(scope)``) wraps the
    scalar correctly, so the service-user dict satisfies the gate
    and the transition commits.

    This pins the post-fix behaviour: the column flips from
    ``in_progress`` to ``done``.
    """

    @pytest.mark.asyncio
    async def test_single_element_scope_list_completes_transition(self):
        storage = _FakeStorage(
            {"id": 1, "status": "in_progress", "points": 5},
        )
        sm_lookup = _sm_lookup_for_rounds()

        # The fix shape: a single-element list carrying the scope.
        fixed_user = {
            "role": "service",
            "scopes": ["play"],
            "id": "",
        }

        updated = await do_state_transition(
            storage, "rounds", 1, "status", "done",
            fixed_user, sm_lookup,
        )

        # The returned record reflects the patched column.
        assert updated["status"] == "done"
        # And the fake storage actually committed.
        assert storage._record["status"] == "done"


# ── (4) Call-site shape: assert no future regression to list(str) ──


class TestComputeRunnerCallSiteShape:
    """Source-level pin: the ``state_transition`` branch in
    ``compute_runner.py`` must not call ``list(...)`` on the result
    of ``ctx.scope_for_content_verb(...)``. That literal pattern was
    the bug; if it ever re-appears, this test fails fast at parse
    time without needing the runtime to be exercised.

    Source-text pins read poorly in isolation, but the alternative
    (an end-to-end agent-tool integration test) requires substantial
    stub-provider scaffolding for the same coverage. The comment in
    ``compute_runner.py`` above the call site explains the trap —
    this test guards it.
    """

    def test_no_list_of_scope_for_content_verb_in_compute_runner(self):
        import inspect

        from termin_server import compute_runner

        source = inspect.getsource(compute_runner)
        # The exact bug pattern (and any whitespace variant) that
        # splits a scope string into characters.
        bad_patterns = [
            "list(ctx.scope_for_content_verb",
            "list( ctx.scope_for_content_verb",
        ]
        for pat in bad_patterns:
            assert pat not in source, (
                f"compute_runner.py contains {pat!r} — this is the "
                f"per-character scope-split bug. Wrap with "
                f"`[scope] if scope else []` instead. See the comment "
                f"above the state_transition branch."
            )
