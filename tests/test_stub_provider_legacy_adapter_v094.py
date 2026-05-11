# Copyright 2026 Jamie-Leigh Blake and Termin project contributors
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Server #2: stub providers expose a `.legacy` adapter.

The compute runner routes ai-agent and LLM computes through
``provider.legacy`` for SDK-shaped calls. The stub providers
implement the modern Protocol (``invoke`` / ``invoke_streaming`` /
``complete``) but had no ``.legacy`` adapter, so stub-bound
ai-agent computes crashed with ``AttributeError`` after passing
the configuration gate.

This is the shim that makes the runtime's ``.legacy`` calls
translate to the stub's scripted-script behavior. Per the codebase
itself (compute_runner.py:425), ``.legacy`` is slated for removal
in slice (c) of the v0.10 cleanup — the shim is band-aid over
architectural drift the codebase already plans to fix.

Surface area the shim must support (per compute_runner.py call
sites):

Agent provider:
  - ``legacy.agent_loop_with_conversation(system_prompt, messages,
    tools, execute_tool, on_writeback, on_text_delta=None,
    on_text_end=None, should_halt=None, on_event=None,
    max_turns=20) -> dict`` — conv-mode entry point. Returns
    ``{"thinking": str, "summary": str}``. Fires
    ``on_writeback(kind="tool_call", body=..., tool_call_id=...,
    tool_name=..., tool_args=...)`` then
    ``on_writeback(kind="tool_result", body=..., tool_call_id=...,
    is_error=False)`` per scripted call, then
    ``on_writeback(kind="agent", body=...)`` for the final text
    (when ``script["final_result"]`` is a string or dict with
    ``"body"`` key).

  - ``legacy.agent_loop_streaming(system_prompt, user_message,
    tools, execute_tool, on_event) -> dict`` — non-conv +
    event_bus entry point. Fires
    ``on_event({"type": "done", "output": <dict>})`` at the end.

  - ``legacy.agent_loop(system_prompt, user_message, tools,
    execute_tool) -> dict`` — fallback entry point.

LLM provider:
  - ``legacy.complete(system_prompt, user_message, output_tool)
    -> dict`` — returns the matched response's output_value (a
    dict) merged with thinking. The stub's
    ``self._responses[k]["output_value"]`` IS the result dict.
"""

from __future__ import annotations

import pytest


# ── Fixtures ──


def make_agent_stub(scripts=None, default_script=None):
    from termin_server.providers.builtins.compute_agent_stub import (
        StubAgentProvider,
    )
    config = {}
    if scripts is not None:
        config["scripts"] = scripts
    if default_script is not None:
        config["default_script"] = default_script
    return StubAgentProvider(config=config or None)


def make_llm_stub(responses=None, default_response=None):
    from termin_server.providers.builtins.compute_llm_stub import (
        StubLlmProvider,
    )
    config = {}
    if responses is not None:
        config["responses"] = responses
    if default_response is not None:
        config["default_response"] = default_response
    return StubLlmProvider(config=config or None)


# ── Agent provider .legacy surface ──


class TestStubAgentLegacyAttribute:
    def test_provider_has_legacy_attribute(self):
        provider = make_agent_stub()
        assert hasattr(provider, "legacy"), \
            "StubAgentProvider must expose .legacy"

    def test_legacy_has_agent_loop(self):
        provider = make_agent_stub()
        assert hasattr(provider.legacy, "agent_loop"), \
            ".legacy must expose agent_loop"

    def test_legacy_has_agent_loop_streaming(self):
        provider = make_agent_stub()
        assert hasattr(provider.legacy, "agent_loop_streaming"), \
            ".legacy must expose agent_loop_streaming"

    def test_legacy_has_agent_loop_with_conversation(self):
        provider = make_agent_stub()
        assert hasattr(provider.legacy, "agent_loop_with_conversation"), \
            ".legacy must expose agent_loop_with_conversation"


# ── agent_loop (non-streaming fallback) ──


class TestStubAgentLegacyAgentLoop:
    @pytest.mark.asyncio
    async def test_returns_dict_with_thinking_and_summary(self):
        provider = make_agent_stub(default_script={
            "tool_calls": [],
            "final_outcome": "success",
        })

        async def execute_tool(name, args):
            return {}

        result = await provider.legacy.agent_loop(
            "system", "user", [], execute_tool,
        )
        assert isinstance(result, dict)
        assert "thinking" in result
        assert "summary" in result

    @pytest.mark.asyncio
    async def test_executes_scripted_tool_calls(self):
        provider = make_agent_stub(default_script={
            "tool_calls": [
                {"tool": "content.query", "args": {"x": 1}},
                {"tool": "content.create", "args": {"y": 2}},
            ],
            "final_outcome": "success",
        })
        called: list[tuple[str, dict]] = []

        async def execute_tool(name, args):
            called.append((name, dict(args)))
            return {"ok": True}

        await provider.legacy.agent_loop(
            "system", "user", [], execute_tool,
        )
        assert called == [
            ("content.query", {"x": 1}),
            ("content.create", {"y": 2}),
        ]

    @pytest.mark.asyncio
    async def test_matches_scripts_against_prompt(self):
        """The stub's _match() pairs script key against prompt
        substring. Verify the legacy path uses the same matcher."""
        provider = make_agent_stub(scripts={
            "ALPHA_OBJECTIVE": {
                "tool_calls": [{"tool": "alpha_tool", "args": {}}],
                "final_outcome": "success",
            },
            "BETA_OBJECTIVE": {
                "tool_calls": [{"tool": "beta_tool", "args": {}}],
                "final_outcome": "success",
            },
        })
        called: list[str] = []

        async def execute_tool(name, args):
            called.append(name)
            return None

        await provider.legacy.agent_loop(
            "system", "ALPHA_OBJECTIVE here", [], execute_tool,
        )
        assert called == ["alpha_tool"]


# ── agent_loop_streaming (event_bus + non-conv) ──


class TestStubAgentLegacyAgentLoopStreaming:
    @pytest.mark.asyncio
    async def test_fires_done_event_at_end(self):
        provider = make_agent_stub(default_script={
            "tool_calls": [],
            "final_outcome": "success",
            "final_result": {"summary": "all done"},
        })
        events: list[dict] = []

        async def on_event(event):
            events.append(event)

        async def execute_tool(name, args):
            return None

        await provider.legacy.agent_loop_streaming(
            "system", "user", [], execute_tool, on_event,
        )
        assert any(e.get("type") == "done" for e in events), \
            f"Expected a done event, got: {events}"

    @pytest.mark.asyncio
    async def test_executes_tools_in_order(self):
        provider = make_agent_stub(default_script={
            "tool_calls": [
                {"tool": "tool_a", "args": {}},
                {"tool": "tool_b", "args": {}},
            ],
            "final_outcome": "success",
        })
        called: list[str] = []

        async def execute_tool(name, args):
            called.append(name)
            return None

        async def on_event(event):
            pass

        await provider.legacy.agent_loop_streaming(
            "system", "user", [], execute_tool, on_event,
        )
        assert called == ["tool_a", "tool_b"]


# ── agent_loop_with_conversation (conv-mode entry point) ──


class TestStubAgentLegacyAgentLoopWithConversation:
    @pytest.mark.asyncio
    async def test_fires_on_writeback_for_each_tool_call(self):
        """Conversation mode: each scripted tool_call should
        produce a tool_call entry followed by a tool_result
        entry. This is the auto-write-back contract from v0.9.2
        §11.5 — both entries land on the conversation field."""
        provider = make_agent_stub(default_script={
            "tool_calls": [
                {"tool": "content.query", "args": {"q": "x"}},
            ],
            "final_outcome": "success",
        })
        writebacks: list[dict] = []

        async def on_writeback(**kwargs):
            writebacks.append(dict(kwargs))

        async def execute_tool(name, args):
            return {"result": "ok"}

        await provider.legacy.agent_loop_with_conversation(
            system_prompt="sys",
            messages=[{"role": "user", "content": "hi"}],
            tools=[],
            execute_tool=execute_tool,
            on_writeback=on_writeback,
        )
        kinds = [w["kind"] for w in writebacks]
        assert "tool_call" in kinds
        assert "tool_result" in kinds

    @pytest.mark.asyncio
    async def test_tool_call_writeback_includes_metadata(self):
        provider = make_agent_stub(default_script={
            "tool_calls": [
                {"tool": "content.query", "args": {"q": "x"}},
            ],
            "final_outcome": "success",
        })
        writebacks: list[dict] = []

        async def on_writeback(**kwargs):
            writebacks.append(dict(kwargs))

        async def execute_tool(name, args):
            return None

        await provider.legacy.agent_loop_with_conversation(
            system_prompt="sys",
            messages=[{"role": "user", "content": "hi"}],
            tools=[],
            execute_tool=execute_tool,
            on_writeback=on_writeback,
        )
        tool_call_wb = next(
            w for w in writebacks if w["kind"] == "tool_call"
        )
        assert tool_call_wb["tool_name"] == "content.query"
        assert tool_call_wb["tool_args"] == {"q": "x"}
        assert "tool_call_id" in tool_call_wb

    @pytest.mark.asyncio
    async def test_tool_result_pairs_with_tool_call_id(self):
        provider = make_agent_stub(default_script={
            "tool_calls": [
                {"tool": "tool_a", "args": {}},
            ],
            "final_outcome": "success",
        })
        writebacks: list[dict] = []

        async def on_writeback(**kwargs):
            writebacks.append(dict(kwargs))

        async def execute_tool(name, args):
            return {"result": "ok"}

        await provider.legacy.agent_loop_with_conversation(
            system_prompt="sys",
            messages=[{"role": "user", "content": "hi"}],
            tools=[],
            execute_tool=execute_tool,
            on_writeback=on_writeback,
        )
        tool_call_wb = next(
            w for w in writebacks if w["kind"] == "tool_call"
        )
        tool_result_wb = next(
            w for w in writebacks if w["kind"] == "tool_result"
        )
        assert tool_call_wb["tool_call_id"] == tool_result_wb["tool_call_id"]

    @pytest.mark.asyncio
    async def test_fires_agent_writeback_for_final_text(self):
        """When the script declares a final body (the agent's
        end-of-turn text), the adapter fires
        on_writeback(kind="agent", body=<text>)."""
        provider = make_agent_stub(default_script={
            "tool_calls": [],
            "final_outcome": "success",
            "final_result": {"body": "All done, JL."},
        })
        writebacks: list[dict] = []

        async def on_writeback(**kwargs):
            writebacks.append(dict(kwargs))

        async def execute_tool(name, args):
            return None

        await provider.legacy.agent_loop_with_conversation(
            system_prompt="sys",
            messages=[{"role": "user", "content": "hi"}],
            tools=[],
            execute_tool=execute_tool,
            on_writeback=on_writeback,
        )
        agent_wb = [w for w in writebacks if w["kind"] == "agent"]
        assert len(agent_wb) == 1
        assert agent_wb[0]["body"] == "All done, JL."

    @pytest.mark.asyncio
    async def test_fires_text_delta_and_end_when_callbacks_supplied(self):
        """When the runtime supplies on_text_delta / on_text_end
        callbacks (for token streaming UI), the adapter fires
        them around the final agent text."""
        provider = make_agent_stub(default_script={
            "tool_calls": [],
            "final_outcome": "success",
            "final_result": {"body": "Hello world"},
        })
        deltas: list[str] = []
        ends: list[bool] = []

        async def on_writeback(**kwargs):
            pass

        async def on_text_delta(text):
            deltas.append(text)

        async def on_text_end(committed):
            ends.append(committed)

        async def execute_tool(name, args):
            return None

        await provider.legacy.agent_loop_with_conversation(
            system_prompt="sys",
            messages=[{"role": "user", "content": "hi"}],
            tools=[],
            execute_tool=execute_tool,
            on_writeback=on_writeback,
            on_text_delta=on_text_delta,
            on_text_end=on_text_end,
        )
        assert deltas, "Expected at least one delta when final_result.body is set"
        assert ends == [True], \
            "Expected exactly one on_text_end(committed=True) after committed final text"

    @pytest.mark.asyncio
    async def test_should_halt_terminates_early(self):
        """If the runtime signals should_halt() before the next
        tool batch, the loop exits without firing further
        writebacks."""
        provider = make_agent_stub(default_script={
            "tool_calls": [
                {"tool": "tool_a", "args": {}},
                {"tool": "tool_b", "args": {}},
            ],
            "final_outcome": "success",
        })
        writebacks: list[dict] = []
        halt_state = {"value": False}

        async def on_writeback(**kwargs):
            writebacks.append(dict(kwargs))
            # Flip halt after first tool_call writeback
            if kwargs.get("kind") == "tool_call":
                halt_state["value"] = True

        async def execute_tool(name, args):
            return None

        def should_halt():
            return halt_state["value"]

        await provider.legacy.agent_loop_with_conversation(
            system_prompt="sys",
            messages=[{"role": "user", "content": "hi"}],
            tools=[],
            execute_tool=execute_tool,
            on_writeback=on_writeback,
            should_halt=should_halt,
        )
        # Should have written tool_call + tool_result for tool_a
        # but NOT proceeded to tool_b
        tool_names = [
            w.get("tool_name") for w in writebacks
            if w["kind"] == "tool_call"
        ]
        assert "tool_a" in tool_names
        assert "tool_b" not in tool_names, \
            "should_halt should have prevented tool_b from running"

    @pytest.mark.asyncio
    async def test_returns_thinking_summary_dict(self):
        provider = make_agent_stub(default_script={
            "tool_calls": [],
            "final_outcome": "success",
        })

        async def on_writeback(**kwargs):
            pass

        async def execute_tool(name, args):
            return None

        result = await provider.legacy.agent_loop_with_conversation(
            system_prompt="sys",
            messages=[{"role": "user", "content": "hi"}],
            tools=[],
            execute_tool=execute_tool,
            on_writeback=on_writeback,
        )
        assert isinstance(result, dict)
        assert "thinking" in result
        assert "summary" in result


# ── LLM provider .legacy surface ──


class TestStubLlmLegacyAttribute:
    def test_provider_has_legacy_attribute(self):
        provider = make_llm_stub()
        assert hasattr(provider, "legacy")

    def test_legacy_has_complete(self):
        provider = make_llm_stub()
        assert hasattr(provider.legacy, "complete")


class TestStubLlmLegacyComplete:
    @pytest.mark.asyncio
    async def test_returns_output_value_merged_with_thinking(self):
        """The runtime expects legacy.complete to return a dict
        with the output field values plus a 'thinking' key.
        The stub's responses[<k>]["output_value"] IS that dict;
        the adapter merges in thinking + any extras."""
        provider = make_llm_stub(default_response={
            "outcome": "success",
            "output_value": {"answer": "42", "confidence": "high"},
        })
        result = await provider.legacy.complete(
            "system_prompt",
            "user_message",
            {"name": "set_output", "input_schema": {}},
        )
        assert isinstance(result, dict)
        assert result.get("answer") == "42"
        assert result.get("confidence") == "high"
        assert "thinking" in result  # may be empty string

    @pytest.mark.asyncio
    async def test_matches_responses_against_prompt(self):
        provider = make_llm_stub(responses={
            "EVALUATE": {
                "outcome": "success",
                "output_value": {"score": 7},
            },
            "SUMMARIZE": {
                "outcome": "success",
                "output_value": {"score": 3},
            },
        })
        result = await provider.legacy.complete(
            "system",
            "Please EVALUATE this work",
            {"name": "set_output"},
        )
        assert result.get("score") == 7
