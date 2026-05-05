# Copyright 2026 Jamie-Leigh Blake and Termin project contributors
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""v0.9.2 L7.1+L7.2+L7.3 — conversation materialization + auto-write-back.

Two surfaces under test:

1. ``materialize_to_anthropic(entries)`` — the §11.4 canonical kind →
   Anthropic-shape mapping. Pure function; unit-testable per kind,
   per merging rule, per attachment shape.

2. The conversation-aware dispatch in
   ``_execute_agent_compute``: when ``Conversation is X.Y`` is set on
   the compute, the runtime materializes the field, calls a
   conversation-aware agent loop on the provider, and auto-appends
   each tool_call / tool_result / final assistant text back to the
   field with parent_id linkage to the triggering user entry. Per
   tech-design §11.5.
"""

from __future__ import annotations

import json
import time

import pytest


# ── L7.2: materialize_to_anthropic — pure mapping helper ──


class TestMaterializeUserKind:
    def test_user_text_only(self):
        from termin_server.ai_provider import materialize_to_anthropic
        entries = [
            {"kind": "user", "body": "hello", "id": "e-1"},
        ]
        msgs = materialize_to_anthropic(entries)
        assert msgs == [
            {"role": "user", "content": [{"type": "text", "text": "hello"}]},
        ]

    def test_user_with_image_attachment(self):
        from termin_server.ai_provider import materialize_to_anthropic
        entries = [
            {
                "kind": "user", "body": "what's in this?", "id": "e-1",
                "attachments": [
                    {
                        "media_type": "image/png",
                        "source": {
                            "type": "base64", "media_type": "image/png",
                            "data": "iVBORAA==",
                        },
                    },
                ],
            },
        ]
        msgs = materialize_to_anthropic(entries)
        assert len(msgs) == 1
        assert msgs[0]["role"] == "user"
        assert msgs[0]["content"][0] == {
            "type": "text", "text": "what's in this?",
        }
        # The image rides alongside the text in the same content array.
        assert msgs[0]["content"][1]["type"] == "image"
        assert msgs[0]["content"][1]["source"]["media_type"] == "image/png"

    def test_user_with_pdf_attachment(self):
        from termin_server.ai_provider import materialize_to_anthropic
        entries = [
            {
                "kind": "user", "body": "summarize", "id": "e-1",
                "attachments": [
                    {
                        "media_type": "application/pdf",
                        "source": {
                            "type": "base64",
                            "media_type": "application/pdf",
                            "data": "JVBERi0=",
                        },
                    },
                ],
            },
        ]
        msgs = materialize_to_anthropic(entries)
        assert msgs[0]["content"][1]["type"] == "document"


class TestMaterializeAssistantKind:
    def test_assistant_text(self):
        from termin_server.ai_provider import materialize_to_anthropic
        entries = [
            {"kind": "user", "body": "hi", "id": "e-1"},
            {"kind": "assistant", "body": "hello back", "id": "e-2"},
        ]
        msgs = materialize_to_anthropic(entries)
        assert msgs[1] == {
            "role": "assistant",
            "content": [{"type": "text", "text": "hello back"}],
        }

    def test_assistant_refusal_type_maps_identically(self):
        """Per §11.4: `type` is a Termin discriminator for audit and
        chat rendering only — refusal-type entries are sent to
        Anthropic identically to response-type assistant entries."""
        from termin_server.ai_provider import materialize_to_anthropic
        entries = [
            {"kind": "user", "body": "do harm", "id": "e-1"},
            {
                "kind": "assistant", "body": "I can't do that",
                "type": "refusal", "id": "e-2",
            },
        ]
        msgs = materialize_to_anthropic(entries)
        # Same shape as a response-type assistant entry.
        assert msgs[1] == {
            "role": "assistant",
            "content": [{"type": "text", "text": "I can't do that"}],
        }


class TestMaterializeToolCallKind:
    def test_tool_call_becomes_assistant_tool_use_block(self):
        from termin_server.ai_provider import materialize_to_anthropic
        entries = [
            {"kind": "user", "body": "what time is it?", "id": "e-1"},
            {
                "kind": "tool_call", "body": "current_time({})",
                "tool_call_id": "toolu_01", "tool_name": "current_time",
                "tool_args": {}, "id": "e-2",
            },
        ]
        msgs = materialize_to_anthropic(entries)
        assert msgs[1]["role"] == "assistant"
        block = msgs[1]["content"][0]
        assert block["type"] == "tool_use"
        assert block["id"] == "toolu_01"
        assert block["name"] == "current_time"
        assert block["input"] == {}


class TestMaterializeToolResultKind:
    def test_tool_result_becomes_user_tool_result_block(self):
        from termin_server.ai_provider import materialize_to_anthropic
        entries = [
            {"kind": "user", "body": "what time?", "id": "e-1"},
            {
                "kind": "tool_call", "body": "current_time({})",
                "tool_call_id": "toolu_01", "tool_name": "current_time",
                "tool_args": {}, "id": "e-2",
            },
            {
                "kind": "tool_result", "body": "2026-05-04T10:00:00Z",
                "tool_call_id": "toolu_01", "id": "e-3",
            },
        ]
        msgs = materialize_to_anthropic(entries)
        # The tool_result lives in a user-role message (per Anthropic).
        result_msg = msgs[2]
        assert result_msg["role"] == "user"
        block = result_msg["content"][0]
        assert block["type"] == "tool_result"
        assert block["tool_use_id"] == "toolu_01"
        assert block["content"] == "2026-05-04T10:00:00Z"
        assert "is_error" not in block

    def test_tool_result_with_error_marks_is_error(self):
        from termin_server.ai_provider import materialize_to_anthropic
        entries = [
            {"kind": "user", "body": "x", "id": "e-1"},
            {
                "kind": "tool_call", "body": "broken({})",
                "tool_call_id": "toolu_99", "tool_name": "broken",
                "tool_args": {}, "id": "e-2",
            },
            {
                "kind": "tool_result", "body": "Error: kaboom",
                "tool_call_id": "toolu_99", "is_error": True, "id": "e-3",
            },
        ]
        msgs = materialize_to_anthropic(entries)
        block = msgs[2]["content"][0]
        assert block.get("is_error") is True


class TestMaterializeSystemEventKind:
    def test_system_event_wrapped_in_source_prefix_on_user_role(self):
        """Per §11.4: system_event maps to user role with body wrapped
        as `[<source>] <body>` so the in-band context is distinguishable
        from real user input."""
        from termin_server.ai_provider import materialize_to_anthropic
        entries = [
            {"kind": "user", "body": "hi", "id": "e-1"},
            {
                "kind": "assistant", "body": "hello", "id": "e-2",
            },
            {
                "kind": "system_event",
                "body": "User has been idle for 5 minutes",
                "source": "OVERSEER", "id": "e-3",
            },
        ]
        msgs = materialize_to_anthropic(entries)
        assert msgs[2]["role"] == "user"
        text = msgs[2]["content"][0]["text"]
        assert text.startswith("[OVERSEER]")
        assert "idle for 5 minutes" in text


class TestMaterializeAdjacentRoleMerging:
    """Anthropic requires alternating user/assistant. Adjacent same-role
    entries must merge into one message with multiple content blocks."""

    def test_two_user_entries_merge_into_one_message(self):
        from termin_server.ai_provider import materialize_to_anthropic
        entries = [
            {"kind": "user", "body": "first", "id": "e-1"},
            {"kind": "user", "body": "second", "id": "e-2"},
        ]
        msgs = materialize_to_anthropic(entries)
        assert len(msgs) == 1
        assert msgs[0]["role"] == "user"
        assert len(msgs[0]["content"]) == 2
        assert msgs[0]["content"][0]["text"] == "first"
        assert msgs[0]["content"][1]["text"] == "second"

    def test_assistant_text_then_tool_call_merge(self):
        """A turn that thinks-then-calls produces both an assistant
        text block and an assistant tool_use block in one message."""
        from termin_server.ai_provider import materialize_to_anthropic
        entries = [
            {"kind": "user", "body": "what time?", "id": "e-1"},
            {
                "kind": "assistant",
                "body": "let me check", "id": "e-2",
            },
            {
                "kind": "tool_call", "body": "current_time({})",
                "tool_call_id": "toolu_01", "tool_name": "current_time",
                "tool_args": {}, "id": "e-3",
            },
        ]
        msgs = materialize_to_anthropic(entries)
        assert len(msgs) == 2
        assert msgs[1]["role"] == "assistant"
        assert len(msgs[1]["content"]) == 2
        types = [b["type"] for b in msgs[1]["content"]]
        assert types == ["text", "tool_use"]

    def test_system_event_after_user_merges_user_role(self):
        """system_event maps to user role; adjacent to a real user
        entry it should merge into one user message with two text
        blocks."""
        from termin_server.ai_provider import materialize_to_anthropic
        entries = [
            {"kind": "user", "body": "hi", "id": "e-1"},
            {
                "kind": "system_event", "body": "user idle",
                "source": "OVERSEER", "id": "e-2",
            },
        ]
        msgs = materialize_to_anthropic(entries)
        assert len(msgs) == 1
        assert msgs[0]["role"] == "user"
        assert len(msgs[0]["content"]) == 2


class TestMaterializeToolLinkageValidation:
    """Tech-design §11.4: every tool_result must have a tool_call_id
    matching a preceding tool_call. Orphan tool_results are rejected."""

    def test_orphan_tool_result_raises(self):
        from termin_server.ai_provider import (
            materialize_to_anthropic, ConversationMaterializationError,
        )
        entries = [
            {"kind": "user", "body": "x", "id": "e-1"},
            {
                "kind": "tool_result", "body": "irrelevant",
                "tool_call_id": "ghost", "id": "e-2",
            },
        ]
        with pytest.raises(ConversationMaterializationError):
            materialize_to_anthropic(entries)

    def test_tool_result_with_matching_prior_call_passes(self):
        from termin_server.ai_provider import materialize_to_anthropic
        entries = [
            {"kind": "user", "body": "x", "id": "e-1"},
            {
                "kind": "tool_call", "body": "ok",
                "tool_call_id": "toolu_1", "tool_name": "x",
                "tool_args": {}, "id": "e-2",
            },
            {
                "kind": "tool_result", "body": "ok",
                "tool_call_id": "toolu_1", "id": "e-3",
            },
        ]
        msgs = materialize_to_anthropic(entries)
        assert len(msgs) == 3


class TestMaterializeEmpty:
    def test_empty_entries_returns_empty_list(self):
        from termin_server.ai_provider import materialize_to_anthropic
        assert materialize_to_anthropic([]) == []


# ── L7.1+L7.3: integration — conversation flow with auto-write-back ──


def _compile_chat_app(tmp_path, with_tool: bool = False):
    """Compile a small conversational ai-agent .termin program suitable
    for stub-provider integration testing. Returns (app, ir_json).

    ``with_tool`` is retained as a hook for future variants that need
    a real Compute behind an Invokes declaration; v0.9.2 stubs route
    the synthetic tool name through ``_execute_tool``'s unknown-tool
    fallback (returns ``{"error": "Unknown tool: ..."}``), which is
    sufficient for the write-back assertions."""
    from termin import peg_parser, analyzer, lower
    from termin_core.ir.serialize import serialize_ir
    from termin_server import create_termin_app

    source = '''Application: L7 Conv Test
  Description: v0.9.2 L7.1+L7.3 fixture
Id: 6e7c1b2e-8f4a-4b1c-9d8e-2f5a3b7c8d94

Identity:
  Scopes are "chat.use"
  An "anonymous" has "chat.use"

Content called "chat_threads":
  Each chat_thread has a title which is text, default "Conversation"
  Each chat_thread has a conversation which is conversation
  Anyone with "chat.use" can view chat_threads
  Anyone with "chat.use" can create chat_threads
  Anyone with "chat.use" can append to chat_threads.conversation

Compute called "reply":
  Provider is "ai-agent"
  Trigger on event "chat_threads.conversation.appended" where `appended_entry.kind == "user"`
  Conversation is chat_threads.conversation
  Anyone with "chat.use" can execute this
  Audit level: actions
  Anyone with "chat.use" can audit
  Directive is ```
    You are a helpful assistant.
  ```

As anonymous, I want to chat so that I can converse:
  Show a page called "Chat"
'''
    program, perr = peg_parser.parse_peg(source)
    assert perr.ok, perr.format()
    aerr = analyzer.analyze(program)
    assert aerr.ok, aerr.format()
    spec = lower.lower(program)
    ir_json = serialize_ir(spec)
    db_path = str(tmp_path / "l7_conv.db")
    app = create_termin_app(ir_json, db_path=db_path)
    return app, ir_json


class _TextOnlyStubLegacy:
    """Stub legacy AIProvider whose conversation-aware loop returns a
    single assistant text response — the simplest case."""
    def __init__(self, response_text: str = "the time is 10am"):
        self._text = response_text
        self.last_messages = None
        self.last_directive = None

    async def agent_loop_with_conversation(
        self, directive, messages, tools, execute_tool,
        on_writeback, on_text_delta=None, on_text_end=None,
        should_halt=None, on_event=None, max_turns=20,
    ):
        self.last_directive = directive
        self.last_messages = messages
        # Single turn: just emit a final assistant text via writeback.
        await on_writeback(
            kind="assistant", body=self._text,
        )
        return {"thinking": self._text, "summary": "ok"}


class _ToolUsingStubLegacy:
    """Stub that simulates a tool-call turn followed by a final
    assistant text. Used to exercise tool_call/tool_result write-back
    + parent_id linkage."""
    def __init__(self):
        self.last_messages = None

    async def agent_loop_with_conversation(
        self, directive, messages, tools, execute_tool,
        on_writeback, on_text_delta=None, on_text_end=None,
        should_halt=None, on_event=None, max_turns=20,
    ):
        self.last_messages = messages
        # Turn 1: write a tool_call entry, execute the tool, write a
        # tool_result entry.
        tool_args = {}
        await on_writeback(
            kind="tool_call",
            body=f'current_time({json.dumps(tool_args)})',
            tool_call_id="toolu_stub_1",
            tool_name="current_time",
            tool_args=tool_args,
        )
        tool_out = await execute_tool("current_time", tool_args)
        await on_writeback(
            kind="tool_result",
            body=json.dumps(tool_out),
            tool_call_id="toolu_stub_1",
        )
        # Turn 2: final assistant text.
        await on_writeback(
            kind="assistant",
            body="The current time is 2026-05-04T10:00:00Z.",
        )
        return {"thinking": "computed", "summary": "ok"}


class _StubProvider:
    is_configured = True
    service = "stub"
    model = "stub-1"
    _config_hash = "sha256:stub"

    def __init__(self, legacy):
        self.legacy = legacy


class TestConversationMaterializationDispatch:
    """When `Conversation is X.Y` is set, the runtime loads the field
    contents, materializes them via materialize_to_anthropic, and
    passes the messages array to agent_loop_with_conversation."""

    def test_assistant_text_appended_with_parent_id(self, tmp_path):
        from fastapi.testclient import TestClient
        app, _ir = _compile_chat_app(tmp_path)

        stub_legacy = _TextOnlyStubLegacy(response_text="hi back")
        with TestClient(app) as client:
            ctx = getattr(client.app.state, "ctx", None) or getattr(
                client.app, "_termin_ctx", None,
            )
            assert ctx is not None
            ctx.compute_providers = {"reply": _StubProvider(stub_legacy)}

            create = client.post(
                "/api/v1/chat_threads", json={"title": "L7 test"})
            assert create.status_code in (200, 201), create.text
            thread_id = create.json()["id"]

            ap = client.post(
                f"/api/v1/chat_threads/{thread_id}/conversation:append",
                json={"kind": "user", "body": "hi"})
            assert ap.status_code == 201, ap.text

            # Background thread runs the compute. Wait briefly.
            time.sleep(0.5)

            get = client.get(f"/api/v1/chat_threads/{thread_id}")
            assert get.status_code == 200
            raw = get.json().get("conversation")
            entries = json.loads(raw) if isinstance(raw, str) else raw
            assert isinstance(entries, list)
            assert len(entries) == 2, entries
            user_entry = entries[0]
            assistant_entry = entries[1]
            assert user_entry["kind"] == "user"
            assert assistant_entry["kind"] == "assistant"
            # Assistant write-back has no `type` field on response.
            assert "type" not in assistant_entry, assistant_entry
            assert assistant_entry["body"] == "hi back"
            assert assistant_entry["parent_id"] == user_entry["id"]

    def test_messages_passed_to_provider_use_anthropic_shape(self, tmp_path):
        """Sanity-check that the materialization output (Anthropic
        messages list) is the value the provider receives."""
        from fastapi.testclient import TestClient
        app, _ir = _compile_chat_app(tmp_path)

        stub_legacy = _TextOnlyStubLegacy("ack")
        with TestClient(app) as client:
            ctx = getattr(client.app.state, "ctx", None) or getattr(
                client.app, "_termin_ctx", None,
            )
            ctx.compute_providers = {"reply": _StubProvider(stub_legacy)}

            create = client.post(
                "/api/v1/chat_threads", json={"title": "shape"})
            thread_id = create.json()["id"]
            client.post(
                f"/api/v1/chat_threads/{thread_id}/conversation:append",
                json={"kind": "user", "body": "ping"})
            time.sleep(0.5)

        # The materialized messages list passed to the provider
        # contains the user entry (and only the user entry — the
        # assistant write-back happens inside the loop). Shape =
        # Anthropic's content-blocks form.
        msgs = stub_legacy.last_messages
        assert isinstance(msgs, list)
        assert len(msgs) == 1
        assert msgs[0]["role"] == "user"
        assert msgs[0]["content"][0]["type"] == "text"
        assert msgs[0]["content"][0]["text"] == "ping"


class TestConversationToolCallWriteback:
    """Per §11.5: tool calls and tool results are auto-appended to the
    conversation field in source order, all sharing parent_id =
    triggering user entry id."""

    def test_tool_call_result_text_all_share_parent_id(self, tmp_path):
        from fastapi.testclient import TestClient
        app, _ir = _compile_chat_app(tmp_path, with_tool=True)

        stub_legacy = _ToolUsingStubLegacy()
        with TestClient(app) as client:
            ctx = getattr(client.app.state, "ctx", None) or getattr(
                client.app, "_termin_ctx", None,
            )
            ctx.compute_providers = {"reply": _StubProvider(stub_legacy)}

            create = client.post(
                "/api/v1/chat_threads", json={"title": "tool"})
            thread_id = create.json()["id"]
            ap = client.post(
                f"/api/v1/chat_threads/{thread_id}/conversation:append",
                json={"kind": "user", "body": "what time?"})
            assert ap.status_code == 201
            time.sleep(0.7)

            get = client.get(f"/api/v1/chat_threads/{thread_id}")
            raw = get.json().get("conversation")
            entries = json.loads(raw) if isinstance(raw, str) else raw

        # Expected source order: user, tool_call, tool_result, assistant
        kinds = [e["kind"] for e in entries]
        assert kinds == [
            "user", "tool_call", "tool_result", "assistant",
        ], entries
        user_id = entries[0]["id"]
        # All three auto-write-back entries link back to the user entry.
        for auto in entries[1:]:
            assert auto.get("parent_id") == user_id, auto
        # tool_call carries id + name + args; tool_result carries id.
        tc = entries[1]
        assert tc["tool_call_id"] == "toolu_stub_1"
        assert tc["tool_name"] == "current_time"
        assert tc["tool_args"] == {}
        # body is the unsummarized tool_name(args) form (Q2 confirmed:
        # no truncation; future v0.9.3+ adds an optional `purpose`
        # field for short display).
        assert tc["body"] == "current_time({})"
        tr = entries[2]
        assert tr["tool_call_id"] == "toolu_stub_1"


class TestStreamingTextDeltas:
    """v0.9.2 close-out streaming path: agent_loop_with_conversation
    accepts on_text_delta + on_text_end callbacks. The runtime wires
    them to publish on `content.<source>.<field>.streaming` so the
    chat UI renders token-by-token without waiting for the turn to
    commit. Tested via stubs that drive the callback directly — the
    real Anthropic streaming surface is exercised by JL's manual
    end-to-end test (no live API in CI)."""

    def test_text_delta_callback_publishes_on_streaming_channel(
            self, tmp_path):
        """Stub legacy fires on_text_delta(delta) twice + on_text_end
        (committed=True) + on_writeback final. The runtime must
        publish per-delta + end events on the streaming channel."""
        from fastapi.testclient import TestClient
        app, _ir = _compile_chat_app(tmp_path)

        captured_events = []

        class _StreamingStub:
            async def agent_loop_with_conversation(
                self, directive, messages, tools, execute_tool,
                on_writeback, on_text_delta=None, on_text_end=None,
                should_halt=None, on_event=None, max_turns=20,
            ):
                if on_text_delta:
                    await on_text_delta("Hello ")
                    await on_text_delta("there")
                # Commit the final text via on_writeback (this is
                # what the production loop does) and fire end with
                # committed=True so the chat UI knows the matching
                # appended event will follow.
                await on_writeback(kind="assistant", body="Hello there")
                if on_text_end:
                    await on_text_end(committed=True)
                return {"thinking": "", "summary": "ok"}

        with TestClient(app) as client:
            ctx = getattr(client.app.state, "ctx", None) or getattr(
                client.app, "_termin_ctx", None,
            )
            ctx.compute_providers = {"reply": _StubProvider(_StreamingStub())}

            # Tap the event bus — every published event lands here.
            original_publish = ctx.event_bus.publish
            async def _capture_publish(event):
                captured_events.append(event)
                await original_publish(event)
            ctx.event_bus.publish = _capture_publish

            create = client.post(
                "/api/v1/chat_threads", json={"title": "stream"})
            thread_id = create.json()["id"]
            client.post(
                f"/api/v1/chat_threads/{thread_id}/conversation:append",
                json={"kind": "user", "body": "hi"})
            time.sleep(0.5)

        # The streaming channel saw two delta events + one end.
        stream_channel = (
            "content.chat_threads.conversation.streaming"
        )
        stream_events = [
            e for e in captured_events
            if e.get("channel_id") == stream_channel
        ]
        deltas = [e for e in stream_events if e.get("type") == "delta"]
        ends = [e for e in stream_events if e.get("type") == "end"]
        assert len(deltas) == 2, (
            f"expected 2 delta events on streaming channel; "
            f"got {len(deltas)}: {deltas!r}"
        )
        assert deltas[0]["text"] == "Hello "
        assert deltas[1]["text"] == "there"
        assert len(ends) == 1, ends
        assert ends[0]["committed"] is True
        # Each event has the record_id so the chat UI can filter
        # for the active thread.
        for e in stream_events:
            assert e.get("record_id") == thread_id

    def test_tool_call_turn_emits_text_end_committed_false(self, tmp_path):
        """When the agent's turn ends with tool calls (no final
        assistant text committed), the runtime fires
        on_text_end(committed=False) so the chat UI clears the
        pending bubble. The mid-turn streamed text was 'thinking',
        not a commit-bound reply."""
        from fastapi.testclient import TestClient
        app, _ir = _compile_chat_app(tmp_path, with_tool=True)

        captured_events = []

        class _ThinkingStub:
            async def agent_loop_with_conversation(
                self, directive, messages, tools, execute_tool,
                on_writeback, on_text_delta=None, on_text_end=None,
                should_halt=None, on_event=None, max_turns=20,
            ):
                # Turn 1: stream thinking text, then call a tool.
                if on_text_delta:
                    await on_text_delta("Let me check ")
                    await on_text_delta("the time...")
                await on_writeback(
                    kind="tool_call",
                    body="current_time({})",
                    tool_call_id="tc_1",
                    tool_name="current_time",
                    tool_args={},
                )
                await execute_tool("current_time", {})
                await on_writeback(
                    kind="tool_result", body="1pm",
                    tool_call_id="tc_1",
                )
                # Turn 1 ended with tool calls — clear pending bubble.
                if on_text_end:
                    await on_text_end(committed=False)
                # Turn 2: final reply.
                if on_text_delta:
                    await on_text_delta("It's 1pm.")
                await on_writeback(kind="assistant", body="It's 1pm.")
                if on_text_end:
                    await on_text_end(committed=True)
                return {"thinking": "", "summary": "ok"}

        with TestClient(app) as client:
            ctx = getattr(client.app.state, "ctx", None) or getattr(
                client.app, "_termin_ctx", None,
            )
            ctx.compute_providers = {"reply": _StubProvider(_ThinkingStub())}

            original_publish = ctx.event_bus.publish
            async def _capture_publish(event):
                captured_events.append(event)
                await original_publish(event)
            ctx.event_bus.publish = _capture_publish

            create = client.post(
                "/api/v1/chat_threads", json={"title": "thinking"})
            thread_id = create.json()["id"]
            client.post(
                f"/api/v1/chat_threads/{thread_id}/conversation:append",
                json={"kind": "user", "body": "what time?"})
            time.sleep(0.7)

        stream_channel = (
            "content.chat_threads.conversation.streaming"
        )
        ends = [
            e for e in captured_events
            if e.get("channel_id") == stream_channel
            and e.get("type") == "end"
        ]
        assert len(ends) == 2, ends
        assert ends[0]["committed"] is False  # tool turn
        assert ends[1]["committed"] is True   # final turn


class TestSystemRefuseToolDescription:
    """v0.9.2 close-out: the build_agent_tools-generated system_refuse
    tool description must match conversation-mode semantics — i.e.,
    it must NOT instruct the agent to call the legacy `set_output`
    sentinel after refusing (set_output doesn't exist on the
    conversation-mode tool surface). The agent reads this string
    and acts on it; a stale instruction means the agent calls
    refuse + tries to call set_output (which 404s as an unknown
    tool) + keeps generating text. JL caught this in manual
    testing 2026-05-04 evening when the agent reasoned 'I should
    call set_output to terminate the loop' after refusing."""

    def test_description_does_not_mention_set_output(self):
        from termin_server.ai_provider import build_agent_tools
        tools = build_agent_tools(["chat_threads"], {})
        refuse = next(
            (t for t in tools if t.get("name") == "system_refuse"),
            None,
        )
        assert refuse is not None, (
            "system_refuse must always be in the agent tool surface"
        )
        desc = refuse.get("description", "")
        assert "set_output" not in desc.lower(), (
            f"system_refuse description must not reference the "
            f"legacy set_output sentinel (doesn't exist in "
            f"conversation mode). Got:\n{desc}"
        )

    def test_description_directs_agent_to_stop_generating(self):
        """The description must tell the agent that calling
        system_refuse terminates its response — no further text
        or tool calls."""
        from termin_server.ai_provider import build_agent_tools
        tools = build_agent_tools(["chat_threads"], {})
        refuse = next(
            (t for t in tools if t.get("name") == "system_refuse"),
            None,
        )
        desc = (refuse.get("description") or "").lower()
        # One of these phrasings must appear; the exact wording can
        # evolve but the directive must be unmistakable.
        keywords = (
            "do not generate", "no further", "no additional",
            "terminate", "stop", "end your response",
        )
        assert any(k in desc for k in keywords), (
            f"system_refuse description must direct the agent to "
            f"stop generating after the call. Got:\n{refuse.get('description')}"
        )


class TestRuntimeSuppressesTextDeltasAfterRefusal:
    """v0.9.2 close-out: defense-in-depth for the refusal halt
    contract. The provider's mid-turn text stream may already be
    yielding deltas when a tool_use block for system_refuse arrives
    in the same response. The runtime's _on_text_delta callback
    must suppress further publication on the streaming channel
    once refusal_state is set so the chat UI's pending bubble
    doesn't accumulate post-refusal text. This is the runtime-side
    enforcement; the provider-side optimization (break out of the
    Anthropic stream context manager early) is a separate
    cost-saving measure tested via TestProviderBreaksOnRefuse."""

    def test_text_deltas_after_refusal_are_not_published(self, tmp_path):
        from fastapi.testclient import TestClient
        app, _ir = _compile_chat_app(tmp_path)

        captured_events = []

        class _RefuseMidStreamLegacy:
            async def agent_loop_with_conversation(
                self, directive, messages, tools, execute_tool,
                on_writeback, on_text_delta=None, on_text_end=None,
                should_halt=None, on_event=None, max_turns=20,
            ):
                # Stream pre-refusal text — should publish.
                if on_text_delta:
                    await on_text_delta("Let me think...")
                # Fire system_refuse mid-turn.
                await execute_tool(
                    "system_refuse",
                    {"reason": "policy: cannot continue"},
                )
                # A misbehaving provider might continue to fire
                # text deltas after the refuse tool returns. The
                # runtime must suppress these.
                if on_text_delta:
                    await on_text_delta(" but actually I'll just keep talking")
                if on_text_end:
                    await on_text_end(committed=False)
                return {"thinking": "", "summary": "halted"}

        with TestClient(app) as client:
            ctx = getattr(client.app.state, "ctx", None) or getattr(
                client.app, "_termin_ctx", None,
            )
            ctx.compute_providers = {
                "reply": _StubProvider(_RefuseMidStreamLegacy()),
            }

            original_publish = ctx.event_bus.publish
            async def _capture_publish(event):
                captured_events.append(event)
                await original_publish(event)
            ctx.event_bus.publish = _capture_publish

            create = client.post(
                "/api/v1/chat_threads", json={"title": "suppress"})
            thread_id = create.json()["id"]
            client.post(
                f"/api/v1/chat_threads/{thread_id}/conversation:append",
                json={"kind": "user", "body": "trigger refuse"})
            time.sleep(0.6)

        stream_channel = (
            "content.chat_threads.conversation.streaming"
        )
        delta_events = [
            e for e in captured_events
            if e.get("channel_id") == stream_channel
            and e.get("type") == "delta"
        ]
        delta_texts = [e.get("text", "") for e in delta_events]
        # Pre-refusal "Let me think..." DOES publish.
        assert "Let me think..." in delta_texts, delta_texts
        # Post-refusal " but actually..." MUST NOT publish.
        for txt in delta_texts:
            assert "actually I'll just keep talking" not in txt, (
                f"post-refusal delta leaked through to streaming "
                f"channel: {delta_texts!r}"
            )


class TestProviderBreaksOnRefuse:
    """v0.9.2 close-out: provider-side optimization. When the
    Anthropic stream's content_block_start fires for a tool_use
    block named system_refuse, the producer should stop forwarding
    further text_delta events to the consumer. This complements
    the runtime-side _on_text_delta suppression
    (TestRuntimeSuppressesTextDeltasAfterRefusal) but acts at the
    provider boundary so the bridge queue isn't burdened with
    deltas that will be discarded.

    Tests the producer logic directly via a mock-shaped Anthropic
    stream that yields scripted events.
    """

    def test_text_deltas_after_system_refuse_block_are_dropped(self):
        from termin_server.ai_provider import (
            _producer_for_conversation_stream,
        )

        # Scripted event stream: text_delta, content_block_start
        # for tool_use system_refuse, text_delta (post-refuse).
        class _Event:
            def __init__(self, type, **kw):
                self.type = type
                for k, v in kw.items():
                    setattr(self, k, v)

        class _Delta:
            def __init__(self, type, text=""):
                self.type = type
                self.text = text

        class _ContentBlock:
            def __init__(self, type, name=None):
                self.type = type
                self.name = name

        events = [
            _Event("content_block_delta",
                   delta=_Delta("text_delta", text="Pre-refuse text. ")),
            _Event("content_block_start",
                   content_block=_ContentBlock("tool_use",
                                               name="system_refuse")),
            _Event("content_block_delta",
                   delta=_Delta("text_delta",
                                text="Post-refuse text — drop me.")),
        ]
        emitted: list = []
        def put(item):
            emitted.append(item)
        _producer_for_conversation_stream(events, put)
        delta_texts = [
            i.get("text", "") for i in emitted
            if i.get("type") == "text_delta"
        ]
        assert "Pre-refuse text. " in delta_texts, delta_texts
        for t in delta_texts:
            assert "Post-refuse text" not in t, (
                f"producer must drop text_deltas after system_refuse "
                f"content_block_start; emitted: {delta_texts!r}"
            )


class TestRefusalTerminatesLoop:
    """v0.9.2 close-out: per compute-contract.md §6.1, system_refuse
    must terminate the agent loop. The runtime supplies should_halt
    as a closure over refusal_state; the AI provider checks it
    between turns and between mid-turn tool calls. Post-refusal
    on_writeback calls short-circuit so no further entries land on
    the conversation field — the refusal entry (appended by L7.4
    post-loop) is the last commit.

    Without this enforcement, an agent could call system_refuse
    and then keep calling other tools (content_create, etc.) —
    side effects would still fire, audit incoherence
    (outcome=refused but conversation has post-refusal entries),
    and the platform's enforcement-over-vigilance promise would
    be advisory at best."""

    def test_should_halt_skips_subsequent_turns(self, tmp_path):
        from fastapi.testclient import TestClient
        app, _ir = _compile_chat_app(tmp_path)

        turns_executed = {"count": 0}

        class _ChattyAfterRefuseLegacy:
            async def agent_loop_with_conversation(
                self, directive, messages, tools, execute_tool,
                on_writeback, on_text_delta=None, on_text_end=None,
                should_halt=None, on_event=None, max_turns=20,
            ):
                # Turn 1: refuse, then "continue" to attempt more
                # tool calls + a text commit. The runtime's
                # short-circuits should prevent any of the
                # post-refusal effects from landing.
                for turn in range(5):
                    if should_halt and should_halt():
                        return {"thinking": "", "summary": "halted"}
                    turns_executed["count"] += 1
                    if turn == 0:
                        # First turn: fire system_refuse.
                        await execute_tool(
                            "system_refuse",
                            {"reason": "policy: cannot do this"},
                        )
                        # An agent might continue trying things.
                        # Each subsequent on_writeback should be
                        # short-circuited; each subsequent
                        # execute_tool should error out.
                        continue
                    # Post-refuse turns: try to commit text + call
                    # a tool. These should all be no-ops.
                    await on_writeback(
                        kind="assistant",
                        body=f"chatty followup turn {turn}",
                    )
                return {"thinking": "", "summary": "completed"}

        with TestClient(app) as client:
            ctx = getattr(client.app.state, "ctx", None) or getattr(
                client.app, "_termin_ctx", None,
            )
            ctx.compute_providers = {
                "reply": _StubProvider(_ChattyAfterRefuseLegacy()),
            }

            create = client.post(
                "/api/v1/chat_threads", json={"title": "halt"})
            thread_id = create.json()["id"]
            client.post(
                f"/api/v1/chat_threads/{thread_id}/conversation:append",
                json={"kind": "user", "body": "do something bad"})
            time.sleep(0.7)

            get = client.get(f"/api/v1/chat_threads/{thread_id}")
            raw = get.json().get("conversation")
            entries = json.loads(raw) if isinstance(raw, str) else raw

        # Only the user entry + the refusal entry are on the field.
        # No "chatty followup turn N" assistant entries committed
        # despite the stub trying to write them. should_halt
        # caused the loop to bail; on_writeback short-circuited
        # the writes that did fire.
        kinds = [e["kind"] for e in entries]
        assert kinds == ["user", "assistant"], entries
        assert entries[1].get("type") == "refusal"
        assert "policy" in entries[1]["body"]
        # No body text from the stub's chatty followups landed.
        bodies = [e.get("body", "") for e in entries]
        for b in bodies:
            assert "chatty followup" not in b, (
                f"post-refusal commit leaked through: {bodies!r}"
            )

    def test_post_refusal_tool_call_returns_error(self, tmp_path):
        """Defensive: even if a future provider doesn't honor
        should_halt and tries to call another tool after refusal,
        the runtime's _execute_tool gate returns an error envelope
        rather than executing."""
        from fastapi.testclient import TestClient
        app, _ir = _compile_chat_app(tmp_path)

        post_refuse_results = []

        class _IgnoreHaltLegacy:
            async def agent_loop_with_conversation(
                self, directive, messages, tools, execute_tool,
                on_writeback, on_text_delta=None, on_text_end=None,
                should_halt=None, on_event=None, max_turns=20,
            ):
                # Refuse first.
                await execute_tool(
                    "system_refuse", {"reason": "no"},
                )
                # Pretend the provider ignores should_halt and
                # tries another tool. The runtime's tool gate
                # should return an error envelope.
                result = await execute_tool(
                    "content_query",
                    {"content_name": "chat_threads"},
                )
                post_refuse_results.append(result)
                return {"thinking": "", "summary": "completed"}

        with TestClient(app) as client:
            ctx = getattr(client.app.state, "ctx", None) or getattr(
                client.app, "_termin_ctx", None,
            )
            ctx.compute_providers = {
                "reply": _StubProvider(_IgnoreHaltLegacy()),
            }
            create = client.post(
                "/api/v1/chat_threads", json={"title": "gate"})
            thread_id = create.json()["id"]
            client.post(
                f"/api/v1/chat_threads/{thread_id}/conversation:append",
                json={"kind": "user", "body": "x"})
            time.sleep(0.5)

        assert len(post_refuse_results) == 1
        assert "error" in post_refuse_results[0], (
            f"post-refusal tool call should return an error envelope; "
            f"got {post_refuse_results[0]!r}"
        )


class TestRefusalRegressionUnderL71:
    """L7.4's refusal-append path must keep working under L7.1's new
    dispatch — refusal still wins over normal write-back when
    system_refuse fires."""

    def test_refusal_still_appends_assistant_refusal_entry(self, tmp_path):
        from fastapi.testclient import TestClient
        app, _ir = _compile_chat_app(tmp_path)

        class _RefuseLegacy:
            async def agent_loop_with_conversation(
                self, directive, messages, tools, execute_tool,
                on_writeback, on_text_delta=None, on_text_end=None,
        should_halt=None, on_event=None, max_turns=20,
            ):
                # Refuse via the runtime-gated tool; do NOT call
                # on_writeback for normal text — refusal path owns the
                # entry.
                await execute_tool("system_refuse", {"reason": "nope"})
                return {"thinking": "refused", "summary": ""}

        with TestClient(app) as client:
            ctx = getattr(client.app.state, "ctx", None) or getattr(
                client.app, "_termin_ctx", None,
            )
            ctx.compute_providers = {"reply": _StubProvider(_RefuseLegacy())}

            create = client.post(
                "/api/v1/chat_threads", json={"title": "refuse"})
            thread_id = create.json()["id"]
            client.post(
                f"/api/v1/chat_threads/{thread_id}/conversation:append",
                json={"kind": "user", "body": "do harm"})
            time.sleep(0.5)

            get = client.get(f"/api/v1/chat_threads/{thread_id}")
            raw = get.json().get("conversation")
            entries = json.loads(raw) if isinstance(raw, str) else raw
            kinds = [e["kind"] for e in entries]
            assert kinds == ["user", "assistant"], entries
            assert entries[1]["type"] == "refusal"
            assert entries[1]["body"] == "nope"


# ── L11: examples/agent_chatbot.termin (the v0.9.2 canonical example)
#
# The integration tests below boot the actual `examples/agent_chatbot.termin`
# program — not a synthesized fixture — and stub the legacy provider so
# we can exercise the full conversation-mode path without a live API
# key. This is the "tests pass != it works" rule met halfway: the
# example compiles and the runtime materializes / writes back per
# §11.5 against a stub. End-to-end with a real Anthropic key is
# verified out-of-band.


def _compile_agent_chatbot(tmp_path):
    """Compile the canonical examples/agent_chatbot.termin and boot it
    against a fresh per-test SQLite DB. Returns the FastAPI app."""
    from pathlib import Path
    from termin import peg_parser, analyzer, lower
    from termin_core.ir.serialize import serialize_ir
    from termin_server import create_termin_app

    # Resolve the example relative to the termin-compiler repo. We
    # walk up from this test file: termin-server/tests/<this> ->
    # termin-server -> ClaudeWorkspace -> termin-compiler/examples.
    example_path = (
        Path(__file__).resolve().parent.parent.parent
        / "termin-compiler" / "examples" / "agent_chatbot.termin"
    )
    source = example_path.read_text(encoding="utf-8")
    program, perr = peg_parser.parse_peg(source)
    assert perr.ok, perr.format()
    aerr = analyzer.analyze(program)
    assert aerr.ok, aerr.format()
    spec = lower.lower(program)
    ir_json = serialize_ir(spec)
    db_path = str(tmp_path / "agent_chatbot.db")
    return create_termin_app(ir_json, db_path=db_path)


class TestAgentChatbotV092EndToEnd:
    """L11: the v0.9.2 examples/agent_chatbot.termin example compiles
    and runs the conversation-mode agent loop. This is the example
    test surface — it needs to keep passing every time someone
    touches the conversation-mode dispatch."""

    def test_multi_turn_conversation_threads_through_agent(self, tmp_path):
        """Two user turns → two assistant replies, all parent-linked,
        all in source order on the conversation field."""
        from fastapi.testclient import TestClient

        # Stub legacy that echoes the latest user message back. The
        # `messages` array the runtime hands the legacy is the
        # materialized §11.4 shape; the stub digs the latest user
        # text out of it for a deterministic reply.
        class _EchoLegacy:
            def __init__(self):
                self.calls = []

            async def agent_loop_with_conversation(
                self, directive, messages, tools, execute_tool,
                on_writeback, on_text_delta=None, on_text_end=None,
        should_halt=None, on_event=None, max_turns=20,
            ):
                self.calls.append({"messages": messages})
                # Pull the most recent user-role text block.
                last_user = ""
                for msg in reversed(messages):
                    if msg.get("role") == "user":
                        for block in msg.get("content") or []:
                            if block.get("type") == "text":
                                last_user = block.get("text", "")
                                break
                        if last_user:
                            break
                await on_writeback(
                    kind="assistant",
                    body=f"echo: {last_user}",
                )
                return {"thinking": "", "summary": "ok"}

        app = _compile_agent_chatbot(tmp_path)
        stub = _EchoLegacy()
        with TestClient(app) as client:
            ctx = getattr(client.app.state, "ctx", None) or getattr(
                client.app, "_termin_ctx", None,
            )
            assert ctx is not None
            ctx.compute_providers = {"reply": _StubProvider(stub)}

            # Create a thread, drive two user turns through the
            # append endpoint, verify the agent reply lands on each.
            create = client.post(
                "/api/v1/chat_threads", json={"title": "demo"})
            assert create.status_code in (200, 201), create.text
            thread_id = create.json()["id"]

            ap1 = client.post(
                f"/api/v1/chat_threads/{thread_id}/conversation:append",
                json={"kind": "user", "body": "hi"})
            assert ap1.status_code == 201
            time.sleep(0.4)

            ap2 = client.post(
                f"/api/v1/chat_threads/{thread_id}/conversation:append",
                json={"kind": "user", "body": "again"})
            assert ap2.status_code == 201
            time.sleep(0.4)

            get = client.get(f"/api/v1/chat_threads/{thread_id}")
            raw = get.json().get("conversation")
            entries = json.loads(raw) if isinstance(raw, str) else raw

        kinds = [e["kind"] for e in entries]
        assert kinds == [
            "user", "assistant", "user", "assistant",
        ], entries
        # Each assistant reply parent-links to the user message
        # *that triggered its turn* — not the most recent overall.
        assert entries[1]["parent_id"] == entries[0]["id"]
        assert entries[3]["parent_id"] == entries[2]["id"]
        assert entries[1]["body"] == "echo: hi"
        assert entries[3]["body"] == "echo: again"
        # The second turn's materialized history must include the
        # first turn (not just the new user message). The agent sees
        # the full conversation each time per §11.5's no-truncation
        # stance for v0.9.2.
        second_turn_msgs = stub.calls[1]["messages"]
        # After adjacent-role merging: turn 2's history is
        # [user("hi"), assistant("echo: hi"), user("again")] — three
        # messages alternating roles.
        assert len(second_turn_msgs) == 3
        assert second_turn_msgs[0]["role"] == "user"
        assert second_turn_msgs[1]["role"] == "assistant"
        assert second_turn_msgs[2]["role"] == "user"
        assert second_turn_msgs[2]["content"][0]["text"] == "again"

    def test_refusal_renders_inline_via_v092_path(self, tmp_path):
        """The refusal-as-assistant-with-type=refusal path (L7.4)
        works against the canonical example."""
        from fastapi.testclient import TestClient

        class _RefuseLegacy:
            async def agent_loop_with_conversation(
                self, directive, messages, tools, execute_tool,
                on_writeback, on_text_delta=None, on_text_end=None,
        should_halt=None, on_event=None, max_turns=20,
            ):
                await execute_tool(
                    "system_refuse",
                    {"reason": "fabricating sources is off-policy"},
                )
                return {"thinking": "", "summary": ""}

        app = _compile_agent_chatbot(tmp_path)
        with TestClient(app) as client:
            ctx = getattr(client.app.state, "ctx", None) or getattr(
                client.app, "_termin_ctx", None,
            )
            ctx.compute_providers = {"reply": _StubProvider(_RefuseLegacy())}

            create = client.post(
                "/api/v1/chat_threads", json={"title": "refuse demo"})
            thread_id = create.json()["id"]
            client.post(
                f"/api/v1/chat_threads/{thread_id}/conversation:append",
                json={"kind": "user",
                      "body": "Make up a real-sounding citation."})
            time.sleep(0.4)

            get = client.get(f"/api/v1/chat_threads/{thread_id}")
            raw = get.json().get("conversation")
            entries = json.loads(raw) if isinstance(raw, str) else raw

        assert [e["kind"] for e in entries] == ["user", "assistant"]
        assert entries[1]["type"] == "refusal"
        assert "fabricating" in entries[1]["body"]


# ── v0.9.2 final close-out: `purpose` field on tool_call entries ──
#
# Per the original v0.9.2 spec (§11.5 + JL's Q2 from earlier today):
# tool_call entries can carry an optional `purpose` field — a short
# (6 words ideal, 12-word hard cap with ellipsis truncation) display
# string the agent supplies for each tool call. Lets chat UIs show
# a meaningful label without parsing the JSON args.


class TestPurposeFieldTruncation:
    """Per spec: 6 words ideal, hard cap at 12 words with ellipsis."""

    def test_under_12_words_passes_through_unchanged(self):
        from termin_server.ai_provider import _truncate_purpose
        assert _truncate_purpose("checking the time") == "checking the time"
        # Exactly 12 words still passes through.
        twelve = " ".join(f"w{i}" for i in range(12))
        assert _truncate_purpose(twelve) == twelve

    def test_thirteen_or_more_words_truncates_with_ellipsis(self):
        from termin_server.ai_provider import _truncate_purpose
        thirteen = " ".join(f"w{i}" for i in range(13))
        out = _truncate_purpose(thirteen)
        # First 12 words + ellipsis (no trailing space before).
        assert out == " ".join(f"w{i}" for i in range(12)) + "..."

    def test_empty_returns_empty(self):
        from termin_server.ai_provider import _truncate_purpose
        assert _truncate_purpose("") == ""

    def test_handles_excess_whitespace(self):
        """Word count uses split() default — collapses runs of whitespace."""
        from termin_server.ai_provider import _truncate_purpose
        out = _truncate_purpose("  checking   the  time  ")
        assert out == "checking the time"


class TestPurposeFieldOnToolCallEntry:
    """End-to-end: when the conversation-mode loop writes a tool_call
    entry, the `purpose` value the agent supplied (if any) lands on
    the persisted entry, truncated per spec."""

    def test_purpose_persists_when_supplied(self, tmp_path):
        from fastapi.testclient import TestClient
        app, _ir = _compile_chat_app(tmp_path, with_tool=True)

        class _PurposefulLegacy:
            async def agent_loop_with_conversation(
                self, directive, messages, tools, execute_tool,
                on_writeback, on_text_delta=None, on_text_end=None,
        should_halt=None, on_event=None, max_turns=20,
            ):
                await on_writeback(
                    kind="tool_call",
                    body="current_time({})",
                    tool_call_id="toolu_p1",
                    tool_name="current_time",
                    tool_args={},
                    purpose="checking the time",
                )
                tool_out = await execute_tool("current_time", {})
                await on_writeback(
                    kind="tool_result",
                    body=json.dumps(tool_out),
                    tool_call_id="toolu_p1",
                )
                await on_writeback(
                    kind="assistant", body="It's 10am.",
                )
                return {"thinking": "", "summary": "ok"}

        with TestClient(app) as client:
            ctx = getattr(client.app.state, "ctx", None) or getattr(
                client.app, "_termin_ctx", None,
            )
            ctx.compute_providers = {"reply": _StubProvider(_PurposefulLegacy())}

            create = client.post(
                "/api/v1/chat_threads", json={"title": "purpose"})
            thread_id = create.json()["id"]
            client.post(
                f"/api/v1/chat_threads/{thread_id}/conversation:append",
                json={"kind": "user", "body": "what time?"})
            time.sleep(0.5)

            get = client.get(f"/api/v1/chat_threads/{thread_id}")
            raw = get.json().get("conversation")
            entries = json.loads(raw) if isinstance(raw, str) else raw

        kinds = [e["kind"] for e in entries]
        assert kinds == [
            "user", "tool_call", "tool_result", "assistant",
        ], entries
        tc = entries[1]
        assert tc.get("purpose") == "checking the time"

    def test_purpose_absent_when_not_supplied(self, tmp_path):
        """The classic L7.3 behavior is unchanged: absent purpose is
        not added as null — the field is omitted from the entry."""
        from fastapi.testclient import TestClient
        app, _ir = _compile_chat_app(tmp_path, with_tool=True)

        with TestClient(app) as client:
            ctx = getattr(client.app.state, "ctx", None) or getattr(
                client.app, "_termin_ctx", None,
            )
            ctx.compute_providers = {"reply": _StubProvider(_ToolUsingStubLegacy())}

            create = client.post(
                "/api/v1/chat_threads", json={"title": "no purpose"})
            thread_id = create.json()["id"]
            client.post(
                f"/api/v1/chat_threads/{thread_id}/conversation:append",
                json={"kind": "user", "body": "what time?"})
            time.sleep(0.5)

            get = client.get(f"/api/v1/chat_threads/{thread_id}")
            raw = get.json().get("conversation")
            entries = json.loads(raw) if isinstance(raw, str) else raw

        tc = entries[1]
        assert tc["kind"] == "tool_call"
        assert "purpose" not in tc, (
            f"purpose should be omitted when not supplied; got {tc!r}"
        )


# ── v0.9.2 final close-out: Invokes runtime wiring ──
#
# Per the original v0.9.2 §12 / §16 spec: `Invokes "<compute_name>"`
# on an ai-agent compute makes the named compute callable as an
# agent tool. The tool's input schema is built from the compute's
# input params; tool dispatch evaluates the compute (CEL expression
# body, via the symbol environment built from tool_args).
#
# v0.9.2 supports default-CEL invokable tools only. Invoking an
# `llm` or `ai-agent` compute as a tool is reserved for future
# slices.


class TestBuildInvokableComputeTools:
    """Per §11.4 / §16 in the v0.9.2 design doc: tool schema per
    declared Invokes entry."""

    def test_empty_invokes_returns_empty_list(self):
        from termin_server.ai_provider import build_invokable_compute_tools
        assert build_invokable_compute_tools([], {}) == []

    def test_unknown_invokes_skipped(self):
        from termin_server.ai_provider import build_invokable_compute_tools
        out = build_invokable_compute_tools(["nonexistent"], {})
        assert out == [], out

    def test_cel_compute_with_one_param_yields_tool(self):
        from termin_server.ai_provider import build_invokable_compute_tools
        computes_lookup = {
            "current_time": {
                "name": {"snake": "current_time", "display": "current_time"},
                "provider": "cel",
                "input_params": [{"name": "query", "type_name": "text"}],
                "directive": None,
            },
        }
        out = build_invokable_compute_tools(["current_time"], computes_lookup)
        assert len(out) == 1
        tool = out[0]
        assert tool["name"] == "current_time"
        assert "input_schema" in tool
        # The compute's `query` param surfaces as a property.
        assert "query" in tool["input_schema"]["properties"]

    def test_non_cel_compute_skipped(self):
        """Invoking llm or ai-agent computes as tools is reserved
        for future slices; v0.9.2 supports default-CEL only."""
        from termin_server.ai_provider import build_invokable_compute_tools
        computes_lookup = {
            "agent_x": {
                "name": {"snake": "agent_x", "display": "agent_x"},
                "provider": "ai-agent",
                "input_params": [],
                "directive": "you are an agent",
            },
        }
        out = build_invokable_compute_tools(["agent_x"], computes_lookup)
        assert out == [], out


class TestInvokesEndToEnd:
    """End-to-end: agent compute with `Invokes "current_time"` can
    actually call current_time as a tool, gets back the CEL eval
    result, and the tool_call/tool_result entries land on the
    conversation field with full linkage."""

    def test_agent_invokes_cel_compute_and_gets_result(self, tmp_path):
        """The fixture (`_compile_agent_chatbot_with_invokes`) builds
        an agent_chatbot-shaped app that declares a `current_time`
        Compute and an `Invokes "current_time"` line on `reply`. The
        stub agent calls current_time; the runtime evaluates the CEL
        body and returns the result; the auto-write-back captures
        both the tool_call and the tool_result with parent_id linkage."""
        from fastapi.testclient import TestClient

        # Build a small program with an Invokable CEL compute.
        from termin import peg_parser, analyzer, lower
        from termin_core.ir.serialize import serialize_ir
        from termin_server import create_termin_app

        source = '''Application: Invokes Test
  Description: v0.9.2 Invokes runtime wiring fixture
Id: 7e8c1b2e-8f4a-4b1c-9d8e-2f5a3b7c8d99

Identity:
  Scopes are "chat.use"
  Anonymous has "chat.use"

Content called "chat_threads":
  Each chat_thread has a title which is text, defaults to "Conversation"
  Each chat_thread has a conversation which is conversation
  Anyone with "chat.use" can view chat_threads
  Anyone with "chat.use" can create chat_threads
  Anyone with "chat.use" can append to chat_threads.conversation

Compute called "current_time":
  Transform: takes a chat_thread, produces a chat_thread
  `"2026-05-04T10:00:00Z"`
  Anyone with "chat.use" can execute this

Compute called "reply":
  Provider is "ai-agent"
  Trigger on event "chat_threads.conversation.appended" where `appended_entry.kind == "user"`
  Conversation is chat_threads.conversation
  Invokes "current_time"
  Anyone with "chat.use" can execute this
  Audit level: actions
  Anyone with "chat.use" can audit
  Directive is ```
    Use the current_time tool when asked about time.
  ```

As an anonymous, I want to chat:
  Show a page called "Chat"
'''
        program, perr = peg_parser.parse_peg(source)
        assert perr.ok, perr.format()
        aerr = analyzer.analyze(program)
        assert aerr.ok, aerr.format()
        spec = lower.lower(program)
        ir_json = serialize_ir(spec)
        db_path = str(tmp_path / "invokes_test.db")
        app = create_termin_app(ir_json, db_path=db_path)

        captured_tools = {}

        class _InvokesStubLegacy:
            async def agent_loop_with_conversation(
                self, directive, messages, tools, execute_tool,
                on_writeback, on_text_delta=None, on_text_end=None,
        should_halt=None, on_event=None, max_turns=20,
            ):
                # Verify current_time is in the tool surface.
                captured_tools["names"] = [t["name"] for t in tools]
                # Call it with an arg.
                await on_writeback(
                    kind="tool_call",
                    body='current_time({"query":"now"})',
                    tool_call_id="toolu_inv_1",
                    tool_name="current_time",
                    tool_args={"query": "now"},
                    purpose="checking the time",
                )
                result = await execute_tool(
                    "current_time", {"query": "now"})
                captured_tools["result"] = result
                await on_writeback(
                    kind="tool_result",
                    body=str(result),
                    tool_call_id="toolu_inv_1",
                )
                await on_writeback(
                    kind="assistant",
                    body="The time is 10am.",
                )
                return {"thinking": "", "summary": "ok"}

        with TestClient(app) as client:
            ctx = getattr(client.app.state, "ctx", None) or getattr(
                client.app, "_termin_ctx", None,
            )
            ctx.compute_providers = {"reply": _StubProvider(_InvokesStubLegacy())}

            create = client.post(
                "/api/v1/chat_threads", json={"title": "invokes"})
            thread_id = create.json()["id"]
            client.post(
                f"/api/v1/chat_threads/{thread_id}/conversation:append",
                json={"kind": "user", "body": "what time?"})
            time.sleep(0.5)

            get = client.get(f"/api/v1/chat_threads/{thread_id}")
            raw = get.json().get("conversation")
            entries = json.loads(raw) if isinstance(raw, str) else raw

        # current_time was in the tool surface.
        assert "current_time" in captured_tools.get("names", []), (
            f"current_time tool not surfaced; got "
            f"{captured_tools.get('names')!r}"
        )
        # The compute returned its CEL result.
        assert captured_tools.get("result"), captured_tools
        # The four expected entries landed.
        kinds = [e["kind"] for e in entries]
        assert kinds == [
            "user", "tool_call", "tool_result", "assistant",
        ], entries
        # Purpose persists.
        assert entries[1].get("purpose") == "checking the time"
