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
        on_writeback, on_event=None, max_turns=20,
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
        on_writeback, on_event=None, max_turns=20,
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
                on_writeback, on_event=None, max_turns=20,
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
