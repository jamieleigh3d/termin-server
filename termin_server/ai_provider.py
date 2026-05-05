# Copyright 2026 Jamie-Leigh Blake and Termin project contributors
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0

"""AI Provider — calls LLM APIs for Compute execution.

Supports two modes:
- Level 1 (Provider is "llm"): Field-to-field completion with tool_use output
- Level 3 (Provider is "ai-agent"): Autonomous agent with ComputeContext tools

Built-in providers: Anthropic (Claude) and OpenAI (GPT).
Provider is selected via deploy config ai_provider section.
"""

import asyncio
import json
import logging
from typing import Any, Optional

logger = logging.getLogger("termin.ai_provider")


class AIProviderError(Exception):
    """Error from AI provider (API failure, missing config, etc.)."""
    pass


class ConversationMaterializationError(Exception):
    """v0.9.2 §11.4: a conversation entry list violated the canonical
    materialization contract (e.g. a tool_result whose tool_call_id
    doesn't match any preceding tool_call). Raised by
    ``materialize_to_anthropic``; the runtime treats it as a
    server-side error (the source-of-truth field has bad data the
    runtime can't translate into a valid provider call)."""


# ── v0.9.2 L7.2: canonical kind → Anthropic mapping ──
#
# Per tech-design §11.4 (verified against Anthropic API docs):
#   Termin kind     | Anthropic role | content blocks
#   user            | user           | {type:text} + image/document blocks
#   assistant       | assistant      | {type:text} (+ image blocks where supported)
#   tool_call       | assistant      | {type:tool_use, id, name, input}
#   tool_result     | user           | {type:tool_result, tool_use_id, content, is_error?}
#   system_event    | user           | {type:text, text:"[<source>] <body>"}
#
# The `assistant.type == "refusal"` discriminator is **not** sent to
# Anthropic — refusal-type assistant entries map identically to
# response-type ones. The type field is for Termin's audit and chat
# rendering only. Per tech-design §11.4.
#
# Adjacent same-role entries are merged into one message with multiple
# content blocks (Anthropic requires alternating user/assistant
# roles).
#
# Tool linkage rule: every tool_result must reference a tool_call_id
# that appears in a prior tool_call entry. Orphan tool_results raise
# ConversationMaterializationError.

_KINDS_USER_ROLE = frozenset({"user", "tool_result", "system_event"})
_KINDS_ASSISTANT_ROLE = frozenset({"assistant", "tool_call"})


# ── v0.9.2 close-out: `purpose` field on tool_call entries ──
#
# Per JL's Q2 (today): tool_call entries can carry an optional
# `purpose` field — a short (6 words ideal, 12-word hard cap with
# ellipsis truncation) display string the agent supplies for each
# tool call. Lets chat UIs show a meaningful label without parsing
# the JSON args.
#
# The runtime extracts `purpose` from the agent's tool_use input
# during materialization, truncates at the hard cap, and persists
# it on the conversation entry as a top-level field. The agent's
# remaining args (with `purpose` stripped) go to the tool callback.

_PURPOSE_MAX_WORDS = 12

_PURPOSE_TOOL_DESCRIPTION = (
    "Short, 6-words-or-less, plain-English description of why you're "
    "calling this tool — for chat-UI display. Examples: "
    "'checking the time', 'looking up the order', 'updating the "
    "ticket status'. Hard truncated at 12 words with ellipsis on "
    "persistence."
)


def _truncate_purpose(text: str, max_words: int = _PURPOSE_MAX_WORDS) -> str:
    """Hard-truncate a purpose string to ``max_words`` (default 12)
    with ellipsis when over. Collapses runs of whitespace via
    ``str.split()`` default semantics."""
    if not text:
        return ""
    words = text.split()
    if len(words) <= max_words:
        return " ".join(words)
    return " ".join(words[:max_words]) + "..."


def _purpose_property() -> dict:
    """The Anthropic JSON-schema property dict for the `purpose`
    field. Added to every tool's input_schema by the conversation-
    mode tool-surface builder so the agent is consistently prompted
    to supply intent."""
    return {
        "type": "string",
        "description": _PURPOSE_TOOL_DESCRIPTION,
    }


def _add_purpose_to_tool(tool: dict) -> dict:
    """Return a shallow copy of ``tool`` with `purpose` added to its
    input_schema's properties (idempotent — no-op if already present).
    Does NOT mark `purpose` as required; agents are encouraged to
    supply it but the chat UI falls back to body when absent."""
    tool = dict(tool)
    schema = dict(tool.get("input_schema") or {})
    props = dict(schema.get("properties") or {})
    if "purpose" not in props:
        props["purpose"] = _purpose_property()
    schema["properties"] = props
    tool["input_schema"] = schema
    return tool


# ── v0.9.2 close-out: Invokes runtime wiring ──
#
# `Invokes "<compute_name>"` declarations on an ai-agent compute
# surface the named compute as an agent tool. v0.9.2 supports
# default-CEL invokable tools only; LLM and ai-agent invocations
# from another agent are reserved for future slices (the wiring
# here doesn't emit tools for non-CEL providers — cleanly ignored).


def build_invokable_compute_tools(
    invokes_list: list[str],
    computes_lookup: dict,
) -> list[dict]:
    """Build Anthropic-shape tool schemas for each compute named in
    the agent's Invokes declarations. Per the v0.9.2 design §11:
    tool name = compute snake_name; description from the compute's
    display_name + first-line directive (when present); input schema
    derived from the compute's input_params.

    v0.9.2 supports default-CEL providers only. Computes with
    provider="llm" or "ai-agent" are skipped (logged as future).
    Unknown invokes (compute not in lookup) are also skipped — the
    analyzer should have caught this at compile time, but the
    runtime is forgiving.
    """
    tools = []
    for invoke_name in invokes_list:
        comp = computes_lookup.get(invoke_name)
        if comp is None:
            continue
        provider = comp.get("provider") or "cel"
        # Only default-CEL is wired in v0.9.2.
        if provider not in ("cel", "default-CEL", None, ""):
            continue
        name = comp.get("name", {})
        snake = name.get("snake") if isinstance(name, dict) else invoke_name
        display = name.get("display") if isinstance(name, dict) else invoke_name
        directive = (comp.get("directive") or "").strip().splitlines()
        first_line = directive[0] if directive else ""
        description = display
        if first_line:
            description = f"{display} — {first_line}"
        properties = {}
        required = []
        for param in comp.get("input_params") or ():
            pname = param.get("name") if isinstance(param, dict) else getattr(param, "name", None)
            ptype = param.get("type_name") if isinstance(param, dict) else getattr(param, "type_name", None)
            if not pname:
                continue
            # Param types are content-type singulars; v0.9.2 ships
            # an object-typed schema for each (the agent passes a
            # record-shaped dict; the runtime evaluates the body
            # with that dict bound to the param name).
            properties[pname] = {
                "type": "object",
                "description": (
                    f"The {ptype} record this compute operates on."
                    if ptype else
                    f"The {pname} input."
                ),
                "additionalProperties": True,
            }
            required.append(pname)
        schema = {
            "type": "object",
            "properties": properties,
            "required": required,
        }
        tools.append({
            "name": snake,
            "description": description,
            "input_schema": schema,
        })
    return tools


def _entry_role(kind: str) -> str:
    """Map a Termin entry kind to its Anthropic role."""
    if kind in _KINDS_ASSISTANT_ROLE:
        return "assistant"
    if kind in _KINDS_USER_ROLE:
        return "user"
    # Defensive: callers should validate kind upstream (the append
    # endpoint enforces _CANONICAL_KINDS); fall back to user role.
    return "user"


def _build_content_blocks(entry: dict) -> list[dict]:
    """Build the Anthropic content-blocks list for a single entry.

    The block shape depends on the entry's kind:
      - user / assistant / system_event → text block (+ attachments
        for user)
      - tool_call → tool_use block
      - tool_result → tool_result block
    """
    kind = entry.get("kind", "")
    body = entry.get("body", "")

    if kind == "tool_call":
        return [{
            "type": "tool_use",
            "id": entry.get("tool_call_id", ""),
            "name": entry.get("tool_name", ""),
            "input": entry.get("tool_args") or {},
        }]

    if kind == "tool_result":
        block: dict = {
            "type": "tool_result",
            "tool_use_id": entry.get("tool_call_id", ""),
            "content": body,
        }
        if entry.get("is_error"):
            block["is_error"] = True
        return [block]

    # text-bearing kinds: user, assistant, system_event
    if kind == "system_event":
        # Wrap with source prefix so the in-band context is
        # distinguishable from real user input. Per §11.4.
        source = entry.get("source", "system") or "system"
        text = f"[{source}] {body}"
    else:
        text = body

    blocks: list[dict] = [{"type": "text", "text": text}]

    # Attachments ride alongside text in the same content array
    # (image/document blocks per §11.4 attachments rule). Only
    # user-kind entries carry attachments in v0.9.2; assistant
    # attachments are out of scope for v0.9.2 (depends on
    # Anthropic's per-model assistant-image acceptance).
    if kind == "user":
        for att in entry.get("attachments") or ():
            media_type = (att.get("media_type") or "").lower()
            source = att.get("source") or {}
            if media_type.startswith("image/"):
                blocks.append({
                    "type": "image",
                    "source": source,
                })
            elif media_type == "application/pdf":
                blocks.append({
                    "type": "document",
                    "source": source,
                })
            # Unknown media types are dropped silently — the runtime's
            # append-time validation should have caught them.

    return blocks


def materialize_to_anthropic(entries) -> list[dict]:
    """Translate a Termin conversation field's entry list into
    Anthropic's messages array per §11.4.

    Returns: ``list[{role, content: list[block]}]`` ready to pass to
    ``anthropic.messages.create(messages=...)``.

    Raises ``ConversationMaterializationError`` on:
      - tool_result with tool_call_id that doesn't match a prior
        tool_call entry (orphan).

    Adjacent same-role entries merge into one message with the
    blocks concatenated (Anthropic requires alternating user/
    assistant roles).
    """
    if not entries:
        return []

    seen_tool_call_ids: set[str] = set()
    messages: list[dict] = []

    for entry in entries:
        if not isinstance(entry, dict):
            continue
        kind = entry.get("kind", "")
        if not kind:
            continue

        # Tool linkage validation — done before we map the entry so
        # we don't half-build a turn before failing.
        if kind == "tool_call":
            tcid = entry.get("tool_call_id", "")
            if tcid:
                seen_tool_call_ids.add(tcid)
        elif kind == "tool_result":
            tcid = entry.get("tool_call_id", "")
            if not tcid or tcid not in seen_tool_call_ids:
                raise ConversationMaterializationError(
                    f"tool_result entry references unknown "
                    f"tool_call_id {tcid!r}; no preceding tool_call "
                    f"with that id."
                )

        role = _entry_role(kind)
        blocks = _build_content_blocks(entry)
        if not blocks:
            continue

        if messages and messages[-1]["role"] == role:
            # Adjacent same-role merge.
            messages[-1]["content"].extend(blocks)
        else:
            messages.append({"role": role, "content": list(blocks)})

    return messages


class StreamingJsonFieldExtractor:
    """Parses growing JSON text from a tool-use stream and emits per-field
    events: field_delta (string-value chunk) and field_done (complete
    value with its typed value).

    Design constraints (v0.8 scope):
      - Flat, single-level JSON object (tool inputs for set_output are flat).
      - String-valued fields stream character-by-character as each chunk
        of partial_json arrives. Per-character streaming matches what
        users see in a chat UI.
      - Non-string fields (numbers, booleans, null) are parsed and
        emitted as a single field_done event at finish(). They cannot
        meaningfully stream chars.
      - Escaped quotes (\\") inside strings do not terminate the field.

    The extractor is stateful: feed() appends a chunk, advances a
    scanner, and returns a list of events. finish() flushes any
    remaining completed value and returns any final events (e.g. non-
    string fields that were unparseable until the full object arrived).

    Events are dicts with shape:
      {"type": "field_delta", "field": <name>, "delta": <text chunk>}
      {"type": "field_done",  "field": <name>, "value": <final value>}
    """

    # Scanner states
    _S_SEEK_KEY = 0           # outside a string, looking for next "<key>":
    _S_IN_KEY = 1             # inside the key string
    _S_SEEK_COLON = 2         # after key's closing quote, waiting for :
    _S_SEEK_VALUE = 3         # after colon, waiting for first char of value
    _S_IN_STRING_VALUE = 4    # inside a string value (emit each char as delta)
    _S_IN_OTHER_VALUE = 5     # inside a non-string value (number/bool/null)
    _S_DONE = 99              # after closing brace

    def __init__(self):
        self._buffer = ""
        self._pos = 0
        self._state = self._S_SEEK_KEY
        self._current_key = ""
        self._current_value_chars = []  # for strings: chars emitted so far
        self._value_start_pos = 0       # buffer index where current value began
        self._escape = False            # last char was an unescaped backslash
        # Object-nesting depth so nested objects (e.g. content_create's
        # `data: { body: "..." }`) still stream leaf string fields.
        # Entering "{" in a value context increments; matching "}" at
        # depth>0 returns us to SEEK_KEY for the enclosing level. At
        # depth 0, a closing "}" terminates the object (S_DONE).
        self._depth = 0

    def feed(self, chunk: str) -> list:
        """Append chunk and emit any events this advance uncovered."""
        if not chunk:
            return []
        self._buffer += chunk
        return self._scan()

    def finish(self) -> list:
        """Flush any remaining non-string fields by parsing the
        accumulated buffer as JSON. String fields that were already
        closed have emitted their field_done events; numeric/boolean
        fields emit theirs here.

        Returns events that weren't already emitted via feed()."""
        events = []
        # Try to parse the buffer as JSON and emit field_done for any
        # non-string field that isn't already emitted.
        try:
            # Strip anything after the closing brace (e.g., trailing
            # whitespace from a slow event loop). Simple bracketing.
            text = self._buffer.strip()
            # Permissive close: append } if unbalanced (we already saw
            # all string value closures, but an abrupt stream end might
            # leave us at S_IN_OTHER_VALUE).
            open_braces = text.count("{") - text.count("}")
            if open_braces > 0:
                text = text + ("}" * open_braces)
            parsed = json.loads(text)
            if isinstance(parsed, dict):
                # For each field NOT of string type and NOT already
                # emitted, emit a field_done.
                for k, v in parsed.items():
                    if isinstance(v, str):
                        continue  # string fields already emitted via scan
                    events.append({"type": "field_done",
                                    "field": k, "value": v})
        except json.JSONDecodeError:
            pass  # nothing more to emit
        return events

    def parsed_object(self) -> dict:
        """Return the fully parsed object if buffer is complete JSON;
        {} otherwise. Used for the invocation-done event's output dict."""
        try:
            text = self._buffer.strip()
            open_braces = text.count("{") - text.count("}")
            if open_braces > 0:
                text = text + ("}" * open_braces)
            parsed = json.loads(text)
            return parsed if isinstance(parsed, dict) else {}
        except json.JSONDecodeError:
            return {}

    def _scan(self) -> list:
        """Advance the scanner through newly-appended text. Returns
        events emitted during this scan."""
        events = []
        while self._pos < len(self._buffer):
            c = self._buffer[self._pos]

            if self._state == self._S_SEEK_KEY:
                if c == '"':
                    self._current_key = ""
                    self._state = self._S_IN_KEY
                elif c == "{":
                    # Outer-level open brace or a nested object's open.
                    self._depth += 1
                elif c == "}":
                    self._depth -= 1
                    if self._depth <= 0:
                        self._state = self._S_DONE
                # skip whitespace, commas
                self._pos += 1

            elif self._state == self._S_IN_KEY:
                if self._escape:
                    self._current_key += c
                    self._escape = False
                    self._pos += 1
                elif c == "\\":
                    self._escape = True
                    self._pos += 1
                elif c == '"':
                    self._state = self._S_SEEK_COLON
                    self._pos += 1
                else:
                    self._current_key += c
                    self._pos += 1

            elif self._state == self._S_SEEK_COLON:
                if c == ":":
                    self._state = self._S_SEEK_VALUE
                self._pos += 1  # skip whitespace until colon, then past colon

            elif self._state == self._S_SEEK_VALUE:
                if c.isspace():
                    self._pos += 1
                elif c == '"':
                    # String value starts. Next chars stream as delta.
                    self._current_value_chars = []
                    self._state = self._S_IN_STRING_VALUE
                    self._pos += 1
                elif c == "{":
                    # Nested-object value. Descend — subsequent keys
                    # will be processed as if at the current level,
                    # and leaf-key string fields still stream. We
                    # lose the outer key context (no path prefix),
                    # which keeps the client-side "match by leaf
                    # field name" semantics simple.
                    self._depth += 1
                    self._state = self._S_SEEK_KEY
                    self._pos += 1
                else:
                    # Non-string primitive (number/bool/null) or array.
                    # Don't stream chars; the field lands in finish()
                    # after full parse.
                    self._state = self._S_IN_OTHER_VALUE
                    self._value_start_pos = self._pos
                    self._pos += 1

            elif self._state == self._S_IN_STRING_VALUE:
                if self._escape:
                    # Previous char was backslash. Interpret the escape.
                    decoded = self._decode_escape(c)
                    self._current_value_chars.append(decoded)
                    events.append({"type": "field_delta",
                                    "field": self._current_key,
                                    "delta": decoded})
                    self._escape = False
                    self._pos += 1
                elif c == "\\":
                    self._escape = True
                    self._pos += 1
                elif c == '"':
                    # End of string value — emit field_done.
                    final_value = "".join(self._current_value_chars)
                    events.append({"type": "field_done",
                                    "field": self._current_key,
                                    "value": final_value})
                    self._current_key = ""
                    self._current_value_chars = []
                    self._state = self._S_SEEK_KEY
                    self._pos += 1
                else:
                    self._current_value_chars.append(c)
                    events.append({"type": "field_delta",
                                    "field": self._current_key,
                                    "delta": c})
                    self._pos += 1

            elif self._state == self._S_IN_OTHER_VALUE:
                # Advance until we hit a separator (',' or '}') at the
                # current depth. No delta events emitted here; finish()
                # parses these fields from the buffer.
                if c == ",":
                    self._state = self._S_SEEK_KEY
                    self._pos += 1
                elif c == "}":
                    self._depth -= 1
                    if self._depth <= 0:
                        self._state = self._S_DONE
                    else:
                        # Closed a nested primitive value; back to
                        # SEEK_KEY at the enclosing level.
                        self._state = self._S_SEEK_KEY
                    self._pos += 1
                else:
                    self._pos += 1

            elif self._state == self._S_DONE:
                # Buffer past the object — skip whitespace etc.
                self._pos += 1
            else:
                self._pos += 1

        return events

    @staticmethod
    def _decode_escape(c: str) -> str:
        """Minimal JSON escape decoding for common cases."""
        return {"n": "\n", "t": "\t", "r": "\r", "\"": "\"",
                "\\": "\\", "/": "/", "b": "\b", "f": "\f"}.get(c, c)


class AIProvider:
    """Manages LLM API calls for Compute execution."""

    def __init__(self, deploy_config: dict):
        self._config = deploy_config.get("ai_provider", {})
        self._service = self._config.get("service", "")
        self._model = self._config.get("model", "")
        self._api_key = self._config.get("api_key", "")
        self._client = None

    @property
    def is_configured(self) -> bool:
        return bool(self._service and self._api_key and "${" not in self._api_key)

    @property
    def service(self) -> str:
        return self._service

    @property
    def model(self) -> str:
        return self._model

    def startup(self):
        """Initialize the LLM client."""
        if not self.is_configured:
            if not self._service:
                logger.warning("AI provider not configured — no 'ai_provider' section in deploy config")
            elif not self._api_key:
                logger.warning(f"AI provider '{self._service}' has no API key — set the environment variable (e.g., ANTHROPIC_API_KEY)")
            elif "${" in self._api_key:
                # Show the variable name but not the value
                import re
                var_match = re.search(r'\$\{([^}]+)\}', self._api_key)
                var_name = var_match.group(1) if var_match else "unknown"
                logger.warning(f"AI provider API key contains unresolved variable ${{{var_name}}} — export it in your shell: export {var_name}=\"your-key\"")
            elif "\n" in self._api_key or "\r" in self._api_key:
                logger.error(f"AI provider API key contains a newline character — check your .bashrc or environment for a line break in the key")
            else:
                logger.warning("AI provider not configured — LLM Computes will be skipped")
            return

        if self._service == "anthropic":
            try:
                import anthropic
                self._client = anthropic.Anthropic(api_key=self._api_key)
                logger.info(f"Anthropic client initialized (model: {self._model})")
            except ImportError:
                logger.error("anthropic package not installed. Run: pip install anthropic")
        elif self._service == "openai":
            try:
                import openai
                self._client = openai.OpenAI(api_key=self._api_key)
                logger.info(f"OpenAI client initialized (model: {self._model})")
            except ImportError:
                logger.error("openai package not installed. Run: pip install openai")
        else:
            logger.error(f"Unknown AI provider service: {self._service}")

    async def stream_complete(self, system_prompt: str, user_message: str):
        """Yield (delta, done) tuples as the LLM generates text.

        Text-only streaming (no tool_use). Intended for chat-style
        Computes where the LLM produces free-form text that a client
        renders token-by-token via the compute.stream.<invocation_id>
        event channel (see docs/termin-streaming-protocol.md).

        Contract matches simulate_stream:
          - Every delta except the last yields done=False.
          - The last delta yields done=True with its content.
          - Empty response yields ("", True) as a single terminal event.

        Raises AIProviderError on API failure (init issue or mid-stream
        error from the provider). The caller's invocation record should
        be updated appropriately.
        """
        if not self._client:
            raise AIProviderError("AI provider not initialized")
        if self._service == "anthropic":
            async for item in self._anthropic_stream(system_prompt, user_message):
                yield item
        elif self._service == "openai":
            async for item in self._openai_stream(system_prompt, user_message):
                yield item
        else:
            raise AIProviderError(
                f"Unknown service for stream_complete: {self._service}")

    async def _bridge_sync_stream_to_queue(self, producer_fn):
        """Shared pattern: run a sync producer in a thread that pushes
        text chunks onto an asyncio.Queue, then consume with
        lookahead-by-one so we can mark the last chunk with done=True.

        producer_fn is a callable taking (queue, sentinel, exception_putter)
        where exception_putter(exc) thread-safely forwards an exception
        to the consuming coroutine.

        Yields (delta, done) tuples conforming to the stream_complete
        contract.
        """
        import asyncio
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue = asyncio.Queue()
        SENTINEL = object()

        def _thread_safe_put(item):
            # Schedule queue.put on the consumer's loop.
            asyncio.run_coroutine_threadsafe(queue.put(item), loop).result()

        def _runner():
            try:
                producer_fn(_thread_safe_put)
            except Exception as exc:
                _thread_safe_put(exc)
            finally:
                _thread_safe_put(SENTINEL)

        # Run the producer in a thread so the event loop stays responsive
        # while we wait for each chunk from the provider's sync iterator.
        producer_task = asyncio.create_task(asyncio.to_thread(_runner))

        prev_chunk = None
        try:
            while True:
                item = await queue.get()
                if item is SENTINEL:
                    break
                if isinstance(item, Exception):
                    raise AIProviderError(f"stream error: {item}") from item
                if prev_chunk is not None:
                    yield (prev_chunk, False)
                prev_chunk = item
        finally:
            # Ensure the producer task doesn't leak.
            if not producer_task.done():
                await producer_task

        if prev_chunk is not None:
            yield (prev_chunk, True)
        else:
            # No text chunks received — emit a single terminal event.
            yield ("", True)

    async def _anthropic_stream(self, system_prompt: str, user_message: str):
        """Bridge Anthropic's sync messages.stream context manager into
        our async (delta, done) contract."""
        def producer(put):
            with self._client.messages.stream(
                model=self._model or "claude-sonnet-4-6",
                max_tokens=4096,
                system=system_prompt,
                messages=[{"role": "user", "content": user_message}],
            ) as stream:
                for text in stream.text_stream:
                    if text:
                        put(text)

        async for item in self._bridge_sync_stream_to_queue(producer):
            yield item

    async def stream_agent_response(self, system_prompt: str,
                                    user_message: str, output_tool: dict):
        """Yield tool-use field events as an agent generates its
        set_output tool call.

        This is the primary streaming path for agent Computes —
        agents respond exclusively via set_output and cannot emit
        free-form text as their "response."

        Event shapes (see docs/termin-streaming-protocol.md):
          {"type": "field_delta", "field": <name>, "delta": <text>}
          {"type": "field_done",  "field": <name>, "value": <final>}
          {"type": "done",        "output": <full tool-call input dict>}

        String-valued fields stream character-by-character. Non-string
        fields (number, bool, null) land as a single field_done event
        when the tool-use content block completes.

        Raises AIProviderError on init issues or mid-stream failure.
        """
        if not self._client:
            raise AIProviderError("AI provider not initialized")
        if self._service == "anthropic":
            async for item in self._anthropic_agent_stream(
                    system_prompt, user_message, output_tool):
                yield item
        elif self._service == "openai":
            async for item in self._openai_agent_stream(
                    system_prompt, user_message, output_tool):
                yield item
        else:
            raise AIProviderError(
                f"Unknown service for stream_agent_response: {self._service}")

    async def agent_loop_streaming(self, system_prompt: str,
                                    user_message: str, tools: list,
                                    execute_tool,
                                    on_event=None,
                                    max_turns: int = 20):
        """Streaming-capable agent loop.

        Like agent_loop(): iterates LLM calls + tool execution until
        the agent invokes set_output, then returns the set_output
        input dict.

        Unlike agent_loop(): each turn uses the provider's streaming
        API. For set_output tool calls specifically, accumulates
        input_json_delta events through StreamingJsonFieldExtractor
        and fires on_event for each field_delta / field_done / done
        event. Other tool calls (content_query, content_create, etc.)
        execute as before — non-user-visible.

        on_event is an async callable. It is the bridge to the event
        bus, letting the compute_runner publish stream events to WS
        subscribers without tangling the provider with runtime
        concerns.

        Only Anthropic is supported in v0.8; OpenAI agent-loop
        streaming follows the same contract and can be added without
        changing callers.

        Raises AIProviderError on failure.
        """
        if not self._client:
            raise AIProviderError("AI provider not initialized")
        if self._service == "anthropic":
            return await self._anthropic_agent_loop_streaming(
                system_prompt, user_message, tools, execute_tool,
                on_event, max_turns)
        elif self._service == "openai":
            # Fall back to non-streaming agent_loop; emit a synthetic
            # terminal event with the final output so the client path
            # still works.
            result = await self.agent_loop(system_prompt, user_message,
                                            tools, execute_tool)
            if on_event:
                await on_event({"type": "done", "output": result})
            return result
        else:
            raise AIProviderError(
                f"Unknown service for agent_loop_streaming: {self._service}")

    async def _anthropic_agent_loop_streaming(self, system_prompt: str,
                                               user_message: str,
                                               tools: list,
                                               execute_tool,
                                               on_event,
                                               max_turns: int):
        """Anthropic streaming agent-loop implementation.

        Pattern per turn:
          1. Open messages.stream with tools.
          2. Iterate events in a producer thread:
             - content_block_start for tool_use: if set_output, create
               a per-block extractor; otherwise mark the block as
               non-streaming.
             - content_block_delta with input_json_delta on a
               set_output block: feed the extractor and push events
               to on_event.
          3. After the stream closes, get_final_message and process
             tool_use content blocks: set_output returns; other tools
             execute + feed back.
          4. Repeat until set_output is called or max_turns reached.
        """
        import json as _json
        messages = [{"role": "user", "content": user_message}]

        for turn in range(max_turns):
            # Per-turn state: one extractor per tool_use content block
            # (keyed by block index) so concurrent tool calls in a
            # single response each stream their fields independently.
            # Previously we only streamed set_output blocks, but agent
            # chat (agent_chatbot) uses content_create to persist each
            # reply — the user-visible "message" is the `body` inside
            # content_create's `data` input. Streaming every tool_use
            # block covers that case and generalizes for other tools
            # whose input has a field the chat UI wants to render.
            state = {
                "final_message": None,
                "extractors": {},        # block_index -> extractor
                "block_tool_name": {},   # block_index -> tool name
                "error": None,
            }

            def producer(put):
                try:
                    with self._client.messages.stream(
                        model=self._model or "claude-sonnet-4-6",
                        max_tokens=4096,
                        system=system_prompt,
                        messages=messages,
                        tools=tools,
                    ) as stream:
                        for event in stream:
                            etype = getattr(event, "type", None)
                            if etype == "content_block_start":
                                idx = getattr(event, "index", 0)
                                cb = getattr(event, "content_block", None)
                                if cb and getattr(cb, "type", None) == "tool_use":
                                    tool_name = getattr(cb, "name", "") or ""
                                    state["extractors"][idx] = \
                                        StreamingJsonFieldExtractor()
                                    state["block_tool_name"][idx] = tool_name
                            elif etype == "content_block_delta":
                                idx = getattr(event, "index", 0)
                                ex = state["extractors"].get(idx)
                                if ex is None:
                                    continue  # not a tool_use block
                                delta = getattr(event, "delta", None)
                                if delta is None:
                                    continue
                                if getattr(delta, "type", None) != "input_json_delta":
                                    continue
                                partial = getattr(delta, "partial_json", "") or ""
                                tool_name = state["block_tool_name"].get(idx, "")
                                for emitted in ex.feed(partial):
                                    # Tag the event with the source tool
                                    # so the compute_runner knows which
                                    # channel/tool label to publish on.
                                    emitted = dict(emitted)
                                    emitted["tool"] = tool_name
                                    put(emitted)
                            elif etype == "content_block_stop":
                                idx = getattr(event, "index", 0)
                                ex = state["extractors"].get(idx)
                                if ex is not None:
                                    tool_name = state["block_tool_name"].get(idx, "")
                                    for emitted in ex.finish():
                                        emitted = dict(emitted)
                                        emitted["tool"] = tool_name
                                        put(emitted)
                        # Stream closed — capture the final assembled message.
                        state["final_message"] = stream.get_final_message()
                except Exception as exc:
                    state["error"] = exc

            # Consume events produced during this turn.
            async for item in self._bridge_events_to_queue(producer):
                if on_event:
                    await on_event(item)

            if state["error"] is not None:
                raise AIProviderError(
                    f"Anthropic stream error on turn {turn}: {state['error']}"
                ) from state["error"]

            final_msg = state["final_message"]
            if final_msg is None:
                raise AIProviderError(
                    f"Anthropic stream produced no final message on turn {turn}")

            # Inspect the final message for tool_use blocks.
            tool_calls = [b for b in getattr(final_msg, "content", [])
                          if getattr(b, "type", None) == "tool_use"]

            if not tool_calls:
                # Agent finished without calling set_output — fall back.
                text_blocks = [b.text for b in final_msg.content
                               if getattr(b, "type", None) == "text"]
                result = {
                    "thinking": " ".join(text_blocks),
                    "summary": "Agent completed without set_output",
                }
                if on_event:
                    await on_event({"type": "done", "output": result})
                return result

            # Check for set_output among tool calls.
            set_output_call = next(
                (tc for tc in tool_calls if getattr(tc, "name", "") == "set_output"),
                None,
            )
            if set_output_call is not None:
                # Emit terminal done event. The per-block extractor
                # already flushed its non-string fields at content_block_stop;
                # the `done` event here carries the full set_output input
                # as a canonical snapshot for clients that joined mid-stream.
                if on_event:
                    await on_event(
                        {"type": "done", "output": set_output_call.input})
                return set_output_call.input

            # Non-set_output tools — execute, feed results back, iterate.
            messages.append({"role": "assistant",
                              "content": final_msg.content})
            tool_results = []
            for tc in tool_calls:
                try:
                    result = await execute_tool(tc.name, tc.input)
                    content_str = (_json.dumps(result)
                                   if isinstance(result, (dict, list))
                                   else str(result))
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": tc.id,
                        "content": content_str,
                    })
                except Exception as exc:
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": tc.id,
                        "content": f"Error: {exc}",
                        "is_error": True,
                    })
            messages.append({"role": "user", "content": tool_results})

        raise AIProviderError(f"Agent exceeded maximum turns ({max_turns})")

    async def _anthropic_agent_stream(self, system_prompt: str,
                                       user_message: str,
                                       output_tool: dict):
        """Bridge Anthropic's streaming events into field-level
        (delta, done) events. Listens for input_json_delta events on
        the set_output tool's content block and pipes partial_json
        through a StreamingJsonFieldExtractor."""
        extractor = StreamingJsonFieldExtractor()

        def producer(put):
            with self._client.messages.stream(
                model=self._model or "claude-sonnet-4-6",
                max_tokens=4096,
                system=system_prompt,
                messages=[{"role": "user", "content": user_message}],
                tools=[output_tool],
                tool_choice={"type": "tool", "name": "set_output"},
            ) as stream:
                for event in stream:
                    etype = getattr(event, "type", None)
                    if etype == "content_block_delta":
                        delta_obj = getattr(event, "delta", None)
                        if delta_obj is None:
                            continue
                        dtype = getattr(delta_obj, "type", None)
                        if dtype == "input_json_delta":
                            partial = getattr(delta_obj, "partial_json", "") or ""
                            for emitted in extractor.feed(partial):
                                put(emitted)

        async for item in self._bridge_events_to_queue(producer):
            yield item

        # After the stream closes, flush any non-string fields.
        for e in extractor.finish():
            yield e
        # Emit the final invocation-done event with the full output dict.
        yield {"type": "done", "output": extractor.parsed_object()}

    async def _openai_agent_stream(self, system_prompt: str,
                                    user_message: str,
                                    output_tool: dict):
        """Bridge OpenAI streaming function_call.arguments deltas into
        the same field-level event contract as Anthropic."""
        extractor = StreamingJsonFieldExtractor()
        oai_tool = {
            "type": "function",
            "function": {
                "name": output_tool["name"],
                "description": output_tool.get("description", ""),
                "parameters": output_tool["input_schema"],
            },
        }

        def producer(put):
            stream = self._client.chat.completions.create(
                model=self._model or "gpt-4o",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message},
                ],
                tools=[oai_tool],
                tool_choice={"type": "function",
                             "function": {"name": "set_output"}},
                stream=True,
            )
            for chunk in stream:
                choices = getattr(chunk, "choices", None) or []
                if not choices:
                    continue
                delta = getattr(choices[0], "delta", None)
                if not delta:
                    continue
                tool_calls = getattr(delta, "tool_calls", None) or []
                for tc in tool_calls:
                    fn = getattr(tc, "function", None)
                    if not fn:
                        continue
                    args_chunk = getattr(fn, "arguments", None)
                    if args_chunk:
                        for emitted in extractor.feed(args_chunk):
                            put(emitted)

        async for item in self._bridge_events_to_queue(producer):
            yield item

        for e in extractor.finish():
            yield e
        yield {"type": "done", "output": extractor.parsed_object()}

    async def _bridge_events_to_queue(self, producer_fn):
        """Pump arbitrary event dicts from a sync producer thread into
        an async generator. Similar to _bridge_sync_stream_to_queue but
        yields whatever objects the producer puts on the queue (not
        lookahead-paired tuples). Used by the agent-streaming path
        where the contract is {"type": ..., ...} dicts, not (delta, done).
        """
        import asyncio
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue = asyncio.Queue()
        SENTINEL = object()

        def _thread_safe_put(item):
            asyncio.run_coroutine_threadsafe(queue.put(item), loop).result()

        def _runner():
            try:
                producer_fn(_thread_safe_put)
            except Exception as exc:
                _thread_safe_put(exc)
            finally:
                _thread_safe_put(SENTINEL)

        producer_task = asyncio.create_task(asyncio.to_thread(_runner))
        try:
            while True:
                item = await queue.get()
                if item is SENTINEL:
                    break
                if isinstance(item, Exception):
                    raise AIProviderError(f"stream error: {item}") from item
                yield item
        finally:
            if not producer_task.done():
                await producer_task

    async def _openai_stream(self, system_prompt: str, user_message: str):
        """Bridge OpenAI's stream=True chat completions into our async
        (delta, done) contract. Role-only chunks (delta.content is None)
        are filtered."""
        def producer(put):
            stream = self._client.chat.completions.create(
                model=self._model or "gpt-4o",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message},
                ],
                stream=True,
            )
            for chunk in stream:
                # Each chunk has choices[0].delta.content which may be
                # None (the opening role-only chunk) or a text string.
                choices = getattr(chunk, "choices", None) or []
                if not choices:
                    continue
                delta = getattr(choices[0], "delta", None)
                content = getattr(delta, "content", None) if delta else None
                if content:
                    put(content)

        async for item in self._bridge_sync_stream_to_queue(producer):
            yield item

    async def simulate_stream(self, deltas: list):
        """Test helper: yield (delta, done) from a scripted list.

        Shape matches what a real streaming provider (e.g., Anthropic
        messages.stream) produces. Empty input yields a single terminal
        event with empty text, so callers never need to special-case
        zero-delta completions.

        Used by tests and by dev-mode to exercise the streaming protocol
        without a live LLM API. The real stream_complete() implementation
        (v0.8.1) will conform to the same yield shape.
        """
        if not deltas:
            yield ("", True)
            return
        last_idx = len(deltas) - 1
        for i, delta in enumerate(deltas):
            yield (delta, i == last_idx)

    async def complete(self, system_prompt: str, user_message: str,
                       output_tool: dict) -> dict:
        """Level 1: Single completion with forced tool_use output.

        Args:
            system_prompt: The Directive (system message).
            user_message: The Objective with input field values interpolated.
            output_tool: The auto-generated set_output tool schema.

        Returns:
            Dict with 'thinking' and output field values from the tool call.

        Raises:
            AIProviderError on failure.
        """
        if not self._client:
            raise AIProviderError("AI provider not initialized")

        if self._service == "anthropic":
            return await self._anthropic_complete(system_prompt, user_message, output_tool)
        elif self._service == "openai":
            return await self._openai_complete(system_prompt, user_message, output_tool)
        else:
            raise AIProviderError(f"Unknown service: {self._service}")

    async def agent_loop_with_conversation(
        self,
        system_prompt: str,
        messages: list[dict],
        tools: list[dict],
        execute_tool: Any,
        on_writeback: Any,
        on_text_delta: Any = None,
        on_text_end: Any = None,
        should_halt: Any = None,
        on_event: Any = None,
        max_turns: int = 20,
    ) -> dict:
        """v0.9.2 §11.5: agent loop driven by a pre-materialized
        Anthropic-shape ``messages`` array (not a single user_message
        string). Calls ``on_writeback(kind=..., body=..., **fields)``
        per source-order action so the runtime can append each
        tool_call / tool_result / final assistant text back to the
        conversation field.

        No forced ``set_output``: the agent communicates by ending its
        turn naturally. The final assistant text is delivered via
        ``on_writeback(kind="assistant", body=text)``.

        ``on_writeback`` is async; the runtime awaits it for each
        action so write-back ordering matches source order. The
        runtime is responsible for setting parent_id on the persisted
        entries — the provider supplies tool_call_id / tool_name /
        tool_args (for tool_call) or tool_call_id / is_error (for
        tool_result) and lets the runtime layer in metadata.

        v0.9.2 streaming (post-close-out):
          - ``on_text_delta(text: str)`` (optional, async): fires per
            text token as the agent streams its response. Lets the
            runtime publish in-flight text on a streaming channel
            so chat UIs can render token-by-token without waiting
            for the turn to commit.
          - ``on_text_end(committed: bool)`` (optional, async): fires
            once per turn after the stream closes. ``committed=True``
            means the runtime called ``on_writeback(kind="assistant",
            ...)`` with the full text — the chat UI's pending bubble
            should wait for the matching `appended` event to swap
            for the persisted entry. ``committed=False`` means the
            turn ended with tool calls (no final assistant text
            committed yet) — the chat UI should clear any pending
            bubble; subsequent turns may resume streaming new text.

        Returns: ``{"thinking": "...", "summary": "..."}`` for audit /
        legacy-compat consumers.
        """
        if not self._client:
            raise AIProviderError("AI provider not initialized")
        if self._service == "anthropic":
            return await self._anthropic_agent_loop_with_conversation(
                system_prompt, messages, tools, execute_tool,
                on_writeback, on_text_delta, on_text_end,
                should_halt, on_event, max_turns,
            )
        # OpenAI conversation-mode is out of v0.9.2 scope; the
        # analyzer-side check (TERMIN-S061 etc) doesn't gate on
        # provider service, so adding OpenAI is a future-friendly
        # extension. For now, refuse loudly.
        raise AIProviderError(
            f"agent_loop_with_conversation: {self._service!r} not "
            f"yet implemented; v0.9.2 ships Anthropic only."
        )

    async def _anthropic_agent_loop_with_conversation(
        self,
        system_prompt: str,
        messages: list[dict],
        tools: list[dict],
        execute_tool,
        on_writeback,
        on_text_delta,
        on_text_end,
        should_halt,
        on_event,
        max_turns: int,
    ) -> dict:
        """Anthropic implementation of ``agent_loop_with_conversation``.

        Per turn:
          1. Open ``messages.stream`` with the running messages list.
             Iterate stream events:
               - text_delta events fire on_text_delta(delta) so the
                 runtime can publish in-flight text on the streaming
                 channel. Accumulate locally for the eventual
                 commit.
               - input_json_delta and other tool-use events are
                 collected by the SDK; final assembled tool_use
                 blocks are read from get_final_message().
          2. For each tool_use block in the final message: emit
             on_writeback("tool_call", body=f"{name}({json.dumps(args)})",
             tool_call_id, tool_name, tool_args, purpose), execute
             via ``execute_tool``, emit on_writeback("tool_result",
             body=result_str, tool_call_id, is_error), append the
             assistant turn and tool_results to the local messages
             array. Fire on_text_end(committed=False) — the text the
             agent streamed during this turn was "thinking", not a
             commit-bound reply; the chat UI clears any pending
             bubble.
          3. For text-only end of turn (no tool calls): emit
             on_writeback("assistant", body=accumulated_text) and
             on_text_end(committed=True). The chat UI's pending
             bubble waits for the matching `appended` event and
             swaps for the persisted entry.
        """
        import anthropic
        running_messages = list(messages)

        for turn in range(max_turns):
            # v0.9.2 close-out: refusal-terminates-loop. The
            # runtime supplies `should_halt` as a closure over its
            # `refusal_state`; if system_refuse fired during a
            # prior turn, we stop here instead of issuing another
            # provider call. The post-loop refusal-append path
            # (L7.4) writes the refusal entry as the last commit
            # on the conversation field. Per compute-contract.md
            # §6.1: "the loop terminates; staged outputs are
            # discarded."
            if should_halt is not None:
                halt = should_halt() if not asyncio.iscoroutinefunction(
                    should_halt) else await should_halt()
                if halt:
                    return {
                        "thinking": "",
                        "summary": "halted (refused)",
                    }
            # Per-turn state. The producer thread iterates the
            # synchronous Anthropic stream and pushes text_delta
            # events onto a queue; the async consumer (us) drains
            # it and dispatches each delta to on_text_delta. After
            # the producer signals SENTINEL we have the final
            # assembled message via stream.get_final_message().
            state = {"final_message": None, "error": None}
            accumulated_text = ""

            def producer(put):
                try:
                    with self._client.messages.stream(
                        model=self._model or "claude-sonnet-4-6",
                        max_tokens=4096,
                        system=system_prompt,
                        messages=running_messages,
                        tools=tools,
                    ) as stream:
                        for event in stream:
                            etype = getattr(event, "type", None)
                            if etype != "content_block_delta":
                                continue
                            delta = getattr(event, "delta", None)
                            if delta is None:
                                continue
                            if getattr(delta, "type", None) == "text_delta":
                                text = getattr(delta, "text", "") or ""
                                if text:
                                    put({"type": "text_delta", "text": text})
                        state["final_message"] = stream.get_final_message()
                except Exception as exc:
                    state["error"] = exc

            try:
                async for item in self._bridge_events_to_queue(producer):
                    if item.get("type") == "text_delta":
                        text = item.get("text", "")
                        accumulated_text += text
                        if on_text_delta:
                            await on_text_delta(text)
            except AIProviderError as e:
                raise AIProviderError(
                    f"Anthropic streaming error on turn {turn}: {e}"
                )

            if state["error"] is not None:
                raise AIProviderError(
                    f"Anthropic streaming error on turn {turn}: "
                    f"{state['error']}"
                ) from state["error"]

            final_msg = state["final_message"]
            if final_msg is None:
                raise AIProviderError(
                    f"Anthropic stream produced no final message on "
                    f"turn {turn}"
                )

            tool_calls = [b for b in final_msg.content
                          if getattr(b, "type", None) == "tool_use"]

            if not tool_calls:
                # End-of-turn: commit the streamed text as the
                # assistant entry. Empty text yields no entry — the
                # caller can decide whether that's an error.
                final_text = accumulated_text.strip()
                if final_text:
                    await on_writeback(
                        kind="assistant", body=final_text,
                    )
                    if on_text_end:
                        await on_text_end(committed=True)
                else:
                    if on_text_end:
                        await on_text_end(committed=False)
                return {
                    "thinking": final_text,
                    "summary": "completed",
                }

            # Mid-turn: tool calls are present. Tell the chat UI the
            # streamed text on this turn was "thinking" — clear any
            # pending bubble before tool entries start arriving.
            if on_text_end:
                await on_text_end(committed=False)
            # Write each tool_call, execute it, write each tool_result,
            # then loop. The final_msg.content (which may interleave
            # text + tool_use blocks) goes into the assistant turn so
            # Anthropic sees its own thinking back on the next call.
            running_messages.append({
                "role": "assistant", "content": final_msg.content,
            })
            tool_results = []
            for tc in tool_calls:
                # Mid-turn halt check: if a prior tool call in this
                # same response (e.g., system_refuse) flipped the
                # halt signal, drop the rest of the batch. The
                # post-loop refusal-append path still writes the
                # refusal entry as the last commit on the field.
                if should_halt is not None:
                    halt = should_halt() if not asyncio.iscoroutinefunction(
                        should_halt) else await should_halt()
                    if halt:
                        return {
                            "thinking": "",
                            "summary": "halted (refused)",
                        }
                raw_args = tc.input if isinstance(tc.input, dict) else {}
                # v0.9.2 close-out: extract `purpose` (if supplied)
                # before passing args to the tool. The runtime
                # truncates per spec and persists on the tool_call
                # entry; the tool itself doesn't see `purpose`.
                purpose_raw = raw_args.get("purpose") if isinstance(raw_args, dict) else None
                tool_args = {
                    k: v for k, v in raw_args.items() if k != "purpose"
                } if isinstance(raw_args, dict) else {}
                summary_body = f"{tc.name}({json.dumps(tool_args)})"
                writeback_kwargs = {
                    "tool_call_id": tc.id,
                    "tool_name": tc.name,
                    "tool_args": tool_args,
                }
                if purpose_raw:
                    writeback_kwargs["purpose"] = _truncate_purpose(
                        str(purpose_raw)
                    )
                await on_writeback(
                    kind="tool_call",
                    body=summary_body,
                    **writeback_kwargs,
                )
                try:
                    result = await execute_tool(tc.name, tool_args)
                    content_str = (
                        json.dumps(result)
                        if isinstance(result, (dict, list))
                        else str(result)
                    )
                    await on_writeback(
                        kind="tool_result",
                        body=content_str,
                        tool_call_id=tc.id,
                    )
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": tc.id,
                        "content": content_str,
                    })
                except Exception as exc:
                    err_str = f"Error: {exc}"
                    await on_writeback(
                        kind="tool_result",
                        body=err_str,
                        tool_call_id=tc.id,
                        is_error=True,
                    )
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": tc.id,
                        "content": err_str,
                        "is_error": True,
                    })
            running_messages.append({
                "role": "user", "content": tool_results,
            })

        raise AIProviderError(
            f"Conversation agent exceeded maximum turns ({max_turns})"
        )

    async def agent_loop(self, system_prompt: str, user_message: str,
                         tools: list[dict],
                         execute_tool: Any) -> dict:
        """Level 3: Agent loop with tool calls.

        Args:
            system_prompt: Directive + tool descriptions.
            user_message: Objective + triggering record context.
            tools: List of tool schemas (ComputeContext tools + set_output).
            execute_tool: Async callable(tool_name, tool_input) -> result dict.

        Returns:
            The set_output tool call result (thinking + summary).

        Raises:
            AIProviderError on failure.
        """
        if not self._client:
            raise AIProviderError("AI provider not initialized")

        if self._service == "anthropic":
            return await self._anthropic_agent_loop(system_prompt, user_message, tools, execute_tool)
        elif self._service == "openai":
            return await self._openai_agent_loop(system_prompt, user_message, tools, execute_tool)
        else:
            raise AIProviderError(f"Unknown service: {self._service}")

    # ── Anthropic implementation ──

    async def _anthropic_complete(self, system_prompt: str, user_message: str,
                                   output_tool: dict) -> dict:
        import anthropic
        try:
            response = self._client.messages.create(
                model=self._model or "claude-sonnet-4-6",
                max_tokens=4096,
                system=system_prompt,
                messages=[{"role": "user", "content": user_message}],
                tools=[output_tool],
                tool_choice={"type": "tool", "name": "set_output"},
            )
            # Extract tool call result
            for block in response.content:
                if block.type == "tool_use" and block.name == "set_output":
                    return block.input
            raise AIProviderError("LLM did not call set_output tool")
        except anthropic.APIError as e:
            raise AIProviderError(f"Anthropic API error: {e}")

    async def _anthropic_agent_loop(self, system_prompt: str, user_message: str,
                                     tools: list[dict],
                                     execute_tool) -> dict:
        import anthropic
        messages = [{"role": "user", "content": user_message}]
        max_turns = 20

        for turn in range(max_turns):
            try:
                response = self._client.messages.create(
                    model=self._model or "claude-sonnet-4-6",
                    max_tokens=4096,
                    system=system_prompt,
                    messages=messages,
                    tools=tools,
                )
            except anthropic.APIError as e:
                raise AIProviderError(f"Anthropic API error on turn {turn}: {e}")

            # Check if the response has tool calls
            tool_calls = [b for b in response.content if b.type == "tool_use"]
            text_blocks = [b.text for b in response.content if b.type == "text"]

            if text_blocks:
                print(f"[Termin]   Turn {turn} thinking: {text_blocks[0][:100]}...")

            if not tool_calls:
                # Agent finished without calling set_output — extract text
                return {"thinking": " ".join(text_blocks), "summary": "Agent completed without set_output"}

            print(f"[Termin]   Turn {turn}: {len(tool_calls)} tool call(s): {', '.join(tc.name for tc in tool_calls)}")

            # Process tool calls
            messages.append({"role": "assistant", "content": response.content})
            tool_results = []

            for tool_call in tool_calls:
                if tool_call.name == "set_output":
                    # Agent signals completion
                    print(f"[Termin]   Agent called set_output — completing")
                    return tool_call.input

                # Execute the tool via ComputeContext
                try:
                    result = await execute_tool(tool_call.name, tool_call.input)
                    result_summary = json.dumps(result)[:200] if isinstance(result, (dict, list)) else str(result)[:200]
                    print(f"[Termin]     {tool_call.name}() -> {result_summary}")
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": tool_call.id,
                        "content": json.dumps(result) if isinstance(result, (dict, list)) else str(result),
                    })
                except Exception as e:
                    print(f"[Termin]     {tool_call.name}() ERROR: {e}")
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": tool_call.id,
                        "content": f"Error: {e}",
                        "is_error": True,
                    })

            messages.append({"role": "user", "content": tool_results})

            if response.stop_reason == "end_turn":
                return {"thinking": "Agent ended turn", "summary": "Completed"}

        raise AIProviderError(f"Agent exceeded maximum turns ({max_turns})")

    # ── OpenAI implementation ──

    async def _openai_complete(self, system_prompt: str, user_message: str,
                                output_tool: dict) -> dict:
        import openai
        # Convert Anthropic-style tool to OpenAI format
        oai_tool = {
            "type": "function",
            "function": {
                "name": output_tool["name"],
                "description": output_tool.get("description", ""),
                "parameters": output_tool["input_schema"],
            }
        }
        try:
            response = self._client.chat.completions.create(
                model=self._model or "gpt-4o",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message},
                ],
                tools=[oai_tool],
                tool_choice={"type": "function", "function": {"name": "set_output"}},
            )
            # Extract tool call
            choice = response.choices[0]
            if choice.message.tool_calls:
                tc = choice.message.tool_calls[0]
                return json.loads(tc.function.arguments)
            raise AIProviderError("LLM did not call set_output tool")
        except openai.APIError as e:
            raise AIProviderError(f"OpenAI API error: {e}")

    async def _openai_agent_loop(self, system_prompt: str, user_message: str,
                                  tools: list[dict],
                                  execute_tool) -> dict:
        import openai
        # Convert tools to OpenAI format
        oai_tools = []
        for t in tools:
            oai_tools.append({
                "type": "function",
                "function": {
                    "name": t["name"],
                    "description": t.get("description", ""),
                    "parameters": t["input_schema"],
                }
            })

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ]
        max_turns = 20

        for turn in range(max_turns):
            try:
                response = self._client.chat.completions.create(
                    model=self._model or "gpt-4o",
                    messages=messages,
                    tools=oai_tools,
                )
            except openai.APIError as e:
                raise AIProviderError(f"OpenAI API error on turn {turn}: {e}")

            choice = response.choices[0]

            if not choice.message.tool_calls:
                # No tool calls — agent finished
                return {"thinking": choice.message.content or "", "summary": "Completed"}

            messages.append(choice.message)

            for tc in choice.message.tool_calls:
                if tc.function.name == "set_output":
                    return json.loads(tc.function.arguments)

                try:
                    result = await execute_tool(tc.function.name, json.loads(tc.function.arguments))
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": json.dumps(result) if isinstance(result, (dict, list)) else str(result),
                    })
                except Exception as e:
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": f"Error: {e}",
                    })

            if choice.finish_reason == "stop":
                return {"thinking": "Agent stopped", "summary": "Completed"}

        raise AIProviderError(f"Agent exceeded maximum turns ({max_turns})")


def build_output_tool(output_fields: list[tuple[str, str]], content_lookup: dict) -> dict:
    """Build the set_output tool schema from Compute output field declarations.

    Args:
        output_fields: List of (content_ref, field_name) from IR.
        content_lookup: Dict of snake_name -> content schema dict.

    Returns:
        Anthropic-format tool schema dict.
    """
    # Fix 009.3: Don't include 'thinking' unconditionally.
    # It will only appear if the compute's output schema declares a 'thinking' field.
    properties = {}
    required = []

    for content_ref, field_name in output_fields:
        # Resolve content schema to get field type and enum constraints
        # content_ref is the singular (e.g., "completion"), need to find the content
        schema = None
        for name, s in content_lookup.items():
            singular = s.get("singular", "")
            if name == content_ref or singular == content_ref:
                schema = s
                break
        if not schema:
            # Fallback: just use string type
            properties[field_name] = {"type": "string", "description": f"Field: {content_ref}.{field_name}"}
            required.append(field_name)
            continue

        # Find the field definition
        field_def = None
        for f in schema.get("fields", []):
            fname = f.get("name", "")
            if fname == field_name:
                field_def = f
                break

        if field_def:
            prop = {"description": f"Field: {content_ref}.{field_name}"}
            enum_vals = field_def.get("enum_values", [])
            if enum_vals:
                prop["type"] = "string"
                prop["enum"] = list(enum_vals)
            elif field_def.get("column_type") in ("INTEGER", "REAL"):
                prop["type"] = "number"
            elif field_def.get("column_type") == "BOOLEAN":
                prop["type"] = "boolean"
            else:
                prop["type"] = "string"
            properties[field_name] = prop
            required.append(field_name)
        else:
            properties[field_name] = {"type": "string", "description": f"Field: {content_ref}.{field_name}"}
            required.append(field_name)

    return {
        "name": "set_output",
        "description": "Set the output fields for this computation. Always call this tool to provide your response.",
        "input_schema": {
            "type": "object",
            "properties": properties,
            "required": required,
        }
    }


def build_agent_tools(accesses: list[str], content_lookup: dict) -> list[dict]:
    """Build ComputeContext tool schemas for an agent.

    Args:
        accesses: List of content type snake_names the agent can touch.
        content_lookup: Dict of snake_name -> content schema dict.

    Returns:
        List of Anthropic-format tool schemas.
    """
    tools = [
        {
            "name": "content_query",
            "description": "Query records from a content table. Returns a list of records.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "content_name": {
                        "type": "string",
                        "description": f"Content type to query. Allowed: {', '.join(accesses)}",
                        "enum": accesses,
                    },
                    "filters": {
                        "type": "object",
                        "description": "Optional key-value filters (field_name: value).",
                        "additionalProperties": True,
                    },
                },
                "required": ["content_name"],
            },
        },
        {
            "name": "content_create",
            "description": "Create a new record in a content table. Returns the created record with id.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "content_name": {
                        "type": "string",
                        "description": f"Content type to create in. Allowed: {', '.join(accesses)}",
                        "enum": accesses,
                    },
                    "data": {
                        "type": "object",
                        "description": "Field values for the new record.",
                        "additionalProperties": True,
                    },
                },
                "required": ["content_name", "data"],
            },
        },
        {
            "name": "content_update",
            "description": "Update an existing record by id.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "content_name": {
                        "type": "string",
                        "enum": accesses,
                    },
                    "record_id": {"type": "integer", "description": "The record id to update."},
                    "data": {
                        "type": "object",
                        "description": "Fields to update.",
                        "additionalProperties": True,
                    },
                },
                "required": ["content_name", "record_id", "data"],
            },
        },
        {
            "name": "state_transition",
            "description": (
                "Transition a record's state machine to a new state. "
                "If the content has more than one state machine, "
                "machine_name must be supplied to disambiguate which "
                "machine to drive."),
            "input_schema": {
                "type": "object",
                "properties": {
                    "content_name": {"type": "string", "enum": accesses},
                    "record_id": {"type": "integer"},
                    "machine_name": {
                        "type": "string",
                        "description": (
                            "Snake-case name of the state machine to drive "
                            "(also the SQL column). Optional when the "
                            "content has exactly one state machine."),
                    },
                    "target_state": {"type": "string", "description": "The state to transition to."},
                },
                "required": ["content_name", "record_id", "target_state"],
            },
        },
        # v0.9 Phase 3 slice (e): system.refuse always-available tool.
        # The agent calls this when a request conflicts with system
        # policy or training-time restrictions. The runtime captures
        # the call in the audit + compute_refusals sidecar; the
        # invocation outcome becomes "refused" with the supplied
        # reason. Per BRD §6.3.3 + design §3.7.
        {
            "name": "system_refuse",
            "description": (
                "Refuse the requested work because it conflicts with "
                "system policy or training-time constraints. Provide a "
                "clear, structured reason. After calling this you "
                "should still call set_output to terminate the loop "
                "(set_output's content does not matter — the runtime "
                "discards it on refusal)."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "reason": {
                        "type": "string",
                        "description": "Why the work is being refused.",
                    },
                },
                "required": ["reason"],
            },
        },
    ]
    return tools
