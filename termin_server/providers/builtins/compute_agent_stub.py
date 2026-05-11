# Copyright 2026 Jamie-Leigh Blake and Termin project contributors
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0

"""Stub ai-agent compute provider — first-party plugin against the
v0.9 ai-agent contract surface (BRD §6.3.3).

Scripted-script stub for deterministic tests and local development.
Does not call any LLM SDK; instead replays a configured sequence of
tool calls and a final result. The runtime's gate function still
runs against the configured tool calls — so denied tools surface as
ToolNotDeclared even in the stub, which is exactly the conformance
behavior tests want.

Per BRD §10 ("Stub providers required for every contract"), every
named contract ships with a stub product so dev/test deploy configs
can bind to a deterministic implementation.

Configuration shape (deploy_config["bindings"]["compute"]["<name>"]
["config"]):
    {
        "scripts": {
            "<directive_or_objective_substring>": {
                "tool_calls": [
                    {"tool": "content.query", "args": {...}},
                    ...
                ],
                "final_outcome": "success" | "refused" | "error",
                "final_result": <any>,                # if success
                "refusal_reason": "...",              # if refused
                "error_detail": "...",                # if error
                "reasoning_summary": "..."            # optional
            }
        },
        "default_script": {...},                      # optional
        "model_identifier": "stub-agent-1"
    }
"""

from __future__ import annotations

import asyncio
import json
from typing import Any, AsyncIterator, Mapping, Optional

from termin_core.providers.contracts import Category, ContractRegistry
from termin_core.providers.compute_contract import (
    AgentContext, AgentEvent, AgentResult, AuditableAction, AuditRecord,
    Completed, ToolCall, ToolCalled, ToolNotDeclared, ToolResult,
    ToolSurface,
)
from ._provider_hash import hash_provider_config

from termin_server import __version__


class StubAgentProvider:
    """Scripted ai-agent for tests.

    Replays a configured tool-call sequence then emits a final result.
    Each scripted tool call goes through `context.tool_callback` so
    the runtime's gate function is exercised — denied tools surface
    via `ToolNotDeclared`, which the script can then react to (or
    propagate as an error).
    """

    def __init__(self, config: Mapping[str, Any] | None = None) -> None:
        self._config = dict(config or {})
        self._scripts: dict[str, dict] = dict(self._config.get("scripts", {}))
        self._default: Optional[dict] = self._config.get("default_script")
        self._model_id: str = str(
            self._config.get("model_identifier", "stub-agent-1")
        )
        self._config_hash = hash_provider_config(self._config)

    @property
    def is_configured(self) -> bool:
        """v0.9.4 (server issue #1): the stub provider is always
        configured. The compute runner gates execution on this
        property (`compute_runner.py:428,612`); without it,
        `getattr(provider, "is_configured", False)` returns False
        and stub-bound computes are silently skipped at every
        invocation. The whole point of the stub is to exercise
        plumbing without external dependencies — there's nothing
        to wait for, so it's always ready.
        """
        return True

    @property
    def service(self) -> str:
        """v0.9.4 (server issue #1 follow-up): the compute runner
        reads `provider.service` for log lines and to special-case
        Anthropic-vs-other plumbing (`compute_runner.py:438,512,869`).
        Returns "stub" so the log line shape stays consistent and
        the Anthropic-specific paths skip cleanly."""
        return "stub"

    @property
    def model(self) -> str:
        """Surfaces the configured model identifier for log /
        audit. Stub providers default to "stub-agent-1" unless
        overridden via deploy config `model_identifier`."""
        return self._model_id

    @property
    def legacy(self) -> "_StubAgentLegacyAdapter":
        """v0.9.4 (server issue #2): the compute runner routes
        ai-agent computes through ``provider.legacy`` for SDK-
        shaped calls (`agent_loop_with_conversation`,
        `agent_loop_streaming`, `agent_loop`). The stub
        implements the modern Protocol shape (`invoke`,
        `invoke_streaming`); this adapter translates the
        runtime's legacy calls into scripted-script behavior
        so stub-bound ai-agent computes work end-to-end
        without external dependencies.

        Per compute_runner.py:425, ``.legacy`` is slated for
        removal in v0.10 slice (c). The adapter is band-aid
        over architectural drift the codebase already plans to
        fix; when slice (c) lands, the runtime will call the
        modern Protocol directly and this property goes away.
        """
        return _StubAgentLegacyAdapter(self)

    async def invoke(
        self,
        directive: str,
        objective: str,
        context: AgentContext,
        tools: ToolSurface,
    ) -> AgentResult:
        prompt = f"{directive}\n{objective}"
        script = self._match(prompt)
        return await self._run_script(prompt, script, context)

    async def invoke_streaming(
        self,
        directive: str,
        objective: str,
        context: AgentContext,
        tools: ToolSurface,
    ) -> AsyncIterator[AgentEvent]:
        prompt = f"{directive}\n{objective}"
        script = self._match(prompt)
        # Replay tool calls as ToolCalled / ToolResult events, then
        # emit Completed at the end.
        executed: list[ToolCall] = []
        actions: list[AuditableAction] = []
        for idx, call in enumerate(script.get("tool_calls", [])):
            tool = call["tool"]
            args = call.get("args", {})
            call_id = f"call-{idx}"
            yield ToolCalled(tool=tool, args=args, call_id=call_id)
            try:
                result = await _maybe_await(
                    context.tool_callback, tool, args
                ) if context.tool_callback else None
                yield ToolResult(call_id=call_id, result=result, is_error=False)
                executed.append(ToolCall(
                    tool=tool, args=dict(args), result=result,
                    is_error=False, latency_ms=0,
                ))
                actions.append(AuditableAction(
                    tool=tool,
                    target=str(args.get("content_type") or args.get("channel_name") or ""),
                    succeeded=True,
                ))
            except ToolNotDeclared as e:
                yield ToolResult(
                    call_id=call_id, result=str(e), is_error=True,
                )
                executed.append(ToolCall(
                    tool=tool, args=dict(args), result=str(e),
                    is_error=True, latency_ms=0,
                ))
                actions.append(AuditableAction(
                    tool=tool, target=None, succeeded=False,
                ))
        # Build final AgentResult and emit Completed.
        result = self._build_result(prompt, script, executed, actions)
        yield Completed(result=result)

    async def _run_script(
        self, prompt: str, script: dict, context: AgentContext,
    ) -> AgentResult:
        executed: list[ToolCall] = []
        actions: list[AuditableAction] = []
        for call in script.get("tool_calls", []):
            tool = call["tool"]
            args = call.get("args", {})
            try:
                result = await _maybe_await(
                    context.tool_callback, tool, args
                ) if context.tool_callback else None
                executed.append(ToolCall(
                    tool=tool, args=dict(args), result=result,
                    is_error=False, latency_ms=0,
                ))
                actions.append(AuditableAction(
                    tool=tool,
                    target=str(args.get("content_type") or args.get("channel_name") or ""),
                    succeeded=True,
                ))
            except ToolNotDeclared as e:
                executed.append(ToolCall(
                    tool=tool, args=dict(args), result=str(e),
                    is_error=True, latency_ms=0,
                ))
                actions.append(AuditableAction(
                    tool=tool, target=None, succeeded=False,
                ))
        return self._build_result(prompt, script, executed, actions)

    def _build_result(
        self, prompt: str, script: dict,
        executed: list[ToolCall], actions: list[AuditableAction],
    ) -> AgentResult:
        outcome = script.get("final_outcome", "success")
        audit = AuditRecord(
            provider_product="stub",
            model_identifier=self._model_id,
            provider_config_hash=self._config_hash,
            prompt_as_sent=prompt,
            sampling_params={},
            tool_calls=tuple(executed),
            outcome=outcome,
            refusal_reason=script.get("refusal_reason"),
            error_detail=script.get("error_detail"),
            cost=None,
            latency_ms=0,
        )
        return AgentResult(
            outcome=outcome,
            actions_taken=tuple(actions),
            reasoning_summary=script.get("reasoning_summary"),
            refusal_reason=script.get("refusal_reason"),
            error_detail=script.get("error_detail"),
            audit_record=audit,
            output_value=script.get("final_result"),
        )

    def _match(self, prompt: str) -> dict:
        for key, script in self._scripts.items():
            if key in prompt:
                return script
        if self._default is not None:
            return dict(self._default)
        # Empty default: zero tool calls, success outcome.
        return {"final_outcome": "success", "tool_calls": []}


async def _maybe_await(callback, tool: str, args: Mapping[str, Any]):
    """Invoke the gated tool callback. Supports both sync and async
    callbacks for stub flexibility."""
    import inspect
    result = callback(tool, args)
    if inspect.isawaitable(result):
        return await result
    return result


# ── Legacy adapter (v0.9.4 server issue #2) ──


class _StubAgentLegacyAdapter:
    """Adapter that translates the runtime's legacy SDK-shaped calls
    (`agent_loop`, `agent_loop_streaming`,
    `agent_loop_with_conversation`) into scripted-script behavior on
    the wrapped ``StubAgentProvider``.

    Lives here as a band-aid over the compute runner's `.legacy`
    indirection (compute_runner.py:425 — "Slice (b) interim, slice
    (c) deletes `.legacy`"). Once slice (c) lands, the runtime calls
    the modern Protocol (`invoke`, `invoke_streaming`) directly and
    this adapter goes away.

    Behavior shapes the adapter must preserve (matching the Anthropic
    legacy implementation in `ai_provider.py`):

    - `agent_loop_with_conversation` fires on_writeback per scripted
      tool_call (kind="tool_call" with tool_call_id/tool_name/
      tool_args) followed by on_writeback for each tool_result
      (kind="tool_result" with matching tool_call_id, is_error
      reflecting whether execute_tool raised). After all scripted
      calls complete, if the script declares a final body string,
      fires on_writeback(kind="agent", body=<text>) followed by
      on_text_end(committed=True). If no final body, fires
      on_text_end(committed=False).
    - `should_halt()` is checked between tool calls; halts the loop
      when truthy (matches Anthropic legacy behavior at line 1191).
    - All three methods return ``{"thinking": str, "summary": str}``
      for audit consumers (compute_runner.py extracts ``thinking``
      and logs it).
    """

    def __init__(self, provider: "StubAgentProvider") -> None:
        self._provider = provider

    async def agent_loop(
        self,
        system_prompt: str,
        user_message: str,
        tools: list,
        execute_tool: Any,
    ) -> dict:
        """Non-streaming, non-conversation fallback. Runs the
        scripted tool calls sequentially via ``execute_tool`` and
        returns a thinking/summary dict."""
        prompt = f"{system_prompt}\n{user_message}"
        script = self._provider._match(prompt)
        thinking_parts: list[str] = []
        for call in script.get("tool_calls", []):
            tool = call["tool"]
            args = call.get("args", {}) or {}
            try:
                result = await execute_tool(tool, dict(args))
                if result is not None:
                    thinking_parts.append(f"{tool} -> ok")
            except Exception as exc:  # noqa: BLE001
                thinking_parts.append(f"{tool} -> error: {exc}")
        final_body = _extract_final_body(script)
        if final_body:
            thinking_parts.append(final_body)
        return {
            "thinking": "\n".join(thinking_parts),
            "summary": script.get("final_outcome", "success"),
        }

    async def agent_loop_streaming(
        self,
        system_prompt: str,
        user_message: str,
        tools: list,
        execute_tool: Any,
        on_event: Any,
    ) -> dict:
        """Streaming non-conversation entry point. Fires on_event
        events as tools execute (event-bus consumers — compute
        stream channels — render these). Always emits a final
        ``{"type": "done", ...}`` event so subscribers know to
        unsubscribe."""
        prompt = f"{system_prompt}\n{user_message}"
        script = self._provider._match(prompt)
        for idx, call in enumerate(script.get("tool_calls", [])):
            tool = call["tool"]
            args = call.get("args", {}) or {}
            call_id = f"stub-call-{idx}"
            if on_event:
                await on_event({
                    "type": "tool_call",
                    "tool": tool,
                    "args": dict(args),
                    "call_id": call_id,
                })
            try:
                result = await execute_tool(tool, dict(args))
                if on_event:
                    await on_event({
                        "type": "tool_result",
                        "tool": tool,
                        "call_id": call_id,
                        "result": result,
                        "is_error": False,
                    })
            except Exception as exc:  # noqa: BLE001
                if on_event:
                    await on_event({
                        "type": "tool_result",
                        "tool": tool,
                        "call_id": call_id,
                        "result": str(exc),
                        "is_error": True,
                    })
        final_body = _extract_final_body(script)
        result_dict = {
            "thinking": final_body or "",
            "summary": script.get("final_outcome", "success"),
        }
        if isinstance(script.get("final_result"), dict):
            # Merge the script's final_result fields into the output
            # so set_output-style consumers see them.
            for k, v in script["final_result"].items():
                if k not in result_dict:
                    result_dict[k] = v
        if on_event:
            await on_event({"type": "done", "output": result_dict})
        return result_dict

    async def agent_loop_with_conversation(
        self,
        system_prompt: str,
        messages: list,
        tools: list,
        execute_tool: Any,
        on_writeback: Any,
        on_text_delta: Any = None,
        on_text_end: Any = None,
        should_halt: Any = None,
        on_event: Any = None,
        max_turns: int = 20,
    ) -> dict:
        """v0.9.2 §11.5 conversation entry point.

        Per scripted tool_call: fire on_writeback(kind="tool_call",
        ...) → execute_tool → fire on_writeback(kind="tool_result",
        ...). After all tool calls (or after should_halt fires),
        if the script declares a final body, fire on_text_delta
        with the full text + on_writeback(kind="agent", body=text)
        + on_text_end(committed=True). If no final body, fire
        on_text_end(committed=False).
        """
        # Materialize the messages list into a single objective
        # string so the stub's substring matcher can find a script.
        # The user content is what carries the prompt text in the
        # Anthropic message shape.
        objective_parts: list[str] = []
        for msg in messages or []:
            content = msg.get("content")
            if isinstance(content, str):
                objective_parts.append(content)
            elif isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "text":
                        objective_parts.append(block.get("text", ""))
        prompt = f"{system_prompt}\n" + "\n".join(objective_parts)
        script = self._provider._match(prompt)

        for idx, call in enumerate(script.get("tool_calls", []) or []):
            # Halt check before each tool call (matches Anthropic
            # legacy at line 1191; the runtime uses this to short-
            # circuit on system_refuse).
            if should_halt is not None:
                halt_val = should_halt()
                if asyncio.iscoroutine(halt_val):
                    halt_val = await halt_val
                if halt_val:
                    return {
                        "thinking": "",
                        "summary": "halted (refused)",
                    }
            tool = call["tool"]
            args = call.get("args", {}) or {}
            call_id = f"stub-call-{idx}"
            summary_body = f"{tool}({json.dumps(dict(args))})"
            await on_writeback(
                kind="tool_call",
                body=summary_body,
                tool_call_id=call_id,
                tool_name=tool,
                tool_args=dict(args),
            )
            try:
                result = await execute_tool(tool, dict(args))
                content_str = (
                    json.dumps(result)
                    if isinstance(result, (dict, list))
                    else str(result) if result is not None else ""
                )
                await on_writeback(
                    kind="tool_result",
                    body=content_str,
                    tool_call_id=call_id,
                    is_error=False,
                )
            except Exception as exc:  # noqa: BLE001
                await on_writeback(
                    kind="tool_result",
                    body=f"Error: {exc}",
                    tool_call_id=call_id,
                    is_error=True,
                )

        # Final agent text. The script may declare a body either as
        # ``final_result["body"]`` (recommended for chat-shape stubs)
        # or as ``final_result`` itself when it's a string.
        final_body = _extract_final_body(script)
        if final_body:
            if on_text_delta:
                await on_text_delta(final_body)
            await on_writeback(kind="agent", body=final_body)
            if on_text_end:
                await on_text_end(committed=True)
        else:
            if on_text_end:
                await on_text_end(committed=False)
        return {
            "thinking": final_body or "",
            "summary": script.get("final_outcome", "success"),
        }


def _extract_final_body(script: dict) -> str:
    """Pull the agent's end-of-turn text from a script. Accepts
    either ``final_result["body"]`` (dict shape) or
    ``final_result`` directly (string shape). Empty string when
    neither is present."""
    fr = script.get("final_result")
    if isinstance(fr, dict):
        body = fr.get("body")
        if isinstance(body, str) and body:
            return body
    elif isinstance(fr, str) and fr:
        return fr
    return ""


# ── Registration ──


def _stub_agent_factory(config: Mapping[str, Any]) -> StubAgentProvider:
    return StubAgentProvider(config)


def register_stub_agent(
    provider_registry, contract_registry: ContractRegistry | None = None
) -> None:
    """Register the stub agent provider against (compute, "ai-agent")."""
    provider_registry.register(
        category=Category.COMPUTE,
        contract_name="ai-agent",
        product_name="stub",
        factory=_stub_agent_factory,
        conformance="passing",
        version=__version__,
        contract_registry=contract_registry,
    )
