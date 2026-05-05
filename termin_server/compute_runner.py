# Copyright 2026 Jamie-Leigh Blake and Termin project contributors
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0

"""Compute execution — LLM, Agent, and CEL compute invocation + audit traces.

Handles Level 1 LLM (field-to-field), Level 3 Agent (autonomous tool use),
and CEL server-side Compute execution. D-20 audit trace writing and redaction.
"""

import asyncio
import datetime as _dt
import json
import threading
import uuid

from fastapi import HTTPException, Request

from .context import RuntimeContext
from .storage import (
    get_db, create_record, get_record, update_record,
    list_records, filtered_query, update_fields, insert_raw, select_column,
)
from .state import do_state_transition
from .ai_provider import AIProviderError, build_output_tool, build_agent_tools
from .confidentiality import (
    check_compute_access, check_taint_integrity, enforce_output_taint,
    check_for_redacted_values,
)
from .errors import TerminError
from .transaction import Transaction, ContentSnapshot
from .boundaries import check_boundary_access
from .fastapi_adapter import make_auth_context
from termin_core.routing import build_the_user_for_cel


def _the_user_for(user: dict) -> dict:
    """Slice 7.5b helper: build the ``the_user`` CEL binding from a
    user-shaped dict (legacy runtime shape).

    Bridges the runtime's existing ``user: dict`` plumbing into the
    BRD #3 §4.2-shaped binding source CEL expects. Slice 7.5 may
    eliminate this when every compute call site receives an
    AuthContext directly.
    """
    return build_the_user_for_cel(make_auth_context(user))


# ── Prompt building (testable, pure functions) ──

def _resolve_directive_at_invocation(comp: dict, record: dict) -> tuple[str, str]:
    """v0.9 Phase 6c (BRD #3 §6.2-§6.3): resolve field-ref Directive
    and Objective text from the triggering record at each invocation.

    For computes with `directive_source.kind == "field"`, returns
    `record[<field>]` as the directive text. Same for objective.
    Deploy-config-sourced directives have already been resolved at
    app startup (see `app._resolve_directive_sources`); for those
    forms the resolved text already lives in `comp["directive"]`.

    A record missing the named field resolves to empty rather than
    raising — keeps the prompt-build path forgiving for partial
    data, same as inline-empty behavior.

    Returns: (directive_text, objective_text). Either may be empty.
    """
    directive = comp.get("directive", "") or ""
    objective = comp.get("objective", "") or ""

    d_src = comp.get("directive_source")
    if isinstance(d_src, dict) and d_src.get("kind") == "field":
        directive = str(record.get(d_src.get("field", ""), "") or "")

    o_src = comp.get("objective_source")
    if isinstance(o_src, dict) and o_src.get("kind") == "field":
        objective = str(record.get(o_src.get("field", ""), "") or "")

    return directive, objective


def _build_llm_prompts(comp: dict, record: dict, content_name: str,
                       singular_lookup: dict) -> tuple[str, str]:
    """Build system and user messages for Level 1 LLM compute.

    Fix 009.1: system = directive + objective (objective was wrongly in user turn).
    Fix 009.2: No default directive injected when only objective is present.

    Returns: (system_message, user_message)
    """
    # v0.9 Phase 6c: field-ref Directive/Objective resolve from the
    # triggering record. No-op for inline / deploy-config forms.
    directive, objective = _resolve_directive_at_invocation(comp, record)

    # Read input fields from record
    input_values = {}
    for content_ref, field_name in comp.get("input_fields", []):
        if field_name in record:
            input_values[field_name] = record[field_name]

    # Interpolate inline expressions in objective (field references)
    if objective:
        singular = singular_lookup.get(
            content_name,
            content_name.rstrip("s") if content_name.endswith("s") else content_name)
        for fname, fval in input_values.items():
            objective = objective.replace(f"{singular}.{fname}", str(fval))

    # System message: directive + objective (both optional, no defaults)
    system_parts = []
    if directive:
        system_parts.append(directive)
    if objective:
        system_parts.append(objective)
    system_msg = "\n\n".join(system_parts) if system_parts else ""

    # User message: input field values ONLY (no objective)
    if input_values:
        user_msg = "\n".join(f"{k}: {v}" for k, v in input_values.items())
    else:
        user_msg = ""

    return system_msg, user_msg


def _build_agent_prompts(comp: dict, record: dict) -> tuple[str, str]:
    """Build system and user messages for Level 3 Agent compute.

    Fix 009.2: No default directive injected when only objective is present.

    Returns: (system_message, user_message)
    """
    # v0.9 Phase 6c: field-ref Directive/Objective resolve from the
    # triggering record. No-op for inline / deploy-config forms.
    directive, objective = _resolve_directive_at_invocation(comp, record)

    # System message: directive + objective
    system_parts = []
    if directive:
        system_parts.append(directive)
    if objective:
        system_parts.append(objective)
    system_msg = "\n\n".join(system_parts) if system_parts else ""

    # User message: triggering record context
    user_msg = f"Triggering record:\n{json.dumps(record, indent=2, default=str)}"

    return system_msg, user_msg


def _build_agent_set_output(comp: dict, content_lookup: dict) -> dict:
    """Build the set_output tool for agent computes.

    Fix 009.3: Only includes 'thinking' if the compute's output schema declares it.
    Always includes 'summary' for completion signal.
    """
    properties = {
        "summary": {"type": "string", "description": "Result summary."},
    }
    required = ["summary"]

    # Add output fields from the compute's declaration
    for content_ref, field_name in comp.get("output_fields", []):
        schema = None
        for name, s in content_lookup.items():
            singular = s.get("singular", "")
            if name == content_ref or singular == content_ref:
                schema = s
                break
        if schema:
            field_def = None
            for f in schema.get("fields", []):
                if f.get("name", "") == field_name:
                    field_def = f
                    break
            if field_def:
                prop = {"description": f"Field: {content_ref}.{field_name}"}
                if field_def.get("column_type") in ("INTEGER", "REAL"):
                    prop["type"] = "number"
                else:
                    prop["type"] = "string"
                properties[field_name] = prop
                required.append(field_name)
                continue
        # Fallback
        properties[field_name] = {"type": "string", "description": f"Field: {content_ref}.{field_name}"}
        required.append(field_name)

    return {
        "name": "set_output",
        "description": "Signal that you have completed the task. Call this when done.",
        "input_schema": {
            "type": "object",
            "properties": properties,
            "required": required,
        }
    }


def _build_llm_audit_metadata(
    ctx: RuntimeContext, comp_snake: str,
    system_msg: str, user_msg: str, result: dict | None,
    *, error_str: str | None = None,
) -> dict:
    """Build the BRD §6.3.4 audit metadata dict for an LLM call.

    Slice (d) reads the deploy binding for provider_product, the
    provider instance for provider_config_hash + model_identifier,
    and the result dict for cost. The output dict shape matches what
    write_audit_trace expects under audit_metadata.

    Slice (e) extends this for refusal and structured tool_calls.
    """
    provider_inst = ctx.compute_providers.get(comp_snake)
    # Resolve provider_product from the deploy binding if available.
    bindings = (
        getattr(ctx, "_deploy_bindings", None)
        or {}
    )
    provider_product = ""
    if hasattr(ctx, "compute_providers") and provider_inst is not None:
        provider_product = getattr(provider_inst, "service", "") or ""
    config_hash = getattr(provider_inst, "_config_hash", "") if provider_inst else ""
    model_id = getattr(provider_inst, "model", "") or ""
    prompt_as_sent = f"<system>\n{system_msg}\n</system>\n{user_msg}"
    cost_units = 0
    if result and isinstance(result, dict):
        usage = result.get("_termin_usage") or {}
        if isinstance(usage, dict):
            cost_units = int(usage.get("total_tokens") or 0)
    return {
        "provider_product": provider_product,
        "model_identifier": model_id,
        "provider_config_hash": config_hash,
        "prompt_as_sent": prompt_as_sent,
        "sampling_params_json": "{}",
        "tool_calls_json": "[]",
        "refusal_reason": None,
        "cost_units": cost_units,
        "cost_unit_type": "tokens" if cost_units else "",
        "cost_currency_amount": "",
    }


# v0.9.2 L7.5 (JL Wave 3 callout): _write_refusal_sidecar removed.
# The Phase 3 slice (e) `compute_refusals` Content type is retired —
# the WARN-level audit log entry written on system.refuse is the
# audit-trail surface; the v0.9.2 runtime appends a kind="assistant",
# type="refusal" conversation entry as the chat surface. Two
# surfaces, one event source.


def _build_agent_audit_metadata(
    ctx: RuntimeContext, comp_snake: str,
    system_msg: str, user_msg: str, tool_calls_log: list,
    *, refusal_reason: str | None = None,
) -> dict:
    """Build the BRD §6.3.4 audit metadata dict for an ai-agent call.

    tool_calls_log is a list of {tool, args, result, is_error,
    latency_ms} dicts capturing every tool call the agent made
    during the invocation. Persisted as JSON in the audit
    `tool_calls` column.
    """
    provider_inst = ctx.compute_providers.get(comp_snake)
    provider_product = ""
    if provider_inst is not None:
        provider_product = getattr(provider_inst, "service", "") or ""
    config_hash = getattr(provider_inst, "_config_hash", "") if provider_inst else ""
    model_id = getattr(provider_inst, "model", "") or ""
    prompt_as_sent = f"<system>\n{system_msg}\n</system>\n{user_msg}"
    return {
        "provider_product": provider_product,
        "model_identifier": model_id,
        "provider_config_hash": config_hash,
        "prompt_as_sent": prompt_as_sent,
        "sampling_params_json": "{}",
        "tool_calls_json": json.dumps(tool_calls_log) if tool_calls_log else "[]",
        "refusal_reason": refusal_reason,
        "cost_units": 0,
        "cost_unit_type": "",
        "cost_currency_amount": "",
    }


async def execute_compute(ctx: RuntimeContext, comp: dict, record: dict,
                          content_name: str, main_loop=None,
                          invoked_by=None, triggering_entry: dict = None):
    """Execute a Compute triggered by an event or manual /trigger call.

    ``invoked_by`` is the upstream Principal who caused this run. For
    manual ``POST /api/v1/compute/<name>/trigger`` calls it's the
    HTTP caller's resolved principal; for event-triggered runs it's
    the principal who caused the upstream event (or None if the
    event chain doesn't carry one — system-triggered, scheduler).
    Threaded through to ``write_audit_trace`` so the audit row
    stamps the right principal columns per BRD §6.3.4.

    ``triggering_entry`` (v0.9.2 L7.4) is the appended_entry payload
    when the compute was triggered by a `<X>.<Y>.appended` event.
    Used for setting `parent_id` on auto-write-back conversation
    entries (refusals in this slice; assistant text + tool_call/result
    in L7.3). None on manual /trigger and scheduler paths.
    """
    comp_name = comp["name"]["display"]
    provider = comp.get("provider", "cel")

    if provider == "llm":
        await _execute_llm_compute(
            ctx, comp, record, content_name, main_loop,
            invoked_by=invoked_by,
        )
    elif provider == "ai-agent":
        await _execute_agent_compute(
            ctx, comp, record, content_name, main_loop,
            invoked_by=invoked_by,
            triggering_entry=triggering_entry,
        )
    elif provider in (None, "", "cel", "default-CEL"):
        # v0.9.1: default-CEL via /trigger now writes an audit row.
        # Previously this branch only printed a warning, leaving the
        # spec §5.2 "manual trigger writes audit" requirement
        # unsatisfied. The synchronous endpoint at
        # /api/v1/compute/<name>/ has its own audit path; this
        # branch is the manual-trigger path for CEL computes.
        await _execute_cel_compute(
            ctx, comp, record, content_name,
            invoked_by=invoked_by,
        )
    else:
        print(
            f"[Termin] Compute '{comp_name}': unknown provider "
            f"'{provider}' — skipping"
        )


async def _execute_cel_compute(ctx: RuntimeContext, comp: dict, record: dict,
                                content_name: str, invoked_by=None):
    """Execute a default-CEL Compute — evaluate the CEL body and
    write an audit row.

    Slimmer than the synchronous /api/v1/compute/<name>/ endpoint:
    no preconditions, no postconditions, no transaction
    machinery — manual trigger + event paths use this for the
    audit-only contract per BRD §6.3.4. The full sync endpoint
    remains the right path for callers that need the result back.
    """
    comp_name = comp["name"]["display"]
    invocation_id = str(uuid.uuid4())
    started = _dt.datetime.now(_dt.timezone.utc)
    started_str = started.isoformat().replace("+00:00", "Z")

    body_lines = comp.get("body_lines", [])
    if not body_lines:
        # No CEL body to evaluate — record the invocation as
        # success but leave trace empty.
        completed = _dt.datetime.now(_dt.timezone.utc)
        await write_audit_trace(
            ctx, comp, invocation_id=invocation_id, trigger="manual",
            started_at=started_str,
            completed_at=completed.isoformat().replace("+00:00", "Z"),
            latency_ms=(completed - started).total_seconds() * 1000.0,
            outcome="success",
            trace_data={"compute_type": "cel", "note": "no body"},
            invoked_by=invoked_by,
        )
        return

    cel_body = body_lines[0]
    eval_ctx = {
        "Compute": {
            "Name": comp_name,
            "Provider": "cel",
            "IdentityMode": comp.get("identity_mode", "delegate"),
            "Trigger": "manual",
            "ExecutionId": invocation_id,
            "StartedAt": started_str,
        },
    }
    if isinstance(record, dict):
        eval_ctx.update(record)
        if content_name:
            eval_ctx[content_name] = record

    try:
        ctx.expr_eval.evaluate(cel_body, eval_ctx)
        completed = _dt.datetime.now(_dt.timezone.utc)
        await write_audit_trace(
            ctx, comp, invocation_id=invocation_id, trigger="manual",
            started_at=started_str,
            completed_at=completed.isoformat().replace("+00:00", "Z"),
            latency_ms=(completed - started).total_seconds() * 1000.0,
            outcome="success",
            trace_data={"compute_type": "cel", "cel": cel_body},
            invoked_by=invoked_by,
        )
    except Exception as e:
        completed = _dt.datetime.now(_dt.timezone.utc)
        await write_audit_trace(
            ctx, comp, invocation_id=invocation_id, trigger="manual",
            started_at=started_str,
            completed_at=completed.isoformat().replace("+00:00", "Z"),
            latency_ms=(completed - started).total_seconds() * 1000.0,
            outcome="error",
            error_message=str(e),
            trace_data={"compute_type": "cel", "cel": cel_body, "error": str(e)},
            invoked_by=invoked_by,
        )


async def _execute_llm_compute(ctx: RuntimeContext, comp: dict, record: dict,
                                content_name: str, main_loop=None,
                                invoked_by=None):
    """Execute a Level 1 LLM Compute — field-to-field completion."""
    comp_name = comp["name"]["display"]
    comp_snake = comp["name"]["snake"]
    _llm_started = _dt.datetime.now(_dt.timezone.utc)
    _llm_invocation_id = str(uuid.uuid4())

    # v0.9 Phase 3: per-compute provider lookup. Slice (b) routes
    # through `provider.legacy` (the embedded AIProvider) for SDK
    # calls so prompt building, tool_use forcing, and streaming
    # behavior are byte-identical with v0.8. Slice (d) ports the
    # legacy methods into the contract surface and deletes .legacy.
    provider = ctx.compute_providers.get(comp_snake)
    if provider is None or not getattr(provider, "is_configured", False):
        print(f"[Termin] Compute '{comp_name}': no provider bound, skipped")
        return

    # Build prompts (Fix 009.1 + 009.2)
    system_msg, user_msg = _build_llm_prompts(comp, record, content_name, ctx.singular_lookup)

    # Build output tool
    output_fields = comp.get("output_fields", [])
    output_tool = build_output_tool(output_fields, ctx.content_lookup)

    print(f"[Termin] Compute '{comp_name}': calling {provider.service} (record {record.get('id', '?')})")

    # v0.8.1: LLM-path streaming. When the provider supports
    # stream_agent_response, route the call through it and publish
    # each field_delta / field_done / done event onto the event bus
    # so any component rendering the target field (data_table cells,
    # chat bubbles, detail views) can render tokens as they arrive.
    # Falls back to non-streaming complete() for providers that don't
    # implement the streaming path.
    #
    # Events carry content_name + record_id so the general client
    # hydrator can target `[data-termin-row-id=<id>]
    # [data-termin-field=<field>]` without knowing the component type.
    _llm_stream_base = f"compute.stream.{_llm_invocation_id}"
    _llm_record_id = record.get("id")

    async def _on_llm_stream_event(event):
        if ctx.event_bus is None:
            return
        etype = event.get("type")
        if etype == "field_delta":
            field = event.get("field", "")
            await ctx.event_bus.publish({
                "channel_id": f"{_llm_stream_base}.field.{field}",
                "data": {
                    "invocation_id": _llm_invocation_id,
                    "compute": comp_snake,
                    "mode": "tool_use",
                    "tool": event.get("tool", "set_output"),
                    "content_name": content_name,
                    "record_id": _llm_record_id,
                    "field": field,
                    "delta": event.get("delta", ""),
                    "done": False,
                },
            })
        elif etype == "field_done":
            field = event.get("field", "")
            await ctx.event_bus.publish({
                "channel_id": f"{_llm_stream_base}.field.{field}",
                "data": {
                    "invocation_id": _llm_invocation_id,
                    "compute": comp_snake,
                    "mode": "tool_use",
                    "tool": event.get("tool", "set_output"),
                    "content_name": content_name,
                    "record_id": _llm_record_id,
                    "field": field,
                    "done": True,
                    "value": event.get("value"),
                },
            })
        elif etype == "done":
            await ctx.event_bus.publish({
                "channel_id": _llm_stream_base,
                "data": {
                    "invocation_id": _llm_invocation_id,
                    "compute": comp_snake,
                    "mode": "tool_use",
                    "tool": "set_output",
                    "content_name": content_name,
                    "record_id": _llm_record_id,
                    "done": True,
                    "output": event.get("output") or {},
                },
            })

    try:
        # Slice (b): legacy methods on `provider.legacy` (an internal
        # AIProvider). Same call shape as v0.8 — no behavior change.
        legacy = provider.legacy
        use_streaming = (
            ctx.event_bus is not None
            and hasattr(legacy, "stream_agent_response")
            and provider.service == "anthropic"
        )
        if use_streaming:
            result = {}
            async for event in legacy.stream_agent_response(
                    system_msg, user_msg, output_tool):
                if event.get("type") == "done":
                    result = event.get("output") or {}
                await _on_llm_stream_event(event)
        else:
            result = await legacy.complete(system_msg, user_msg, output_tool)
        thinking = result.pop("thinking", "")
        if thinking:
            print(f"[Termin] Compute '{comp_name}' thinking: {thinking[:100]}")

        # Write output fields back to the record
        if output_fields and record.get("id"):
            update_data = {}
            for content_ref, field_name in output_fields:
                if field_name in result:
                    update_data[field_name] = result[field_name]
            if update_data:
                db = await get_db(ctx.db_path)
                try:
                    await update_fields(db, content_name, record["id"], update_data)
                    print(f"[Termin] Compute '{comp_name}': updated record {record['id']}")
                    updated_record = dict(record)
                    updated_record.update(update_data)
                    event_data = {
                        "channel_id": f"content.{content_name}.updated",
                        "data": updated_record,
                    }
                    if main_loop and main_loop.is_running():
                        asyncio.run_coroutine_threadsafe(
                            ctx.event_bus.publish(event_data), main_loop)
                    else:
                        await ctx.event_bus.publish(event_data)
                finally:
                    await db.close()

        # D-20: Audit trace on success
        _llm_completed = _dt.datetime.now(_dt.timezone.utc)
        _llm_duration = (_llm_completed - _llm_started).total_seconds() * 1000
        audit_level = comp.get("audit_level", "actions")
        trace_data = {"compute_type": "agent", "calls": [{"response": thinking[:200] if thinking else ""}]}
        if audit_level == "debug":
            trace_data["calls"][0]["system_prompt"] = system_msg
            trace_data["calls"][0]["thinking"] = thinking
        # v0.9 Phase 3 slice (d): BRD §6.3.4 audit_metadata.
        audit_metadata = _build_llm_audit_metadata(
            ctx, comp_snake, system_msg, user_msg, result,
        )
        await write_audit_trace(
            ctx, comp, invocation_id=_llm_invocation_id, trigger="event",
            started_at=_llm_started.isoformat().replace("+00:00", "Z"),
            completed_at=_llm_completed.isoformat().replace("+00:00", "Z"),
            latency_ms=_llm_duration, outcome="success",
            trace_data=trace_data,
            audit_metadata=audit_metadata,
            invoked_by=invoked_by,
        )
    except AIProviderError as e:
        print(f"[Termin] [ERROR] Compute '{comp_name}': {e}")
        _llm_err_completed = _dt.datetime.now(_dt.timezone.utc)
        _llm_err_duration = (_llm_err_completed - _llm_started).total_seconds() * 1000
        audit_metadata = _build_llm_audit_metadata(
            ctx, comp_snake, system_msg, user_msg, None, error_str=str(e),
        )
        await write_audit_trace(
            ctx, comp, invocation_id=_llm_invocation_id, trigger="event",
            started_at=_llm_started.isoformat().replace("+00:00", "Z"),
            completed_at=_llm_err_completed.isoformat().replace("+00:00", "Z"),
            latency_ms=_llm_err_duration, outcome="error",
            error_message=str(e),
            trace_data={"compute_type": "agent", "error": str(e)},
            audit_metadata=audit_metadata,
            invoked_by=invoked_by,
        )


async def _execute_agent_compute(ctx: RuntimeContext, comp: dict, record: dict,
                                  content_name: str, main_loop=None,
                                  invoked_by=None, triggering_entry: dict = None):
    """Execute a Level 3 Agent Compute — autonomous with tool calls.

    ``triggering_entry`` (v0.9.2 L7.4) is the entry payload from the
    upstream `<X>.<Y>.appended` event, when the compute was triggered
    by one. Used to set `parent_id` on auto-write-back conversation
    entries so reviewers can trace from a refusal (or, post-L7.3, an
    assistant response or tool_call/result) back to the user message
    that started the turn.
    """
    comp_name = comp["name"]["display"]
    comp_snake = comp["name"]["snake"]
    _agent_started = _dt.datetime.now(_dt.timezone.utc)
    _agent_invocation_id = str(uuid.uuid4())

    # v0.9 Phase 3: per-compute provider lookup (slice b interim
    # via provider.legacy — see _execute_llm_compute for rationale).
    provider = ctx.compute_providers.get(comp_snake)
    if provider is None or not getattr(provider, "is_configured", False):
        print(f"[Termin] Compute '{comp_name}': no provider bound, skipped")
        return

    accesses = comp.get("accesses", [])
    # v0.9 Phase 3 slice (c): Reads grants read-only content access.
    # The agent's tool surface includes content_query / content_get
    # for these types but not content_create / update / delete.
    # State tools (state_transition) come from accesses only.
    reads = comp.get("reads", [])
    # readable = anything in Accesses or Reads; writable = Accesses only.
    readable_set = set(accesses) | set(reads)
    writable_set = set(accesses)

    # Build prompts (Fix 009.1 + 009.2)
    system_msg, user_msg = _build_agent_prompts(comp, record)

    # v0.9.2 L7.1: conversation-mode detection. When the compute
    # declares `Conversation is X.Y`, the agent runs the §11.5
    # auto-write-back path: the user-message string is replaced by
    # a materialized Anthropic-shape messages list, set_output is
    # stripped from the tool surface (no completion sentinel), and
    # each tool_call / tool_result / final assistant text the agent
    # produces is appended back to the conversation field via
    # _do_append. Refusal (L7.4) still wins over normal write-back.
    conversation_source = comp.get("conversation_source")
    is_conv_mode = (
        conversation_source and len(conversation_source) == 2
    )

    # Build tools
    agent_tools = build_agent_tools(accesses, ctx.content_lookup)
    # v0.9.2 close-out: surface author-declared computes as agent
    # tools per `Invokes "<X>"` declarations. CEL only in v0.9.2.
    invokes_list = list(comp.get("invokes") or [])
    invokable_tools = []
    if invokes_list and ctx.compute_lookup:
        from .ai_provider import build_invokable_compute_tools
        invokable_tools = build_invokable_compute_tools(
            invokes_list, ctx.compute_lookup,
        )
    if is_conv_mode:
        # No set_output on conversation mode — the agent communicates
        # by ending its turn with text. Per §11.5 + L7 design.
        # v0.9.2 close-out: every tool gets a `purpose` schema field
        # so the agent is consistently prompted to supply intent for
        # chat-UI display.
        from .ai_provider import _add_purpose_to_tool
        all_tools = [
            _add_purpose_to_tool(t)
            for t in (agent_tools + invokable_tools)
        ]
        # Per §11.3, append the refusal marker so the model knows
        # system.refuse is reachable.
        system_msg = (
            (system_msg or "")
            + "\n\nYou may refuse a request you cannot fulfill by "
              "calling system.refuse(reason)."
        ).strip()
    else:
        set_output = _build_agent_set_output(comp, ctx.content_lookup)
        all_tools = agent_tools + invokable_tools + [set_output]

    # v0.9 Phase 3 slice (e): refusal capture state. Mutated by
    # _execute_tool when the agent calls system_refuse; consulted
    # post-loop to convert the outcome to "refused".
    # v0.9.2 L7.5: the sidecar write the post-loop check used to do
    # is retired (audit log is the audit-trail surface). L7.4 will
    # add a kind="assistant", type="refusal" conversation entry append
    # for the chat surface — runs from the same refusal_state hook.
    refusal_state: dict = {}

    async def _execute_tool(tool_name: str, tool_input: dict) -> dict:
        db = await get_db(ctx.db_path)
        try:
            # v0.9 Phase 3 slice (e): system_refuse capture.
            # v0.9.2 close-out: the AI provider now honors
            # `should_halt` between turns AND between tools in a
            # multi-tool batch — once refusal_state is set, no
            # further tool calls land here. This branch is the
            # defensive guard for any future provider that doesn't
            # check should_halt: a non-refuse tool reaching here
            # AFTER a refusal returns an error envelope rather
            # than executing. The post-loop refusal-append path
            # (L7.4) writes the refusal entry as the last commit
            # on the conversation field per compute-contract.md
            # §6.1.
            if tool_name == "system_refuse":
                if not refusal_state:
                    refusal_state["reason"] = str(
                        tool_input.get("reason", "")
                    ).strip()
                return {"acknowledged": True}
            if refusal_state.get("reason"):
                return {"error": (
                    "Invocation refused; further tool calls are not "
                    "permitted. The runtime has terminated this "
                    "agent loop and is appending the refusal entry."
                )}

            if tool_name == "content_query":
                cname = tool_input.get("content_name", "")
                # v0.9 Phase 3 slice (c): read tools accept either
                # Accesses or Reads as the source-side grant.
                if cname not in readable_set:
                    return {"error": (
                        f"Access denied: {cname} not in Accesses or "
                        f"Reads"
                    )}
                bnd_err = check_boundary_access(
                    ctx.boundary_for_compute, ctx.boundary_for_content,
                    comp_snake, cname)
                if bnd_err:
                    return {"error": bnd_err}
                filters = tool_input.get("filters", {})
                return await filtered_query(db, cname, filters or None)

            elif tool_name == "content_create":
                cname = tool_input.get("content_name", "")
                if cname not in writable_set:
                    return {"error": f"Access denied: {cname} not in Accesses"}
                bnd_err = check_boundary_access(
                    ctx.boundary_for_compute, ctx.boundary_for_content,
                    comp_snake, cname)
                if bnd_err:
                    return {"error": bnd_err}
                data = tool_input.get("data", {})
                # v0.9 multi-SM: sm_info is the list of state-machine specs
                # for this content. create_record() seeds initial values
                # for each machine's column from that list.
                sm_info = ctx.sm_lookup.get(cname, [])
                schema = ctx.content_lookup.get(cname, {})
                rec = await create_record(db, cname, data, schema, sm_info,
                                          ctx.terminator, ctx.event_bus)
                return rec

            elif tool_name == "content_update":
                cname = tool_input.get("content_name", "")
                if cname not in writable_set:
                    return {"error": f"Access denied: {cname} not in Accesses"}
                bnd_err = check_boundary_access(
                    ctx.boundary_for_compute, ctx.boundary_for_content,
                    comp_snake, cname)
                if bnd_err:
                    return {"error": bnd_err}
                rid = tool_input.get("record_id")
                data = tool_input.get("data", {})
                await update_record(db, cname, rid, data, "id",
                                    ctx.terminator, ctx.event_bus)
                return {"ok": True, "id": rid}

            elif tool_name == "state_transition":
                cname = tool_input.get("content_name", "")
                # State tools come from Accesses only — Reads grants
                # do not include state.transition. BRD §6.3.3 explicit.
                if cname not in writable_set:
                    return {"error": f"Access denied: {cname} not in Accesses"}
                bnd_err = check_boundary_access(
                    ctx.boundary_for_compute, ctx.boundary_for_content,
                    comp_snake, cname)
                if bnd_err:
                    return {"error": bnd_err}
                rid = tool_input.get("record_id")
                target = tool_input.get("target_state")
                # v0.9: machine_name is required when content has multiple
                # state machines. Fall back to the single machine when one
                # exists; raise when ambiguous.
                machine = tool_input.get("machine_name", "")
                sm_list = ctx.sm_lookup.get(cname, [])
                if not machine:
                    if len(sm_list) == 1:
                        machine = sm_list[0]["machine_name"]
                    else:
                        return {"error": (
                            f"machine_name is required for state_transition on "
                            f"'{cname}' (has {len(sm_list)} state machines)")}
                # Phase 2.x (d): transitions go through ctx.storage
                # for atomic CAS — same path as the human transition
                # endpoint.
                result = await do_state_transition(
                    ctx.storage, cname, rid, machine, target,
                    {"role": "service", "scopes": list(ctx.scope_for_content_verb(cname, "update") or [])},
                    ctx.sm_lookup, ctx.terminator, ctx.event_bus)
                return result

            elif tool_name in invokes_list:
                # v0.9.2 close-out: agent invokes a declared compute
                # as a tool. Resolve via ctx.compute_lookup, evaluate
                # the CEL body with tool_args bound (and the param's
                # named slot, when one is declared), return the
                # resulting input record (Transform/Reduce-shape
                # convention) to the agent.
                target = ctx.compute_lookup.get(tool_name) if ctx.compute_lookup else None
                if target is None:
                    return {"error": f"Invoked compute '{tool_name}' not found"}
                provider = target.get("provider") or "cel"
                if provider not in ("cel", "default-CEL", None, ""):
                    return {"error": (
                        f"Invokes wiring for provider '{provider}' is "
                        f"reserved for future slices; v0.9.2 supports "
                        f"default-CEL only."
                    )}
                body_lines = target.get("body_lines") or []
                if not body_lines:
                    return {"error": f"Compute '{tool_name}' has no body"}
                cel_body = body_lines[0]
                # Build the eval context: top-level keys are the
                # compute's input param names (typically named after
                # the content singular), with the agent-supplied
                # arg dict as the value. The body mutates that dict
                # (Transform/Reduce convention) and we return it.
                eval_ctx = {}
                for param in target.get("input_params") or ():
                    pname = (
                        param.get("name") if isinstance(param, dict)
                        else getattr(param, "name", None)
                    )
                    if not pname:
                        continue
                    eval_ctx[pname] = (
                        tool_input.get(pname)
                        if isinstance(tool_input, dict) and pname in tool_input
                        else (tool_input if isinstance(tool_input, dict) else {})
                    )
                # Also flatten top-level args into the eval context
                # so CEL bodies that reference args by name (not via
                # the param record) still work.
                if isinstance(tool_input, dict):
                    for k, v in tool_input.items():
                        if k not in eval_ctx:
                            eval_ctx[k] = v
                try:
                    expr_result = ctx.expr_eval.evaluate(cel_body, eval_ctx)
                except Exception as exc:
                    return {"error": f"Invoked compute failed: {exc}"}
                # Return the mutated input record (the convention
                # for Transform/Reduce shapes). When the body is a
                # pure expression returning a value (no mutation),
                # surface that as `value`.
                output = {}
                for param in target.get("output_params") or ():
                    pname = (
                        param.get("name") if isinstance(param, dict)
                        else getattr(param, "name", None)
                    )
                    if pname and pname in eval_ctx:
                        output[pname] = eval_ctx[pname]
                if not output:
                    output = {"value": expr_result}
                return output

            else:
                return {"error": f"Unknown tool: {tool_name}"}
        finally:
            await db.close()

    print(f"[Termin] Compute '{comp_name}': starting agent loop ({provider.service})")

    # v0.8 #7: stream set_output field deltas to the compute.stream.*
    # channel family so connected clients (chat UI) can render
    # token-by-token. Event-bus publication is cheap and no-op when
    # nobody is subscribed, so we always go through the streaming path
    # when an event bus is available. Fallback to the non-streaming
    # agent_loop only if the bus is unavailable (defensive).
    _stream_base_channel = f"compute.stream.{_agent_invocation_id}"
    _agent_record_id = record.get("id") if record else None

    async def _on_stream_event(event):
        """Push each agent-stream event onto the event bus on the
        appropriate channel per the v0.8 streaming protocol.

        content_name + record_id (when known) are included so the
        general client-side hydrator can target DOM elements keyed by
        (row_id, field_name) — the same shape as `content.*.updated`
        events but streamed. This keeps streaming orthogonal to the
        presentation component type (data_table, chat, detail view).
        """
        if ctx.event_bus is None:
            return
        etype = event.get("type")
        tool_name = event.get("tool", "set_output")
        if etype == "field_delta":
            field = event.get("field", "")
            await ctx.event_bus.publish({
                "channel_id": f"{_stream_base_channel}.field.{field}",
                "data": {
                    "invocation_id": _agent_invocation_id,
                    "compute": comp_snake,
                    "mode": "tool_use",
                    "tool": tool_name,
                    "content_name": content_name,
                    "record_id": _agent_record_id,
                    "field": field,
                    "delta": event.get("delta", ""),
                    "done": False,
                },
            })
        elif etype == "field_done":
            field = event.get("field", "")
            await ctx.event_bus.publish({
                "channel_id": f"{_stream_base_channel}.field.{field}",
                "data": {
                    "invocation_id": _agent_invocation_id,
                    "compute": comp_snake,
                    "mode": "tool_use",
                    "tool": tool_name,
                    "content_name": content_name,
                    "record_id": _agent_record_id,
                    "field": field,
                    "done": True,
                    "value": event.get("value"),
                },
            })
        elif etype == "done":
            await ctx.event_bus.publish({
                "channel_id": _stream_base_channel,
                "data": {
                    "invocation_id": _agent_invocation_id,
                    "compute": comp_snake,
                    "mode": "tool_use",
                    "tool": "set_output",
                    "content_name": content_name,
                    "record_id": _agent_record_id,
                    "done": True,
                    "output": event.get("output") or {},
                },
            })

    # v0.9.2 L7.1+L7.3: conversation-mode write-back state. Captured
    # so we don't fire write-back if the agent later refuses (refusal
    # path owns the entry per L7.4 — the writes to the conversation
    # field are exclusive between normal-completion and refusal).
    _writeback_log: list[dict] = []
    _conv_user_dict: dict = {}
    if invoked_by is not None:
        _conv_user_dict = {"id": getattr(invoked_by, "id", "") or ""}

    async def _on_writeback(*, kind: str, body: str, **fields):
        """v0.9.2 L7.3: append one auto-write-back entry to the
        conversation field. Each entry shares parent_id = triggering
        user entry id so reviewers can reconstruct turn boundaries.

        The runtime owns parent_id (resolved from triggering_entry);
        the provider supplies kind, body, and the structured fields
        per kind (tool_call_id, tool_name, tool_args for tool_call;
        tool_call_id, is_error for tool_result).

        v0.9.2 close-out: refusal short-circuit. Once
        ``refusal_state`` is set (system_refuse fired earlier this
        invocation), no further commits land on the conversation
        field through this callback. The post-loop refusal-append
        path writes the refusal entry as the last commit. Per
        compute-contract.md §6.1: "the loop terminates; staged
        outputs are discarded."
        """
        if refusal_state.get("reason"):
            return
        from .routes import (
            _do_append, AppendValidationError, AppendNotFoundError,
        )
        conv_content, conv_field = conversation_source
        payload: dict = {"kind": kind, "body": body}
        if triggering_entry:
            tparent = triggering_entry.get("id")
            if tparent:
                payload["parent_id"] = tparent
        # Pass through provider-supplied structured fields. The
        # _do_append passthrough list (routes.py) already accepts
        # tool_call_id / tool_name / tool_args / parent_id / etc.
        for k, v in fields.items():
            payload[k] = v
        try:
            entry = await _do_append(
                ctx,
                content_ref=conv_content,
                key_val=record.get("id") if record else None,
                field_name=conv_field,
                payload=payload,
                user=_conv_user_dict,
                row_filter=None,
            )
            _writeback_log.append(entry)
        except (AppendValidationError, AppendNotFoundError) as e:
            print(
                f"[Termin] [WARN] Compute '{comp_name}': "
                f"failed to append {kind!r} entry to conversation "
                f"{conv_content}.{conv_field}: {e}"
            )

    try:
        legacy = provider.legacy
        if is_conv_mode and hasattr(
                legacy, "agent_loop_with_conversation"):
            # v0.9.2 L7.1: load the conversation field, materialize
            # to Anthropic shape, run the §11.5 conversation loop.
            from .ai_provider import (
                materialize_to_anthropic,
                ConversationMaterializationError,
            )
            raw_field = (record or {}).get(conversation_source[1])
            if raw_field in (None, ""):
                conv_entries: list = []
            elif isinstance(raw_field, list):
                conv_entries = raw_field
            else:
                try:
                    conv_entries = json.loads(raw_field)
                    if not isinstance(conv_entries, list):
                        conv_entries = []
                except (TypeError, ValueError):
                    conv_entries = []
            try:
                messages = materialize_to_anthropic(conv_entries)
            except ConversationMaterializationError as e:
                raise AIProviderError(
                    f"conversation materialization failed: {e}"
                ) from e

            # v0.9.2 streaming (post-close-out): wire on_text_delta /
            # on_text_end to a per-record streaming channel so the
            # chat UI can render token-by-token. The channel name
            # mirrors the existing `<X>.<Y>.appended` shape:
            # `content.<source>.<field>.streaming`. Each delta event
            # carries record_id + text; the `end` event carries
            # record_id + committed (true if a matching `appended`
            # event will follow with the persisted entry, false if
            # the streamed text was tool-call thinking and should
            # be cleared from the chat UI's pending bubble).
            conv_content, conv_field = conversation_source
            stream_channel = (
                f"content.{conv_content}.{conv_field}.streaming"
            )

            async def _on_text_delta(text: str):
                if ctx.event_bus is None:
                    return
                envelope = {
                    "channel_id": stream_channel,
                    "type": "delta",
                    "content_name": conv_content,
                    "field_name": conv_field,
                    "record_id": record.get("id") if record else None,
                    "text": text,
                    "invocation_id": _agent_invocation_id,
                }
                envelope["data"] = dict(envelope)
                await ctx.event_bus.publish(envelope)

            async def _on_text_end(committed: bool):
                if ctx.event_bus is None:
                    return
                envelope = {
                    "channel_id": stream_channel,
                    "type": "end",
                    "content_name": conv_content,
                    "field_name": conv_field,
                    "record_id": record.get("id") if record else None,
                    "committed": bool(committed),
                    "invocation_id": _agent_invocation_id,
                }
                envelope["data"] = dict(envelope)
                await ctx.event_bus.publish(envelope)

            # v0.9.2 close-out: should_halt is the runtime->provider
            # signal that the agent loop must terminate (e.g.,
            # system_refuse fired). Per compute-contract.md §6.1
            # the loop must terminate on refusal; should_halt is
            # the mechanism that lets the provider check between
            # turns + between mid-turn tool calls.
            def _should_halt():
                return bool(refusal_state.get("reason"))

            result = await legacy.agent_loop_with_conversation(
                system_msg, messages, all_tools, _execute_tool,
                on_writeback=_on_writeback,
                on_text_delta=_on_text_delta,
                on_text_end=_on_text_end,
                should_halt=_should_halt,
            )
        elif ctx.event_bus is not None and hasattr(
                legacy, "agent_loop_streaming"):
            result = await legacy.agent_loop_streaming(
                system_msg, user_msg, all_tools, _execute_tool,
                on_event=_on_stream_event)
        else:
            result = await legacy.agent_loop(
                system_msg, user_msg, all_tools, _execute_tool)
        thinking = result.get("thinking", "")
        if thinking:
            print(f"[Termin] Compute '{comp_name}' completed: {thinking[:100]}")

        _agent_completed = _dt.datetime.now(_dt.timezone.utc)
        _agent_duration = (_agent_completed - _agent_started).total_seconds() * 1000
        audit_level = comp.get("audit_level", "actions")

        # v0.9.2 L7.5: refusal handling. The compute_refusals sidecar
        # write is retired (per JL Wave 3 callout) — the WARN-level
        # audit log entry is the audit-trail surface; L7.4 (next
        # commit) appends a kind="assistant", type="refusal"
        # conversation entry as the chat surface. The refusal event
        # publish (compute.<name>.refused) is also retired — audit log
        # is the queryable source of truth. Refusal is logged
        # unconditionally regardless of audit_level (BRD contract
        # invariant).
        if refusal_state.get("reason"):
            agent_audit_metadata = _build_agent_audit_metadata(
                ctx, comp_snake, system_msg, user_msg, tool_calls_log=[],
                refusal_reason=refusal_state["reason"],
            )
            await write_audit_trace(
                ctx, comp, invocation_id=_agent_invocation_id,
                trigger="event",
                started_at=_agent_started.isoformat().replace("+00:00", "Z"),
                completed_at=_agent_completed.isoformat().replace("+00:00", "Z"),
                latency_ms=_agent_duration, outcome="refused",
                trace_data={
                    "compute_type": "agent",
                    "refused": True,
                    "reason": refusal_state["reason"],
                },
                audit_metadata=agent_audit_metadata,
                invoked_by=invoked_by,
            )
            # v0.9.2 L7.4: append the refusal as a conversation entry
            # so the chat provider renders it inline at source position.
            # Per tech-design §7.2 + §11.5, refusal is an assistant-kind
            # entry with type="refusal" — not a separate kind. The
            # entry is appended to the compute's conversation_source
            # (an L6 IR field). Computes without a Conversation source
            # (legacy ai-agent computes still using the messages-
            # collection pattern) get the audit row only — they have
            # no field to append to.
            conversation_source = comp.get("conversation_source")
            if conversation_source and len(conversation_source) == 2:
                from .routes import _do_append, AppendValidationError, AppendNotFoundError
                conv_content, conv_field = conversation_source
                # Resolve parent_id from the upstream appended_entry
                # (set by the .appended event dispatch path). None on
                # other trigger paths — the entry's parent_id stays
                # unset, which is harmless.
                parent_id = None
                if triggering_entry:
                    parent_id = triggering_entry.get("id")
                refusal_payload = {
                    "kind": "assistant",
                    "type": "refusal",
                    "body": refusal_state["reason"],
                }
                if parent_id:
                    refusal_payload["parent_id"] = parent_id
                # User dict for _do_append. The runtime has the
                # invoked_by Principal; map to the legacy user-dict
                # shape _do_append expects until the WS auth refactor
                # (slice 7.5).
                user_dict = {}
                if invoked_by is not None:
                    user_dict = {"id": getattr(invoked_by, "id", "") or ""}
                try:
                    await _do_append(
                        ctx,
                        content_ref=conv_content,
                        key_val=record.get("id") if record else None,
                        field_name=conv_field,
                        payload=refusal_payload,
                        user=user_dict,
                        row_filter=None,
                    )
                except (AppendValidationError, AppendNotFoundError) as e:
                    # The audit row already captured the refusal — log
                    # the secondary failure but don't propagate.
                    print(
                        f"[Termin] [WARN] Compute '{comp_name}': "
                        f"refusal recorded in audit but failed to append "
                        f"to conversation {conv_content}.{conv_field}: {e}"
                    )
            return

        trace_data = {"compute_type": "agent", "calls": [{"response": thinking[:200] if thinking else ""}]}
        if audit_level == "debug":
            trace_data["calls"][0]["system_prompt"] = system_msg
            trace_data["calls"][0]["thinking"] = thinking
        # v0.9 Phase 3 slice (d): BRD §6.3.4 audit_metadata. The
        # legacy AIProvider doesn't yet expose a structured tool-calls
        # list back to the runner, so we pass the empty list here.
        agent_audit_metadata = _build_agent_audit_metadata(
            ctx, comp_snake, system_msg, user_msg, tool_calls_log=[],
        )
        await write_audit_trace(
            ctx, comp, invocation_id=_agent_invocation_id, trigger="event",
            started_at=_agent_started.isoformat().replace("+00:00", "Z"),
            completed_at=_agent_completed.isoformat().replace("+00:00", "Z"),
            latency_ms=_agent_duration, outcome="success",
            trace_data=trace_data,
            audit_metadata=agent_audit_metadata,
            invoked_by=invoked_by,
        )
    except AIProviderError as e:
        print(f"[Termin] [ERROR] Compute '{comp_name}': {e}")
        _agent_err_completed = _dt.datetime.now(_dt.timezone.utc)
        _agent_err_duration = (_agent_err_completed - _agent_started).total_seconds() * 1000
        agent_audit_metadata = _build_agent_audit_metadata(
            ctx, comp_snake, system_msg, user_msg, tool_calls_log=[],
        )
        await write_audit_trace(
            ctx, comp, invocation_id=_agent_invocation_id, trigger="event",
            started_at=_agent_started.isoformat().replace("+00:00", "Z"),
            completed_at=_agent_err_completed.isoformat().replace("+00:00", "Z"),
            latency_ms=_agent_err_duration, outcome="error",
            error_message=str(e),
            trace_data={"compute_type": "agent", "error": str(e)},
            audit_metadata=agent_audit_metadata,
            invoked_by=invoked_by,
        )


# ── D-20: Audit trace recording ──

async def write_audit_trace(ctx: RuntimeContext, comp: dict, invocation_id: str,
                            trigger: str, started_at: str, completed_at: str,
                            latency_ms: float = 0.0, outcome: str = "success",
                            trace_data: dict = None, error_message: str = None,
                            total_input_tokens: int = 0, total_output_tokens: int = 0,
                            invoked_by=None,
                            audit_metadata: dict = None):
    """Write a trace record to the compute's audit log Content table.

    Per BRD §6.3.4 (v0.9 Phase 3 slice (d)):
      - `latency_ms` is the canonical column name (renamed from
        `duration_ms` in v0.8). The transitional `duration_ms=` kwarg
        was dropped along with the back-compat shim before Phase 7.
      - `audit_metadata` carries the BRD §6.3.4 reproducibility-grade
        fields for LLM/agent invocations: provider_product,
        model_identifier, provider_config_hash, prompt_as_sent,
        sampling_params (JSON), tool_calls (JSON), refusal_reason,
        cost_{units,unit_type,currency_amount}. Missing keys default
        to safe values (empty strings / 0 / null). CEL computes pass
        None.

    invoked_by: optional Principal who triggered the compute. For
        event-triggered computes this is the principal who caused
        the upstream event; for system-triggered computes (scheduler,
        startup hooks) it's None and the audit fields are empty.
        For delegate-mode agent principals, on_behalf_of is also
        recorded so the audit trail captures 'agent X acting for
        user Y did Z'.
    """
    audit_level = comp.get("audit_level", "actions")
    audit_ref = comp.get("audit_content_ref")
    if audit_level == "none" or not audit_ref:
        return

    # Per BRD §6.3.4, principal info on the audit record.
    # v0.9.1: anonymous principals get a synthesized
    # ``anonymous:<short>`` id rather than empty-string so audit
    # rows always carry a "proper auditable type" — operators can
    # filter ``invoked_by_principal_id LIKE 'anonymous:%'`` to
    # find anonymous-caller rows. The short suffix is derived
    # from the invocation_id so each anonymous row is uniquely
    # identifiable within the audit trail; the prefix preserves
    # the type information operators care about.
    invoked_by_id = ""
    invoked_by_name = ""
    on_behalf_of_id = ""

    def _synth_anon(raw_id: str) -> str:
        """If the raw id marks an anonymous principal, decorate it
        with the invocation_id short prefix; otherwise return as-is."""
        if raw_id in ("", "anonymous", None):
            short = (invocation_id or "").replace("-", "")[:8] or "unknown"
            return f"anonymous:{short}"
        return raw_id

    if invoked_by is not None:
        raw_id = getattr(invoked_by, "id", "") or ""
        invoked_by_id = _synth_anon(raw_id)
        raw_name = getattr(invoked_by, "display_name", "") or ""
        if not raw_name and invoked_by_id.startswith("anonymous:"):
            invoked_by_name = "Anonymous"
        else:
            invoked_by_name = raw_name

        # Per BRD §6.3.4: on_behalf_of is the *chain target* — only
        # populated when the Principal carries an explicit
        # ``on_behalf_of`` reference (delegate-mode agents
        # constructed with the upstream user attached). For a
        # principal acting as themselves (humans, anonymous, or
        # service-mode agents) the column stays empty — the
        # column's signal is precisely whether the row represents
        # an X-acting-for-Y chain. This matches the v0.9.0
        # ``test_v09_identity_contract.py::TestAuditLogPrincipalRecording``
        # invariants.
        obo = getattr(invoked_by, "on_behalf_of", None)
        if obo is not None:
            obo_raw = getattr(obo, "id", "") or ""
            on_behalf_of_id = _synth_anon(obo_raw)

    trace_json = json.dumps(trace_data) if trace_data else "{}"
    record_data = {
        "compute_name": comp["name"]["display"],
        "invocation_id": invocation_id,
        "trigger": trigger,
        "started_at": started_at,
        "completed_at": completed_at,
        "latency_ms": latency_ms,
        "outcome": outcome,
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
        "trace": trace_json,
        "error_message": error_message or "",
        "invoked_by_principal_id": invoked_by_id,
        "invoked_by_display_name": invoked_by_name,
        "on_behalf_of_principal_id": on_behalf_of_id,
    }

    # v0.9 Phase 3 slice (d): LLM/agent invocations carry the
    # reproducibility-grade audit columns from BRD §6.3.4. CEL
    # computes don't get these columns in their audit table — the
    # writer only populates them when the schema includes them
    # (provider in {"llm", "ai-agent"}).
    if comp.get("provider") in ("llm", "ai-agent"):
        m = audit_metadata or {}
        record_data.update({
            "provider_product": m.get("provider_product", ""),
            "model_identifier": m.get("model_identifier", ""),
            "provider_config_hash": m.get("provider_config_hash", ""),
            "prompt_as_sent": m.get("prompt_as_sent", ""),
            "sampling_params": m.get("sampling_params_json", "{}"),
            "tool_calls": m.get("tool_calls_json", "[]"),
            "refusal_reason": m.get("refusal_reason") or "",
            "cost_units": m.get("cost_units") or 0,
            "cost_unit_type": m.get("cost_unit_type", "") or "",
            "cost_currency_amount": m.get("cost_currency_amount", "") or "",
        })

    try:
        db = await get_db(ctx.db_path)
        try:
            await insert_raw(db, audit_ref, record_data)
        finally:
            await db.close()
    except Exception as e:
        print(f"[Termin] [WARN] Failed to write audit trace for '{comp['name']['display']}': {e}")


async def redact_audit_traces(ctx: RuntimeContext, records: list,
                              audit_table_name: str, user_scopes: set) -> list:
    """Apply redaction to audit trace records based on caller scopes."""
    comp = None
    for c in ctx.ir.get("computes", []):
        if c.get("audit_content_ref") == audit_table_name:
            comp = c
            break
    if not comp:
        return records

    all_content_refs = set(
        comp.get("input_content", []) + comp.get("output_content", []) + comp.get("accesses", []))
    redact_fields = []
    for cr in all_content_refs:
        schema = ctx.content_lookup.get(cr, {})
        for field_def in schema.get("fields", []):
            conf_scopes = tuple(field_def.get("confidentiality_scopes", []))
            if conf_scopes and not all(s in user_scopes for s in conf_scopes):
                redact_fields.append((cr, field_def["name"]))

    if not redact_fields:
        return records

    redact_values = []
    try:
        db = await get_db(ctx.db_path)
        try:
            for content_name, field_name in redact_fields:
                try:
                    col_values = await select_column(db, content_name, field_name)
                    for val_raw in col_values:
                        val = str(val_raw) if val_raw is not None else ""
                        if len(val) >= 4:
                            redact_values.append((val, field_name))
                except Exception:
                    pass
        finally:
            await db.close()
    except Exception:
        return records

    if not redact_values:
        return records

    for rec in records:
        trace_val = rec.get("trace", "")
        if trace_val:
            for val, fname in redact_values:
                trace_val = trace_val.replace(val, f"[REDACTED:{fname}]")
            rec["trace"] = trace_val
        err_val = rec.get("error_message", "")
        if err_val:
            for val, fname in redact_values:
                err_val = err_val.replace(val, f"[REDACTED:{fname}]")
            rec["error_message"] = err_val

    return records


def register_compute_endpoint(app, ctx: RuntimeContext):
    """Register the server-side Compute invocation endpoint."""

    @app.post("/api/v1/compute/{compute_name}")
    async def invoke_compute(compute_name: str, request: Request):
        """Execute a Compute server-side with confidentiality checks (Checks 1-4)."""
        comp = ctx.compute_lookup.get(compute_name)
        if not comp:
            raise HTTPException(status_code=404, detail=f"Compute '{compute_name}' not found")

        user = ctx.get_current_user(request)
        user_scopes = set(user.get("scopes", []))
        body = await request.json()
        input_data = body.get("input", {})

        # Check execution permission
        req_scope = comp.get("required_scope")
        if req_scope and req_scope not in user_scopes:
            raise HTTPException(status_code=403, detail=f"Requires scope '{req_scope}' to execute")

        # Check 1: Confidentiality gate
        gate_err = check_compute_access(comp, user_scopes)
        if gate_err:
            ctx.terminator.route(TerminError(
                source=comp["name"]["display"], kind="confidentiality_gate_rejected",
                message=gate_err))
            raise HTTPException(status_code=403, detail=gate_err)

        # Check 2: Taint integrity
        if isinstance(input_data, list) and comp.get("identity_mode") == "service":
            for input_content_name in comp.get("input_content", []):
                schema = ctx.content_lookup.get(input_content_name, {})
                taint_err = check_taint_integrity(input_data, schema, user_scopes)
                if taint_err:
                    ctx.terminator.route(TerminError(
                        source="confidentiality", kind="taint_violation",
                        message=taint_err))
                    raise HTTPException(status_code=500, detail=taint_err)

        # D-20: Audit timing
        _audit_started = _dt.datetime.now(_dt.timezone.utc)
        _audit_started_str = _audit_started.isoformat().replace("+00:00", "Z")

        tx = Transaction()

        compute_ctx = {
            "Compute": {
                "Name": comp["name"]["display"],
                "Provider": comp.get("provider") or "cel",
                "IdentityMode": comp.get("identity_mode", "delegate"),
                "Scopes": list(user_scopes),
                "ExecutionId": tx.id,
                "Trigger": "api",
                "StartedAt": tx.started_at,
            },
            # Slice 7.5b: bind ``the_user`` instead of the legacy ``User``
            # PascalCase shape. Source CEL spells references as
            # ``the user.X`` or ``user.X``; both resolve to ``the_user``.
            "the_user": _the_user_for(user),
        }

        # Evaluate preconditions
        for i, precond in enumerate(comp.get("preconditions", [])):
            try:
                result = ctx.expr_eval.evaluate(precond, compute_ctx)
                if not result:
                    tx.rollback()
                    detail = f"Precondition {i+1} failed: {precond}"
                    ctx.terminator.route(TerminError(
                        source=comp["name"]["display"], kind="precondition_failed",
                        message=detail))
                    raise HTTPException(status_code=412, detail=detail)
            except HTTPException:
                raise
            except Exception as e:
                tx.rollback()
                raise HTTPException(status_code=500, detail=f"Precondition evaluation error: {e}")

        # Block C: Boundary enforcement
        comp_snake_name = comp["name"]["snake"]
        for acc_content in comp.get("accesses", []):
            bnd_err = check_boundary_access(
                ctx.boundary_for_compute, ctx.boundary_for_content,
                comp_snake_name, acc_content)
            if bnd_err:
                tx.rollback()
                raise HTTPException(status_code=403, detail=bnd_err)

        # Execute the CEL body
        body_lines = comp.get("body_lines", [])
        if not body_lines:
            raise HTTPException(status_code=400, detail="Compute has no body to execute")

        cel_body = body_lines[0]
        try:
            eval_ctx = dict(compute_ctx)
            if isinstance(input_data, dict):
                eval_ctx.update(input_data)
            elif isinstance(input_data, list):
                for input_name in comp.get("input_content", []):
                    eval_ctx[input_name] = input_data

            # Check 3: CEL redaction guard
            redacted_err = check_for_redacted_values(eval_ctx)
            if redacted_err:
                tx.rollback()
                ctx.terminator.route(TerminError(
                    source="expression", kind="redacted_field_access",
                    message=redacted_err))
                raise HTTPException(status_code=500, detail=redacted_err)

            result = ctx.expr_eval.evaluate(cel_body, eval_ctx)
        except HTTPException:
            raise
        except Exception as e:
            tx.rollback()
            _audit_err_completed = _dt.datetime.now(_dt.timezone.utc)
            _audit_err_duration = (_audit_err_completed - _audit_started).total_seconds() * 1000
            await write_audit_trace(
                ctx, comp, invocation_id=tx.id, trigger="api",
                started_at=_audit_started_str,
                completed_at=_audit_err_completed.isoformat().replace("+00:00", "Z"),
                latency_ms=_audit_err_duration, outcome="error",
                error_message=str(e),
                trace_data={"compute_type": "cel", "expression": cel_body, "error": str(e)},
            )
            raise HTTPException(status_code=500, detail=f"Compute evaluation failed: {e}")

        output = {"result": result, "transaction_id": tx.id}

        # Before/After snapshots for postconditions
        before_data = {"result": None}
        after_data = {"result": result}

        try:
            db = await get_db(ctx.db_path)
            all_content_refs = set(
                comp.get("input_content", []) + comp.get("output_content", [])
                + comp.get("accesses", []))
            for content_name in all_content_refs:
                records = await list_records(db, content_name)
                before_data[content_name] = records
                after_data[content_name] = await tx.read_all(content_name, records)
            await db.close()
        except Exception:
            pass

        before_snapshot_obj = ContentSnapshot(
            {k: v for k, v in before_data.items() if k != "result"}, result=None)
        after_snapshot_obj = ContentSnapshot(
            {k: v for k, v in after_data.items() if k != "result"}, result=result)

        # Evaluate postconditions
        post_ctx = dict(compute_ctx)
        post_ctx["After"] = after_data
        post_ctx["Before"] = before_data
        for i, postcond in enumerate(comp.get("postconditions", [])):
            try:
                check = ctx.expr_eval.evaluate(postcond, post_ctx)
                if not check:
                    tx.rollback()
                    detail = f"Postcondition {i+1} failed: {postcond}"
                    ctx.terminator.route(TerminError(
                        source=comp["name"]["display"], kind="postcondition_failed",
                        message=detail))
                    raise HTTPException(status_code=409, detail=detail)
            except HTTPException:
                raise
            except Exception:
                pass

        # Check 4: Output taint enforcement
        final_output, taint_err = enforce_output_taint(output, comp, user_scopes)
        if taint_err:
            tx.rollback()
            ctx.terminator.route(TerminError(
                source=comp["name"]["display"], kind="output_taint_blocked",
                message=taint_err))
            raise HTTPException(status_code=403, detail=taint_err)

        # D-20: Audit trace on success
        _audit_completed = _dt.datetime.now(_dt.timezone.utc)
        _audit_duration = (_audit_completed - _audit_started).total_seconds() * 1000
        audit_level = comp.get("audit_level", "actions")
        trace_data = {"compute_type": "cel", "expression": cel_body, "output": result}
        if audit_level == "debug":
            trace_data["input"] = input_data
        await write_audit_trace(
            ctx, comp, invocation_id=tx.id, trigger="api",
            started_at=_audit_started_str,
            completed_at=_audit_completed.isoformat().replace("+00:00", "Z"),
            latency_ms=_audit_duration, outcome="success",
            trace_data=trace_data,
        )

        return final_output

    # Slice 7.2.x: bridge POST /api/v1/compute/{compute_name}/trigger
    # to the pure trigger_compute_handler in termin-core. The handler
    # reads check_compute_access through ctx; stash it on first
    # registration so the bridge is the only place this binding
    # lives.
    if not hasattr(ctx, "check_compute_access"):
        ctx.check_compute_access = check_compute_access

    from termin_core.routing import trigger_compute_handler
    from .fastapi_adapter import (
        make_auth_context,
        to_fastapi_response,
        to_termin_request,
    )

    @app.post("/api/v1/compute/{compute_name}/trigger")
    async def trigger_compute(compute_name: str, request: Request):
        """Manually trigger any Compute regardless of declared trigger type.

        The sibling endpoint POST /api/v1/compute/{compute_name} runs CEL
        computes synchronously and returns the result. This endpoint is
        for the provider types whose normal trigger is an event or a
        schedule — llm and ai-agent — so they can be invoked on demand
        for testing, dev-loop iteration, or "re-run on this record"
        workflows.

        Bridge to termin_core.routing.trigger_compute_handler.
        """
        user = ctx.get_current_user(request)
        auth = make_auth_context(user)
        termin_req = await to_termin_request(
            request,
            path_params={"compute_name": compute_name},
            auth=auth,
        )
        response = await trigger_compute_handler(termin_req, ctx)
        return to_fastapi_response(response)


# ── LLM streaming support (v0.8 #7) ──
#
# Two publishers, one for each streaming mode:
#   publish_stream_deltas        — text streaming (stream_complete)
#   publish_agent_stream_events  — tool-use streaming (stream_agent_response)
#
# See docs/termin-streaming-protocol.md for the full protocol.


async def publish_agent_stream_events(event_bus, invocation_id: str,
                                       compute_name: str, stream,
                                       tool_name: str = "set_output"):
    """Pump tool-use stream events from stream_agent_response onto the
    event bus on the tool-use channels described in the protocol:

      compute.stream.<invocation_id>                   (done event)
      compute.stream.<invocation_id>.field.<name>      (field_delta/done)

    Returns the final output dict from the agent's set_output call so
    the caller can persist the result.

    Args:
        event_bus: runtime EventBus.
        invocation_id: UUID assigned at invocation start.
        compute_name: Compute's snake_name.
        stream: async generator yielding event dicts from
            AIProvider.stream_agent_response — shapes:
              {"type":"field_delta","field":<name>,"delta":<text>}
              {"type":"field_done","field":<name>,"value":<final>}
              {"type":"done","output":<dict>}
        tool_name: the tool whose input is being streamed (default
            "set_output").
    """
    base_channel = f"compute.stream.{invocation_id}"
    output = {}
    async for ev in stream:
        etype = ev.get("type")
        if etype == "field_delta":
            field = ev.get("field", "")
            await event_bus.publish({
                "channel_id": f"{base_channel}.field.{field}",
                "data": {
                    "invocation_id": invocation_id,
                    "compute": compute_name,
                    "mode": "tool_use",
                    "tool": tool_name,
                    "field": field,
                    "delta": ev.get("delta", ""),
                    "done": False,
                },
            })
        elif etype == "field_done":
            field = ev.get("field", "")
            value = ev.get("value")
            output[field] = value
            await event_bus.publish({
                "channel_id": f"{base_channel}.field.{field}",
                "data": {
                    "invocation_id": invocation_id,
                    "compute": compute_name,
                    "mode": "tool_use",
                    "tool": tool_name,
                    "field": field,
                    "done": True,
                    "value": value,
                },
            })
        elif etype == "done":
            provider_output = ev.get("output") or {}
            final_output = {**output, **provider_output}
            await event_bus.publish({
                "channel_id": base_channel,
                "data": {
                    "invocation_id": invocation_id,
                    "compute": compute_name,
                    "mode": "tool_use",
                    "tool": tool_name,
                    "done": True,
                    "output": final_output,
                },
            })
            return final_output
    # Stream exited without a top-level done event — emit one.
    await event_bus.publish({
        "channel_id": base_channel,
        "data": {
            "invocation_id": invocation_id,
            "compute": compute_name,
            "mode": "tool_use",
            "tool": tool_name,
            "done": True,
            "output": output,
        },
    })
    return output


async def publish_stream_deltas(event_bus, invocation_id: str,
                                compute_name: str, stream):
    """Iterate the stream generator, publishing each delta to the event
    bus, and return the concatenated final text. Used for text-mode
    streaming (stream_complete).

    Args:
        event_bus: runtime EventBus.
        invocation_id: UUID assigned at invocation start.
        compute_name: the Compute's snake_name (used in event payloads).
        stream: async generator yielding (delta: str, done: bool).

    Returns:
        The concatenated final_text.
    """
    channel = f"compute.stream.{invocation_id}"
    parts = []
    async for delta, done in stream:
        if done:
            # Terminal event: include final_text for latecomers.
            parts.append(delta)
            final_text = "".join(parts)
            await event_bus.publish({
                "channel_id": channel,
                "data": {
                    "invocation_id": invocation_id,
                    "compute": compute_name,
                    "delta": delta,
                    "done": True,
                    "final_text": final_text,
                },
            })
            return final_text
        parts.append(delta)
        await event_bus.publish({
            "channel_id": channel,
            "data": {
                "invocation_id": invocation_id,
                "compute": compute_name,
                "delta": delta,
                "done": False,
            },
        })
    # Stream exited without a done=True signal — treat as terminal.
    final_text = "".join(parts)
    await event_bus.publish({
        "channel_id": channel,
        "data": {
            "invocation_id": invocation_id,
            "compute": compute_name,
            "delta": "",
            "done": True,
            "final_text": final_text,
        },
    })
    return final_text
