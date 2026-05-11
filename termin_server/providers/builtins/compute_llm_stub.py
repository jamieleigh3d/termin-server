# Copyright 2026 Jamie-Leigh Blake and Termin project contributors
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0

"""Stub LLM compute provider — first-party plugin against the v0.9
llm contract surface (BRD §6.3.2).

Scripted-response stub for deterministic tests and local development.
Does not call any LLM SDK; returns pre-configured outputs (or refuses
when configured to refuse). Same loading path as the real Anthropic
provider.

Per BRD §10 ("Stub providers required for every contract"), every
named contract ships with a stub product so dev/test deploy configs
can bind to a deterministic implementation.

Configuration shape (deploy_config["bindings"]["compute"]["<name>"]
["config"]):
    {
        "responses": {
            "<directive_or_objective_substring>": {
                "outcome": "success",
                "output_value": <any>,
            },
            ...
        },
        "default_response": { ... },        # optional fallback
        "model_identifier": "stub-llm-1",   # for audit record
    }

Tests that need scripted refusals or errors set the corresponding
outcome value plus refusal_reason / error_detail.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional

from termin_core.providers.contracts import Category, ContractRegistry
from termin_core.providers.compute_contract import (
    AuditRecord, CompletionResult, LlmComputeProvider,
)
from ._provider_hash import hash_provider_config

from termin_server import __version__


class StubLlmProvider:
    """Scripted LLM completions for tests.

    The provider matches the directive + objective against a config-
    supplied response map. Match shape is "first key whose substring
    appears in directive+objective"; this lets tests configure
    distinct responses for distinct prompts without exact-match
    brittleness.

    If no key matches and `default_response` is set, returns that.
    Otherwise returns a generic success with output_value=None.
    """

    def __init__(self, config: Mapping[str, Any] | None = None) -> None:
        self._config = dict(config or {})
        self._responses: dict[str, dict] = dict(self._config.get("responses", {}))
        self._default: Optional[dict] = self._config.get("default_response")
        self._model_id: str = str(self._config.get("model_identifier", "stub-llm-1"))
        self._config_hash = hash_provider_config(self._config)

    @property
    def is_configured(self) -> bool:
        """v0.9.4 (server issue #1): the stub provider is always
        configured. The compute runner gates execution on this
        property; without it, stub-bound computes are silently
        skipped. The stub exists to exercise plumbing without
        external dependencies — nothing to wait for.
        """
        return True

    @property
    def service(self) -> str:
        """v0.9.4 (server issue #1 follow-up): the compute runner
        reads `provider.service` for log lines + special-casing
        Anthropic plumbing. Returns "stub" so log shape stays
        consistent."""
        return "stub"

    @property
    def model(self) -> str:
        """Stub providers default to "stub-llm-1" unless overridden
        via deploy config `model_identifier`."""
        return self._model_id

    @property
    def legacy(self) -> "_StubLlmLegacyAdapter":
        """v0.9.4 (server issue #2): the compute runner routes
        Level-1 (LLM) computes through ``provider.legacy.complete``.
        This adapter translates that to the stub's response-map
        matcher so stub-bound LLM computes work without external
        dependencies. Slated for removal in v0.10 slice (c)."""
        return _StubLlmLegacyAdapter(self)

    async def complete(
        self,
        directive: str,
        objective: str,
        input_value: Any,
        output_schema: Optional[Mapping[str, Any]] = None,
        sampling_params: Optional[Mapping[str, Any]] = None,
    ) -> CompletionResult:
        prompt = f"{directive}\n{objective}"
        scripted = self._match(prompt)
        outcome = scripted.get("outcome", "success")

        audit = AuditRecord(
            provider_product="stub",
            model_identifier=self._model_id,
            provider_config_hash=self._config_hash,
            prompt_as_sent=prompt,
            sampling_params=dict(sampling_params or {}),
            outcome=outcome,
            refusal_reason=scripted.get("refusal_reason"),
            error_detail=scripted.get("error_detail"),
            cost=None,
            latency_ms=0,
        )
        return CompletionResult(
            outcome=outcome,
            output_value=scripted.get("output_value"),
            refusal_reason=scripted.get("refusal_reason"),
            error_detail=scripted.get("error_detail"),
            audit_record=audit,
        )

    def _match(self, prompt: str) -> dict:
        for key, response in self._responses.items():
            if key in prompt:
                return response
        if self._default is not None:
            return dict(self._default)
        return {"outcome": "success", "output_value": None}


# ── Legacy adapter (v0.9.4 server issue #2) ──


class _StubLlmLegacyAdapter:
    """Adapter exposing ``complete(system_prompt, user_message,
    output_tool)`` for the runtime's legacy LLM call path
    (compute_runner.py:522). Translates to the stub's response-map
    matcher; the matched response's ``output_value`` IS the dict
    the runtime expects from a forced-tool_use call.

    Per compute_runner.py:425, ``.legacy`` is slated for removal in
    v0.10 slice (c). When that lands, the runtime calls the modern
    Protocol (``complete``) directly and this adapter goes away.
    """

    def __init__(self, provider: "StubLlmProvider") -> None:
        self._provider = provider

    async def complete(
        self,
        system_prompt: str,
        user_message: str,
        output_tool: dict,
    ) -> dict:
        """Match the response map against system_prompt + user_message
        and return the response's output_value merged with a
        ``thinking`` key (the runtime pops thinking off the dict and
        treats the rest as set_output field values)."""
        prompt = f"{system_prompt}\n{user_message}"
        scripted = self._provider._match(prompt)
        output_value = scripted.get("output_value")
        result: dict = {}
        if isinstance(output_value, dict):
            result.update(output_value)
        # The runtime pops "thinking" via result.pop("thinking", "")
        # and prints it to logs. Carry through any thinking the
        # script declared, otherwise empty string.
        if "thinking" not in result:
            result["thinking"] = scripted.get("thinking", "")
        return result


# ── Registration ──


def _stub_llm_factory(config: Mapping[str, Any]) -> StubLlmProvider:
    return StubLlmProvider(config)


def register_stub_llm(
    provider_registry, contract_registry: ContractRegistry | None = None
) -> None:
    """Register the stub LLM provider against (compute, "llm")."""
    provider_registry.register(
        category=Category.COMPUTE,
        contract_name="llm",
        product_name="stub",
        factory=_stub_llm_factory,
        conformance="passing",
        version=__version__,
        contract_registry=contract_registry,
    )
