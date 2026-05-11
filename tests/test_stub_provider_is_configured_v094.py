# Copyright 2026 Jamie-Leigh Blake and Termin project contributors
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Server #1: stub providers must report is_configured=True.

The compute runner gates execution on
``provider.is_configured`` (compute_runner.py lines 428 + 612):

    if provider is None or not getattr(provider, "is_configured", False):
        print(f"[Termin] Compute '{comp_name}': no provider bound, skipped")
        return

The Anthropic provider returns True when api_key is set; the
stub providers (compute_agent_stub, compute_llm_stub) didn't
define is_configured at all, so getattr fell through to False
and stub-bound computes were silently skipped at every
invocation. Surfaced by Airlock-on-Termin slice A3b smoke
(plumbing-verification mode unusable without working stub).

Fix: stub providers report is_configured=True. They're not
waiting on external configuration — that's the whole point.
"""

from __future__ import annotations

import pytest


class TestStubAgentProviderIsConfigured:
    def test_default_construction(self):
        from termin_server.providers.builtins.compute_agent_stub import (
            StubAgentProvider,
        )
        provider = StubAgentProvider()
        assert provider.is_configured is True

    def test_service_property_returns_stub(self):
        """v0.9.4 follow-up: the compute runner reads
        provider.service for log shape + Anthropic special-casing.
        Stub returns 'stub' so log lines render cleanly and the
        Anthropic-specific paths skip."""
        from termin_server.providers.builtins.compute_agent_stub import (
            StubAgentProvider,
        )
        assert StubAgentProvider().service == "stub"

    def test_model_property_returns_default(self):
        from termin_server.providers.builtins.compute_agent_stub import (
            StubAgentProvider,
        )
        assert StubAgentProvider().model == "stub-agent-1"

    def test_model_property_honors_config_override(self):
        from termin_server.providers.builtins.compute_agent_stub import (
            StubAgentProvider,
        )
        provider = StubAgentProvider(config={"model_identifier": "custom-id"})
        assert provider.model == "custom-id"

    def test_with_config(self):
        from termin_server.providers.builtins.compute_agent_stub import (
            StubAgentProvider,
        )
        provider = StubAgentProvider(config={"scripts": {}})
        assert provider.is_configured is True

    def test_with_empty_config(self):
        from termin_server.providers.builtins.compute_agent_stub import (
            StubAgentProvider,
        )
        provider = StubAgentProvider(config={})
        assert provider.is_configured is True


class TestStubLlmProviderIsConfigured:
    def test_default_construction(self):
        from termin_server.providers.builtins.compute_llm_stub import (
            StubLlmProvider,
        )
        provider = StubLlmProvider()
        assert provider.is_configured is True

    def test_with_config(self):
        from termin_server.providers.builtins.compute_llm_stub import (
            StubLlmProvider,
        )
        provider = StubLlmProvider(config={"model_identifier": "test"})
        assert provider.is_configured is True
