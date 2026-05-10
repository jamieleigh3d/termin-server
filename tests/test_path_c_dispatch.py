# Copyright 2026 Jamie-Leigh Blake and Termin project contributors
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""v0.9.4 Path C: per-component contract dispatch.

Today the SSR pipeline at `presentation.py::render_component`
dispatches by `node["type"]`, ignoring `node["contract"]`. That means
`Using "<custom-namespace>.<contract>"` overrides in `.termin` source
reach the IR but never reach the bound provider — they get rendered
by whatever default contract the underlying type resolves to (e.g.
`tailwind-default`'s data_table renderer).

Three coordinated changes close that gap:

  1. `_populate_presentation_providers` learns to expand namespace
     bindings for unknown namespaces by consulting the bound
     provider's `declared_contracts`. Without this, `bindings.
     presentation.<custom-ns>: {provider: ...}` silently drops on
     the floor when there's no contract-package YAML.

  2. `render_component` checks `node["contract"]` first; if a
     provider is bound for that contract, dispatches via the
     provider (SSR-capable: call `render_ssr`, inline result;
     CSR-only: emit a mount-point div the bundle hydrates).

  3. `static/termin.js` adds `hydrateCsrMounts()` — walks
     `[data-termin-csr-mount][data-termin-contract]` after bundles
     load and calls `getRenderer(contract)`. (Tested by
     conformance/JS, not here.)

This test file covers (1) and (2) at the Python unit-test level."""

from __future__ import annotations

from typing import Any

import pytest


# ── Fakes used across both groups of tests ──

class _FakeCsrOnlyProvider:
    """Mimics the AirlockProvider shape: declares custom-namespace
    contracts, CSR-only, render_ssr raises."""

    def __init__(self, declared: tuple[str, ...] = ("airlock.cosmic-orb",
                                                     "airlock.scenario-narrative")):
        self.declared_contracts = declared
        self.render_modes = ("csr",)

    def render_ssr(self, contract, ir_fragment, data, principal_context):
        raise NotImplementedError("CSR-only provider")

    def csr_bundle_url(self):
        return "/_termin/providers/airlock-fake/bundle.js"


class _FakeSsrCapableProvider:
    """Mimics a hypothetical SSR-capable custom provider."""

    def __init__(self, declared: tuple[str, ...] = ("custom.greeting",)):
        self.declared_contracts = declared
        self.render_modes = ("ssr", "csr")

    def render_ssr(self, contract, ir_fragment, data, principal_context):
        # Return a tiny markup string that the dispatcher inlines.
        name = (ir_fragment or {}).get("props", {}).get("name", "world")
        return f'<span data-custom-greeting="{contract}">Hello, {name}!</span>'

    def csr_bundle_url(self):
        return None


# ── Group 1: namespace-expansion fallback in _populate_presentation_providers ──

class TestNamespaceExpansionFallback:
    """When the deploy config binds a namespace whose name isn't
    `presentation-base` and isn't in the contract-package registry,
    the populator should still expand the binding — by asking the
    instantiated provider which contracts it declares in that
    namespace. Without this, custom-provider deployments need either
    per-contract bindings (verbose) or a contract-package YAML
    (overhead for the simple case)."""

    def _make_ctx_and_registries(self, deploy_config: dict, provider_factory):
        """Spin up just enough of the runtime to run
        `_populate_presentation_providers`. Avoids the full app-boot
        path because the populator is pure given its inputs."""
        from termin_core.providers.contracts import (
            Category, ContractDefinition, ContractRegistry, Tier,
        )
        from termin_core.providers.registry import ProviderRegistry

        provider_registry = ProviderRegistry()
        contract_registry = ContractRegistry()

        # Register the provider factory under product name "airlock-fake"
        # against a custom contract so the populator can find it. The
        # registry's `all_records()` walks every registered product —
        # the populator's `_get_or_create` looks for ANY presentation
        # record with the matching product name, so registering against
        # one contract is enough.
        contract_registry.register_contract(ContractDefinition(
            name="airlock.cosmic-orb",
            category=Category.PRESENTATION,
            tier=Tier.TIER_2,
            naming="named",
            description="test contract",
        ))
        provider_registry.register(
            category=Category.PRESENTATION,
            contract_name="airlock.cosmic-orb",
            product_name="airlock-fake",
            factory=lambda config: provider_factory(),
            version="0.0.0-test",
            contract_registry=contract_registry,
        )

        # Minimal ctx surface for the populator.
        class _Ctx:
            def __init__(self):
                self.presentation_providers = []
                self.contract_package_registry = None

        return _Ctx(), provider_registry, contract_registry

    def test_unknown_namespace_expands_via_declared_contracts(self):
        """`bindings.presentation.airlock` should expand to every
        `airlock.*` contract the provider declares — without a
        contract-package YAML."""
        from termin_server.app import _populate_presentation_providers

        deploy = {
            "bindings": {
                "presentation": {
                    "airlock": {"provider": "airlock-fake", "config": {}},
                },
            },
        }
        ctx, prov_reg, contract_reg = self._make_ctx_and_registries(
            deploy, lambda: _FakeCsrOnlyProvider(),
        )
        _populate_presentation_providers(ctx, deploy, prov_reg, contract_reg)

        bound_contracts = {c for c, _p, _i in ctx.presentation_providers}
        assert "airlock.cosmic-orb" in bound_contracts, (
            f"namespace binding should expand to airlock.cosmic-orb; "
            f"got {bound_contracts}")
        assert "airlock.scenario-narrative" in bound_contracts, (
            f"namespace binding should expand to airlock.scenario-narrative; "
            f"got {bound_contracts}")

    def test_unknown_namespace_skips_contracts_in_other_namespaces(self):
        """A provider that happens to declare contracts in multiple
        namespaces should only get bound for the namespace the deploy
        config names."""
        from termin_server.app import _populate_presentation_providers

        deploy = {
            "bindings": {
                "presentation": {
                    "airlock": {"provider": "airlock-fake", "config": {}},
                },
            },
        }
        ctx, prov_reg, contract_reg = self._make_ctx_and_registries(
            deploy,
            lambda: _FakeCsrOnlyProvider(
                declared=("airlock.cosmic-orb", "other-ns.thing"),
            ),
        )
        _populate_presentation_providers(ctx, deploy, prov_reg, contract_reg)

        bound_contracts = {c for c, _p, _i in ctx.presentation_providers}
        assert "airlock.cosmic-orb" in bound_contracts
        assert "other-ns.thing" not in bound_contracts, (
            "namespace expansion should not leak across namespaces")

    def test_per_contract_binding_still_works(self):
        """The pre-existing per-contract binding shape (key contains
        a dot) must not regress."""
        from termin_server.app import _populate_presentation_providers

        deploy = {
            "bindings": {
                "presentation": {
                    "airlock.cosmic-orb": {
                        "provider": "airlock-fake", "config": {},
                    },
                },
            },
        }
        ctx, prov_reg, contract_reg = self._make_ctx_and_registries(
            deploy, lambda: _FakeCsrOnlyProvider(),
        )
        _populate_presentation_providers(ctx, deploy, prov_reg, contract_reg)

        bound_contracts = {c for c, _p, _i in ctx.presentation_providers}
        assert bound_contracts == {"airlock.cosmic-orb"}, (
            f"per-contract binding should bind exactly one contract; "
            f"got {bound_contracts}")


# ── Group 2: contract-first dispatch in render_component ──

class TestRenderComponentContractDispatch:
    """`render_component` must check `node["contract"]` before falling
    back to type-based dispatch. CSR-only providers emit a mount-point
    div; SSR-capable providers inline their `render_ssr` output."""

    def test_no_contract_falls_back_to_type_dispatch(self):
        """A node with no `contract` key uses the type-based
        renderer table — preserving the legacy code path."""
        from termin_server.presentation import render_component

        node = {
            "type": "text",
            "props": {"content": "Hello"},
        }
        # No providers list → legacy path.
        out = render_component(node)
        assert "Hello" in out
        # Type-based class signature from _render_text.
        assert 'class="text-lg' in out

    def test_unbound_contract_falls_back_to_type_dispatch(self):
        """A node with a `contract` set but no provider bound for it
        also falls back to type dispatch (defensive — the populator
        may have skipped a misconfigured binding)."""
        from termin_server.presentation import render_component

        node = {
            "type": "text",
            "contract": "airlock.unknown",
            "props": {"content": "Fallback content"},
        }
        out = render_component(node, presentation_providers=[])
        assert "Fallback content" in out
        assert 'class="text-lg' in out

    def test_csr_only_contract_emits_mount_point(self):
        """When a CSR-only provider is bound for the contract, the
        SSR pipeline emits a mount-point div with the IR serialized
        as a data attribute. No call to render_ssr (the provider
        doesn't implement it)."""
        from termin_server.presentation import render_component

        provider = _FakeCsrOnlyProvider()
        node = {
            "type": "data_table",  # legacy type — would normally render a table
            "contract": "airlock.cosmic-orb",
            "props": {"source": "scenes"},
        }
        providers = [
            ("airlock.cosmic-orb", "airlock-fake", provider),
        ]
        out = render_component(node, presentation_providers=providers)

        assert "data-termin-csr-mount" in out, (
            "CSR-only dispatch should emit a mount-point marker")
        assert 'data-termin-contract="airlock.cosmic-orb"' in out, (
            "mount-point must carry the contract name for the JS hydrator")
        # Type-based markup (the data_table renderer's <table> tag)
        # MUST NOT appear — that's the bug Path C closes.
        assert "<table" not in out, (
            "contract dispatch must short-circuit the type-based renderer; "
            f"got: {out!r}")
        assert "data-termin-ir" in out, (
            "mount-point must carry the IR fragment so the renderer "
            "can read its props")

    def test_csr_mount_point_ir_round_trip(self):
        """The serialized IR on the mount-point must round-trip through
        JSON — the JS hydrator parses it back. Special chars in props
        (quotes, ampersands) must not break the HTML attribute."""
        import html
        import json
        from termin_server.presentation import render_component

        provider = _FakeCsrOnlyProvider(
            declared=("airlock.scenario-narrative",))
        node = {
            "type": "data_table",
            "contract": "airlock.scenario-narrative",
            "props": {
                "lines": [
                    {"text": 'A "quoted" line', "kind": "narrative"},
                    {"text": "Another & line", "kind": "alert"},
                ],
            },
        }
        providers = [
            ("airlock.scenario-narrative", "airlock-fake", provider),
        ]
        out = render_component(node, presentation_providers=providers)

        # Extract the data-termin-ir attribute value and decode it.
        import re
        match = re.search(r'data-termin-ir="([^"]*)"', out)
        assert match, f"mount-point should expose data-termin-ir; got {out!r}"
        ir_attr = html.unescape(match.group(1))
        round_tripped = json.loads(ir_attr)
        assert round_tripped["contract"] == "airlock.scenario-narrative"
        assert round_tripped["props"]["lines"][0]["text"] == 'A "quoted" line'

    def test_ssr_capable_contract_inlines_provider_output(self):
        """When the bound provider's render_modes includes 'ssr', the
        dispatcher calls render_ssr and inlines the result — no
        mount-point markup."""
        from termin_server.presentation import render_component

        provider = _FakeSsrCapableProvider()
        node = {
            "type": "text",
            "contract": "custom.greeting",
            "props": {"name": "JL"},
        }
        providers = [
            ("custom.greeting", "custom-fake", provider),
        ]
        out = render_component(node, presentation_providers=providers)
        assert "Hello, JL!" in out
        assert 'data-custom-greeting="custom.greeting"' in out
        # No mount-point marker — SSR provider rendered inline.
        assert "data-termin-csr-mount" not in out

    def test_dispatch_does_not_match_partial_contract_names(self):
        """Lookup must be exact-match on the full contract name —
        otherwise `airlock.cosmic-orb` could spuriously match
        `airlock.cosmic-orb-v2`. Uses tuple equality not prefix."""
        from termin_server.presentation import render_component

        provider = _FakeCsrOnlyProvider(
            declared=("airlock.cosmic-orb-v2",))
        node = {
            "type": "data_table",
            "contract": "airlock.cosmic-orb",  # NOT bound
            "props": {"source": "scenes"},
        }
        providers = [
            ("airlock.cosmic-orb-v2", "airlock-fake", provider),
        ]
        out = render_component(node, presentation_providers=providers)
        # Should fall back to type dispatch, not the v2 provider.
        assert "data-termin-csr-mount" not in out


# ── Group 3: end-to-end via build_page_template ──

class TestPageTemplatePropagatesProviders:
    """`build_page_template` must thread `presentation_providers`
    through to `render_component` so contract dispatch fires for
    every child of the page."""

    def test_build_page_template_dispatches_via_contract(self):
        from termin_server.presentation import build_page_template

        provider = _FakeCsrOnlyProvider(declared=("airlock.cosmic-orb",))
        page = {
            "name": "Smoke",
            "children": [
                {
                    "type": "data_table",
                    "contract": "airlock.cosmic-orb",
                    "props": {"source": "scenes"},
                },
            ],
        }
        providers = [("airlock.cosmic-orb", "airlock-fake", provider)]
        template = build_page_template(page, presentation_providers=providers)
        rendered = template.render()
        assert "data-termin-csr-mount" in rendered
        assert 'data-termin-contract="airlock.cosmic-orb"' in rendered
        assert "<table" not in rendered

    def test_build_merged_page_template_dispatches_via_contract(self):
        from termin_server.presentation import build_merged_page_template

        provider = _FakeCsrOnlyProvider(declared=("airlock.cosmic-orb",))
        pages = [
            {
                "name": "Smoke",
                "role": "Anonymous",
                "children": [
                    {
                        "type": "data_table",
                        "contract": "airlock.cosmic-orb",
                        "props": {"source": "scenes"},
                    },
                ],
            },
        ]
        providers = [("airlock.cosmic-orb", "airlock-fake", provider)]
        template = build_merged_page_template(
            pages, presentation_providers=providers)
        rendered = template.render(current_role="Anonymous")
        assert "data-termin-csr-mount" in rendered
