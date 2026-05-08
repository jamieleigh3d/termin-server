# Copyright 2026 Jamie-Leigh Blake and Termin project contributors
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Concrete provider implementations namespace for termin-server.

The Provider Protocols + ContractRegistry + ProviderRegistry +
deploy-config parser + binding resolver live in
``termin_core.providers``. Import from there for the contract
surface — this package only carries the IO-bound concrete providers
under ``builtins/`` (SqliteStorageProvider, AnthropicAIProvider,
TailwindDefault SSR renderer, channel webhook/email/messaging
stubs).

v0.9.3 (2026-05-07): the slice 7.1 re-export shims (binding,
contracts, registry, deploy_config, *_contract.py) were deleted
per the no-shims policy. Server-internal code imports from
``termin_core.providers`` directly. External callers
(termin-spectrum-provider, termin-conformance) likewise.
"""
