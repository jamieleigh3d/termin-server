# Copyright 2026 Jamie-Leigh Blake and Termin project contributors
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0

"""Termin Server — the reference FastAPI hosting layer for compiled
Termin applications.

Slice 7.3 of Phase 7 (2026-04-30) split this package out of
``termin-compiler/termin_runtime/``. Existing imports of
``termin_runtime.X`` continue to work via re-export shims in the
compiler tree; the canonical home is here. The shims drop in
slice 7.5.

Usage:
    from termin_server import create_termin_app
    app = create_termin_app(ir_json_string)
"""

# Canonical package version per docs/version-policy.md §2.1 in
# termin-compiler. release.py bumps THIS value; everywhere else that
# needs the package version (provider records, runtime_version
# reflection, test assertions) imports it from here.
#
# IMPORTANT: this assignment must come BEFORE the .app import below.
# Submodules (routes, providers/builtins/*) do `from . import
# __version__` at their module-load time; if __version__ isn't yet
# defined when those imports trigger, Python raises ImportError.
__version__ = "0.9.3"

from .app import create_termin_app  # noqa: E402 — see __version__ note

__all__ = ["create_termin_app", "__version__"]
