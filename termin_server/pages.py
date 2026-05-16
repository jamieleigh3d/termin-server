# Copyright 2026 Jamie-Leigh Blake and Termin project contributors
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0

"""Page route generation — presentation layer routing and form handling.

Registers page GET routes (with data loading, CEL evaluation, and template
rendering) and form POST routes (with validation, default evaluation, and
redirect logic).
"""

import json
import re

from fastapi import HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse

from termin_core.expression.compute_js import build_compute_js
from termin_core.presentation.compose import extract_page_reqs

from .context import RuntimeContext
from .storage import get_db, create_record, update_record, list_records, find_by_field
from termin_core.providers import Eq, QueryOptions
from termin_core.confidentiality.redaction import redact_records
from .presentation import build_nav_html, build_base_template, build_page_template, build_merged_page_template
from termin_core.validation import evaluate_field_defaults
from .bootstrap import page_should_use_shell, render_shell_response


def register_page_routes(app, ctx: RuntimeContext):
    """Register all page GET/POST routes."""

    nav_html = build_nav_html(ctx.ir.get("nav_items", []), list(ctx.roles.keys()))
    base_template = build_base_template(ctx.ir.get("name", "Termin App"), nav_html)

    # Group pages by slug
    pages_by_slug: dict[str, list] = {}
    for page in ctx.ir.get("pages", []):
        pages_by_slug.setdefault(page["slug"], []).append(page)

    # v0.9.4 Path C: thread the bound presentation providers into the
    # template builders so per-component contract overrides reach
    # their bound providers. Without this, `Using "<ns>.<contract>"`
    # silently drops at render time and the type-default renderer
    # (e.g. tailwind-default `_render_data_table`) handles the node.
    presentation_providers = getattr(ctx, "presentation_providers", []) or []
    page_templates = {}
    for slug, pages_list in pages_by_slug.items():
        if len(pages_list) == 1:
            page_templates[slug] = build_page_template(
                pages_list[0],
                presentation_providers=presentation_providers,
            )
        else:
            page_templates[slug] = build_merged_page_template(
                pages_list,
                presentation_providers=presentation_providers,
            )

    compute_js = build_compute_js(ctx.ir)

    # Home redirect
    if ctx.ir.get("pages"):
        first_slug = ctx.ir["pages"][0]["slug"]

        @app.get("/", response_class=HTMLResponse)
        async def home():
            return RedirectResponse(url=f"/{first_slug}")

    # Page routes — one per unique slug
    emitted_slugs: set = set()
    for page in ctx.ir.get("pages", []):
        slug = page["slug"]
        if slug in emitted_slugs:
            continue
        emitted_slugs.add(slug)
        reqs = extract_page_reqs(page)

        # v0.9.4 Phase 2: detail-page binding. When PageEntry carries
        # a non-empty record_binding, register the route as
        # `/<slug>/{id}` instead of `/<slug>`. The handler fetches
        # the record server-side (404 if missing, ownership-scoped),
        # then renders the page through the same template path —
        # bound-record propagation happens client-side: the
        # React component reads the {id} from the URL and fetches
        # via /api/v1/<plural>/<id>. The v0.10 bound_data pass will
        # thread the record dict into the IR fragments natively;
        # for v0.9.4 the URL-driven fetch keeps the runtime change
        # to one new handler.
        record_binding = page.get("record_binding") or ""
        if record_binding:
            _register_detail_page_get(
                app, ctx, page, slug, record_binding,
                page_templates, base_template, compute_js,
            )
            continue  # detail pages don't have form posts

        _register_page_get(app, ctx, page, slug, reqs, page_templates,
                           base_template, compute_js)

        if reqs["form_target"]:
            _register_form_post(app, ctx, page, slug, reqs)


def _register_page_get(app, ctx, page, slug, page_reqs, page_templates,
                       base_template, compute_js):
    @app.get(f"/{slug}", response_class=HTMLResponse)
    async def page_route(request: Request, _pg=page, _sl=slug, _reqs=page_reqs):
        # v0.9 Phase 5b.4 B' loop: page-route cut-over. When the bound
        # presentation provider for `presentation-base.page` is CSR-only
        # (e.g., spectrum), the SSR-Tailwind pipeline below would render
        # markup the bundle then has to reconcile React against — wasted
        # work and a flash of wrong content. Short-circuit to the shell
        # HTML in that case; the bundle's renderer takes over from the
        # bootstrap payload. Per the page_should_use_shell contract this
        # is a no-op when no CSR-only provider is bound (legacy default).
        if page_should_use_shell(ctx):
            return await render_shell_response(ctx, request, f"/{_sl}")

        user = ctx.get_current_user(request)
        q = request.query_params.get("q", "")
        db = await get_db(ctx.db_path)
        try:
            all_transitions = {}
            # Per-content transition lists, used by the edit modal JS to
            # filter state dropdowns to valid targets reachable from the
            # current row state and allowed by the user's scopes.
            # Shape: {content_ref: [{from, to, scope, machine_name}, ...]}.
            # With multi-SM the list may contain entries from multiple
            # machines; the edit modal JS filters by `machine_name` per
            # dropdown.
            sm_transitions_by_content = {}
            # Per-(content, machine_name) transitions for templates that
            # want machine-scoped lookup. Backward-compatibility flat
            # `_sm_transitions` (a union over all machines on all contents)
            # is preserved for templates that key directly on (from, to).
            sm_transitions_by_machine = {}
            for sm_content, sm_list in ctx.sm_lookup.items():
                sm_transitions_by_content[sm_content] = []
                for sm in sm_list:
                    trans = sm.get("transitions", {})
                    machine = sm.get("machine_name", "")
                    # v0.9.4 Gap #3: transition values are now dicts
                    # ({required_scope, condition_expr}) instead of
                    # bare scope strings. The legacy edit-modal JS
                    # dropdown templates expect a flat scope string in
                    # `scope`, so flatten the value at this read site.
                    # Forward-compat: legacy bare-string values still
                    # work via the isinstance check.
                    def _scope_of(gate):
                        if isinstance(gate, dict):
                            return gate.get("required_scope", "")
                        return gate or ""
                    flat_trans = {k: _scope_of(v) for k, v in trans.items()}
                    all_transitions.update(flat_trans)
                    sm_transitions_by_machine[(sm_content, machine)] = flat_trans
                    sm_transitions_by_content[sm_content].extend([
                        {"from": f, "to": t, "scope": _scope_of(s),
                         "machine_name": machine}
                        for (f, t), s in trans.items()
                    ])

            # Slice 7.5b: drop the legacy ``User`` PascalCase binding;
            # source CEL spells the caller as ``the user.X`` /
            # ``user.X``, which both resolve to ``the_user`` after
            # the rewrite. Build via ``build_the_user_for_cel``.
            import datetime
            from termin_core.routing import build_the_user_for_cel
            from termin_server.fastapi_adapter import make_auth_context
            cel_ctx = {
                "the_user": build_the_user_for_cel(make_auth_context(user)),
                "now": datetime.datetime.now(datetime.timezone.utc).isoformat().replace("+00:00", "Z"),
                "today": datetime.date.today().isoformat(),
            }

            def _termin_eval(expression):
                try:
                    return ctx.expr_eval.evaluate(expression, cel_ctx)
                except Exception:
                    return "..."

            # Flash notification params
            flash_msg = request.query_params.get("_flash")
            flash_style = request.query_params.get("_flash_style", "toast")
            flash_level = request.query_params.get("_flash_level", "success")
            flash_dismiss = request.query_params.get("_flash_dismiss")

            # Structural is_anonymous flag derived from the typed
            # Principal — templates should use this rather than string-
            # comparing current_role, which is fragile across casing
            # (v0.9 canonicalized the role name to "Anonymous" but
            # historical templates compared to "anonymous"). Falls back
            # to a case-insensitive role-name check if Principal isn't
            # in the user dict (defensive — every code path through
            # identity.py now puts it there).
            principal = user.get("Principal")
            is_anonymous = (
                principal.is_anonymous if principal is not None
                else str(user.get("role", "")).lower() == "anonymous"
            )
            template_ctx = {
                "page_title": _pg["name"],
                "current_role": user["role"],
                "current_user_name": user["profile"]["DisplayName"],
                "is_anonymous": is_anonymous,
                "user_profile_json": json.dumps(user["profile"]),
                "roles": list(ctx.roles.keys()),
                "q": q,
                "termin_compute_js": compute_js,
                "_sm_transitions": all_transitions,
                "_sm_transitions_by_content": sm_transitions_by_content,
                "_sm_transitions_by_machine": sm_transitions_by_machine,
                "user_scopes": set(user["scopes"]),
                "termin_eval": _termin_eval,
                "flash_msg": flash_msg,
                "flash_style": flash_style,
                "flash_level": flash_level,
                "flash_dismiss": int(flash_dismiss) if flash_dismiss else None,
            }

            # Load data sources via the storage contract. v0.9 Phase 2:
            # page rendering reads through ctx.storage.query — same path
            # the auto-CRUD list route uses. limit=1000 matches the
            # legacy "return all" behavior; large content sets should
            # paginate via the auto-CRUD endpoint.
            user_scopes = set(user.get("scopes", []))
            for src in _reqs["sources"]:
                page = await ctx.storage.query(
                    src, None, QueryOptions(limit=1000),
                )
                records = [dict(r) for r in page.records]
                schema = ctx.content_lookup.get(src, {})
                template_ctx["items"] = redact_records(records, schema, user_scopes)

            # Form reference lists — same path.
            for ref in _reqs["ref_lists"]:
                page = await ctx.storage.query(
                    ref, None, QueryOptions(limit=1000),
                )
                template_ctx[f"{ref}_list"] = [dict(r) for r in page.records]

            content_html = page_templates[_sl].render(**template_ctx)
            return base_template.render(content=content_html, **template_ctx)
        finally:
            await db.close()


def _register_detail_page_get(app, ctx, page, slug, record_binding,
                              page_templates, base_template, compute_js):
    """v0.9.4 Phase 2 — register a detail-page route at /<slug>/{id}.

    Differences from a regular page route:
      - Path includes the {id} segment so each request names one
        record of `record_binding` (the bound plural).
      - The handler fetches the record before rendering so it can
        return 404 cleanly when the id doesn't match (and so the
        404 happens server-side rather than after the React bundle
        has hydrated and failed its own /api/v1/<plural>/<id>
        fetch).
      - Ownership: if the bound content declares `is owned by
        <field>`, the handler verifies the record's owner field
        matches the caller's principal id. A mismatch returns 404
        (same surface as missing, so ownership doesn't leak
        existence — same pattern as the append route in routes.py
        and the auto-CRUD GET handler).
      - The record dict itself is NOT propagated to the client in
        v0.9.4 — the React component reads the {id} from
        `window.location.pathname` and fetches /api/v1/<plural>/<id>
        for the full record. The v0.10 bound_data pass will thread
        the record into IR fragments natively; this v0.9.4 handler
        keeps the server-side change minimal.
    """
    @app.get(f"/{slug}/{{id}}", response_class=HTMLResponse)
    async def detail_route(
        request: Request, id: str, _pg=page, _sl=slug,
        _binding=record_binding,
    ):
        if page_should_use_shell(ctx):
            return await render_shell_response(
                ctx, request, f"/{_sl}/{id}",
            )

        user = ctx.get_current_user(request)

        # Fetch the bound record — returns None if missing.
        record = await ctx.storage.read(_binding, id)
        if not record:
            raise HTTPException(status_code=404, detail="Not found")

        # Ownership filter (same shape as routes.py:611-625) — 404
        # rather than 403 so ownership doesn't leak existence.
        owner_field = None
        if hasattr(ctx, "_owner_field_for_content"):
            owner_field = ctx._owner_field_for_content.get(_binding)
        if owner_field:
            user_id = (user or {}).get("id") if user else None
            if not user_id and isinstance(user, dict):
                the_user = user.get("the_user") or {}
                user_id = the_user.get("id")
            principal = user.get("Principal") if user else None
            if not user_id and principal is not None:
                user_id = getattr(principal, "id", None)
            if owner_field in record and record.get(owner_field) != user_id:
                raise HTTPException(status_code=404, detail="Not found")

        # Build template_ctx the same way the regular page route
        # does — the page template is shared. Defensive: minimal
        # data sources (the detail page binds one record, not a
        # list; the IR-level sources are still passed through for
        # any non-bound directive children).
        import datetime
        from termin_core.routing import build_the_user_for_cel
        from termin_server.fastapi_adapter import make_auth_context

        cel_ctx = {
            "the_user": build_the_user_for_cel(make_auth_context(user)),
            "now": datetime.datetime.now(
                datetime.timezone.utc,
            ).isoformat().replace("+00:00", "Z"),
            "today": datetime.date.today().isoformat(),
        }

        def _termin_eval(expression):
            try:
                return ctx.expr_eval.evaluate(expression, cel_ctx)
            except Exception:
                return "..."

        principal = user.get("Principal")
        is_anonymous = (
            principal.is_anonymous if principal is not None
            else str(user.get("role", "")).lower() == "anonymous"
        )

        template_ctx = {
            "page_title": _pg["name"],
            "current_role": user["role"],
            "current_user_name": user["profile"]["DisplayName"],
            "is_anonymous": is_anonymous,
            "user_profile_json": json.dumps(user["profile"]),
            "roles": list(ctx.roles.keys()),
            "q": "",
            "termin_compute_js": compute_js,
            "_sm_transitions": {},
            "_sm_transitions_by_content": {},
            "_sm_transitions_by_machine": {},
            "user_scopes": set(user.get("scopes", [])),
            "termin_eval": _termin_eval,
            "flash_msg": None,
            "flash_style": "toast",
            "flash_level": "success",
            "flash_dismiss": None,
            # v0.9.4 Phase 2: the bound record. Templates and
            # contracts may read this; the React component on the
            # client also re-fetches via /api/v1/<binding>/{id} so
            # changes-after-render reflect.
            "bound_record": record,
            "bound_record_id": id,
            "items": [record],  # legacy compat for templates that
                                # iterate items even on detail pages
        }

        content_html = page_templates[_sl].render(**template_ctx)
        return base_template.render(content=content_html, **template_ctx)


def _register_form_post(app, ctx, page, slug, reqs):
    ft = reqs["form_target"]
    # v0.9: sm_info is a list of state-machine dicts (one per SM on this
    # content). Empty list = no state machines on this content.
    sm_info = ctx.sm_lookup.get(ft, [])
    create_as = reqs["create_as"]
    unique_fields = reqs["unique_fields"]
    after_save = reqs["after_save"]

    @app.post(f"/{slug}", response_class=HTMLResponse)
    async def form_post(request: Request, _pg=page, _sl=slug, _ft=ft,
                        _sm=sm_info, _ca=create_as,
                        _uf=unique_fields, _as=after_save):
        form = await request.form()
        data = dict(form)
        edit_id = data.pop("edit_id", "")
        record = None
        schema = ctx.content_lookup.get(_ft, {})

        # v0.9 Phase 2: unique-field check uses ctx.storage.query with
        # an Eq predicate — replaces the legacy find_by_field helper.
        if not edit_id and _uf:
            for uf in _uf:
                val = data.get(uf, "")
                if val:
                    page_result = await ctx.storage.query(
                        _ft, Eq(field=uf, value=val), QueryOptions(limit=1),
                    )
                    if page_result.records:
                        raise HTTPException(
                            status_code=409,
                            detail=f"A record with {uf} '{val}' already exists")

        if edit_id:
            try:
                updated = await ctx.storage.update(_ft, edit_id, data)
            except Exception as e:
                from termin_core.errors import TerminError
                if ctx.terminator:
                    ctx.terminator.route(TerminError(
                        source=_ft, kind="validation", message=str(e)))
                raise
            if updated is not None:
                if ctx.event_bus:
                    await ctx.event_bus.publish({
                        "type": f"{_ft}_updated",
                        "channel_id": f"content.{_ft}.updated",
                        "content_name": _ft,
                        "data": dict(updated),
                    })
        else:
            user = ctx.get_current_user(request)
            from termin_server.fastapi_adapter import make_auth_context
            evaluate_field_defaults(
                data, schema, ctx.expr_eval, auth=make_auth_context(user),
            )

            # v0.9 multi-SM: state-machine column initial values are
            # the route's responsibility (provider stays SM-agnostic).
            # `create_as` overrides the initial of the first SM on
            # this content; remaining machines get their declared
            # initial state.
            for sm in (_sm or []):
                col = sm.get("machine_name", "")
                if col and not data.get(col):
                    data[col] = sm.get("initial", "")
            if _ca and _sm:
                data[_sm[0]["machine_name"]] = _ca

            try:
                record = await ctx.storage.create(_ft, data)
            except Exception as e:
                from termin_core.errors import TerminError
                if ctx.terminator:
                    ctx.terminator.route(TerminError(
                        source=_ft, kind="validation", message=str(e)))
                raise
            record = dict(record)
            if ctx.event_bus:
                await ctx.event_bus.publish({
                    "type": f"{_ft}_created",
                    "channel_id": f"content.{_ft}.created",
                    "content_name": _ft,
                    "data": record,
                })
            db = await get_db(ctx.db_path)
            try:
                await ctx.run_event_handlers(db, _ft, "created", record)
            finally:
                await db.close()

        # AJAX response
        accept = request.headers.get("accept", "")
        is_ajax = ("application/json" in accept
                   or request.headers.get("x-requested-with", "").lower() == "xmlhttprequest")
        if is_ajax:
            if edit_id:
                return JSONResponse({"ok": True, "id": edit_id, "action": "updated"})
            elif record:
                return JSONResponse(record)
            else:
                return JSONResponse({"ok": True})

        redirect_url = f"/{_sl}"
        if _as and _as.startswith("return_to:"):
            target_slug = _as.split(":", 1)[1].strip()
            redirect_url = f"/{target_slug}"
        return RedirectResponse(url=redirect_url, status_code=303)
