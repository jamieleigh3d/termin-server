/**
 * Termin Client Runtime — SSR hydration + WebSocket reactive updates.
 *
 * Loaded as an ES module after SSR page render. Connects to the server
 * via WebSocket, subscribes to content changes, and patches the DOM
 * when push events arrive. Progressive enhancement: page works without JS.
 *
 * No build step. No npm dependencies. CEL evaluator loaded from CDN (already in page).
 */

// ── Flash Notification (toast/banner) ──

function showFlashNotification(message, style, level, dismissSeconds) {
  // Remove any existing flash
  document.querySelectorAll("[data-termin-toast], [data-termin-banner]").forEach(el => el.remove());

  const isError = level === "error";
  const el = document.createElement("div");
  el.setAttribute("role", style === "banner" ? "alert" : "status");

  if (style === "banner") {
    el.setAttribute("data-termin-banner", "");
    el.setAttribute("data-level", level);
    el.className = `mb-4 p-4 rounded-lg border ${isError ? "bg-red-50 border-red-200 text-red-800" : "bg-green-50 border-green-200 text-green-800"}`;
    el.innerHTML = `<div class="flex items-center justify-between"><span>${message}</span><button onclick="this.parentElement.parentElement.remove()" class="ml-4 text-lg font-bold opacity-50 hover:opacity-100">&times;</button></div>`;

    // Insert at top of <main>
    const main = document.querySelector("main");
    if (main) main.insertBefore(el, main.firstChild);
  } else {
    // Toast: fixed position bottom-right
    el.setAttribute("data-termin-toast", "");
    el.setAttribute("data-level", level);
    // v0.9 Phase 5a.5: bg-{red,green}-700 brings white-on-color
    // contrast ≥4.5:1 (WCAG AA). Pre-5a.5 used -600 which was 3.3:1.
    el.className = `fixed bottom-4 right-4 z-50 p-4 rounded-lg shadow-lg ${isError ? "bg-red-700 text-white" : "bg-green-700 text-white"}`;
    el.textContent = message;
    document.body.appendChild(el);
  }

  // Auto-dismiss
  const dismiss = dismissSeconds != null ? dismissSeconds : (style === "toast" ? 5 : 0);
  if (dismiss > 0) {
    setTimeout(() => {
      el.style.transition = "opacity 0.3s";
      el.style.opacity = "0";
      setTimeout(() => el.remove(), 300);
    }, dismiss * 1000);
  }
}

const TERMIN_VERSION = "0.3.0";

// ── State ──
const state = {
  registry: null,
  bootstrap: null,
  identity: null,
  ws: null,
  wsUrl: null,
  connected: false,
  subscriptions: new Map(),   // channel_id -> Set<callback>
  cache: new Map(),           // "content.<name>" -> Map<id, record>
  pendingRequests: new Map(), // ref -> { resolve, reject }
  refCounter: 0,
  reconnectDelay: 1000,
  reconnectTimer: null,
  indicator: null,
};

// ── Bootstrap ──

async function init() {
  try {
    // Fetch registry and bootstrap in parallel
    const [registryRes, bootstrapRes] = await Promise.all([
      fetch("/runtime/registry"),
      fetch("/runtime/bootstrap"),
    ]);

    if (!registryRes.ok || !bootstrapRes.ok) {
      console.warn("[Termin] Bootstrap endpoints not available, running in SSR-only mode");
      return;
    }

    state.registry = await registryRes.json();
    state.bootstrap = await bootstrapRes.json();
    state.identity = state.bootstrap.identity;

    // Register client-safe compute functions (if available)
    registerComputes(state.bootstrap.computes || []);

    // Derive WebSocket URL from page origin (same host)
    const proto = location.protocol === "https:" ? "wss:" : "ws:";
    state.wsUrl = `${proto}//${location.host}/runtime/ws`;

    // Create connection indicator
    createIndicator();

    // Connect WebSocket
    connectWebSocket();

    // Hydrate DOM
    hydrateAll();

    // v0.9 Phase 5b.4 platform: load CSR presentation provider
    // bundles. Fire-and-forget — SSR rendering is unaffected by
    // CSR bundle availability; bundles enhance later as they load.
    loadCsrBundles();

    console.log(`[Termin] Client runtime ${TERMIN_VERSION} initialized`);
  } catch (err) {
    console.warn("[Termin] Bootstrap failed, SSR-only mode:", err.message);
  }
}

// ── Compute Registration ──

function registerComputes(computes) {
  // Compute functions are registered on the page-level context object
  // by the server-generated inline script (ctx["FuncName"] = function...).
  // This function is a placeholder for future client-side compute
  // registration from the bootstrap API response.
  // Currently, server-side JS generation handles this in app.py.
}

// ── WebSocket ──

function connectWebSocket() {
  if (!state.wsUrl) return;

  try {
    state.ws = new WebSocket(state.wsUrl);
  } catch (err) {
    console.warn("[Termin] WebSocket creation failed:", err.message);
    scheduleReconnect();
    return;
  }

  state.ws.onopen = () => {
    state.connected = true;
    state.reconnectDelay = 1000;
    updateIndicator(true);
    console.log("[Termin] WebSocket connected");

    // Re-subscribe all active subscriptions. Legacy SSR-mode
    // hydrators register in `state.subscriptions`; provider-mode
    // bundles register in `_subscriptionHandlers` via Termin.subscribe.
    // BOTH must be replayed on (re)connect — provider subscriptions
    // typically race the WebSocket-open event during initial mount,
    // so the in-band sendFrame inside _addSubscription is a no-op
    // until this onopen fires.
    for (const [ch] of state.subscriptions) {
      sendFrame("subscribe", ch, {});
    }
    for (const ch in _subscriptionHandlers) {
      sendFrame("subscribe", ch, {});
    }
  };

  state.ws.onmessage = (event) => {
    try {
      const frame = JSON.parse(event.data);
      console.debug("[Termin] WS recv:", frame.op, frame.ch, frame.ref || "");
      handleFrame(frame);
    } catch (err) {
      console.warn("[Termin] Bad frame:", err.message);
    }
  };

  state.ws.onclose = () => {
    state.connected = false;
    updateIndicator(false);
    console.log("[Termin] WebSocket disconnected");
    scheduleReconnect();
  };

  state.ws.onerror = () => {
    // onclose will fire after onerror
  };
}

function scheduleReconnect() {
  if (state.reconnectTimer) return;
  state.reconnectTimer = setTimeout(() => {
    state.reconnectTimer = null;
    state.reconnectDelay = Math.min(state.reconnectDelay * 1.5, 30000);
    connectWebSocket();
  }, state.reconnectDelay);
}

function sendFrame(op, ch, payload, ref) {
  if (!state.ws || state.ws.readyState !== WebSocket.OPEN) return null;
  ref = ref || `ref-${++state.refCounter}`;
  state.ws.send(JSON.stringify({ v: 1, ch, op, ref, payload }));
  return ref;
}

// ── Frame Handler ──

function handleFrame(frame) {
  const { ch, op, ref, payload } = frame;

  if (op === "push") {
    // Update cache and notify subscribers
    if (ch === "runtime.identity") {
      state.identity = payload;
      return;
    }
    console.log("[Termin] Push:", ch, payload && payload.id ? `id=${payload.id}` : "");
    updateCache(ch, payload);
    notifySubscribers(ch, payload);
    // v0.9 Phase 5b.4 B' loop: parallel dispatch to provider-
    // registered subscription handlers (those registered via
    // Termin.subscribe(channel, handler) from a CSR provider's
    // bundle). The two paths exist side-by-side: legacy SSR mode
    // hydrators run via notifySubscribers; B'-mode provider
    // renderers run via _dispatchToProviderSubscriptions. A single
    // page may use both during cut-over windows.
    _dispatchToProviderSubscriptions(ch, payload);
  } else if (op === "response") {
    // Resolve pending request
    const pending = state.pendingRequests.get(ref);
    if (pending) {
      pending.resolve(payload);
      state.pendingRequests.delete(ref);
    }
    // If this is a subscribe response with current data, populate cache
    if (payload && payload.current) {
      const parts = ch.split(".");
      if (parts.length >= 2 && parts[0] === "content") {
        const contentName = parts[1];
        const cacheKey = `content.${contentName}`;
        const recordMap = new Map();
        for (const record of payload.current) {
          recordMap.set(record.id, record);
        }
        state.cache.set(cacheKey, recordMap);
      }
    }
  } else if (op === "error") {
    const pending = state.pendingRequests.get(ref);
    if (pending) {
      pending.reject(new Error(payload.message || "Unknown error"));
      state.pendingRequests.delete(ref);
    }
  }
}

// ── Cache ──

function updateCache(channelId, data) {
  // Parse: content.<name>.created/updated/deleted
  const parts = channelId.split(".");
  if (parts.length < 3 || parts[0] !== "content") return;

  const contentName = parts[1];
  const action = parts[2]; // created, updated, deleted
  const cacheKey = `content.${contentName}`;

  if (!state.cache.has(cacheKey)) {
    state.cache.set(cacheKey, new Map());
  }
  const records = state.cache.get(cacheKey);

  if (action === "deleted") {
    const id = data.record_id || (data.id);
    if (id != null) records.delete(id);
  } else if (data && data.id != null) {
    records.set(data.id, data);
  }
}

function getCachedRecords(contentName) {
  const cacheKey = `content.${contentName}`;
  const records = state.cache.get(cacheKey);
  return records ? Array.from(records.values()) : [];
}

// ── Subscriptions ──

function subscribe(channelId, callback) {
  if (!state.subscriptions.has(channelId)) {
    state.subscriptions.set(channelId, new Set());
    // Send subscribe frame to server
    sendFrame("subscribe", channelId, {});
  }
  state.subscriptions.get(channelId).add(callback);

  // Return unsubscribe function
  return () => {
    const cbs = state.subscriptions.get(channelId);
    if (cbs) {
      cbs.delete(callback);
      if (cbs.size === 0) {
        state.subscriptions.delete(channelId);
        sendFrame("unsubscribe", channelId, {});
      }
    }
  };
}

function notifySubscribers(channelId, data) {
  // Exact match
  const exact = state.subscriptions.get(channelId);
  if (exact) exact.forEach(cb => cb(channelId, data));

  // Prefix match: "content.products.created" notifies "content.products" subscribers
  const parts = channelId.split(".");
  for (let i = parts.length - 1; i >= 2; i--) {
    const prefix = parts.slice(0, i).join(".");
    const prefixCbs = state.subscriptions.get(prefix);
    if (prefixCbs) prefixCbs.forEach(cb => cb(channelId, data));
  }
}

// Test/dev hook: exposes notifySubscribers so browser-driven tests
// can inject synthetic push events without a live WebSocket round-trip.
// Safe in production — it's just a pointer to the internal dispatcher
// and only activates when test code explicitly calls it.
if (typeof window !== "undefined") {
  window.__TERMIN_NOTIFY__ = notifySubscribers;
}

// ── DOM Hydration ──

function hydrateAll() {
  hydrateDataTables();
  hydrateChatComponents();
  hydrateAggregations();
  hydrateForms();
  hydrateComputeStream();
}

// General-purpose streaming hydrator.
//
// Subscribes once to compute.stream.* for the page and dispatches
// each field_delta / field_done / invocation-done event to the DOM
// element that should render it. The targeting model mirrors the
// distributed-runtime-model doc: a streamed field is a logical
// Channel on a Content field, and any component rendering that field
// is a subscriber. No component type is special — data_table cells,
// text components, form inputs can all receive streaming updates.
//
// Payload shape (post v0.8.1):
//   data.content_name, data.record_id  — targeting keys
//   data.field                         — which schema field
//   data.delta / data.value            — content to render
//   data.done                          — terminal flag
//
// Chat components retain a separate pending-bubble handler for the
// pre-commit case (agent calls content_create — the record doesn't
// yet have an id; invocation_id is the only key). That handler is
// still registered inside hydrateChatComponents() and operates in
// parallel with this one.
function hydrateComputeStream() {
  // Only activate if the page has at least one targetable element
  // (otherwise the subscription is wasted).
  const hasTargets = document.querySelector(
    "[data-termin-row-id] [data-termin-field]");
  if (!hasTargets) return;

  subscribe("compute.stream", (ch, data) => {
    if (!data) return;
    const field = data.field;
    const recordId = data.record_id;
    if (field == null || recordId == null) return;
    // Find every cell/input for this (row, field). Ordinarily one,
    // but a detail view and a table row could both be on the page.
    const selector =
      `[data-termin-row-id="${cssEscape(String(recordId))}"] ` +
      `[data-termin-field="${cssEscape(field)}"]`;
    const targets = document.querySelectorAll(selector);
    if (targets.length === 0) return;
    targets.forEach((el) => {
      if (data.done) {
        // Terminal field event — set the final value if provided,
        // else leave the last delta-accumulated text in place.
        if (data.value != null) el.textContent = String(data.value);
      } else if (typeof data.delta === "string" && data.delta.length > 0) {
        // Intermediate delta — append. The cell's pre-stream
        // textContent (empty for a just-created row) is the starting
        // point. If the cell was already populated by a prior
        // non-stream render, we overwrite on the first delta.
        if (!el.dataset.terminStreaming) {
          el.textContent = "";
          el.dataset.terminStreaming = "1";
        }
        el.textContent = (el.textContent || "") + data.delta;
      }
    });
  });
}

function cssEscape(s) {
  // Quote chars that would break the attribute selector. Modern
  // browsers have CSS.escape(); fall back for older runtimes.
  if (window.CSS && typeof window.CSS.escape === "function") {
    return window.CSS.escape(s);
  }
  return String(s).replace(/["\\]/g, "\\$&");
}

function hydrateChatComponents() {
  const chats = document.querySelectorAll("[data-termin-chat]");
  for (const chat of chats) {
    const source = chat.dataset.terminSource;
    if (!source) continue;

    // v0.9.2 L9: branch on the binding shape. Conversation-field chats
    // (per the v0.9.2 conversation field type tech design §14) bind to a
    // JSON column on a parent record; messages-collection chats (the
    // legacy form) bind to a Content type. The two share the wrapper
    // and the input area but diverge on subscription channel, message
    // shape, and send path.
    if (chat.dataset.terminChatBinding === "conversation-field") {
      hydrateConversationFieldChat(chat);
      continue;
    }

    const messagesContainer = chat.querySelector("[data-termin-chat-messages]");
    if (!messagesContainer) continue;

    // Auto-scroll to bottom on load
    messagesContainer.scrollTop = messagesContainer.scrollHeight;

    // Determine field names from the rendered HTML or fallback to defaults
    const roleField = chat.querySelector("[data-termin-role]")
      ? "data-termin-role" : null;

    // Subscribe for live message updates
    const channelPrefix = `content.${source}`;
    console.log("[Termin] Chat subscribing to:", channelPrefix);
    subscribe(channelPrefix, (ch, data) => {
      const action = ch.split(".")[2]; // created, updated
      if (action === "created" && data) {
        console.log("[Termin] Chat new message:", data);
        // If there is a pending stream bubble for this arrival, remove
        // it — the persisted message now takes over. Match is a
        // best-effort one-pending-bubble-at-a-time heuristic; complex
        // interleaving can be handled in a future iteration.
        const pending = chat.querySelector("[data-termin-chat-pending]");
        if (pending) pending.remove();
        appendChatMessage(messagesContainer, data, chat);
      }
    });

    // v0.8 #7: subscribe to compute.stream.* for token-by-token deltas.
    // Handles both streaming modes per docs/termin-streaming-protocol.md:
    //   mode="text": {invocation_id, delta, done, final_text}
    //   mode="tool_use": {invocation_id, field, delta|value, done}
    //
    // For tool-use mode, the chat component displays only the field
    // whose name matches the chat's content_field (the column shown in
    // rendered messages). Other fields — e.g., a confidence score —
    // may still land on the same stream and are ignored here; a
    // dashboard subscriber could consume them.
    //
    // The chat component's content_field is derived from the rendered
    // markup: the last <div> inside each chat-message is the body. We
    // read it from the data-termin-content-field attribute on the chat
    // element; fallback to common field names.
    const contentField = chat.dataset.terminContentField ||
      inferContentFieldFromMarkup(chat) || "body";
    // Subscribe to the compute.stream prefix. The client-side prefix
    // matcher in notifySubscribers() splits the incoming channel on "."
    // and checks subscription keys without a trailing dot, so we pass
    // the dotted-namespace prefix without trailing dot here. The server
    // uses startswith() which matches either form.
    subscribe("compute.stream", (ch, data) => {
      if (!data) return;
      if (data.error) {
        renderChatStreamError(messagesContainer, data.error, chat);
        return;
      }
      const invId = data.invocation_id;
      if (!invId) return;
      const mode = data.mode || "text";

      if (mode === "tool_use") {
        // Only the chat's content field is rendered into the bubble.
        if (data.field !== contentField) return;
        let pending = chat.querySelector(
          `[data-termin-chat-pending][data-invocation-id="${invId}"]`);
        if (!pending) {
          pending = createPendingChatBubble(messagesContainer, invId);
        }
        const bodyDiv = pending.querySelector(
          "[data-termin-chat-pending-body]");
        if (data.delta && bodyDiv) {
          bodyDiv.textContent = (bodyDiv.textContent || "") + data.delta;
        }
        if (data.done && data.value !== undefined && bodyDiv) {
          // field_done — safety-net overwrite with the final parsed value.
          bodyDiv.textContent = data.value;
        }
      } else {
        // Text mode.
        let pending = chat.querySelector(
          `[data-termin-chat-pending][data-invocation-id="${invId}"]`);
        if (!pending) {
          pending = createPendingChatBubble(messagesContainer, invId);
        }
        const bodyDiv = pending.querySelector(
          "[data-termin-chat-pending-body]");
        if (data.delta && bodyDiv) {
          bodyDiv.textContent = (bodyDiv.textContent || "") + data.delta;
        }
        if (data.done && data.final_text && bodyDiv &&
            bodyDiv.textContent !== data.final_text) {
          bodyDiv.textContent = data.final_text;
        }
      }
      messagesContainer.scrollTop = messagesContainer.scrollHeight;
    });
  }
}

function inferContentFieldFromMarkup(chatEl) {
  // Heuristic: the chat component renders each message's body inside
  // <div>{{ item.<field> }}</div>. We can't introspect the Jinja source
  // at runtime. Fall back to inspecting rendered messages for a known
  // field name. Returns null if we can't infer.
  return null;
}

function createPendingChatBubble(container, invId) {
  const wrapper = document.createElement("div");
  wrapper.className = "flex justify-start";
  wrapper.setAttribute("data-termin-chat-pending", "");
  wrapper.setAttribute("data-invocation-id", invId);
  const bubble = document.createElement("div");
  bubble.className = "bg-gray-200 text-gray-800 rounded-lg px-4 py-2 max-w-[70%] opacity-80";
  const roleLabel = document.createElement("div");
  roleLabel.className = "text-xs opacity-70 mb-1";
  roleLabel.textContent = "assistant";
  const bodyDiv = document.createElement("div");
  bodyDiv.setAttribute("data-termin-chat-pending-body", "");
  bubble.appendChild(roleLabel);
  bubble.appendChild(bodyDiv);
  wrapper.appendChild(bubble);
  container.appendChild(wrapper);
  container.scrollTop = container.scrollHeight;
  return wrapper;
}

function renderChatStreamError(container, errorMsg, chatEl) {
  const wrapper = document.createElement("div");
  wrapper.className = "flex justify-start";
  wrapper.setAttribute("data-termin-chat-stream-error", "");
  const bubble = document.createElement("div");
  bubble.className = "bg-red-100 text-red-800 border border-red-300 rounded-lg px-4 py-2 max-w-[70%]";
  bubble.textContent = "Stream error: " + errorMsg;
  wrapper.appendChild(bubble);
  container.appendChild(wrapper);
}

function appendChatMessage(container, data, chatEl) {
  // Determine role and content field names from the chat component
  const firstMsg = chatEl.querySelector("[data-termin-chat-message]");
  // Extract role from data — try common field names
  const role = data.role || data.sender || "user";
  const body = data.body || data.content || data.message || "";

  const isUser = role === "user";
  const wrapper = document.createElement("div");
  wrapper.className = `flex ${isUser ? "justify-end" : "justify-start"}`;
  wrapper.setAttribute("data-termin-chat-message", "");
  wrapper.setAttribute("data-termin-role", role);

  const bubble = document.createElement("div");
  bubble.className = `${isUser ? "bg-blue-500 text-white" : "bg-gray-200 text-gray-800"} rounded-lg px-4 py-2 max-w-[70%]`;

  const roleLabel = document.createElement("div");
  roleLabel.className = "text-xs opacity-70 mb-1";
  roleLabel.textContent = role;

  const bodyDiv = document.createElement("div");
  bodyDiv.textContent = body;

  bubble.appendChild(roleLabel);
  bubble.appendChild(bodyDiv);
  wrapper.appendChild(bubble);
  container.appendChild(wrapper);

  // Auto-scroll to bottom
  container.scrollTop = container.scrollHeight;
}

// ── v0.9.2 L9: conversation-field chat ──
//
// The Tailwind built-in chat provider for the new
// `Show a chat for <content>.<field>` binding form. Per the v0.9.2
// conversation field type tech design §14, this branch:
//
// 1. Fetches the parent record (most-recent visible) via the standard
//    `/api/v1/<source>` listing endpoint, picks one, and reads the
//    `<conversation_field>` JSON column.
// 2. Renders one bubble per §7.2 entry, dispatched on `kind` and (for
//    assistant entries) `type`. Sets data-termin-{kind,type,source,
//    tool-name} on each entry root so app CSS can override appearance
//    per the §14.3 customization layer without touching source.
// 3. Subscribes to `content.<source>.<field>.appended` (L5 event)
//    and appends new entries with a fade-in animation.
// 4. Wires the input form to the L4 WebSocket `append` frame; sends
//    `{type:"append", resource, id, field, payload:{kind:"user", body}}`.
//
// Pairs `tool_call` and `tool_result` entries by `tool_call_id` and
// renders the result as a collapsible inline detail under the call.
//
// JL is colorblind: every kind/type distinction also carries a label,
// icon, position, and aria attribute — color is supplementary, never
// the only signal.
function hydrateConversationFieldChat(chat) {
  const source = chat.dataset.terminSource;
  const field = chat.dataset.terminConversationField;
  const messagesContainer = chat.querySelector("[data-termin-chat-messages]");
  if (!source || !field || !messagesContainer) return;

  // Track the active record id once resolved. The conversation lives on
  // a record; the hydrator picks one to display. v0.9.2 default:
  // most-recent visible record. Authors who want a pinned record can
  // set `data-termin-record-id` on the chat root before hydration.
  let recordId = chat.dataset.terminRecordId || null;

  // Initial render: load the parent record (creating one if none
  // exists), walk its conversation list, render entries.
  resolveActiveRecord(source, recordId).then((rec) => {
    if (!rec) {
      renderChatPlaceholder(messagesContainer,
        "No conversation yet. Send a message to start.");
      return;
    }
    recordId = rec.id;
    chat.dataset.terminRecordId = String(recordId);
    const placeholder = messagesContainer.querySelector(
      "[data-termin-chat-placeholder]");
    if (placeholder) placeholder.remove();
    const entries = Array.isArray(rec[field]) ? rec[field] : [];
    renderConversationEntries(messagesContainer, entries);
    messagesContainer.scrollTop = messagesContainer.scrollHeight;
  }).catch((err) => {
    renderChatPlaceholder(messagesContainer,
      "Could not load conversation: " + (err && err.message || err));
  });

  // Subscribe to the field-specific appended event (L5).
  const channel = `content.${source}.${field}.appended`;
  console.log("[Termin] Conversation chat subscribing to:", channel);
  subscribe(channel, (ch, data) => {
    if (!data || !data.appended_entry) return;
    // Filter by record id once the active record is known. Until then,
    // the first append we see (typically our own) determines it.
    if (recordId == null) {
      recordId = data.record_id;
      chat.dataset.terminRecordId = String(recordId);
    } else if (data.record_id != null && String(data.record_id) !== String(recordId)) {
      return;
    }
    const placeholder = messagesContainer.querySelector(
      "[data-termin-chat-placeholder]");
    if (placeholder) placeholder.remove();
    appendConversationEntry(messagesContainer, data.appended_entry, /*isLive*/true);
    messagesContainer.scrollTop = messagesContainer.scrollHeight;
  });

  // Wire the input form to the L4 WS append frame.
  const form = chat.querySelector("[data-termin-chat-form]");
  if (form) {
    form.addEventListener("submit", (e) => {
      e.preventDefault();
      const input = form.querySelector("input[name='body']");
      const body = input && input.value ? String(input.value).trim() : "";
      if (!body) return;
      if (recordId == null) {
        // No record yet — create one synchronously so we have an id to
        // append against. The default content type usually has a
        // `title` field; let storage defaults fill the rest.
        ensureRecord(source).then((rec) => {
          recordId = rec.id;
          chat.dataset.terminRecordId = String(recordId);
          sendAppendFrame(source, recordId, field, {kind: "user", body});
          input.value = "";
        }).catch((err) => {
          console.warn("[Termin] Could not create record for conversation:", err);
        });
        return;
      }
      sendAppendFrame(source, recordId, field, {kind: "user", body});
      input.value = "";
    });
  }
}

function renderChatPlaceholder(container, text) {
  let placeholder = container.querySelector("[data-termin-chat-placeholder]");
  if (!placeholder) {
    placeholder = document.createElement("div");
    placeholder.className = "text-sm text-gray-500";
    placeholder.setAttribute("data-termin-chat-placeholder", "");
    container.appendChild(placeholder);
  }
  placeholder.textContent = text;
}

async function resolveActiveRecord(source, pinnedId) {
  if (pinnedId != null) {
    const r = await fetch(`/api/v1/${source}/${pinnedId}`);
    if (r.ok) return await r.json();
  }
  const r = await fetch(`/api/v1/${source}`);
  if (!r.ok) return null;
  const records = await r.json();
  if (!Array.isArray(records) || records.length === 0) return null;
  // Most-recent visible record. The list endpoint orders by id (UUIDv7
  // for content with that default; otherwise insertion order). The last
  // entry in the array is the latest in either case.
  return records[records.length - 1];
}

async function ensureRecord(source) {
  const r = await fetch(`/api/v1/${source}`, {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify({}),
  });
  if (!r.ok) throw new Error(`Could not create ${source}: ${r.status}`);
  return await r.json();
}

function sendAppendFrame(resource, id, field, payload) {
  // The L4 append frame doesn't ride the standard {v, ch, op, ref}
  // envelope — the AppendFrameInterceptor on the server matches on
  // top-level `type === "append"` (per tech design §8.3). We send raw.
  if (!state.ws || state.ws.readyState !== WebSocket.OPEN) {
    console.warn("[Termin] WS not connected; cannot send append");
    return null;
  }
  const ref = `ref-${++state.refCounter}`;
  state.ws.send(JSON.stringify({
    type: "append",
    ref,
    resource,
    id,
    field,
    payload,
  }));
  return ref;
}

function renderConversationEntries(container, entries) {
  // Pair tool_calls with their tool_results so collapsibles can render
  // inline. Pair key: tool_call_id. Unmatched tool_results render
  // standalone with a "(orphan)" label.
  const resultByCallId = new Map();
  for (const e of entries) {
    if (e && e.kind === "tool_result" && e.tool_call_id) {
      resultByCallId.set(e.tool_call_id, e);
    }
  }
  for (const e of entries) {
    if (e && e.kind === "tool_result" && e.tool_call_id &&
        resultByCallId.get(e.tool_call_id) === e) {
      // Skip — rendered inline under its tool_call.
      continue;
    }
    appendConversationEntry(container, e, /*isLive*/false,
      e.kind === "tool_call" ? resultByCallId.get(e.tool_call_id) : null);
  }
}

function appendConversationEntry(container, entry, isLive, pairedResult) {
  if (!entry || typeof entry !== "object") return;
  const kind = entry.kind || "user";
  const type = entry.type || "";
  const wrapper = document.createElement("div");
  wrapper.setAttribute("data-termin-chat-entry", "");
  wrapper.setAttribute("data-termin-kind", kind);
  if (type) wrapper.setAttribute("data-termin-type", type);
  if (entry.source) wrapper.setAttribute("data-termin-source", entry.source);
  if (entry.tool_name) wrapper.setAttribute("data-termin-tool-name", entry.tool_name);
  if (entry.id) wrapper.setAttribute("data-termin-entry-id", entry.id);
  if (isLive) wrapper.classList.add("termin-fade-in");

  const renderer = CONVERSATION_ENTRY_RENDERERS[kind] || _renderEntryUnknown;
  renderer(wrapper, entry, pairedResult);
  container.appendChild(wrapper);
}

// Per-kind renderers. Each fills `wrapper` with the entry's HTML.
// Color is one signal among many — every distinction also carries a
// label/icon/position/aria so the surface is comprehensible to
// colorblind reviewers (per the universal CLAUDE.md UI rule).
function _renderEntryUser(wrapper, entry) {
  wrapper.className = "flex justify-end";
  if (entry.id) wrapper.classList.add("termin-fade-in-prep");
  const bubble = document.createElement("div");
  bubble.className = "bg-blue-500 text-white rounded-lg px-4 py-2 max-w-[70%]";
  bubble.setAttribute("aria-label", "User message");
  const label = document.createElement("div");
  label.className = "text-xs opacity-70 mb-1";
  label.textContent = "You";
  const body = document.createElement("div");
  body.textContent = entry.body || "";
  bubble.appendChild(label);
  bubble.appendChild(body);
  _renderAttachments(bubble, entry.attachments);
  wrapper.appendChild(bubble);
}

function _renderEntryAssistant(wrapper, entry) {
  wrapper.className = "flex justify-start";
  const isRefusal = entry.type === "refusal";
  const bubble = document.createElement("div");
  bubble.className = isRefusal
    ? "bg-amber-50 border border-amber-300 text-amber-900 rounded-lg px-4 py-2 max-w-[70%]"
    : "bg-gray-200 text-gray-800 rounded-lg px-4 py-2 max-w-[70%]";
  bubble.setAttribute("aria-label",
    isRefusal ? "Assistant refusal" : "Assistant message");
  const label = document.createElement("div");
  label.className = "text-xs opacity-70 mb-1 flex items-center gap-1";
  if (isRefusal) {
    // Color is one signal; the explicit "Refused" label + warning glyph
    // + amber-bordered bubble shape together convey the distinction
    // without depending on color perception.
    const icon = document.createElement("span");
    icon.setAttribute("aria-hidden", "true");
    icon.textContent = "⚠";
    label.appendChild(icon);
    const txt = document.createElement("span");
    txt.textContent = "Refused";
    label.appendChild(txt);
  } else {
    label.textContent = "Assistant";
  }
  const body = document.createElement("div");
  body.textContent = entry.body || "";
  bubble.appendChild(label);
  bubble.appendChild(body);
  _renderAttachments(bubble, entry.attachments);
  wrapper.appendChild(bubble);
}

function _renderEntryToolCall(wrapper, entry, pairedResult) {
  wrapper.className = "flex justify-start";
  const details = document.createElement("details");
  details.className = "bg-slate-100 border border-slate-300 rounded-lg px-3 py-2 max-w-[70%] text-sm text-slate-700";
  const summary = document.createElement("summary");
  summary.className = "cursor-pointer flex items-center gap-1";
  summary.setAttribute("aria-label",
    `Tool call: ${entry.tool_name || "unknown"}`);
  const icon = document.createElement("span");
  icon.setAttribute("aria-hidden", "true");
  icon.textContent = "⚙";
  summary.appendChild(icon);
  const txt = document.createElement("span");
  txt.textContent = `Called ${entry.tool_name || "tool"}`;
  summary.appendChild(txt);
  details.appendChild(summary);
  const argsLabel = document.createElement("div");
  argsLabel.className = "mt-2 text-xs uppercase tracking-wide opacity-60";
  argsLabel.textContent = "Arguments";
  details.appendChild(argsLabel);
  const args = document.createElement("pre");
  args.className = "mt-1 text-xs whitespace-pre-wrap break-words";
  args.textContent = entry.tool_args
    ? JSON.stringify(entry.tool_args, null, 2)
    : (entry.body || "");
  details.appendChild(args);
  if (pairedResult) {
    const resultLabel = document.createElement("div");
    resultLabel.className = "mt-2 text-xs uppercase tracking-wide opacity-60";
    resultLabel.textContent = "Result";
    details.appendChild(resultLabel);
    const result = document.createElement("div");
    result.className = "mt-1 text-xs whitespace-pre-wrap break-words";
    result.textContent = pairedResult.body || "";
    details.appendChild(result);
  }
  wrapper.appendChild(details);
}

function _renderEntryToolResult(wrapper, entry) {
  // Reached only for orphan tool_results (no matching tool_call in
  // this conversation). Render standalone with the orphan label.
  wrapper.className = "flex justify-start";
  const bubble = document.createElement("div");
  bubble.className = "bg-slate-100 border border-slate-300 rounded-lg px-3 py-2 max-w-[70%] text-sm text-slate-700";
  bubble.setAttribute("aria-label", "Orphan tool result");
  const label = document.createElement("div");
  label.className = "text-xs opacity-70 mb-1 flex items-center gap-1";
  const icon = document.createElement("span");
  icon.setAttribute("aria-hidden", "true");
  icon.textContent = "⚙";
  label.appendChild(icon);
  const txt = document.createElement("span");
  txt.textContent = `Tool result (${entry.tool_name || "unknown"}, orphan)`;
  label.appendChild(txt);
  const body = document.createElement("div");
  body.textContent = entry.body || "";
  bubble.appendChild(label);
  bubble.appendChild(body);
  wrapper.appendChild(bubble);
}

function _renderEntrySystemEvent(wrapper, entry) {
  wrapper.className = "flex justify-center";
  const notice = document.createElement("div");
  notice.className = "bg-indigo-50 border-l-4 border-indigo-400 text-indigo-900 px-3 py-2 max-w-[80%] text-sm";
  notice.setAttribute("role", "status");
  const labelRow = document.createElement("div");
  labelRow.className = "text-xs uppercase tracking-wide opacity-70 mb-1 flex items-center gap-1";
  const icon = document.createElement("span");
  icon.setAttribute("aria-hidden", "true");
  icon.textContent = "ℹ";
  labelRow.appendChild(icon);
  const labelTxt = document.createElement("span");
  labelTxt.textContent = entry.source ? entry.source : "System";
  labelRow.appendChild(labelTxt);
  const body = document.createElement("div");
  body.textContent = entry.body || "";
  notice.appendChild(labelRow);
  notice.appendChild(body);
  wrapper.appendChild(notice);
}

function _renderEntryUnknown(wrapper, entry) {
  wrapper.className = "flex justify-start";
  const bubble = document.createElement("div");
  bubble.className = "bg-gray-100 border border-gray-300 rounded-lg px-3 py-2 max-w-[70%] text-sm text-gray-700";
  bubble.textContent =
    `[unknown entry kind: ${entry.kind || "?"}] ${entry.body || ""}`;
  wrapper.appendChild(bubble);
}

const CONVERSATION_ENTRY_RENDERERS = {
  user: _renderEntryUser,
  assistant: _renderEntryAssistant,
  tool_call: _renderEntryToolCall,
  tool_result: _renderEntryToolResult,
  system_event: _renderEntrySystemEvent,
};

function _renderAttachments(parent, attachments) {
  if (!Array.isArray(attachments) || attachments.length === 0) return;
  const wrap = document.createElement("div");
  wrap.className = "mt-2 space-y-1";
  wrap.setAttribute("data-termin-chat-attachments", "");
  for (const a of attachments) {
    if (!a || !a.mime_type) continue;
    const mime = String(a.mime_type);
    const node = document.createElement("div");
    node.setAttribute("data-termin-attachment", "");
    node.setAttribute("data-termin-attachment-mime", mime);
    if (mime.indexOf("image/") === 0) {
      const img = document.createElement("img");
      img.alt = a.file_name || "attachment";
      img.src = `data:${mime};base64,${a.body || ""}`;
      img.className = "max-h-48 rounded";
      node.appendChild(img);
    } else if (mime === "application/pdf") {
      const link = document.createElement("a");
      link.className = "underline text-sm";
      link.href = `data:${mime};base64,${a.body || ""}`;
      link.download = a.file_name || "document.pdf";
      link.textContent = `📎 ${a.file_name || "document.pdf"}`;
      node.appendChild(link);
    } else {
      // text-ish: collapsed expandable
      const details = document.createElement("details");
      details.className = "text-xs";
      const summary = document.createElement("summary");
      summary.className = "cursor-pointer";
      summary.textContent = `📎 ${a.file_name || "attachment"} (${mime})`;
      details.appendChild(summary);
      const pre = document.createElement("pre");
      pre.className = "mt-1 whitespace-pre-wrap break-words";
      pre.textContent = a.body || "";
      details.appendChild(pre);
      node.appendChild(details);
    }
    wrap.appendChild(node);
  }
  parent.appendChild(wrap);
}

function hydrateDataTables() {
  const tables = document.querySelectorAll("[data-termin-component='data_table']");
  for (const table of tables) {
    const source = table.dataset.terminSource;
    if (!source) continue;

    const channelPrefix = `content.${source}`;
    subscribe(channelPrefix, (ch, data) => {
      const action = ch.split(".")[2]; // created, updated, deleted

      if (action === "updated" && data && data.id != null) {
        // Update existing row
        const row = table.querySelector(`tr[data-termin-row-id="${data.id}"]`);
        if (row) {
          updateRow(row, data);
          // Re-hydrate form interceptors on any new buttons inserted by updateActionButtons
          row.querySelectorAll("form[method='post']").forEach(form => {
            if (!form._terminHydrated) {
              form._terminHydrated = true;
              hydrateOneForm(form);
            }
          });
          flashRow(row);
        }
      } else if (action === "created" && data && data.id != null) {
        // Append new row — but skip if already exists
        const existing = table.querySelector(`tr[data-termin-row-id="${data.id}"]`);
        if (existing) {
          // Row already exists — just update it in case fields changed
          updateRow(existing, data);
          flashRow(existing);
        } else {
          const tbody = table.querySelector("tbody");
          if (tbody) {
            const row = createRow(table, data);
            tbody.appendChild(row);
            flashRow(row);
          }
        }
      } else if (action === "deleted") {
        const id = data.record_id || data.id;
        if (id != null) {
          const row = table.querySelector(`tr[data-termin-row-id="${id}"]`);
          if (row) row.remove();
        }
      }
    });
  }
}

function updateRow(row, data) {
  const cells = row.querySelectorAll("td[data-termin-field]");
  for (const cell of cells) {
    const field = cell.dataset.terminField;
    if (field in data) {
      cell.textContent = data[field] ?? "";
    }
  }
  // v0.9.1: re-evaluate transition action buttons after every push.
  // Pre-v0.9 used a generic ``status`` column and only re-evaluated
  // when the push carried that field; the v0.9 multi-state-machine
  // refactor renamed the column per-machine (``product_lifecycle``,
  // ``approval_status``, etc.), which made the legacy ``status``
  // check always-false and left action buttons stale until refresh.
  // We always re-evaluate now because (a) action visibility may
  // depend on a transition column we don't strictly know about
  // here, and (b) the cost of reading a few data attributes +
  // recomputing per-span is trivial compared to the click-to-stale
  // UX bug it fixes.
  updateActionButtons(row);
}

function updateActionButtons(row) {
  // Find the table's source content name
  const table = row.closest("[data-termin-component='data_table']");
  if (!table) return;
  const source = table.dataset.terminSource;
  if (!source) return;

  // v0.9 multi-SM bootstrap shape: transitions[source][machine_name][from|to] = scope.
  // Pre-v0.9 was the flat transitions[source][from|to] = scope shape; this
  // hydrator was migrated alongside the runtime change as part of v0.9.1.
  const transitionsByMachine = state.bootstrap && state.bootstrap.transitions
    ? state.bootstrap.transitions[source] || {}
    : {};
  const userScopes = state.identity ? new Set(state.identity.scopes) : new Set();
  const recordId = row.dataset.terminRowId;

  // Update each transition button wrapper. Each span declares its
  // own machine name (data-machine-name), so we read the current
  // state from THAT machine's column on the row (which updateRow
  // just refreshed for any push that carried it).
  row.querySelectorAll("[data-termin-transition]").forEach(span => {
    const targetState = span.dataset.targetState;
    const machineName = span.dataset.machineName || "";
    const behavior = span.dataset.behavior || "disable";

    // Read the current state from the corresponding column cell.
    // The state column for a state machine is the machine name
    // itself in v0.9 (one column per machine).
    const stateCell = machineName
      ? row.querySelector(`td[data-termin-field="${machineName}"]`)
      : null;
    const currentState = stateCell
      ? (stateCell.textContent || "").trim()
      : "";

    const machineTransitions = machineName
      ? (transitionsByMachine[machineName] || {})
      : {};
    const transKey = `${currentState}|${targetState}`;
    const requiredScope = machineTransitions[transKey];
    const isValid = requiredScope !== undefined;
    const hasScope = isValid && (requiredScope === "" || userScopes.has(requiredScope));
    const safeTarget = targetState.replace(/ /g, "_");
    const safeMachine = machineName.replace(/ /g, "_");

    if (isValid && hasScope) {
      // Show enabled button. Form posts to the v0.9 4-segment
      // transition path: /_transition/<source>/<machine>/<id>/<target>.
      span.innerHTML =
        `<form method="post" action="/_transition/${source}/${safeMachine}/${recordId}/${safeTarget}" style="display:inline">` +
        `<button type="submit" class="text-indigo-600 hover:text-indigo-800 text-xs">${span.dataset.label || targetState}</button></form>`;
    } else if (behavior === "hide") {
      span.innerHTML = "";
    } else {
      // Disabled button
      span.innerHTML =
        `<button disabled class="text-gray-400 text-xs cursor-not-allowed">${span.dataset.label || targetState}</button>`;
    }
  });
}

function createRow(table, data) {
  // Determine column fields from existing rows or thead
  const fields = [];
  const tbody = table.querySelector("tbody");
  const existingRow = tbody && tbody.querySelector("tr[data-termin-row-id] td[data-termin-field]");
  if (existingRow) {
    // Copy field order from existing row that has cells
    existingRow.closest("tr").querySelectorAll("td[data-termin-field]").forEach(td => {
      fields.push(td.dataset.terminField);
    });
  }
  if (fields.length === 0) {
    // Fall back to thead column headers
    table.querySelectorAll("thead th").forEach(th => {
      const text = th.textContent.trim().toLowerCase().replace(/\s+/g, "_");
      if (text && text !== "actions") {
        fields.push(text);
      }
    });
  }

  const tr = document.createElement("tr");
  tr.className = "border-t";
  tr.setAttribute("data-termin-row-id", data.id);
  for (const field of fields) {
    const td = document.createElement("td");
    td.className = "px-4 py-2 text-sm";
    td.setAttribute("data-termin-field", field);
    td.textContent = data[field] ?? "";
    tr.appendChild(td);
  }
  return tr;
}

function flashRow(row) {
  row.classList.add("termin-updated");
  setTimeout(() => row.classList.remove("termin-updated"), 600);
}

function hydrateAggregations() {
  const aggs = document.querySelectorAll("[data-termin-component='aggregation'], [data-termin-component='stat_breakdown']");
  for (const agg of aggs) {
    const source = agg.dataset.terminSource;
    if (!source) continue;

    subscribe(`content.${source}`, () => {
      // Re-fetch aggregation value on any change
      // For now, show "updating..." briefly — full compute requires server round-trip
      const valueEl = agg.querySelector("[data-termin-agg]");
      if (valueEl) {
        valueEl.classList.add("termin-updating");
        // The SSR page will need a full reload for server-computed aggregations
        // Future: compute client-side from cache for simple count/sum
        setTimeout(() => valueEl.classList.remove("termin-updating"), 1000);
      }
    });
  }
}

function hydrateOneForm(form) {
  // Skip the role-switcher form
  if (form.action && form.action.includes("/set-role")) return;
  // Skip forms that opt out of the generic AJAX-submit interceptor.
  // v0.9.2: the conversation-field chat form owns its own submit
  // path (sendAppendFrame via the WS L4 append frame); the generic
  // interceptor would just try to fetch the form's action URL and
  // either succeed at the page URL (wrong endpoint) or fail (when
  // action is javascript:void(0)). Either way, wrong shape.
  if (form.hasAttribute("data-termin-no-default-submit")) return;
  if (form._terminHydrated) return;
  form._terminHydrated = true;

  form.addEventListener("submit", async (e) => {
      e.preventDefault();
      const formData = new FormData(form);
      const url = form.action || window.location.href;

      try {
        const resp = await fetch(url, {
          method: "POST",
          body: formData,
          headers: {
            "Accept": "application/json",
            "X-Requested-With": "XMLHttpRequest",
          },
        });

        if (resp.ok) {
          // Check for flash notification in JSON response (transition feedback)
          try {
            const data = await resp.json();
            if (data._flash) {
              showFlashNotification(data._flash, data._flash_style || "toast",
                                   data._flash_level || "success", data._flash_dismiss);
            }
          } catch {
            // Not JSON — normal form response, ignore
          }

          // Clear the form inputs
          form.querySelectorAll("input[type='text'], textarea").forEach(input => {
            input.value = "";
          });
          form.querySelectorAll("select").forEach(select => {
            select.selectedIndex = 0;
          });
          // Don't add the row here — the WebSocket subscription will push
          // the new record to the table. Adding from both sources causes
          // duplicate rows. Let WebSocket be the single source of truth.
          //
          // For transition forms, the WS push carries the updated status.
          // But if WS is not connected, fall back to a page reload.
          if (!state.ws || state.ws.readyState !== WebSocket.OPEN) {
            window.location.reload();
          }
        } else {
          // Show error — check for flash notification in error response
          try {
            const err = await resp.json();
            if (err.detail && err.detail._flash) {
              showFlashNotification(err.detail._flash, err.detail._flash_style || "banner",
                                   err.detail._flash_level || "error", err.detail._flash_dismiss);
            } else {
              alert(err.detail || "Error saving record");
            }
          } catch {
            alert("Error saving record (HTTP " + resp.status + ")");
          }
        }
      } catch (fetchErr) {
        // Network error — fall back to normal form submit
        console.warn("[Termin] AJAX submit failed, falling back to form POST:", fetchErr);
        form.submit();
      }
    });
}

function hydrateForms() {
  document.querySelectorAll("form[method='post'], form:not([method])").forEach(form => {
    hydrateOneForm(form);
  });
}

// ── Connection Indicator ──

function createIndicator() {
  const el = document.createElement("div");
  el.id = "termin-status";
  el.style.cssText = "position:fixed;bottom:8px;right:8px;padding:4px 10px;border-radius:4px;" +
    "font-size:11px;font-family:system-ui;z-index:9999;transition:opacity 0.3s;pointer-events:none;";
  document.body.appendChild(el);
  state.indicator = el;
  updateIndicator(false);
}

function updateIndicator(connected) {
  if (!state.indicator) return;
  if (connected) {
    state.indicator.textContent = "Connected";
    state.indicator.style.background = "#10b981";
    state.indicator.style.color = "#fff";
    state.indicator.style.opacity = "0.6";
    // Fade out after 3 seconds
    setTimeout(() => {
      if (state.connected && state.indicator) state.indicator.style.opacity = "0.2";
    }, 3000);
  } else {
    state.indicator.textContent = "Reconnecting\u2026";
    state.indicator.style.background = "#ef4444";
    state.indicator.style.color = "#fff";
    state.indicator.style.opacity = "1";
  }
}

// ── CSS for flash animation ──

const style = document.createElement("style");
style.textContent = `
  .termin-updated {
    animation: termin-flash 0.6s ease-out;
  }
  @keyframes termin-flash {
    0% { background-color: #fef3c7; }
    100% { background-color: transparent; }
  }
  .termin-updating {
    opacity: 0.5;
    transition: opacity 0.2s;
  }
`;
document.head.appendChild(style);

// ── v0.9 Phase 5b.4 platform: Presentation provider extension API ──
//
// Per BRD #2 §7.4 + JL-resolved option (d) from the briefings:
// CSR bundles register their per-contract render functions through
// `Termin.registerRenderer(contract, fn)`. Loaded at boot via
// `loadCsrBundles()` from the discovery endpoint.

const _renderers = Object.create(null);

function registerRenderer(contract, fn) {
  if (typeof contract !== "string" || typeof fn !== "function") {
    console.warn("[Termin] registerRenderer: contract must be string, fn must be function");
    return;
  }
  _renderers[contract] = fn;
}

function getRenderer(contract) {
  return _renderers[contract] || null;
}

async function loadCsrBundles() {
  try {
    const resp = await fetch("/_termin/presentation/bundles", { credentials: "same-origin" });
    if (!resp.ok) return;
    const body = await resp.json();
    const bundles = (body && body.bundles) || [];
    // Dedupe on URL — one bundle file may serve multiple contracts,
    // and the B'-mode shell template already injects <script defer>
    // tags for each bundle. Skip any URL that's already in the
    // document so we don't double-execute the bundle.
    const seen = new Set();
    for (const entry of bundles) {
      if (!entry || !entry.url || seen.has(entry.url)) continue;
      seen.add(entry.url);
      // Honor existing static script tags from the shell template.
      // querySelector with an attribute-equals match — if the bundle
      // is already there, skip the dynamic append.
      if (document.querySelector(
        `script[src="${entry.url.replace(/"/g, '\\"')}"]`)) {
        continue;
      }
      const script = document.createElement("script");
      script.src = entry.url;
      script.async = true;
      script.dataset.terminCsrBundle = entry.contract;
      script.dataset.terminCsrProvider = entry.provider || "";
      document.head.appendChild(script);
    }
  } catch (err) {
    // Bundle load failures are non-fatal — SSR remains in effect.
    console.warn("[Termin] CSR bundle load failed:", err.message);
  }
}

// ── v0.9 Phase 5b.4 B' plumbing: SPA navigation + action dispatch ──
//
// Per the Spectrum-provider design Q2 (B' = server-authoritative +
// JS-as-renderer), the provider's bundle calls these to drive
// page transitions and user-initiated mutations. The runtime owns
// the trust plane; these helpers are the typed seam between the
// provider and the runtime.

const SHELL_CONTRACT = "__app_shell__";
const _subscriptionHandlers = Object.create(null);  // channel -> Set<handler>

async function navigate(path) {
  if (typeof path !== "string" || !path) {
    console.warn("[Termin] navigate: path must be a non-empty string");
    return;
  }
  try {
    const url = new URL("/_termin/page-data", window.location.origin);
    url.searchParams.set("path", path);
    const resp = await fetch(url.toString(), { credentials: "same-origin" });
    if (!resp.ok) {
      console.warn(`[Termin] navigate: ${resp.status} for path ${path}`);
      return;
    }
    const payload = await resp.json();

    // Browser history — back/forward must work without a full
    // page reload. The state stash lets us re-render on popstate
    // without a second fetch (cache the most recent payload).
    history.pushState({ termin: true, path }, "", path);

    // Hand off to the registered shell renderer. If no provider
    // has registered "__app_shell__" yet, log and fall back to a
    // full page load so the user isn't stranded.
    const renderer = getRenderer(SHELL_CONTRACT);
    if (typeof renderer === "function") {
      renderer(
        payload.component_tree_ir,
        payload.bound_data,
        payload.principal_context,
        payload.subscriptions_to_open,
        payload.app_chrome,
      );
    } else {
      window.location.href = path;
    }
  } catch (err) {
    console.warn("[Termin] navigate failed:", err.message);
  }
}

// Termin.action(payload) — client-side dispatcher to the existing
// REST surface every conforming runtime implements per BRD #2 §11.
// No /_termin/action server endpoint exists; the JS provider gets
// a stable typed seam, the runtime gets zero new plumbing, and
// alternate runtimes (e.g. an AWS-native Termin runtime) inherit
// this for free.
//
// Payload shape:
//   { kind: "create",     content, payload }
//   { kind: "update",     content, id, payload }
//   { kind: "delete",     content, id }
//   { kind: "transition", content, id, machine_name, target_state }
//   { kind: "compute",    compute_name, input }
//
// Returns: { ok, status, kind, data } on success; { ok: false,
// status, kind, error } on HTTP / validation / network failure.
async function action(payload) {
  if (!payload || typeof payload !== "object" || !payload.kind) {
    console.warn("[Termin] action: payload must be { kind: <string>, ... }");
    return { ok: false, error: "payload must be an object with a `kind` field" };
  }
  const kind = payload.kind;
  let url, method, body;

  if (kind === "create") {
    if (!payload.content) {
      return { ok: false, kind, error: "`content` required for create" };
    }
    url = `/api/v1/${encodeURIComponent(payload.content)}`;
    method = "POST";
    body = payload.payload || {};
  } else if (kind === "update") {
    if (!payload.content || payload.id == null) {
      return { ok: false, kind, error: "`content` and `id` required for update" };
    }
    url = `/api/v1/${encodeURIComponent(payload.content)}/${encodeURIComponent(payload.id)}`;
    method = "PUT";
    body = payload.payload || {};
  } else if (kind === "delete") {
    if (!payload.content || payload.id == null) {
      return { ok: false, kind, error: "`content` and `id` required for delete" };
    }
    url = `/api/v1/${encodeURIComponent(payload.content)}/${encodeURIComponent(payload.id)}`;
    method = "DELETE";
    body = null;
  } else if (kind === "transition") {
    if (!payload.content || payload.id == null
        || !payload.machine_name || !payload.target_state) {
      return {
        ok: false, kind,
        error: "`content`, `id`, `machine_name`, `target_state` required for transition",
      };
    }
    // Generic transition endpoint (transitions.py) — 4 path segments
    // after `/_transition`. Underscores in target_state survive URL
    // encoding and are converted back to spaces server-side.
    const targetSafe = String(payload.target_state).replace(/ /g, "_");
    url = "/_transition/"
      + encodeURIComponent(payload.content) + "/"
      + encodeURIComponent(payload.machine_name) + "/"
      + encodeURIComponent(payload.id) + "/"
      + encodeURIComponent(targetSafe);
    method = "POST";
    body = null;
  } else if (kind === "compute") {
    if (!payload.compute_name) {
      return { ok: false, kind, error: "`compute_name` required for compute" };
    }
    url = `/api/v1/compute/${encodeURIComponent(payload.compute_name)}`;
    method = "POST";
    body = payload.input || {};
  } else {
    console.warn(`[Termin] action: unknown kind ${kind}`);
    return { ok: false, kind, error: `unknown kind: ${kind}` };
  }

  const init = {
    method,
    credentials: "same-origin",
    headers: { "Accept": "application/json" },
  };
  if (body !== null) {
    init.headers["Content-Type"] = "application/json";
    init.body = JSON.stringify(body);
  }

  try {
    const resp = await fetch(url, init);
    let data = null;
    const ct = resp.headers.get("content-type") || "";
    if (ct.includes("application/json")) {
      try { data = await resp.json(); } catch { data = null; }
    }
    if (!resp.ok) {
      return {
        ok: false, kind, status: resp.status,
        error: (data && (data.detail || data.error)) || resp.statusText,
      };
    }
    return { ok: true, kind, status: resp.status, data };
  } catch (err) {
    console.warn("[Termin] action failed:", err.message);
    return { ok: false, kind, error: err.message };
  }
}

function _addSubscription(channel, handler) {
  if (typeof channel !== "string" || typeof handler !== "function") {
    console.warn("[Termin] subscribe: channel must be string, handler must be function");
    return;
  }
  let set = _subscriptionHandlers[channel];
  if (!set) {
    set = new Set();
    _subscriptionHandlers[channel] = set;
    // Open the WebSocket-side subscription directly via sendFrame.
    // We deliberately do NOT call the legacy `subscribe(channel)`
    // here: that helper is the SSR-hydrator surface and stores
    // callbacks in `state.subscriptions`. Calling it without a
    // callback poisons that Set with `undefined`, which then
    // throws "cb is not a function" when notifySubscribers fires
    // for any matching channel. Provider-side subscriptions live
    // in `_subscriptionHandlers` and dispatch through
    // `_dispatchToProviderSubscriptions` instead.
    sendFrame("subscribe", channel, {});
  }
  set.add(handler);
}

function _removeSubscription(channel, handler) {
  const set = _subscriptionHandlers[channel];
  if (!set) return;
  set.delete(handler);
  if (set.size === 0) {
    delete _subscriptionHandlers[channel];
    // The existing subscription state allows multiple subscribers;
    // we leave the channel-level subscription open if any caller
    // (including this file's internal subscribe) still wants it.
    // Per-channel teardown is a runtime decision; provider-level
    // remove just clears the handler list.
  }
}

// Hook for the existing WebSocket message dispatcher to route
// incoming events into provider-registered subscription handlers.
// Existing `state.handlers` keep working; this adds a parallel
// dispatch surface for the Termin global.
function _dispatchToProviderSubscriptions(channel, payload) {
  // Prefix-match: a handler registered for "content.tickets"
  // also receives "content.tickets.created" / ".updated" / ".deleted".
  for (const registered in _subscriptionHandlers) {
    if (channel === registered || channel.startsWith(registered + ".")) {
      for (const handler of _subscriptionHandlers[registered]) {
        try { handler(payload, channel); }
        catch (err) {
          console.warn(`[Termin] subscriber for ${registered} threw:`, err.message);
        }
      }
    }
  }
}

// Browser back/forward — re-fetch and render. Pure-history
// SPA: state.termin marks frames we own; foreign frames are
// passed through to the browser.
window.addEventListener("popstate", function (ev) {
  if (ev.state && ev.state.termin && ev.state.path) {
    navigate(ev.state.path).catch(() => {});
  }
});

// ── Public API ──

window.Termin = {
  registerRenderer,
  getRenderer,
  navigate,
  action,
  subscribe: _addSubscription,
  unsubscribe: _removeSubscription,
  _dispatchToProviderSubscriptions,  // internal hook; do not rely on
};
window.__termin = { state, subscribe, getCachedRecords, TERMIN_VERSION };

// ── Initialize on DOM ready ──

if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", init);
} else {
  init();
}
