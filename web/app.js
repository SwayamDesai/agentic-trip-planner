/* Atlas front end.
 *
 * Two flows:
 *   chat     POST /api/chat per message until the trip is plannable
 *   plan     SSE from /api/plan, streaming agent progress then the plan
 *
 * Everything the server sends is treated as text, never HTML: agent output is
 * model-generated and interpolating it as markup would be an injection route.
 */

const AGENT_ROLE = {
  flight: "required", hotels: "required", itinerary: "required",
  weather: "optional", budget: "optional",
};

const AGENT_BLURB = {
  flight: "Searching live fares",
  hotels: "Checking room rates",
  weather: "Reading the forecast",
  itinerary: "Finding things to do",
  budget: "Reconciling costs",
};

const el = (id) => document.getElementById(id);
const log = el("chat-log");
const input = el("chat-input");
const form = el("chat-form");
const sendBtn = el("chat-send");
const stateLabel = el("assistant-state");
const suggestions = el("suggestions");

let history = [];
let planning = false;
let clockTimer = null;

/* ── helpers ─────────────────────────────────────────────────────── */

function node(tag, cls, text) {
  const n = document.createElement(tag);
  if (cls) n.className = cls;
  if (text !== undefined && text !== null) n.textContent = String(text);
  return n;
}

/* Cents matter here — a run costs fractions of one, so `money()` rounding to
   the nearest dollar would render every plan as "$0". */
function dollars(value) {
  const n = Number(value);
  if (!Number.isFinite(n)) return "—";
  if (n === 0) return "$0.00";
  return n < 0.01 ? `$${n.toFixed(4)}` : `$${n.toFixed(2)}`;
}

function money(value) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return "—";
  return "$" + Math.round(Number(value)).toLocaleString("en-US");
}

function dayName(iso) {
  const d = new Date(iso + "T12:00:00");
  if (Number.isNaN(d.getTime())) return iso;
  return d.toLocaleDateString("en-US", { weekday: "short", month: "short", day: "numeric" });
}

function scrollLog() { log.scrollTop = log.scrollHeight; }

/* ── chat ────────────────────────────────────────────────────────── */

function say(role, text) {
  const cls = role === "user" ? "msg msg--user"
            : role === "error" ? "msg msg--error" : "msg msg--bot";
  log.appendChild(node("div", cls, text));
  scrollLog();
}

function showTyping() {
  const wrap = node("div", "msg msg--bot");
  wrap.id = "typing";
  const dots = node("span", "typing");
  for (let i = 0; i < 3; i++) dots.appendChild(node("span"));
  wrap.appendChild(dots);
  log.appendChild(wrap);
  scrollLog();
}

function hideTyping() { el("typing")?.remove(); }

function setBusy(busy, label) {
  sendBtn.disabled = busy;
  input.disabled = busy;
  stateLabel.textContent = label;
}

async function send(message) {
  if (!message.trim() || planning) return;

  suggestions.hidden = true;
  say("user", message);
  history.push({ role: "user", content: message });
  input.value = "";
  input.style.height = "auto";

  setBusy(true, "Thinking…");
  showTyping();

  let data;
  try {
    const res = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message, history: history.slice(0, -1) }),
    });
    if (!res.ok) throw new Error("HTTP " + res.status);
    data = await res.json();
  } catch (err) {
    hideTyping();
    setBusy(false, "Ready");
    say("error", "Could not reach the planner: " + err.message);
    return;
  }

  hideTyping();
  say("bot", data.reply);
  history.push({ role: "assistant", content: data.reply });

  if (data.ready) {
    setBusy(true, "Planning…");
    startPlanning(data.trip);
  } else {
    setBusy(false, "Waiting for details");
  }
}

form.addEventListener("submit", (e) => { e.preventDefault(); send(input.value); });

input.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); send(input.value); }
});

input.addEventListener("input", () => {
  input.style.height = "auto";
  input.style.height = Math.min(input.scrollHeight, 132) + "px";
});

suggestions.addEventListener("click", (e) => {
  const chip = e.target.closest(".chip");
  if (chip) send(chip.textContent.trim());
});

/* ── planning ────────────────────────────────────────────────────── */

function renderSummary(trip) {
  el("sum-origin").textContent = trip.origin;
  el("sum-destination").textContent = trip.destination;

  const facts = el("sum-facts");
  facts.replaceChildren();

  const nights = (() => {
    const a = new Date(trip.start_date), b = new Date(trip.end_date);
    const n = Math.round((b - a) / 86400000);
    return Number.isFinite(n) ? n : null;
  })();

  const entries = [
    ["Dates", `${dayName(trip.start_date)} – ${dayName(trip.end_date)}`],
    ["Nights", nights === null ? "—" : String(nights)],
    ["Travellers", String(trip.travelers)],
    ["Budget", trip.budget_usd ? money(trip.budget_usd) : "Not set"],
  ];
  if (trip.preferences?.length) entries.push(["Interests", trip.preferences.join(", ")]);

  for (const [k, v] of entries) {
    const box = node("div");
    box.append(node("dt", null, k), node("dd", null, v));
    facts.appendChild(box);
  }
  el("summary").hidden = false;
}

function renderAgentRows(names) {
  const list = el("agents");
  list.replaceChildren();
  for (const name of names) {
    const row = node("li", "agent");
    row.dataset.state = "waiting";
    row.dataset.agent = name;
    row.append(
      node("span", "agent__dot", ""),
      node("span", "agent__name", name),
      node("span", "agent__note", AGENT_BLURB[name] || "Working"),
    );
    const role = node("span", "agent__role", AGENT_ROLE[name] || "required");
    role.dataset.role = AGENT_ROLE[name] || "required";
    row.appendChild(role);
    list.appendChild(row);
  }
  el("progress").hidden = false;
}

function updateAgent(event) {
  const row = document.querySelector(`.agent[data-agent="${event.agent}"]`);
  if (!row) return;
  const note = row.querySelector(".agent__note");
  const dot = row.querySelector(".agent__dot");
  const ev = event.event || "";

  if (ev.startsWith("start")) {
    row.dataset.state = "running";
    note.textContent = AGENT_BLURB[event.agent] || "Working";
  } else if (ev.startsWith("tool ")) {
    row.dataset.state = "running";
    note.textContent = "Calling " + ev.slice(5).replace(/_/g, " ");
  } else if (ev.startsWith("skip")) {
    row.dataset.state = "done";
    dot.textContent = "✓";
    note.textContent = "Reused from an earlier run";
  } else if (ev.startsWith("done")) {
    row.dataset.state = "done";
    dot.textContent = "✓";
    const m = ev.match(/(\d+) tool calls/);
    note.textContent = m ? `Done — ${m[1]} tool call${m[1] === "1" ? "" : "s"}` : "Done";
  } else if (ev.startsWith("FAILED")) {
    row.dataset.state = "failed";
    dot.textContent = "!";
    note.textContent = "Failed — " + (AGENT_ROLE[event.agent] === "optional"
      ? "plan continues without it" : "this plan needs it");
  } else if (ev.startsWith("LIMIT")) {
    note.textContent = "Limit reached — wrapping up";
  }
}

function startPlanning(trip) {
  planning = true;
  el("empty-state").hidden = true;
  el("notice").hidden = true;
  el("plan").replaceChildren();

  const q = new URLSearchParams({
    origin: trip.origin, destination: trip.destination, start: trip.start_date,
  });
  if (trip.end_date) q.set("end", trip.end_date);
  if (trip.nights) q.set("nights", trip.nights);
  if (trip.travelers) q.set("travelers", trip.travelers);
  if (trip.budget_usd) q.set("budget", trip.budget_usd);
  if (trip.preferences?.length) q.set("prefer", trip.preferences.join(","));

  const started = Date.now();
  clockTimer = setInterval(() => {
    el("progress-clock").textContent = ((Date.now() - started) / 1000).toFixed(1) + "s";
  }, 100);

  const source = new EventSource("/api/plan?" + q.toString());

  source.addEventListener("resolved", (e) => {
    const data = JSON.parse(e.data);
    renderSummary(data.request);
    renderAgentRows(data.agents);
    if (data.scope_reason) {
      say("bot", `You didn't give a trip length, so I've used ${data.request.start_date} to ${data.request.end_date} — ${data.scope_reason}`);
    }
  });

  source.addEventListener("agent", (e) => updateAgent(JSON.parse(e.data)));

  source.addEventListener("plan", (e) => {
    finish(source);
    renderPlan(JSON.parse(e.data));
  });

  source.addEventListener("failed", (e) => {
    finish(source);
    say("error", "Planning failed: " + JSON.parse(e.data).error);
  });

  source.onerror = () => {
    if (planning) { finish(source); say("error", "Connection to the planner was lost."); }
  };
}

function finish(source) {
  source.close();
  planning = false;
  clearInterval(clockTimer);
  setBusy(false, "Ready");
}

/* ── rendering the plan ──────────────────────────────────────────── */

function card(title, count, source) {
  const c = node("section", "card");
  const head = node("div", "card__head");
  head.appendChild(node("h3", null, title));
  if (count) head.appendChild(node("span", "card__count", count));
  if (source) head.appendChild(node("span", "card__source", source));
  c.appendChild(head);
  const body = node("div", "card__body");
  c.appendChild(body);
  return { card: c, body };
}

function renderFlights(data, target) {
  if (!data?.options?.length) return;
  const { card: c, body } = card("Flights", `${data.options.length} options`, "Live fares");
  const cheapest = Math.min(...data.options.map((o) => o.price_usd));

  data.options.forEach((o) => {
    const row = node("div", "offer");
    const main = node("div", "offer__main");

    const title = node("div", "offer__title");
    title.appendChild(node("span", null, o.airline));
    if (o.stops === 0) title.appendChild(node("span", "tag", "Nonstop"));
    if (o.price_usd === cheapest) title.appendChild(node("span", "tag tag--best", "Cheapest"));
    main.appendChild(title);

    main.appendChild(node("div", "offer__meta",
      `${o.departure} → ${o.arrival} · ${o.duration}` +
      (o.stops ? ` · ${o.stops} stop${o.stops > 1 ? "s" : ""}` : "")));
    if (o.notes) main.appendChild(node("div", "offer__note", o.notes));

    const price = node("div", "offer__price");
    price.append(node("div", "offer__amount", money(o.price_usd)),
                 node("div", "offer__unit", "per person"));

    row.append(main, price);
    body.appendChild(row);
  });
  target.appendChild(c);
}

function renderHotels(data, target) {
  if (!data?.options?.length) return;
  const { card: c, body } = card("Where to stay", `${data.options.length} options`, "Live rates");
  const cheapest = Math.min(...data.options.map((o) => o.price_per_night_usd));

  data.options.forEach((h) => {
    const row = node("div", "offer");
    const main = node("div", "offer__main");

    const title = node("div", "offer__title");
    title.appendChild(node("span", null, h.name));
    if (h.price_per_night_usd === cheapest) title.appendChild(node("span", "tag tag--best", "Lowest rate"));
    main.appendChild(title);

    const meta = node("div", "offer__meta");
    if (h.rating) {
      const filled = Math.round(h.rating);
      meta.appendChild(node("span", "stars", "★".repeat(filled) + "☆".repeat(Math.max(0, 5 - filled))));
      meta.appendChild(node("span", null, ` ${h.rating.toFixed(1)} · `));
    }
    meta.appendChild(node("span", null, h.area));
    main.appendChild(meta);
    if (h.notes) main.appendChild(node("div", "offer__note", h.notes));

    const price = node("div", "offer__price");
    price.append(node("div", "offer__amount", money(h.price_per_night_usd)),
                 node("div", "offer__unit", "per night"));

    row.append(main, price);
    body.appendChild(row);
  });
  target.appendChild(c);
}

function renderWeather(data, target) {
  if (!data?.daily?.length) return;
  const { card: c, body } = card("Weather", null, "Open-Meteo");
  const strip = node("div", "weather");

  data.daily.forEach((d) => {
    const day = node("div", "weather__day");
    day.append(
      node("div", "weather__date", dayName(d.date)),
      node("div", "weather__temp", `${Math.round(d.high_c)}° / ${Math.round(d.low_c)}°`),
      node("div", "weather__cond", d.condition),
    );
    if (d.precipitation_chance) {
      day.appendChild(node("div", "weather__rain", `${d.precipitation_chance}% rain`));
    }
    strip.appendChild(day);
  });
  body.appendChild(strip);
  if (data.packing_advice) body.appendChild(node("div", "advice", data.packing_advice));
  target.appendChild(c);
}

function renderItinerary(data, target) {
  if (!data?.days?.length) return;
  const { card: c, body } = card("Itinerary", `${data.days.length} days`, "Verified places");

  data.days.forEach((d, i) => {
    const day = node("div", "day");
    const head = node("div", "day__head");
    head.append(node("span", "day__label", `Day ${i + 1}`),
                node("span", "day__date", dayName(d.date)));
    day.appendChild(head);

    (d.activities || []).forEach((a) => {
      const slot = node("div", "slot");
      slot.appendChild(node("div", "slot__when", a.time_of_day));

      const bodyEl = node("div", "slot__body");
      const name = node("div", "slot__name", a.name);
      if (!a.cost_usd) name.appendChild(node("span", "pill pill--free", "Free"));
      else name.appendChild(node("span", "pill", money(a.cost_usd) + " pp"));
      if (a.indoor) name.appendChild(node("span", "pill pill--indoor", "Indoor"));
      bodyEl.appendChild(name);

      const bits = [`${a.duration_hours}h`];
      if (a.notes) bits.push(a.notes);
      bodyEl.appendChild(node("div", "slot__meta", bits.join(" · ")));

      slot.appendChild(bodyEl);
      day.appendChild(slot);
    });
    body.appendChild(day);
  });
  target.appendChild(c);
}

function renderCost(data, target) {
  if (!data?.breakdown) return;
  const b = data.breakdown;
  const tier = b.tier === "cheapest" ? "cheapest" : "mid-range";
  const { card: c, body } = card("Cost", null, `${tier} options`);
  const table = node("div", "cost");

  const rows = [
    ["Flights", `${tier} × ${b.travelers} traveller${b.travelers > 1 ? "s" : ""}`, b.flights_usd],
    ["Lodging", `${tier} × ${b.nights} night${b.nights === 1 ? "" : "s"}`, b.lodging_usd],
    ["Activities", "entry costs", b.activities_usd],
  ];
  for (const [label, hint, value] of rows) {
    const row = node("div", "cost__row");
    const left = node("div", "cost__label", label);
    left.appendChild(node("span", "cost__hint", hint));
    row.append(left, node("div", "cost__value", money(value)));
    table.appendChild(row);
  }

  const total = node("div", "cost__row cost__row--total");
  total.append(node("div", "cost__label", "Estimated total"),
               node("div", "cost__value", money(b.subtotal_usd)));
  table.appendChild(total);
  body.appendChild(table);

  if (b.budget_usd === null || b.budget_usd === undefined) {
    body.appendChild(node("div", "verdict verdict--none",
      "No budget was set, so this is costed at mid-range options rather than the cheapest available."));
  } else if (b.feasible === false) {
    const v = node("div", "verdict verdict--over");
    v.appendChild(node("strong", null, `Not achievable within ${money(b.budget_usd)}. `));
    v.appendChild(node("span", null,
      `The cheapest flights and lodging alone come to ${money(b.travel_only_usd)}, before any activities.`));
    body.appendChild(v);
  } else if (b.within_budget) {
    const v = node("div", "verdict verdict--ok");
    v.appendChild(node("strong", null, "Within budget. "));
    v.appendChild(node("span", null,
      `${money(Math.abs(b.over_under_usd))} to spare against ${money(b.budget_usd)}.`));
    body.appendChild(v);
  } else {
    const v = node("div", "verdict verdict--over");
    v.appendChild(node("strong", null, `Over budget by ${money(Math.abs(b.over_under_usd))}. `));
    v.appendChild(node("span", null, `Budget was ${money(b.budget_usd)}.`));
    body.appendChild(v);
  }

  if (b.missing?.length) {
    body.appendChild(node("div", "advice",
      `Incomplete: no data for ${b.missing.join(", ")}, so this total is understated.`));
  }

  if (data.advice) {
    if (data.advice.assessment) body.appendChild(node("div", "advice", data.advice.assessment));
    if (data.advice.suggestions?.length) {
      const wrap = node("div", "suggest");
      wrap.appendChild(node("h4", null, "How to save"));
      const list = node("ul");
      data.advice.suggestions.forEach((s) => list.appendChild(node("li", null, s)));
      wrap.appendChild(list);
      body.appendChild(wrap);
    }
    if (data.advice.unbudgeted?.length) {
      body.appendChild(node("div", "offer__note",
        "Not included: " + data.advice.unbudgeted.join(", ") + "."));
    }
  }
  target.appendChild(c);
}

/* What the plan cost to PRODUCE, as opposed to what the trip costs.
   Kept in its own card, and labelled "at list price", because these are two
   dollar figures on one screen and confusing them would be easy: the trip
   total is real money a traveller would spend, this one is what the inference
   would have cost on a paid account. Every model here is on a free tier, so the
   actual invoice is zero — which is precisely why the list-price figure is the
   informative one. */
function renderRunCost(metrics, target) {
  const t = metrics?.totals;
  if (!t) return;

  const { card: c, body } = card("What this plan cost to produce", null,
    "model spend, at list price");
  const table = node("div", "cost");

  const rows = [
    ["Inference", `${t.llm_calls} model call${t.llm_calls === 1 ? "" : "s"}`,
     dollars(t.cost_usd)],
    ["Tokens", "prompt + completion", (t.total_tokens || 0).toLocaleString("en-US")],
    ["Wall clock", "fan-out, so less than the sum of the agents",
     `${t.wall_seconds}s`],
  ];
  for (const [label, hint, value] of rows) {
    const row = node("div", "cost__row");
    const left = node("div", "cost__label", label);
    left.appendChild(node("span", "cost__hint", hint));
    row.append(left, node("div", "cost__value", value));
    table.appendChild(row);
  }
  body.appendChild(table);

  /* An unpriced call means the total is understated. Saying so is the whole
     point: a cost figure that quietly omits calls is worse than none. */
  if (t.unpriced_calls) {
    body.appendChild(node("div", "advice",
      `${t.unpriced_calls} call${t.unpriced_calls === 1 ? "" : "s"} had no known ` +
      `price, so this figure is a floor.`));
  }

  const filtered = metrics.filtered && Object.keys(metrics.filtered);
  if (filtered?.length) {
    body.appendChild(node("div", "advice",
      `Instruction-like text was found in data from ${filtered.join(", ")} and ` +
      `neutralised before the model saw it.`));
  }

  target.appendChild(c);
}

function renderNotice(payload) {
  const box = el("notice");
  if (payload.status === "ok") { box.hidden = true; return; }

  box.className = "notice notice--" + (payload.status === "failed" ? "failed" : "degraded");
  box.replaceChildren();
  box.appendChild(node("strong", "notice__title",
    payload.status === "failed"
      ? "Incomplete plan — required data is missing"
      : "Partial plan — some details are unavailable"));

  const list = node("ul");
  (payload.errors || []).forEach((e) => list.appendChild(node("li", null, e)));
  box.appendChild(list);

  if (payload.status === "failed") {
    box.appendChild(node("p", null,
      "What follows is partial. Don't rely on it as a full plan — send another message to retry the failed agents."));
  }
  box.hidden = false;
}

function renderPlan(payload) {
  renderNotice(payload);

  const target = el("plan");
  target.replaceChildren();
  renderFlights(payload.flight, target);
  renderHotels(payload.hotels, target);
  renderWeather(payload.weather, target);
  renderItinerary(payload.itinerary, target);
  renderCost(payload.budget, target);
  renderRunCost(payload.metrics, target);

  if (payload.warnings?.length) {
    const box = node("details", "flags");
    box.appendChild(node("summary", null, `${payload.warnings.length} automated check(s) flagged something`));
    const list = node("ul");
    payload.warnings.forEach((w) => list.appendChild(node("li", null, w)));
    box.appendChild(list);
    target.appendChild(box);
  }

  const summary = payload.status === "failed"
    ? "I couldn't complete this one — see the notice above."
    : payload.status === "degraded"
      ? "Done, though some details were unavailable. Ask me to adjust anything."
      : "Your plan is ready. Ask me to change the dates, budget or interests.";
  say("bot", summary);
}

/* ── first paint ─────────────────────────────────────────────────── */

say("bot", "Hello — I plan trips end to end. Tell me where you're going from, where to, and roughly when. I'll work out the rest, or ask if something's missing.");
input.focus();
