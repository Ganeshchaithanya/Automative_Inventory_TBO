const $ = (selector) => document.querySelector(selector);
const form = $("#scenario-form");
const controls = form.elements;
const number = new Intl.NumberFormat("en-IN");
const currency = new Intl.NumberFormat("en-IN", { style: "currency", currency: "INR", maximumFractionDigits: 0 });
const actions = {
  ABSTAIN: "Take no new action", WAIT: "Wait for more evidence", REJECT: "Close opportunity",
  REQUEST_TERMS: "Request supplier terms", BUY_300: "Buy 300 room-nights",
  BUY_500: "Buy 500 room-nights", HOLD_300_PLUS_300: "Take 300 firm + 300 optional",
  EXERCISE_300: "Exercise the 300-night option", RELEASE_OPTION: "Release option",
};
let settings;
let previous;
let refreshingSettings = false;
let latestRun = 0;

function node(tag, text, className) {
  const item = document.createElement(tag);
  if (text !== undefined) item.textContent = text;
  if (className) item.className = className;
  return item;
}

function setPhaseHint() {
  $("#phase-hint").textContent = controls.phase.value === "deadline"
    ? "Starts with 300 firm nights already committed and an option on 300 more."
    : "Opening decision with no new allocation.";
}

function setSignal(value) {
  controls.signal.value = value;
  $("#signal-value").textContent = Number(controls.signal.value).toFixed(2);
}

function setPreset(preset) {
  const preferred = preset === "volatile" ? "expo" : "icc";
  const chosen = settings.events.find((event) => event.id === preferred) || settings.events[0];
  controls.event_id.value = chosen.id;
  controls.phase.value = preset === "volatile" ? "deadline" : "opening";
  controls.volatile.checked = preset === "volatile";
  controls.quote_verified.checked = true;
  controls.evidence_state.value = "current";
  controls.budget.value = preset === "cash" ? 700000 : settings.budget;
  controls.downside_limit.value = settings.downside_limit;
  setSignal(preset === "volatile" ? 0.35 : preset === "routine" ? 0.05 : chosen.signal);
  setPhaseHint();
}

function metric(label, value) {
  const item = node("div", undefined, "impact-metric");
  item.append(node("strong", number.format(value)), node("span", label));
  return item;
}

function section(title, description) {
  const card = node("section", undefined, "review-card lab-readout");
  card.append(node("h2", title));
  if (description) card.append(node("p", description, "lab-note"));
  return card;
}

function whySelected(data, selected) {
  if (!data.actor_ran) {
    if (!data.trigger.active && data.phase === "opening") return "The opening demand signal stayed below the market trigger, so the allocation policy remained idle.";
    return data.supplier_evidence.reason + (data.phase === "deadline"
      ? " The option needs review before any new action."
      : " The system requested updated terms before evaluating a commitment.");
  }
  const weight = data.action_weights?.[data.selected_action];
  const choice = weight == null
    ? "The policy selected this action from the eligible options."
    : "The policy ranked this highest among " + Object.keys(data.action_weights).length + " eligible actions with a " +
      (100 * weight).toFixed(1) + "% selection weight.";
  if (selected?.peak_cash) {
    return choice + " Expected contribution is " + currency.format(selected.expected_profit) +
      ", with " + currency.format(selected.downside_cvar90) + " modeled downside against a " +
      currency.format(data.downside_limit) + " limit.";
  }
  return choice + " This action makes no new inventory commitment.";
}

function stage(list, label, description) {
  const item = node("li");
  item.append(node("strong", label), node("span", description));
  list.append(item);
}

function candidateTable(data) {
  const wrapper = node("div", undefined, "lab-table");
  const table = node("table");
  const head = node("thead");
  const headers = node("tr");
  for (const label of ["Candidate action", "Peak cash", "Expected contribution", "90% downside", "Expected sell-through", "Policy weight", "Eligibility"]) {
    const field = node("th", label);
    field.scope = "col";
    headers.append(field);
  }
  head.append(headers);
  const body = node("tbody");
  for (const item of data.candidates) {
    const selected = item.action === data.selected_action;
    const row = node("tr", undefined, selected ? "chosen-row" : "");
    const weight = data.action_weights?.[item.action];
    row.append(
      node("td", actions[item.action] + (selected ? " · Selected" : "")),
      node("td", currency.format(item.peak_cash)), node("td", currency.format(item.expected_profit)),
      node("td", currency.format(item.downside_cvar90)),
      node("td", item.expected_sell_through_pct == null ? "—" : Math.round(item.expected_sell_through_pct) + "%"),
      node("td", weight == null ? "—" : (100 * weight).toFixed(1) + "%"),
      node("td", item.excluded_by.length ? item.excluded_by.join(", ") : "Eligible",
        item.excluded_by.length ? "lab-blocked" : ""),
    );
    body.append(row);
  }
  table.append(head, body);
  wrapper.append(table);
  return wrapper;
}

function skillTable(reviews) {
  const wrapper = node("div", undefined, "lab-table");
  const table = node("table");
  const head = node("thead");
  const headers = node("tr");
  for (const label of ["Allocation", "Domain · sell-through", "Domain · interval coverage",
    "Company · loss risk", "Company · full sellout"]) {
    const header = node("th", label);
    header.scope = "col";
    headers.append(header);
  }
  head.append(headers);
  const body = node("tbody");
  for (const review of reviews) {
    const row = node("tr");
    row.append(node("td", actions[review.action] + (review.eligible ? "" : " · Blocked")));
    for (const score of [...Object.values(review.domain), ...Object.values(review.company)]) {
      const challenger = score.source === "simulator baseline"
        ? " · learned estimate " + Math.round(score.challenger * 100) + "%" : "";
      const cell = node("td", Math.round(score.value * 100) + "% · " + score.source + challenger);
      row.append(cell);
    }
    body.append(row);
  }
  table.append(head, body);
  wrapper.append(table);
  return wrapper;
}

function render(data) {
  const verdict = node("div", undefined, "lab-verdict");
  const evidence = data.supplier_evidence;
  verdict.append(node("p", data.actor_ran ? "MASKED PPO DECISION" :
    evidence.eligible ? "MONITORING DECISION" : "EVIDENCE REVIEW", "eyebrow"));
  verdict.append(node("h2", actions[data.selected_action]));
  verdict.append(node("p", data.event + " · " + data.market));
  const selected = data.candidates.find((item) => item.action === data.selected_action);
  const explanation = section("Why this action was selected", whySelected(data, selected));
  const assumptions = node("ul", undefined, "lab-assumptions");
  for (const label of [
    (data.phase === "opening" ? "Opening decision" : "Option deadline"),
    (data.volatile ? "Volatile market" : "Stable market"),
    "Demand signal " + data.signal.toFixed(2),
    "Cash headroom " + currency.format(data.cash_headroom),
    number.format(data.existing_room_nights) + " existing room-nights",
  ]) assumptions.append(node("li", label));
  explanation.append(assumptions);
  const reviews = {
    queued: "Allocation sent for human review.",
    "existing request": "This market already has a request in the approval inbox.",
    "already approved": "This market’s opening allocation has already been approved.",
    declined: "This market’s opening allocation has been declined.",
  };
  if (data.review_status) {
    const message = node("p", reviews[data.review_status] + " ");
    const inbox = node("a", "Open Approval Inbox");
    inbox.href = "/approvals";
    message.append(inbox);
    verdict.append(message);
  }
  if (previous) {
    verdict.append(node("p", "Previous run: " + actions[previous.selected_action] +
      " · median forecast " + number.format(previous.forecast.p50) + " room-nights", "prior-run"));
  }
  const forecast = section("Demand forecast", "Expected total room-nights across the simulated event window.");
  const metrics = node("div", undefined, "impact-metrics lab-metrics");
  metrics.append(
    metric("P10 room-nights", data.forecast.p10), metric("Median room-nights", data.forecast.p50),
    metric("P90 room-nights", data.forecast.p90),
  );
  forecast.append(metrics);
  const details = section("How the system works");
  const trace = node("ol", undefined, "model-trace");
  stage(trace, "Market trigger", "Robust z-score " + data.trigger.robust_z +
    ". " + (data.trigger.active ? "Threshold crossed." : "Below the opening threshold."));
  stage(trace, "Product workflow", data.workflow.activated.join(" · ") + ".");
  stage(trace, "Supplier evidence", evidence.reason || "Current verified terms for this simulated decision window.");
  const fitted = data.fit.training_events ? " Fitted on " + data.fit.training_events +
    " synthetic events, including " + data.fit.stockout_events + " with inventory censoring." : "";
  const calibration = data.forecast_skill.calibrated
    ? " Forecast calibrated separately for each lead time and market regime." : "";
  stage(trace, "Demand reconstruction and forecast", "Censored negative-binomial model." + fitted + calibration);
  const evaluation = Object.entries(data.forecast_validation || {}).map(([suite, metrics]) =>
    suite + " " + metrics.coverage_pct + "% (" + metrics.events + " events)");
  if (evaluation.length) {
    stage(trace, "Synthetic P10–P90 coverage", "Target 80% · " + evaluation.join(" · ") + ".");
  }
  stage(trace, "Supplier and risk gates", (data.quote_verified ? "Supplier terms verified." : "Supplier terms unverified.") +
    " Cash headroom " + currency.format(data.cash_headroom) + ". " + number.format(data.existing_room_nights) +
    " existing room-nights.");
  stage(trace, "Strategic action", data.actor_ran
    ? (actions[data.selected_action] + (data.option_open && data.selected_action === "ABSTAIN"
      ? ". The optional allocation would lapse under these assumptions." : "."))
    : evidence.eligible
      ? "The opening trigger did not fire. The forecast above is available for inspection."
      : data.phase === "deadline"
        ? "Review the outstanding option. The system took no new commitment or release action."
        : actions[data.selected_action] + ". The system will only consider a commitment after supplier evidence is verified.");
  details.append(trace);
  const candidates = section("Candidate comparison",
    "Policy weights show how the model ranks eligible actions. Downside is the average modeled loss in the worst 10% of outcomes.");
  candidates.append(candidateTable(data));
  const panels = [verdict, explanation, forecast, details, candidates];
  if (data.skill_reviews?.length) {
    const skills = section("Domain and company skill review",
      "Advisory estimates from the simulator and learned models.");
    skills.append(skillTable(data.skill_reviews));
    panels.push(skills);
  }
  $("#result").replaceChildren(...panels);
  previous = data;
}

async function run(queueForReview = false) {
  const generation = ++latestRun;
  const button = $("#run-scenario");
  button.disabled = true;
  button.textContent = "Running…";
  $("#scenario-error").textContent = "";
  const inputs = {
    event_id: controls.event_id.value, phase: controls.phase.value, offer: controls.offer.value,
    signal: Number(controls.signal.value),
    volatile: controls.volatile.checked, quote_verified: controls.quote_verified.checked,
    evidence_state: controls.evidence_state.value,
    budget: Number(controls.budget.value), downside_limit: Number(controls.downside_limit.value),
    queue_for_review: queueForReview,
  };
  try {
    const response = await fetch("/api/scenarios", {
      method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(inputs),
    });
    const data = await response.json();
    if (!response.ok) throw new Error(typeof data.detail === "string" ? data.detail : "Check the scenario inputs and try again.");
    if (generation === latestRun && inputs.event_id === controls.event_id.value) render(data);
  } catch (error) {
    if (generation === latestRun) $("#scenario-error").textContent = error.message;
  } finally {
    if (generation === latestRun) {
      button.disabled = false;
      button.textContent = "Run scenario";
    }
  }
}

controls.signal.addEventListener("input", () => {
  $("#signal-value").textContent = Number(controls.signal.value).toFixed(2);
});
controls.phase.addEventListener("change", setPhaseHint);
controls.event_id.addEventListener("change", () => {
  const market = settings?.events.find((event) => event.id === controls.event_id.value);
  if (market) {
    setSignal(market.signal);
    if (form.reportValidity()) run(true);
  }
});
form.addEventListener("submit", (event) => {
  event.preventDefault();
  run(true);
});

async function refreshSettings() {
  if (refreshingSettings) return;
  refreshingSettings = true;
  try {
    const response = await fetch("/api/scenarios", { cache: "no-store" });
    if (!response.ok) throw new Error("Scenario inputs are unavailable.");
    const next = await response.json();
    if (JSON.stringify(next) === JSON.stringify(settings)) return;
    const selected = controls.event_id.value;
    settings = next;
    controls.event_id.replaceChildren(...settings.events.map((event) => {
      const option = node("option", event.destination + " · " + event.title);
      option.value = event.id;
      return option;
    }));
    if (settings.events.some((event) => event.id === selected)) {
      controls.event_id.value = selected;
    } else if (selected) {
      setSignal(settings.events[0].signal);
    }
    if (previous && !settings.events.some((event) => event.id === previous.event_id)) {
      $("#result").replaceChildren(node("p", "The allocation was declined. Select an active market to run a scenario.", "empty"));
      previous = undefined;
    }
  } finally {
    refreshingSettings = false;
  }
}

for (const button of document.querySelectorAll("[data-preset]")) {
  button.addEventListener("click", () => {
    if (!settings) return;
    setPreset(button.dataset.preset);
    if (form.reportValidity()) run();
  });
}

try {
  await refreshSettings();
  setPreset("stable");
  await run();
} catch (error) {
  $("#scenario-error").textContent = error.message;
}
const pollSettings = () => refreshSettings().catch(() => {});
window.addEventListener("focus", pollSettings);
setInterval(pollSettings, 2000);
