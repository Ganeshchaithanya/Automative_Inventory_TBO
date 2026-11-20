const $ = (selector) => document.querySelector(selector);
const currency = new Intl.NumberFormat("en-IN", { style: "currency", currency: "INR", maximumFractionDigits: 0 });
const number = new Intl.NumberFormat("en-IN");
let report;
let snapshot = "";

function node(tag, text, className) {
  const item = document.createElement(tag);
  if (text !== undefined) item.textContent = text;
  if (className) item.className = className;
  return item;
}

function link(text, url) {
  const item = node("a", text);
  item.target = "_blank";
  item.rel = "noopener noreferrer";
  item.href = url;
  return item;
}

function labelled(value, label) {
  const item = node("div", undefined, "impact-metric");
  item.append(node("strong", value), node("span", label));
  return item;
}

function section(title) {
  const item = node("section", undefined, "review-card evidence-card");
  item.append(node("h2", title));
  return item;
}

const actions = {
  HOLD_300_PLUS_300: "Take phased allocation",
  EXERCISE_300: "Exercise option",
  ABSTAIN: "Take no new action",
  RELEASE_OPTION: "Release option",
  BUY_300: "Buy 300 room-nights",
  BUY_500: "Buy 500 room-nights",
  WAIT: "Wait for more evidence",
  REJECT: "Close opportunity",
};

function render() {
  if (!report?.properties.length) return;
  const market = report.properties.find((row) => row.id === $("#event-select").value) || report.properties[0];
  const metrics = node("div", undefined, "impact-metrics");
  metrics.append(
    labelled(number.format(market.added), "Agent-approved room-nights"),
    labelled(number.format(market.featured_stock), "Event allocation still promoted"),
    labelled(number.format(market.featured_sold), "Promoted room-nights booked"),
    labelled(currency.format(market.revenue), "Demo booking revenue"),
  );

  const evidence = section("Event and historical evidence");
  const intro = node("p", market.destination + " · " + market.event_date + " · ");
  intro.append(market.event_source ? link("Organizer announcement", market.event_source) : document.createTextNode("Simulated event"));
  evidence.append(intro);
  const record = market.evidence;
  if (record.as_of) evidence.append(node("p", "Research checked " + new Date(record.as_of).toLocaleString() +
    " · MCP " + record.discovery_state));
  const facts = node("ul", undefined, "evidence-list");
  for (const fact of record.facts) {
    const item = node("li", fact.field + " · " + fact.value + " · ");
    item.append(link(fact.source_title, fact.source_url), document.createTextNode(" · " + fact.method + " · " + fact.use));
    if (fact.excerpt) item.append(node("p", "“" + fact.excerpt + "”"));
    facts.append(item);
  }
  evidence.append(facts, node("p", "Published attendance and visits are demand proxies. Supplier terms for the demo come from simulated contracts."));

  const properties = section("Real hotel research");
  properties.append(node("p", "Search window " + market.check_in + " to " + market.check_out + ". Public properties are independent of the simulated stock."));
  const candidates = node("ul", undefined, "evidence-list");
  for (const property of market.real_hotels) {
    const item = node("li");
    item.append(link(property.name, property.url), document.createTextNode(" · Availability unverified"));
    candidates.append(item);
  }
  properties.append(candidates, link("Check rates and dates in Google Hotels", market.hotel_search));
  properties.append(node("p", market.live_hotels.status));
  if (market.live_hotels.offers.length) {
    const live = node("ul", undefined, "evidence-list");
    for (const property of market.live_hotels.offers) {
      const item = node("li");
      item.append(link(property.name, property.url), document.createTextNode(" · Live offer found for 1 room, 2 adults"));
      if (property.total !== null && property.currency) {
        const total = new Intl.NumberFormat("en-IN", { style: "currency", currency: property.currency }).format(property.total);
        item.append(document.createTextNode(" · estimated stay total " + total));
      }
      live.append(item);
    }
    properties.append(live);
  }

  const model = section("Model decision");
  model.append(
    node("p", "Censored negative binomial · demand estimated from simulated booking and stockout data." +
      (market.forecast_calibrated ? " Calibrated separately by lead time and volatility." : "")),
    node("p", market.signal_prior
      ? "Comparable historical attendance adds +" + market.signal_prior.toFixed(3) + " to the demo’s initial demand signal."
      : "Historical attendance provides context. This market uses its synthetic demand signal."),
  );
  if (market.forecast) {
    const { p10, p50, p90 } = market.forecast;
    model.append(node("p", "Simulated room-night forecast · P10 " + number.format(p10) + " · P50 " + number.format(p50) + " · P90 " + number.format(p90)));
  }
  const stage = market.policy_stage === -10 ? "Option deadline" : "Opening allocation";
  model.append(node("p", stage + (market.volatile ? " · volatile scenario" : " · stable scenario")));
  model.append(node("p", market.event_active
    ? "Masked PPO decision · " + (actions[market.policy_action] || market.policy_action || "No active review")
    : "Opening allocation declined. Event promotion withdrawn."));
  model.append(node("p", "Company thresholds · " + currency.format(market.budget) + " budget · " +
    currency.format(market.downside_limit) + " downside limit"));
  if (market.candidate_assessment?.peak_cash) {
    model.append(node("p", "90% modeled downside CVaR · " + currency.format(market.candidate_assessment.downside_cvar90)));
  }

  const rates = section("Rate and commercial impact");
  const details = node("ul", undefined, "evidence-list");
  for (const line of [
    "Current demo room rate · " + currency.format(market.night_rate) + " per room-night",
    "Package transfer add-on · " + currency.format(market.transfer_rate) + " per room, per stay",
    "Modeled acquisition cost · " + currency.format(market.unit_cost) + " per room-night",
    "Committed cost of agent-approved stock · " + currency.format(market.committed_cost),
    "Gross margin on demo sales to date · " + currency.format(market.gross_margin_on_sold),
    "Total room-nights booked · " + number.format(market.sold) + " · " + number.format(market.available) + " still available",
  ]) details.append(node("li", line));
  rates.append(details);
  const panels = node("div", undefined, "impact-panels");
  panels.append(evidence, properties, model, rates);

  const sales = section("Recent demo bookings");
  const receipts = report.bookings.filter((row) => row.hotel_id === market.id);
  if (receipts.length) {
    const list = node("ul", undefined, "evidence-list");
    for (const booking of receipts) {
      list.append(node("li", booking.reference + " · " + (booking.offer === "package" ? "Hotel + transfer" : "Hotel") +
        " · " + booking.room_nights + " room-nights · " + currency.format(booking.subtotal_inr)));
    }
    sales.append(list);
  } else {
    sales.append(node("p", "Create a demo booking from the storefront to see sales here."));
  }
  $("#impact").replaceChildren(metrics, panels, sales);
}

async function refresh() {
  try {
    const response = await fetch("/api/impact", { cache: "no-store" });
    if (!response.ok) throw new Error("Inventory impact is unavailable.");
    const data = await response.json();
    const next = JSON.stringify(data);
    if (snapshot === next) return;
    snapshot = next;
    const selection = $("#event-select").value;
    report = data;
    $("#event-select").replaceChildren(...data.properties.map((market) => {
      const option = node("option", market.destination + " · " + market.event);
      option.value = market.id;
      option.selected = market.id === selection;
      return option;
    }));
    render();
  } catch (error) {
    if (!report) $("#impact").replaceChildren(node("p", error.message, "empty"));
  }
}

$("#event-select").addEventListener("change", render);
await refresh();
setInterval(refresh, 2000);
