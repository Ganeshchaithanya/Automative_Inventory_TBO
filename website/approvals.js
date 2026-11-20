const requests = document.querySelector("#requests");
const messages = document.querySelector("#messages");
const money = new Intl.NumberFormat("en-IN", { style: "currency", currency: "INR", maximumFractionDigits: 0 });
let requestSnapshot = "";
let activitySnapshot = "";
let responding = false;

function node(tag, value, className) {
  const item = document.createElement(tag);
  if (value !== undefined) item.textContent = value;
  if (className) item.className = className;
  return item;
}

async function respond(request, accepted, checked, controls) {
  if (responding) return;
  responding = true;
  controls.forEach((button) => { button.disabled = true; });
  const path = "/api/agent/requests/" + request.id + (accepted ? "/approve" : "/decline");
  try {
    const response = await fetch(path, {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ proposal_id: request.proposal.id, acknowledge_simulation: checked }),
    });
    const result = await response.json();
    if (!response.ok) {
      throw new Error(result.detail || "Unable to resolve the request.");
    }
    messages.textContent = accepted
      ? (result.firm_added
        ? "Approved. The firm demo room-nights are now in the storefront."
        : "Release approved. The existing firm allocation stays in the storefront.")
      : request.stage === "Opening allocation"
        ? "Declined. The event was removed from active scenarios and storefront promotion."
        : "Optional allocation declined. Existing firm inventory remains available.";
  } catch (error) {
    messages.textContent = error.message;
    controls.forEach((button) => { button.disabled = false; });
  } finally {
    responding = false;
    await refresh();
  }
}

function card(request) {
  const record = node("article", undefined, "review-card");
  const proposal = request.proposal;
  const label = node("label");
  const check = node("input");
  check.type = "checkbox";
  check.dataset.proposal = proposal.id;
  label.append(check, document.createTextNode(" I’m approving demo inventory only."));
  const release = proposal.action === "RELEASE_OPTION";
  const approve = node("button", release ? "Approve release" : "Approve allocation", "button emerald");
  const decline = node("button", "Decline", "button outline");
  const pdf = node("a", "Download review PDF", "button outline pdf-link");
  pdf.href = "/api/agent/requests/" + request.id + "/review.pdf?proposal_id=" + encodeURIComponent(proposal.id);
  approve.disabled = true;
  check.addEventListener("change", () => { approve.disabled = !check.checked; });
  approve.addEventListener("click", () => respond(request, true, check.checked, [approve, decline]));
  decline.addEventListener("click", () => respond(request, false, false, [approve, decline]));
  record.append(
    node("span", request.stage, "chip"), node("h3", request.title),
    node("p", request.destination + " · " + request.event_title),
    node("p", release ? "Release the optional lot · No additional purchase"
      : proposal.firm_units + " firm room-nights · " + money.format(proposal.cost_inr) + " simulated cost"),
    node("p", proposal.option_units + " optional room-nights · modeled maximum " + money.format(proposal.maximum_cash_inr)),
    node("p", request.reason),
    node("p", "Supplier evidence · " + request.supplier_evidence.state + " for this simulated decision window."),
    pdf, label, approve, decline,
  );
  return record;
}

async function refresh() {
  try {
    if (responding) return;
    const [requestsResponse, activityResponse] = await Promise.all([
      fetch("/api/agent/requests", { cache: "no-store" }),
      fetch("/api/agent/activity", { cache: "no-store" }),
    ]);
    if (!requestsResponse.ok || !activityResponse.ok) throw new Error("Approval requests are unavailable.");
    const [items, history] = await Promise.all([requestsResponse.json(), activityResponse.json()]);
    if (responding) return;
    const nextRequests = JSON.stringify(items);
    if (nextRequests !== requestSnapshot) {
      const checked = new Set([...requests.querySelectorAll("input:checked")].map((item) => item.dataset.proposal));
      requestSnapshot = nextRequests;
      requests.replaceChildren(...(items.length ? items.map(card) : [node("p", "No allocations are awaiting approval.", "empty")]));
      for (const input of requests.querySelectorAll("input")) {
        if (checked.has(input.dataset.proposal)) {
          input.checked = true;
          input.dispatchEvent(new Event("change"));
        }
      }
    }
    const nextActivity = JSON.stringify(history);
    if (nextActivity !== activitySnapshot) {
      activitySnapshot = nextActivity;
      document.querySelector("#activity-panel").hidden = !history.length;
      document.querySelector("#activity").replaceChildren(...history.map(({ event, status }) => node("li", event + " · " + status)));
    }
  } catch (error) {
    if (!requestSnapshot) requests.replaceChildren(node("p", error.message, "empty"));
  }
}

await refresh();
setInterval(refresh, 2000);
