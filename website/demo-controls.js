const toolbar = document.querySelector("[data-demo-controls]");
let revision = null;

function node(tag, text, className) {
  const element = document.createElement(tag);
  if (text) element.textContent = text;
  if (className) element.className = className;
  return element;
}

const status = node("details", "", "mcp-status");
const summary = node("summary", "MCP · checking");
const checks = node("ul");
status.append(summary, checks);
const flash = node("span");
flash.setAttribute("role", "status");
flash.textContent = sessionStorage.getItem("ascis-reset-message") || "";
sessionStorage.removeItem("ascis-reset-message");
const reset = node("button", "Reset demo", "button outline");
reset.type = "button";
reset.disabled = true;
toolbar.append(status, flash, reset);

const dialog = node("dialog", "", "reset-dialog");
const title = node("h2", "Reset this demo?");
title.id = "reset-demo-title";
dialog.setAttribute("aria-labelledby", title.id);
const form = node("form");
form.method = "dialog";
const cancel = node("button", "Cancel", "button outline");
cancel.autofocus = true;
const confirm = node("button", "Reset demo data", "button emerald");
confirm.type = "button";
const feedback = node("p");
feedback.setAttribute("role", "alert");
form.append(
  title,
  node("p", "This clears simulated bookings, approvals, declines, option activity, and sales. The storefront returns to its starter stock and the agent creates fresh proposals."),
  cancel,
  confirm,
  feedback,
);
dialog.append(form);
document.body.append(dialog);

reset.addEventListener("click", () => {
  feedback.textContent = "";
  dialog.showModal();
});

confirm.addEventListener("click", async () => {
  confirm.disabled = true;
  try {
    const response = await fetch("/api/demo/reset", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ confirm: "reset-demo", revision }),
    });
    if (!response.ok) {
      const error = await response.json();
      throw new Error(typeof error.detail === "string" ? error.detail : "Unable to reset the demo.");
    }
    sessionStorage.setItem("ascis-reset-message", "Demo reset. Fresh proposals are ready.");
    location.reload();
  } catch (error) {
    feedback.textContent = error.message;
    confirm.disabled = false;
  }
});

async function refreshStatus() {
  try {
    const response = await fetch("/api/demo/status", { cache: "no-store" });
    if (!response.ok) throw new Error();
    const data = await response.json();
    if (revision !== null && data.revision > revision) {
      location.reload();
      return;
    }
    if (revision !== null && data.revision < revision) return;
    revision = data.revision;
    reset.disabled = false;
    const states = data.markets.map((market) => market.state);
    const label = states.includes("connected") ? "connected" : states.includes("checking") ? "checking" : "unavailable";
    summary.textContent = "MCP · " + label;
    checks.replaceChildren(...data.markets.map((market) => {
      const providers = Object.entries(market.providers).map(([name, state]) => name + ": " + state).join(", ");
      return node("li", market.destination + " — " + market.state +
        (providers ? " · " + providers + " · " + market.sources + " source links" : ""));
    }), node("li", "Live hotel partner lookup " + (data.live_hotel_provider_configured ? "configured" : "not configured")));
  } catch {
    summary.textContent = "MCP · status unavailable";
    reset.disabled = true;
  }
}

await refreshStatus();
setInterval(refreshStatus, 3000);
