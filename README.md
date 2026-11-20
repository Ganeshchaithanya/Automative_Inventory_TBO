# ASCIS synthetic procurement prototype

Run the hypothetical Bengaluru cricket-final replay from the workspace:

```powershell
python -m pip install -r requirements.txt
python -m ascis train
python -m ascis.skills
python -m ascis.evaluation
python -m ascis demo
python -m ascis demo --approve --report runs/bengaluru.json
python -m ascis demo --volatile --approve --report runs/bengaluru-volatile.json
python -m ascis adapt
python -m unittest discover -s tests
```

Training fits demand from censored bookings, corrects forecast bias by lead time and volatility, and calibrates forecast ranges on a separate set of synthetic events. It then imitates examples generated from the synthetic company policy and trains masked PPO on decisions at opening and option expiry. The checkpoint records before-and-after forecast coverage and error on fresh normal and stress cohorts, plus replay results against the company-rule heuristic.

The policy trains on normal and stressed markets with unsold-inventory and downside penalties. Train two seeds, select the checkpoint on separate validation markets, and score the final choice on untouched normal and stress cohorts. To benchmark against an existing checkpoint, run `python -m ascis --checkpoint runs/policy-new.pt train --reference runs/policy.pt`.

The domain mixture learns adjustments for forecast coverage and sell-through. The company mixture uses the domain estimates to adjust loss risk and full-sellout probability. The training command compares the two-MLP mixture with a mixture of a 32-unit gated residual network and a two-layer rank-8 feature-cross network. It trains both architectures at AdamW learning rates of `0.003` and `0.05` on the same synthetic splits. Validation selects one candidate before the independent normal and stress holdouts are scored. A candidate must improve every active learned output on both holdouts before replacing an existing checkpoint. Scores for new learned outputs are compared against the simulator. The approval and eligibility rules remain deterministic.

Procured sell-through is simulated room-nights sold divided by room-nights purchased. Stockout demand served is the share fulfilled above the base inventory of 600 room-nights. A successful allocation sells at least 90% of its purchased quantity and yields nonnegative simulated contribution. Read these together with average unmet demand, downside CVaR and masked violations. Actual storefront conversion requires real visits and bookings.

The replay reports the market trigger, supplier pool deduplication, versioned strategic reports, two stages of skills, scenario estimates and eligibility checks. Omitting `--approve` stops at the first proposal that requires a mock approval. Adding it simulates approval and writes confirmed inventory into a local SQLite ledger for that replay. The volatile replay changes the evidence before the option deadline.

The adaptation command detects recurring losses in a separate volatile replay set, trains a candidate when the pattern recurs, and compares it on held-out stress cases. It saves any candidate separately and displays the comparison. Select an alternate checkpoint with `python -m ascis --checkpoint <path> demo`.

This slice uses synthetic supplier terms and a rules-based research router. The design covers live MCP integrations, trainable latent attention routing, SARIMAX/Mamba, and a B2B interface as later stages. The replay emits distribution-ready records for mock listings.

See [the architecture plan](TBO-architecture-and-build-plan.md) for the full design.

## Interactive demo website

```powershell
python -m pip install -r requirements.txt
python -m ascis train
python -m ascis.skills
python -m ascis.web
```

Open [the storefront](http://127.0.0.1:8080) to browse current demo stock by destination, property and stay dates. The fictional properties have initial demo stock so the catalog is populated before any new proposal is approved.

Open [the operations demo](http://127.0.0.1:8083/) for the model and inventory impact view or [the approval inbox](http://127.0.0.1:8083/approvals) to review the agent’s requests. The two sites share a local SQLite ledger under `runs/`. On startup, the trained checkpoint evaluates the demo markets and queues proposals that require approval. Each proposal has a downloadable PDF bound to its exact terms. Accepting adds the firm quantity to the storefront. Declining leaves stock as-is.

Use [Scenario Lab](http://127.0.0.1:8083/scenarios) for what-if runs. Start with a stable market, volatile deadline, tight budget or routine demand, then change the event, timing, demand, cash, risk limit and quote verification. The lab calls the trained checkpoint and displays the forecast interval, candidate constraints and selected action. It runs independently of the booking and approval ledger.

**Reset demo** on the operations site clears simulated bookings, sales, procurement approvals, declines, and option activity after confirmation. It restores the seeded catalog and opens a new set of proposals. Open demo pages refresh to the new revision. The trained checkpoint, source configuration and exported PDFs are files outside the demo ledger.

Keep the storefront and approval inbox open in separate tabs. Approve an opening allocation to see its firm room-nights appear automatically. Twenty seconds after an opening hold is approved, the internal simulation advances to the option deadline. The agent can request approval to exercise, or allow the option to lapse when the market becomes volatile. Use Johannesburg for the stable scenario and Belgrade for the volatility scenario. The inbox updates automatically as decisions arrive. Set `ASCIS_LEDGER` to a new SQLite path to run a fresh demo without altering past approvals.

After approval, the storefront promotes acquired inventory and sells those room-nights before its starter stock on each stay date. Hotel-only and hotel-plus-transfer bookings share the same per-date room inventory. Packages also consume arrival transfer capacity. The impact view tracks booked room-nights, acquired inventory sold, demo rates, committed cost and margin on demo sales.

Public event lookup runs behind the scenes through a read-only MCP tool. Set `FIRECRAWL_API_KEY` in the server environment for Firecrawl web and news results. The default fallbacks are Wikipedia, GDELT news and pinned organizer sources. The operations site displays organizer-reported historical attendance as a demand proxy and verified public property links. A configured `BOOKING_DEMAND_TOKEN` and `BOOKING_AFFILIATE_ID` enable read-only event-date availability searches in the production partner catalog. External offers are separate from the fictional procurement and storefront.

Expand **MCP status** in the operations site to see whether its in-process MCP tool answered and which external search providers returned results. A connected MCP tool can still report an unavailable upstream source. Search results support event research. The forecast model still uses synthetic booking data and the PPO checkpoint still uses the demo company policy.

### Local evidence layer

The Impact view lists curated event and historical facts with source links. The local parser also extracts dates and attendance counts from MCP search excerpts and shows the source passage. Search mentions remain research context. The workflow displays curated facts if MCP is unavailable.

In Scenario Lab, choose Missing, Expired, or Conflicting supplier evidence. When the opening trigger fires, the workflow requests terms and skips PPO. At the option deadline it abstains and flags the option for review. Test these paths with `python -m ascis.evaluation`. Independent public context remains visible while supplier evidence is under review.

The storefront and approval inbox are for local use with the Python/PyTorch checkpoint. The demo performs no payment or supplier reservation. Use the CLI replays for the internal option deadline and volatility scenarios.

## Deploy the two demo sites

The GitHub repository name is `Automative_Inventory_TBO`. The Render service hosts the operations site at `/impact`, `/approvals`, and `/scenarios`. The Vercel project serves the storefront and forwards its catalog and booking endpoints to Render. Both sites use the same simulated inventory ledger.

1. Push this repository to GitHub and import it as a [Render Blueprint](https://dashboard.render.com/blueprints). Set `ASCIS_STOREFRONT_URL` to the intended Vercel project URL when prompted. The Blueprint loads the checked-in synthetic checkpoints and checks `/healthz` after startup.
2. Import the same repository as a [Vercel project](https://vercel.com/new), using the root directory. Set `ASCIS_OPERATIONS_URL` to the Render service origin, for example `https://automative-inventory-tbo-ops.onrender.com` if that is the hostname assigned by Render. Vercel builds the storefront with `vercel.json`.
3. Update `ASCIS_STOREFRONT_URL` in Render if Vercel assigns a different hostname. Verify the storefront offers load, then approve a demo allocation on Render and check that the storefront inventory reflects it.

The default Render Blueprint uses the Free plan and stores the shared demo ledger in `/tmp`. A redeploy or instance replacement can reset approvals and bookings. The hosted demo is a shared simulation and people with the operations link can reset it. The checked-in weights contain synthetic training results. The local ledger and local reports are excluded from Git.
