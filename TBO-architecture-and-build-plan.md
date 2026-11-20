# ASCIS architecture and build plan for TBO

This design treats PS4 inventory procurement as the principal use case and PS5 supplier identity, capacity and contract comparison as prerequisites. The proposed hackathon implementation starts with hotel room-nights and hotel-plus-transfer packages for a single event market. Flights, activities and broader bundles use the same data and decision contracts. All commercial terms, board preference configurations, demand and transactions in the prototype are synthetic. Configuration values below are initial experiment settings.

## Architecture and authority

~~~mermaid
flowchart TD
    E["Event calendars, suppliers, travel systems, searches, bookings"] --> MCP["Authorized MCP adapters"]
    MCP --> O["As-of observations and evidence store"]
    O --> A["Central agent, Mode A: monitor and score triggers"]
    A -->|Routine| A
    A -->|Material trigger| S["Open or resume strategic session"]
    S --> B["Central agent, Mode B: retrieve analogs and route workflows"]
    MCP --> B
    B --> DAG["Dependency DAG: hotel, flight, activity, transfer, package"]
    DAG --> P["PS5: resolve entities, normalize terms, deduplicate capacity"]
    P --> D["Reconstruct censored demand"]
    D --> F["Forecast demand, supply, price and uncertainty"]
    F --> R["Reconcile products, evaluate contract economics, publish report Vn"]
    R --> M1["Stage I: professional and domain skills"]
    R --> M2["Stage II: company preference and alignment skills"]
    M1 --> M2
    R --> U["Uncertainty, historical cases and scenario simulation"]
    M1 --> U
    M2 --> U
    U -->|Evidence gap, within iteration budget| S
    U --> GATE["Apply company thresholds to candidate actions"]
    GATE -->|State and eligible action mask| RL["Company-policy pretrained RL judges evidence and selects action"]
    RL -->|New commitment or inventory action| V["Static executor validates terms, capacity and budget"]
    RL -->|Abstain from new action| A
    V -->|Material change| S
    V --> H["Authorized person approves exact commitment"]
    H -->|Approved| X["Static executor commits, reconciles and updates inventory"]
    H -->|Rejected or expired| N["Close or reassess session"]
    X --> M["B2B inventory ranking and distribution"]
    M --> L["Outcomes and offline model evaluation"]
    X --> L
    L --> I["Detect recurring failures, train and validate candidate policies"]
    I --> B
    I --> RL
    S <--> ES["Versioned episodic memory"]
    U --> ES
~~~

The two modes belong to one logical central agent. MCP carries authorized observations and tool responses. The central agent owns analysis and financial calculations. Deterministic services calculate values and write a shared report payload with a human-readable rendering. Two groups of specialized skills review the report and produce advisory evidence. The scenario engine can request evidence and restrict applicability according to explicit rules. RL selects a strategic action among feasible candidates. A human can approve or reject a commitment. The executor validates and applies the exact approved action. Marketing receives confirmed, publishable inventory. [Official MCP specification](https://modelcontextprotocol.io/specification/2025-11-25)

**Lifecycle.** Monitor → triggered → researching → deliberating → stress testing → policy selected → validating → awaiting approval → executing → confirming → distributing → settled. Rejected, expired and evidence-insufficient cases have explicit terminal or reassessment paths. Keep deadlines inside the state machine and wake it before quote expiry or supplier release. A wait action includes a review time and its governing deadline. Use one session per overlapping opportunity cluster (event, market, service date and product scope) with cooldown and severity escalation. A contested supplier pool or cash reservation is shared across sessions.

**Dependencies.** Canonical IDs, contract terms and timestamped evidence precede demand fitting. Known stockout or availability history precedes reconstruction. Common demand draws and approved supplier constraints precede economic comparisons. The skill orchestration stage consumes a versioned report. The policy receives calibrated forecasts, current commitments, skill findings and alignment scores, stress metrics and a feasible action mask. A new quote, approval or cancellation invalidates only graph descendants. DAG tasks use input hashes, freshness ranges and source IDs to reuse retrieval. Analysis uses read-scoped tools. Transaction write credentials belong to the execution service.

## Monitoring, event retrieval and routing

For metric $x_t$ use comparable trailing weekday and booking-lead buckets, starting with 8 weeks. On sparse buckets use a pooled baseline and a scale floor. Starting robust score, EWMA and look-to-book are

$$z_t=0.67449\frac{x_t-\mathrm{median}(x)}{\max(\mathrm{MAD}(x),\epsilon)},\qquad e_t=0.2z_t+0.8e_{t-1},\qquad LTB_t=\frac{\text{eligible searches}}{\max(1,\text{confirmed bookings})}.$$

Track zero bookings explicitly and require a minimum denominator population before treating LTB as stable. Deduplicate searches and monitor tracking outages. Start with $|z|\ge3.5$ for a sudden move or two buckets with $|e|\ge2$ for persistence. Orient changes commercially: a price spike, search surge, supplier outage and falling conversion have different meanings. Isolation Forest is an optional multivariate detector (200 trees, 256 observations per tree, initial alarm at validation 99th-percentile anomaly severity). Alert thresholds come from event-level commercial labels and a false-alert budget. Allow governance or impending financial deadlines to trigger independently.

For event $e$, estimate incremental room-night pressure in zone $z$ with a unit-aware event demand pull index

$$EDPI_{ez}=\frac{A_e u_e o_e n_e}{g_e}\,m_e\,
\frac{w_z\exp(-t_{ez}/\tau)}{\sum_{z'}w_{z'}\exp(-t_{ez'}/\tau)}.$$

Here $A$ is unique attendance, $u$ nonlocal share, $o$ overnight share among nonlocals, $n$ nights per overnight visitor, $g$ guests per room, $m$ TBO-addressable share, $w_z$ zone attractiveness and $t_{ez}$ transit minutes. Start with $\tau=30$ minutes and propagate attendance and share ranges. Multi-match attendance is deduplicated before adding pressure. Initial commercial score on [0,1] weights anomaly, event pressure, projected shortage and exposure at 0.25, 0.20, 0.35 and 0.20. Open a case at 0.70 and re-arm below 0.50, with a separate priority override for critical stockouts or risk.

**Analog retrieval.** Rank events by $0.35\,s_{\mathrm{text}}+0.50\,s_{\mathrm{structured}}+0.15\,s_{\mathrm{lead/season}}$. Use a frozen sentence encoder with normalized cosine, Gower similarity for mixed numeric and categorical features, and a distance decay on announcement lead time and service season. Structured fields include attendance, visitor origin, lodging/transport mix, spend band, transit topology, competition/event type and supplier depth. Require enough shared fields, expose missingness and weights, and retrieve eight previous events initially. Learn weighted similarity only with graded analog judgments. Cases include failed procurements and age-adjusted data, with realized outcomes restricted to those known at query time.

**Standalone routing network.** Use a low-rank multi-head cross-attention encoder inspired by MLA. Tokens represent event, zone, segment, supplier and retrieved case features with source/missing-value indicators. Each product or investigation has a learned query. For input $X$ and workflow queries $Q_w$, project keys and values through a shared latent $C=XW_c$, then

$$H_h=\operatorname{softmax}\left(\frac{(Q_w W_h^Q)(C W_h^K)^\top}{\sqrt{d_h}}+M\right)(C W_h^V),\qquad
p_w=\sigma\left(\mathrm{MLP}_w([\mathrm{concat}_h H_h,\ q_w])\right).$$

$M$ masks missing/padded tokens. Train the output heads using weighted binary cross-entropy, with optional dependency penalty $\lambda\sum_{c\to p}[p_c-p_p]_+$ for hierarchical labels. Commercially useful workflow labels come from simulated incremental evidence or reviewer-labeled historical cases, with an independently authored utility rule for validation. Product routing probabilities come from the supervised output heads. Raw attention weights can be inspected for diagnostics. Include an event-only outcome, always fetch required financial evidence, close selected tasks over mandatory dependencies and add investigations when evidence gaps remain.

Start with width 64, four heads of 16, latent rank 24, two blocks, hidden width 128, at most 64 tokens, dropout 0.10, AdamW learning rate $3\cdot10^{-4}$, weight decay $10^{-4}$, batch 64, up to 30 epochs with patience 5. Start routing at $p_w\ge0.5$ and tune thresholds per workflow for high recall of mandatory evidence. Compare matched-budget standard attention, MLP and boosted trees, plus rules. The design borrows latent compression from MLA and trains the router independently. [DeepSeek-V2 technical report](https://arxiv.org/abs/2405.04434)

## Supplier intelligence and demand reconstruction

Resolve hotel identity first using city and geographic blocks, then rank by name/address similarity, distance, phone and trusted IDs. Start with regularized logistic regression ($C=1$) and compare deterministic matching and gradient boosting. Initial auto-merge probability 0.98, review band 0.80–0.98, subject to independent precision calibration and hard identity conflicts. Train on positives and hard negatives such as nearby chain properties. Audit cluster consistency and overmerges.

Normalize room attributes and booking terms before comparison. Keep meal plan, occupancy, bed type, refund window, currency, taxes, mandatory fees and inclusions attached to the offer. Compare activity duration, transfers, itinerary and cancelability in the same way. Enrich listings with evidence-linked licensed content and images. Track physical allocation pools separately from hotel and supplier identities. Two 80-unit listings pointing to one 80-unit pool have a combined capacity of at most 80. For unverified pool relationships, use a conservative bound and hold ambiguous additive commitments for review. Estimate supplier fulfillment using shrunk rates such as $(s+19)/(n+20)$ for successes $s$ out of $n$, a synthetic Beta(19,1) prior to be recalibrated. Stress correlated failures across suppliers sharing an upstream provider. Rank valid offers by landed cost, terms, reliability, and service fit.

For an interval with known capacity $C_t$ and validated stockout evidence, let $Y_t=\min(D_t,C_t)$, where $Y$ is gross booked units and $D$ is requested demand at the observable price and service conditions. Use a negative binomial with $\operatorname{Var}(D)=\mu+\mu^2/k$ and $\log\mu_t=\beta^\top x_t$, including lagged search, lead time, availability exposure, event features and price. Optimize

$$\ell=\sum_{t\in U}\log P_\theta(D=Y_t\mid x_t)+\sum_{t\in S}\log P_\theta(D\ge C_t\mid x_t).$$

$U$ are verified uncensored intervals and $S$ are known binding-capacity intervals. Use verified lower bounds and the same survival term for partial observation where justified. A zero-exposure or ambiguously censored interval gives little or no direct demand information. Split intraday windows at stockout, refill and cancellation timestamps where possible. Model cancellations and return-to-stock explicitly. Use failed booking reason codes and deduplicated searches as features or evidence, with distinct treatment for technical failures. Use $k=10$ as an initialization and fit dispersion jointly. Compare observed bookings, correction using in-stock conversion, and censored NB. Draw from the conditional tail for stockouts and carry imputation uncertainty forward into the forecast. Audit sensitivity to supply endogeneity and sparse in-stock examples. Report lower bounds or wide intervals where the censored tail is weakly identified.

## Forecasts, cold starts and uncertainty

Forecast gross customer intent and cancellation-adjusted fulfillable demand at daily service-date and lead-time grains, then assemble origin-destination, hotel room-night, arrival/departure, flight seat, activity and package outputs. Keep bookings by booking date for velocity. Forecast supplier quote prices and capacity separately as needed. Use event schedule and holidays as known future covariates. Lag searches, booking and cancellation signals or provide a separate forecast/scenario for their future paths. Model price as externally supplied or scenario-dependent. Treat observational price coefficients as predictive until separately identified.

**SARIMAX.** For $y_t=\log(1+D_t)$ use $y_t=\beta^\top X_t+u_t$ and

$$\Phi_P(B^7)\phi_p(B)(1-B)^d(1-B^7)^D u_t=\Theta_Q(B^7)\theta_q(B)\varepsilon_t.$$

Start at $(1,0,1)\times(1,0,0)_7$, with small grids $p,q,P,Q\in\{0,1\}$ and $d,D\in\{0,1\}$ only where diagnostics support differencing. Encode annual/holiday variation with calendar features for sufficiently long series. Simulate on the modeled scale then back-transform the draws. Compare seasonal naive and a pooled regularized count or gradient-boosted quantile model. [statsmodels SARIMAX documentation](https://www.statsmodels.org/stable/generated/statsmodels.tsa.statespace.sarimax.SARIMAX.html)

**Global Mamba SSM.** Train across destinations and event categories. Its selective recurrence has the schematic form $h_t=\bar A(\Delta_t)h_{t-1}+\bar B(\Delta_t,B_t)x_t,\ y_t=C_t h_t+d_{\mathrm{skip}}x_t$, with input-dependent $\Delta_t,B_t,C_t$. Feed time-dependent history and static embeddings to direct horizon heads conditioned on known future covariates. Start at width 64, two blocks, state size 16, expansion 2, convolution width 4, 56 history days and 28 days of direct outputs. Use batch 32, AdamW at 0.001, dropout 0.10 on projected features, clipping at 1, up to 40 epochs with patience 5. Train P10/P50/P90 heads with pinball loss $\rho_\tau(e)=e(\tau-\mathbf 1_{e<0})$, enforce ordered outputs and benchmark against the same inputs in a GRU or temporal convolution. Use a supported GPU environment for an official-kernel Mamba experiment. A CPU baseline keeps the main demo runnable and reports the Mamba experiment as pending if the environment is unavailable. [Mamba paper](https://arxiv.org/abs/2312.00752), [Mamba implementation](https://github.com/state-spaces/mamba)

**Cold start and ensemble.** Pool by city, event category and service region. Train a regularized analog-uplift baseline on top of destination seasonality and compare it with global forecasts. Weight analogs by retrieval similarity and shrink sparse groups toward the category mean. Update as search, ticket and booking observations arrive. Evaluate retrieval and forecasting on events and destination regimes held out as groups. Estimate nonnegative forecast-distribution mixture weights summing to one on rolling validation origins. A grid in 0.1 steps or a constrained optimization of proper scoring rules suffices. Promote an ensemble only if it improves on the strongest member with acceptable tail calibration. Refit weights using validation only.

**Reconciliation.** Forecast disjoint purchase intents $b$ and derive components using the bill of materials $y=Sb,\ b\ge0$. For hotel-plus-transfer orders, map both components and the service dates they consume. When combining independent aggregate forecasts, reconcile each joint scenario by $\arg\min_{b\ge0}(Sb-\hat y)^\top W^{-1}(Sb-\hat y)$ with diagonal forecast-error covariance first. Compute intervals after reconciliation. Simulate shared event, date and supplier shocks so components and dates remain dependent. Keep substitution and package conversion explicit.

**Calibration and volatility.** On a held-out rolling calibration block use scores $\max(q_{.10}-D,D-q_{.90})$ and expand the interval by the empirical conformal correction. Check coverage by lead time and novelty, use pooled fallback for small groups and widen unfamiliar cases based on held-out event errors. Time dependence or regime shifts can invalidate exchangeability assumptions, so measure achieved coverage. Use GARCH(1,1)-X on suitable residuals or log price changes only after residual ARCH diagnostics and rolling validation. Start $\alpha=0.05,\beta=0.90$, constrain $\alpha,\beta\ge0,\alpha+\beta<1$ for the base recursion $v_t=\omega+\alpha e_{t-1}^2+\beta v_{t-1}+\gamma^\top z_{t-1}$, and require nonnegative variance and stationary exogenous dynamics where assumed. Start with 250 regularly spaced residuals as a feasibility check, then compare to EWMA volatility. [Conformalized Quantile Regression](https://arxiv.org/abs/1905.03222)

## Commercial calculations and risk

All values compare incremental economics to existing commitments and the same no-new-procurement baseline. Use supplier-authorized selling prices $p_i$ and the contract's commissionable base $b_i$ per room-night or ticket. For an eligible settled sale indicator $m_i$, commission is $\Pi_{\rm comm}=\sum_i m_i b_i r_i-\mathrm{incremental\ fees}$, with canceled/refunded lines treated according to their contract. Illustrative synthetic rates are 2.5–4% for flights and 6–15% for hotels and packages, with package exceptions explicitly modeled.

For resale, $\Pi_{\rm resale}=\sum_i p_i^{\rm net} x_i-\sum_j c_jq_j+V_{\rm recovery}-F_{\rm hold}-F_{\rm release}-F_{\rm operating}$, where $q$ is the billed commitment and $x$ the net sold quantity. Acquisition cost already includes purchased unsold units. Add commission to a resale line only where the signed agreement supports both. Track tax, FX, refund timing, deposit and settlement separately. Reserve peak cash at risk, including existing obligations, and enforce the available budget path up to settlement. Make supplier margins and permitted prices external contract inputs.

Model release as a timed second-stage action. A hold $Q$ commits a minimum $m$, allows $m\le q_{\rm retain}\le Q$ before the deadline, and incurs its specified fee or penalty. Enforce each stay date, quantity step, minimum obligation, exclusivity and pool cap. Cross-date stays and package components must all be fulfillable together.

Simulate common random paths for each candidate on correlated attendance, booking lead curves, cancellations, supplier outages and price/FX changes. Start with 2,000 paths, increase near ranking ties or unstable tails, and evaluate against structural stresses such as event postponement and late booking acceleration. Share an as-of information history across a scenario tree. Simulated release or purchase decisions use only what has arrived by their respective decision times. For downside $L=\max(0,-\Delta\Pi)$,

$$\operatorname{CVaR}_{\alpha}(L)=\min_\eta\left[\eta+\frac{1}{1-\alpha}\mathbb E(L-\eta)_+\right].$$

Use $\alpha=0.90$ for development, report 0.95 in stress testing with tail sample counts. Include profit distribution, loss probability, unsold units, unmet demand and peak capital. Compare action outcomes under common shocks. Higher price or marketing uplift counterfactuals are scenario assumptions unless a validated causal design supports them. [Rockafellar and Uryasev on CVaR](https://sites.math.washington.edu/~rtr/papers/rtr179-CVaR1.pdf)

## Skill orchestration and reanalysis

**Report.** The report schema includes session/version, as-of cutoff, trigger and market map, sources and provenance, retrieved analogs, routing and DAG trace, reconstruction assumptions, forecast draws and coverage, supplier and pool comparisons, cash flows, governing constraints, alternatives, marketing hypotheses and open questions. Render JSON and readable views from the same payload.

**Skill contract.** Register each skill with an ID, version, stage, activation predicate, typed inputs and outputs, required evidence, and method. Each run records the report version, applicable candidate IDs, findings with units and evidence IDs, assumptions, and open questions. A skill can return insufficient evidence and request a named input. Independent skills run in parallel within a stage. Persist outputs by input hash so a revised report reruns only affected skills. Start with explicit activation rules. The MLA-inspired router operates earlier and activates product research workflows.

| Stage | Example skill | Inputs | Output |
|---|---|---|---|
| I | Forecast calibration | Forecast intervals, holdout coverage and event similarity | Coverage by lead time, novelty flag and a request for wider intervals when warranted |
| I | Contract comparison | Dated quotes, release terms, minimums, and allocation pool IDs | Comparable offers, duplicated capacity, missing obligations and affected candidates |
| I | Liquidity analysis | Budget, commitments and settlement schedule | Peak cash by candidate and the date of any funding gap |
| II | Capital preservation | Stage I cash and contract findings, scenario loss estimates and synthetic company preferences | Candidate alignment scores and the terms driving the rankings |
| II | Availability and growth | Expected fulfillment, addressable agency segments and synthetic priorities | Service gap, growth alignment and ranked alternatives |
| II | Supplier relationship alignment | Concentration, reliability and synthetic relationship priorities | Portfolio trade-offs and company-alignment findings |

**Composition.** Stage I applies domain methods. Stage II consumes the report and applicable Stage I findings to evaluate company alignment. Additional registered skills can cover economics, tourism, consumer behavior, supply chain, risk and strategy. For synthetic company preference weights, use $u_j(a)=w_j^\top f(a)$ on standardized candidate features and aggregate preference votes as $\sum_j \omega_j\,\mathbf 1[a=\arg\max u_j]$. Board profiles configure weights and priorities. If enough authorized decision pairs become available, fit regularized Bradley–Terry preferences $P_j(a\succ b)=\sigma(u_j(a)-u_j(b))$. Votes and scores are advisory inputs for scenario review and the RL policy. Skills can use deterministic code, statistical models or evidence-grounded text generation as their method requires. Evaluate each skill against its own output contract and assess whether its contribution improves the scenario review.

**Reanalysis.** Keep statistical intervals, model disagreement, source uncertainty and unresolved commercial terms visible. Trigger another pass for stale quotes, unknown pools, poor analog coverage, conflicting skill findings that could change the action, or a stress result near a budget/risk boundary. Stress test material commitments even when skill findings align. Start with two reanalysis passes and stop earlier on evidence sufficiency or contractual deadline. Unresolved mandatory terms restrict the action mask to eligible wait, request-quote or reject actions. Session state stores source snapshots, report versions, candidate history, prior skill outputs and the DAG invalidation trace.

## Company-policy pretrained RL, abstention and adaptation

This block runs after both skill stages and the counterfactual review, before the static executor and human approval. It decides whether the upstream predictions and supporting evidence are reliable enough to justify a specific action now. The forecasts, skill findings and scenario results inform that decision alongside company priorities. A strong demand forecast can still lead to abstention when the potential loss, model disagreement or market volatility makes a commitment hard to justify.

Formulate a finite-horizon constrained decision process over the event lifecycle. Include as-of forecasts, interval width and historical calibration, model disagreement, volatile-market and unfamiliar-regime indicators, evidence gaps, existing inventory and cash commitments, supplier terms and deadlines in $s_t$. For each candidate include its peak cash, downside, fulfillment and performance relative to abstention under alternative plausible forecasts. Candidates include buying a specific lot, taking a flexible hold, diversifying across suppliers, waiting until a specified review time, rejecting, requesting terms, and retaining or releasing inventory.

**Abstain from new action.** Include an explicit no-new-commitment action in every decision state. The policy can choose it even when a purchase is technically feasible. The system records why it abstained and returns the opportunity to passive monitoring. A material change in evidence can trigger a new decision. A wait action schedules a specific follow-up while the opportunity remains open. A reject action closes the current opportunity. Existing payment, release and approval deadlines stay active when the policy abstains. If abstention would leave a required action unresolved, escalate the existing obligation to authorized operations and follow the signed contractual default.

For each candidate $a$, calculate required peak cash $K(a)$ and scenario downside $\operatorname{CVaR}_\alpha(L_a)$. Let $B_t$ be available cash headroom after existing commitments and $\bar L$ the configured loss limit. For a commitment candidate, the action mask is

$$
m_t(a)=\mathbf 1\big[K(a)\le B_t,\ \operatorname{CVaR}_\alpha(L_a)\le\bar L,\ \mathrm{eligibleTerms}(s_t,a)\big].
$$

Eligible terms require fresh evidence, sufficient compatible capacity, permitted suppliers and terms, and a feasible approval and execution window. Each action type gets its corresponding checks. For example, waiting requires a feasible next review time, and rejecting the opportunity cannot cancel an existing obligation. When mandatory evidence is missing, only actions that remain contractually permissible can pass. The actor scores the remaining candidates and makes the demo decision deterministically using $a_t^*=\arg\max_{a:\,m_t(a)=1}\pi_\theta(a\mid s_t,m_t)$. The threshold gate supplies eligibility. The pretrained actor supplies the preference among eligible actions. The executor revalidates the choice against fresh data before approval and execution.

**Company-policy pretraining.** Translate the authorized company policy version into hard restrictions, preference weights and representative decision examples. Include budget and liquidity limits, downside tolerance, supplier restrictions, availability objectives and evidence requirements. For the hackathon, use a clearly labeled synthetic policy and hypothetical decision histories. Where suitable demonstrations exist, initialize the actor by imitating company-approved actions using $L_{\mathrm{BC}}=-\mathbb E[\log\pi_\theta(a_{\mathrm{approved}}\mid s,m)]$. Then pretrain masked PPO in the simulator under the same restrictions and company-weighted rewards. Sample volatile and unfamiliar markets, including episodes where abstention avoids losses and episodes where excessive abstention causes shortages. Select a checkpoint using held-out events and stress scenarios. Version the training policy configuration with the checkpoint and apply the current approved hard rules at inference.

Define reward in normalized currency with a contribution term, incremental service shortfall, time-weighted committed capital and operational cost. For example, use $r_t=[\Delta\Pi_t-c_U U_t-c_KK_t]/S$ with procurement and settlement cash effects recorded once, and $K_t$ in currency-days. Penalize unmet demand and capital separately from financial cost where the terms express explicit policy preferences. An episodic downside constraint $\operatorname{CVaR}_{\alpha}(\max(0,-\Delta\Pi))\le\bar L$ is a separate policy-training experiment to assess on independent rollouts. Use masked PPO with

$$L_{\rm clip}=\mathbb E\big[\min(\rho_t\hat A_t,\operatorname{clip}(\rho_t,1-\epsilon,1+\epsilon)\hat A_t)\big],\quad
\rho_t=\frac{\pi_\theta(a_t\mid s_t)}{\pi_{\rm old}(a_t\mid s_t)}.$$

Use an MLP state encoder [128,128], shared candidate encoder [64,64], scalar scoring head per candidate, masked softmax, and a separate value head. Start with $\gamma=0.99$, GAE 0.95, PPO clip 0.2, learning rate $3\cdot10^{-4}$, entropy coefficient 0.01, value coefficient 0.5, 2,048 steps per rollout, minibatch 128 and 5 epochs. Evaluate 100k–500k simulator steps across at least five seeds. Begin with action masking and the deterministic risk gate. Compare a Lagrangian risk-cost variant only when the independent simulator shows a meaningful trade-off. Benchmarks are no-buy, reorder, newsvendor, stochastic two-stage programming with the same information and hard constraints, and a clairvoyant oracle as a labeled ceiling. RL earns its role when timing, releases or repeated decisions deliver value over the simpler policies. The hackathon checkpoint stays in simulation. Log failure clusters, retrain offline, run paired stress and held-out tests, and use a versioned approval and rollback process for policy promotion. [PPO paper](https://arxiv.org/abs/1707.06347)

**Why RL and how it adapts.** Buying inventory now changes cash, availability and the choices available at later release deadlines. Settlement then reveals delayed outcomes. RL learns from those sequences and outcomes. After a poor outcome, compare the decision-time evidence, chosen action and realized result. Attribute whether the problem came from a forecast, supplier failure, commercial preference, execution or the learned policy. Use feasible replay or modeled alternatives when estimating regret and label their uncertainty. Cluster similar failures by decision-time signals and action, and require evidence across independent events before treating a pattern as recurring.

The learning pipeline automatically generates a candidate policy version for a recurring decision failure. Replay and augment those cases in the simulator, add similar successful cases, and retrain against the approved company rules. For example, repeated losses from large firm buys under high model disagreement and unreliable capacity can lead a new policy to prefer smaller flexible holds or abstention in that state. Compare old and new policies on independent cases and adjacent regimes. Check whether recurrence falls without exceeding risk limits, increasing avoidable stockouts, or causing excessive abstention. Only promote a candidate after validation and the authorized deployment process. The learned action policy can change through this loop. Changes to official company rules and financial limits require company approval.

## Execution, B2B distribution and learning

Bind human approval to action ID/hash, report version, quantity, supplier/pool, price, currency, quote expiry and payment obligations. Commit inventory and ledger reservations transactionally in the internal store. Use idempotency keys and an outbox for supplier calls, reconcile ambiguous timeouts before retrying, and record compensation/manual exception paths if payment succeeds before supplier confirmation. The same approval applies only within its explicit scope. Send changed terms or strategic release choices back to the policy through the session. The prototype can replay duplicates, stale quotes and timeout responses with a mock supplier and mock payment ledger.

Rank confirmed inventory for eligible travel agents, tour operators and OTAs with segment fit, agent origin and destination history, product relevance, service quality, inventory availability, remaining time and verified terms. Start with interpretable ranking or regularized logistic conversion model ($C=1$), then compare gradient boosting; evaluate incremental impact with a randomized agency-level holdout, eligibility logs, intent-to-treat conversion, sell-through by deadline and confidence intervals. Raw click rates are biased by exposure and placement. Platform listings publish only inventory and terms backed by supplier confirmation. Use feedback for recalibration, supplier reliability, policy learning and marketing. Persistent stores cover working task context, versioned episodic sessions and institutional outcomes with as-of access controls.

## Synthetic data and training contracts

| Dataset | Main grain and fields |
|---|---|
| Events and venues | Event/version with announced time, observed time, service window, category, location, attendance estimate, origin mix, team/artist tags and source. |
| Demand and interactions | Booking date × service date × zone × segment × canonical SKU with price exposure, eligible searches, gross bookings, failed booking reasons, cancellations and sellout intervals. |
| Supplier snapshots and offers | Supplier, canonical product, listing and pool IDs, timestamp, quote expiry, capacity, cost, commission, authorized sell price, tax and FX, inclusions, minimum, release deadline, penalties and reliability history. |
| Product and package catalog | Canonical ID, attributes, listing mappings, component quantities and service offsets, supplier lineage and image/content provenance. |
| Decisions and ledger | Session/report/candidate IDs, policy version, observed feature snapshot, approval/expiry, idempotency key, holds, invoices, refunds, cash dates, settlement, unmet units and profit. |
| Skill inputs, feedback and routing labels | Scenario/version, observable context, useful workflow labels, skill input/output versions, evidence utility, typed objections and synthetic preference pairs. |
| Distribution | Agency cohort, eligibility, impressions, assignment, clicks, bookings and sell-through with source time. |
| Simulator truth | Latent demand and attendance, shared pools, shocks, true substitutions, preference drivers and alternate outcomes. Isolated from policy and live feature views. |

Generate event and destination baselines first. Draw latent visitors and segments, travel choices, party size, travel dates, and disjoint standalone versus bundle intent. Add shared event shocks and seasonality, supplier pricing responsive to forecasts, finite inventory and release contracts. Generate searches with bot/noise/abandonment, then transact via chronological stock and price exposure, cancellations and replenishment. Separately simulate supply outages, delayed updates, marketing assignment and outcome reactions. Keep a second demand generator or shifted choice/elasticity mechanism reserved for stress evaluation. A starting dataset of 24 months × 10 cities × 4 product families is roughly 29,000 aggregated rows before service date, lead-time and segment expansion. Start with 150–300 synthetic events and enough hard supplier negatives to estimate auto-merge precision. Store timestamped snapshots in Parquet, use DuckDB for replay and SQLite for a local transactional demo. A single process with persisted jobs and read-only MCP adapters is sufficient initially.

Each row records effective time and first available time. Filter on both relative to the decision cutoff as appropriate. Group all snapshots and copies of the same event before temporal splitting. Hold out entire events, at least one region, and a generator regime. Fit preprocessing, score thresholds, analog indexes, reconstruction models and calibration on their respective training or past-only windows. Use reconstruction posterior samples available from fold-safe fitting. Simulator ground truth is permitted in supervised labels, rewards and scoring through explicit interfaces. Future outcomes and stress seed identifiers stay out of observations. Evaluate same-seed paired comparisons and report event-clustered uncertainty.

## Baselines, failure checks and build order

| Stage | Deliverable and acceptance check |
|---|---|
| 1. Data and replay | Build timestamped synthetic generator, PS5 normalization, mock MCP reads and session store. Replay a routine day, a material event, a sold-out interval and a shared supplier pool. |
| 2. Forecast and financial spine | Fit a baseline, censored NB and SARIMAX, generate joint draws, reconcile bundles and score firm, wait and flex proposals. Prove cash-flow identities, no duplicate capacity, and pathwise release timing. |
| 3. Research and deliberation | Add routing baselines, train the standalone latent router, add the two skill stages and replay an uncertainty-driven V1 → V2 update. Compare routing utility and MCP call counts with run-all and rules. |
| 4. Strategic policy and execution | Pretrain PPO using the synthetic company policy, compare stochastic and heuristic policies, and demonstrate volatile-market abstention alongside approval, inventory update and release. Verify zero hard constraint violations in replay, concurrency and retry cases. Inject a repeated decision failure and compare the automatically generated candidate policy with the deployed version in held-out replay. |
| 5. Distribution and model experiments | Show agency ranking and conversion measurement design. Benchmark Mamba and GARCH where data and hardware support them. Record seed-level results, calibration and stress failures. |

Track trigger precision/recall, false alerts and detection latency. Measure routing macro-F1 and critical workflow recall, wasted MCP calls and latency. Evaluate match precision/recall and false cluster merges. Score reconstruction versus hidden simulated demand by censoring severity, forecast MAE/RMSE/WAPE with zero-volume handling, pinball loss and interval coverage by horizon/event novelty. Measure analog Recall@k or nDCG using judged relevance, skill output accuracy and unresolved conflicts, profit and CVaR, peak cash, stockouts, unsold room-nights, fulfillment, supplier failure concentration, marketing uplift, regret versus a feasible comparator and stability after policy updates.

Largest failure risks are missing sold-out flags, events present on both sides of the split, future supplier prices used as features, double counting of pools or packages, incorrectly modeled deadline choices and policies exploiting simulator quirks. Test feed outages and lag, duplicate procurement messages, failed model calibration, cancellation shocks, correlated upstream outages and near-deadline approval. Keep a CPU demonstration path for reconstruction, SARIMAX, routing, skills, simulations and PPO. Cap tool calls and report revisions, cache immutable history, and bound large models to an offline experiment. A simple model can win any benchmark.

## Primary walkthrough for the hypothetical Bengaluru cricket final

Use the master brief's synthetic demonstration scenario. At T−60 days, estimated attendance is 50,000, the non-local share is 60%, and the procurement budget is ₹30 lakh. The intent mix is 20% match-only, 35% match plus hotel, 20% match plus hotel and transport, 10% match plus hotel and sightseeing, and 15% complete package. These are mutually exclusive intent segments. Convert the relevant segments into addressable room-nights using the attendance, stay length, party size and market share terms in EDPI. The central agent routes investigation to hotels, transport, nearby activities and compatible packages, then matches suppliers and accounts for stockouts in analogous events.

Illustrative point forecasts are 1,350 room-nights from SARIMAX and 1,650 from Mamba. The arithmetic midpoint of 1,500 demonstrates an equal-weight ensemble, whose actual use depends on validation. With 600 existing room-nights, that implies a point shortage of 900. V1 carries the full distribution, terms, and supporting evidence. The forecast calibration skill checks novelty and uncertainty. The contract comparison skill flags missing optional-allotment terms. Company skills expose how capital preservation and fulfillment rank the same alternatives differently.

With a synthetic purchase cost of ₹3,600 and authorized net resale price of ₹4,000 per night, prepaid contribution is $4000\min(q,D_{\mathrm{residual}})-3600q$. The scenarios below use demand above the existing 600 nights and omit additional fees.

| Incremental demand | Buy 300 | Buy 500 | Buy 700 |
|---:|---:|---:|---:|
| 300 | +₹1.20 lakh | −₹6.00 lakh | −₹13.20 lakh |
| 500 | +₹1.20 lakh | +₹2.00 lakh | −₹5.20 lakh |
| 700 | +₹1.20 lakh | +₹2.00 lakh | +₹2.80 lakh |

The scenario review sends the missing contract terms back to the central agent. V2 adds a synthetic contract with 300 firm nights at ₹10.80 lakh and an option on 300 more at the same unit price. For this walkthrough the option has no separate premium or minimum exercise, expires at T−10 days, and lapses on unexercised inventory without a penalty. On these assumptions maximum exposure is ₹21.60 lakh, leaving ₹8.40 lakh of budget. The simulated policy may select phased procurement after stress testing and send its initial commitment for approval. A second policy action determines whether to exercise the option before expiry using then-current evidence. The demo distributes the 300 firm nights on confirmed terms through the B2B channel, and publishes additional nights when their procurement is confirmed. Record sell-through and final contribution against the report, skill outputs, policy actions and approvals.

In a volatile replay, the models diverge sharply before the option expires and ticket demand falls after a schedule change. The policy can abstain from exercising, allow the optional lot to lapse under the synthetic contract, and continue monitoring and servicing the original 300 confirmed nights. If similar cases repeatedly led the previous policy to exercise and lose money, the adaptation pipeline generates and evaluates a revised policy against both those cases and strong-demand cases where exercising was valuable.

## Supplementary example for the 2027 Cricket World Cup

The ICC lists an India–Pakistan match in Johannesburg on 10 October 2027. The scenario uses a toy book of hotel room-nights around that match, with all prices in illustrative INR-equivalent after assumed FX conversion. [ICC fixture announcement](https://www.icc-cricket.com/news/icc-men-s-cricket-world-cup-2027-schedule-revealed-and-ballot-system-opens)

At T−28 days, simulated search robust z-score reaches 4.1 and the commercial score reaches 0.78. The routing model's illustrative workflow scores activate hotel (0.94), transfer (0.78) and hotel-plus-transfer package (0.67), with flight (0.38) and sightseeing (0.31) below the starting threshold. Matching identifies two listings drawing on the same physical allocation. A comparable event hit its 100-room-night cap, so the reconstruction model incorporates the censored observation as $D\ge100$. In the toy forecast, total demand over the service window has outcomes 100, 160 and 220 at probabilities 25%, 50% and 25%. The 100 currently fulfillable nights leave 0, 60 or 120 unmet. The model counts standalone and package room-nights in the same pool. In the toy calculation all nights are interchangeable inside the specified window.

V1 records uncertainty about overlapping capacity and financial terms. The contract comparison skill flags overlapping capacity, and the liquidity analysis skill requests the release quote. V2 verifies the common pool and models a hold of 80 with a 20-night minimum. At T−7 the early observable signal is weak for the low case and half of high cases. The other cases are strong. For this example, a hold costs ₹12,000 plus ₹3,300 per retained night. A firm buy costs ₹3,000 per night today. Wait produces a firm 40-night offer at ₹3,700 only on strong-signal paths at T−7 in this toy tree. The authorized net selling price is ₹4,800. Assume no other incremental fees, no salvage and no commission on resale. The release action uses the signal at T−7. Weak retains 20, strong retains 80.

| Scenario | Probability | Remaining demand | Signal at T−7 | Firm 60 profit | Flex profit | Wait profit |
|---|---:|---:|---|---:|---:|---:|
| Low | 25% | 0 | Weak | −₹180,000 | −₹78,000 | ₹0 |
| Medium | 50% | 60 | Strong | ₹108,000 | ₹12,000 | ₹44,000 |
| High, early signal | 12.5% | 120 | Strong | ₹108,000 | ₹108,000 | ₹44,000 |
| High, late surge | 12.5% | 120 | Weak | ₹108,000 | ₹18,000 | ₹0 |

| Alternative | Expected profit | CVaR 90% downside | Peak cash | Added expected room-nights fulfilled |
|---|---:|---:|---:|---:|
| Reject | ₹0 | ₹0 | ₹0 | 0 |
| Wait | ₹27,500 | ₹0 | ₹148,000 | 25 |
| Firm 60 | ₹36,000 | ₹180,000 | ₹180,000 | 45 |
| Hold 80 with a 20-night minimum | ₹2,250 | ₹78,000 | ₹276,000 | 42.5 |
| Firm 120 | −₹72,000 | ₹360,000 | ₹360,000 | 60 |

A synthetic ₹300,000 cash limit and ₹100,000 CVaR limit remove both firm-now actions. Wait has better expected profit, while flex supplies 17.5 more expected room-nights. The incremental availability value needed for flex to exceed wait under the simple expected-profit-plus-service objective is ₹25,250 ÷ 17.5 ≈ ₹1,443 per extra room-night. For example, with ₹1,800 per incremental room-night and the stated limits, a trained policy could select the flex action. That walkthrough is an illustrative policy outcome. Supplier failures, softer package conversion and postponement would be separate stress paths with potentially different eligibility.

The executor requests approval for the hold using the exact cost and deadline. For a strong T−7 signal, the next policy action could be retain 80. Validate the live terms and obtain any additional approval required by the hold's authorization scope. Publish confirmed hotel rooms and eligible packages to agencies serving the Johannesburg match market, then measure sell-through and agency-level conversion. In the medium path the system settles 60 incremental nights, 20 retained nights go unsold, and profit is ₹12,000. Record the outcome against the report, contract, approvals and policy version so future training can distinguish forecast error, selection error and supplier performance.
