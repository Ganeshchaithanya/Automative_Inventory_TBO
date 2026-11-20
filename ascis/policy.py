"""Company-policy conditioning, behavior cloning and two-step masked PPO."""

from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import nn
from torch.distributions import Categorical

from ascis.intelligence import (
    Action, Analysis, CompanyPolicy, Position, apply_position, evaluate, settlement,
)
from ascis.market import BASE_STOCK, DemandModel, Episode, ForecastCalibration, episodes, trigger


GAMMA = 0.99
ACTION_COUNT = len(Action)


class ActorCritic(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(width, 128), nn.Tanh(), nn.Linear(128, 128), nn.Tanh())
        self.actor = nn.Linear(128, ACTION_COUNT)
        self.critic = nn.Linear(128, 1)

    def forward(self, features: torch.Tensor, mask: torch.Tensor):
        state = self.encoder(features)
        logits = self.actor(state).masked_fill(~mask, -1e9)
        return Categorical(logits=logits), self.critic(state).squeeze(-1)

    @torch.no_grad()
    def choose(self, analysis: Analysis) -> Action:
        distribution, _ = self(torch.tensor(analysis.features), torch.tensor(analysis.mask))
        return Action(int(distribution.logits.argmax()))

    @torch.no_grad()
    def action_weights(self, analysis: Analysis) -> dict[str, float]:
        distribution, _ = self(torch.tensor(analysis.features), torch.tensor(analysis.mask))
        return {action.name: float(distribution.probs[action]) for action in Action if analysis.mask[action]}


@dataclass
class MarketBatch:
    x0: torch.Tensor
    mask0: torch.Tensor
    next_x: torch.Tensor
    next_mask: torch.Tensor
    continues: torch.Tensor
    immediate: torch.Tensor
    next_reward: torch.Tensor
    teacher0: torch.Tensor
    next_teacher: torch.Tensor


def company_example(analysis: Analysis, company: CompanyPolicy) -> int:
    def utility(action: Action, assessment: dict) -> float:
        if assessment["excluded_by"]:
            return float("-inf")
        if action == Action.ABSTAIN:
            return 0.0
        if action == Action.WAIT:
            return -5_000.0 if analysis.features[1] else 4_000.0
        if action in (Action.REJECT, Action.RELEASE_OPTION, Action.REQUEST_TERMS):
            return -10_000.0
        return (
            assessment["expected_profit"]
            + company.availability_value * assessment["expected_fulfilled"]
            - company.capital_weight * assessment["peak_cash"]
            - 0.5 * assessment["downside_cvar90"]
            + (45_000 if action == Action.HOLD_300_PLUS_300 else 0)
        )

    return max(
        range(ACTION_COUNT),
        key=lambda index: utility(Action(index), analysis.report["counterfactuals"][index]),
    )


def outcome_reward(result: dict, demand: int, company: CompanyPolicy) -> float:
    baseline_gap = max(0, demand - BASE_STOCK)
    service_gain = baseline_gap - result["unmet_room_nights"]
    contribution = result["realized_contribution"]
    downside = company.downside_penalty * max(0, -contribution)
    unsold = company.unsold_inventory_penalty * result["unsold_procured_nights"]
    return (contribution + service_gain * company.availability_value - downside - unsold) / 300_000


def prepare_batch(data: list[Episode], demand: DemandModel, company: CompanyPolicy) -> MarketBatch:
    x0, mask0, next_x, next_mask, continues, immediate, next_reward = ([] for _ in range(7))
    teacher0, next_teacher = [], []
    for episode in data:
        initial = evaluate(episode.opening, Position(), demand, company, scenarios=96)
        positions = {
            action: apply_position(Position(), initial.candidates[action])
            for action in (Action.WAIT, Action.HOLD_300_PLUS_300)
        }
        continuations = {
            action: evaluate(episode.deadline, position, demand, company, scenarios=96)
            for action, position in positions.items()
        }
        x0.append(initial.features)
        mask0.append(initial.mask)
        teacher0.append(company_example(initial, company))
        per_x, per_mask, per_cont, per_immediate, per_reward, per_teacher = ([] for _ in range(6))
        fallback = continuations[Action.WAIT]
        for candidate in initial.candidates:
            action = candidate.action
            future = continuations.get(action, fallback)
            position = apply_position(Position(), candidate)
            charge = company.capital_weight * candidate.cost / 300_000
            active = action in continuations and initial.mask[action]
            per_x.append(future.features)
            per_mask.append(future.mask)
            per_cont.append(active)
            per_immediate.append(-charge if active else (
                outcome_reward(settlement(position, episode.demand), episode.demand, company) - charge
            ))
            per_reward.append([
                outcome_reward(settlement(apply_position(position, later), episode.demand), episode.demand, company)
                - company.capital_weight * later.cost / 300_000
                for later in future.candidates
            ])
            per_teacher.append(company_example(future, company))
        next_x.append(per_x)
        next_mask.append(per_mask)
        continues.append(per_cont)
        immediate.append(per_immediate)
        next_reward.append(per_reward)
        next_teacher.append(per_teacher)
    return MarketBatch(*(torch.tensor(value) for value in (
        x0, mask0, next_x, next_mask, continues, immediate, next_reward, teacher0, next_teacher,
    )))


def train(
    data: list[Episode], demand: DemandModel, company: CompanyPolicy, *, seed: int, updates: int,
    previous: ActorCritic | None = None,
) -> tuple[ActorCritic, dict]:
    if not data or updates < 1:
        raise ValueError("Training requires data and at least one PPO update.")
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    batch = prepare_batch(data, demand, company)
    policy = ActorCritic(batch.x0.shape[1])
    optimizer = torch.optim.Adam(policy.parameters(), lr=3e-4)
    if previous is None:
        demo_x = torch.cat([batch.x0, batch.next_x[batch.continues]])
        demo_masks = torch.cat([batch.mask0, batch.next_mask[batch.continues]])
        demo_actions = torch.cat([batch.teacher0, batch.next_teacher[batch.continues]])
        for _ in range(90):
            ids = torch.randint(len(demo_x), (128,))
            distribution, _ = policy(demo_x[ids], demo_masks[ids])
            loss = -distribution.log_prob(demo_actions[ids]).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    else:
        policy.load_state_dict(previous.state_dict())

    steps = 0
    batch_return = 0.0
    for _ in range(updates):
        ids = torch.randint(len(data), (256,))
        start_x, start_mask = batch.x0[ids], batch.mask0[ids]
        with torch.no_grad():
            dist0, value0 = policy(start_x, start_mask)
            action0 = dist0.sample()
            active = batch.continues[ids, action0]
            later_ids, earlier_actions = ids[active], action0[active]
            end_x = batch.next_x[later_ids, earlier_actions]
            end_mask = batch.next_mask[later_ids, earlier_actions]
            if len(end_x):
                dist1, value1 = policy(end_x, end_mask)
                action1 = dist1.sample()
                log_prob1 = dist1.log_prob(action1)
            else:
                value1 = value0.new_empty(0)
                action1 = action0.new_empty(0)
                log_prob1 = value0.new_empty(0)
            reward1 = batch.next_reward[later_ids, earlier_actions, action1]
            reward0 = batch.immediate[ids, action0]
            target0 = reward0.clone()
            target0[active] += GAMMA * reward1
            advantage0 = reward0 - value0
            advantage0[active] += GAMMA * (value1 + 0.95 * (reward1 - value1))
            advantages = torch.cat([advantage0, reward1 - value1])
            advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-6)
            actions = torch.cat([action0, action1])
            old_log_prob = torch.cat([dist0.log_prob(action0), log_prob1])
            targets = torch.cat([target0, reward1])
            inputs = torch.cat([start_x, end_x])
            masks = torch.cat([start_mask, end_mask])
            steps += len(actions)
            batch_return = float(target0.mean())

        for _ in range(5):
            for indices in torch.randperm(len(actions)).split(128):
                distribution, values = policy(inputs[indices], masks[indices])
                ratio = torch.exp(distribution.log_prob(actions[indices]) - old_log_prob[indices])
                advantage = advantages[indices]
                objective = torch.minimum(ratio * advantage, ratio.clamp(0.8, 1.2) * advantage)
                value_loss = (values - targets[indices]).square().mean()
                loss = -objective.mean() + 0.5 * value_loss - 0.01 * distribution.entropy().mean()
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
                optimizer.step()
    policy.eval()
    return policy, {"ppo_updates": updates, "training_transitions": steps, "last_batch_mean_return": round(batch_return, 4)}


def replay_episode(episode: Episode, model: ActorCritic | None, demand: DemandModel, company: CompanyPolicy) -> dict:
    opening = evaluate(episode.opening, Position(), demand, company)
    action = Action.ABSTAIN
    if trigger(episode.opening)["active"]:
        action = model.choose(opening) if model else Action(company_example(opening, company))
    position = apply_position(Position(), opening.candidates[action])
    masked_violations = int(not opening.mask[action])
    later = None
    if action in (Action.HOLD_300_PLUS_300, Action.WAIT):
        review = evaluate(episode.deadline, position, demand, company)
        later = model.choose(review) if model else Action(company_example(review, company))
        masked_violations += int(not review.mask[later])
        position = apply_position(position, review.candidates[later])
    return {
        **settlement(position, episode.demand),
        "opening_action": action.name,
        "deadline_action": later.name if later is not None else None,
        "volatile": episode.deadline.volatile,
        "masked_violations": masked_violations,
    }


def benchmark(data: list[Episode], model: ActorCritic | None, demand: DemandModel, company: CompanyPolicy) -> dict:
    rows = [replay_episode(episode, model, demand, company) for episode in data]
    profits = sorted(row["realized_contribution"] for row in rows)
    tail = profits[:max(1, (len(rows) + 9) // 10)]
    purchased = sum(row["procured_room_nights"] for row in rows)
    sold = sum(row["incremental_sale_nights"] for row in rows)
    stockouts = [row for row in rows if row["realized_demand"] > BASE_STOCK]
    incremental_demand = sum(row["realized_demand"] - BASE_STOCK for row in stockouts)
    allocations = [row for row in rows if row["procured_room_nights"] > 0]
    successful = [
        row for row in allocations if row["realized_contribution"] >= 0
        and 10 * row["incremental_sale_nights"] >= 9 * row["procured_room_nights"]
    ]
    return {
        "episodes": len(rows),
        "mean_contribution": round(sum(profits) / len(rows)),
        "mean_unmet_room_nights": round(sum(row["unmet_room_nights"] for row in rows) / len(rows), 1),
        "loss_cvar90": round(sum(max(0, -profit) for profit in tail) / len(tail)),
        "volatile_losses": sum(row["volatile"] and row["realized_contribution"] < 0 for row in rows),
        "abstentions": sum(row["opening_action"] == "ABSTAIN" or row["deadline_action"] == "ABSTAIN" for row in rows),
        "purchased_room_nights": purchased,
        "sold_procured_room_nights": sold,
        "procured_sell_through_pct": round(100 * sold / purchased, 1) if purchased else None,
        "stockout_demand_served_pct": round(100 * sold / incremental_demand, 1) if incremental_demand else None,
        "allocation_episodes": len(allocations),
        "successful_allocation_pct": round(100 * len(successful) / len(allocations), 1) if allocations else None,
        "masked_violations": sum(row["masked_violations"] for row in rows),
    }


def save(path: Path, model: ActorCritic, demand: DemandModel, company: CompanyPolicy, metadata: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save({
        "schema": 1, "width": model.encoder[0].in_features, "weights": model.state_dict(),
        "demand": asdict(demand), "company": asdict(company), "metadata": metadata,
    }, temporary)
    temporary.replace(path)


def load(path: Path) -> tuple[ActorCritic, DemandModel, CompanyPolicy, dict]:
    saved = torch.load(path, map_location="cpu", weights_only=True)
    if saved["schema"] != 1:
        raise ValueError("Unsupported ASCIS checkpoint schema.")
    model = ActorCritic(saved["width"])
    model.load_state_dict(saved["weights"])
    model.eval()
    demand_config = saved["demand"]
    if demand_config.get("calibration") is not None:
        demand_config = {
            **demand_config,
            "calibration": {
                key: ForecastCalibration(**group) for key, group in demand_config["calibration"].items()
            },
        }
    return model, DemandModel(**demand_config), CompanyPolicy(**saved["company"]), saved["metadata"]


def adaptation_candidate(model: ActorCritic, demand: DemandModel, company: CompanyPolicy, updates: int):
    cases = episodes(4_110, 240, shock_mode=True)
    replay = [(episode, replay_episode(episode, model, demand, company)) for episode in cases]
    failures = [
        episode for episode, result in replay
        if result["volatile"] and result["realized_contribution"] < 0
    ]
    if len(failures) < 3:
        return None, {"failure_cases": len(failures), "status": "insufficient recurrence"}
    proposed, training = train(
        cases + failures * 3, demand, company, seed=73, updates=updates, previous=model,
    )
    holdout = episodes(4_112, 120, shock_mode=True)
    before = benchmark(holdout, model, demand, company)
    after = benchmark(holdout, proposed, demand, company)
    return proposed, {
        "status": "candidate only",
        "failure_cases": len(failures), "training": training,
        "held_out_baseline": before, "held_out_candidate": after,
        "improved_stress_check": (
            after["volatile_losses"] < before["volatile_losses"]
            and after["loss_cvar90"] <= before["loss_cvar90"]
            and after["mean_unmet_room_nights"] <= before["mean_unmet_room_nights"] + 15
            and after["mean_contribution"] >= before["mean_contribution"] - 10_000
        ),
    }
