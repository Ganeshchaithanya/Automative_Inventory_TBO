"""Two learned mixtures of skills for advisory inventory assessment."""

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
from hashlib import sha256

import numpy as np
import torch
from torch import nn

from ascis.intelligence import Action, Analysis, CompanyPolicy, Position, evaluate
from ascis.market import SELL_PRICE, DemandModel, Observation, episodes


METRICS = ("sell_through", "forecast_coverage", "loss_probability", "full_sellout")
WIDTH = 17
HIDDEN = 32
LEARNING_RATES = (0.003, 0.05)
ARCHITECTURES = ("mlp", "residual_cross")


class GatedResidualSkill(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.project = nn.Linear(width, HIDDEN)
        self.transform = nn.Sequential(nn.LayerNorm(HIDDEN), nn.Linear(HIDDEN, 2 * HIDDEN), nn.GLU())
        self.output = nn.Linear(HIDDEN, 2)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        hidden = self.project(inputs)
        return self.output(hidden + self.transform(hidden))


class CrossFeatureSkill(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.crosses = nn.ModuleList(
            nn.Sequential(nn.Linear(width, 8, bias=False), nn.Linear(8, width)) for _ in range(2)
        )
        self.output = nn.Linear(width, 2)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        hidden = inputs
        for cross in self.crosses:
            hidden = hidden + inputs * cross(hidden)
        return self.output(hidden)


class SkillMixture(nn.Module):
    def __init__(self, width: int, architecture: str = "mlp"):
        super().__init__()
        self.architecture = architecture
        self.gate = nn.Linear(width, 2)
        if architecture == "mlp":
            self.skills = nn.ModuleList(
                nn.Sequential(nn.Linear(width, HIDDEN), nn.SiLU(), nn.Linear(HIDDEN, 2)) for _ in range(2)
            )
        elif architecture == "residual_cross":
            self.skills = nn.ModuleList([GatedResidualSkill(width), CrossFeatureSkill(width)])
        else:
            raise ValueError(f"Unknown skill architecture: {architecture}")

    def forward(self, inputs: torch.Tensor, baseline: torch.Tensor) -> torch.Tensor:
        weights = self.gate(inputs).softmax(dim=-1)
        reference = torch.logit(baseline.clamp(0.005, 0.995)).unsqueeze(-2)
        adjustments = torch.stack([skill(inputs) for skill in self.skills], dim=-2).tanh()
        predictions = (reference + adjustments).sigmoid()
        return torch.sum(predictions * weights.unsqueeze(-1), dim=-2)


@dataclass
class DualSkills:
    domain: SkillMixture
    company: SkillMixture
    selected: tuple[bool, ...]
    validation: dict

    @torch.inference_mode()
    def assess(self, observation: Observation, analysis: Analysis) -> list[dict]:
        actions, features, reference = candidate_evidence(observation, analysis)
        estimates = predict(self.domain, self.company, features, reference)
        return [
            {
                "action": action.name,
                "eligible": analysis.mask[action],
                "domain": skill_scores(METRICS[:2], estimate[:2], baseline[:2], self.selected[:2]),
                "company": skill_scores(METRICS[2:], estimate[2:], baseline[2:], self.selected[2:]),
            }
            for action, estimate, baseline in zip(actions, estimates, reference)
        ]


def skill_scores(names, estimates, baselines, selected) -> dict:
    return {
        name: {
            "value": round(float(estimate if use_model else baseline), 4),
            "source": "learned mixture" if use_model else "simulator baseline",
            "challenger": round(float(estimate), 4),
        }
        for name, estimate, baseline, use_model in zip(names, estimates, baselines, selected)
    }


def candidate_evidence(observation: Observation, analysis: Analysis) -> tuple[list[Action], np.ndarray, np.ndarray]:
    report = analysis.report
    forecast = report["forecast_room_nights"]
    stock = report["existing_room_nights"]
    headroom = report["company_skills"][0]["headroom"]
    actions = []
    features = []
    reference = []
    for candidate, current in zip(analysis.candidates, report["counterfactuals"]):
        if not candidate.units or "decision window closed" in current["excluded_by"]:
            continue
        if "no exercisable option" in current["excluded_by"] or "reassess existing option before a separate purchase" in current["excluded_by"]:
            continue
        sale_rate = current["expected_sell_through_pct"] / 100
        features.append((
            observation.signal, float(observation.volatile), float(observation.day == -10),
            float(observation.quote_verified), forecast["p10"] / 2000, forecast["p50"] / 2000, forecast["p90"] / 2000,
            stock / 1500, candidate.units / 600, candidate.optional_units / 600,
            candidate.cost / (candidate.units * SELL_PRICE), candidate.peak_cash / 3_000_000,
            headroom / 3_000_000, current["expected_profit"] / 1_000_000,
            current["downside_cvar90"] / 1_000_000, sale_rate, current["loss_probability"],
        ))
        reference.append((sale_rate, 0.80, current["loss_probability"], current["full_sellout_probability"]))
        actions.append(candidate.action)
    return (actions, np.asarray(features, dtype=np.float32).reshape(-1, WIDTH),
            np.asarray(reference, dtype=np.float32).reshape(-1, len(METRICS)))


def sample_batches(seed: int, count: int, demand: DemandModel, company: CompanyPolicy, *, stress: bool = False):
    inputs, labels, baselines = [], [], []
    for episode in episodes(seed, count, shock_mode=stress):
        for observation, position in ((episode.opening, Position()), (episode.deadline, Position(300, 1_080_000, True))):
            analysis = evaluate(observation, position, demand, company)
            actions, features, baseline = candidate_evidence(observation, analysis)
            band = analysis.report["forecast_room_nights"]
            covered = band["p10"] <= episode.demand <= band["p90"]
            unsupplied = max(0, episode.demand - analysis.report["existing_room_nights"])
            for action in actions:
                candidate = analysis.candidates[action]
                sold = min(candidate.units, unsupplied)
                loss = sold * SELL_PRICE < candidate.cost
                labels.append((sold / candidate.units, covered, loss, sold == candidate.units))
            inputs.extend(features)
            baselines.extend(baseline)
    return tuple(torch.tensor(np.asarray(rows), dtype=torch.float32) for rows in (inputs, labels, baselines))


def fit(model: SkillMixture, inputs: torch.Tensor, labels: torch.Tensor,
        baselines: torch.Tensor, validation: tuple, *, learning_rate: float) -> None:
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.001)
    best = float("inf")
    for _ in range(28):
        model.train()
        for indices in torch.randperm(len(inputs)).split(256):
            baseline = baselines[indices]
            predictions = model(inputs[indices], baseline)
            loss = (nn.functional.binary_cross_entropy(predictions, labels[indices])
                    + 0.1 * (predictions - baseline).square().mean())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            vx, vy, vb = validation
            score = float(nn.functional.binary_cross_entropy(model(vx, vb), vy))
        if score < best:
            best = score
            weights = {name: value.clone() for name, value in model.state_dict().items()}
    model.load_state_dict(weights)


def predict(domain: SkillMixture, company: SkillMixture, features, baselines) -> torch.Tensor:
    inputs = torch.as_tensor(features, dtype=torch.float32)
    reference = torch.as_tensor(baselines, dtype=torch.float32)
    domain_scores = domain(inputs, reference[:, :2])
    company_scores = company(torch.column_stack((inputs, domain_scores)), reference[:, 2:])
    return torch.column_stack((domain_scores, company_scores))


def scores(model: DualSkills, batch: tuple) -> dict:
    inputs, actual, baseline = batch
    with torch.no_grad():
        predicted = predict(model.domain, model.company, inputs, baseline)
    return {
        name: {
            "learned_mse": round(float((predicted[:, index] - actual[:, index]).square().mean()), 7),
            "baseline_mse": round(float((baseline[:, index] - actual[:, index]).square().mean()), 7),
        }
        for index, name in enumerate(METRICS)
    }


def train_candidate(batches: dict, architecture: str, learning_rate: float) -> DualSkills:
    torch.manual_seed(29011)
    domain = SkillMixture(WIDTH, architecture)
    alignment = SkillMixture(WIDTH + 2, architecture)
    domain_x, domain_y, domain_b = concatenate(batches["domain_fit_normal"], batches["domain_fit_stress"])
    validation_x, validation_y, validation_b = concatenate(batches["validate_normal"], batches["validate_stress"])
    fit(domain, domain_x, domain_y[:, :2], domain_b[:, :2],
        (validation_x, validation_y[:, :2], validation_b[:, :2]), learning_rate=learning_rate)
    company_x, company_y, company_b = concatenate(batches["company_fit_normal"], batches["company_fit_stress"])
    with torch.no_grad():
        company_x = torch.column_stack((company_x, domain(company_x, company_b[:, :2])))
        alignment_validation = torch.column_stack((validation_x, domain(validation_x, validation_b[:, :2])))
    fit(alignment, company_x, company_y[:, 2:], company_b[:, 2:],
        (alignment_validation, validation_y[:, 2:], validation_b[:, 2:]), learning_rate=learning_rate)
    return DualSkills(domain, alignment, (False,) * len(METRICS), {
        "architecture": architecture, "learning_rate": learning_rate,
    })


def compare(candidate: dict, reference: dict | None, active: tuple[bool, ...]) -> dict[str, float]:
    """Return the smaller relative gain across normal and stress for each output."""
    gains = {}
    for metric, released in zip(METRICS, active):
        by_cohort = []
        for name, suite in candidate.items():
            baseline = reference[name][metric]["learned_mse"] if released else suite[metric]["baseline_mse"]
            by_cohort.append((baseline - suite[metric]["learned_mse"]) / baseline)
        gains[metric] = min(by_cohort)
    return gains


def train(demand: DemandModel, company: CompanyPolicy, incumbent: DualSkills | None) -> tuple[DualSkills, dict]:
    torch.set_num_threads(1)
    cohorts = ("normal", "stress")
    batches = {
        name: sample_batches(seed, count, demand, company, stress=stress)
        for name, seed, count, stress in (
            ("domain_fit_normal", 29012, 2400, False), ("domain_fit_stress", 29013, 2400, True),
            ("company_fit_normal", 29014, 2400, False), ("company_fit_stress", 29015, 2400, True),
            ("validate_normal", 29140, 1600, False), ("validate_stress", 29141, 1600, True),
            ("test_normal", 29142, 2500, False), ("test_stress", 29143, 2500, True),
        )
    }
    incumbent_active = incumbent.selected if incumbent else (False,) * len(METRICS)
    prior_validation = (
        {name: scores(incumbent, batches[f"validate_{name}"]) for name in cohorts} if incumbent else None
    )
    candidates = []
    for architecture in ARCHITECTURES:
        for rate in LEARNING_RATES:
            candidate = train_candidate(batches, architecture, rate)
            validation = {name: scores(candidate, batches[f"validate_{name}"]) for name in cohorts}
            gains = compare(validation, prior_validation, incumbent_active)
            preserves_active = all(gains[name] > 0 for name, active in zip(METRICS, incumbent_active) if active)
            score = (preserves_active, sum(max(0.0, gain) for gain in gains.values()))
            details = {
                **candidate.validation, "validation": validation, "minimum_validation_gain": gains,
                "better_on_all_active_outputs": preserves_active,
            }
            candidates.append((score, candidate, details))
    _, winner, chosen = max(candidates, key=lambda row: row[0])
    holdout = {name: scores(winner, batches[f"test_{name}"]) for name in cohorts}
    prior_holdout = {name: scores(incumbent, batches[f"test_{name}"]) for name in cohorts} if incumbent else None
    gains = compare(holdout, prior_holdout, incumbent_active)
    released = tuple(
        chosen["minimum_validation_gain"][metric] > 0 and gains[metric] > 0 for metric in METRICS
    )
    publish = any(released) and all(new or not old for new, old in zip(released, incumbent_active))
    results = {
        "mode": "synthetic advisory skills",
        "training_events_per_stage": 4800, "validation_events_per_cohort": 1600, "holdout_events_per_cohort": 2500,
        "settings": {"seed": 29011, "epochs": 28, "batch_size": 256, "weight_decay": 0.001},
        "validation_trials": [details for _, _, details in candidates],
        "validation_winner": winner.validation,
        "holdout": holdout, "incumbent_holdout": prior_holdout,
        "minimum_holdout_gain": gains, "challenger_passed_on": dict(zip(METRICS, released)),
        "promoted": publish,
        "effective_learned_outputs": dict(zip(METRICS, released if publish else incumbent_active)),
    }
    winner.selected = released
    winner.validation = results
    return winner, results


def concatenate(*batches: tuple) -> tuple:
    return tuple(torch.cat(items) for items in zip(*batches))


def checkpoint_hash(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def load_skills(path: Path, policy: Path) -> DualSkills | None:
    if not path.is_file():
        return None
    saved = torch.load(path, map_location="cpu", weights_only=True)
    if saved.get("schema") not in (2, 3) or saved["policy_checkpoint_hash"] != checkpoint_hash(policy):
        return None
    architecture = saved.get("architecture", "mlp")
    domain, company = SkillMixture(WIDTH, architecture), SkillMixture(WIDTH + 2, architecture)
    domain.load_state_dict(saved["domain"])
    company.load_state_dict(saved["company"])
    return DualSkills(domain.eval(), company.eval(), tuple(saved["selected"]), saved["evaluation"])


if __name__ == "__main__":
    from ascis.policy import load

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", type=Path, default=Path("runs/policy.pt"))
    parser.add_argument("--output", type=Path, default=Path("runs/dual-skills.pt"))
    args = parser.parse_args()
    _, demand_model, company_policy, _ = load(args.policy)
    incumbent = load_skills(args.output, args.policy)
    model, report = train(demand_model, company_policy, incumbent)
    if report["promoted"] or incumbent is None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_suffix(".tmp")
        torch.save({
            "schema": 3, "architecture": model.domain.architecture,
            "domain": model.domain.state_dict(), "company": model.company.state_dict(),
            "selected": model.selected, "evaluation": report, "policy_checkpoint_hash": checkpoint_hash(args.policy),
        }, temporary)
        temporary.replace(args.output)
    print(json.dumps(report, indent=2))
