"""Reproducible synthetic checks for the stages surrounding inventory decisions."""

import argparse
from dataclasses import replace
import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit

from ascis.intelligence import contract_comparison, route_workflow, supplier_quotes
from ascis.market import (
    TRIGGER_THRESHOLD, Observation, censored_samples, episodes, fit_demand, forecast_benchmark, trigger_score,
)
from ascis.policy import load
from ascis.evidence import supplier_evidence
from ascis.intelligence import Action, Position
from ascis.web import assess_decision


def reconstruction_benchmark() -> dict:
    train_signal, _, _, bookings, _ = censored_samples(91, 360)
    design = np.column_stack([np.ones(len(train_signal)), train_signal])
    booking_curve = np.linalg.lstsq(design, np.log(bookings), rcond=None)[0]
    model, _ = fit_demand()

    signal, demand, _, _, sold_out = censored_samples(29001, 2000)
    estimates = {
        "bookings_only": np.exp(booking_curve[0] + booking_curve[1] * signal),
        "censored_fit": np.exp(model.intercept + model.slope * signal),
    }
    return {
        key: {
            "events": int(mask.sum()),
            **{
                method: {
                    "mae": round(float(np.mean(np.abs(prediction[mask] - demand[mask]))), 1),
                    "bias": round(float(np.mean(prediction[mask] - demand[mask])), 1),
                }
                for method, prediction in estimates.items()
            },
        }
        for key, mask in (("all", np.ones(len(demand), dtype=bool)), ("sold_out", sold_out))
    }


def monitor_samples(seed: int, count: int, *, noisy: bool = False) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    features = []
    labels = []
    for index in range(count):
        center = rng.uniform(100, 1200)
        noise = rng.uniform(0.04, 0.22 if noisy else 0.14)
        baseline = tuple(np.maximum(0, rng.normal(center, center * noise, 28)).astype(int))
        surge = rng.random() < 0.30
        disruption = not surge and rng.random() < (0.22 if noisy else 0.08)
        uplift = rng.uniform(0.16, 0.75) if surge or disruption else 0
        searches = max(0, int(center * (1 + uplift + rng.normal(0, noise))))
        evidence = rng.normal(0.55 if surge else 0.10, 0.34 if noisy else 0.25)
        observation = Observation(str(index), -60, searches, baseline, float(evidence), noisy, True)
        features.append((1, np.clip(trigger_score(observation), -5, 12) / 4, evidence))
        labels.append(surge)
    return np.asarray(features), np.asarray(labels)


def alert_metrics(labels: np.ndarray, active: np.ndarray) -> dict:
    tp = int(np.count_nonzero(labels & active))
    fp = int(np.count_nonzero(~labels & active))
    missed = int(np.count_nonzero(labels & ~active))
    return {
        "precision_pct": round(100 * tp / max(1, tp + fp), 1),
        "recall_pct": round(100 * tp / max(1, tp + missed), 1),
        "f1_pct": round(200 * tp / max(1, 2 * tp + fp + missed), 1),
        "false_alerts": fp,
        "missed_surges": missed,
    }


def monitor_benchmark() -> dict:
    training = (monitor_samples(29002, 2000), monitor_samples(29003, 2000, noisy=True))
    features = np.concatenate([group[0] for group in training])
    labels = np.concatenate([group[1] for group in training])

    def loss(weights: np.ndarray) -> float:
        logits = features @ weights
        return float(np.mean(np.logaddexp(0, logits) - labels * logits) + 0.005 * np.sum(weights[1:] ** 2))

    fit = minimize(loss, np.zeros(features.shape[1]), method="BFGS")
    if not fit.success:
        raise RuntimeError(f"Monitor fitting failed: {fit.message}")
    validation = (monitor_samples(29004, 800), monitor_samples(29005, 800, noisy=True))
    threshold = max(
        np.arange(0.20, 0.81, 0.025),
        key=lambda threshold: min(
            alert_metrics(actual, expit(values @ fit.x) >= threshold)["f1_pct"]
            for values, actual in validation
        ),
    )
    results = {"challenger_threshold": round(float(threshold), 3), "training_events": len(labels), "holdout": {}}
    for name, data in (
        ("normal", monitor_samples(29006, 1200)),
        ("noisy", monitor_samples(29007, 1200, noisy=True)),
    ):
        values, actual = data
        results["holdout"][name] = {
            "events": len(actual),
            "production_robust_z": alert_metrics(actual, values[:, 1] * 4 >= TRIGGER_THRESHOLD),
            "experimental_logistic": alert_metrics(actual, expit(values @ fit.x) >= threshold),
        }
    return results


def evidence_benchmark(model, demand, company) -> dict:
    results = {}
    for name, regime in (("normal", False), ("stress", True)):
        samples = episodes(29220 + int(regime), 250, shock_mode=regime)
        decisions = [
            inputs
            for episode in samples
            for inputs in (
                (replace(episode.opening, quote_verified=True), Position()),
                (episode.deadline, Position(300, 1_080_000, True)),
            )
        ]
        suite = {}
        for state in ("current", "missing", "stale", "conflicting"):
            commitments = skipped = 0
            for observation, position in decisions:
                evidence = supplier_evidence(observation.day, observation.quote_verified, state)
                analysis, action, actor_ran, _ = assess_decision(model, demand, company, observation, position, evidence)
                commitments += int(bool(analysis.candidates[action].units) or action == Action.RELEASE_OPTION)
                skipped += int(not actor_ran)
            suite[state] = {"decisions": len(decisions), "inventory_actions": commitments, "policy_calls_skipped": skipped}
        results[name] = suite
    return results


def run(checkpoint: Path) -> dict:
    policy, model, company, training = load(checkpoint)
    baseline, _ = fit_demand()
    quotes = supplier_quotes(-60)
    conflict = (quotes[0], replace(quotes[1], capacity=quotes[1].capacity + 100), quotes[2])
    return {
        "mode": "synthetic replay and deterministic verification",
        "reconstruction": reconstruction_benchmark(),
        "forecast": {
            name: {"baseline": forecast_benchmark(baseline, cases), "calibrated": forecast_benchmark(model, cases)}
            for name, cases in (
                ("normal", episodes(27107, 1000)),
                ("stress", episodes(27108, 1000, shock_mode=True)),
            )
        },
        "monitor": monitor_benchmark(),
        "evidence_faults": evidence_benchmark(policy, model, company),
        "routing": {offer: route_workflow(offer) for offer in ("hotel", "package")},
        "supplier": {
            "matching_declared_pool_capacity": contract_comparison(quotes, True)["pool_capacity"],
            "conflicting_pools_requiring_review": contract_comparison(conflict, True)["conflicting_pools"],
        },
        "policy": training.get("policy_evaluation", {}),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=Path("runs/policy.pt"))
    args = parser.parse_args()
    print(json.dumps(run(args.checkpoint), indent=2))
