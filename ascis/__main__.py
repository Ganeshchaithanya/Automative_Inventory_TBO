"""Commands for training, replaying and adapting the local ASCIS prototype."""

import argparse
from dataclasses import replace
import json
from pathlib import Path
import sqlite3

from ascis.execution import MockExecutor, proposal
from ascis.intelligence import Action, CompanyPolicy, Position, evaluate, settlement
from ascis.market import SELL_PRICE, bengaluru, episodes, fit_calibration, fit_demand, forecast_benchmark, trigger
from ascis.policy import adaptation_candidate, benchmark, load, save, train


DEFAULT_CHECKPOINT = Path("runs/policy.pt")


def demo(checkpoint: Path, *, volatile: bool, approve: bool) -> dict:
    policy, demand, company, training = load(checkpoint)
    episode = bengaluru(volatile=volatile)
    triggered = trigger(episode.opening)
    result = {
        "mode": "synthetic local replay", "trigger": triggered, "event": episode.opening.event,
        "company_policy": company.version, "checkpoint": str(checkpoint), "training": training,
        "reports": [], "decisions": [], "commercial_outcome": None, "distribution": [],
    }
    if not triggered["active"]:
        return result
    position = Position()
    with sqlite3.connect(":memory:") as connection:
        executor = MockExecutor(connection)
        report_version = 1
        observation = episode.opening
        for _ in range(2):
            analysis = evaluate(observation, position, demand, company, report_version=report_version)
            result["reports"].append(analysis.report)
            if not observation.quote_verified:
                result["reanalysis"] = {
                    "reason": "Contract comparison requires verified terms and allocation.",
                    "refreshed": ["supplier quote", "financial exposure by candidate", "company skill findings"],
                }
                observation = replace(observation, quote_verified=True)
                report_version += 1
                analysis = evaluate(observation, position, demand, company, report_version=report_version)
                result["reports"].append(analysis.report)
            action = policy.choose(analysis)
            candidate = analysis.candidates[action]
            decision = {"day": observation.day, "report_version": report_version, "action": action.name}
            result["decisions"].append(decision)
            if candidate.units or action == Action.RELEASE_OPTION:
                requested = proposal(analysis, action, company)
                decision["proposal"] = requested
                if not approve:
                    decision["status"] = "awaiting mock approval"
                    return result
                receipt = executor.commit(requested, requested["id"], observation, demand, company)
                decision["status"] = "approved and executed in mock ledger"
                position = Position(**receipt["position"])
                if candidate.units:
                    result["distribution"].append({
                        "audience": "B2B agencies serving Bengaluru cricket travel",
                        "confirmed_room_nights": candidate.units,
                        "authorized_price_inr": SELL_PRICE,
                        "listing_status": "mock listing ready",
                    })
            elif action == Action.ABSTAIN:
                decision["status"] = "no new inventory commitment"
                decision["decision_context"] = {
                    "volatile_market": observation.volatile,
                    "forecast_room_nights": analysis.report["forecast_room_nights"],
                    "excluded_commitments": {
                        item["action"]: item["excluded_by"]
                        for item in analysis.report["counterfactuals"]
                        if item["peak_cash"] and item["excluded_by"]
                    },
                }
            else:
                decision["status"] = "no inventory transaction"
            if action in (Action.WAIT, Action.HOLD_300_PLUS_300):
                observation = episode.deadline
                report_version += 1
            else:
                break
        if position.option_open:
            result["option_outcome"] = "unexercised option lapses at T−10 under the synthetic contract"
            position = replace(position, option_open=False)
        result["commercial_outcome"] = settlement(position, episode.demand)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m ascis")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    commands = parser.add_subparsers(dest="command", required=True)
    training = commands.add_parser("train", help="pretrain under a synthetic company policy")
    training.add_argument("--updates", type=int, default=150)
    training.add_argument("--reference", type=Path, help="optional incumbent checkpoint scored on the same final cohorts")
    replay = commands.add_parser("demo", help="replay the hypothetical Bengaluru event")
    replay.add_argument("--volatile", action="store_true")
    replay.add_argument("--approve", action="store_true", help="simulate approval of the exact proposals")
    replay.add_argument("--report", type=Path, help="write the complete strategic report and decision trace")
    adapt = commands.add_parser("adapt", help="generate and compare a candidate policy after recurring losses")
    adapt.add_argument("--updates", type=int, default=55)
    adapt.add_argument("--output", type=Path, default=Path("runs/policy-candidate.pt"))
    args = parser.parse_args()
    if args.command == "train":
        baseline, fit = fit_demand()
        correction_cases = episodes(27103, 4000) + episodes(27104, 2000, shock_mode=True)
        calibration_cases = episodes(27105, 3000) + episodes(27106, 1500, shock_mode=True)
        demand = fit_calibration(baseline, correction_cases, calibration_cases)
        fit["calibration"] = {
            "groups": "opening and deadline, split by stable and volatile markets",
            "bias_fit_events": len(correction_cases), "calibration_events": len(calibration_cases),
            "nominal_coverage_pct": 80,
        }
        company = CompanyPolicy(downside_penalty=0.75, unsold_inventory_penalty=120)
        training_episodes = episodes(105, 1600) + episodes(27110, 1200, shock_mode=True)
        validation = {"normal": episodes(27111, 320), "stress": episodes(27112, 320, shock_mode=True)}
        contenders = []
        for seed in (42, 73):
            policy, report = train(training_episodes, demand, company, seed=seed, updates=args.updates)
            scores = {name: benchmark(data, policy, demand, company) for name, data in validation.items()}
            # Account for lost sales so a checkpoint cannot win by simply buying less.
            score = min(
                item["mean_contribution"] - 0.35 * item["loss_cvar90"]
                - company.availability_value * item["mean_unmet_room_nights"]
                for item in scores.values()
            )
            contenders.append((score, policy, {**report, "seed": seed, "validation": scores}))
        _, model, report = max(contenders, key=lambda contender: contender[0])
        holdout = episodes(106, 120)
        scenarios = {"normal": episodes(27107, 1000), "stress": episodes(27108, 1000, shock_mode=True)}
        reference = load(args.reference) if args.reference else None
        policy_evaluation = {}
        for name, data in scenarios.items():
            policy_evaluation[name] = {
                "ppo": benchmark(data, model, demand, company),
                "company_rules": benchmark(data, None, demand, company),
            }
            if reference:
                old_policy, old_demand, old_company, _ = reference
                policy_evaluation[name]["reference"] = benchmark(data, old_policy, old_demand, old_company)
        result = {
            "fit": fit, "training": {**report, "training_events": len(training_episodes)},
            "held_out_ppo": benchmark(holdout, model, demand, company),
            "held_out_company_heuristic": benchmark(holdout, None, demand, company),
            "forecast_evaluation": {
                name: {"baseline": forecast_benchmark(baseline, data), "calibrated": forecast_benchmark(demand, data)}
                for name, data in scenarios.items()
            },
            "policy_evaluation": policy_evaluation,
        }
        save(args.checkpoint, model, demand, company, result)
    elif args.command == "demo":
        if not args.checkpoint.is_file():
            parser.error("Run 'python -m ascis train' to create the local policy checkpoint.")
        result = demo(args.checkpoint, volatile=args.volatile, approve=args.approve)
        if args.report:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        result = {key: value for key, value in result.items() if key not in ("reports", "training")}
    else:
        if not args.checkpoint.is_file():
            parser.error("Run 'python -m ascis train' to create the local policy checkpoint.")
        if args.output.resolve() == args.checkpoint.resolve():
            parser.error("Save an adaptation candidate to a different path.")
        model, demand, company, prior = load(args.checkpoint)
        candidate, result = adaptation_candidate(model, demand, company, args.updates)
        if candidate is not None:
            demand_metadata = {key: prior[key] for key in ("fit", "forecast_evaluation") if key in prior}
            save(args.output, candidate, demand, company, {**demand_metadata, **result})
            result["candidate_checkpoint"] = str(args.output)
    print(json.dumps(result, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
