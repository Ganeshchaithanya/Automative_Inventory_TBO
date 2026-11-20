"""Synthetic observations and censored demand forecasting."""

from dataclasses import dataclass, replace

import numpy as np
from scipy.optimize import minimize
from scipy.stats import nbinom


BASE_STOCK = 600
SELL_PRICE = 4_000
UNIT_COST = 3_600
DISPERSION = 38
TRIGGER_THRESHOLD = 3.5


@dataclass(frozen=True)
class Observation:
    event: str
    day: int
    searches: int
    baseline_searches: tuple[int, ...]
    signal: float
    volatile: bool
    quote_verified: bool


@dataclass(frozen=True)
class Episode:
    opening: Observation
    deadline: Observation
    demand: int


def episodes(seed: int, count: int, *, shock_mode: bool = False) -> list[Episode]:
    """Keep realized demand separate from the two available-at-decision snapshots."""
    rng = np.random.default_rng(seed)
    result = []
    for index in range(count):
        signal = float(rng.uniform(-0.2, 1.15))
        volatile = bool(rng.random() < 0.28)
        early = float(signal + rng.normal(0, 0.30 if volatile else 0.17))
        baseline = tuple(int(value) for value in rng.poisson(250, size=28))
        event = f"replay-{seed}-{index:04d}"
        opening = Observation(
            event, -60, int(300 + 160 * max(early, 0)), baseline, early,
            volatile, bool(rng.random() > 0.05),
        )
        deadline = Observation(
            event, -10, int(300 + 160 * max(signal, 0)), baseline, signal,
            volatile, True,
        )
        mean = 900 * float(np.exp(0.72 * signal))
        if volatile and rng.random() < (0.72 if shock_mode else 0.45):
            mean *= 0.50
        demand = int(rng.negative_binomial(DISPERSION, DISPERSION / (DISPERSION + mean)))
        result.append(Episode(opening, deadline, demand))
    return result


def bengaluru(*, volatile: bool) -> Episode:
    baseline = (224, 245, 232, 255, 261, 248, 243, 237, 255, 260, 246, 237)
    event = "bengaluru-volatile" if volatile else "bengaluru-base"
    opening = Observation(event, -60, 495, baseline, 0.78, False, False)
    deadline = Observation(
        event, -10, 330 if volatile else 505, baseline, 0.35 if volatile else 0.88,
        volatile, True,
    )
    return Episode(opening, deadline, 680 if volatile else 1_465)


def trigger_score(observation: Observation) -> float:
    baseline = np.asarray(observation.baseline_searches)
    center = np.median(baseline)
    mad = np.median(np.abs(baseline - center))
    return float(0.67449 * (observation.searches - center) / max(mad, 10))


def trigger(observation: Observation) -> dict:
    score = trigger_score(observation)
    return {"robust_z": round(score, 2), "active": score >= TRIGGER_THRESHOLD}


@dataclass(frozen=True)
class ForecastCalibration:
    offset: float
    slope_adjustment: float
    residuals: tuple[float, ...]


def forecast_group(observation: Observation) -> str:
    stage = "opening" if observation.day < -10 else "deadline"
    regime = "volatile" if observation.volatile else "stable"
    return f"{stage}_{regime}"


@dataclass(frozen=True)
class DemandModel:
    intercept: float
    slope: float
    dispersion: int
    calibration: dict[str, ForecastCalibration] | None = None

    def scenarios(self, observation: Observation, *, count: int = 512) -> np.ndarray:
        """Use a shared quantile grid so candidate comparisons share demand paths."""
        quantiles = (np.arange(count) + 0.5) / count
        log_mean = self.intercept + self.slope * observation.signal
        if self.calibration is not None:
            group = self.calibration[forecast_group(observation)]
            residuals = group.residuals
            # Split-calibration ranks use n+1 rather than rescaling from the sample extrema.
            ranks = np.arange(1, len(residuals) + 1) / (len(residuals) + 1)
            log_draws = log_mean + group.offset + group.slope_adjustment * observation.signal
            log_draws = log_draws + np.interp(quantiles, ranks, residuals)
            return np.rint(np.exp(log_draws)).astype(int)

        mean = float(np.exp(log_mean))
        draws = nbinom.ppf(quantiles, self.dispersion, self.dispersion / (self.dispersion + mean))
        if observation.volatile:
            # Synthetic stress branch for postponement/cancellation and demand shocks.
            draws[: count // 3] *= 0.56
        return draws.astype(int)


def censored_samples(seed: int, count: int) -> tuple[np.ndarray, ...]:
    rng = np.random.default_rng(seed)
    signal = rng.uniform(-0.35, 1.20, count)
    mean = np.exp(np.log(900) + 0.72 * signal)
    demand = rng.negative_binomial(DISPERSION, DISPERSION / (DISPERSION + mean))
    capacity = rng.choice([650, 900, 1_150, 2_000], count)
    bookings = np.minimum(demand, capacity)
    return signal, demand, capacity, bookings, demand >= capacity


def fit_demand(seed: int = 91, count: int = 360) -> tuple[DemandModel, dict]:
    signal, _, capacity, bookings, censored = censored_samples(seed, count)

    def loss(parameters: np.ndarray) -> float:
        mu = np.exp(parameters[0] + signal * parameters[1])
        probability = DISPERSION / (DISPERSION + mu)
        exact = nbinom.logpmf(bookings[~censored], DISPERSION, probability[~censored])
        tail = nbinom.logsf(capacity[censored] - 1, DISPERSION, probability[censored])
        return float(-(exact.sum() + tail.sum()))

    fit = minimize(loss, [np.log(900), 0.5], bounds=[(5, 9), (-2, 2)], method="L-BFGS-B")
    if not fit.success:
        raise RuntimeError(f"Censored demand fit failed: {fit.message}")
    model = DemandModel(float(fit.x[0]), float(fit.x[1]), DISPERSION)
    return model, {
        "training_events": count,
        "stockout_events": int(censored.sum()),
        "observed_booking_mean": round(float(bookings.mean()), 1),
        "fitted_demand_mean": round(float(np.exp(model.intercept + model.slope * signal).mean()), 1),
        "method": "negative binomial with right-censored stockouts",
    }


def fit_calibration(model: DemandModel, training: list[Episode], calibration: list[Episode]) -> DemandModel:
    """Fit stage/regime-dependent bias and reserve independent outcomes for interval calibration."""
    if {episode.opening.event for episode in training} & {episode.opening.event for episode in calibration}:
        raise ValueError("Forecast correction and calibration must use separate events.")

    def grouped_residuals(data: list[Episode]) -> dict[str, np.ndarray]:
        groups: dict[str, list[tuple[float, float]]] = {}
        for episode in data:
            for observation in (episode.opening, episode.deadline):
                signal = observation.signal
                log_mean = model.intercept + model.slope * signal
                residual = float(np.log(max(1, episode.demand)) - log_mean)
                groups.setdefault(forecast_group(observation), []).append((signal, residual))
        return {key: np.asarray(rows) for key, rows in groups.items()}

    train_groups = grouped_residuals(training)
    calibration_groups = grouped_residuals(calibration)
    groups = {}
    for key in ("opening_stable", "opening_volatile", "deadline_stable", "deadline_volatile"):
        training_rows = train_groups.get(key, ())
        calibration_rows = calibration_groups.get(key, ())
        if len(training_rows) < 100 or len(calibration_rows) < 100:
            raise ValueError(f"Calibration requires at least 100 independent observations for {key} per split.")
        training_signal, training_residual = training_rows.T
        fit = np.linalg.lstsq(
            np.column_stack([np.ones(len(training_signal)), training_signal]), training_residual, rcond=None,
        )[0]
        signal, residual = calibration_rows.T
        corrected_residual = residual - fit[0] - fit[1] * signal
        groups[key] = ForecastCalibration(
            float(fit[0]), float(fit[1]), tuple(float(value) for value in np.sort(corrected_residual)),
        )
    return replace(model, calibration=groups)


def forecast_benchmark(model: DemandModel, data: list[Episode]) -> dict:
    """Score the same forecast bands that the website presents, split by lead time and regime."""
    groups: dict[str, list[tuple]] = {}
    for episode in data:
        for stage, observation in (("opening", episode.opening), ("deadline", episode.deadline)):
            lower, median, upper = np.rint(np.quantile(model.scenarios(observation, count=256), [0.1, 0.5, 0.9]))
            row = (episode.demand, lower, median, upper)
            groups.setdefault(stage, []).append(row)
            groups.setdefault(forecast_group(observation), []).append(row)
    result = {}
    for key, rows in groups.items():
        actual, low, median, high = np.asarray(rows).T
        error = median - actual
        result[key] = {
            "events": len(rows),
            "mae": round(float(np.mean(np.abs(error))), 1),
            "rmse": round(float(np.sqrt(np.mean(error ** 2))), 1),
            "wape_pct": round(float(100 * np.sum(np.abs(error)) / actual.sum()), 1) if actual.sum() else None,
            "bias": round(float(np.mean(error)), 1),
            "coverage_pct": round(float(100 * np.mean((actual >= low) & (actual <= high))), 1),
            "mean_width": round(float(np.mean(high - low)), 1),
            "pinball": {
                f"p{int(level * 100)}": round(float(np.mean(np.maximum(
                    level * (actual - prediction), (level - 1) * (actual - prediction),
                ))), 1)
                for level, prediction in ((0.1, low), (0.5, median), (0.9, high))
            },
        }
    return result
