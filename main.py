"""GTA VI trophy rarity forecast (gta6-trophy-rarity-forecast-01).

Forecasts every PS5 trophy's Exophase earn rate at the case's target as
five quantiles, scored by pinball loss on the logit scale.

Model, per trophy, on the logit scale (logit(p) = ln(p / (100 - p))):
  x = ln(1 + hours since the game's release), y = logit(rate).
  1. Weighted least squares y = a + b*x over the reads in the last 48h
     before the checkpoint (weights halve every 24h). The slope is shrunk
     toward the global (median) slope across trophies, more so when the
     trophy's own history is flat or short -- flat trophies carry no trend
     signal of their own.
  2. The median is a damped extrapolation to the target, averaged in
     logit space between two views of the same dynamics: the line's
     reading of the rate itself, and a counts view (earners over the
     players tracked at the target, each fit separately against
     log-time). The counts view sees dilution explicitly -- when the
     tracked set outgrows the earners, the rate falls even if the
     rate's own line looks flat. The trend keeps running but not at
     full strength, because player progression saturates over a
     4.5-day horizon. The move from the last read is clamped to
     [-2, +3] logit units.
  3. The spread combines a random-walk term that grows with the horizon
     (the house baseline's 0.5 * sqrt(days)), the trophy's own fit
     residual (noisy histories earn wider bands), and a refresh allowance:
     Exophase refreshes its counts on its own schedule, and a refresh can
     move every rate at once (launch night: the tracked set jumped
     638 -> 3,868 players and every rate but the 100% one fell).
  4. Launch-dilution prior: when the history is flat, the game is under
     a day old and the tracked set is still tiny, the next refresh is
     expected to dilute the rates, so the median shifts down a logit
     unit. This only fires in that launch regime; the final case (2.5
     days of history) never triggers it.
  5. The trophy sitting at 100% gets a tight band: it survived the
     launch refresh at 100%. The platinum's quantiles are capped at the
     minimum of the other trophies' quantiles at the same level, because
     the platinum needs every other trophy.

Deterministic: no randomness, no wall clock (every time comes from the
case file), trophies processed in the case's own order. Standard library
only, no network. Runs as `python3 -P -s main.py`; reads
$MAKEFUN_INPUT_DIR (else /input) and writes $MAKEFUN_OUTPUT_DIR
(else /output), one <id>.out per <id>.in.
"""

import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from statistics import NormalDist

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

# How far back the trend fit looks, and how fast old reads fade.
WINDOW_HOURS = 48.0
HALF_LIFE_HOURS = 24.0
# The trend keeps running at this fraction of its fitted slope.
DAMP = 0.75
# Spread pieces, in logit units.
RW_SIGMA_PER_SQRT_DAY = 0.5
REFRESH_SIGMA = 0.5
FLAT_SIGMA = 0.25
# Launch-dilution prior (fires only on flat history, tiny tracked set,
# game under a day old): shift the median down by this many logit units.
LAUNCH_SHIFT = 1.0
LAUNCH_MAX_PLAYERS = 2000
LAUNCH_MAX_AGE_HOURS = 24.0
# Clamps on the median's move from the last read, in logit units.
MOVE_DOWN = 2.0
MOVE_UP = 3.0


def clamp_pct(x):
    return min(99.99, max(0.01, x))


def logit(pct):
    p = clamp_pct(pct) / 100.0
    return math.log(p / (1.0 - p))


def inv_sigmoid(lg):
    return 100.0 / (1.0 + math.exp(-lg))


def hours_since_epoch(ts):
    """ISO-8601 UTC timestamp -> hours since the Unix epoch."""
    return datetime.fromisoformat(ts.replace("Z", "+00:00")).timestamp() / 3600.0


def fit_line(points):
    """Weighted least squares y = a + b*x over (x, y, weight) points.

    Returns (intercept, slope, rmse). A degenerate set (one distinct x)
    reports a flat line through the weighted mean.
    """
    total_w = sum(w for _, _, w in points)
    if total_w <= 0:
        return None
    mean_x = sum(w * x for x, _, w in points) / total_w
    mean_y = sum(w * y for _, y, w in points) / total_w
    sxx = sum(w * (x - mean_x) ** 2 for x, _, w in points)
    if sxx <= 1e-12:
        return (mean_y, 0.0, 0.0)
    sxy = sum(w * (x - mean_x) * (y - mean_y) for x, y, w in points)
    slope = sxy / sxx
    intercept = mean_y - slope * mean_x
    var = sum(w * (y - (intercept + slope * x)) ** 2
              for x, y, w in points) / total_w
    return (intercept, slope, math.sqrt(max(var, 0.0)))


def fallback_rows(trophies):
    """A safe forecast when a case carries no usable history: a wide
    band around 1% for every trophy. Never let a case go invalid."""
    rows = []
    for trophy in trophies:
        rows.append({"id": trophy["id"],
                     "q": [0.05, 0.3, 1.0, 3.0, 15.0]})
    return {"trophies": rows}


def forecast_case(case):
    trophies = case["trophies"]
    history = case.get("history") or []
    if not history:
        return fallback_rows(trophies)

    levels = list(case.get("levels") or [0.05, 0.25, 0.5, 0.75, 0.95])
    z_values = [NormalDist().inv_cdf(tau) for tau in levels]

    release_h = hours_since_epoch(case["game"]["release_date"] + "T00:00:00Z")
    at_h = hours_since_epoch(case["at"])
    target_h = hours_since_epoch(case["target"])
    horizon_days = case["horizon_minutes"] / 1440.0
    x_target = math.log1p(max(target_h - release_h, 0.0))

    # The trend window: reads in the last 48h, else everything there is.
    recent = [h for h in history
              if at_h - hours_since_epoch(h["at"]) <= WINDOW_HOURS]
    if len(recent) < 3:
        recent = history
    recent_h = [hours_since_epoch(h["at"]) for h in recent]

    # One weighted line fit per trophy, plus its last non-null rate.
    fits = []
    last_rates = []
    for i in range(len(trophies)):
        points = []
        for read, read_h in zip(recent, recent_h):
            rate = read["rates"][i]
            if rate is None:
                continue
            x = math.log1p(max(read_h - release_h, 0.0))
            weight = 0.5 ** ((at_h - read_h) / HALF_LIFE_HOURS)
            points.append((x, logit(rate), weight))
        last_rate = None
        for read in reversed(history):
            if read["rates"][i] is not None:
                last_rate = read["rates"][i]
                break
        last_rates.append(0.5 if last_rate is None else last_rate)
        if len(points) >= 2:
            fits.append(fit_line(points))
        else:
            fits.append((logit(last_rates[-1]), 0.0, 0.0))

    # The global slope: the median of the trophies that show any slope.
    slopes = sorted(f[1] for f in fits if abs(f[1]) > 1e-9)
    global_slope = slopes[len(slopes) // 2] if slopes else 0.0

    # A second view of the same dynamics, in counts: how fast the
    # tracked set grows, and how fast each trophy's earners grow.
    # log N and log E are each fit against ln(1 + hours) over the same
    # window; their ratio at the target is a second median for the
    # rate, one that sees dilution (N outgrowing E) explicitly. When
    # the two views agree it changes nothing; when they disagree it
    # hedges the line fit's reading of refresh steps.
    n_points = []
    for read, read_h in zip(recent, recent_h):
        players = read.get("players_tracked") or 0
        if players > 0:
            x = math.log1p(max(read_h - release_h, 0.0))
            weight = 0.5 ** ((at_h - read_h) / HALF_LIFE_HOURS)
            n_points.append((x, math.log(players), weight))
    n_fit = fit_line(n_points) if len(n_points) >= 2 else None

    e_fits = []
    for i in range(len(trophies)):
        points = []
        for read, read_h in zip(recent, recent_h):
            rate = read["rates"][i]
            players = read.get("players_tracked") or 0
            if rate is None or players <= 0:
                continue
            earners = max(rate * players / 100.0, 0.05)
            x = math.log1p(max(read_h - release_h, 0.0))
            weight = 0.5 ** ((at_h - read_h) / HALF_LIFE_HOURS)
            points.append((x, math.log(earners), weight))
        e_fits.append(fit_line(points) if len(points) >= 2 else None)
    e_slopes = sorted(f[1] for f in e_fits
                      if f is not None and abs(f[1]) > 1e-9)
    e_global = e_slopes[len(e_slopes) // 2] if e_slopes else 0.0

    # Launch regime: flat history, tiny tracked set, young game. The
    # next Exophase refresh is expected to dilute the rates.
    players_last = history[-1].get("players_tracked") or 0
    age_hours = at_h - release_h
    launch = (players_last <= LAUNCH_MAX_PLAYERS
              and age_hours <= LAUNCH_MAX_AGE_HOURS
              and all(abs(f[1]) < 1e-9 for f in fits))

    rows = []
    for i, trophy in enumerate(trophies):
        intercept, slope, rmse = fits[i]
        y_last = logit(last_rates[i])

        # How many distinct rates this trophy showed lately: the fewer,
        # the more its slope is pulled toward the global one.
        seen = set()
        for read in recent:
            rate = read["rates"][i]
            if rate is not None:
                seen.add(rate)
        shrink = len(seen) / (len(seen) + 4.0)
        slope = shrink * slope + (1.0 - shrink) * global_slope

        # The line's median, averaged in logit space with the counts
        # view (earners over tracked players at the target).
        y_line = intercept + slope * x_target
        y_mid = y_line
        if n_fit is not None and e_fits[i] is not None:
            e_intercept, e_slope, _ = e_fits[i]
            e_slope = shrink * e_slope + (1.0 - shrink) * e_global
            log_e = e_intercept + e_slope * x_target
            log_n = n_fit[0] + n_fit[1] * x_target
            # Earners can never exceed the players tracked.
            log_e = min(log_e, log_n + math.log(0.999))
            rate_counts = 100.0 * math.exp(log_e - log_n)
            y_mid = 0.5 * (y_line + logit(rate_counts))

        y_med = y_last + DAMP * (y_mid - y_last)
        y_med = min(max(y_med, y_last - MOVE_DOWN), y_last + MOVE_UP)
        if launch:
            y_med -= LAUNCH_SHIFT

        # Spread: random walk + own residual + refresh allowance, with
        # an extra piece when the trophy never moved in the window.
        rw = RW_SIGMA_PER_SQRT_DAY * math.sqrt(horizon_days)
        resid = min(max(rmse, 0.10), 0.80)
        sigma = math.sqrt(rw * rw + resid * resid
                          + REFRESH_SIGMA * REFRESH_SIGMA)
        if len(seen) <= 1:
            sigma = math.sqrt(sigma * sigma + FLAT_SIGMA * FLAT_SIGMA)

        if last_rates[i] >= 99.99:
            # Survived the launch refresh at 100%: keep it tight.
            y_med = logit(99.99)
            sigma = 0.20

        quantiles = [inv_sigmoid(y_med + z * sigma) for z in z_values]
        quantiles = [min(max(q, 0.01), 99.99) for q in quantiles]
        for k in range(1, len(quantiles)):
            quantiles[k] = max(quantiles[k], quantiles[k - 1])
        rows.append({"id": trophy["id"],
                     "q": [round(q, 4) for q in quantiles]})

    # The platinum needs every other trophy, so it can never sit above
    # the lowest of them at any level.
    platinum = next((r for r in rows if r["id"] == 1), None)
    if platinum is not None:
        others = [r for r in rows if r["id"] != 1]
        if others:
            for k in range(len(levels)):
                cap = min(r["q"][k] for r in others)
                if platinum["q"][k] > cap:
                    platinum["q"][k] = cap
            for k in range(1, len(levels)):
                platinum["q"][k] = max(platinum["q"][k],
                                       platinum["q"][k - 1])
    return {"trophies": rows}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # Every case file, in name order, so two runs write the same bytes.
    for path in sorted(INPUT_DIR.glob("*.in")):
        try:
            case = json.loads(path.read_text(encoding="utf-8"))
            forecast = forecast_case(case)
        except Exception:
            # A broken case file must still produce a valid forecast.
            try:
                trophies = json.loads(
                    path.read_text(encoding="utf-8"))["trophies"]
            except Exception:
                trophies = []
            forecast = fallback_rows(trophies)
        (OUTPUT_DIR / (path.stem + ".out")).write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n",
            encoding="utf-8")


if __name__ == "__main__":
    main()
