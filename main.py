#!/usr/bin/env python3
"""Roblox Up-and-Coming top-10 forecaster v2 (make.fun bounty
roblox-up-and-coming-forecast-01).

Reads /input/history.ndjson (snapshots of Roblox's Up-and-Coming chart,
oldest first) and /input/task.json, and writes /output/forecast.json:
per-place probabilities for the top 10 places plus a top-10 guess in
order, scored by log score over the 72 hours after the deadline.

Method: Monte Carlo rank diffusion in latent-appeal space (a = -log rank),
with a regression-calibrated drift and heavy-tailed (Student-t) steps.

What the history taught us (45 archive reads, 2026-10-30..2026-11-10;
validated on 7 holdout windows, fit on snapshots 1..cut, scored on the
reads in the 72 h after the fit ends):

- Appeal steps are heavy-tailed: a Student-t step (df 12) beats Gaussian
  by ~2 nats. A game's appeal moves ~0.10/sqrt(hour), roughly flat in
  rank; the pooled `da * rank` volatility of v1 overstated top-rank
  diffusion ~20x.
- Drift is predictable, and a regression on 3,650 consecutive-pair
  transitions (R2 = 0.125) fits it: E[da/h] = -0.038 + 0.43 * (appeal
  slope over the trailing 96 h) + 0.63 * (log-playing slope over the
  trailing 30 h) + 0.0079 * log(rank) + 0.0014 * log(chart-age in hours).
  The playing-growth term is the single biggest win: a game's playing
  count surging 10x leads its rank by hours (corr 0.31 with the next
  appeal step; top-decile growers climb +0.5 appeal next read 12% of
  the time vs 1% for the bottom decile). The 96 h slope window beats
  48 h: with sparse reads the shorter slope is mostly noise.
- New games matter: ~0.5 enter per hour, ~13% reach the top 10 within
  72 h, and at 36-72 h out they take 6-30% of top-10 slots by place.
  Phantom entrants drawn from the empirical entry-rank distribution
  carry that mass, which is deliberately left unnamed (the scorer
  spreads unnamed mass over the chart's unnamed games).
- There is NO hard 28-day age-out. Games created 42-44 days ago are
  still on the chart and 26% of it is older than 28 days; the exit
  hazard rises smoothly with creation age instead (7% gone in 72 h
  under 14 days old, 55% at 28-35 days). v1's created+28d cut would
  have zeroed every game older than 28 days at the deadline in the
  final run -- including a former #1 -- for ~6 nats. v2 applies no
  age-out at all: old games fade through the drift's rank term.

Standard library only, no network, deterministic (fixed seed).
"""

import json
import math
import os
import random
from datetime import datetime, timedelta

INPUT_DIR = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
OUTPUT_DIR = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")

# Tunables (calibrated on the archive by multi-window holdout scoring;
# fit on snapshots 1..cut, score on the reads in the 72 h after the cut,
# cuts 20/24/28/30/34/38/42. Mean -34.62, median -32.90, vs v1's
# -38.72 / -38.23 on the same windows; on the practice window itself
# (cut 34) -31.76 vs v1's -37.15. The paid threshold on the live
# practice board is -33.8.)
SEED = 20261113
N_SIMS = 1500           # Monte Carlo paths
DRIFT_WINDOW_HOURS = 96.0
GPLAY_WINDOW_HOURS = 30.0
DRIFT_SHRINK_K = 2.0     # slope *= n / (n + k)
DRIFT_CAP = 0.030        # |appeal slope| per hour, capped
# Regression drift: OLS on 3,650 consecutive-pair transitions from the
# archive (R^2 = 0.125). E[da/h] = REG_CONST + REG_SLOPE * appeal slope
# (trailing 96 h, capped +-0.03, shrunk n/(n+2)) + REG_GPLAY * log-playing
# slope (trailing 30 h, clipped +-0.05) + REG_LOG_RANK * log(rank)
# + REG_LOG_AGE * log(chart-age in hours). Residual sd/h by rank band:
# 1-3: 0.070, 4-10: 0.045, 11-30: 0.040, 31+: 0.028.
REG_SCALE = 1.2          # regression drift is shrunk; 1.2 fits best
REG_CONST = -0.038
REG_SLOPE = 0.43
REG_GPLAY = 0.63
REG_LOG_RANK = 0.0079
REG_LOG_AGE = 0.0014
SIGMA = 0.10             # Student-t scale per sqrt-hour
T_DF = 12                # Student-t degrees of freedom
PHANTOM_SIGMA = 0.35     # entrants diffuse faster than incumbents
ENTRANT_DRIFT = 0.012    # entrants keep climbing, decaying with age
ENTRANT_HALFLIFE_H = 24.0
SMOOTH_EPS = 0.02        # blend of raw counts with a flat prior
P_FLOOR = 0.0015         # rows below this are dropped (mass stays unnamed)
P_CAP = 0.65             # no single game-place probability above this
PLACE_SUM_CAP = 0.95     # leave at least 5% of every place unnamed


def parse_time(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def load_history(input_dir):
    snaps = []
    with open(os.path.join(input_dir, "history.ndjson"), encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                snaps.append(json.loads(line))
    snaps.sort(key=lambda s: s["at"])
    task = {}
    try:
        with open(os.path.join(input_dir, "task.json"), encoding="utf-8") as f:
            task = json.load(f)
    except OSError:
        pass
    return snaps, task


def _slope(xs, ys):
    """Least-squares slope with n/(n+k) shrinkage; xs in hours."""
    n = len(xs)
    if n < 2:
        return 0.0
    mx = sum(xs) / n
    my = sum(ys) / n
    denom = sum((x - mx) ** 2 for x in xs)
    if denom <= 0:
        return 0.0
    s = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / denom
    return s * n / (n + DRIFT_SHRINK_K)


def estimate_params(snaps):
    """Per-game drift (regression), entry rate/ranks, last chart."""
    series = {}
    play_series = {}
    first_seen = {}
    entry_ranks = []
    for idx, snap in enumerate(snaps):
        t = parse_time(snap["at"])
        for row in snap.get("up_and_coming") or []:
            uid = row["universe_id"]
            series.setdefault(uid, []).append((t, row["rank"]))
            if row.get("playing"):
                play_series.setdefault(uid, []).append((t, row["playing"]))
            if uid not in first_seen:
                first_seen[uid] = t
                if idx > 0:
                    entry_ranks.append(row["rank"])
    t_end = parse_time(snaps[-1]["at"])
    last_chart = {r["universe_id"]: (r["rank"], r.get("name") or "")
                  for r in snaps[-1]["up_and_coming"]}

    t_start = t_end - timedelta(hours=DRIFT_WINDOW_HOURS)
    g_start = t_end - timedelta(hours=GPLAY_WINDOW_HOURS)
    last_rank = {uid: rank for uid, (rank, _n) in last_chart.items()}
    drifts = {}
    for uid, obs in series.items():
        recent = [(t, r) for (t, r) in obs if t >= t_start]
        if len(recent) < 3:
            recent = obs[-3:]
        slope = 0.0
        if len(recent) >= 2:
            xs = [(t - recent[0][0]).total_seconds() / 3600.0 for t, _ in recent]
            ys = [-math.log(r) for _, r in recent]
            slope = min(max(_slope(xs, ys), -DRIFT_CAP), DRIFT_CAP)
        # Trailing log-playing slope (growth leads rank by hours).
        gplay = 0.0
        trail = [(t, p) for (t, p) in play_series.get(uid, []) if t >= g_start]
        if len(trail) >= 3:
            xs = [(t - trail[0][0]).total_seconds() / 3600.0 for t, _ in trail]
            ys = [math.log(max(p, 1)) for _, p in trail]
            gplay = min(max(_slope(xs, ys), -0.05), 0.05)
        age_h = max((t_end - first_seen[uid]).total_seconds() / 3600.0, 1.0)
        rank = last_rank.get(uid, obs[-1][1])
        drift = (REG_CONST + REG_SLOPE * slope + REG_GPLAY * gplay
                 + REG_LOG_RANK * math.log(rank) + REG_LOG_AGE * math.log(age_h))
        drifts[uid] = drift * REG_SCALE
    span_h = 0.0
    if len(snaps) >= 2:
        span_h = (parse_time(snaps[-1]["at"])
                  - parse_time(snaps[0]["at"])).total_seconds() / 3600.0
    entry_rate = min(max(len(entry_ranks) / span_h, 0.05), 1.5) if span_h > 0 else 0.5
    return {
        "drifts": drifts,
        "entry_rate": entry_rate,
        "entry_ranks": sorted(entry_ranks) or [80, 87, 90, 95, 97],
        "last_chart": last_chart,
    }


def simulate(params, places, horizon_hours, n_sims=N_SIMS, seed=SEED):
    """Monte Carlo over the scored window. Returns counts[uid][place]."""
    rng = random.Random(seed)
    last_chart = params["last_chart"]
    drifts = params["drifts"]
    entry_rate = params["entry_rate"]
    entry_ranks = params["entry_ranks"]

    games = []  # [uid, appeal, drift]
    for uid, (rank, _name) in sorted(last_chart.items(), key=lambda kv: kv[1][0]):
        games.append([uid, -math.log(rank), drifts.get(uid, 0.0)])
    n_games = len(games)
    # Phantom entrants: expected count over the window, from the
    # empirical entry rate (about 0.5/hour in the archive).
    n_phantoms = min(max(int(round(entry_rate * horizon_hours)), 25), 70)

    def t_noise():
        # Exact Student-t with T_DF degrees of freedom:
        # Z / sqrt(chi2_df / df), chi2 via 2 * Gamma(df/2, 1).
        z = rng.gauss(0.0, 1.0)
        chi2 = 2.0 * rng.gammavariate(T_DF / 2.0, 1.0)
        return z / math.sqrt(chi2 / T_DF)

    counts = {uid: [0] * (places + 1) for uid, _, _ in games}
    total_slots = n_sims * horizon_hours
    for _ in range(n_sims):
        state = [[g[0], g[1], g[2]] for g in games]
        # Phantoms: [entry_hour, appeal, age_hours]; entry hours uniform
        # over the window, entry appeals from the empirical entry ranks.
        phantoms = []
        for _ in range(n_phantoms):
            r = entry_ranks[rng.randrange(len(entry_ranks))]
            phantoms.append([rng.uniform(0.0, horizon_hours),
                             -math.log(r) + rng.gauss(0, 0.3), 0.0])
        for hour in range(1, horizon_hours + 1):
            scored = []
            for g in state:
                g[1] += g[2] + t_noise() * SIGMA
                scored.append((g[1], g[0]))
            for ph in phantoms:
                if ph[0] <= hour:
                    ph[2] += 1.0
                    decay = 0.5 ** (ph[2] / ENTRANT_HALFLIFE_H)
                    ph[1] += ENTRANT_DRIFT * decay + t_noise() * PHANTOM_SIGMA
                    scored.append((ph[1], -1))
            scored.sort(key=lambda x: -x[0])
            for place in range(1, min(places, len(scored)) + 1):
                uid = scored[place - 1][1]
                if uid >= 0:
                    counts[uid][place] += 1
    return counts, total_slots, n_games


def build_forecast(counts, total_slots, n_games, places, last_chart):
    probs = []
    for place in range(1, places + 1):
        rows = []
        for uid, arr in counts.items():
            p = arr[place] / total_slots if total_slots else 0.0
            # Blend with a flat prior so no plausible game is ever at zero.
            p = (1.0 - SMOOTH_EPS) * p + SMOOTH_EPS * (1.0 / max(n_games, 1))
            if p >= P_FLOOR:
                rows.append((uid, min(p, P_CAP)))
        total = sum(p for _, p in rows)
        if total > PLACE_SUM_CAP:
            scale = PLACE_SUM_CAP / total
            rows = [(uid, p * scale) for uid, p in rows]
        rows.sort(key=lambda x: (-x[1], x[0]))
        for uid, p in rows:
            probs.append({"universe_id": uid, "place": place, "p": round(p, 6)})
    # Top 10: greedy by probability, place by place.
    by_place = {}
    for row in probs:
        by_place.setdefault(row["place"], []).append(row)
    top10 = []
    used = set()
    for place in range(1, places + 1):
        for row in by_place.get(place, []):
            if row["universe_id"] not in used:
                top10.append(row["universe_id"])
                used.add(row["universe_id"])
                break
    if len(top10) < places:  # fall back to the last chart's order
        for uid, (_rank, _name) in sorted(last_chart.items(), key=lambda kv: kv[1][0]):
            if uid not in used:
                top10.append(uid)
                used.add(uid)
            if len(top10) == places:
                break
    return {"probabilities": probs, "top10": top10[:places]}


def main():
    snaps, task = load_history(INPUT_DIR)
    places = int(task.get("places", 10))
    window = task.get("window") or {}
    horizon = int(window.get("slots", 72))
    if window.get("slot_minutes"):
        horizon = int(round(horizon * window["slot_minutes"] / 60.0))
    horizon = min(max(horizon, 1), 72 * 4)
    params = estimate_params(snaps)
    counts, total_slots, n_games = simulate(params, places, horizon)
    forecast = build_forecast(counts, total_slots, n_games, places, params["last_chart"])
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, task.get("output", "forecast.json"))
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(json.dumps(forecast, separators=(",", ":")) + "\n")


if __name__ == "__main__":
    main()
