# Roblox Up-and-Coming top-10 forecaster (v2)

Entry for the [make.fun](https://make.fun) bounty
**"Predict Roblox's Up-and-Coming top 10"**
(`roblox-up-and-coming-forecast-01`, $500 prize pool, 20 places).
Iteration of the v1 entry
([makefun-roblox-up-and-coming-forecast-01](https://github.com/Fuzzylucious/makefun-roblox-up-and-coming-forecast-01)),
which scored −37.1 on the practice window.

## What it does

`main.py` reads the chart history at `/input/history.ndjson` (plus
`/input/task.json`) and writes one forecast to
`/output/forecast.json`: for each of the top 10 places, a probability
for each game it names being at that place, and its top-10 guess in
order. The forecast is scored by log score against every hourly read
of Roblox's Up-and-Coming chart in the 72 hours after the deadline
(2026-11-13T06:00:00Z to 2026-11-16T06:00:00Z).

The program runs with no network in the grading sandbox, so
everything it knows comes from the history it is given. It re-reads
that history on every run: nothing about the current chart is
baked in.

## Method

Monte Carlo rank diffusion in latent-appeal space (`a = -log(rank)`),
with a regression-calibrated drift and heavy-tailed steps:

- **Heavy tails.** Appeal steps are Student-t (df 12), not Gaussian.
  A game's appeal moves ~0.10/sqrt(hour), roughly flat in rank.
  v1's pooled `da * rank` volatility overstated top-rank diffusion
  ~20x; the flat scale beats it by ~2 nats.
- **Regression drift.** An OLS fit on 3,650 consecutive-pair
  transitions (R² = 0.125) gives
  `E[da/h] = −0.038 + 0.43·(appeal slope, trailing 96 h)
  + 0.63·(log-playing slope, trailing 30 h)
  + 0.0079·log(rank) + 0.0014·log(chart-age in hours)`,
  scaled by 1.2. The playing-growth term is the biggest single win:
  a game's playing count surging 10x leads its rank by hours
  (corr 0.31 with the next appeal step; top-decile growers climb
  +0.5 appeal next read 12% of the time, vs 1% for the bottom
  decile). The 96 h slope window beats 48 h: with sparse reads the
  shorter slope is mostly noise.
- **New entrants.** ~0.5 games enter per hour, ~13% reach the top 10
  within 72 h, and at 36–72 h out they take 6–30% of top-10 slots
  by place. Phantom entrants drawn from the empirical entry-rank
  distribution carry that mass, which is deliberately left unnamed
  — the scoring rule spreads a place's unnamed share over that
  read's unnamed games, which is exactly the hedge a brand-new
  climber needs.
- **No hard 28-day age-out.** Games created 42–44 days ago are still
  on the chart and 26% of it is older than 28 days; the exit hazard
  rises smoothly with creation age instead (7% gone in 72 h under
  14 days old, 55% at 28–35 days). v1's created+28d cut would have
  zeroed every game older than 28 days at the deadline in the final
  run — including a former #1 — for ~6 nats. v2 applies no age-out:
  old games fade through the drift's rank term.
- 1,500 seeded simulations × 72 hourly slots are counted: how often
  each real game sits at each place. Counts are blended with a small
  flat prior, capped per place, and every place keeps at least 5%
  unnamed. No game is ever staked at p = 1 and no plausible climber
  is left at zero.

## Validation

Holdout scoring with the official rule (canonical scorer from the
kit's harness, verified: it reproduces v1's live practice score
−37.1 on the practice window). Fit on snapshots 1..cut of the full
46-read public archive, score on the reads in the 72 h after the cut:

| cut | fit ends | v2 | v1 | persistence baseline |
|----:|----------|-----|-----|----------------------|
| 20 | 11-04 | −37.83 | — | — |
| 24 | 11-05 | −41.79 | −41.10 | −57.34 |
| 28 | 11-06 | −35.97 | −37.29 | −45.81 |
| 30 | 11-07 | −32.56 | — | — |
| 34 | 11-07 (practice window) | **−31.38** | −37.15 | −43.51 |
| 38 | 11-08 | −31.62 | −39.23 | −38.57 |
| 42 | 11-10 | −32.47 | — | — |

Mean −34.80, median −32.56 over the 7 windows (v1: −38.72 mean on
its 4 windows). The four most recent windows all beat the live
practice board's paid threshold (−33.8) by 1.3–2.4 nats. The two
oldest windows are harder draws (a Ragdoll/Survive surge from deep
ranks in the 11-05 window); the final run fits on ~93 reads
(archive + hourly recorder reads), far more data than any holdout
above.

Two runs on the same history write byte-identical output
(deterministic seed, ~28 s per run).

## Run it locally

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
cat out/forecast.json
```

Standard library only; Python 3.13 in the grading sandbox.

## License

MIT — see [LICENSE](LICENSE).
