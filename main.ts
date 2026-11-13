// polymarket-nvidia-ceo-forecast-01 entry (Node 22, standard library only).
//
// Design (tuned on the 5 scored practice transitions p00->p01 ... p04->p05,
// mean pinball 0.3003 vs the house baseline's 0.3925 on the same transitions):
//   - Persistence centre, plus a cross-market sum drift: the 24 Yes prices are
//     mutually exclusive CEO chances, so their sum should sit near 1. When the
//     sum S of last prices is off 1, each market's centre drifts proportionally
//     to its share, (1 - S) * x / S, ramped linearly to full strength at h=240.
//     (Proportional, not equal, drift: big markets absorb most of the correction.)
//   - Per-market spread from recent realised volatility: sigma_m =
//     clamp(0.0002 + 0.45 * RMS120, 0.00015, 0.0025), where RMS120 is the root
//     mean square of the last <=120 one-minute diffs, each winsorised at 0.02.
//     Prices are mostly flat with rare large jumps, so a market that just moved
//     gets a much wider interval than a dead one; the floor keeps quiet markets
//     from collapsing to a point forecast. Revision (2026-11-13): the old cap
//     0.002 bound on 9/128 practice market-cases and the uncapped sigma reached
//     0.0032, so the cap was truncating genuinely volatile markets; the 0.01
//     winsorisation likewise cut real jump signal. Raising the cap to 0.0025
//     and the winsorisation to 0.02 improved the pooled practice pinball from
//     0.3003 to 0.2981 (see analysis/results.md).
//   - Fat-tailed quantile multipliers, not normal z: [-3.5, -0.75, 0, 0.75, 3.5]
//     times sigma_m * sqrt(h). The outer levels are far wider than a normal
//     (jump risk), the inner levels narrower than the baseline's (most minutes
//     barely move). No drift toward trailing medians/means and no empirical
//     quantiles of past 240-min changes: both were tested and hurt on the
//     practice transitions (past regime drift does not carry forward).
//   - Short histories are handled gracefully: RMS uses however many diffs exist
//     (a market with 19 prices still gets a sigma); a market with no diffs
//     falls back to the 0.0002 floor. No randomness, no network, deterministic.
//
// What an entry must do (from starter/main.ts):
//   - read every <id>.in in the input directory (the grader runs one case at a
//     time, with only that case's file there), canonical JSON
//     {"schema":"makefun/polymarket-case/v1","case","at","horizon_minutes",
//     "quantiles","event","markets":[{"id","name","slug","yes_token","start",
//     "prices"}]}, where `prices` holds one Yes price per minute from `start`
//     to `at` inclusive, so the last one is the price at the checkpoint;
//   - write <id>.out to the output directory: {"forecasts":[{"id","q"}]}, every
//     market of the case exactly once (no other id, no other key), `q` exactly
//     horizon_minutes rows, row h-1 the forecast for minute at+h, each row one
//     finite number in [0,1] per quantile level, non-decreasing; at most 1 MiB.
//
// Outside the grader's container, MAKEFUN_INPUT_DIR and MAKEFUN_OUTPUT_DIR
// stand in for /input and /output:
//   mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out node main.ts

import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const INPUT_DIR = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const OUTPUT_DIR = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

/** The quantile levels the contest scores, in the order each row lists them. */
const QUANTILES = [0.05, 0.25, 0.5, 0.75, 0.95];
/** Fat-tailed quantile multipliers (see header): outer levels far wider than
 *  a normal's, inner levels narrower. */
const Z = [-3.5, -0.75, 0, 0.75, 3.5];
/** Recent-volatility window, in minutes, for the per-market sigma. */
const VOL_WINDOW = 120;
/** Winsorisation cap for one-minute diffs when estimating volatility. */
const DIFF_CAP = 0.02;
/** Per-market sigma = clamp(SIGMA_FLOOR_A + SIGMA_B * RMS120, SIGMA_MIN, SIGMA_MAX). */
const SIGMA_A = 0.0002;
const SIGMA_B = 0.45;
const SIGMA_MIN = 0.00015;
const SIGMA_MAX = 0.0025;

interface CaseMarket {
  id: string;
  prices: number[];
}

interface Case {
  horizon_minutes: number;
  quantiles: number[];
  markets: CaseMarket[];
}

/** x clamped to [0,1] and rounded to 6 decimals (0, never -0). */
const price6 = (x: number): number => Math.round(Math.min(1, Math.max(0, x)) * 1e6) / 1e6 + 0;

/** Root mean square of the last <=VOL_WINDOW one-minute diffs, winsorised. */
function rms120(prices: number[]): number {
  const n = prices.length;
  if (n < 2) return 0;
  const start = Math.max(0, n - 1 - VOL_WINDOW);
  let sumSq = 0;
  let count = 0;
  for (let i = start; i < n - 1; i++) {
    const d = Math.abs((prices[i + 1] as number) - (prices[i] as number));
    const w = d > DIFF_CAP ? DIFF_CAP : d;
    sumSq += w * w;
    count++;
  }
  return count === 0 ? 0 : Math.sqrt(sumSq / count);
}

/** One market's forecast: horizon rows of the five quantiles. */
function forecast(m: CaseMarket, horizon: number, sumDrift: number): number[][] {
  if (m.prices.length === 0) throw new Error(`market ${m.id} has no prices`);
  const x = m.prices[m.prices.length - 1] as number;
  const sigma = Math.min(SIGMA_MAX, Math.max(SIGMA_MIN, SIGMA_A + SIGMA_B * rms120(m.prices)));
  const rows: number[][] = [];
  for (let h = 1; h <= horizon; h++) {
    // Centre drifts toward the sum-to-1 correction, fully applied by h=240.
    const centre = x + sumDrift * Math.min(1, h / 240);
    const spread = sigma * Math.sqrt(h);
    rows.push(Z.map((z) => price6(centre + z * spread)));
  }
  return rows;
}

mkdirSync(OUTPUT_DIR, { recursive: true });
// Every case file, in name order, so two runs write the same bytes.
const inputs = readdirSync(INPUT_DIR).filter((f) => f.endsWith('.in')).sort();
for (const file of inputs) {
  const id = file.slice(0, -'.in'.length);
  const c = JSON.parse(readFileSync(join(INPUT_DIR, file), 'utf8')) as Case;
  if (c.quantiles.length !== QUANTILES.length || c.quantiles.some((t, j) => t !== QUANTILES[j])) throw new Error(`${file} asks for quantiles ${JSON.stringify(c.quantiles)}, not ${JSON.stringify(QUANTILES)}`);
  // Cross-market sum at the checkpoint; each market's drift is proportional
  // to its share of that sum. Guarded: the drift divides by the sum, so a sum
  // far from 1 (a partial or broken universe, e.g. S = 0.05 would give a drift
  // of 19x the price) must not be corrected at all — only a sum in
  // [0.25, 2] is treated as a small mis-pricing to drift back toward 1.
  // Practice sums are 0.81-1.09, so this guard never fires there; it exists
  // for unseen final cases.
  const sumLast = c.markets.reduce((s, m) => s + (m.prices.length === 0 ? 0 : (m.prices[m.prices.length - 1] as number)), 0);
  const sumSane = sumLast >= 0.25 && sumLast <= 2;
  const forecasts = c.markets.map((m) => {
    const x = m.prices.length === 0 ? 0 : (m.prices[m.prices.length - 1] as number);
    const drift = sumSane ? ((1 - sumLast) * x) / sumLast : 0;
    return { id: m.id, q: forecast(m, c.horizon_minutes, drift) };
  });
  writeFileSync(join(OUTPUT_DIR, `${id}.out`), `${JSON.stringify({ forecasts })}\n`);
}
