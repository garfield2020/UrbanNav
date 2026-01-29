# Elevator DOE Report

## Executive Summary

- **Total runs**: 216
- **Pass**: 210 (97.2%)
- **Fail**: 6

## Mode Comparison (A vs B vs C)

| Mode | Runs | Mean RMSE (m) | Mean P90 (m) | Max P90 (m) | Mean DNH Ratio | Pass Rate |
|------|------|--------------|-------------|-------------|----------------|-----------|
| A (Baseline) | 72 | 13.624 | 17.018 | 73.603 | 1.04 | 100.0% |

| B (Robust Ignore) | 72 | 41.881 | 67.607 | 190.464 | 0.996 | 100.0% |

| C (Source-Aware) | 72 | 4.915 | 7.816 | 20.999 | 0.991 | 91.7% |


## Do-No-Harm Compliance (≤ 1.10)

- Mode B violations: 0 / 72
- Mode C violations: 6 / 72

## Mode C Near-Shaft Benefit vs Mode B

- Mode B near-shaft P90: 60.456 m
- Mode C near-shaft P90: 7.121 m
- Reduction: 88.2% (gate ≥20%: **PASS**)

## Worst Case Failures (Top 10)

| Run | Mode | Archetype | Approach | Speed | P90 (m) | DNH | Reasons |
|-----|------|-----------|----------|-------|---------|-----|---------|
| 135 | C | stop_and_go | APPROACH_MEDIUM | ELEV_SLOW | 20.999 | 1.245 | Do-no-harm: 1.245 > 1.10 |
| 198 | C | dual_shaft | APPROACH_NEAR | ELEV_FAST | 15.754 | 1.658 | Do-no-harm: 1.658 > 1.10 |
| 196 | C | dual_shaft | APPROACH_NEAR | ELEV_FAST | 15.711 | 1.683 | Do-no-harm: 1.683 > 1.10 |
| 125 | C | stop_and_go | APPROACH_NEAR | ELEV_FAST | 12.646 | 1.196 | Do-no-harm: 1.196 > 1.10 |
| 142 | C | stop_and_go | APPROACH_MEDIUM | ELEV_FAST | 9.635 | 1.144 | Do-no-harm: 1.144 > 1.10 |
| 117 | C | stop_and_go | APPROACH_NEAR | ELEV_SLOW | 8.981 | 1.246 | Do-no-harm: 1.246 > 1.10 |

## Acceptance Criteria Summary

| Gate | Requirement | Status |
|------|------------|--------|
| Do-no-harm (Mode B) | P90 ratio ≤ 1.10 | PASS |
| Mode C benefit | ≥20% P90 reduction near shaft | PASS |
