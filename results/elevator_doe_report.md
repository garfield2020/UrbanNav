# Elevator DOE Report

## Executive Summary

- **Total runs**: 216
- **Pass**: 216 (100.0%)
- **Fail**: 0

## Mode Comparison (A vs B vs C)

| Mode | Runs | Mean RMSE (m) | Mean P90 (m) | Max P90 (m) | Mean DNH Ratio | Pass Rate |
|------|------|--------------|-------------|-------------|----------------|-----------|
| A (Baseline) | 72 | 13.624 | 17.018 | 73.603 | 1.04 | 100.0% |

| B (Robust Ignore) | 72 | 41.881 | 67.607 | 190.464 | 0.996 | 100.0% |

| C (Source-Aware) | 72 | 27.525 | 46.364 | 125.82 | 0.997 | 100.0% |


## Do-No-Harm Compliance (≤ 1.10)

- Mode B violations: 0 / 72
- Mode C violations: 0 / 72

## Mode C Near-Shaft Benefit vs Mode B

- Mode B near-shaft P90: 60.456 m
- Mode C near-shaft P90: 42.23 m
- Reduction: 30.1% (gate ≥20%: **PASS**)

## Acceptance Criteria Summary

| Gate | Requirement | Status |
|------|------------|--------|
| Do-no-harm (Mode B) | P90 ratio ≤ 1.10 | PASS |
| Mode C benefit | ≥20% P90 reduction near shaft | PASS |
