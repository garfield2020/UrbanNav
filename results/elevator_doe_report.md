# Elevator DOE Report

## Executive Summary

- **Total runs**: 216
- **Pass**: 216 (100.0%)
- **Fail**: 0

## Mode Comparison (A vs B vs C)

| Mode | Runs | Mean RMSE (m) | Mean P90 (m) | Max P90 (m) | Mean DNH Ratio | Pass Rate |
|------|------|--------------|-------------|-------------|----------------|-----------|
| A (Baseline) | 72 | 0.431 | 0.681 | 1.262 | 0.967 | 100.0% |

| B (Robust Ignore) | 72 | 0.624 | 1.01 | 2.209 | 0.983 | 100.0% |

| C (Source-Aware) | 72 | 0.505 | 0.794 | 1.452 | 0.964 | 100.0% |


## Do-No-Harm Compliance (≤ 1.10)

- Mode B violations: 0 / 72
- Mode C violations: 0 / 72

## Mode C Near-Shaft Benefit vs Mode B

- Mode B near-shaft P90: 0.987 m
- Mode C near-shaft P90: 0.768 m
- Reduction: 22.1% (gate ≥20%: **PASS**)

## Acceptance Criteria Summary

| Gate | Requirement | Status |
|------|------------|--------|
| Do-no-harm (Mode B) | P90 ratio ≤ 1.10 | PASS |
| Mode C benefit | ≥20% P90 reduction near shaft | PASS |
