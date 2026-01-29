# Acceptance Criteria

Measurable KPIs for UrbanNav qualification. All metrics evaluated over
the qualification DOE (Design of Experiments) scenario suite.

## Position Accuracy

| Metric | Threshold | Environment |
|--------|-----------|-------------|
| Horizontal RMSE | ≤ 2.0 m | Hallway, no sources active |
| Horizontal RMSE | ≤ 3.0 m | Multi-source (elevator + vehicles) |
| Vertical RMSE | ≤ 0.5 m | Within-floor (no floor transition) |
| Floor detection accuracy | ≥ 95% | Elevator ride, stairwell |
| Max horizontal error (99th %ile) | ≤ 8.0 m | All scenarios |

## Source Detection and Tracking

| Metric | Threshold | Notes |
|--------|-----------|-------|
| Elevator detection latency | ≤ 30 s | Time from SNR > 3 to TRACK_CONFIRMED |
| Vehicle detection latency | ≤ 45 s | Time from SNR > 3 to TRACK_CONFIRMED |
| Source position RMSE | ≤ 3.0 m | Confirmed sources only |
| False positive rate | ≤ 5% | Fraction of ghost sources promoted |
| Source retirement latency | ≤ 60 s | Time from source departure to TRACK_RETIRED |

## Map Learning

| Metric | Threshold | Notes |
|--------|-----------|-------|
| Background map RMSE | ≤ 50 nT | After convergence, source-free regions |
| Map convergence (50% error reduction) | ≤ 5 minutes | From cold start |
| Background not poisoned | Δ(tile_coef) < 2σ | Near active sources |
| Teachability rate | ≥ 60% | Fraction of measurements used for learning |

## Safety and Robustness

| Metric | Threshold | Notes |
|--------|-----------|-------|
| NEES consistency | 0.5 ≤ mean NEES ≤ 2.0 | Position states, d=3 |
| Nav corruption from bad source | RMSE increase ≤ 50% | vs no-source baseline |
| Rollback success rate | 100% | All triggered rollbacks restore clean state |
| No divergence | 0 events | Position error must remain bounded |

## Timing

| Metric | Threshold | Notes |
|--------|-----------|-------|
| Fast loop (10 Hz) | ≤ 50 ms | Measurement processing |
| Slow loop (1 Hz) | ≤ 500 ms | Map and source updates |
| Timing overrun rate | ≤ 1% | Across all scenarios |

## Qualification DOE Coverage

The qualification must exercise:
- **Trajectories**: Hallway patrol, elevator ride, spiral ramp, lobby crossing
- **Environments**: Office building, parking garage, lobby/atrium
- **Source configurations**: No sources, single elevator, multiple vehicles, mixed
- **Sensor degradation**: IMU bias drift, odometry dropout, barometer fault
- **Seeds**: Minimum 10 random seeds per scenario for statistical significance
