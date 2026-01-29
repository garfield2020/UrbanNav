# UrbanNav Navigation System

## Project Overview

Ground-based navigation system for urban environments using magnetic field sensing for GPS-denied localization. The system fuses magnetic anomaly measurements with traditional navigation sensors (IMU, odometry, barometer) using factor graph optimization.

## Critical Quality Standard

**All physics models, algorithms, and implementations must be able to withstand the scrutiny of a skeptical physicist.**

This means:
- Equations must be dimensionally consistent
- Physical constants must use correct SI units with citations
- Approximations must state their validity regimes
- Coordinate frame conventions must be explicit and consistent
- Uncertainty propagation must follow proper statistical principles
- No "magic numbers" - all thresholds must have physical or statistical justification

## Architecture Philosophy

- **Product code** lives in the core navigation package only
- Scenario runners, experimental estimators, and device-specific models stay outside the core package to prevent cross-contamination
- Contracts define authoritative interfaces between subsystems
- All sensor models must be pluggable via a registry pattern

## Key Physics Concepts

### Magnetic Field Measurements
- **d=3 mode**: Field vector B only (3 components)
- **d=8 mode**: Field + gradient tensor (3 + 5 independent components)
  - Gradient tensor is symmetric and traceless (Maxwell): Gzz = -(Gxx + Gyy)
  - 9 components → 5 independent: [Gxx, Gyy, Gxy, Gxz, Gyz]

### Chi-Square Gating
- Whitened residual: γ = r'Σ⁻¹r ~ χ²(d)
- d=3 thresholds: mild=11.345 (p=0.01), strong=16.266 (p=0.001)
- d=8 thresholds: mild=20.09 (p=0.01), strong=26.12 (p=0.001)

### Dipole Model
- Field: B(r) = (μ₀/4π) [3(m·r̂)r̂ - m] / |r|³
- μ₀ = 4π × 10⁻⁷ T·m/A (permeability of free space)

## Repository Structure

```
UrbanNav/
├── claude.md                    # This file - project context for AI assistants
├── docs/
│   ├── ACCEPTANCE.md            # Acceptance test criteria
│   ├── ARCHITECTURE.md          # System architecture overview
│   ├── SYSTEM_CONTRACT.md       # API contracts and guarantees
│   └── NAV_ENVELOPE.yaml        # Operational envelope definition
│
├── nav_core/                    # Core navigation package (PRODUCT CODE)
│   ├── Project.toml
│   ├── Manifest.toml
│   └── src/
│       ├── NavCore.jl           # Main module entry point
│       ├── contracts/           # Authoritative interfaces
│       │   ├── StateContract.jl       # Vehicle state definition
│       │   ├── MeasurementContract.jl # AbstractMeasurement interface
│       │   ├── FactorContract.jl      # Factor graph factor interface
│       │   ├── HealthContract.jl      # Health state machine (HEALTHY/DEGRADED/UNRELIABLE)
│       │   └── TelemetryContract.jl   # Telemetry schemas
│       ├── math/                # Core mathematics
│       │   ├── physics.jl             # Dipole model, μ₀, harmonic basis
│       │   ├── whitening.jl           # Covariance whitening, χ² computation
│       │   ├── sigma_total.jl         # Innovation covariance composition
│       │   ├── rotations.jl           # SO(3) utilities
│       │   └── uncertainty.jl         # Uncertainty propagation
│       ├── graph/               # Factor graph engine
│       │   ├── FactorGraph.jl         # Generic factor graph structure
│       │   ├── optimization.jl        # Gauss-Newton / Levenberg-Marquardt
│       │   ├── conditioning.jl        # Numerical conditioning
│       │   └── factors.jl             # Urban-specific factors (IMU, odometry, etc.)
│       ├── estimation/          # State estimation
│       │   ├── StateEstimator.jl      # Main estimator interface
│       │   ├── residual_manager.jl    # χ² gating, feature lifecycle
│       │   ├── dipole_fitter.jl       # LM dipole parameter estimation
│       │   └── covariance.jl          # Covariance management
│       ├── sensors/             # Sensor models
│       │   ├── SensorRegistry.jl      # Pluggable sensor registry
│       │   └── sensors.jl             # IMU, odometry, barometer models
│       ├── features/            # Feature system
│       │   ├── FeatureRegistry.jl     # Feature type registry
│       │   ├── dipole.jl              # Dipole feature type
│       │   ├── dipole_mle.jl          # Multi-source MLE
│       │   ├── spatial_clustering.jl  # Clustering with pose uncertainty
│       │   ├── feature_lifecycle.jl   # Promotion/demotion logic
│       │   ├── feature_disambiguation.jl  # Association, split, merge
│       │   ├── feature_absorption.jl  # Feature-to-map absorption
│       │   └── feature_marginalization.jl # Schur complement
│       ├── mapping/             # Tile-based mapping
│       │   ├── tile_coefficients.jl   # Harmonic basis with Hessians
│       │   ├── tile_manager.jl        # Spatial indexing
│       │   ├── map_update_policy.jl   # Learn vs freeze decisions
│       │   ├── gradient_integration.jl # d=8 full tensor support
│       │   ├── tensor_selectivity.jl  # SNR-based mode selection
│       │   ├── temporal_coherence.jl  # Detection validation
│       │   └── map_store.jl           # Persistence and versioning
│       ├── health/              # Health monitoring
│       │   ├── health_types.jl        # MonitoredHealthState, HealthReport
│       │   └── HealthMonitor.jl       # Checkers (Covariance, Innovation, etc.)
│       └── io/                  # I/O and infrastructure
│           ├── determinism.jl         # Reproducibility utilities
│           ├── external_interface.jl  # External API
│           ├── timing_budget.jl       # Real-time timing
│           ├── telemetry.jl           # Telemetry publishing
│           └── logging.jl             # Structured logging
│
├── scenarios/                   # Scenario runners (NOT imported by nav_core)
│   ├── scenario.jl              # Scenario framework
│   └── ...
│
├── experiments/                 # Experimental code (NOT imported by nav_core)
│   └── ...
│
└── tests/
    ├── runtests.jl              # Top-level test runner
    ├── unit/                    # Unit tests
    │   ├── runtests.jl
    │   └── ...
    ├── integration/
    │   └── ...
    └── acceptance/
        └── ...
```

## What Stays OUT of nav_core

To prevent cross-contamination:
- Scenario runners and performance harnesses → `scenarios/`
- Alternate/experimental estimator architectures → `scenarios/`
- Device-specific interference models (until calibrated) → `experiments/`
- World generation → `experiments/`

## Coding Conventions

- **Language**: Julia 1.9+
- **Types**: Use StaticArrays for fixed-size vectors (SVector{3}, SMatrix{3,3})
- **Units**: SI throughout (meters, seconds, Tesla, radians)
- **Frames**: Explicit NED (North-East-Down) body and world frames
- **Tests**: Each capability has unit tests with clear test IDs
