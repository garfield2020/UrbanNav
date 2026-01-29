# System Invariants

Non-negotiable properties that must hold at all times during operation.

## INV-01: Background Must Not Absorb Source Signatures

The tile-based background map must never absorb 1/r³ dipole signatures from
tracked sources. If a source is being tracked by the source estimator, its
predicted field contribution must be subtracted from the measurement before
any tile update occurs.

**Enforcement**: The `MapLearningGate` receives clean residuals only. The
`SourceSeparation` module subtracts all active source predictions before
forwarding to tile learning. Tile updates are frozen within the spatial
exclusion zone of any confirmed source (default 15 m radius).

**Test**: Inject a known dipole at fixed position. After 100 steps of tile
learning with the source tracked, verify that tile coefficients near the
source have not grown beyond their initial uncertainty. The tile must NOT
have absorbed the dipole field.

## INV-02: Source Estimator Degrades Gracefully

A wrongly-initialized or diverging source estimate must not corrupt the
navigation state. The source-coupled update operates in three modes:

1. **SHADOW**: Source is tracked but not used in navigation (new sources)
2. **COV_ONLY**: Source uncertainty inflates navigation covariance but does not shift the mean
3. **SUBTRACT**: Source prediction is subtracted from measurements (fully trusted)

Promotion through modes requires passing observability gates (FIM rank),
consistency gates (NEES < 3.0), and persistence gates (minimum observation count).

**Enforcement**: `SourceSafetyController` monitors per-source NEES and demotes
sources that exceed thresholds. `SourceCoupledUpdate` implements the three modes.

**Test**: Initialize a source with 10× wrong moment. Verify navigation RMSE
increases by no more than 50% relative to no-source baseline (graceful degradation,
not divergence).

## INV-03: Clean Residual Contract

The residual passed to the map learning subsystem must satisfy:
  clean_residual = measured - background_predicted - Σ source_predicted

This residual should be dominated by:
1. Measurement noise (when map and sources are well-estimated)
2. Unmodeled background variation (drives map learning)

It must NOT contain:
- Known source contributions (those are subtracted)
- Navigation state errors (those are absorbed by the nav filter)

**Enforcement**: `SourceSeparation.extract_clean_measurement()` computes the
clean residual and sets `teachable=false` when source covariance dominates.

**Test**: With a known source active, verify `clean_residual` has zero mean
and variance consistent with measurement noise + map uncertainty. Verify
`teachable` flips to `false` when source_σ > 0.5 × measurement_σ.

## INV-04: Measurement Conservation

At every measurement step:
  background_pred + Σ source_pred + residual ≈ measured

The sum of all predicted contributions plus the residual must equal the raw
measurement to within floating-point tolerance (< 1e-12 T).

**Enforcement**: `OnlineSourceSLAM.get_source_contributions()` returns the
full attribution. Conservation is checked in the measurement processing loop.

**Test**: `test_measurement_attribution_conservation.jl` verifies this identity
at every step of a multi-source simulation.

## INV-05: Covariance Monotonicity Under Source Addition

Adding a source to the state vector must not decrease position uncertainty.
The augmented covariance must satisfy:
  P_pos_augmented ≥ P_pos_original (in the PSD sense)

This prevents false confidence from poorly-observed sources.

**Enforcement**: Source augmentation uses conservative cross-covariance
initialization. `SourceCoupledUpdate` in COV_ONLY mode inflates position
covariance by the source-position cross-coupling.

## INV-06: Floor Detection Consistency

Floor transitions detected by barometer must be consistent with elevator
source tracking. If the source tracker identifies an elevator in motion,
the floor detection module must not register a floor change from the
elevator's magnetic signature alone.

**Enforcement**: Floor detection uses barometric altitude as primary,
magnetic signatures as confirmatory only.

## INV-07: Deterministic Replay

Given the same seed, sensor inputs, and configuration, the system must
produce bit-identical state trajectories. This enables regression testing
and debugging.

**Enforcement**: All random number generation uses `NavCore` determinism
utilities. No floating-point non-determinism from threading.

## INV-08: Checkpoint-Rollback Safety

The system can create checkpoints and roll back to them without corrupting
state. After rollback, all subsequent processing produces the same results
as if the rolled-back states never existed.

**Enforcement**: `OnlineSafetyController` creates checkpoints at configurable
intervals. Rollback restores navigation, tile, and source states atomically.
