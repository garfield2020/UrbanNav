"""
    UrbanNav

Urban Navigation Core - Factor graph estimation for ground-based urban navigation.

The Core owns the truth; everything else plugs in.

# Architecture
- `contracts/` - State, Measurement, Factor, Health, Telemetry contracts
- `math/` - Linear algebra, rotations, uncertainty propagation, physics
- `graph/` - Factor graph structure and optimization engine
- `estimation/` - State estimation, covariance management, residuals
- `sensors/` - Sensor models (IMU, Odometry, Barometer, FTM) via registry
- `features/` - Feature types (dipole, etc.) via registry
- `fleet/` - Multi-device fusion policies
- `health/` - System health monitoring and diagnostics
- `io/` - Logging, serialization, telemetry, determinism

# Public API
- `UrbanNav.run_mission(config, world, sensors)` - Run a complete mission
- `UrbanNav.step!(estimator, measurements)` - Single estimation step
- `UrbanNav.export_state(state)` - Export current state estimate
"""
module UrbanNav

using LinearAlgebra
using StaticArrays
using SparseArrays
using Rotations
using Random
using Statistics: mean

# ============================================================================
# Contracts (the authoritative interfaces)
# ============================================================================

# StateContract defines core types used by everything else
include("contracts/StateContract.jl")

# MeasurementContract defines abstract types and interfaces
include("contracts/MeasurementContract.jl")

# FactorContract defines factor graph factor interfaces
include("contracts/FactorContract.jl")

# HealthContract defines health state machine
include("contracts/HealthContract.jl")

# MapBasisContract defines magnetic field basis math (single source of truth)
include("contracts/MapBasisContract.jl")

# ============================================================================
# Core math utilities
# ============================================================================

# Rotation utilities (quaternion, DCM, Euler)
include("math/rotations.jl")

# Uncertainty propagation (covariance, Mahalanobis, sigma points)
include("math/uncertainty.jl")

# Physics foundations (dipole, conductor, harmonic potential, Maxwell gate)
include("math/physics.jl")

# Whitening and chi-square computation
include("math/whitening.jl")

# Sigma total - full innovation covariance
include("math/sigma_total.jl")

# ============================================================================
# I/O and infrastructure
# ============================================================================

# Determinism utilities for reproducibility
include("io/determinism.jl")

# External interface for logging and serialization
include("io/external_interface.jl")

# Real-time timing budget and throttling
include("io/timing_budget.jl")

# ============================================================================
# Sensors
# ============================================================================

# Sensor registry (pluggable sensor models)
include("sensors/SensorRegistry.jl")

# Sensor models (Odometry, Depth, FTM, IMU)
include("sensors/sensors.jl")

# ============================================================================
# Factor graph engine
# ============================================================================

# Generic factor graph data structure
include("graph/FactorGraph.jl")

# Numerical conditioning (scaling, damping, preconditioning)
include("graph/conditioning.jl")

# Factor graph optimization (Gauss-Newton / LM)
include("graph/optimization.jl")

# UrbanNav-specific factor definitions (IMU, Odometry, etc.)
include("graph/factors.jl")

# ============================================================================
# State estimation
# ============================================================================

# Residual management and anomaly detection
include("estimation/residual_manager.jl")

# Dipole parameter estimation
include("estimation/dipole_fitter.jl")

# Main state estimator interface (public API)
include("estimation/StateEstimator.jl")

# NEES calibration infrastructure
include("estimation/calibration.jl")

# Physically-motivated process noise
include("estimation/process_noise.jl")

# ============================================================================
# Features (Pack 2: Dipole feature system)
# ============================================================================

# Feature registry (pluggable feature types)
include("features/FeatureRegistry.jl")

# Core dipole feature types and registry
include("features/dipole.jl")

# Multi-source MLE via coordinate descent
include("features/dipole_mle.jl")

# Spatial clustering with pose uncertainty
include("features/spatial_clustering.jl")

# Feature lifecycle and retirement
include("features/feature_lifecycle.jl")

# Feature disambiguation (association, split, merge)
include("features/feature_disambiguation.jl")

# Feature-to-map absorption
include("features/feature_absorption.jl")

# Feature marginalization (Schur complement)
include("features/feature_marginalization.jl")

# ============================================================================
# Mapping (Pack 1: Tile-based field estimation)
# ============================================================================

# Tile coefficients and harmonic basis
include("mapping/tile_coefficients.jl")

# Tile manager for spatial indexing
include("mapping/tile_manager.jl")

# Map update policy (learn vs freeze)
include("mapping/map_update_policy.jl")

# Gradient integration (d=8 full tensor)
include("mapping/gradient_integration.jl")

# Tensor selectivity (mode selection)
include("mapping/tensor_selectivity.jl")

# Temporal coherence validation
include("mapping/temporal_coherence.jl")

# Map persistence and versioning
include("mapping/map_store.jl")

# ============================================================================
# Health Monitoring (Pack 3: Per-subsystem health tracking)
# ============================================================================

# Health types and subsystem definitions
include("health/health_types.jl")

# Health monitor with checkers
include("health/HealthMonitor.jl")

# ============================================================================
# Fleet (Pack 4: Multi-vehicle coordination)
# ============================================================================

# Fleet registry for fusion policies
include("fleet/FleetRegistry.jl")

# Fleet types (VehicleState, FleetState, messages)
include("fleet/fleet_types.jl")

# Inter-vehicle ranging
include("fleet/ranging.jl")

# Fusion policy implementations
include("fleet/fusion_policies.jl")

# Note: FleetMapFusion is included after MapContract (it depends on MapTileID, etc.)

# ============================================================================
# Testing Infrastructure (fault injection)
# ============================================================================

include("testing/fault_injection.jl")
include("testing/doe.jl")
include("testing/qualification.jl")
include("testing/scenario_loader.jl")
include("testing/tiered_gates.jl")
include("testing/seed_grid.jl")
include("testing/nees_diagnostics.jl")
include("testing/ttd_metrics.jl")
include("testing/observability_classification.jl")
include("testing/failure_atlas.jl")
include("testing/qualification_evidence.jl")
include("testing/regression_suite.jl")
include("testing/qualification_runner.jl")

# Elevator DOE testing infrastructure
include("testing/elevator_mode_config.jl")
include("testing/elevator_doe_metrics.jl")
include("testing/elevator_map_poisoning.jl")
include("testing/elevator_doe.jl")

# ============================================================================
# Public API - the ONLY entry points for external code
# ============================================================================
export run_mission, step!, export_state
export validate_config, load_config

# Re-export core types from contracts
export Vec3, Mat3, Mat15
export UrbanNavState, Keyframe
export position, velocity, orientation, bias_gyro, bias_accel, altitude
export rotation_matrix, euler_angles, state_dim, error_state_vector
export apply_error_state!, transform_to_body, transform_to_world
export position_uncertainty, position_std
export create_initial_state, add_measurement!, get_measurement

# Re-export measurement interfaces
export AbstractMeasurement, AbstractMagneticField, AbstractSensor, AbstractEstimator
export AbstractResidualManager, ScenarioConfig, ScenarioResult
export field_at, gradient_at, field_at_body, gradient_at_body
export measure, measurement_covariance, measurement_dimension
export transform_field_to_body, transform_field_to_world
export transform_gradient_to_body, transform_gradient_to_world

# Re-export factor contract (internal, but needed by some tests)
export AbstractFactor, FactorMetadata
export residual, jacobian, noise_model, whiten, factor_dim, connected_states
export PriorFactor, BetweenFactor, MeasurementFactor

# Re-export health state machine (from HealthContract)
export HealthState, HEALTH_HEALTHY, HEALTH_DEGRADED, HEALTH_UNRELIABLE
export SensorStatus, SENSOR_OK, SENSOR_DEGRADED, SENSOR_FAILED
export SafeStateAction, ACTION_NONE, ACTION_REDUCE_SPEED, ACTION_HOLD_POSITION, ACTION_SURFACE
export TransitionRule, TransitionInputs, TransitionRecord
export HealthStateMachineConfig, DEFAULT_HSM_CONFIG
export HealthStateMachine, evaluate_transition!, get_current_state, get_nav_state
export StateAction, get_state_actions
export print_transition_log, export_transition_log_json
export build_transition_inputs
export valid_health_transitions, is_valid_transition

# Re-export health monitoring (Pack 3)
export MonitoredHealthState, NOMINAL, CAUTION, WARNING, CRITICAL
export HEALTH_SUBSYSTEMS, HealthTransition, HealthReport
export is_healthy, get_degraded_subsystems
export HealthMetrics, metrics_to_dict
export map_to_nav_health, build_transition_inputs_from_report
export HealthMonitor, HealthContext, AbstractHealthChecker
export check_health, get_report, register_checker!, run_check
export create_default_monitor, reset!, get_subsystem_state
export CovarianceChecker, InnovationChecker, ResidualChecker
export SensorChecker, TimingChecker

# Re-export fleet (Pack 4)
export AbstractFleetPolicy, FleetRegistry
export register_fleet_policy!, get_fleet_policy, has_fleet_policy
export list_fleet_policies, clear_fleet_policies!
export VehicleId, VehicleState, FleetState
export vehicle_position, vehicle_position_std, is_stale
export update_vehicle!, remove_vehicle!, get_vehicle
export active_vehicles, healthy_vehicles, num_vehicles
export fleet_centroid, fleet_spread
export FleetMessageType, FleetMessage
export MSG_STATE_UPDATE, MSG_POSITION_ONLY, MSG_RANGING_REQUEST
export MSG_RANGING_RESPONSE, MSG_HEALTH_STATUS
export state_update_message, position_only_message
export ranging_request_message, ranging_response_message
export FleetConfig, DEFAULT_FLEET_CONFIG
export SOUND_SPEED_AIR, DEFAULT_RANGING_STD
export RangingProtocol, PROTOCOL_ONE_WAY, PROTOCOL_TWO_WAY
export RangingMeasurement, RangingModel, DEFAULT_RANGING_MODEL
export expected_range, range_residual, range_chi2
export ranging_noise, is_valid_range, simulate_ranging
export RangingFactor, RangingSchedule
export should_range, record_ranging!, pairs_to_range
export CentralizedFusion, DecentralizedFusion, HierarchicalFusion
export covariance_intersection, optimize_omega
export set_hierarchy!, is_root, is_leaf, all_children_reported
export register_default_fleet_policies!

# Re-export fleet map fusion (Phase B)
export FleetMapMessageType, MAP_MSG_TILE_UPDATE, MAP_MSG_VERSION_QUERY
export MAP_MSG_VERSION_RESPONSE, MAP_MSG_VALIDATION_RESULT
export TileUpdatePayload, compute_tile_quality
export FleetMapMessage, tile_update_message, version_query_message, version_response_message
export FleetMapFusionConfig, DEFAULT_FLEET_MAP_FUSION_CONFIG
export map_covariance_intersection, optimize_map_omega
export FleetMapFusionStatistics, update_fusion_statistics!, fusion_success_rate
export FleetMapFusionResult
export FleetMapFusionManager, receive_tile_update!, fuse_tile!
export prune_stale_peers!, peers_with_tile
export MapVersionConflict, detect_version_conflicts
export format_fusion_statistics, format_fusion_result

# Re-export Phase B: Fleet Metrics (Qualification)
export VehicleMetrics, record_position!, record_residual!, record_update!, record_fusion!
export compute_rmse, compute_nees_mean, compute_nees_consistency
export compute_residual_quantiles, compute_update_acceptance_rate, compute_fusion_success_rate
export FleetMetrics, get_vehicle_metrics, record_comms!, record_conflict!, record_rollback!
export compute_fleet_rmse_mean, compute_fleet_rmse_worst, compute_fleet_nees_mean
export compute_fleet_nees_consistency, compute_rollback_rate, compute_fleet_map_quality
export compute_comms_per_km, compute_bytes_per_km
export FleetImprovementMetrics, compute_improvement
export FleetQualificationResult, evaluate_qualification
export format_vehicle_metrics, format_fleet_metrics, format_qualification_result

# Re-export rotations
export quat_multiply, quat_conjugate, quat_rotate, quat_to_dcm, dcm_to_quat
export euler_to_quat, quat_to_euler, quat_normalize

# Re-export uncertainty
export propagate_covariance, mahalanobis_distance, sigma_points
export covariance_union, ensure_positive_definite, condition_number

# Re-export physics
export μ₀, μ₀_4π
export MagneticDipole, ConductorSegment, ConductorPath, HarmonicPotential
export field, gradient, potential
export MaxwellGateResult, apply_maxwell_gate, apply_maxwell_gate_5component

# Re-export whitening
export WhiteningResult, whiten, whiten_full
export compute_chi2_direct, compute_chi2_cholesky, compute_chi2_whitened_sum
export verify_chi2_consistency
export whiten_field_only, whiten_field_gradient, build_combined_covariance
export TotalCovarianceComponents, build_total_covariance

# Re-export sigma total
export SigmaTotalConfig, DEFAULT_SIGMA_TOTAL_CONFIG
export compute_sigma_meas, compute_sigma_total, compute_sigma_total_simple
export ensure_spd

# Re-export sensor registry
export AbstractSensorModel, SensorRegistry
export register_sensor!, get_sensor, list_sensors, has_sensor, clear_sensors!

# Re-export feature registry
export AbstractFeatureType, FeatureRegistry
export register_feature!, get_feature, list_features, has_feature, clear_features!

# Re-export sensors
export OdometryParams, OdometryMeasurement, OdometryModel
export BarometerParams, BarometerMeasurement, BarometerModel
export FTMParams, FTMMeasurement, FTMModel
export IMUParams, IMUMeasurement, IMUModel
export simulate_measurement
export gradient_to_vector, vector_to_gradient

# Re-export AUV-specific factors
export Factor, jacobians, information_matrix
export IMUPreintegration, integrate!, correct_bias
export IMUPreintegrationFactor, OdometryFactor, BarometerFactor, MagneticFactor
export UrbanNavFactorGraph, add_node!, compute_total_error
export skew

# Re-export generic FactorGraph (internal, for migration)
export FactorGraph

# Re-export conditioning
export ScalingConfig, DEFAULT_SCALING_CONFIG
export build_scaling_matrix, compute_auto_scales
export LMDampingPolicy, LM_FIXED, LM_MULTIPLICATIVE, LM_NIELSEN, LM_TRUST_REGION
export LMDampingState, update_damping!, apply_lm_damping, apply_lm_damping_identity
export JacobiPreconditioner, build_jacobi_preconditioner
export apply_preconditioner, unapply_preconditioner
export ConditioningStats, compute_conditioning_stats
export check_spd, regularize_hessian
export ConditioningPipeline, SolveResult, solve_conditioned

# Re-export residual manager
export GatingDecision, INLIER, MILD_OUTLIER, STRONG_OUTLIER
export GatingConfig, chi2_threshold, compute_chi2, gate_measurement
export ResidualStatistics, compute_statistics
export PhysicsGateConfig, PhysicsGatedStatistics
export FeatureStatus, CANDIDATE, CONFIRMED, DEMOTED, ABSORBED
export FeatureCandidate, ResidualManager
export process_measurement!, get_confirmed_features, get_active_candidates

# Re-export dipole fitter
export DipoleFitConfig, DipoleFitResult, ConfidenceLevel
export fit_dipole, fit_dipole_with_gradient
export CONFIDENCE_LOW, CONFIDENCE_MEDIUM, CONFIDENCE_HIGH
export compute_fit_covariance, assess_observability
export DEFAULT_DIPOLE_FIT_CONFIG

# Re-export determinism
export DEFAULT_SEED, set_global_seed!, get_global_rng, reset_global_rng!
export with_seed, seeded_randn, seeded_rand

# Re-export I/O
export LogLevel, LOG_DEBUG, LOG_INFO, LOG_WARN, LOG_ERROR, LOG_CRITICAL
export LogEntry, NavLogger, log!, flush_logs!
export NavStateExport, to_json, to_binary
export AbstractTelemetryPublisher, publish!, NullPublisher, LoggingPublisher

# Re-export timing budget
export TimingBudget, DEFAULT_TIMING_BUDGET, HIGH_PERF_TIMING_BUDGET, EMBEDDED_TIMING_BUDGET
export TaskTiming, TimingTracker
export start_keyframe!, end_keyframe!, time_task!
export remaining_budget, can_afford
export ThrottleDecision, NO_THROTTLE, SKIP_FEATURE_WORK, SKIP_MAP_OPTIMIZE
export SKIP_FULL_OPTIMIZE, EMERGENCY_THROTTLE
export throttle_decision, should_skip_feature_work, should_skip_optimization
export record_skipped_work!
export AdaptiveBudgetConfig, adapt_budget!
export TimingStats, get_timing_stats, print_timing_report
export with_timing_budget
export LoadSpikeDetector, update_load!

# Re-export StateEstimator (PUBLIC API)
export StateEstimator, EstimatorConfig, StepResult
export create_estimator, get_state, get_covariance

# Re-export process noise
export ProcessNoiseConfig, DEFAULT_PROCESS_NOISE_CONFIG
export BiasRandomWalkConfig, DEFAULT_BIAS_RANDOM_WALK_CONFIG
export ProcessNoiseState, ProcessNoiseResult
export discretize_process_noise, compute_position_process_noise
export compute_velocity_process_noise, compute_attitude_process_noise
export compute_bias_process_noise, compute_full_process_noise
export ProcessNoiseCalibrator, update_process_noise_calibration!
export recommend_process_noise_tuning

# Re-export fault injection (testing)
export FaultType, FAULT_NONE, FAULT_BIAS_DRIFT, FAULT_SENSOR_DROPOUT
export FAULT_MAP_ERROR, FAULT_MAP_OFFSET, FAULT_MAP_SCALE
export FAULT_TIMING_SPIKE, FAULT_TIMING_JITTER, FAULT_MEMORY_PRESSURE
export FAULT_OUTLIER_BURST, FAULT_NOISE_INCREASE
export FAULT_GRADUAL_DRIFT, FAULT_SUDDEN_JUMP, FAULT_OSCILLATION
export FaultConfig, FaultInjector, inject_fault!, clear_faults!, get_active_faults
export update_injector!
export DivergenceConfig, DivergenceState, DivergenceResult
export DivergenceDetector, update_divergence!, check_divergence, is_silent_divergence
export FaultInjectionGate, GateResult, GATE_PASS, GATE_FAIL, GATE_INCONCLUSIVE
export run_fault_gate, run_fault_scenario, FaultScenario
export SilentDivergenceChecker
export lawnmower_turn_scenario, sensor_dropout_scenario, map_error_scenario

# Re-export DOE (Design of Experiments)
export DOEFactor, DOELevel, DOEDesign, DOERun, DOEResult
export TrajectoryType, TRAJ_STRAIGHT, TRAJ_LAWNMOWER, TRAJ_SPIRAL, TRAJ_HOVER
export TRAJ_CIRCLE, TRAJ_FIGURE_EIGHT
export FieldStrength, FIELD_WEAK, FIELD_NOMINAL, FIELD_STRONG
export GradientStrength, GRAD_WEAK, GRAD_NOMINAL, GRAD_STRONG
export SensorConfig, SENSORS_FULL, SENSORS_NO_ODOMETRY, SENSORS_NO_COMPASS, SENSORS_MINIMAL
export trajectory_factor, field_strength_factor, gradient_strength_factor, sensor_config_factor
export ObservabilityDOE, PerformanceDOE
export ObservabilityMetrics, PerformanceMetrics
export create_full_factorial, create_fractional_factorial
export run_doe!, analyze_observability, analyze_performance
export DOEAnalysis, find_observability_boundary, compute_main_effects
export compute_observability_metrics, is_observable
export compute_performance_metrics, passes_performance
export create_observability_doe, create_performance_doe
export quick_observability_doe, comprehensive_observability_doe, lawnmower_observability_doe

# Re-export Elevator DOE
export ElevatorNavMode, NAV_MODE_A_BASELINE, NAV_MODE_B_ROBUST_IGNORE, NAV_MODE_C_SOURCE_AWARE
export ElevatorModeConfig, configure_mode_a, configure_mode_b, configure_mode_c, configure_elevator_mode
export ElevatorDOEMetrics, SegmentMetrics
export compute_elevator_metrics, compute_segment_metrics
export compute_do_no_harm, compute_map_contamination, compute_innovation_burst
export MapPoisoningResult, run_map_poisoning_test
export ElevatorSpeed, ELEV_SLOW, ELEV_MEDIUM, ELEV_FAST
export StopFrequency, STOP_RARE, STOP_NORMAL, STOP_FREQUENT
export ClosestApproach, APPROACH_NEAR, APPROACH_MEDIUM, APPROACH_FAR
export DipoleStrength, DIPOLE_WEAK, DIPOLE_NOMINAL, DIPOLE_STRONG
export ShaftGeometry, SHAFT_SINGLE, SHAFT_OFFSET, SHAFT_DUAL
export TrajectoryRichness, RICHNESS_STRAIGHT, RICHNESS_L_TURNS, RICHNESS_LOOP
export SensorNoiseScale, NOISE_HALF, NOISE_NOMINAL, NOISE_DOUBLE
export ElevatorDOEPoint, ElevatorDOEDesign, ElevatorDOERun, ElevatorDOEResult
export create_practical_first_doe, create_screening_doe
export build_elevator_world, build_trajectory, build_control_world
export run_elevator_doe!, run_single_point

# Re-export Qualification Framework (Step 10)
export QualificationMode, MODE_OBSERVABILITY, MODE_PERFORMANCE
export QualificationStatus, QUAL_PASS, QUAL_FAIL, QUAL_CONDITIONAL, QUAL_NOT_RUN
export QualificationConfig, DEFAULT_QUALIFICATION_CONFIG
export ObservabilityQualResult, PerformanceQualResult, QualificationResult
export QualificationReport, QualificationGate
export QualificationRunner, add_scenario!
export run_observability_qualification!, run_performance_qualification!, run_qualification!
export create_default_qualification, qualification_summary
export create_observability_gates, create_performance_gates
export quick_qualification_check
export validate_nees_calibration, validate_innovation_covariance, validate_process_noise

# Re-export Scenario Loader (V1.0 Qualification)
export ScenarioMetadata, TrajectoryConfig, SensorSuiteConfig
export IMUSensorConfig, OdometrySensorConfig, DepthSensorConfig
export CompassSensorConfig, FTMSensorConfig
export WorldConfig, DipoleConfig, ClutterConfig, ClutterObject
export MapErrorConfig, MapErrorEntry
export FaultScheduleEntry, FaultScheduleConfig
export ScenarioPassCriteria, ScenarioDefinition
export ValidationResult, validate_scenario
export load_scenario, load_all_scenarios
export scenario_to_qualification_config
export ScenarioQualificationResult, QualificationSuiteResult
export run_scenario_qualification, run_all_qualification_scenarios
export scenario_summary

# Re-export Tiered Gates (V1.0 Qualification Step 2)
export GateTier, TIER_EXTERNAL, TIER_INTERNAL
export TieredGate, TieredGateResult
export V1_0_EXTERNAL_THRESHOLDS, V1_0_INTERNAL_THRESHOLDS
export TieredGateConfig, DEFAULT_V1_0_GATE_CONFIG
export create_v1_0_external_gates, create_v1_0_internal_gates
export create_all_v1_0_gates
export TieredQualificationResult, run_tiered_qualification
export evaluate_tiered_gates, format_tiered_report

# Re-export Seed Grid (V1.0 Qualification Step 3)
export SeedGridMode, MODE_QUICK, MODE_FULL, MODE_CUSTOM
export SeedGridConfig, QUICK_SEED_CONFIG, FULL_SEED_CONFIG
export SeedRunResult, SeedGridResult, ScenarioSeedResults
export run_seed_grid, aggregate_seed_results
export SeedGridRunner, get_progress, get_results
export format_seed_grid_summary, analyze_seed_failures
export generate_seeds, generate_scenario_seeds

# Re-export NEES Diagnostics (V1.0 Qualification Step 4)
export NEESComponents, compute_nees_components
export NEESSample, NEESTimeSeries, add_sample!
export NEESWindowStats, compute_window_stats
export ChiSquaredTest, chi2_cdf, chi2_quantile, run_chi2_test
export NEESConsistencyResult, check_nees_consistency
export NEESTrend, detect_nees_trend
export CovarianceDiagnostics, diagnose_covariance
export NEESRootCause, CAUSE_NONE, CAUSE_Q_TOO_SMALL, CAUSE_Q_TOO_LARGE
export CAUSE_R_TOO_SMALL, CAUSE_R_TOO_LARGE, CAUSE_BIAS_UNMODELED
export CAUSE_OBSERVABILITY, CAUSE_DIVERGENCE
export NEESDiagnosticResult, run_nees_diagnostics
export CalibrationRecommendation, generate_recommendations
export format_nees_diagnostic_report

# Re-export TTD Metrics (V1.0 Qualification Step 5)
export TTDSample, TTDStatistics, TTDTracker
export TTDConfig, DEFAULT_TTD_CONFIG
export TTDGateResult, TTDGate, evaluate_ttd_gate
export start_fault!, record_detection!, get_ttd_samples
export compute_ttd_statistics, reset_tracker!, update_tracker!
export TTDScenarioResult, TTDSuiteResult
export run_ttd_scenario, analyze_ttd_results
export format_ttd_report, format_ttd_summary
export create_ttd_external_gate, create_ttd_internal_gate

# Re-export Observability Classification (V1.0 Qualification Step 6)
export ObservabilityFailureCategory
export OBS_EXPECTED_TRAJECTORY, OBS_EXPECTED_SENSOR
export OBS_EXPECTED_ENVIRONMENT, OBS_UNEXPECTED_SYSTEM
export OBS_UNEXPECTED_UNKNOWN, OBS_RECOVERABLE, OBS_PERSISTENT
export ObservabilityFailureClass
export OBSFAIL_Y_AXIS_STRAIGHT, OBSFAIL_HEADING_STATIC
export OBSFAIL_Odometry_DROPOUT, OBSFAIL_COMPASS_FAILURE
export OBSFAIL_WEAK_GRADIENT, OBSFAIL_MAGNETIC_INTERFERENCE
export OBSFAIL_FILTER_DIVERGENCE, OBSFAIL_UNKNOWN
export OBSFAIL_HOVER_POSITION, OBSFAIL_DEPTH_FAILURE
export OBSFAIL_FTM_DROPOUT, OBSFAIL_UNIFORM_FIELD
export OBSFAIL_COVARIANCE_SINGULAR, OBSFAIL_NUMERICAL_INSTABILITY
export ObservabilityFailure, ObservabilityClassifier
export ObservabilityClassificationConfig, DEFAULT_OBS_CLASS_CONFIG
export classify_observability_failure, is_expected_failure
export get_failure_mitigation, get_failure_explanation
export ObservabilityFailureReport, generate_failure_report
export format_observability_classification
export KnownLimitation, V1_0_KNOWN_LIMITATIONS
export is_known_limitation, get_known_limitation
export update_classifier!
export ObservabilityQualificationResult, evaluate_observability_qualification

# Re-export Failure Atlas (V1.0 Qualification Step 7)
export FailureMode, FailureTrigger, FailureBoundary
export FailureCorrelation, FailureAtlas
export FailureAtlasConfig, DEFAULT_ATLAS_CONFIG
export AtlasEntry, AtlasSection
export discover_failure_modes, identify_failure_boundaries
export compute_failure_correlations, generate_failure_atlas
export format_failure_atlas, export_failure_atlas_markdown
export FailureCluster, cluster_failures
export OperationalEnvelope, compute_operational_envelope
export format_trigger, identify_failure_triggers

# Re-export Qualification Evidence (V1.0 Qualification Step 8)
export QualificationEvidence, EvidenceSection, EvidenceSummary
export QualificationEvidenceConfig, DEFAULT_EVIDENCE_CONFIG
export ScenarioEvidence, GateEvidence
export EvidenceStatus, EVIDENCE_PASS, EVIDENCE_CONDITIONAL, EVIDENCE_FAIL
export generate_qualification_evidence, run_qualification_evidence
export format_evidence_package, export_evidence_markdown
export export_evidence_json, create_evidence_archive
export collect_scenario_evidence, collect_gate_evidence
export count_evidence_status, generate_evidence_summary

# Re-export Regression Suite (V1.0 Qualification Step 9)
export RegressionMode, REG_SMOKE, REG_QUICK, REG_FULL, REG_NIGHTLY
export RegressionConfig, DEFAULT_REGRESSION_CONFIG
export SMOKE_REGRESSION_CONFIG, QUICK_REGRESSION_CONFIG, FULL_REGRESSION_CONFIG
export RegressionResult, ScenarioRegressionResult
export RegressionSummary, RegressionReport
export run_regression_suite, run_regression_scenarios
export format_regression_report, export_regression_junit
export check_regression_gates, get_regression_exit_code

# Re-export Qualification Runner (V1.0 Qualification Step 10)
export RunnerMode, RUNNER_QUICK, RUNNER_FULL, RUNNER_NIGHTLY
export ObsClassification, OBS_FULL, OBS_PARTIAL, OBS_NONE
export NEESBreakdown, NEESDiagnostics, TTDResult, ObservabilityClassification
export QualificationRunnerConfig, DEFAULT_RUNNER_CONFIG
export QUICK_RUNNER_CONFIG, FULL_RUNNER_CONFIG, NIGHTLY_RUNNER_CONFIG
export QualificationRunnerResult, QualificationArtifacts
export run_full_qualification, run_qualification_cli
export generate_all_artifacts, validate_qualification_result
export format_qualification_summary, export_qualification_artifacts

# Re-export FactorGraph (internal, but needed for migration)
export Variable, add_variable!, get_variable
export num_variables, num_factors, total_state_dim
export OptimizationParams, OptimizationResult
export marginal_covariance

# ============================================================================
# Re-export Feature Pack (Pack 2)
# ============================================================================

# Dipole feature types
export DipoleFeatureState, DIPOLE_FEATURE_DIM
export to_state_vector, from_state_vector, to_magnetic_dipole
export DipoleLifecycleState, DIPOLE_CANDIDATE, DIPOLE_ACTIVE, DIPOLE_RETIRED, DIPOLE_DEMOTED
export DipoleFeatureNode, DipoleFeatureCandidate
export add_candidate_measurement!, update_candidate_statistics!
export candidate_duration, candidate_support_count
export PromotionGateConfig, DEFAULT_PROMOTION_GATE_CONFIG
export DipoleFeatureRegistry
export dipoles_enabled, set_dipoles_enabled!
export create_feature_candidate!, get_feature_candidate, get_dipole_feature
export n_feature_candidates, n_dipole_features
export remove_feature_candidate!, promote_feature_candidate!
export retire_dipole_feature!, demote_dipole_feature!
export feature_field, feature_field_jacobian
export PromotionCheckResult, check_promotion_gates

# Multi-source MLE
export DipoleSourceEstimate, FieldMeasurementSet
export dipole_field_estimate, dipole_field_jacobians
export total_field_from_sources, field_excluding_source
export SingleSourceResult, single_source_mle
export CoordinateDescentConfig, DEFAULT_COORDINATE_DESCENT_CONFIG
export MultiSourceResult, coordinate_descent_mle
export source_separation

# Spatial clustering
export SpatialDetection, SpatialCluster
export euclidean_dist, mahalanobis_dist, mahalanobis_dist_sq
export SpatialClusteringConfig, DEFAULT_SPATIAL_CLUSTERING_CONFIG
export greedy_spatial_cluster, add_to_spatial_cluster!, max_cluster_extent
export CompactnessResult, evaluate_compactness
export SpatialClusteringPipeline
export add_spatial_detection!, process_spatial_detections!, get_confirmed_cluster_sources

# Feature lifecycle
export FeatureLifecycleConfig, DEFAULT_LIFECYCLE_CONFIG
export FeatureLifecycleState, FeatureLifecycleManager
export init_feature_lifecycle!, get_feature_lifecycle, remove_feature_lifecycle!
export compute_effective_support, decay_all_feature_support!, add_feature_observation!
export compute_contribution_metric, evaluate_feature_contribution
export FeatureRetirementReason
export FEATURE_NOT_OBSERVED, FEATURE_LOW_CONTRIBUTION, FEATURE_LOW_SUPPORT
export FEATURE_CONFIDENCE_DEGRADED, FEATURE_ABSORBED_BY_MAP, FEATURE_MANUAL
export FeatureLifecycleDecision, check_feature_retirement, process_feature_lifecycle!
export update_feature_observation!
export FeatureLifecycleStats, get_feature_lifecycle_stats

# Feature disambiguation
export FeatureDisambiguationConfig, DEFAULT_DISAMBIGUATION_CONFIG
export feature_mahalanobis_distance, feature_mahalanobis_distance_sq
export FeatureAssociationDecision
export ASSOCIATE_CLEAR, ASSOCIATE_AMBIGUOUS, ASSOCIATE_NO_MATCH, ASSOCIATE_SPLIT
export FeatureAssociationResult, associate_feature_measurement
export FeatureSplitResult, detect_feature_split, simple_2means_clustering
export FeatureMergeResult, detect_feature_merge, find_feature_merge_candidates
export FeatureDisambiguationManager, process_feature_disambiguation!

# Feature absorption
export FeatureAbsorptionConfig, DEFAULT_ABSORPTION_CONFIG
export FeatureAbsorptionCriteria, check_feature_absorption_criteria
export FeatureAbsorptionState, update_absorption_blend!, get_absorption_blended_field
export FeatureAbsorptionManager
export start_feature_absorption!, update_feature_absorptions!, get_total_absorption_field
export process_feature_absorptions!

# Feature marginalization
export FeatureMarginalizationConfig, DEFAULT_MARGINALIZATION_CONFIG
export schur_complement
export FeatureMarginalizedPrior, apply_marginalized_prior!
export FeatureMarginalizationResult, marginalize_feature
export compute_feature_information
export prune_retired_features!
export FeatureGraphMemoryStats, estimate_feature_graph_memory, check_feature_graph_growth

# ============================================================================
# Re-export Mapping Pack (Pack 1)
# ============================================================================

# Tile coefficients and harmonic basis
export get_harmonic_basis, get_harmonic_basis_with_hessian
export pack_gradient_tensor, unpack_gradient_tensor
export TileCoefficients, n_coefficients, local_position, in_tile
export evaluate_tile_field, evaluate_tile_field_jacobian
export evaluate_tile_gradient, evaluate_tile_gradient_jacobian
export evaluate_tile_field_and_gradient

# Tile manager
export TileManager
export tile_index, tile_center
export get_or_create_tile!, get_tile, get_tile_by_index, has_tile
export get_neighboring_tiles, get_8_neighboring_tiles
export all_tiles, all_tile_indices, n_tiles
export evaluate_field_at, evaluate_field_and_gradient_at
export TileStatistics, compute_tile_statistics

# Map update policy
export UpdatePolicy, TileUpdateDecision, TileUpdateContext
export AdaptiveUpdatePolicy, ConservativeUpdatePolicy, AggressiveUpdatePolicy
export ManualFreezePolicy, PolicyManager
export DEFAULT_UPDATE_POLICY
export decide_update, decide!, get_policy_statistics, reset_policy_statistics!
export freeze_decision, update_decision, partial_update_decision
export freeze_tile!, unfreeze_tile!
export apply_weighted_update, apply_weighted_covariance_update
export compute_innovation_sigma, recommend_policy

# Gradient integration (d=8)
export GradientConfig, DEFAULT_GRADIENT_CONFIG, GRADIENT_DISABLED_CONFIG
export CHI2_D3_MILD_THRESHOLD, CHI2_D3_STRONG_THRESHOLD
export CHI2_D8_MILD_THRESHOLD, CHI2_D8_STRONG_THRESHOLD
export chi2_threshold_for_d
export GradientMeasurement, gradient_measurement_vector, gradient_measurement_covariance
export GradientResidualStatistics, compute_gradient_residual_statistics
export GradientConfidenceModel, GRADIENT_CONFIDENCE_MODEL, FIELD_CONFIDENCE_MODEL
export compute_gradient_confidence, confidence_label
export GradientBaseline, update_gradient_baseline!, get_baseline_comparison

# Tensor selectivity
export MeasurementMode, MODE_FIELD_ONLY, MODE_GRADIENT_ONLY, MODE_FULL_TENSOR
export measurement_dimension
export TensorSelectivityConfig, DEFAULT_SELECTIVITY_CONFIG
export estimate_gradient_snr, estimate_snr_at_range, crossover_range
export ModeSelectionResult, select_measurement_mode, select_measurement_mode_with_covariance
export GradientFisherInfo, compute_gradient_fisher_gain, optimal_gradient_range
export AdaptiveModeSelector, update_mode_selector!, get_selector_statistics
export apply_mode_selection, build_mode_covariance

# Temporal coherence
export TemporalMeasurement, MeasurementBuffer
export add_to_buffer!, get_time_window, get_spatial_window
export PersistenceConfig, PersistenceResult, test_persistence
export ProfileConfig, ProfileResult, test_profile_consistency
export SpectralConfig, SpectralResult, test_spectral_stability
export CausalConfig, CausalResult, test_causal_consistency
export TemporalCoherenceConfig, DEFAULT_TEMPORAL_COHERENCE_CONFIG
export TemporalCoherenceResult, test_temporal_coherence

# Map storage and versioning
export TileSnapshot, restore_tile
export MapVersion, generate_version_id
export TileDiff, MapDiff, compute_tile_diff, compute_map_diff
export MapStore
export save_store_index!, load_store_index!
export save_map_version!, load_map_version
export save_map!, load_map, load_latest_map
export list_map_versions, diff_map_versions, rollback_map!
export get_version_chain, compute_version_statistics

# ============================================================================
# Frozen Map Mode (Phase A: Map-backed localization)
# ============================================================================

# Map contract (authoritative interface)
include("contracts/MapContract.jl")

# Map basis and fitting
include("map/MapBasis.jl")

# Map provider interface and implementations
include("map/MapProvider.jl")

# Map update pipeline (Phase B)
include("map/MapUpdate.jl")

# Map learning gate (Phase B) - separates navigation from learning residuals
include("map/MapLearningGate.jl")

# Map versioning and rollback (Phase B)
include("map/MapVersioning.jl")

# Manifold collapse metrics (Phase B)
include("map/ManifoldCollapse.jl")

# Basis order ladder (Phase B)
include("map/BasisOrderLadder.jl")

# Source separation (Phase B) - clean object coupling
include("map/SourceSeparation.jl")

# Fleet map fusion (Phase B) - multi-vehicle map learning
# Note: Must be included AFTER MapContract because it depends on MapTileID, MapTileData, etc.
include("fleet/FleetMapFusion.jl")

# Fleet metrics (Phase B) - qualification metrics for fleet fusion testing
include("fleet/FleetMetrics.jl")

# Re-export Frozen Map Mode (Phase A)
export Vec3Map, Mat3Map
export MapBasisType, MAP_BASIS_LINEAR, MAP_BASIS_QUADRATIC, MAP_BASIS_CUBIC
export MapFrame, MapMetadata
export MapTileID, tile_id_at
export MapQueryResult, pack_gradient, unpack_gradient
export MapTileData, MapModel, has_coverage, get_tile
export AbstractMapProvider
export FrozenFileMapProvider, save_map, convert_harmonic_model_to_map
export NullMapProvider
export LinearHarmonicModel, predict_field, predict_gradient, predict_with_uncertainty
export MapFitSample, MapFitResult, fit_linear_harmonic_model
export fit_from_field_samples, fit_from_field_and_gradient_samples
export compute_fit_residuals
export compute_gradient_energy, compute_gradient_condition, compute_directional_observability

# Re-export Phase B: Map Update Contract
export MapVersionInfo, is_frozen
export MapUpdateSource, UPDATE_SOURCE_NAVIGATION, UPDATE_SOURCE_SURVEY
export UPDATE_SOURCE_FLEET, UPDATE_SOURCE_MANUAL
export MapUpdateMessage, MapUpdateResult
export AbstractMapUpdater, apply_update, apply_batch
export MutableMapModel, freeze, update_tile!, get_version

# Re-export Phase B: Map Update Pipeline
export MapUpdaterConfig, DEFAULT_MAP_UPDATER_CONFIG
export InformationFormUpdater
export chi2_outlier_threshold, default_outlier_threshold
export compute_normalized_innovation_squared, is_outlier
export compute_pose_field_coupling
export verify_commutativity, verify_covariance_reduction
export CHI2_D3_P001, CHI2_D3_P0001, CHI2_D8_P001, CHI2_D8_P0001
export DEFAULT_MAX_POSE_UNCERTAINTY_M2

# Re-export Phase B: Map Learning Gate
export NavigationResidual, MapLearningResidual
export MapLearningConfig, DEFAULT_MAP_LEARNING_CONFIG
export compute_pose_to_measurement_covariance
export create_navigation_residual, create_map_learning_residual
export compute_learning_weight
export LearningGateStatistics, update_statistics!, teachability_rate

# Re-export Phase B: Map Versioning and Rollback
export MapCheckpoint, MapProvenance
export frozen_provenance, learning_provenance, rollback_provenance
export ValidationMetrics, ValidationResult, ValidationConfig
export DEFAULT_VALIDATION_CONFIG, validate_metrics
export MapVersionHistory
export current_checkpoint, get_checkpoint, list_versions, latest_version
export create_checkpoint!, rollback!, record_validation!, get_validation
export find_last_good_version, auto_rollback_if_degraded!
export restore_checkpoint, compute_scenario_hash, format_version_history

# Re-export Phase B: Manifold Collapse Metrics
export CollapseSnapshot, CollapseTrajectory, add_snapshot!
export CollapseMetrics, compute_collapse_metrics
export ConvergenceCriteria, DEFAULT_CONVERGENCE_CRITERIA
export ConvergenceResult, check_convergence
export MissionComparison, compare_missions, prove_mission_improvement
export AcceptanceCurve, extract_acceptance_curves, normalized_acceptance_curve
export format_collapse_metrics, format_convergence_result

# Re-export Phase B: Basis Order Ladder
export BasisOrder, ORDER_CONSTANT, ORDER_LINEAR, ORDER_QUADRATIC
export n_parameters, minimum_observations, effective_observations
export OrderSelectionConfig, DEFAULT_ORDER_SELECTION_CONFIG
export ModelFitStatistics, compute_aic, compute_bic
export f_test, compare_aic
export OrderSelectionResult, select_basis_order
export can_upgrade_order, should_downgrade_order
export OrderTransition, OrderLadderState
export lock_order!, unlock_order!, update_ladder!
export format_selection_result, format_ladder_state

# Re-export Phase B: Source Separation
export SourceType, SOURCE_BACKGROUND, SOURCE_OBJECT, SOURCE_VEHICLE, SOURCE_UNKNOWN
export SourceContribution
export background_contribution, object_contribution, vehicle_contribution
export MeasurementAttribution, attribute_measurement
export SourceSeparationConfig, DEFAULT_SOURCE_SEPARATION_CONFIG
export check_background_teachability
export CleanMeasurement, extract_clean_measurement
export SourceSeparationStatistics, teachability_rate
export SourceSeparator, process_measurement!
export format_separation_statistics, format_attribution

# ============================================================================
# Phase C: Online MagSLAM (Real-Time Online Learning + Fleet)
# ============================================================================

# SLAM state contract (augmented state: nav + map + sources)
include("contracts/SlamStateContract.jl")

# Online SLAM scheduler (dual-time-scale inference)
include("slam/OnlineSlamScheduler.jl")

# Online map provider (map query with uncertainty)
include("slam/OnlineMapProvider.jl")

# Online tile updater (information-form updates)
include("slam/OnlineTileUpdater.jl")

# Online source SLAM (dipole tracking and source separation)
include("slam/OnlineSourceSLAM.jl")

# Online manifold collapse tracking (real-time convergence)
include("slam/OnlineManifoldCollapse.jl")

# Online safety controller (NEES monitoring, rollback)
include("slam/OnlineSafetyController.jl")

# Online fleet learning (fleet-based map fusion)
include("slam/OnlineFleetLearning.jl")

# Map persistence for multi-mission support (Phase C)
# Must be after SlamStateContract and SLAM modules due to dependencies
include("map/MapPersistence.jl")

# Re-export Phase C: SLAM State Contract
export SlamMode, SLAM_FROZEN, SLAM_ONLINE, SLAM_SURVEY
export SlamConfig, DEFAULT_SLAM_CONFIG
export is_online_learning, is_source_tracking
export SlamTileState, tile_state_dim, to_tile_data
export SlamSourceState, SOURCE_STATE_DIM
export source_position, source_moment, to_dipole_state
export SlamAugmentedState
export nav_state_dim, map_state_dim, source_state_dim, total_state_dim
export n_sources, n_active_sources, n_probationary_sources
export state_partition
export get_nav_state, get_tile_state, get_source_state
export query_slam_map
export SlamCheckpoint, create_checkpoint, restore_from_checkpoint!

# Re-export Phase C: Online SLAM Scheduler (Step 3)
export KeyframeTrigger, TRIGGER_TIME, TRIGGER_DISTANCE, TRIGGER_ROTATION
export TRIGGER_OBSERVABILITY, TRIGGER_RESIDUAL, TRIGGER_MANUAL
export KeyframePolicyConfig, DEFAULT_KEYFRAME_POLICY_CONFIG
export KeyframeState, KeyframeDecision
export evaluate_keyframe_policy, record_keyframe!, record_observation!
export SchedulerLoopType, LOOP_FAST, LOOP_SLOW, LOOP_IDLE
export OnlineSlamSchedulerConfig, DEFAULT_SCHEDULER_CONFIG
export SchedulerStatistics, DeferredMapUpdate
export OnlineSlamScheduler, informed_jacobian_rank, active_k_histogram_string
export is_learning_enabled, is_source_tracking_enabled
export FastLoopInput, FastLoopResult, process_fast_loop!
export SlowLoopResult, should_run_slow_loop, process_slow_loop!
export avg_fast_loop_time, avg_slow_loop_time
export fast_loop_overrun_rate, slow_loop_overrun_rate
export format_scheduler_statistics

# Re-export Phase C: Online Map Provider (Step 4)
export OnlineMapQueryConfig, DEFAULT_ONLINE_MAP_QUERY_CONFIG
export OnlineMapQueryResult, to_map_query_result
export OnlineMapProvider, DEFAULT_ONLINE_MAP_PROVIDER
export query_online_map
export compute_field_gradient_jacobian, compute_tile_quality_metric
export evaluate_tile_prediction, evaluate_source_contributions
export is_tier2_eligible
export SigmaMapComponents, compute_sigma_map_components
export build_sigma_total_with_map

# Re-export Phase C: Online Tile Updater (Step 5)
export TIER2_MAX_ACTIVE_DIM, JACOBIAN_COL_NORM_THRESHOLD, canonical_informed_dim
export GRADIENT_MIN_SPAN_M, GRADIENT_MIN_OBS
export TIER2_MIN_SPAN_M, TIER2_MIN_SPAN_BOTH, TIER2_MIN_OBS
export TIER2_MIN_INFO_GAIN_RATIO, TIER2_MIN_RANK_QUAD, TIER2_MAX_CROSS_COUPLING
export TIER2_MIN_RELATIVE_COND, TIER2_RELOCK_RELATIVE_COND, TIER2_RELOCK_SPAN_M
export OnlineTileUpdaterConfig, DEFAULT_ONLINE_TILE_UPDATER_CONFIG
export TileUpdateObservation
export TileUpdateRejection, REJECT_NONE, REJECT_LOW_TEACHABILITY
export REJECT_OUTLIER, REJECT_LOW_WEIGHT, REJECT_SINGULAR, REJECT_LOW_GRADIENT
export TileUpdateResult
export OnlineTileUpdater, DEFAULT_ONLINE_TILE_UPDATER
export can_apply_update, compute_tile_jacobian
export apply_batch_update!, apply_update_result!
export SmoothnessPrior, compute_smoothness_information
export is_likely_dipole_signature
export TileUpdaterStatistics, acceptance_rate
export format_updater_statistics

# Re-export Phase C: Online Source SLAM (Step 6)
export SourceDetectionConfig, DEFAULT_SOURCE_DETECTION_CONFIG
export SourcePromotionConfig, DEFAULT_SOURCE_PROMOTION_CONFIG
export SourceUpdateConfig, DEFAULT_SOURCE_UPDATE_CONFIG
export SourceRetirementConfig, DEFAULT_SOURCE_RETIREMENT_CONFIG
export OnlineSourceSLAMConfig, DEFAULT_SOURCE_SLAM_CONFIG
export SourceCandidate, check_promotion
export SourceObservation, source_residual
export compute_source_jacobian, dipole_field_at
export SourceUpdateResult, update_source_state!
export SourceRetirementReason, RETIRE_NONE, RETIRE_TIME_OUT
export RETIRE_LOW_CONTRIBUTION, RETIRE_COVARIANCE_GROWTH, RETIRE_ABSORBED, RETIRE_MANUAL
export check_source_retirement
export OnlineSourceSLAM, SourceProcessingResult
export process_source_observation!, get_source_contributions
export SourceSLAMStatistics, format_source_slam_statistics

# Re-export Phase C: Online Manifold Collapse (Step 7)
export OnlineCollapseConfig, DEFAULT_ONLINE_COLLAPSE_CONFIG
export IncrementalStatistics, get_variance, get_std
export CollapseState, COLLAPSE_INACTIVE, COLLAPSE_LEARNING, COLLAPSE_CONVERGING
export COLLAPSE_CONVERGED, COLLAPSE_STALLED, COLLAPSE_DIVERGING
export OnlineCollapseSnapshot
export OnlineCollapseTracker, reset_tracker!
export OnlineCollapseStatus, update_tracker!
export compute_reduction_rate
export RealTimeCollapseMetrics, get_realtime_metrics
export OnlineConvergenceCriteria, DEFAULT_ONLINE_CONVERGENCE_CRITERIA
export OnlineConvergenceResult, check_online_convergence
export export_to_trajectory
export format_collapse_state, format_online_collapse_status, format_realtime_metrics

# Re-export Phase C: Online Safety Controller (Step 8)
export SafetyControllerConfig, DEFAULT_SAFETY_CONFIG
export SafetyAction, ACTION_NONE, ACTION_WARN, ACTION_THROTTLE
export ACTION_PAUSE_LEARNING, ACTION_ROLLBACK, ACTION_EMERGENCY_STOP
export SafetyAlert, ALERT_NONE, ALERT_NEES_HIGH, ALERT_COVARIANCE_COLLAPSE
export ALERT_COVARIANCE_EXPLOSION, ALERT_DIVERGENCE, ALERT_STALL
export ALERT_TIMING_OVERRUN, ALERT_OUTLIER_BURST
export SafetyMonitorState, SafetyCheckpoint
export SafetyCheckResult, OnlineSafetyController
export check_safety!, get_rollback_checkpoint, execute_rollback!
export SafetyStatistics, get_safety_statistics
export format_safety_statistics, format_safety_result

# Re-export Phase C: Online Fleet Learning (Step 9)
export FleetLearningConfig, DEFAULT_FLEET_LEARNING_CONFIG
export OnlineMapUpdate, OnlineSourceUpdate
export compute_online_tile_quality
export PeerLearningState, FleetLearningState
export OnlineFleetLearning
export generate_updates, FleetUpdatePriority
export FleetFusionResult, process_tile_update!, process_source_update!
export fuse_tile_ci!, fuse_source_ci!, optimize_ci_omega
export FleetConvergenceMetrics, compute_fleet_convergence
export FleetLearningStatistics, get_fleet_statistics

"""
    run_mission(config, world, sensors) -> MissionResult

Execute a complete navigation mission with the given configuration.
"""
function run_mission end

"""
    step!(estimator, measurements) -> StepResult

Perform a single estimation step with new measurements.
"""
function step! end

"""
    export_state(state) -> ExportedState

Export the current state estimate in a portable format.
"""
function export_state end

"""
    validate_config(config) -> ValidationResult

Validate configuration, failing fast on inconsistencies.
"""
function validate_config end

"""
    load_config(path::String) -> NavConfig

Load configuration from TOML file.
"""
function load_config end

# ============================================================================
# Phase E modules (opt-in estimation enhancements)
# ============================================================================

# Fixed-lag smoother for pose refinement
include("estimation/FixedLagSmoother.jl")

# Observability metrics (FIM-based trajectory diagnostics)
include("estimation/ObservabilityMetrics.jl")

# Hybrid estimation (source-aware magnetic updates)
include("estimation/HybridEstimation.jl")

# Information-optimal learning rates
include("mapping/InformationOptimalLearning.jl")

# Basis enrichment (higher-order map basis)
include("map/BasisEnrichment.jl")

# ============================================================================
# Phase F: Information-Geometry–Driven Quadratic Value Extraction
# ============================================================================

# Observability budget (quadratic-block FIM for Tier-2 gating)
include("map/ObservabilityBudget.jl")

# Trajectory excitation generators (scenario utilities included in NavCore
# because trajectory_excitation_score depends on core FIM computation)
include("scenarios/TrajectoryExcitation.jl")

# Tier-2 efficacy metrics (d=15 vs d=8 comparison)
include("map/Tier2EfficacyMetrics.jl")

# Re-export Phase F: Observability Budget
export compute_quadratic_fim, observability_scalar, ObservabilityBudget

# Re-export Phase F: Trajectory Excitation
export lissajous_trajectory, box_diagonals_trajectory, spiral_trajectory
export altitude_modulated, trajectory_excitation_score

# Re-export Phase F: Tier-2 Efficacy Metrics
export EvaluationScope, SCOPE_UNLOCKED, SCOPE_BOUNDARY, SCOPE_GLOBAL
export Tier2EfficacyReport, TileProbe
export evaluate_tier2_efficacy, unlock_coverage

# ============================================================================
# Phase G: Source Localization (Dipole vs Background)
# ============================================================================

# Source coupling contract (Step 1)
include("contracts/SourceContract.jl")

# Source detection front-end (Step 3)
include("slam/SourceDetectionFrontEnd.jl")

# Source observability (Step 4)
include("map/SourceObservability.jl")

# Source tracker (Step 5)
include("slam/SourceTracker.jl")

# Source-coupled update (Step 6)
include("slam/SourceCoupledUpdate.jl")

# Source refinement (Step 7)
include("slam/SourceRefinement.jl")

# Source metrics (Step 8)
include("testing/source_metrics.jl")

# Source safety controller (Step 10)
include("slam/SourceSafetyController.jl")

# Source persistence (Step 10)
include("slam/SourcePersistence.jl")

# Phase G+: Source Localization Maturation
include("contracts/SourceMapSeparationContract.jl")  # Step 1: separation contract
include("slam/SourceFirstResidualRouter.jl")          # Step 4: residual router

# Re-export Phase G: Source Contract (Step 1)
export SourceCouplingMode, SOURCE_SHADOW, SOURCE_COV_ONLY, SOURCE_SUBTRACT
export SourceCouplingGates, all_gates_pass, n_gates_passing
export SourceGateThresholds, DEFAULT_SOURCE_GATE_THRESHOLDS
export SourceCouplingConfig, DEFAULT_SOURCE_COUPLING_CONFIG
export evaluate_coupling_gates, effective_coupling_mode
export apply_source_coupling!

# Re-export Phase G: Detection Front-End (Step 3)
export DetectionEvent, SourceDetectionFrontEnd
export detect_anomalies, cluster_detections, associate_or_create!

# Re-export Phase G: Source Observability (Step 4)
export SourceObservabilityBudget
export compute_source_fim, source_observability_scalar
export meets_observability_gate

# Re-export Phase G: Source Tracker (Step 5)
export SourceTrackStatus
export TRACK_CANDIDATE, TRACK_PROVISIONAL, TRACK_CONFIRMED
export TRACK_LOCKED, TRACK_RETIRED, TRACK_DEMOTED
export SourceTrack, SourceTrackerConfig, DEFAULT_SOURCE_TRACKER_CONFIG
export SourceTrackingResult, SourceTracker
export refit_track!

# Re-export Phase G: Source-Coupled Update (Step 6)
export SourceCoupledResidual
export compute_source_coupled_residual
export source_coupled_tile_update!, source_coupled_nav_update!

# Re-export Phase G: Source Refinement (Step 7)
export RefinementConfig, DEFAULT_REFINEMENT_CONFIG
export RefinementResult
export refine_sources!, model_selection_aic

# Re-export Phase G: Source Metrics (Step 8)
export SourceLocalizationMetrics, ScenarioSourceMetrics
export source_nees, source_position_nees
export evaluate_source_metrics

# Re-export Phase G: Source Safety (Step 10)
export SourceSafetyConfig, DEFAULT_SOURCE_SAFETY_CONFIG
export SourceSafetyAction
export SOURCE_SAFETY_NONE, SOURCE_SAFETY_DEMOTE, SOURCE_SAFETY_ROLLBACK, SOURCE_SAFETY_THROTTLE
export SourceSafetyController
export check_source_safety!
export checkpoint_sources, restore_sources!
export serialize_source_tracks, deserialize_source_tracks

# Re-export Phase G: Source Persistence (Step 10)
export SourceCheckpoint
export create_source_checkpoint, restore_source_checkpoint!
export save_source_tracks, load_source_tracks!

# Re-export Phase G+: Source-Map Separation Contract (Step 1)
export ResidualOwner, OWNER_NAV, OWNER_TILE, OWNER_SOURCE, OWNER_UNATTRIBUTED
export MeasurementAttribution, SeparationConfig
export attribute_residual, should_freeze_tile, should_freeze_tile_spatial, partition_residual

# Re-export Phase G+: Enhanced Observability (Step 3)
export spatial_excitation_score, meets_excitation_gate, incremental_fim_update!

# Re-export Phase G+: Residual Router (Step 4)
export ResidualRoutingResult, SourceFirstResidualRouter
export route_residual!, update_freeze_zones!

# Re-export Phase G+: Adaptive Inflation (Step 5)
export compute_adaptive_inflation, apply_adaptive_inflation!

# Re-export Phase G+: Lifecycle Hysteresis (Step 6)
export LifecycleHysteresis, DEFAULT_LIFECYCLE_HYSTERESIS
export gate_score, should_promote, should_demote

# Re-export Phase G+: Conditioned Metrics (Step 8)
export ConditionedSourceMetrics, snr_regime, evaluate_conditioned_metrics

end # module UrbanNav
