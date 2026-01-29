# ============================================================================
# Map Versioning and Rollback (Phase B)
# ============================================================================
#
# Provides version control for learned maps with rollback capability.
#
# Purpose: "We can prove learning didn't corrupt the map."
#
# Key Features:
# 1. Semantic versioning with full provenance
# 2. Checkpoint snapshots before learning updates
# 3. Rollback to last known good version
# 4. Canary validation (automated health checking)
#
# Version Chain:
#   v0 (frozen) → v1 (mission_2) → v2 (mission_3) → ...
#
# Each version stores:
# - Map coefficients and covariance
# - Provenance (what missions contributed)
# - Validation metrics (RMSE, NEES at creation time)
# - Timestamp and update statistics
#
# ============================================================================

using LinearAlgebra
using Dates
using SHA
using Statistics: mean, std

# ============================================================================
# Version Metadata
# ============================================================================

"""
    MapProvenance

Tracks the origin and history of map updates.

# Fields
- `parent_version::Int`: Version this was derived from (-1 for initial)
- `mission_ids::Vector{String}`: Missions that contributed updates
- `update_count::Int`: Number of measurement updates applied
- `scenario_hash::UInt64`: Hash of scenario for reproducibility
- `method::Symbol`: How this version was created (:frozen, :learning, :rollback)
- `notes::String`: Human-readable notes
"""
struct MapProvenance
    parent_version::Int
    mission_ids::Vector{String}
    update_count::Int
    scenario_hash::UInt64
    method::Symbol
    notes::String
end

"""
    MapCheckpoint

Immutable snapshot of map state at a point in time.

# Fields
- `version::Int`: Version number (0 = frozen baseline)
- `timestamp::DateTime`: When this checkpoint was created
- `coefficients::Vector{Float64}`: Map coefficients α
- `covariance::Matrix{Float64}`: Coefficient covariance P_α
- `tile_id::MapTileID`: Which tile this is for
- `observation_count::Int`: Total observations incorporated
- `provenance::MapProvenance`: What created this version

# Immutability
Checkpoints are immutable once created. To modify, create a new version.
"""
struct MapCheckpoint
    version::Int
    timestamp::DateTime
    coefficients::Vector{Float64}
    covariance::Matrix{Float64}
    tile_id::MapTileID
    observation_count::Int
    provenance::MapProvenance
    active_dim::Int  # Number of active (learned) coefficients at checkpoint time.
                     # Coefficients (active_dim+1):end are frozen/inactive.
                     # Used to re-enforce cross-block zeros on load.
end

"""Create provenance for frozen (Phase A) map."""
function frozen_provenance(mission_id::String)
    MapProvenance(-1, [mission_id], 0, UInt64(0), :frozen, "Initial frozen map")
end

"""Create provenance for learned update."""
function learning_provenance(parent::MapProvenance, mission_id::String, update_count::Int;
                             scenario_hash::UInt64 = UInt64(0), notes::String = "")
    MapProvenance(
        parent.parent_version + 1,  # This will be corrected when checkpoint is created
        vcat(parent.mission_ids, [mission_id]),
        parent.update_count + update_count,
        scenario_hash,
        :learning,
        notes
    )
end

"""Create provenance for rollback."""
function rollback_provenance(target_version::Int, reason::String)
    MapProvenance(
        target_version,
        String[],
        0,
        UInt64(0),
        :rollback,
        "Rollback to v$target_version: $reason"
    )
end

# ============================================================================
# Validation Metrics
# ============================================================================

"""
    ValidationMetrics

Metrics captured when validating a map version.

# Fields
- `rmse::Float64`: Root mean square error [m] on validation trajectory
- `nees_mean::Float64`: Mean NEES (should be ≈1 for calibrated filter)
- `nees_std::Float64`: NEES standard deviation
- `consistency::Float64`: Fraction of NEES values in [0.1, 10]
- `coverage_fraction::Float64`: Fraction of trajectory in map coverage
- `timestamp::DateTime`: When validation was performed
- `trajectory_length::Int`: Number of points in validation trajectory

# Health Thresholds (with justification)
- RMSE threshold: Typically 20% degradation triggers concern
- NEES range: [0.1, 10] is generous; [0.5, 2.0] is ideal
- Consistency: >80% of NEES values should be in healthy range
"""
struct ValidationMetrics
    rmse::Float64
    nees_mean::Float64
    nees_std::Float64
    consistency::Float64
    coverage_fraction::Float64
    timestamp::DateTime
    trajectory_length::Int
end

"""Create validation metrics from arrays."""
function ValidationMetrics(;
    errors::Vector{Float64},
    nees_values::Vector{Float64},
    coverage_flags::Vector{Bool}
)
    rmse = sqrt(mean(errors.^2))
    nees_mean = mean(nees_values)
    nees_std = std(nees_values)
    consistency = mean(0.1 .< nees_values .< 10.0)
    coverage_fraction = mean(coverage_flags)

    ValidationMetrics(
        rmse, nees_mean, nees_std, consistency, coverage_fraction,
        now(UTC), length(errors)
    )
end

"""
    ValidationResult

Result of validating a map version.

# Fields
- `passed::Bool`: Whether validation passed
- `metrics::ValidationMetrics`: Computed metrics
- `baseline_metrics::Union{Nothing, ValidationMetrics}`: Baseline for comparison
- `degradation_pct::Float64`: RMSE degradation vs baseline (0 if no baseline)
- `failure_reasons::Vector{Symbol}`: Why validation failed (if it did)
"""
struct ValidationResult
    passed::Bool
    metrics::ValidationMetrics
    baseline_metrics::Union{Nothing, ValidationMetrics}
    degradation_pct::Float64
    failure_reasons::Vector{Symbol}
end

# ============================================================================
# Validation Configuration
# ============================================================================

"""
    ValidationConfig

Configuration for map validation / canary testing.

# Fields
- `max_rmse_degradation_pct::Float64`: Max allowed RMSE increase (default: 20%)
- `nees_min::Float64`: Minimum acceptable NEES (default: 0.1)
- `nees_max::Float64`: Maximum acceptable NEES (default: 5.0)
- `min_consistency::Float64`: Minimum NEES consistency (default: 0.7)
- `min_coverage::Float64`: Minimum coverage fraction (default: 0.8)

# Threshold Justifications

**max_rmse_degradation_pct = 20%**:
A 20% increase in RMSE is operationally significant but not catastrophic.
Smaller thresholds (e.g., 10%) would trigger more false positives.
Larger thresholds risk accepting corrupted maps.

**NEES bounds [0.1, 5.0]**:
- NEES < 0.1: Filter is overconfident (covariance too small)
- NEES > 5.0: Filter is underconfident or diverging
- Ideal is NEES ≈ 1.0

**min_consistency = 0.7**:
70% of NEES values in healthy range allows for occasional transients
while catching systematic issues.

**min_coverage = 0.8**:
Validation trajectory should be 80% within map coverage to be meaningful.
"""
struct ValidationConfig
    max_rmse_degradation_pct::Float64
    nees_min::Float64
    nees_max::Float64
    min_consistency::Float64
    min_coverage::Float64

    function ValidationConfig(;
        max_rmse_degradation_pct::Float64 = 20.0,
        nees_min::Float64 = 0.1,
        nees_max::Float64 = 5.0,
        min_consistency::Float64 = 0.7,
        min_coverage::Float64 = 0.8
    )
        @assert max_rmse_degradation_pct > 0
        @assert 0 < nees_min < nees_max
        @assert 0 < min_consistency <= 1
        @assert 0 < min_coverage <= 1
        new(max_rmse_degradation_pct, nees_min, nees_max, min_consistency, min_coverage)
    end
end

const DEFAULT_VALIDATION_CONFIG = ValidationConfig()

"""
    validate_metrics(metrics::ValidationMetrics, config::ValidationConfig;
                    baseline::Union{Nothing, ValidationMetrics} = nothing)

Validate metrics against configuration thresholds.

Returns ValidationResult with pass/fail and reasons.
"""
function validate_metrics(metrics::ValidationMetrics, config::ValidationConfig;
                         baseline::Union{Nothing, ValidationMetrics} = nothing)
    failure_reasons = Symbol[]
    degradation_pct = 0.0

    # Check RMSE degradation if baseline provided
    if baseline !== nothing
        degradation_pct = (metrics.rmse - baseline.rmse) / baseline.rmse * 100
        if degradation_pct > config.max_rmse_degradation_pct
            push!(failure_reasons, :rmse_degradation)
        end
    end

    # Check NEES bounds
    if metrics.nees_mean < config.nees_min
        push!(failure_reasons, :nees_too_low)
    end
    if metrics.nees_mean > config.nees_max
        push!(failure_reasons, :nees_too_high)
    end

    # Check consistency
    if metrics.consistency < config.min_consistency
        push!(failure_reasons, :low_consistency)
    end

    # Check coverage
    if metrics.coverage_fraction < config.min_coverage
        push!(failure_reasons, :low_coverage)
    end

    passed = isempty(failure_reasons)

    return ValidationResult(passed, metrics, baseline, degradation_pct, failure_reasons)
end

# ============================================================================
# Version History Manager
# ============================================================================

"""
    MapVersionHistory

Manages version history for a single tile with rollback support.

# Fields
- `tile_id::MapTileID`: Which tile this history is for
- `checkpoints::Vector{MapCheckpoint}`: All checkpoints (append-only)
- `current_version::Int`: Currently active version
- `validations::Dict{Int, ValidationResult}`: Validation results by version
- `rollback_log::Vector{Tuple{DateTime, Int, Int, String}}`: (time, from, to, reason)

# Invariants
- Checkpoints are append-only (never modified or deleted)
- current_version points to an existing checkpoint
- Version 0 is always the frozen baseline (if present)
"""
mutable struct MapVersionHistory
    tile_id::MapTileID
    checkpoints::Vector{MapCheckpoint}
    current_version::Int
    validations::Dict{Int, ValidationResult}
    rollback_log::Vector{Tuple{DateTime, Int, Int, String}}
end

"""Create empty version history for a tile."""
function MapVersionHistory(tile_id::MapTileID)
    MapVersionHistory(
        tile_id,
        MapCheckpoint[],
        -1,  # No current version yet
        Dict{Int, ValidationResult}(),
        Tuple{DateTime, Int, Int, String}[]
    )
end

"""Create version history initialized with frozen baseline."""
function MapVersionHistory(tile::MapTileData, mission_id::String)
    history = MapVersionHistory(tile.id)

    # Create frozen baseline checkpoint
    checkpoint = MapCheckpoint(
        0,  # Version 0 = frozen
        now(UTC),
        copy(tile.coefficients),
        copy(tile.covariance),
        tile.id,
        tile.observation_count,
        frozen_provenance(mission_id)
    )

    push!(history.checkpoints, checkpoint)
    history.current_version = 0

    return history
end

"""Get current checkpoint."""
function current_checkpoint(history::MapVersionHistory)
    if history.current_version < 0 || isempty(history.checkpoints)
        return nothing
    end
    # Find checkpoint with current version
    for cp in history.checkpoints
        if cp.version == history.current_version
            return cp
        end
    end
    return nothing
end

"""Get checkpoint by version number."""
function get_checkpoint(history::MapVersionHistory, version::Int)
    for cp in history.checkpoints
        if cp.version == version
            return cp
        end
    end
    return nothing
end

"""Get all version numbers in history."""
function list_versions(history::MapVersionHistory)
    return [cp.version for cp in history.checkpoints]
end

"""Get latest version number."""
function latest_version(history::MapVersionHistory)
    if isempty(history.checkpoints)
        return -1
    end
    return maximum(cp.version for cp in history.checkpoints)
end

"""
    create_checkpoint!(history::MapVersionHistory, tile::MapTileData,
                       mission_id::String, update_count::Int;
                       scenario_hash::UInt64 = UInt64(0),
                       notes::String = "")

Create a new checkpoint from current tile state.

Returns the new checkpoint.
"""
function create_checkpoint!(history::MapVersionHistory, tile::MapTileData,
                           mission_id::String, update_count::Int;
                           scenario_hash::UInt64 = UInt64(0),
                           notes::String = "")
    # Get parent provenance
    current_cp = current_checkpoint(history)
    if current_cp === nothing
        parent_prov = frozen_provenance("unknown")
    else
        parent_prov = current_cp.provenance
    end

    # Create new version number
    new_version = latest_version(history) + 1

    # Create provenance
    provenance = MapProvenance(
        history.current_version,
        vcat(parent_prov.mission_ids, [mission_id]),
        parent_prov.update_count + update_count,
        scenario_hash,
        :learning,
        notes
    )

    # Create checkpoint
    checkpoint = MapCheckpoint(
        new_version,
        now(UTC),
        copy(tile.coefficients),
        copy(tile.covariance),
        tile.id,
        tile.observation_count,
        provenance
    )

    # Append to history (never modify existing)
    push!(history.checkpoints, checkpoint)
    history.current_version = new_version

    return checkpoint
end

"""
    rollback!(history::MapVersionHistory, target_version::Int, reason::String)

Rollback to a previous version.

Returns the target checkpoint, or nothing if version not found.

# Note
This does NOT delete newer checkpoints - they remain in history for audit.
It only changes current_version pointer.
"""
function rollback!(history::MapVersionHistory, target_version::Int, reason::String)
    target_cp = get_checkpoint(history, target_version)
    if target_cp === nothing
        return nothing
    end

    # Log the rollback
    push!(history.rollback_log, (
        now(UTC),
        history.current_version,
        target_version,
        reason
    ))

    # Update current version
    history.current_version = target_version

    return target_cp
end

"""
    record_validation!(history::MapVersionHistory, version::Int, result::ValidationResult)

Record validation result for a version.
"""
function record_validation!(history::MapVersionHistory, version::Int, result::ValidationResult)
    history.validations[version] = result
end

"""
    get_validation(history::MapVersionHistory, version::Int)

Get validation result for a version (or nothing if not validated).
"""
function get_validation(history::MapVersionHistory, version::Int)
    return get(history.validations, version, nothing)
end

"""
    find_last_good_version(history::MapVersionHistory)

Find the most recent version that passed validation.

Returns version number, or -1 if none found.
"""
function find_last_good_version(history::MapVersionHistory)
    # Search from newest to oldest
    versions = sort(list_versions(history), rev=true)

    for v in versions
        result = get_validation(history, v)
        if result !== nothing && result.passed
            return v
        end
    end

    # If no validated versions, return frozen baseline if it exists
    if 0 in versions
        return 0
    end

    return -1
end

"""
    auto_rollback_if_degraded!(history::MapVersionHistory, config::ValidationConfig)

Automatically rollback to last good version if current version is degraded.

Returns (rolled_back::Bool, target_version::Int, reason::String)
"""
function auto_rollback_if_degraded!(history::MapVersionHistory, config::ValidationConfig)
    current_val = get_validation(history, history.current_version)

    if current_val === nothing
        return (false, history.current_version, "No validation for current version")
    end

    if current_val.passed
        return (false, history.current_version, "Current version is healthy")
    end

    # Find last good version
    target = find_last_good_version(history)

    if target < 0
        return (false, history.current_version, "No validated good version to rollback to")
    end

    if target == history.current_version
        return (false, history.current_version, "Already at last good version")
    end

    # Perform rollback
    reason = "Auto-rollback due to: $(join(string.(current_val.failure_reasons), ", "))"
    rollback!(history, target, reason)

    return (true, target, reason)
end

# ============================================================================
# Convenience Functions
# ============================================================================

"""
    restore_checkpoint(checkpoint::MapCheckpoint)

Create a MapTileData from a checkpoint.
"""
function restore_checkpoint(checkpoint::MapCheckpoint)
    return MapTileData(
        checkpoint.tile_id,
        Vec3Map(0, 0, 0),  # Center not stored in checkpoint
        copy(checkpoint.coefficients),
        copy(checkpoint.covariance),
        checkpoint.observation_count
    )
end

"""
    compute_scenario_hash(seed::Int, trajectory_length::Int, noise_level::Float64)

Compute a hash for scenario reproducibility.
"""
function compute_scenario_hash(seed::Int, trajectory_length::Int, noise_level::Float64)
    data = "seed=$seed,traj=$trajectory_length,noise=$noise_level"
    hash_bytes = sha256(Vector{UInt8}(data))
    return reinterpret(UInt64, hash_bytes[1:8])[1]
end

"""
    format_version_history(history::MapVersionHistory)

Format version history as human-readable string.
"""
function format_version_history(history::MapVersionHistory)
    lines = String[]
    push!(lines, "Map Version History for Tile $(history.tile_id)")
    push!(lines, "=" ^ 50)
    push!(lines, "Current version: $(history.current_version)")
    push!(lines, "")

    for cp in history.checkpoints
        status = cp.version == history.current_version ? " [CURRENT]" : ""
        val = get_validation(history, cp.version)
        val_str = val === nothing ? "not validated" : (val.passed ? "✓ PASS" : "✗ FAIL")

        push!(lines, "v$(cp.version)$status")
        push!(lines, "  Created: $(cp.timestamp)")
        push!(lines, "  Method: $(cp.provenance.method)")
        push!(lines, "  Updates: $(cp.provenance.update_count)")
        push!(lines, "  Validation: $val_str")
        if val !== nothing && !val.passed
            push!(lines, "  Failures: $(join(string.(val.failure_reasons), ", "))")
        end
        push!(lines, "")
    end

    if !isempty(history.rollback_log)
        push!(lines, "Rollback Log:")
        push!(lines, "-" ^ 50)
        for (time, from_v, to_v, reason) in history.rollback_log
            push!(lines, "  $time: v$from_v → v$to_v")
            push!(lines, "    Reason: $reason")
        end
    end

    return join(lines, "\n")
end

# ============================================================================
# Exports
# ============================================================================

export MapCheckpoint, MapProvenance
export frozen_provenance, learning_provenance, rollback_provenance
export ValidationMetrics, ValidationResult, ValidationConfig
export DEFAULT_VALIDATION_CONFIG, validate_metrics
export MapVersionHistory
export current_checkpoint, get_checkpoint, list_versions, latest_version
export create_checkpoint!, rollback!, record_validation!, get_validation
export find_last_good_version, auto_rollback_if_degraded!
export restore_checkpoint, compute_scenario_hash, format_version_history
