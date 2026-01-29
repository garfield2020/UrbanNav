# ============================================================================
# Dipole Feature Types
# ============================================================================
#
# Ported from AUV-Navigation/src/features.jl
#
# Defines the dipole feature state, candidate, node, and registry for
# persistent magnetic anomaly tracking. Integrates with existing
# MagneticDipole physics and chi-square gating.
#
# Feature lifecycle: CANDIDATE → ACTIVE → (RETIRED | DEMOTED)
# ============================================================================

using LinearAlgebra
using StaticArrays

# ============================================================================
# Feature State
# ============================================================================

"""
    DipoleFeatureState

State vector for a magnetic dipole feature.

Wraps position and moment for factor graph integration.
Uses existing MagneticDipole from physics.jl for field computation.
"""
struct DipoleFeatureState
    position::SVector{3, Float64}  # [x, y, z] in meters (world frame)
    moment::SVector{3, Float64}    # [mx, my, mz] in A·m²
end

# Convenience constructors
DipoleFeatureState(pos::AbstractVector, mom::AbstractVector) =
    DipoleFeatureState(SVector{3}(pos), SVector{3}(mom))

"""Zero-initialized feature state."""
DipoleFeatureState() = DipoleFeatureState(zeros(SVector{3}), zeros(SVector{3}))

"""Feature state dimension (6 DOF: 3 position + 3 moment)."""
const DIPOLE_FEATURE_DIM = 6

"""Pack feature state into vector."""
function to_state_vector(f::DipoleFeatureState)
    return vcat(f.position, f.moment)
end

"""Unpack vector into feature state."""
function from_state_vector(::Type{DipoleFeatureState}, v::AbstractVector)
    @assert length(v) == DIPOLE_FEATURE_DIM
    return DipoleFeatureState(SVector{3}(v[1:3]), SVector{3}(v[4:6]))
end

"""Convert to MagneticDipole for physics calculations."""
function to_magnetic_dipole(f::DipoleFeatureState)
    MagneticDipole(f.position, f.moment)
end

"""Create from MagneticDipole."""
function DipoleFeatureState(d::MagneticDipole)
    DipoleFeatureState(d.position, d.moment)
end

# ============================================================================
# Feature Lifecycle
# ============================================================================

"""
    DipoleLifecycleState

Lifecycle state for dipole features.
"""
@enum DipoleLifecycleState begin
    DIPOLE_CANDIDATE  # Awaiting promotion (accumulating observations)
    DIPOLE_ACTIVE     # Active in factor graph
    DIPOLE_RETIRED    # Absorbed into map or no longer observed
    DIPOLE_DEMOTED    # Failed validation, removed
end

# ============================================================================
# Feature Node (Factor Graph)
# ============================================================================

"""
    DipoleFeatureNode

Node in the factor graph representing a persistent magnetic feature.

Tracks state estimate, uncertainty, and lifecycle metadata.
"""
mutable struct DipoleFeatureNode
    id::Int
    state::DipoleFeatureState
    covariance::Matrix{Float64}  # 6×6

    # Lifecycle tracking
    lifecycle::DipoleLifecycleState
    created_time::Float64
    last_observed::Float64
    support_count::Int
    residual_reduction::Float64  # 0-1, fraction of residual explained

    # Promotion gate record (for auditing)
    promotion_gamma::Float64      # γ at promotion
    promotion_confidence::Float64 # nav confidence at promotion
end

"""Create a new candidate feature node."""
function DipoleFeatureNode(id::Int, state::DipoleFeatureState, covariance::Matrix{Float64};
                           created_time::Float64 = 0.0)
    @assert size(covariance) == (DIPOLE_FEATURE_DIM, DIPOLE_FEATURE_DIM)
    return DipoleFeatureNode(
        id, state, covariance,
        DIPOLE_CANDIDATE, created_time, created_time,
        0, 0.0,  # support_count, residual_reduction
        0.0, 0.0 # promotion_gamma, promotion_confidence
    )
end

"""Create candidate with default covariance."""
function DipoleFeatureNode(id::Int, state::DipoleFeatureState; created_time::Float64 = 0.0)
    # Default covariance: 10m position, 100 A·m² moment uncertainty
    default_cov = Diagonal([10.0, 10.0, 10.0, 100.0, 100.0, 100.0].^2)
    return DipoleFeatureNode(id, state, Matrix(default_cov); created_time = created_time)
end

# ============================================================================
# Feature Candidate (Pre-Promotion)
# ============================================================================

"""
    DipoleFeatureCandidate

Candidate feature before promotion to full DipoleFeatureNode.

Tracks measurements and statistics needed for promotion decision.
Must pass all gates before becoming a DipoleFeatureNode:
1. γ gate: χ² strong gate exceeded persistently
2. Temporal persistence: exists for ≥ T seconds
3. Spatial compactness: cluster radius < R
4. Nav confidence: confidence ≥ MEDIUM during lifetime
"""
mutable struct DipoleFeatureCandidate
    id::Int

    # Accumulated measurements
    positions::Vector{SVector{3, Float64}}    # measurement positions
    residuals::Vector{SVector{3, Float64}}    # field residuals at each position
    times::Vector{Float64}                     # timestamps
    gammas::Vector{Float64}                    # γ values at each measurement
    confidences::Vector{Float64}              # nav confidence at each measurement

    # Statistics (updated incrementally)
    centroid::SVector{3, Float64}
    cluster_radius::Float64
    mean_gamma::Float64
    min_confidence::Float64

    # Creation time
    created_time::Float64
end

"""Create empty feature candidate."""
function DipoleFeatureCandidate(id::Int; created_time::Float64 = 0.0)
    return DipoleFeatureCandidate(
        id,
        SVector{3, Float64}[],
        SVector{3, Float64}[],
        Float64[],
        Float64[],
        Float64[],
        zeros(SVector{3}),
        0.0,
        0.0,
        1.0,  # min_confidence starts high
        created_time
    )
end

"""Add measurement to candidate."""
function add_candidate_measurement!(c::DipoleFeatureCandidate, pos::AbstractVector,
                                    residual::AbstractVector, t::Float64,
                                    gamma::Float64, confidence::Float64)
    push!(c.positions, SVector{3}(pos))
    push!(c.residuals, SVector{3}(residual))
    push!(c.times, t)
    push!(c.gammas, gamma)
    push!(c.confidences, confidence)

    # Update statistics
    update_candidate_statistics!(c)
end

"""Update candidate statistics from accumulated measurements."""
function update_candidate_statistics!(c::DipoleFeatureCandidate)
    n = length(c.positions)
    if n == 0
        return
    end

    # Centroid
    c.centroid = sum(c.positions) / n

    # Cluster radius (max distance from centroid)
    if n > 1
        c.cluster_radius = maximum(norm(p - c.centroid) for p in c.positions)
    else
        c.cluster_radius = 0.0
    end

    # Mean gamma
    c.mean_gamma = sum(c.gammas) / n

    # Minimum confidence during lifetime
    c.min_confidence = minimum(c.confidences)
end

"""Duration of candidate existence."""
function candidate_duration(c::DipoleFeatureCandidate)
    isempty(c.times) && return 0.0
    return maximum(c.times) - c.created_time
end

"""Number of supporting measurements."""
candidate_support_count(c::DipoleFeatureCandidate) = length(c.positions)

# ============================================================================
# Promotion Gates (Configuration)
# ============================================================================

"""
    PromotionGateConfig

Configuration for promotion gate thresholds.

Features must pass all gates to be promoted from candidate to active.
"""
struct PromotionGateConfig
    gamma_threshold::Float64       # Mean γ threshold (χ²(3, 0.999) = 16.266)
    min_duration::Float64          # Minimum existence time (seconds)
    max_cluster_radius::Float64    # Maximum spatial extent (meters)
    min_confidence::Float64        # Minimum navigation confidence (0-1)
    min_residual_reduction::Float64  # Minimum residual reduction from fit (0-1)
    min_moment::Float64            # Minimum plausible moment (A·m²)
    max_moment::Float64            # Maximum plausible moment (A·m²)
end

function PromotionGateConfig(;
    gamma_threshold::Real = 16.266,    # χ²(3, 0.999)
    min_duration::Real = 2.0,          # seconds
    max_cluster_radius::Real = 5.0,    # meters
    min_confidence::Real = 0.5,        # MEDIUM confidence
    min_residual_reduction::Real = 0.6, # 60%
    min_moment::Real = 0.1,            # A·m² (small submunition)
    max_moment::Real = 1000.0          # A·m² (large bomb)
)
    PromotionGateConfig(
        Float64(gamma_threshold),
        Float64(min_duration),
        Float64(max_cluster_radius),
        Float64(min_confidence),
        Float64(min_residual_reduction),
        Float64(min_moment),
        Float64(max_moment)
    )
end

const DEFAULT_PROMOTION_GATE_CONFIG = PromotionGateConfig()

# ============================================================================
# Dipole Registry
# ============================================================================

"""
    DipoleFeatureRegistry

Manages feature candidates and active features.

Responsibilities:
- Track candidates awaiting promotion
- Track active features in the graph
- Enforce lifecycle rules
- Prevent map pollution
"""
mutable struct DipoleFeatureRegistry
    # Counters
    next_candidate_id::Int
    next_feature_id::Int

    # Storage
    candidates::Dict{Int, DipoleFeatureCandidate}
    features::Dict{Int, DipoleFeatureNode}

    # Retired/demoted (for auditing)
    retired::Vector{DipoleFeatureNode}
    demoted::Vector{DipoleFeatureNode}

    # Configuration
    feature_enabled::Bool  # Feature flag for safe integration
    promotion_config::PromotionGateConfig
end

"""Create empty feature registry."""
function DipoleFeatureRegistry(;
    enabled::Bool = false,
    promotion_config::PromotionGateConfig = DEFAULT_PROMOTION_GATE_CONFIG
)
    return DipoleFeatureRegistry(
        1, 1,
        Dict{Int, DipoleFeatureCandidate}(),
        Dict{Int, DipoleFeatureNode}(),
        DipoleFeatureNode[],
        DipoleFeatureNode[],
        enabled,
        promotion_config
    )
end

"""Check if features are enabled."""
dipoles_enabled(r::DipoleFeatureRegistry) = r.feature_enabled

"""Enable/disable feature integration."""
function set_dipoles_enabled!(r::DipoleFeatureRegistry, enabled::Bool)
    r.feature_enabled = enabled
end

"""Create new candidate in registry."""
function create_feature_candidate!(r::DipoleFeatureRegistry, time::Float64)
    !r.feature_enabled && return nothing

    id = r.next_candidate_id
    r.next_candidate_id += 1
    candidate = DipoleFeatureCandidate(id; created_time = time)
    r.candidates[id] = candidate
    return candidate
end

"""Get candidate by ID."""
get_feature_candidate(r::DipoleFeatureRegistry, id::Int) = get(r.candidates, id, nothing)

"""Get feature by ID."""
get_dipole_feature(r::DipoleFeatureRegistry, id::Int) = get(r.features, id, nothing)

"""Number of active candidates."""
n_feature_candidates(r::DipoleFeatureRegistry) = length(r.candidates)

"""Number of active features."""
n_dipole_features(r::DipoleFeatureRegistry) = length(r.features)

"""Remove candidate (after promotion or demotion)."""
function remove_feature_candidate!(r::DipoleFeatureRegistry, id::Int)
    delete!(r.candidates, id)
end

"""Promote candidate to feature."""
function promote_feature_candidate!(r::DipoleFeatureRegistry, candidate::DipoleFeatureCandidate,
                                    state::DipoleFeatureState, covariance::Matrix{Float64},
                                    gamma::Float64, confidence::Float64)
    !r.feature_enabled && return nothing

    # Create feature node
    feature_id = r.next_feature_id
    r.next_feature_id += 1

    feature = DipoleFeatureNode(feature_id, state, covariance;
                                created_time = candidate.created_time)
    feature.lifecycle = DIPOLE_ACTIVE
    feature.support_count = candidate_support_count(candidate)
    feature.promotion_gamma = gamma
    feature.promotion_confidence = confidence

    # Add to registry
    r.features[feature_id] = feature

    # Remove candidate
    remove_feature_candidate!(r, candidate.id)

    return feature
end

"""Retire dipole (absorbed by basis or no longer relevant)."""
function retire_dipole_feature!(r::DipoleFeatureRegistry, id::Int)
    dipole = get_dipole_feature(r, id)
    isnothing(dipole) && return

    dipole.lifecycle = DIPOLE_RETIRED
    push!(r.retired, dipole)
    delete!(r.features, id)
end

"""Demote dipole (validation failed)."""
function demote_dipole_feature!(r::DipoleFeatureRegistry, id::Int)
    dipole = get_dipole_feature(r, id)
    isnothing(dipole) && return

    dipole.lifecycle = DIPOLE_DEMOTED
    push!(r.demoted, dipole)
    delete!(r.features, id)
end

# ============================================================================
# Dipole Field Computation (wrapper using physics.jl)
# ============================================================================

"""
    feature_field(pos, feature)

Compute magnetic field from dipole feature at position pos.
Delegates to physics.jl MagneticDipole field computation.
"""
function feature_field(pos::AbstractVector, feature::DipoleFeatureState)
    dipole = to_magnetic_dipole(feature)
    return field(dipole, SVector{3}(pos))
end

"""
    feature_field_jacobian(pos, feature)

Compute Jacobian of dipole field w.r.t. feature state [position, moment].

Returns 3×6 matrix: ∂B/∂[p, m]
"""
function feature_field_jacobian(pos::AbstractVector, feature::DipoleFeatureState)
    r = SVector{3}(pos) - feature.position
    r_mag = norm(r)

    # Avoid singularity
    if r_mag < 0.01
        return zeros(SMatrix{3, 6})
    end

    r_hat = r / r_mag
    m = feature.moment

    # Jacobian w.r.t. moment (analytic)
    # ∂B/∂m = μ₀/(4π) * [3 * r̂ ⊗ r̂ - I] / r³
    dB_dm = μ₀_4π * (3 * r_hat * r_hat' - I) / r_mag^3

    # Jacobian w.r.t. position (finite differences for stability)
    ε = 1e-6
    dB_dp = zeros(3, 3)
    B0 = feature_field(pos, feature)
    for i in 1:3
        dp = zeros(SVector{3})
        dp = setindex(dp, ε, i)
        f_perturbed = DipoleFeatureState(feature.position + dp, feature.moment)
        B_perturbed = feature_field(pos, f_perturbed)
        dB_dp[:, i] = (B_perturbed - B0) / ε
    end

    return SMatrix{3, 6}(hcat(dB_dp, Matrix(dB_dm)))
end

# ============================================================================
# Promotion Gate Checking
# ============================================================================

"""
    PromotionCheckResult

Result of checking promotion gates for a candidate.
"""
struct PromotionCheckResult
    candidate_id::Int
    passes_gamma::Bool
    passes_duration::Bool
    passes_compactness::Bool
    passes_confidence::Bool
    passes_all_gates::Bool

    # Gate values
    mean_gamma::Float64
    duration_s::Float64
    cluster_radius::Float64
    min_confidence::Float64

    # Overall
    should_promote::Bool
end

"""
    check_promotion_gates(candidate, config)

Check all promotion gates for a candidate.

Gates (all must pass):
1. γ gate: mean γ > gamma_threshold
2. Temporal persistence: duration > min_duration
3. Spatial compactness: cluster radius < max_cluster_radius
4. Nav confidence: min confidence >= min_confidence

CRITICAL: Never promotes when navigation confidence was LOW or UNRELIABLE.
"""
function check_promotion_gates(candidate::DipoleFeatureCandidate;
                               config::PromotionGateConfig = DEFAULT_PROMOTION_GATE_CONFIG)
    # Check individual gates
    passes_gamma = candidate.mean_gamma > config.gamma_threshold
    passes_duration = candidate_duration(candidate) >= config.min_duration
    passes_compactness = candidate.cluster_radius < config.max_cluster_radius
    passes_confidence = candidate.min_confidence >= config.min_confidence

    passes_all_gates = passes_gamma && passes_duration &&
                       passes_compactness && passes_confidence

    return PromotionCheckResult(
        candidate.id,
        passes_gamma, passes_duration, passes_compactness, passes_confidence,
        passes_all_gates,
        candidate.mean_gamma, candidate_duration(candidate),
        candidate.cluster_radius, candidate.min_confidence,
        passes_all_gates  # Will also check fit quality later
    )
end

# ============================================================================
# Exports
# ============================================================================

export DipoleFeatureState, DIPOLE_FEATURE_DIM
export to_state_vector, from_state_vector, to_magnetic_dipole
export DipoleLifecycleState, DIPOLE_CANDIDATE, DIPOLE_ACTIVE, DIPOLE_RETIRED, DIPOLE_DEMOTED
export DipoleFeatureNode
export DipoleFeatureCandidate
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
