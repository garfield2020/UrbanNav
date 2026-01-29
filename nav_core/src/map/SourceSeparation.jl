# ============================================================================
# Source Separation (Phase B)
# ============================================================================
#
# Couples object sources cleanly without polluting the background map.
#
# Purpose: "Objects explain themselves; they don't corrupt the map."
#
# The Problem:
# -----------
# Magnetic measurements contain contributions from multiple sources:
#   z = B_background(x) + B_object_1(x) + B_object_2(x) + ... + noise
#
# If we naively learn the map from all measurements, object signatures
# get absorbed into the background, causing:
# 1. Map coefficients drift toward object locations
# 2. Position estimates biased when object moves/disappears
# 3. Poor generalization to new missions
#
# The Solution:
# ------------
# Explicitly attribute each measurement to sources, then:
# 1. Subtract object contributions before map learning
# 2. Track attribution uncertainty
# 3. Only update map from "clean" measurements (high background attribution)
#
# Key Equation:
#   z_clean = z_raw - Σᵢ B_object_i(x̂) - Σᵢ J_i · δx_object_i
#
# where J_i is the Jacobian of object i's field w.r.t. its parameters.
#
# ============================================================================

using LinearAlgebra
using StaticArrays
using Statistics: mean

# ============================================================================
# Source Attribution
# ============================================================================

"""
    SourceType

Classification of magnetic field sources.

# Types
- `SOURCE_BACKGROUND`: Regional field from geological/external sources
- `SOURCE_OBJECT`: Localized anomaly (dipole feature)
- `SOURCE_VEHICLE`: Self-interference from vehicle (motors, electronics)
- `SOURCE_UNKNOWN`: Unattributed residual
"""
@enum SourceType begin
    SOURCE_BACKGROUND = 1
    SOURCE_OBJECT = 2
    SOURCE_VEHICLE = 3
    SOURCE_UNKNOWN = 4
    SOURCE_ELEVATOR = 5
    SOURCE_DOOR = 6
end

"""
    SourceContribution

Attribution of measurement to a specific source.

# Fields
- `source_type::SourceType`: What kind of source
- `source_id::String`: Unique identifier (empty for background)
- `field_contribution::Vector{Float64}`: Estimated B contribution [T]
- `contribution_covariance::Matrix{Float64}`: Uncertainty in contribution [T²]
- `attribution_confidence::Float64`: Confidence in attribution ∈ [0, 1]

# Attribution Confidence
- 1.0: Certain attribution (e.g., vehicle contribution from calibration)
- 0.5-0.9: Moderate confidence (detected object with some uncertainty)
- <0.5: Low confidence (might be noise, might be weak object)

The attribution confidence affects how we weight the subtraction:
- High confidence: subtract full contribution
- Low confidence: conservative subtraction to avoid over-correction
"""
struct SourceContribution
    source_type::SourceType
    source_id::String
    field_contribution::Vector{Float64}
    contribution_covariance::Matrix{Float64}
    attribution_confidence::Float64

    function SourceContribution(
        source_type::SourceType,
        source_id::String,
        field_contribution::AbstractVector,
        contribution_covariance::AbstractMatrix,
        attribution_confidence::Float64
    )
        @assert length(field_contribution) == 3 "Field contribution must be 3D"
        @assert size(contribution_covariance) == (3, 3) "Covariance must be 3×3"
        @assert 0 <= attribution_confidence <= 1 "Confidence must be in [0, 1]"
        new(
            source_type,
            source_id,
            Vector{Float64}(field_contribution),
            Matrix{Float64}(contribution_covariance),
            attribution_confidence
        )
    end
end

"""Create background source contribution."""
function background_contribution(B_pred::AbstractVector, Σ_map::AbstractMatrix)
    SourceContribution(
        SOURCE_BACKGROUND,
        "",
        B_pred,
        Σ_map,
        1.0  # Background attribution is always certain
    )
end

"""Create object source contribution."""
function object_contribution(
    object_id::String,
    B_object::AbstractVector,
    Σ_object::AbstractMatrix,
    confidence::Float64
)
    SourceContribution(
        SOURCE_OBJECT,
        object_id,
        B_object,
        Σ_object,
        confidence
    )
end

"""Create vehicle self-interference contribution."""
function vehicle_contribution(B_vehicle::AbstractVector, Σ_vehicle::AbstractMatrix)
    SourceContribution(
        SOURCE_VEHICLE,
        "self",
        B_vehicle,
        Σ_vehicle,
        0.95  # Vehicle contribution typically well-calibrated
    )
end

"""Create elevator source contribution."""
elevator_contribution(B::Vec3Map, σ::Float64) = SourceContribution(SOURCE_ELEVATOR, B, σ)

"""Create door source contribution."""
door_contribution(B::Vec3Map, σ::Float64) = SourceContribution(SOURCE_DOOR, B, σ)

# ============================================================================
# Source Separation Configuration
# ============================================================================

"""
    SourceSeparationConfig

Configuration for source separation and clean background extraction.

# Fields
- `max_object_contribution_ratio::Float64`: Max ratio of object to total variance
- `min_attribution_confidence::Float64`: Minimum confidence for subtraction
- `residual_chi2_threshold::Float64`: Max χ² for teachable measurement
- `object_proximity_threshold::Float64`: Distance [m] within which object dominates
- `conservative_subtraction::Bool`: Scale subtraction by confidence

# Threshold Justifications

**max_object_contribution_ratio = 0.3**:
If object variance exceeds 30% of total measurement variance, the measurement
is dominated by object uncertainty. Background signal is not cleanly extractable.
This threshold ensures background signal-to-noise ratio ≥ 2.3.

**min_attribution_confidence = 0.7**:
Below 70% confidence, we're uncertain whether the detected anomaly is really
an object or a background feature. Conservative approach: don't subtract.

**residual_chi2_threshold = 11.345**:
P(χ²(3) > 11.345) = 0.01. If residual exceeds this, either:
- Unmodeled source present
- Attribution error
Either way, not suitable for background learning.

**object_proximity_threshold = 10.0 m**:
Within 10m of a dipole, field gradients are steep and linearization errors
are significant. Measurements too close to objects are unreliable for
background learning regardless of attribution confidence.
"""
struct SourceSeparationConfig
    max_object_contribution_ratio::Float64
    min_attribution_confidence::Float64
    residual_chi2_threshold::Float64
    object_proximity_threshold::Float64
    conservative_subtraction::Bool

    function SourceSeparationConfig(;
        max_object_contribution_ratio::Float64 = 0.3,
        min_attribution_confidence::Float64 = 0.7,
        residual_chi2_threshold::Float64 = 11.345,  # χ²(3, p=0.01)
        object_proximity_threshold::Float64 = 10.0,
        conservative_subtraction::Bool = true
    )
        @assert 0 < max_object_contribution_ratio < 1
        @assert 0 < min_attribution_confidence <= 1
        @assert residual_chi2_threshold > 0
        @assert object_proximity_threshold > 0
        new(
            max_object_contribution_ratio,
            min_attribution_confidence,
            residual_chi2_threshold,
            object_proximity_threshold,
            conservative_subtraction
        )
    end
end

const DEFAULT_SOURCE_SEPARATION_CONFIG = SourceSeparationConfig()

# ============================================================================
# Measurement Attribution
# ============================================================================

"""
    MeasurementAttribution

Full attribution of a measurement to all sources.

# Fields
- `timestamp::Float64`: Measurement time [s]
- `position::Vector{Float64}`: Measurement position [m]
- `raw_measurement::Vector{Float64}`: Raw measurement z [T]
- `sensor_covariance::Matrix{Float64}`: Sensor noise R [T²]
- `sources::Vector{SourceContribution}`: All attributed sources
- `residual::Vector{Float64}`: Unattributed residual [T]
- `residual_covariance::Matrix{Float64}`: Residual uncertainty [T²]
- `background_teachable::Bool`: Whether residual is suitable for map learning

# Residual Calculation
residual = raw_measurement - Σᵢ (confidence_i × contribution_i)

The residual should be small if all sources are correctly attributed.
Large residual suggests unmodeled source or attribution error.
"""
struct MeasurementAttribution
    timestamp::Float64
    position::Vector{Float64}
    raw_measurement::Vector{Float64}
    sensor_covariance::Matrix{Float64}
    sources::Vector{SourceContribution}
    residual::Vector{Float64}
    residual_covariance::Matrix{Float64}
    background_teachable::Bool
end

"""
    attribute_measurement(timestamp, position, z_raw, R_sensor,
                          B_background, Σ_background,
                          object_contributions::Vector{SourceContribution};
                          config::SourceSeparationConfig)

Attribute a measurement to background and object sources.

# Arguments
- `timestamp`: Measurement time [s]
- `position`: Measurement position [m]
- `z_raw`: Raw measurement [T]
- `R_sensor`: Sensor noise covariance [T²]
- `B_background`: Predicted background field [T]
- `Σ_background`: Background prediction uncertainty [T²]
- `object_contributions`: Contributions from known objects
- `config`: Separation configuration

# Returns
MeasurementAttribution with residual and teachability flag.
"""
function attribute_measurement(
    timestamp::Float64,
    position::AbstractVector,
    z_raw::AbstractVector,
    R_sensor::AbstractMatrix,
    B_background::AbstractVector,
    Σ_background::AbstractMatrix,
    object_contributions::Vector{SourceContribution},
    config::SourceSeparationConfig
)
    # Collect all sources
    sources = SourceContribution[]

    # Add background
    push!(sources, background_contribution(B_background, Σ_background))

    # Add objects
    append!(sources, object_contributions)

    # Compute expected measurement from all sources
    z_expected = zeros(3)
    Σ_expected = zeros(3, 3)

    for src in sources
        # Weighted by attribution confidence
        z_expected += src.attribution_confidence * src.field_contribution
        # Covariance includes attribution uncertainty
        Σ_expected += src.attribution_confidence^2 * src.contribution_covariance
    end

    # Compute residual
    residual = z_raw - z_expected

    # Residual covariance includes sensor noise and prediction uncertainty
    residual_covariance = R_sensor + Σ_expected

    # Determine teachability for background learning
    background_teachable = check_background_teachability(
        residual, residual_covariance, object_contributions, config
    )

    MeasurementAttribution(
        timestamp,
        Vector{Float64}(position),
        Vector{Float64}(z_raw),
        Matrix{Float64}(R_sensor),
        sources,
        residual,
        residual_covariance,
        background_teachable
    )
end

# ============================================================================
# Teachability Check
# ============================================================================

"""
    check_background_teachability(residual, residual_cov, object_contributions, config)

Determine if measurement is suitable for background map learning.

# Criteria (all must pass)
1. No object dominates (object variance < max_ratio × total variance)
2. All objects have sufficient attribution confidence
3. Residual passes χ² test
4. Not too close to any object

Returns true if measurement can safely teach the background map.
"""
function check_background_teachability(
    residual::AbstractVector,
    residual_cov::AbstractMatrix,
    object_contributions::Vector{SourceContribution},
    config::SourceSeparationConfig
)
    # Criterion 1: No object dominates
    total_variance = tr(residual_cov)
    for obj in object_contributions
        obj_variance = tr(obj.contribution_covariance)
        if obj_variance / total_variance > config.max_object_contribution_ratio
            return false
        end
    end

    # Criterion 2: Attribution confidence sufficient
    for obj in object_contributions
        if obj.attribution_confidence < config.min_attribution_confidence
            # Low confidence means we're uncertain about subtraction
            # If the object contribution is significant, reject
            obj_magnitude = norm(obj.field_contribution)
            if obj_magnitude > sqrt(total_variance / 3)
                return false
            end
        end
    end

    # Criterion 3: Residual χ² test
    # χ² = r' Σ⁻¹ r
    residual_cov_reg = residual_cov + 1e-20 * I(3)
    chi2 = residual' * (residual_cov_reg \ residual)
    if chi2 > config.residual_chi2_threshold
        return false
    end

    return true
end

# ============================================================================
# Clean Measurement Extraction
# ============================================================================

"""
    CleanMeasurement

Measurement with object contributions removed.

# Fields
- `position::Vector{Float64}`: Measurement position [m]
- `clean_field::Vector{Float64}`: Field with objects subtracted [T]
- `clean_covariance::Matrix{Float64}`: Uncertainty after subtraction [T²]
- `subtracted_sources::Vector{String}`: IDs of subtracted sources
- `subtraction_uncertainty::Matrix{Float64}`: Additional uncertainty from subtraction [T²]
- `quality_score::Float64`: Confidence in cleanliness ∈ [0, 1]
"""
struct CleanMeasurement
    position::Vector{Float64}
    clean_field::Vector{Float64}
    clean_covariance::Matrix{Float64}
    subtracted_sources::Vector{String}
    subtraction_uncertainty::Matrix{Float64}
    quality_score::Float64
end

"""
    extract_clean_measurement(attribution::MeasurementAttribution,
                              config::SourceSeparationConfig)

Extract background-only measurement by subtracting object contributions.

# Method
For each object source with sufficient confidence:
  z_clean = z_raw - Σᵢ (confidence_i × B_object_i)

Covariance inflates to account for subtraction uncertainty:
  R_clean = R_sensor + Σᵢ (confidence_i² × Σ_object_i) + Σ_subtraction_error

The subtraction error accounts for:
- Position uncertainty affecting object field prediction
- Object parameter uncertainty
- Linearization error near objects
"""
function extract_clean_measurement(
    attribution::MeasurementAttribution,
    config::SourceSeparationConfig
)
    z_clean = copy(attribution.raw_measurement)
    subtracted = String[]
    Σ_subtraction = zeros(3, 3)

    for src in attribution.sources
        if src.source_type == SOURCE_OBJECT
            # Check if we should subtract this object
            if src.attribution_confidence >= config.min_attribution_confidence
                # Subtract with confidence weighting if conservative
                if config.conservative_subtraction
                    weight = src.attribution_confidence
                else
                    weight = 1.0
                end

                z_clean -= weight * src.field_contribution
                push!(subtracted, src.source_id)

                # Add subtraction uncertainty
                # Conservative mode: Add FULL object covariance (we're uncertain about
                # the object model itself) plus margin for the portion we didn't subtract.
                # This gives larger covariance = more conservative.
                #
                # Aggressive mode: Add weight² × Σ (just the variance of what we subtracted)
                if config.conservative_subtraction
                    # Full object uncertainty + margin for under-subtraction
                    # Σ_subtraction = Σ_object + (1-w)² × Σ_object
                    Σ_subtraction += src.contribution_covariance
                    Σ_subtraction += (1 - weight)^2 * src.contribution_covariance
                else
                    # Aggressive: just the scaled variance from subtraction
                    Σ_subtraction += weight^2 * src.contribution_covariance
                end
            end
        end
    end

    # Total clean covariance
    R_clean = attribution.sensor_covariance + Σ_subtraction

    # Quality score based on:
    # - How much variance from objects vs sensor
    # - Attribution confidence
    # - Residual magnitude
    sensor_var = tr(attribution.sensor_covariance)
    subtraction_var = tr(Σ_subtraction)
    quality = sensor_var / (sensor_var + subtraction_var)

    # Penalize for low attribution confidence
    min_conf = isempty(attribution.sources) ? 1.0 :
               minimum(s.attribution_confidence for s in attribution.sources
                       if s.source_type == SOURCE_OBJECT; init=1.0)
    quality *= min_conf

    CleanMeasurement(
        attribution.position,
        z_clean,
        R_clean,
        subtracted,
        Σ_subtraction,
        quality
    )
end

# ============================================================================
# Source Separation Statistics
# ============================================================================

"""
    SourceSeparationStatistics

Statistics for monitoring source separation quality.

# Fields
- `total_measurements::Int`: Total measurements processed
- `teachable_count::Int`: Measurements suitable for background learning
- `rejected_object_dominated::Int`: Rejected due to object dominance
- `rejected_low_confidence::Int`: Rejected due to low attribution confidence
- `rejected_chi2::Int`: Rejected due to large residual
- `mean_quality_score::Float64`: Average quality of clean measurements
- `object_contribution_ratio::Float64`: Mean object/total variance ratio
"""
mutable struct SourceSeparationStatistics
    total_measurements::Int
    teachable_count::Int
    rejected_object_dominated::Int
    rejected_low_confidence::Int
    rejected_chi2::Int
    mean_quality_score::Float64
    object_contribution_ratio::Float64
end

function SourceSeparationStatistics()
    SourceSeparationStatistics(0, 0, 0, 0, 0, 0.0, 0.0)
end

"""Update statistics with new attribution."""
function update_statistics!(
    stats::SourceSeparationStatistics,
    attribution::MeasurementAttribution,
    clean::CleanMeasurement,
    config::SourceSeparationConfig
)
    stats.total_measurements += 1

    if attribution.background_teachable
        stats.teachable_count += 1
    else
        # Determine rejection reason
        total_var = tr(attribution.residual_covariance)

        # Check object dominance
        for src in attribution.sources
            if src.source_type == SOURCE_OBJECT
                obj_var = tr(src.contribution_covariance)
                if obj_var / total_var > config.max_object_contribution_ratio
                    stats.rejected_object_dominated += 1
                    break
                end
            end
        end

        # Check confidence
        for src in attribution.sources
            if src.source_type == SOURCE_OBJECT &&
               src.attribution_confidence < config.min_attribution_confidence
                stats.rejected_low_confidence += 1
                break
            end
        end

        # Check χ²
        residual_cov_reg = attribution.residual_covariance + 1e-20 * I(3)
        chi2 = attribution.residual' * (residual_cov_reg \ attribution.residual)
        if chi2 > config.residual_chi2_threshold
            stats.rejected_chi2 += 1
        end
    end

    # Running average of quality score
    n = stats.total_measurements
    stats.mean_quality_score = ((n-1) * stats.mean_quality_score + clean.quality_score) / n

    # Track object contribution ratio
    total_var = tr(attribution.sensor_covariance)
    obj_var = sum(tr(s.contribution_covariance) for s in attribution.sources
                  if s.source_type == SOURCE_OBJECT; init=0.0)
    ratio = obj_var / (obj_var + total_var)
    stats.object_contribution_ratio = ((n-1) * stats.object_contribution_ratio + ratio) / n
end

"""Compute teachability rate."""
function teachability_rate(stats::SourceSeparationStatistics)
    if stats.total_measurements == 0
        return 1.0
    end
    return stats.teachable_count / stats.total_measurements
end

# ============================================================================
# Source Separator Manager
# ============================================================================

"""
    SourceSeparator

Manager for source separation during navigation.

# Fields
- `config::SourceSeparationConfig`: Configuration
- `statistics::SourceSeparationStatistics`: Running statistics
- `known_objects::Dict{String, ObjectState}`: Currently tracked objects
- `clean_buffer::Vector{CleanMeasurement}`: Buffer of clean measurements

# Usage
```julia
separator = SourceSeparator()

# For each measurement:
object_contribs = [object_contribution("obj_1", B, Σ, conf), ...]
attr = attribute_measurement(t, pos, z, R, B_bg, Σ_bg, object_contribs, separator.config)
clean = extract_clean_measurement(attr, separator.config)

if attr.background_teachable
    # Safe to use for map learning
    push!(background_learning_residuals, clean)
end
```
"""
mutable struct SourceSeparator
    config::SourceSeparationConfig
    statistics::SourceSeparationStatistics
end

function SourceSeparator(config::SourceSeparationConfig = DEFAULT_SOURCE_SEPARATION_CONFIG)
    SourceSeparator(config, SourceSeparationStatistics())
end

"""Process measurement through source separator."""
function process_measurement!(
    separator::SourceSeparator,
    timestamp::Float64,
    position::AbstractVector,
    z_raw::AbstractVector,
    R_sensor::AbstractMatrix,
    B_background::AbstractVector,
    Σ_background::AbstractMatrix,
    object_contributions::Vector{SourceContribution}
)
    # Attribute measurement
    attribution = attribute_measurement(
        timestamp, position, z_raw, R_sensor,
        B_background, Σ_background,
        object_contributions, separator.config
    )

    # Extract clean measurement
    clean = extract_clean_measurement(attribution, separator.config)

    # Update statistics
    update_statistics!(separator.statistics, attribution, clean, separator.config)

    return (attribution = attribution, clean = clean)
end

# ============================================================================
# Formatting and Display
# ============================================================================

"""Format source separation statistics."""
function format_separation_statistics(stats::SourceSeparationStatistics)
    lines = String[]
    push!(lines, "Source Separation Statistics")
    push!(lines, "=" ^ 40)
    push!(lines, "Total measurements: $(stats.total_measurements)")
    push!(lines, "Teachable: $(stats.teachable_count) ($(round(teachability_rate(stats) * 100, digits=1))%)")
    push!(lines, "")
    push!(lines, "Rejection Reasons:")
    push!(lines, "  Object dominated: $(stats.rejected_object_dominated)")
    push!(lines, "  Low confidence: $(stats.rejected_low_confidence)")
    push!(lines, "  Large residual: $(stats.rejected_chi2)")
    push!(lines, "")
    push!(lines, "Quality Metrics:")
    push!(lines, "  Mean quality score: $(round(stats.mean_quality_score, digits=3))")
    push!(lines, "  Mean object ratio: $(round(stats.object_contribution_ratio * 100, digits=1))%")

    return join(lines, "\n")
end

"""Format measurement attribution."""
function format_attribution(attribution::MeasurementAttribution)
    lines = String[]
    push!(lines, "Measurement Attribution")
    push!(lines, "-" ^ 40)
    push!(lines, "Position: $(round.(attribution.position, digits=2)) m")
    push!(lines, "Raw measurement: $(round.(attribution.raw_measurement .* 1e9, digits=1)) nT")
    push!(lines, "")
    push!(lines, "Sources:")
    for src in attribution.sources
        type_str = string(src.source_type)
        id_str = isempty(src.source_id) ? "" : " ($(src.source_id))"
        B_nT = round.(src.field_contribution .* 1e9, digits=1)
        push!(lines, "  $type_str$id_str: $B_nT nT (conf=$(round(src.attribution_confidence, digits=2)))")
    end
    push!(lines, "")
    push!(lines, "Residual: $(round.(attribution.residual .* 1e9, digits=1)) nT")
    status = attribution.background_teachable ? "✓ TEACHABLE" : "✗ NOT TEACHABLE"
    push!(lines, "Status: $status")

    return join(lines, "\n")
end

# ============================================================================
# Exports
# ============================================================================

export SourceType, SOURCE_BACKGROUND, SOURCE_OBJECT, SOURCE_VEHICLE, SOURCE_UNKNOWN, SOURCE_ELEVATOR, SOURCE_DOOR
export SourceContribution
export background_contribution, object_contribution, vehicle_contribution, elevator_contribution, door_contribution
export MeasurementAttribution, attribute_measurement
export SourceSeparationConfig, DEFAULT_SOURCE_SEPARATION_CONFIG
export check_background_teachability
export CleanMeasurement, extract_clean_measurement
export SourceSeparationStatistics, update_statistics!, teachability_rate
export SourceSeparator, process_measurement!
export format_separation_statistics, format_attribution
