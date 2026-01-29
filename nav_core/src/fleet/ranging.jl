# ============================================================================
# Ranging - Inter-Vehicle Range Measurements
# ============================================================================
#
# Ported from AUV-Navigation/src/ranging.jl
#
# Acoustic ranging between vehicles for cooperative localization.
# Supports both one-way (with synchronized clocks) and two-way
# (self-calibrating) ranging protocols.
# ============================================================================

using LinearAlgebra
using StaticArrays

# ============================================================================
# Constants
# ============================================================================

"""Speed of sound in water [m/s]."""
const SOUND_SPEED_AIR = 343.0

"""Default ranging noise standard deviation [m]."""
const DEFAULT_RANGING_STD = 0.5

# ============================================================================
# Ranging Measurement
# ============================================================================

"""
    RangingProtocol

Ranging protocol type.
"""
@enum RangingProtocol begin
    PROTOCOL_ONE_WAY = 1    # Requires synchronized clocks
    PROTOCOL_TWO_WAY = 2    # Self-calibrating (round-trip)
end

"""
    RangingMeasurement

Inter-vehicle range measurement.

# Fields
- `from_id`: Originating vehicle ID
- `to_id`: Target vehicle ID
- `range`: Measured range [m]
- `range_std`: Range standard deviation [m]
- `timestamp`: Measurement time
- `protocol`: Ranging protocol used
- `valid`: Whether measurement is valid
"""
struct RangingMeasurement
    from_id::VehicleId
    to_id::VehicleId
    range::Float64
    range_std::Float64
    timestamp::Float64
    protocol::RangingProtocol
    valid::Bool
end

function RangingMeasurement(from_id::VehicleId, to_id::VehicleId,
                            range::Float64, range_std::Float64, timestamp::Float64;
                            protocol::RangingProtocol = PROTOCOL_TWO_WAY,
                            valid::Bool = true)
    RangingMeasurement(from_id, to_id, range, range_std, timestamp, protocol, valid)
end

"""Compute expected range from positions."""
function expected_range(pos1::AbstractVector, pos2::AbstractVector)
    return norm(pos1 - pos2)
end

"""Compute range residual."""
function range_residual(meas::RangingMeasurement, pos1::AbstractVector, pos2::AbstractVector)
    return meas.range - expected_range(pos1, pos2)
end

"""Compute χ² statistic for range measurement."""
function range_chi2(meas::RangingMeasurement, pos1::AbstractVector, pos2::AbstractVector)
    r = range_residual(meas, pos1, pos2)
    return (r / meas.range_std)^2
end

# ============================================================================
# Ranging Model
# ============================================================================

"""
    RangingModel

Acoustic ranging sensor model.

# Fields
- `base_std`: Base range standard deviation [m]
- `range_dependent_std`: Range-dependent noise factor [m/km]
- `max_range`: Maximum valid range [m]
- `min_range`: Minimum valid range [m]
- `outlier_threshold`: χ² threshold for outlier detection
"""
struct RangingModel
    base_std::Float64
    range_dependent_std::Float64
    max_range::Float64
    min_range::Float64
    outlier_threshold::Float64
end

function RangingModel(;
    base_std::Real = 0.3,
    range_dependent_std::Real = 0.001,  # 1mm per meter
    max_range::Real = 5000.0,
    min_range::Real = 1.0,
    outlier_threshold::Real = 9.0  # χ²(1, 0.99)
)
    RangingModel(Float64(base_std), Float64(range_dependent_std),
                 Float64(max_range), Float64(min_range), Float64(outlier_threshold))
end

const DEFAULT_RANGING_MODEL = RangingModel()

"""Compute range-dependent measurement noise."""
function ranging_noise(model::RangingModel, range::Float64)
    return sqrt(model.base_std^2 + (model.range_dependent_std * range)^2)
end

"""Check if range is within valid bounds."""
function is_valid_range(model::RangingModel, range::Float64)
    return model.min_range <= range <= model.max_range
end

"""Simulate a ranging measurement."""
function simulate_ranging(model::RangingModel, pos1::AbstractVector, pos2::AbstractVector,
                          from_id::VehicleId, to_id::VehicleId, timestamp::Float64;
                          add_noise::Bool = true)
    true_range = expected_range(pos1, pos2)

    if !is_valid_range(model, true_range)
        return RangingMeasurement(from_id, to_id, true_range, Inf, timestamp,
                                  PROTOCOL_TWO_WAY, false)
    end

    std = ranging_noise(model, true_range)
    measured_range = add_noise ? true_range + std * randn() : true_range

    return RangingMeasurement(from_id, to_id, measured_range, std, timestamp)
end

# ============================================================================
# Ranging Factor (for factor graph)
# ============================================================================

"""
    RangingFactor

Factor for inter-vehicle ranging in factor graph.

Residual: r = measured_range - ||pos_i - pos_j||
Jacobians: ∂r/∂pos_i = -(pos_i - pos_j)' / ||pos_i - pos_j||
           ∂r/∂pos_j = +(pos_i - pos_j)' / ||pos_i - pos_j||
"""
struct RangingFactor
    measurement::RangingMeasurement
    vehicle_i_idx::Int  # Index of first vehicle in state vector
    vehicle_j_idx::Int  # Index of second vehicle in state vector
end

"""Compute range factor residual."""
function residual(factor::RangingFactor, pos_i::AbstractVector, pos_j::AbstractVector)
    return factor.measurement.range - norm(pos_i - pos_j)
end

"""Compute range factor Jacobians."""
function jacobians(factor::RangingFactor, pos_i::AbstractVector, pos_j::AbstractVector)
    diff = pos_i - pos_j
    d = norm(diff)
    if d < 1e-10
        # Avoid division by zero
        unit_vec = SVector{3}(1.0, 0.0, 0.0)
    else
        unit_vec = SVector{3}(diff ./ d)
    end

    # ∂r/∂pos_i = -unit_vec (since r = meas - d, and ∂d/∂pos_i = unit_vec)
    J_i = -unit_vec'  # 1x3

    # ∂r/∂pos_j = +unit_vec
    J_j = unit_vec'   # 1x3

    return J_i, J_j
end

"""Get information (inverse covariance) for ranging factor."""
function information(factor::RangingFactor)
    return 1.0 / factor.measurement.range_std^2
end

# ============================================================================
# Ranging Schedule
# ============================================================================

"""
    RangingSchedule

Schedule for when vehicles should perform ranging.
"""
mutable struct RangingSchedule
    interval::Float64                    # Time between ranging rounds
    last_ranging::Dict{Tuple{Int,Int}, Float64}  # Last ranging time per pair
    priority_order::Vector{Tuple{Int,Int}}       # Priority order for pairs
end

function RangingSchedule(; interval::Float64 = 1.0)
    RangingSchedule(
        interval,
        Dict{Tuple{Int,Int}, Float64}(),
        Tuple{Int,Int}[]
    )
end

"""Check if pair should range at given time."""
function should_range(schedule::RangingSchedule, id1::Int, id2::Int, timestamp::Float64)
    pair = id1 < id2 ? (id1, id2) : (id2, id1)
    last_time = get(schedule.last_ranging, pair, -Inf)
    return timestamp - last_time >= schedule.interval
end

"""Record that ranging occurred."""
function record_ranging!(schedule::RangingSchedule, id1::Int, id2::Int, timestamp::Float64)
    pair = id1 < id2 ? (id1, id2) : (id2, id1)
    schedule.last_ranging[pair] = timestamp
end

"""Get pairs that should range now."""
function pairs_to_range(schedule::RangingSchedule, vehicle_ids::Vector{Int}, timestamp::Float64)
    pairs = Tuple{Int,Int}[]
    n = length(vehicle_ids)
    for i in 1:n
        for j in i+1:n
            if should_range(schedule, vehicle_ids[i], vehicle_ids[j], timestamp)
                push!(pairs, (vehicle_ids[i], vehicle_ids[j]))
            end
        end
    end
    return pairs
end

# ============================================================================
# Exports
# ============================================================================

export SOUND_SPEED_AIR, DEFAULT_RANGING_STD
export RangingProtocol, PROTOCOL_ONE_WAY, PROTOCOL_TWO_WAY
export RangingMeasurement, expected_range, range_residual, range_chi2
export RangingModel, DEFAULT_RANGING_MODEL
export ranging_noise, is_valid_range, simulate_ranging
export RangingFactor
export RangingSchedule, should_range, record_ranging!, pairs_to_range
