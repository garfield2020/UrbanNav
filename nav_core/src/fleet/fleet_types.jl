# ============================================================================
# Fleet Types - Multi-Vehicle State Management
# ============================================================================
#
# Ported from AUV-Navigation/src/fleet_types.jl
#
# Types for managing multiple vehicles in a fleet:
# - VehicleId: Unique vehicle identifier
# - VehicleState: Per-vehicle state with covariance
# - FleetState: Collection of all vehicles
# - FleetMessage: Inter-vehicle communication
# ============================================================================

using LinearAlgebra
using StaticArrays

# ============================================================================
# Vehicle Identification
# ============================================================================

"""
    VehicleId

Unique identifier for a vehicle in the fleet.
"""
struct VehicleId
    id::Int
    name::String
end

VehicleId(id::Int) = VehicleId(id, "vehicle_$id")

Base.:(==)(a::VehicleId, b::VehicleId) = a.id == b.id
Base.hash(v::VehicleId, h::UInt) = hash(v.id, h)
Base.show(io::IO, v::VehicleId) = print(io, "Vehicle($(v.id):$(v.name))")

# ============================================================================
# Vehicle State
# ============================================================================

"""
    VehicleState

State of a single vehicle with covariance.

# Fields
- `id`: Vehicle identifier
- `state`: Full AUV state (position, velocity, orientation, biases)
- `covariance`: State covariance matrix
- `timestamp`: Time of state estimate
- `health`: Health state of this vehicle
- `active`: Whether vehicle is active in fleet
"""
mutable struct VehicleState
    id::VehicleId
    state::UrbanNavState
    covariance::Matrix{Float64}
    timestamp::Float64
    health::HealthState
    active::Bool
end

function VehicleState(id::VehicleId, state::UrbanNavState;
                      covariance::Matrix{Float64} = Matrix(1.0I, 15, 15),
                      timestamp::Float64 = 0.0,
                      health::HealthState = HEALTH_HEALTHY,
                      active::Bool = true)
    VehicleState(id, state, covariance, timestamp, health, active)
end

"""Get position of vehicle."""
function vehicle_position(vs::VehicleState)
    return position(vs.state)
end

"""Get position uncertainty (std) of vehicle."""
function vehicle_position_std(vs::VehicleState)
    return sqrt(vs.covariance[1,1] + vs.covariance[2,2] + vs.covariance[3,3])
end

"""Check if vehicle state is stale."""
function is_stale(vs::VehicleState, current_time::Float64; timeout::Float64 = 10.0)
    return current_time - vs.timestamp > timeout
end

# ============================================================================
# Fleet State
# ============================================================================

"""
    FleetState

State of the entire fleet.

# Fields
- `vehicles`: All vehicle states indexed by ID
- `own_id`: This vehicle's ID (for distributed operation)
- `timestamp`: Fleet state timestamp
- `topology`: Connection topology (:full, :ring, :star)
"""
mutable struct FleetState
    vehicles::Dict{VehicleId, VehicleState}
    own_id::VehicleId
    timestamp::Float64
    topology::Symbol
end

function FleetState(own_id::VehicleId; topology::Symbol = :full)
    FleetState(Dict{VehicleId, VehicleState}(), own_id, 0.0, topology)
end

"""Add or update a vehicle in the fleet."""
function update_vehicle!(fleet::FleetState, vs::VehicleState)
    fleet.vehicles[vs.id] = vs
    fleet.timestamp = max(fleet.timestamp, vs.timestamp)
    return fleet
end

"""Remove a vehicle from the fleet."""
function remove_vehicle!(fleet::FleetState, id::VehicleId)
    delete!(fleet.vehicles, id)
    return fleet
end

"""Get a vehicle state by ID."""
function get_vehicle(fleet::FleetState, id::VehicleId)
    return get(fleet.vehicles, id, nothing)
end

"""Get all active vehicles."""
function active_vehicles(fleet::FleetState)
    return [v for v in values(fleet.vehicles) if v.active]
end

"""Get all healthy vehicles."""
function healthy_vehicles(fleet::FleetState)
    return [v for v in values(fleet.vehicles) if v.health == HEALTH_HEALTHY && v.active]
end

"""Number of vehicles in fleet."""
num_vehicles(fleet::FleetState) = length(fleet.vehicles)

"""Get fleet centroid position."""
function fleet_centroid(fleet::FleetState)
    active = active_vehicles(fleet)
    if isempty(active)
        return zeros(SVector{3, Float64})
    end
    total = sum(vehicle_position(v) for v in active)
    return total / length(active)
end

"""Get fleet spread (max inter-vehicle distance)."""
function fleet_spread(fleet::FleetState)
    active = active_vehicles(fleet)
    n = length(active)
    if n < 2
        return 0.0
    end

    max_dist = 0.0
    for i in 1:n
        for j in i+1:n
            d = norm(vehicle_position(active[i]) - vehicle_position(active[j]))
            max_dist = max(max_dist, d)
        end
    end
    return max_dist
end

# ============================================================================
# Fleet Messages
# ============================================================================

"""
    FleetMessageType

Type of inter-vehicle message.
"""
@enum FleetMessageType begin
    MSG_STATE_UPDATE = 1      # Full state update
    MSG_POSITION_ONLY = 2     # Position-only update (compressed)
    MSG_RANGING_REQUEST = 3   # Request ranging measurement
    MSG_RANGING_RESPONSE = 4  # Ranging measurement response
    MSG_HEALTH_STATUS = 5     # Health status broadcast
end

"""
    FleetMessage

Message passed between vehicles.

# Fields
- `msg_type`: Type of message
- `sender`: Sender vehicle ID
- `receiver`: Receiver vehicle ID (or nothing for broadcast)
- `timestamp`: Message creation time
- `payload`: Message-specific data
"""
struct FleetMessage
    msg_type::FleetMessageType
    sender::VehicleId
    receiver::Union{VehicleId, Nothing}
    timestamp::Float64
    payload::Dict{Symbol, Any}
end

"""Create a state update message."""
function state_update_message(sender::VehicleId, state::VehicleState;
                               receiver::Union{VehicleId, Nothing} = nothing)
    payload = Dict{Symbol, Any}(
        :position => collect(vehicle_position(state)),
        :covariance_diag => diag(state.covariance)[1:6],  # Compress to diagonal
        :health => state.health
    )
    FleetMessage(MSG_STATE_UPDATE, sender, receiver, state.timestamp, payload)
end

"""Create a position-only message (compressed)."""
function position_only_message(sender::VehicleId, state::VehicleState;
                                receiver::Union{VehicleId, Nothing} = nothing)
    payload = Dict{Symbol, Any}(
        :position => collect(vehicle_position(state)),
        :position_std => vehicle_position_std(state)
    )
    FleetMessage(MSG_POSITION_ONLY, sender, receiver, state.timestamp, payload)
end

"""Create a ranging request."""
function ranging_request_message(sender::VehicleId, receiver::VehicleId, timestamp::Float64)
    FleetMessage(MSG_RANGING_REQUEST, sender, receiver, timestamp, Dict{Symbol, Any}())
end

"""Create a ranging response."""
function ranging_response_message(sender::VehicleId, receiver::VehicleId,
                                   range::Float64, range_std::Float64, timestamp::Float64)
    payload = Dict{Symbol, Any}(
        :range => range,
        :range_std => range_std
    )
    FleetMessage(MSG_RANGING_RESPONSE, sender, receiver, timestamp, payload)
end

# ============================================================================
# Fleet Configuration
# ============================================================================

"""
    FleetConfig

Configuration for fleet operations.
"""
struct FleetConfig
    max_vehicles::Int              # Maximum number of vehicles
    stale_timeout::Float64         # Time before state is considered stale [s]
    ranging_interval::Float64      # Interval between ranging measurements [s]
    state_broadcast_interval::Float64  # Interval for state broadcasts [s]
    compression_level::Symbol      # :full, :position_only, :none
    fusion_policy::Symbol          # :centralized, :decentralized, :hierarchical
end

function FleetConfig(;
    max_vehicles::Int = 10,
    stale_timeout::Real = 10.0,
    ranging_interval::Real = 1.0,
    state_broadcast_interval::Real = 0.5,
    compression_level::Symbol = :position_only,
    fusion_policy::Symbol = :decentralized
)
    FleetConfig(max_vehicles, Float64(stale_timeout), Float64(ranging_interval),
                Float64(state_broadcast_interval), compression_level, fusion_policy)
end

const DEFAULT_FLEET_CONFIG = FleetConfig()

# ============================================================================
# Exports
# ============================================================================

export VehicleId, VehicleState, FleetState
export vehicle_position, vehicle_position_std, is_stale
export update_vehicle!, remove_vehicle!, get_vehicle
export active_vehicles, healthy_vehicles, num_vehicles
export fleet_centroid, fleet_spread
export FleetMessageType, MSG_STATE_UPDATE, MSG_POSITION_ONLY
export MSG_RANGING_REQUEST, MSG_RANGING_RESPONSE, MSG_HEALTH_STATUS
export FleetMessage
export state_update_message, position_only_message
export ranging_request_message, ranging_response_message
export FleetConfig, DEFAULT_FLEET_CONFIG
