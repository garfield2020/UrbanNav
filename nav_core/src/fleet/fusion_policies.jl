# ============================================================================
# Fusion Policies - Multi-Vehicle State Fusion
# ============================================================================
#
# Ported from AUV-Navigation/src/fusion_policies.jl
#
# Implementations of fleet fusion policies:
# - CentralizedFusion: Central server collects and fuses all states
# - DecentralizedFusion: Peer-to-peer with covariance intersection
# - HierarchicalFusion: Tree-structured fusion
# ============================================================================

using LinearAlgebra

# ============================================================================
# Centralized Fusion
# ============================================================================

"""
    CentralizedFusion <: AbstractFleetPolicy

Centralized fusion where a single node collects all vehicle states.

# Algorithm
1. All vehicles send state to central node
2. Central node performs joint state estimation
3. Central node broadcasts updated states back

# Fields
- `is_central`: Whether this vehicle is the central node
- `central_id`: ID of the central node
- `states`: Collected vehicle states
- `timeout`: State timeout [s]
"""
mutable struct CentralizedFusion <: AbstractFleetPolicy
    is_central::Bool
    central_id::Union{VehicleId, Nothing}
    states::Dict{VehicleId, VehicleState}
    timeout::Float64
    last_fusion_time::Float64
end

function CentralizedFusion(;
    is_central::Bool = false,
    central_id::Union{VehicleId, Nothing} = nothing,
    timeout::Float64 = 10.0
)
    CentralizedFusion(is_central, central_id,
                      Dict{VehicleId, VehicleState}(),
                      timeout, 0.0)
end

"""Initialize centralized fusion."""
function initialize!(policy::CentralizedFusion, config::FleetConfig, own_id::VehicleId)
    # First vehicle becomes central by default if not specified
    if policy.central_id === nothing
        policy.central_id = own_id
        policy.is_central = true
    else
        policy.is_central = (own_id == policy.central_id)
    end
    empty!(policy.states)
    policy.last_fusion_time = 0.0
    return policy
end

"""Receive state update from a vehicle."""
function receive!(policy::CentralizedFusion, state::VehicleState)
    policy.states[state.id] = state
    return policy
end

"""Perform centralized fusion (only on central node)."""
function fuse!(policy::CentralizedFusion, timestamp::Float64)
    if !policy.is_central
        return nothing
    end

    # Remove stale states
    for (id, state) in collect(policy.states)
        if is_stale(state, timestamp; timeout = policy.timeout)
            delete!(policy.states, id)
        end
    end

    # For now, just collect states (full joint estimation would go here)
    policy.last_fusion_time = timestamp

    return collect(values(policy.states))
end

"""Determine what to share based on policy."""
function share(policy::CentralizedFusion, own_state::VehicleState)
    if policy.is_central
        # Central node broadcasts fused states
        return :broadcast_all
    else
        # Other nodes send to central
        return :send_to_central
    end
end

"""Get fused state for a vehicle."""
function get_fused_state(policy::CentralizedFusion, id::VehicleId)
    return get(policy.states, id, nothing)
end

# ============================================================================
# Decentralized Fusion (Covariance Intersection)
# ============================================================================

"""
    DecentralizedFusion <: AbstractFleetPolicy

Decentralized fusion using Covariance Intersection (CI).

# Algorithm
Each vehicle maintains its own estimate and fuses received estimates
using CI, which is robust to unknown correlations.

# CI Fusion
Given estimates (x₁, P₁) and (x₂, P₂):
P_fused⁻¹ = ω·P₁⁻¹ + (1-ω)·P₂⁻¹
x_fused = P_fused · (ω·P₁⁻¹·x₁ + (1-ω)·P₂⁻¹·x₂)

where ω ∈ [0,1] is chosen to minimize det(P_fused).

# Fields
- `own_id`: This vehicle's ID
- `peer_states`: Received peer states
- `omega`: CI mixing parameter (0 = trust peers, 1 = trust self)
- `auto_omega`: Automatically optimize omega
"""
mutable struct DecentralizedFusion <: AbstractFleetPolicy
    own_id::Union{VehicleId, Nothing}
    peer_states::Dict{VehicleId, VehicleState}
    omega::Float64
    auto_omega::Bool
    timeout::Float64
end

function DecentralizedFusion(;
    omega::Float64 = 0.5,
    auto_omega::Bool = true,
    timeout::Float64 = 10.0
)
    DecentralizedFusion(nothing, Dict{VehicleId, VehicleState}(),
                        omega, auto_omega, timeout)
end

"""Initialize decentralized fusion."""
function initialize!(policy::DecentralizedFusion, config::FleetConfig, own_id::VehicleId)
    policy.own_id = own_id
    empty!(policy.peer_states)
    return policy
end

"""Receive state update from a peer."""
function receive!(policy::DecentralizedFusion, state::VehicleState)
    if state.id != policy.own_id
        policy.peer_states[state.id] = state
    end
    return policy
end

"""
    covariance_intersection(x1, P1, x2, P2; omega=0.5)

Perform Covariance Intersection fusion.

Returns (x_fused, P_fused).
"""
function covariance_intersection(x1::AbstractVector, P1::AbstractMatrix,
                                  x2::AbstractVector, P2::AbstractMatrix;
                                  omega::Float64 = 0.5)
    # Ensure omega is in valid range
    ω = clamp(omega, 0.01, 0.99)

    # Compute information matrices
    try
        P1_inv = inv(P1)
        P2_inv = inv(P2)

        # CI fusion
        P_fused_inv = ω * P1_inv + (1 - ω) * P2_inv
        P_fused = inv(P_fused_inv)

        x_fused = P_fused * (ω * P1_inv * x1 + (1 - ω) * P2_inv * x2)

        return x_fused, P_fused
    catch e
        # Fall back to weighted average if matrices are singular
        @warn "CI fusion failed, using weighted average" exception=e
        x_fused = ω * x1 + (1 - ω) * x2
        P_fused = ω * P1 + (1 - ω) * P2
        return x_fused, P_fused
    end
end

"""Find optimal omega that minimizes det(P_fused)."""
function optimize_omega(P1::AbstractMatrix, P2::AbstractMatrix;
                         num_samples::Int = 20)
    best_omega = 0.5
    best_det = Inf

    for ω in range(0.1, 0.9, length=num_samples)
        try
            P1_inv = inv(P1)
            P2_inv = inv(P2)
            P_fused_inv = ω * P1_inv + (1 - ω) * P2_inv
            P_fused = inv(P_fused_inv)
            d = det(P_fused)
            if d < best_det && d > 0
                best_det = d
                best_omega = ω
            end
        catch
            continue
        end
    end

    return best_omega
end

"""Fuse own state with peer states using CI."""
function fuse!(policy::DecentralizedFusion, own_state::VehicleState, timestamp::Float64)
    # Remove stale peer states
    for (id, state) in collect(policy.peer_states)
        if is_stale(state, timestamp; timeout = policy.timeout)
            delete!(policy.peer_states, id)
        end
    end

    if isempty(policy.peer_states)
        return own_state
    end

    # Extract own position and covariance (just position for simplicity)
    x_own = collect(vehicle_position(own_state))
    P_own = own_state.covariance[1:3, 1:3]

    x_fused = x_own
    P_fused = P_own

    # Sequentially fuse with each peer
    for (_, peer_state) in policy.peer_states
        if !peer_state.active || peer_state.health != HEALTH_HEALTHY
            continue
        end

        x_peer = collect(vehicle_position(peer_state))
        P_peer = peer_state.covariance[1:3, 1:3]

        # Find optimal omega if auto
        ω = policy.auto_omega ? optimize_omega(P_fused, P_peer) : policy.omega

        x_fused, P_fused = covariance_intersection(x_fused, P_fused, x_peer, P_peer; omega=ω)
    end

    # Create fused state (only updating position portion)
    fused_cov = copy(own_state.covariance)
    fused_cov[1:3, 1:3] = P_fused

    # Note: For a full implementation, would update the UrbanNavState
    # Here we return metrics about the fusion
    return (position = SVector{3}(x_fused...),
            covariance = P_fused,
            num_peers = length(policy.peer_states))
end

"""Determine what to share (decentralized shares with all peers)."""
function share(policy::DecentralizedFusion, own_state::VehicleState)
    return :broadcast_peers
end

# ============================================================================
# Hierarchical Fusion
# ============================================================================

"""
    HierarchicalFusion <: AbstractFleetPolicy

Hierarchical (tree-structured) fusion.

Vehicles are organized in a tree where:
- Leaf nodes send to their parent
- Internal nodes fuse children and send to parent
- Root node has the globally fused estimate

# Fields
- `own_id`: This vehicle's ID
- `parent_id`: Parent node ID (nothing for root)
- `children_ids`: Child node IDs
- `level`: Level in hierarchy (0 = root)
- `child_states`: States received from children
"""
mutable struct HierarchicalFusion <: AbstractFleetPolicy
    own_id::Union{VehicleId, Nothing}
    parent_id::Union{VehicleId, Nothing}
    children_ids::Vector{VehicleId}
    level::Int
    child_states::Dict{VehicleId, VehicleState}
    timeout::Float64
end

function HierarchicalFusion(;
    parent_id::Union{VehicleId, Nothing} = nothing,
    children_ids::Vector{VehicleId} = VehicleId[],
    level::Int = 0,
    timeout::Float64 = 10.0
)
    HierarchicalFusion(nothing, parent_id, children_ids, level,
                       Dict{VehicleId, VehicleState}(), timeout)
end

"""Initialize hierarchical fusion."""
function initialize!(policy::HierarchicalFusion, config::FleetConfig, own_id::VehicleId)
    policy.own_id = own_id
    empty!(policy.child_states)
    return policy
end

"""Set hierarchy structure."""
function set_hierarchy!(policy::HierarchicalFusion;
                        parent_id::Union{VehicleId, Nothing} = nothing,
                        children_ids::Vector{VehicleId} = VehicleId[],
                        level::Int = 0)
    policy.parent_id = parent_id
    policy.children_ids = children_ids
    policy.level = level
    return policy
end

"""Check if this node is root."""
is_root(policy::HierarchicalFusion) = policy.parent_id === nothing

"""Check if this node is leaf."""
is_leaf(policy::HierarchicalFusion) = isempty(policy.children_ids)

"""Receive state from a child."""
function receive!(policy::HierarchicalFusion, state::VehicleState)
    if state.id in policy.children_ids
        policy.child_states[state.id] = state
    end
    return policy
end

"""Check if all children have reported."""
function all_children_reported(policy::HierarchicalFusion)
    return length(policy.child_states) == length(policy.children_ids)
end

"""Fuse child states (simple average for now)."""
function fuse!(policy::HierarchicalFusion, own_state::VehicleState, timestamp::Float64)
    if is_leaf(policy)
        # Leaf nodes just return their own state
        return own_state
    end

    # Remove stale child states
    for (id, state) in collect(policy.child_states)
        if is_stale(state, timestamp; timeout = policy.timeout)
            delete!(policy.child_states, id)
        end
    end

    if isempty(policy.child_states)
        return own_state
    end

    # Weighted average fusion (weight by inverse variance)
    all_states = [own_state; collect(values(policy.child_states))...]

    total_weight = 0.0
    x_sum = zeros(3)

    for state in all_states
        w = 1.0 / max(vehicle_position_std(state), 0.01)
        x_sum += w * collect(vehicle_position(state))
        total_weight += w
    end

    fused_position = x_sum / total_weight

    return (position = SVector{3}(fused_position...),
            num_children = length(policy.child_states),
            level = policy.level)
end

"""Determine what to share based on hierarchy."""
function share(policy::HierarchicalFusion, own_state::VehicleState)
    if is_root(policy)
        return :broadcast_children
    else
        return :send_to_parent
    end
end

# ============================================================================
# Policy Registration
# ============================================================================

"""Register default fleet policies."""
function register_default_fleet_policies!()
    register_fleet_policy!(:centralized, CentralizedFusion())
    register_fleet_policy!(:decentralized, DecentralizedFusion())
    register_fleet_policy!(:hierarchical, HierarchicalFusion())
    return nothing
end

# ============================================================================
# Exports
# ============================================================================

export CentralizedFusion, DecentralizedFusion, HierarchicalFusion
export initialize!, receive!, fuse!, share, get_fused_state
export covariance_intersection, optimize_omega
export set_hierarchy!, is_root, is_leaf, all_children_reported
export register_default_fleet_policies!
