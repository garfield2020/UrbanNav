# ============================================================================
# FleetRegistry.jl - Registry pattern for fleet fusion policies
# ============================================================================
#
# Fleet fusion policies determine how multiple vehicles share information.
# Different policies can be registered and selected via configuration.
#
# Usage:
#   register_fleet_policy!(:centralized, CentralizedFusion())
#   policy = get_fleet_policy(:centralized)
#   fuse!(policy, vehicle_states, measurements)
# ============================================================================

export AbstractFleetPolicy, FleetRegistry
export register_fleet_policy!, get_fleet_policy, list_fleet_policies

"""
    AbstractFleetPolicy

Base type for fleet fusion policy implementations.
"""
abstract type AbstractFleetPolicy end

"""
Required interface for fleet policies:
- `initialize!(policy, config, num_vehicles)` - Initialize policy
- `fuse!(policy, states, measurements)` - Perform fusion step
- `share(policy, vehicle_id, state)` - Determine what to share
- `receive!(policy, vehicle_id, shared_data)` - Process received data
"""

"""
    FleetRegistry

Global registry for fleet fusion policies.
"""
struct FleetRegistry
    policies::Dict{Symbol, AbstractFleetPolicy}
    lock::ReentrantLock
end

# Global singleton registry
const _FLEET_REGISTRY = FleetRegistry(Dict{Symbol, AbstractFleetPolicy}(), ReentrantLock())

"""
    register_fleet_policy!(policy_type::Symbol, impl::AbstractFleetPolicy)

Register a fleet fusion policy implementation.
"""
function register_fleet_policy!(policy_type::Symbol, impl::AbstractFleetPolicy)
    lock(_FLEET_REGISTRY.lock) do
        _FLEET_REGISTRY.policies[policy_type] = impl
    end
    return nothing
end

"""
    get_fleet_policy(policy_type::Symbol) -> AbstractFleetPolicy

Retrieve a registered fleet policy.
"""
function get_fleet_policy(policy_type::Symbol)
    lock(_FLEET_REGISTRY.lock) do
        return _FLEET_REGISTRY.policies[policy_type]
    end
end

"""
    has_fleet_policy(policy_type::Symbol) -> Bool

Check if a fleet policy is registered.
"""
function has_fleet_policy(policy_type::Symbol)
    lock(_FLEET_REGISTRY.lock) do
        return haskey(_FLEET_REGISTRY.policies, policy_type)
    end
end

"""
    list_fleet_policies() -> Vector{Symbol}

List all registered fleet policies.
"""
function list_fleet_policies()
    lock(_FLEET_REGISTRY.lock) do
        return collect(keys(_FLEET_REGISTRY.policies))
    end
end

"""
    clear_fleet_policies!()

Clear all registered policies. Primarily for testing.
"""
function clear_fleet_policies!()
    lock(_FLEET_REGISTRY.lock) do
        empty!(_FLEET_REGISTRY.policies)
    end
    return nothing
end

# ============================================================================
# Fleet policy stubs (implementations in separate files)
# ============================================================================

# :centralized - All vehicles report to central node
# :decentralized - Peer-to-peer fusion
# :hierarchical - Tree-structured fusion
