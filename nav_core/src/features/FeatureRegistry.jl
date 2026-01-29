# ============================================================================
# FeatureRegistry.jl - Registry pattern for feature types
# ============================================================================
#
# Features (dipoles, cables, etc.) are pluggable via this registry.
# Each feature type provides its own state representation, measurement model,
# and factor implementations.
#
# Usage:
#   register_feature!(:dipole, DipoleFeatureType())
#   feature = get_feature(:dipole)
#   state = initialize_feature(feature, params)
# ============================================================================

export AbstractFeatureType, FeatureRegistry
export register_feature!, get_feature, list_features, has_feature

"""
    AbstractFeatureType

Base type for feature type implementations.
Each feature type (dipole, cable, etc.) extends this.
"""
abstract type AbstractFeatureType end

"""
Required interface for feature types:
- `state_dim(feature)` - Dimension of feature state
- `initialize(feature, params)` - Create initial feature state
- `measurement_model(feature, state, nav_state)` - Predict measurement
- `jacobian(feature, state, nav_state)` - Jacobian of measurement model
- `create_factor(feature, measurement, state_idx)` - Create factor for graph
"""

"""
    FeatureRegistry

Global registry for feature type implementations.
"""
struct FeatureRegistry
    types::Dict{Symbol, AbstractFeatureType}
    lock::ReentrantLock
end

# Global singleton registry
const _FEATURE_REGISTRY = FeatureRegistry(Dict{Symbol, AbstractFeatureType}(), ReentrantLock())

"""
    register_feature!(feature_type::Symbol, impl::AbstractFeatureType)

Register a feature type implementation.
"""
function register_feature!(feature_type::Symbol, impl::AbstractFeatureType)
    lock(_FEATURE_REGISTRY.lock) do
        _FEATURE_REGISTRY.types[feature_type] = impl
    end
    return nothing
end

"""
    get_feature(feature_type::Symbol) -> AbstractFeatureType

Retrieve a registered feature type.
"""
function get_feature(feature_type::Symbol)
    lock(_FEATURE_REGISTRY.lock) do
        return _FEATURE_REGISTRY.types[feature_type]
    end
end

"""
    has_feature(feature_type::Symbol) -> Bool

Check if a feature type is registered.
"""
function has_feature(feature_type::Symbol)
    lock(_FEATURE_REGISTRY.lock) do
        return haskey(_FEATURE_REGISTRY.types, feature_type)
    end
end

"""
    list_features() -> Vector{Symbol}

List all registered feature types.
"""
function list_features()
    lock(_FEATURE_REGISTRY.lock) do
        return collect(keys(_FEATURE_REGISTRY.types))
    end
end

"""
    clear_features!()

Clear all registered features. Primarily for testing.
"""
function clear_features!()
    lock(_FEATURE_REGISTRY.lock) do
        empty!(_FEATURE_REGISTRY.types)
    end
    return nothing
end

# ============================================================================
# Feature type stubs (implementations in separate files)
# ============================================================================

# Dipole feature - magnetic dipole source
# Cable feature - linear magnetic source (future)
