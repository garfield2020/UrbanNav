# ============================================================================
# FactorContract.jl - Authoritative definition of factor graph factors
# ============================================================================
#
# This contract defines WHAT a factor must implement:
# - Residual computation
# - Jacobian computation
# - Noise model / whitening
#
# All factor implementations must conform to this interface.
# ============================================================================

export AbstractFactor, FactorMetadata
export residual, jacobian, noise_model, whiten
export factor_dim, connected_states

"""
    AbstractFactor

Base type for all factors in the factor graph.
A factor encodes a constraint between state variables.
"""
abstract type AbstractFactor end

"""
    FactorMetadata

Common metadata for all factors.

# Fields
- `id::Int` - Unique factor identifier
- `timestamp::Float64` - Time associated with factor [s]
- `type::Symbol` - Factor type identifier
"""
struct FactorMetadata
    id::Int
    timestamp::Float64
    type::Symbol
end

# ============================================================================
# Required interface methods (must be implemented by all factors)
# ============================================================================

"""
    residual(f::AbstractFactor, states...) -> AbstractVector

Compute the residual (innovation) for this factor.
Returns r = z - h(x) where z is measurement and h(x) is prediction.
"""
function residual end

"""
    jacobian(f::AbstractFactor, states...) -> Tuple{AbstractMatrix...}

Compute Jacobians of residual with respect to each connected state.
Returns tuple of matrices (J₁, J₂, ...) for each state.
"""
function jacobian end

"""
    noise_model(f::AbstractFactor) -> AbstractMatrix

Return the noise covariance Σ for this factor.
Used for whitening: r_whitened = Σ^(-1/2) * r
"""
function noise_model end

"""
    whiten(f::AbstractFactor, r::AbstractVector) -> AbstractVector

Apply whitening to residual using the factor's noise model.
Default implementation uses Cholesky decomposition.
"""
function whiten(f::AbstractFactor, r::AbstractVector)
    Σ = noise_model(f)
    L = cholesky(Symmetric(Σ)).L
    return L \ r
end

"""
    factor_dim(f::AbstractFactor) -> Int

Return the dimension of the residual vector.
"""
function factor_dim end

"""
    connected_states(f::AbstractFactor) -> Tuple{Int...}

Return indices of states connected by this factor.
"""
function connected_states end

# ============================================================================
# Standard factor types
# ============================================================================

"""
    PriorFactor{T} <: AbstractFactor

Prior constraint on a state variable.

# Fields
- `state_idx::Int` - Index of constrained state
- `prior::T` - Prior value
- `Σ::AbstractMatrix` - Prior covariance
- `meta::FactorMetadata`
"""
struct PriorFactor{T} <: AbstractFactor
    state_idx::Int
    prior::T
    Σ::Matrix{Float64}
    meta::FactorMetadata
end

factor_dim(f::PriorFactor) = length(f.prior)
connected_states(f::PriorFactor) = (f.state_idx,)
noise_model(f::PriorFactor) = f.Σ

"""
    BetweenFactor{T} <: AbstractFactor

Relative constraint between two states.

# Fields
- `state_i::Int` - First state index
- `state_j::Int` - Second state index
- `measurement::T` - Measured relative transformation
- `Σ::AbstractMatrix` - Measurement covariance
- `meta::FactorMetadata`
"""
struct BetweenFactor{T} <: AbstractFactor
    state_i::Int
    state_j::Int
    measurement::T
    Σ::Matrix{Float64}
    meta::FactorMetadata
end

factor_dim(f::BetweenFactor) = length(f.measurement)
connected_states(f::BetweenFactor) = (f.state_i, f.state_j)
noise_model(f::BetweenFactor) = f.Σ

"""
    MeasurementFactor{M<:AbstractMeasurement} <: AbstractFactor

Factor connecting a measurement to state.

# Fields
- `state_idx::Int` - State index
- `measurement::M` - The measurement
- `meta::FactorMetadata`
"""
struct MeasurementFactor{M<:AbstractMeasurement} <: AbstractFactor
    state_idx::Int
    measurement::M
    meta::FactorMetadata
end

factor_dim(f::MeasurementFactor) = measurement_dim(f.measurement)
connected_states(f::MeasurementFactor) = (f.state_idx,)
noise_model(f::MeasurementFactor) = Matrix(covariance(f.measurement))

# ============================================================================
# Residual and Jacobian implementations for standard factors
# ============================================================================

"""
    residual(f::PriorFactor, state::AbstractVector) -> Vector

Residual for prior: r = prior - state
"""
function residual(f::PriorFactor, state::AbstractVector)
    return collect(f.prior) - collect(state)
end

"""
    jacobian(f::PriorFactor, state::AbstractVector) -> Tuple{Matrix}

Jacobian of prior residual w.r.t. state: J = -I
"""
function jacobian(f::PriorFactor, state::AbstractVector)
    n = length(state)
    return (-Matrix{Float64}(I, n, n),)
end

"""
    residual(f::BetweenFactor, state_i::AbstractVector, state_j::AbstractVector) -> Vector

Residual for between: r = measurement - (state_j - state_i)
"""
function residual(f::BetweenFactor, state_i::AbstractVector, state_j::AbstractVector)
    return collect(f.measurement) - (collect(state_j) - collect(state_i))
end

"""
    jacobian(f::BetweenFactor, state_i::AbstractVector, state_j::AbstractVector) -> Tuple{Matrix, Matrix}

Jacobians of between residual: Ji = I, Jj = -I
"""
function jacobian(f::BetweenFactor, state_i::AbstractVector, state_j::AbstractVector)
    n = length(f.measurement)
    return (Matrix{Float64}(I, n, n), -Matrix{Float64}(I, n, n))
end
