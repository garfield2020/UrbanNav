# ============================================================================
# Map Provider - Abstract Interface for Map Data Sources (Phase A)
# ============================================================================
#
# Defines the pluggable interface for map-backed magnetic field prediction.
#
# Design Principle:
# The estimator should be agnostic to map source - it only needs:
# 1. A way to load a map
# 2. A way to query predictions with uncertainties
#
# Implementations:
# - FrozenFileMapProvider: Production - loads pre-computed map from disk
# - TruthMapProvider: Simulation only - samples truth world (in experiments/)
# ============================================================================

using LinearAlgebra
using StaticArrays
using JSON3
using Dates

# Import from MapContract and MapBasis (assumed in same module)
# These provide: MapModel, MapMetadata, MapQueryResult, LinearHarmonicModel, etc.

# ============================================================================
# Map Provider Interface
# ============================================================================

"""
    AbstractMapProvider

Abstract interface for map data sources.

All implementations must provide:
- `load_map(provider, source)::MapModel`
- `query_map(provider, map, position)::MapQueryResult`

The interface is designed to be:
- Stateless: provider holds configuration, not map data
- Deterministic: same inputs produce same outputs
- Pure: no side effects on query
"""
abstract type AbstractMapProvider end

# Default implementations that subclasses can override

"""
    supports_query_type(provider::AbstractMapProvider, query_type::Symbol) -> Bool

Check if provider supports a query type.
Supported types: :field_only, :field_and_gradient
"""
function supports_query_type(provider::AbstractMapProvider, query_type::Symbol)
    return query_type in [:field_only, :field_and_gradient]
end

"""
    get_coverage_bounds(provider::AbstractMapProvider, map::MapModel) -> (min, max)

Get map coverage bounds as (min_corner, max_corner) tuples.
"""
function get_coverage_bounds(provider::AbstractMapProvider, map::MapModel)
    return map.metadata.coverage_bounds
end

# ============================================================================
# Frozen File Map Provider (Production)
# ============================================================================

"""
    FrozenFileMapProvider

Production map provider that loads pre-computed maps from JSON files.

# Configuration
- `default_σ_B::Float64`: Default field uncertainty [T] if not in file
- `default_σ_G::Float64`: Default gradient uncertainty [T/m] if not in file
- `extrapolation_penalty::Float64`: Uncertainty multiplier outside coverage

# File Format
Maps are stored as JSON with structure:
```json
{
    "version": "1.0.0",
    "metadata": { ... },
    "model": {
        "type": "linear_harmonic",
        "B0": [Bx, By, Bz],
        "G0": [[Gxx, Gxy, Gxz], [Gyx, Gyy, Gyz], [Gzx, Gzy, Gzz]],
        "x_ref": [x, y, z],
        "coefficients_covariance": [[...], ...]
    }
}
```

# Usage
```julia
provider = FrozenFileMapProvider()
map = load_map(provider, "path/to/map_v1.json")
result = query_map(provider, map, position)
```
"""
struct FrozenFileMapProvider <: AbstractMapProvider
    default_σ_B::Float64        # Default field uncertainty [T]
    default_σ_G::Float64        # Default gradient uncertainty [T/m]
    extrapolation_penalty::Float64  # Uncertainty multiplier outside coverage
end

function FrozenFileMapProvider(;
    default_σ_B::Float64 = 100e-9,   # 100 nT default uncertainty
    default_σ_G::Float64 = 50e-9,    # 50 nT/m default uncertainty
    extrapolation_penalty::Float64 = 10.0  # 10× uncertainty outside coverage
)
    FrozenFileMapProvider(default_σ_B, default_σ_G, extrapolation_penalty)
end

"""
    load_map(provider::FrozenFileMapProvider, path::String) -> MapModel

Load frozen map from JSON file.
"""
function load_map(provider::FrozenFileMapProvider, path::String)
    @assert isfile(path) "Map file not found: $path"

    json_str = read(path, String)
    data = JSON3.read(json_str)

    # Parse metadata
    meta = data[:metadata]
    metadata = MapMetadata(
        version = string(get(data, :version, "1.0.0")),
        mission_id = string(get(meta, :mission_id, "unknown")),
        frame = MapFrame(),  # Default NED world
        basis_type = MAP_BASIS_LINEAR,
        tile_size = Float64(get(meta, :tile_size, 0.0)),
        coverage_min = get(meta, :coverage_min, [-Inf, -Inf, -Inf]),
        coverage_max = get(meta, :coverage_max, [Inf, Inf, Inf])
    )

    # Parse model
    model_data = data[:model]
    model_type = string(get(model_data, :type, "linear_harmonic"))

    @assert model_type == "linear_harmonic" "Only linear_harmonic model supported, got: $model_type"

    B0 = Vec3Map(model_data[:B0]...)
    G0_raw = model_data[:G0]
    G0 = Mat3Map(
        G0_raw[1][1], G0_raw[2][1], G0_raw[3][1],
        G0_raw[1][2], G0_raw[2][2], G0_raw[3][2],
        G0_raw[1][3], G0_raw[2][3], G0_raw[3][3]
    )
    x_ref = Vec3Map(model_data[:x_ref]...)

    # Parse covariance (or use defaults)
    if haskey(model_data, :coefficients_covariance)
        cov_raw = model_data[:coefficients_covariance]
        n = length(cov_raw)
        Σ = zeros(n, n)
        for i in 1:n
            for j in 1:n
                Σ[i, j] = cov_raw[i][j]
            end
        end
    else
        # Default diagonal covariance
        Σ = diagm(vcat(
            fill(provider.default_σ_B^2, 3),
            fill(provider.default_σ_G^2, 5)
        ))
    end

    # Pack into MapModel format
    # Convert LinearHarmonicModel coefficients to MapTileData format
    coefficients = vcat(Vector(B0), Vector(pack_gradient(G0)))
    observation_count = Int(get(model_data, :observation_count, 0))

    return MapModel(
        metadata = metadata,
        coefficients = coefficients,
        covariance = Σ,
        center = Vector(x_ref),
        observation_count = observation_count
    )
end

"""
    query_map(provider::FrozenFileMapProvider, map::MapModel, position::AbstractVector) -> MapQueryResult

Query frozen map at position.

Returns field and gradient predictions with associated uncertainties.
Uncertainties are inflated by extrapolation_penalty if outside coverage.
"""
function query_map(provider::FrozenFileMapProvider, map::MapModel, position::AbstractVector)
    pos = Vec3Map(position...)

    # Get tile data
    tile = get_tile(map, pos)
    if tile === nothing
        # No data - return high uncertainty
        return MapQueryResult(
            B_pred = Vec3Map(0.0, 0.0, 0.0),
            G_pred = Mat3Map(zeros(3, 3)),
            σ_B = provider.default_σ_B * provider.extrapolation_penalty,
            σ_G = provider.default_σ_G * provider.extrapolation_penalty,
            in_coverage = false,
            tile_id = MapTileID(0, 0)
        )
    end

    # Unpack coefficients
    B0 = Vec3Map(tile.coefficients[1:3]...)
    G5 = tile.coefficients[4:8]
    G0 = unpack_gradient(G5)
    x_ref = tile.center

    # Predict field: B(x) = B₀ + G₀·(x - x_ref)
    dx = pos - x_ref
    B_pred = B0 + G0 * dx

    # Gradient is constant for linear model
    G_pred = G0

    # Compute uncertainties from coefficient covariance
    Σ_coeffs = tile.covariance
    Σ_B0 = Σ_coeffs[1:3, 1:3]
    Σ_G5 = Σ_coeffs[4:8, 4:8]

    # Field uncertainty: propagate coefficient uncertainty to prediction
    J_G = compute_field_gradient_jacobian_internal(dx)
    Σ_B_mat = Σ_B0 + J_G * Σ_G5 * J_G'

    # Apply extrapolation penalty if outside coverage
    in_coverage = has_coverage(map, pos)
    if !in_coverage
        penalty = provider.extrapolation_penalty^2
        Σ_B_mat *= penalty
        Σ_G5 *= penalty
    end

    return MapQueryResult(
        Vec3Map(B_pred...),
        Mat3Map(G_pred...),
        Mat3Map(Σ_B_mat...),
        SMatrix{5,5}(Σ_G5...),
        SMatrix{3,5}(zeros(3, 5)),  # Cross-covariance (zero for independent)
        in_coverage,
        tile.id
    )
end

"""Internal helper for field-gradient Jacobian."""
function compute_field_gradient_jacobian_internal(dx::AbstractVector)
    x, y, z = dx[1], dx[2], dx[3]
    return @SMatrix [
        x    0.0   y    z    0.0;
        0.0  y     x    0.0  z  ;
        -z   -z    0.0  x    y
    ]
end

# ============================================================================
# Map File I/O Utilities
# ============================================================================

"""
    save_map(map::MapModel, path::String; model::LinearHarmonicModel=nothing)

Save map to JSON file.
"""
function save_map(map::MapModel, path::String; model = nothing)
    tile = map.global_tile
    if tile === nothing && !isempty(map.tiles)
        tile = first(values(map.tiles))
    end

    @assert tile !== nothing "Map has no tile data"

    # Unpack coefficients
    B0 = tile.coefficients[1:3]
    G5 = tile.coefficients[4:8]
    G0 = unpack_gradient(G5)

    data = Dict(
        :version => map.metadata.version,
        :metadata => Dict(
            :mission_id => map.metadata.mission_id,
            :created => string(map.metadata.created),
            :tile_size => map.metadata.tile_size,
            :coverage_min => Vector(map.metadata.coverage_bounds[1]),
            :coverage_max => Vector(map.metadata.coverage_bounds[2])
        ),
        :model => Dict(
            :type => "linear_harmonic",
            :B0 => B0,
            :G0 => [[G0[1,1], G0[1,2], G0[1,3]],
                    [G0[2,1], G0[2,2], G0[2,3]],
                    [G0[3,1], G0[3,2], G0[3,3]]],
            :x_ref => Vector(tile.center),
            :coefficients_covariance => [tile.covariance[i, :] for i in 1:size(tile.covariance, 1)],
            :observation_count => tile.observation_count
        )
    )

    json_str = JSON3.write(data)
    mkpath(dirname(path))
    write(path, json_str)
end

"""
    convert_harmonic_model_to_map(model::LinearHarmonicModel; mission_id="", coverage_radius=100.0)

Convert LinearHarmonicModel to MapModel for saving.
"""
function convert_harmonic_model_to_map(model;
                                        mission_id::String = "converted",
                                        coverage_radius::Float64 = 100.0)
    x_ref = model.x_ref
    coverage_min = x_ref .- coverage_radius
    coverage_max = x_ref .+ coverage_radius

    metadata = MapMetadata(
        version = "1.0.0",
        mission_id = mission_id,
        basis_type = MAP_BASIS_LINEAR,
        tile_size = 0.0,
        coverage_min = coverage_min,
        coverage_max = coverage_max
    )

    coefficients = vcat(Vector(model.B0), Vector(pack_gradient(model.G0)))

    return MapModel(
        metadata = metadata,
        coefficients = coefficients,
        covariance = model.Σ_coeffs,
        center = Vector(x_ref),
        observation_count = 0
    )
end

# ============================================================================
# Null Provider (for testing/fallback)
# ============================================================================

"""
    NullMapProvider

Fallback provider that returns constant field with high uncertainty.

Useful for:
- Testing estimator without map
- Graceful degradation when map unavailable
"""
struct NullMapProvider <: AbstractMapProvider
    B_default::Vec3Map       # Default field [T]
    σ_B::Float64             # Field uncertainty [T]
    σ_G::Float64             # Gradient uncertainty [T/m]
end

function NullMapProvider(;
    B_default::AbstractVector = [20e-6, 5e-6, -45e-6],  # Typical Earth field
    σ_B::Float64 = 1000e-9,   # 1000 nT (very uncertain)
    σ_G::Float64 = 100e-9     # 100 nT/m
)
    NullMapProvider(Vec3Map(B_default...), σ_B, σ_G)
end

function load_map(provider::NullMapProvider, path::String)
    # Return empty map model
    metadata = MapMetadata(
        version = "0.0.0",
        mission_id = "null",
        basis_type = MAP_BASIS_LINEAR,
        tile_size = 0.0
    )

    return MapModel(
        metadata = metadata,
        coefficients = vcat(Vector(provider.B_default), zeros(5)),
        covariance = diagm(vcat(fill(provider.σ_B^2, 3), fill(provider.σ_G^2, 5))),
        center = [0.0, 0.0, 0.0],
        observation_count = 0
    )
end

function query_map(provider::NullMapProvider, map::MapModel, position::AbstractVector)
    return MapQueryResult(
        B_pred = provider.B_default,
        G_pred = Mat3Map(zeros(3, 3)),
        σ_B = provider.σ_B,
        σ_G = provider.σ_G,
        in_coverage = false,
        tile_id = MapTileID(0, 0)
    )
end

# ============================================================================
# Exports
# ============================================================================

export AbstractMapProvider, supports_query_type, get_coverage_bounds
export FrozenFileMapProvider, load_map, query_map, save_map
export convert_harmonic_model_to_map
export NullMapProvider
