# ============================================================================
# MapArtifact.jl - Standardized Map Persistence Format
# ============================================================================
#
# Provides a stable artifact format for:
# - Audit trails (provenance tracking)
# - Regression diffs (deterministic comparison)
# - Fleet exchange (standardized schema)
# - Investor demos (reproducible results)
#
# Directory convention:
#   artifacts/maps/<world_type>/<scenario>/<seed>/<mission>.json
#
# Example:
#   artifacts/maps/moderate/collapse_curve/42/mission_3.json
#
# ============================================================================

using Dates
using JSON3
using SHA
using LinearAlgebra
using Statistics: mean

# ============================================================================
# Artifact Schema
# ============================================================================

"""
    MapArtifactMetadata

Metadata for a map artifact (immutable once created).

# Fields
- `artifact_version::String`: Schema version (e.g., "1.0.0")
- `created_at::DateTime`: When this artifact was created (UTC)
- `world_type::Symbol`: World class (:quiet, :moderate, :active)
- `world_hash::UInt64`: Hash of world configuration for reproducibility
- `scenario_hash::UInt64`: Hash of scenario parameters
- `seed::Int`: Random seed used
- `mission_index::Int`: Mission number (1-indexed)
- `trajectory_length::Int`: Number of trajectory points
- `trajectory_duration_s::Float64`: Duration in seconds
"""
struct MapArtifactMetadata
    artifact_version::String
    created_at::DateTime
    world_type::Symbol
    world_hash::UInt64
    scenario_hash::UInt64
    seed::Int
    mission_index::Int
    trajectory_length::Int
    trajectory_duration_s::Float64
end

"""
    TileArtifact

Serializable representation of a single map tile.

# Fields
- `tile_id::Tuple{Int, Int}`: Tile grid coordinates (i, j)
- `center::Vector{Float64}`: Tile center [x, y, z] in meters
- `coefficients::Vector{Float64}`: Map coefficients [B0(3), G5(5)]
- `covariance_diag::Vector{Float64}`: Diagonal of covariance (compact)
- `covariance_trace::Float64`: Trace of full covariance
- `observation_count::Int`: Number of observations incorporated
- `last_updated::DateTime`: When tile was last modified
"""
struct TileArtifact
    tile_id::Tuple{Int, Int}
    center::Vector{Float64}
    coefficients::Vector{Float64}
    covariance_diag::Vector{Float64}
    covariance_trace::Float64
    observation_count::Int
    last_updated::DateTime
end

"""
    ValidationSummary

Summary of validation metrics for the artifact.

# Fields
- `rmse_m::Float64`: Root mean square error [m]
- `nees_mean::Float64`: Mean normalized estimation error squared
- `nees_std::Float64`: NEES standard deviation
- `coverage_fraction::Float64`: Fraction of trajectory in map coverage
- `n_validation_points::Int`: Number of validation points
"""
struct ValidationSummary
    rmse_m::Float64
    nees_mean::Float64
    nees_std::Float64
    coverage_fraction::Float64
    n_validation_points::Int
end

"""
    LearningStatistics

Statistics about the learning process for this artifact.

# Fields
- `updates_attempted::Int`: Number of update attempts
- `updates_accepted::Int`: Number of accepted updates
- `updates_rejected_teachability::Int`: Rejected due to low teachability
- `updates_rejected_observability::Int`: Rejected due to poor observability
- `updates_rejected_quality::Int`: Rejected due to quality gate
- `mean_innovation_norm::Float64`: Mean innovation magnitude [T]
- `mean_sigma_map::Float64`: Mean map uncertainty [T]
"""
struct LearningStatistics
    updates_attempted::Int
    updates_accepted::Int
    updates_rejected_teachability::Int
    updates_rejected_observability::Int
    updates_rejected_quality::Int
    mean_innovation_norm::Float64
    mean_sigma_map::Float64
end

"""
    ProvenanceRecord

Full provenance chain for audit.

# Fields
- `parent_artifact_path::Union{Nothing, String}`: Path to parent artifact (or nothing if initial)
- `parent_hash::Union{Nothing, UInt64}`: Hash of parent artifact
- `creation_method::Symbol`: How this was created (:frozen, :learning, :fleet_fusion, :rollback)
- `source_missions::Vector{String}`: Mission IDs that contributed
- `notes::String`: Human-readable notes
"""
struct ProvenanceRecord
    parent_artifact_path::Union{Nothing, String}
    parent_hash::Union{Nothing, UInt64}
    creation_method::Symbol
    source_missions::Vector{String}
    notes::String
end

"""
    MapArtifact

Complete map artifact for persistence.

# Invariants
- Deterministic: same inputs → identical artifact
- Self-describing: contains all metadata for interpretation
- Auditable: full provenance chain
"""
struct MapArtifact
    metadata::MapArtifactMetadata
    tiles::Vector{TileArtifact}
    validation::Union{Nothing, ValidationSummary}
    learning::Union{Nothing, LearningStatistics}
    provenance::ProvenanceRecord
    content_hash::UInt64  # Hash of coefficients for integrity check
end

# ============================================================================
# Constructors
# ============================================================================

const ARTIFACT_VERSION = "1.0.0"

"""Create artifact metadata."""
function MapArtifactMetadata(;
    world_type::Symbol,
    world_hash::UInt64,
    scenario_hash::UInt64,
    seed::Int,
    mission_index::Int,
    trajectory_length::Int,
    trajectory_duration_s::Float64
)
    MapArtifactMetadata(
        ARTIFACT_VERSION,
        now(UTC),
        world_type,
        world_hash,
        scenario_hash,
        seed,
        mission_index,
        trajectory_length,
        trajectory_duration_s
    )
end

"""Create tile artifact from MapTileData."""
function TileArtifact(tile::MapTileData)
    TileArtifact(
        (tile.id.i, tile.id.j),
        Vector(tile.center),
        copy(tile.coefficients),
        diag(tile.covariance),
        tr(tile.covariance),
        tile.observation_count,
        now(UTC)
    )
end

"""Create validation summary from metrics."""
function ValidationSummary(;
    rmse_m::Float64,
    nees_mean::Float64,
    nees_std::Float64 = 0.0,
    coverage_fraction::Float64 = 1.0,
    n_validation_points::Int = 0
)
    ValidationSummary(rmse_m, nees_mean, nees_std, coverage_fraction, n_validation_points)
end

"""Create empty learning statistics."""
function LearningStatistics()
    LearningStatistics(0, 0, 0, 0, 0, 0.0, 0.0)
end

"""Create initial provenance (no parent)."""
function initial_provenance(mission_id::String; method::Symbol = :frozen, notes::String = "")
    ProvenanceRecord(nothing, nothing, method, [mission_id], notes)
end

"""Create learning provenance from parent."""
function learning_provenance(parent_path::String, parent_hash::UInt64, mission_id::String;
                             notes::String = "")
    ProvenanceRecord(parent_path, parent_hash, :learning, [mission_id], notes)
end

# ============================================================================
# Hashing
# ============================================================================

"""Compute content hash for artifact integrity."""
function compute_content_hash(tiles::Vector{TileArtifact})
    # Hash all coefficients in deterministic order
    all_coeffs = Float64[]
    for tile in sort(tiles, by = t -> t.tile_id)
        append!(all_coeffs, tile.coefficients)
    end

    # SHA256 of coefficient bytes
    bytes = reinterpret(UInt8, all_coeffs)
    hash_bytes = sha256(bytes)
    return reinterpret(UInt64, hash_bytes[1:8])[1]
end

"""Compute world configuration hash."""
function compute_world_hash(world_type::Symbol, gradient_scale::Float64, n_dipoles::Int, seed::Int)
    data = "world=$(world_type),grad=$(gradient_scale),dip=$(n_dipoles),seed=$(seed)"
    hash_bytes = sha256(Vector{UInt8}(data))
    return reinterpret(UInt64, hash_bytes[1:8])[1]
end

"""Compute scenario configuration hash."""
function compute_scenario_hash(trajectory_duration::Float64, trajectory_extent::Float64,
                               operating_depth::Float64, dt::Float64)
    data = "dur=$(trajectory_duration),ext=$(trajectory_extent),depth=$(operating_depth),dt=$(dt)"
    hash_bytes = sha256(Vector{UInt8}(data))
    return reinterpret(UInt64, hash_bytes[1:8])[1]
end

# ============================================================================
# Artifact Creation
# ============================================================================

"""
    create_map_artifact(tiles, metadata, provenance; validation, learning)

Create a complete map artifact.

# Arguments
- `tiles`: Vector of TileArtifact or MapTileData
- `metadata`: MapArtifactMetadata
- `provenance`: ProvenanceRecord

# Keyword Arguments
- `validation`: Optional ValidationSummary
- `learning`: Optional LearningStatistics
"""
function create_map_artifact(
    tiles::Vector{<:Union{TileArtifact, MapTileData}},
    metadata::MapArtifactMetadata,
    provenance::ProvenanceRecord;
    validation::Union{Nothing, ValidationSummary} = nothing,
    learning::Union{Nothing, LearningStatistics} = nothing
)
    # Convert MapTileData to TileArtifact if needed
    tile_artifacts = TileArtifact[
        t isa TileArtifact ? t : TileArtifact(t)
        for t in tiles
    ]

    content_hash = compute_content_hash(tile_artifacts)

    MapArtifact(metadata, tile_artifacts, validation, learning, provenance, content_hash)
end

# ============================================================================
# Serialization
# ============================================================================

"""Convert artifact to JSON-serializable dict."""
function to_dict(artifact::MapArtifact)
    Dict(
        "artifact_version" => artifact.metadata.artifact_version,
        "created_at" => string(artifact.metadata.created_at),
        "metadata" => Dict(
            "world_type" => string(artifact.metadata.world_type),
            "world_hash" => string(artifact.metadata.world_hash),
            "scenario_hash" => string(artifact.metadata.scenario_hash),
            "seed" => artifact.metadata.seed,
            "mission_index" => artifact.metadata.mission_index,
            "trajectory_length" => artifact.metadata.trajectory_length,
            "trajectory_duration_s" => artifact.metadata.trajectory_duration_s
        ),
        "tiles" => [
            Dict(
                "tile_id" => [t.tile_id[1], t.tile_id[2]],
                "center" => t.center,
                "coefficients" => t.coefficients,
                "covariance_diag" => t.covariance_diag,
                "covariance_trace" => t.covariance_trace,
                "observation_count" => t.observation_count,
                "last_updated" => string(t.last_updated)
            )
            for t in artifact.tiles
        ],
        "validation" => artifact.validation === nothing ? nothing : Dict(
            "rmse_m" => artifact.validation.rmse_m,
            "nees_mean" => artifact.validation.nees_mean,
            "nees_std" => artifact.validation.nees_std,
            "coverage_fraction" => artifact.validation.coverage_fraction,
            "n_validation_points" => artifact.validation.n_validation_points
        ),
        "learning" => artifact.learning === nothing ? nothing : Dict(
            "updates_attempted" => artifact.learning.updates_attempted,
            "updates_accepted" => artifact.learning.updates_accepted,
            "updates_rejected_teachability" => artifact.learning.updates_rejected_teachability,
            "updates_rejected_observability" => artifact.learning.updates_rejected_observability,
            "updates_rejected_quality" => artifact.learning.updates_rejected_quality,
            "mean_innovation_norm" => artifact.learning.mean_innovation_norm,
            "mean_sigma_map" => artifact.learning.mean_sigma_map
        ),
        "provenance" => Dict(
            "parent_artifact_path" => artifact.provenance.parent_artifact_path,
            "parent_hash" => artifact.provenance.parent_hash === nothing ? nothing : string(artifact.provenance.parent_hash),
            "creation_method" => string(artifact.provenance.creation_method),
            "source_missions" => artifact.provenance.source_missions,
            "notes" => artifact.provenance.notes
        ),
        "content_hash" => string(artifact.content_hash)
    )
end

"""
    save_artifact(artifact::MapArtifact, path::String)

Save artifact to JSON file.

# Directory Convention
Recommended path format:
  artifacts/maps/<world_type>/<scenario>/<seed>/mission_<N>.json
"""
function save_artifact(artifact::MapArtifact, path::String)
    mkpath(dirname(path))
    open(path, "w") do io
        JSON3.write(io, to_dict(artifact))
    end
    return path
end

"""
    save_artifact(artifact::MapArtifact; base_dir, scenario)

Save artifact using convention-based path.
"""
function save_artifact(artifact::MapArtifact;
                       base_dir::String = "artifacts/maps",
                       scenario::String = "default")
    path = joinpath(
        base_dir,
        string(artifact.metadata.world_type),
        scenario,
        string(artifact.metadata.seed),
        "mission_$(artifact.metadata.mission_index).json"
    )
    return save_artifact(artifact, path)
end

"""
    load_artifact(path::String) -> MapArtifact

Load artifact from JSON file.
"""
function load_artifact(path::String)
    data = JSON3.read(read(path, String))

    # Parse metadata
    md = data["metadata"]
    metadata = MapArtifactMetadata(
        data["artifact_version"],
        DateTime(data["created_at"]),
        Symbol(md["world_type"]),
        parse(UInt64, md["world_hash"]),
        parse(UInt64, md["scenario_hash"]),
        md["seed"],
        md["mission_index"],
        md["trajectory_length"],
        md["trajectory_duration_s"]
    )

    # Parse tiles
    tiles = TileArtifact[
        TileArtifact(
            (t["tile_id"][1], t["tile_id"][2]),
            Vector{Float64}(t["center"]),
            Vector{Float64}(t["coefficients"]),
            Vector{Float64}(t["covariance_diag"]),
            t["covariance_trace"],
            t["observation_count"],
            DateTime(t["last_updated"])
        )
        for t in data["tiles"]
    ]

    # Parse validation
    validation = data["validation"] === nothing ? nothing : ValidationSummary(
        data["validation"]["rmse_m"],
        data["validation"]["nees_mean"],
        data["validation"]["nees_std"],
        data["validation"]["coverage_fraction"],
        data["validation"]["n_validation_points"]
    )

    # Parse learning
    learning = data["learning"] === nothing ? nothing : LearningStatistics(
        data["learning"]["updates_attempted"],
        data["learning"]["updates_accepted"],
        data["learning"]["updates_rejected_teachability"],
        data["learning"]["updates_rejected_observability"],
        data["learning"]["updates_rejected_quality"],
        data["learning"]["mean_innovation_norm"],
        data["learning"]["mean_sigma_map"]
    )

    # Parse provenance
    prov = data["provenance"]
    provenance = ProvenanceRecord(
        prov["parent_artifact_path"],
        prov["parent_hash"] === nothing ? nothing : parse(UInt64, prov["parent_hash"]),
        Symbol(prov["creation_method"]),
        Vector{String}(prov["source_missions"]),
        prov["notes"]
    )

    content_hash = parse(UInt64, data["content_hash"])

    MapArtifact(metadata, tiles, validation, learning, provenance, content_hash)
end

# ============================================================================
# Verification
# ============================================================================

"""
    verify_artifact(artifact::MapArtifact) -> Bool

Verify artifact integrity (content hash matches).
"""
function verify_artifact(artifact::MapArtifact)
    computed_hash = compute_content_hash(artifact.tiles)
    return computed_hash == artifact.content_hash
end

"""
    verify_artifact(path::String) -> Bool

Verify artifact file integrity.
"""
function verify_artifact(path::String)
    artifact = load_artifact(path)
    return verify_artifact(artifact)
end

"""
    compare_artifacts(a1::MapArtifact, a2::MapArtifact) -> Dict

Compare two artifacts and return differences.
"""
function compare_artifacts(a1::MapArtifact, a2::MapArtifact)
    diffs = Dict{String, Any}()

    # Content hash comparison
    diffs["content_match"] = a1.content_hash == a2.content_hash

    # Coefficient differences (if same tiles)
    if length(a1.tiles) == length(a2.tiles)
        coeff_diffs = Float64[]
        for (t1, t2) in zip(sort(a1.tiles, by=t->t.tile_id), sort(a2.tiles, by=t->t.tile_id))
            if t1.tile_id == t2.tile_id
                push!(coeff_diffs, norm(t1.coefficients - t2.coefficients))
            end
        end
        diffs["max_coeff_diff"] = maximum(coeff_diffs)
        diffs["mean_coeff_diff"] = mean(coeff_diffs)
    end

    # Validation comparison
    if a1.validation !== nothing && a2.validation !== nothing
        diffs["rmse_diff_m"] = a2.validation.rmse_m - a1.validation.rmse_m
        diffs["nees_diff"] = a2.validation.nees_mean - a1.validation.nees_mean
    end

    return diffs
end

# ============================================================================
# Exports
# ============================================================================

export MapArtifactMetadata, TileArtifact, ValidationSummary, LearningStatistics
export ProvenanceRecord, MapArtifact
export initial_provenance, learning_provenance
export compute_content_hash, compute_world_hash, compute_scenario_hash
export create_map_artifact, save_artifact, load_artifact
export verify_artifact, compare_artifacts
export ARTIFACT_VERSION
