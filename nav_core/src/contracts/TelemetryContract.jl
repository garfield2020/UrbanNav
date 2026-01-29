# ============================================================================
# TelemetryContract.jl - Authoritative definition of logging requirements
# ============================================================================
#
# This contract defines WHAT must be logged for:
# - Deterministic replay
# - Post-mission analysis
# - Debugging and diagnostics
#
# Telemetry is append-only and must not affect system behavior.
# ============================================================================

export TelemetryLevel, TelemetryRecord, TelemetrySchema
export required_fields, optional_fields

"""
    TelemetryLevel

Logging verbosity levels.

- MINIMAL: Only what's needed for replay
- STANDARD: Replay + basic diagnostics
- VERBOSE: Full debugging information
- DEBUG: Everything including internal state
"""
@enum TelemetryLevel begin
    MINIMAL = 1
    STANDARD = 2
    VERBOSE = 3
    DEBUG = 4
end

"""
    TelemetryRecord

A single telemetry record.

# Fields
- `timestamp::Float64` - Record timestamp [s]
- `level::TelemetryLevel` - Record verbosity level
- `category::Symbol` - Record category
- `data::Dict{Symbol,Any}` - Record payload
"""
struct TelemetryRecord
    timestamp::Float64
    level::TelemetryLevel
    category::Symbol
    data::Dict{Symbol,Any}
end

"""
    TelemetrySchema

Schema definition for a telemetry category.

# Fields
- `category::Symbol` - Category identifier
- `required::Vector{Tuple{Symbol,DataType}}` - Required fields
- `optional::Vector{Tuple{Symbol,DataType}}` - Optional fields
- `min_level::TelemetryLevel` - Minimum level to record
"""
struct TelemetrySchema
    category::Symbol
    required::Vector{Tuple{Symbol,DataType}}
    optional::Vector{Tuple{Symbol,DataType}}
    min_level::TelemetryLevel
end

# ============================================================================
# Required telemetry schemas (must be logged for replay)
# ============================================================================

"""
Required schemas for deterministic replay.
"""
const REPLAY_SCHEMAS = [
    TelemetrySchema(
        :measurement,
        [(:sensor_id, Symbol), (:timestamp, Float64), (:value, Vector{Float64})],
        [(:covariance, Matrix{Float64}), (:valid, Bool)],
        MINIMAL
    ),
    TelemetrySchema(
        :state_estimate,
        [(:timestamp, Float64), (:state, Vector{Float64})],
        [(:covariance_diag, Vector{Float64})],
        MINIMAL
    ),
    TelemetrySchema(
        :factor_added,
        [(:factor_id, Int), (:factor_type, Symbol), (:timestamp, Float64)],
        [(:connected_states, Vector{Int})],
        MINIMAL
    ),
    TelemetrySchema(
        :config,
        [(:timestamp, Float64), (:config_hash, String)],
        [(:config_data, Dict{Symbol,Any})],
        MINIMAL
    )
]

"""
Diagnostic schemas for analysis.
"""
const DIAGNOSTIC_SCHEMAS = [
    TelemetrySchema(
        :innovation,
        [(:factor_id, Int), (:timestamp, Float64), (:residual, Vector{Float64})],
        [(:whitened, Vector{Float64}), (:mahalanobis, Float64)],
        STANDARD
    ),
    TelemetrySchema(
        :health_event,
        [(:timestamp, Float64), (:subsystem, Symbol), (:state, Int)],
        [(:reason, String), (:metrics, Dict{Symbol,Float64})],
        STANDARD
    ),
    TelemetrySchema(
        :optimization,
        [(:timestamp, Float64), (:iterations, Int), (:final_error, Float64)],
        [(:lambda, Float64), (:convergence, Bool)],
        VERBOSE
    ),
    TelemetrySchema(
        :timing,
        [(:timestamp, Float64), (:operation, Symbol), (:duration_ms, Float64)],
        [],
        VERBOSE
    )
]

# ============================================================================
# Schema validation
# ============================================================================

"""
    required_fields(schema::TelemetrySchema) -> Vector{Symbol}

Get required field names for a schema.
"""
required_fields(schema::TelemetrySchema) = [f[1] for f in schema.required]

"""
    optional_fields(schema::TelemetrySchema) -> Vector{Symbol}

Get optional field names for a schema.
"""
optional_fields(schema::TelemetrySchema) = [f[1] for f in schema.optional]

"""
    validate_record(record::TelemetryRecord, schema::TelemetrySchema) -> Bool

Validate a telemetry record against its schema.
"""
function validate_record(record::TelemetryRecord, schema::TelemetrySchema)
    record.category == schema.category || return false
    for (field, T) in schema.required
        haskey(record.data, field) || return false
        record.data[field] isa T || return false
    end
    return true
end
