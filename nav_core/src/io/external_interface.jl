# ============================================================================
# External Interface Layer
# ============================================================================
#
# Ported from AUV-Navigation/src/external_interface.jl
#
# Provides:
# 1. Structured logging for post-mission analysis
# 2. State export for external consumers (JSON, binary)
# 3. Measurement input parsing
# 4. Telemetry publisher interface
#
# Note: Actual ROS bindings would require RobotOS.jl package.
# ============================================================================

using JSON3
import Dates: now, UTC, DateTime

export LogLevel, LOG_DEBUG, LOG_INFO, LOG_WARN, LOG_ERROR, LOG_CRITICAL
export LogEntry, NavLogger
export log!, log_debug!, log_info!, log_warn!, log_error!, log_critical!
export flush_logs!
export NavStateExport, to_json, to_binary
export MeasurementPacket
export parse_imu_packet, parse_odometry_packet, parse_depth_packet, parse_ftm_packet
export packet_to_measurement
export sizeof_nav_state, sizeof_imu_measurement, sizeof_ftm_measurement
export AbstractTelemetryPublisher, publish!
export NullPublisher, LoggingPublisher

# ============================================================================
# Structured Logging
# ============================================================================

"""
    LogLevel

Logging severity levels.
"""
@enum LogLevel begin
    LOG_DEBUG = 1
    LOG_INFO = 2
    LOG_WARN = 3
    LOG_ERROR = 4
    LOG_CRITICAL = 5
end

"""
    LogEntry

Structured log entry for post-mission analysis.
"""
struct LogEntry
    timestamp::Float64           # Mission time (seconds)
    wall_time::DateTime          # Wall clock time
    level::LogLevel
    category::String             # e.g., "estimator", "magnetic", "feature"
    message::String
    data::Dict{String, Any}      # Additional structured data
end

"""
    NavLogger

Structured logger for navigation events.
"""
mutable struct NavLogger
    entries::Vector{LogEntry}
    file_path::Union{String, Nothing}
    min_level::LogLevel
    enabled::Bool
    max_entries::Int             # Memory limit
end

function NavLogger(;
    file_path::Union{String, Nothing} = nothing,
    min_level::LogLevel = LOG_INFO,
    max_entries::Int = 100000
)
    NavLogger(LogEntry[], file_path, min_level, true, max_entries)
end

"""
    log!(logger, level, category, message; data=Dict())

Add a log entry.
"""
function log!(logger::NavLogger, level::LogLevel, category::String, message::String;
              timestamp::Float64 = 0.0, data::Dict{String, Any} = Dict{String, Any}())
    if !logger.enabled || level < logger.min_level
        return nothing
    end

    entry = LogEntry(
        timestamp,
        now(UTC),
        level,
        category,
        message,
        data
    )
    push!(logger.entries, entry)

    # Memory management
    if length(logger.entries) > logger.max_entries
        popfirst!(logger.entries)
    end

    return entry
end

# Convenience functions
log_debug!(logger, cat, msg; kw...) = log!(logger, LOG_DEBUG, cat, msg; kw...)
log_info!(logger, cat, msg; kw...) = log!(logger, LOG_INFO, cat, msg; kw...)
log_warn!(logger, cat, msg; kw...) = log!(logger, LOG_WARN, cat, msg; kw...)
log_error!(logger, cat, msg; kw...) = log!(logger, LOG_ERROR, cat, msg; kw...)
log_critical!(logger, cat, msg; kw...) = log!(logger, LOG_CRITICAL, cat, msg; kw...)

"""
    flush_logs!(logger)

Write logs to file if file_path is set.
"""
function flush_logs!(logger::NavLogger)
    if logger.file_path === nothing || isempty(logger.entries)
        return 0
    end

    open(logger.file_path, "a") do io
        for entry in logger.entries
            json_entry = Dict(
                "t" => entry.timestamp,
                "wall" => string(entry.wall_time),
                "level" => string(entry.level),
                "cat" => entry.category,
                "msg" => entry.message,
                "data" => entry.data
            )
            println(io, JSON3.write(json_entry))
        end
    end

    n_written = length(logger.entries)
    empty!(logger.entries)
    return n_written
end

# ============================================================================
# State Export Formats
# ============================================================================

"""
    NavStateExport

Serializable navigation state for external consumers.
"""
struct NavStateExport
    timestamp::Float64
    position::Vector{Float64}          # [x, y, z]
    position_std::Vector{Float64}      # [σx, σy, σz]
    velocity::Vector{Float64}          # [vx, vy, vz]
    velocity_std::Vector{Float64}      # [σvx, σvy, σvz]
    orientation_quat::Vector{Float64}  # [w, x, y, z]
    orientation_std::Float64           # Approximate angular std
    altitude::Float64
    nav_state::String                  # "HEALTHY", "DEGRADED", "LOST"
    confidence::Float64                # 0-1
    confidence_label::String           # "HIGH", "MEDIUM", "LOW", "UNRELIABLE"
end

"""
    to_json(state::NavStateExport)

Convert state to JSON string.
"""
function to_json(state::NavStateExport)
    JSON3.write(Dict(
        "timestamp" => state.timestamp,
        "position" => state.position,
        "position_std" => state.position_std,
        "velocity" => state.velocity,
        "velocity_std" => state.velocity_std,
        "orientation_quat" => state.orientation_quat,
        "orientation_std" => state.orientation_std,
        "altitude" => state.altitude,
        "nav_state" => state.nav_state,
        "confidence" => state.confidence,
        "confidence_label" => state.confidence_label
    ))
end

"""
    to_binary(state::NavStateExport)

Convert state to binary format (fixed-size for C++ interop).

Format (64 bytes total):
- timestamp: Float64 (8 bytes)
- position: 3 × Float64 (24 bytes)
- velocity: 3 × Float64 (24 bytes)
- altitude: Float64 (8 bytes)
"""
function to_binary(state::NavStateExport)
    io = IOBuffer()
    write(io, Float64(state.timestamp))
    for p in state.position
        write(io, Float64(p))
    end
    for v in state.velocity
        write(io, Float64(v))
    end
    write(io, Float64(state.altitude))
    return take!(io)
end

# ============================================================================
# Measurement Input Parsing
# ============================================================================

"""
    MeasurementPacket

Generic measurement packet from external sensor.
"""
struct MeasurementPacket
    sensor_type::Symbol          # :imu, :odometry, :barometer, :ftm
    timestamp::Float64
    data::Vector{Float64}
    valid::Bool
end

"""
    parse_imu_packet(bytes)

Parse IMU measurement from binary packet.

Expected format (56 bytes):
- timestamp: Float64 (8 bytes)
- gyro: 3 × Float64 (24 bytes)
- accel: 3 × Float64 (24 bytes)
"""
function parse_imu_packet(bytes::Vector{UInt8})
    if length(bytes) != 56
        return MeasurementPacket(:imu, 0.0, Float64[], false)
    end

    io = IOBuffer(bytes)
    timestamp = read(io, Float64)
    gyro = [read(io, Float64) for _ in 1:3]
    accel = [read(io, Float64) for _ in 1:3]

    MeasurementPacket(:imu, timestamp, vcat(gyro, accel), true)
end

"""
    parse_odometry_packet(bytes)

Parse Odometry measurement from binary packet.

Expected format (32 bytes):
- timestamp: Float64 (8 bytes)
- velocity: 3 × Float64 (24 bytes)
"""
function parse_odometry_packet(bytes::Vector{UInt8})
    if length(bytes) != 32
        return MeasurementPacket(:odometry, 0.0, Float64[], false)
    end

    io = IOBuffer(bytes)
    timestamp = read(io, Float64)
    velocity = [read(io, Float64) for _ in 1:3]

    MeasurementPacket(:odometry, timestamp, velocity, true)
end

"""
    parse_depth_packet(bytes)

Parse barometer measurement from binary packet.

Expected format (16 bytes):
- timestamp: Float64 (8 bytes)
- altitude: Float64 (8 bytes)
"""
function parse_depth_packet(bytes::Vector{UInt8})
    if length(bytes) != 16
        return MeasurementPacket(:barometer, 0.0, Float64[], false)
    end

    io = IOBuffer(bytes)
    timestamp = read(io, Float64)
    altitude = read(io, Float64)

    MeasurementPacket(:barometer, timestamp, [altitude], true)
end

"""
    parse_ftm_packet(bytes)

Parse FTM (magnetic) measurement from binary packet.

Expected format (56 bytes):
- timestamp: Float64 (8 bytes)
- position: 3 × Float64 (24 bytes)
- field: 3 × Float64 (24 bytes)
"""
function parse_ftm_packet(bytes::Vector{UInt8})
    if length(bytes) != 56
        return MeasurementPacket(:ftm, 0.0, Float64[], false)
    end

    io = IOBuffer(bytes)
    timestamp = read(io, Float64)
    position = [read(io, Float64) for _ in 1:3]
    field = [read(io, Float64) for _ in 1:3]

    MeasurementPacket(:ftm, timestamp, vcat(position, field), true)
end

"""
    packet_to_measurement(packet::MeasurementPacket)

Convert parsed packet to internal measurement type.
"""
function packet_to_measurement(packet::MeasurementPacket)
    if !packet.valid
        return nothing
    end

    # Return raw data - conversion to specific measurement types
    # is handled by the calling code with access to type definitions
    return packet
end

# ============================================================================
# Binary Size Constants
# ============================================================================

sizeof_nav_state() = 64
sizeof_imu_measurement() = 56
sizeof_ftm_measurement() = 56

# ============================================================================
# Telemetry Publisher Interface
# ============================================================================

"""
    AbstractTelemetryPublisher

Abstract interface for publishing telemetry.

Implementations could use:
- UDP broadcast
- ROS topics
- ZeroMQ
- Shared memory
"""
abstract type AbstractTelemetryPublisher end

"""
    publish!(pub, state)

Publish navigation state.
"""
function publish!(pub::AbstractTelemetryPublisher, state::NavStateExport)
    error("publish! not implemented for $(typeof(pub))")
end

"""
    NullPublisher

No-op publisher for testing.
"""
struct NullPublisher <: AbstractTelemetryPublisher end

function publish!(::NullPublisher, state::NavStateExport)
    return true
end

"""
    LoggingPublisher

Publisher that logs to NavLogger.
"""
struct LoggingPublisher <: AbstractTelemetryPublisher
    logger::NavLogger
end

function publish!(pub::LoggingPublisher, state::NavStateExport)
    log_info!(pub.logger, "telemetry", "NavState published";
              timestamp = state.timestamp,
              data = Dict{String, Any}(
                  "position" => state.position,
                  "confidence" => state.confidence
              ))
    return true
end
