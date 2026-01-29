# ============================================================================
# logging.jl - Deterministic logging for replay and diagnostics
# ============================================================================
#
# Logging is APPEND-ONLY and must not affect system behavior.
# All logged data must be sufficient for deterministic replay.
# ============================================================================

export Logger, log!, flush!, create_logger
export LogLevel, DEBUG, INFO, WARN, ERROR

@enum LogLevel begin
    DEBUG = 1
    INFO = 2
    WARN = 3
    ERROR = 4
end

"""
    Logger

Structured logger for navigation system.

# Fields
- `records::Vector{TelemetryRecord}` - Buffered records
- `level::TelemetryLevel` - Minimum level to record
- `output_path::String` - Output file path (empty for in-memory)
- `buffer_size::Int` - Records before auto-flush
- `deterministic::Bool` - Enable deterministic replay logging
"""
mutable struct Logger
    records::Vector{TelemetryRecord}
    level::TelemetryLevel
    output_path::String
    buffer_size::Int
    deterministic::Bool
    lock::ReentrantLock
end

"""
    create_logger(; level=STANDARD, path="", buffer_size=1000, deterministic=true)

Create a new logger.
"""
function create_logger(; level::TelemetryLevel=STANDARD,
                        path::String="",
                        buffer_size::Int=1000,
                        deterministic::Bool=true)
    return Logger(
        TelemetryRecord[],
        level,
        path,
        buffer_size,
        deterministic,
        ReentrantLock()
    )
end

"""
    log!(logger, category, data; level=STANDARD)

Log a record.
"""
function log!(logger::Logger, category::Symbol, data::Dict{Symbol,Any};
              level::TelemetryLevel=STANDARD, timestamp::Float64=time())

    Int(level) < Int(logger.level) && return

    record = TelemetryRecord(timestamp, level, category, data)

    lock(logger.lock) do
        push!(logger.records, record)

        if length(logger.records) >= logger.buffer_size
            _flush_locked!(logger)
        end
    end
end

"""
    flush!(logger)

Flush buffered records to output.
"""
function flush!(logger::Logger)
    lock(logger.lock) do
        _flush_locked!(logger)
    end
end

function _flush_locked!(logger::Logger)
    isempty(logger.records) && return

    if !isempty(logger.output_path)
        open(logger.output_path, "a") do io
            for record in logger.records
                _write_record(io, record)
            end
        end
    end

    empty!(logger.records)
end

function _write_record(io::IO, record::TelemetryRecord)
    # Simple JSON-like format
    println(io, "{\"timestamp\":$(record.timestamp),\"level\":$(Int(record.level)),\"category\":\"$(record.category)\",\"data\":$(repr(record.data))}")
end

# ============================================================================
# Convenience logging functions
# ============================================================================

"""
    log_measurement!(logger, sensor_id, timestamp, value; covariance=nothing)

Log a measurement for replay.
"""
function log_measurement!(logger::Logger, sensor_id::Symbol, timestamp::Float64,
                          value::AbstractVector; covariance=nothing)
    data = Dict{Symbol,Any}(
        :sensor_id => sensor_id,
        :timestamp => timestamp,
        :value => collect(value)
    )
    if covariance !== nothing
        data[:covariance] = Matrix(covariance)
    end
    log!(logger, :measurement, data; level=MINIMAL, timestamp=timestamp)
end

"""
    log_state!(logger, timestamp, state; covariance_diag=nothing)

Log state estimate for replay.
"""
function log_state!(logger::Logger, timestamp::Float64, state::AbstractVector;
                    covariance_diag=nothing)
    data = Dict{Symbol,Any}(
        :timestamp => timestamp,
        :state => collect(state)
    )
    if covariance_diag !== nothing
        data[:covariance_diag] = collect(covariance_diag)
    end
    log!(logger, :state_estimate, data; level=MINIMAL, timestamp=timestamp)
end

"""
    log_health!(logger, timestamp, subsystem, state; reason="", metrics=nothing)

Log health event.
"""
function log_health!(logger::Logger, timestamp::Float64, subsystem::Symbol,
                     state::HealthState; reason::String="", metrics=nothing)
    data = Dict{Symbol,Any}(
        :timestamp => timestamp,
        :subsystem => subsystem,
        :state => Int(state)
    )
    if !isempty(reason)
        data[:reason] = reason
    end
    if metrics !== nothing
        data[:metrics] = Dict(metrics)
    end
    log!(logger, :health_event, data; level=STANDARD, timestamp=timestamp)
end
