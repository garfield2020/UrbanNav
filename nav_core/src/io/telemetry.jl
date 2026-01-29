# ============================================================================
# telemetry.jl - Telemetry output for real-time monitoring
# ============================================================================

export TelemetryPublisher, publish!, create_publisher
export TelemetrySubscriber, subscribe!, unsubscribe!

"""
    AbstractTelemetryBackend

Base type for telemetry output backends.
"""
abstract type AbstractTelemetryBackend end

"""
    TelemetryPublisher

Publishes telemetry to registered backends.

# Fields
- `backends::Vector{AbstractTelemetryBackend}` - Output backends
- `enabled::Bool` - Global enable flag
- `rate_limit_hz::Float64` - Maximum publish rate
- `last_publish::Dict{Symbol, Float64}` - Last publish time per category
"""
mutable struct TelemetryPublisher
    backends::Vector{AbstractTelemetryBackend}
    enabled::Bool
    rate_limit_hz::Float64
    last_publish::Dict{Symbol, Float64}
    lock::ReentrantLock
end

"""
    create_publisher(; rate_limit_hz=100.0)

Create a telemetry publisher.
"""
function create_publisher(; rate_limit_hz::Float64=100.0)
    return TelemetryPublisher(
        AbstractTelemetryBackend[],
        true,
        rate_limit_hz,
        Dict{Symbol, Float64}(),
        ReentrantLock()
    )
end

"""
    add_backend!(publisher, backend)

Add an output backend to the publisher.
"""
function add_backend!(publisher::TelemetryPublisher, backend::AbstractTelemetryBackend)
    lock(publisher.lock) do
        push!(publisher.backends, backend)
    end
end

"""
    publish!(publisher, category, data, timestamp)

Publish telemetry data.
"""
function publish!(publisher::TelemetryPublisher, category::Symbol,
                  data::Dict{Symbol,Any}, timestamp::Float64)

    !publisher.enabled && return

    lock(publisher.lock) do
        # Rate limiting
        min_interval = 1.0 / publisher.rate_limit_hz
        last_time = get(publisher.last_publish, category, -Inf)

        if timestamp - last_time < min_interval
            return
        end

        publisher.last_publish[category] = timestamp

        # Publish to all backends
        for backend in publisher.backends
            try
                _publish(backend, category, data, timestamp)
            catch e
                @warn "Telemetry backend error" exception=e
            end
        end
    end
end

"""
    _publish(backend, category, data, timestamp)

Backend-specific publish implementation.
"""
function _publish end

# ============================================================================
# Standard backends
# ============================================================================

"""
    ConsoleBackend <: AbstractTelemetryBackend

Prints telemetry to console.
"""
struct ConsoleBackend <: AbstractTelemetryBackend
    categories::Set{Symbol}  # Empty = all
    verbose::Bool
end

ConsoleBackend(; categories=Symbol[], verbose=false) =
    ConsoleBackend(Set(categories), verbose)

function _publish(backend::ConsoleBackend, category::Symbol,
                  data::Dict{Symbol,Any}, timestamp::Float64)
    if !isempty(backend.categories) && !(category in backend.categories)
        return
    end

    if backend.verbose
        println("[$(round(timestamp, digits=3))] $category: $data")
    else
        println("[$(round(timestamp, digits=3))] $category")
    end
end

"""
    FileBackend <: AbstractTelemetryBackend

Writes telemetry to file.
"""
struct FileBackend <: AbstractTelemetryBackend
    path::String
    io::IOStream
end

function FileBackend(path::String)
    io = open(path, "w")
    return FileBackend(path, io)
end

function _publish(backend::FileBackend, category::Symbol,
                  data::Dict{Symbol,Any}, timestamp::Float64)
    println(backend.io, "$timestamp,$category,$(repr(data))")
    flush(backend.io)
end

"""
    CallbackBackend <: AbstractTelemetryBackend

Calls a function with telemetry data.
"""
struct CallbackBackend <: AbstractTelemetryBackend
    callback::Function
end

function _publish(backend::CallbackBackend, category::Symbol,
                  data::Dict{Symbol,Any}, timestamp::Float64)
    backend.callback(category, data, timestamp)
end

# ============================================================================
# Standard telemetry messages
# ============================================================================

"""
    publish_state!(publisher, state, covariance, timestamp)

Publish current state estimate.
"""
function publish_state!(publisher::TelemetryPublisher, state::AbstractVector,
                        covariance::AbstractMatrix, timestamp::Float64)
    data = Dict{Symbol,Any}(
        :state => collect(state),
        :covariance_diag => diag(covariance)
    )
    publish!(publisher, :state, data, timestamp)
end

"""
    publish_health!(publisher, report)

Publish health report.
"""
function publish_health!(publisher::TelemetryPublisher, report::HealthReport)
    data = Dict{Symbol,Any}(
        :overall => Int(report.overall),
        :subsystems => Dict(k => Int(v) for (k,v) in report.subsystems),
        :metrics => report.metrics
    )
    publish!(publisher, :health, data, report.timestamp)
end
