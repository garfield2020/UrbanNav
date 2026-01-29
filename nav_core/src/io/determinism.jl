# ============================================================================
# Determinism and Reproducibility
# ============================================================================
#
# Ported from AUV-Navigation/src/determinism.jl
#
# Provides:
# 1. Seeded random number generation
# 2. Deterministic iteration utilities
# 3. Reproducibility audit tools
# 4. State hashing for verification
#
# All randomness in the system should flow through these utilities.
# ============================================================================

using Random
using SHA

export DEFAULT_SEED, set_global_seed!, get_global_rng, reset_global_rng!
export sorted_keys, sorted_pairs, sorted_values, iterate_deterministic
export StateHash, hash_state, hash_dict, compare_hashes, hash_to_hex
export ReproducibilityAudit, record_snapshot!, verify_reproducibility
export DeterminismConfig, DeterminismResult, verify_determinism
export SolverLog, SolverAudit, log_solver_iteration!, compare_solver_audits
export with_seed, seeded_randn, seeded_rand
export fp_equal, fp_hash

# ============================================================================
# Global Seeded RNG
# ============================================================================

"""Default seed for reproducibility."""
const DEFAULT_SEED = 42

"""Global RNG state for deterministic execution."""
const GLOBAL_RNG = Ref{MersenneTwister}(MersenneTwister(DEFAULT_SEED))

"""
    set_global_seed!(seed::Int)

Set the global RNG seed for reproducibility.
Must be called at start of simulation/test for determinism.
"""
function set_global_seed!(seed::Int)
    GLOBAL_RNG[] = MersenneTwister(seed)
    return seed
end

"""
    get_global_rng()

Get the global seeded RNG.
Use this instead of `Random.default_rng()` for determinism.
"""
get_global_rng() = GLOBAL_RNG[]

"""
    reset_global_rng!()

Reset the global RNG to default seed.
"""
reset_global_rng!() = set_global_seed!(DEFAULT_SEED)

# ============================================================================
# Deterministic Iteration
# ============================================================================

"""
    sorted_keys(d::Dict)

Return keys of dictionary in sorted order for deterministic iteration.
"""
sorted_keys(d::Dict) = sort(collect(keys(d)))

"""
    sorted_pairs(d::Dict)

Return key-value pairs in sorted key order for deterministic iteration.
"""
sorted_pairs(d::Dict) = [(k, d[k]) for k in sorted_keys(d)]

"""
    sorted_values(d::Dict)

Return values in sorted key order for deterministic iteration.
"""
sorted_values(d::Dict) = [d[k] for k in sorted_keys(d)]

"""
    iterate_deterministic(d::Dict)

Iterate over dictionary in deterministic (sorted key) order.
"""
iterate_deterministic(d::Dict) = sorted_pairs(d)

# ============================================================================
# State Hashing
# ============================================================================

"""
    StateHash

Hash of system state for reproducibility verification.
"""
struct StateHash
    hash::Vector{UInt8}
    timestamp::Float64
    description::String
end

"""
    hash_state(state::AbstractVector; description="")

Compute hash of numeric state vector.
"""
function hash_state(state::AbstractVector; description::String = "")
    bytes = reinterpret(UInt8, Vector{Float64}(state))
    h = sha256(bytes)
    StateHash(h, time(), description)
end

"""
    hash_dict(d::Dict; description="")

Compute deterministic hash of dictionary contents.
"""
function hash_dict(d::Dict; description::String = "")
    io = IOBuffer()
    for k in sorted_keys(d)
        write(io, string(k))
        write(io, string(d[k]))
    end
    h = sha256(take!(io))
    StateHash(h, time(), description)
end

"""
    compare_hashes(h1::StateHash, h2::StateHash)

Compare two state hashes for equality.
"""
compare_hashes(h1::StateHash, h2::StateHash) = h1.hash == h2.hash

"""
    hash_to_hex(h::StateHash)

Convert hash to hex string for display.
"""
hash_to_hex(h::StateHash) = bytes2hex(h.hash)

# ============================================================================
# Reproducibility Audit
# ============================================================================

"""
    ReproducibilityAudit

Records state snapshots for reproducibility verification.
"""
mutable struct ReproducibilityAudit
    seed::Int
    snapshots::Vector{StateHash}
    tolerances::Dict{String, Float64}
    warnings::Vector{String}
end

function ReproducibilityAudit(seed::Int = DEFAULT_SEED)
    ReproducibilityAudit(
        seed,
        StateHash[],
        Dict{String, Float64}(
            "position" => 1e-10,
            "rotation" => 1e-10,
            "covariance" => 1e-8
        ),
        String[]
    )
end

"""
    record_snapshot!(audit, state; description="")

Record a state snapshot for later comparison.
"""
function record_snapshot!(audit::ReproducibilityAudit, state::AbstractVector;
                          description::String = "")
    h = hash_state(state; description = description)
    push!(audit.snapshots, h)
    return h
end

"""
    verify_reproducibility(audit1, audit2)

Verify two audits have identical snapshots.
Returns (passed, mismatches).
"""
function verify_reproducibility(audit1::ReproducibilityAudit, audit2::ReproducibilityAudit)
    if length(audit1.snapshots) != length(audit2.snapshots)
        return (false, ["Snapshot count mismatch: $(length(audit1.snapshots)) vs $(length(audit2.snapshots))"])
    end

    mismatches = String[]
    for i in 1:length(audit1.snapshots)
        if !compare_hashes(audit1.snapshots[i], audit2.snapshots[i])
            push!(mismatches, "Snapshot $i: $(audit1.snapshots[i].description)")
        end
    end

    return (isempty(mismatches), mismatches)
end

# ============================================================================
# Determinism Verification
# ============================================================================

"""
    DeterminismConfig

Configuration for determinism testing.
"""
struct DeterminismConfig
    seed::Int
    n_runs::Int
    fp_tolerance::Float64
    log_solver_params::Bool
end

function DeterminismConfig(;
    seed::Int = DEFAULT_SEED,
    n_runs::Int = 3,
    fp_tolerance::Real = 1e-10,
    log_solver_params::Bool = true
)
    DeterminismConfig(seed, n_runs, Float64(fp_tolerance), log_solver_params)
end

"""
    DeterminismResult

Result of determinism verification.
"""
struct DeterminismResult
    passed::Bool
    max_deviation::Float64
    deviations::Vector{Float64}
    run_hashes::Vector{String}
    warnings::Vector{String}
end

"""
    verify_determinism(f::Function; config::DeterminismConfig)

Run function multiple times and verify identical outputs.
The function `f(rng)` should take an RNG and return a state vector.
"""
function verify_determinism(f::Function; config::DeterminismConfig = DeterminismConfig())
    results = Vector{Vector{Float64}}()
    hashes = String[]

    for run in 1:config.n_runs
        rng = MersenneTwister(config.seed)
        result = f(rng)
        push!(results, Vector{Float64}(result))
        push!(hashes, bytes2hex(sha256(reinterpret(UInt8, Vector{Float64}(result)))))
    end

    deviations = Float64[]
    warnings = String[]

    for i in 2:config.n_runs
        dev = maximum(abs.(results[1] .- results[i]))
        push!(deviations, dev)

        if dev > config.fp_tolerance
            push!(warnings, "Run $i deviation: $dev > tolerance $(config.fp_tolerance)")
        end
    end

    max_dev = isempty(deviations) ? 0.0 : maximum(deviations)
    passed = max_dev <= config.fp_tolerance

    if !all(h -> h == hashes[1], hashes)
        passed = false
        push!(warnings, "Hash mismatch across runs")
    end

    DeterminismResult(passed, max_dev, deviations, hashes, warnings)
end

# ============================================================================
# Solver Parameter Logging
# ============================================================================

"""
    SolverLog

Log of solver parameters for reproducibility audit.
"""
struct SolverLog
    timestamp::Float64
    iteration::Int
    damping::Float64
    cost::Float64
    cost_reduction::Float64
    max_update::Float64
    converged::Bool
end

"""
    SolverAudit

Collects solver logs for reproducibility verification.
"""
mutable struct SolverAudit
    logs::Vector{SolverLog}
end

SolverAudit() = SolverAudit(SolverLog[])

"""
    log_solver_iteration!(audit, iter, damping, cost, cost_reduction, max_update, converged)

Log a solver iteration for audit.
"""
function log_solver_iteration!(audit::SolverAudit, iter::Int, damping::Float64,
                                cost::Float64, cost_reduction::Float64,
                                max_update::Float64, converged::Bool)
    push!(audit.logs, SolverLog(time(), iter, damping, cost, cost_reduction, max_update, converged))
end

"""
    compare_solver_audits(audit1, audit2; tolerance=1e-10)

Compare two solver audits for reproducibility.
"""
function compare_solver_audits(audit1::SolverAudit, audit2::SolverAudit;
                                tolerance::Float64 = 1e-10)
    if length(audit1.logs) != length(audit2.logs)
        return (false, "Iteration count mismatch")
    end

    for i in 1:length(audit1.logs)
        l1, l2 = audit1.logs[i], audit2.logs[i]

        if abs(l1.damping - l2.damping) > tolerance
            return (false, "Damping mismatch at iteration $i")
        end
        if abs(l1.cost - l2.cost) > tolerance
            return (false, "Cost mismatch at iteration $i")
        end
        if l1.converged != l2.converged
            return (false, "Convergence mismatch at iteration $i")
        end
    end

    return (true, "Solver audits match")
end

# ============================================================================
# Convenience Functions
# ============================================================================

"""
    with_seed(f::Function, seed::Int)

Execute function with specific RNG seed, then restore.
"""
function with_seed(f::Function, seed::Int)
    old_rng = GLOBAL_RNG[]
    set_global_seed!(seed)
    try
        return f()
    finally
        GLOBAL_RNG[] = old_rng
    end
end

"""
    seeded_randn(dims...; rng=get_global_rng())

Generate random normal values with explicit RNG.
"""
seeded_randn(dims...; rng = get_global_rng()) = randn(rng, dims...)

"""
    seeded_rand(dims...; rng=get_global_rng())

Generate random uniform values with explicit RNG.
"""
seeded_rand(dims...; rng = get_global_rng()) = rand(rng, dims...)

# ============================================================================
# Floating Point Comparison
# ============================================================================

"""
    fp_equal(a, b; tolerance=1e-10)

Compare floating point values within tolerance.
"""
function fp_equal(a::Float64, b::Float64; tolerance::Float64 = 1e-10)
    abs(a - b) <= tolerance
end

function fp_equal(a::AbstractArray, b::AbstractArray; tolerance::Float64 = 1e-10)
    size(a) == size(b) && all(abs.(a .- b) .<= tolerance)
end

"""
    fp_hash(x::Float64; precision=10)

Round to precision digits before hashing for FP tolerance.
"""
fp_hash(x::Float64; precision::Int = 10) = round(x, digits = precision)
