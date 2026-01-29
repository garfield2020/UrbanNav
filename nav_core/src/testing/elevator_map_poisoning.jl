# ============================================================================
# elevator_map_poisoning.jl - Section 7 map poisoning test
# ============================================================================
#
# Tests whether a learned map from a mission with an active elevator
# degrades navigation performance on a subsequent mission without the elevator.
#
# Protocol:
# 1. Mission 1: Elevator active → extract learned map coefficients
# 2. Mission 2 baseline: Elevator off, fresh (empty) map
# 3. Mission 2 learned: Elevator off, use Mission 1's learned map
# 4. Compare: P90(M2 learned) / P90(M2 baseline) ≤ 1.05
# ============================================================================

export MapPoisoningResult, run_map_poisoning_test

"""
    MapPoisoningResult

Results of the Section 7 map poisoning test.

# Fields
- `m1_p90::Float64`: P90 position error during Mission 1 (elevator active).
- `m2_baseline_p90::Float64`: P90 position error during M2 with fresh map.
- `m2_learned_p90::Float64`: P90 position error during M2 with M1's map.
- `contamination_ratio::Float64`: P90(M2 learned) / P90(M2 baseline).
- `m1_errors::Vector{Float64}`: Position errors from Mission 1.
- `m2_baseline_errors::Vector{Float64}`: Position errors from M2 baseline.
- `m2_learned_errors::Vector{Float64}`: Position errors from M2 learned.
- `passes::Bool`: True if contamination_ratio ≤ 1.05.
"""
struct MapPoisoningResult
    m1_p90::Float64
    m2_baseline_p90::Float64
    m2_learned_p90::Float64
    contamination_ratio::Float64
    m1_errors::Vector{Float64}
    m2_baseline_errors::Vector{Float64}
    m2_learned_errors::Vector{Float64}
    passes::Bool
end

"""
    _quantile(vals, q) -> Float64

Compute the q-th quantile of a vector.
"""
function _quantile(vals::Vector{Float64}, q::Float64)
    isempty(vals) && return 0.0
    sorted = sort(vals)
    n = length(sorted)
    idx = q * (n - 1) + 1.0
    lo = clamp(floor(Int, idx), 1, n)
    hi = clamp(ceil(Int, idx), 1, n)
    frac = idx - lo
    return sorted[lo] * (1.0 - frac) + sorted[hi] * frac
end

"""
    run_map_poisoning_test(;
        run_mission_fn,
        build_world_fn,
        build_frozen_world_fn,
        build_trajectory_fn,
        extract_map_fn,
        inject_map_fn,
        compute_errors_fn,
        mode_config,
        seed
    ) -> MapPoisoningResult

Execute the full map poisoning test protocol.

# Arguments
- `run_mission_fn`: Function(world, trajectory, mode_config, map; seed) → (result, learned_map).
- `build_world_fn`: Function() → world with active elevator.
- `build_frozen_world_fn`: Function() → world with frozen elevator.
- `build_trajectory_fn`: Function() → trajectory for both missions.
- `extract_map_fn`: Function(result) → map coefficients.
- `inject_map_fn`: Function(map) → prepared map for injection.
- `compute_errors_fn`: Function(result) → Vector{Float64} position errors.
- `mode_config`: ElevatorModeConfig to use.
- `seed::Int`: Random seed.

The function orchestrates the three-mission protocol:
1. M1 with elevator active → extract map
2. M2 baseline with elevator frozen, fresh map
3. M2 learned with elevator frozen, M1's map
4. Compare P90 ratios

# Returns
`MapPoisoningResult` with contamination ratio and pass/fail.
"""
function run_map_poisoning_test(;
    run_mission_fn::Function,
    build_world_fn::Function,
    build_frozen_world_fn::Function,
    build_trajectory_fn::Function,
    extract_map_fn::Function,
    inject_map_fn::Function,
    compute_errors_fn::Function,
    mode_config,
    seed::Int = 42,
)
    traj = build_trajectory_fn()

    # --- Mission 1: Elevator active, learn map ---
    world_m1 = build_world_fn()
    result_m1, learned_map = run_mission_fn(world_m1, traj, mode_config, nothing; seed=seed)
    m1_errors = compute_errors_fn(result_m1)
    m1_p90 = _quantile(m1_errors, 0.9)

    # --- Mission 2 baseline: Elevator frozen, fresh map ---
    world_m2_base = build_frozen_world_fn()
    result_m2_base, _ = run_mission_fn(world_m2_base, traj, mode_config, nothing; seed=seed+1)
    m2_base_errors = compute_errors_fn(result_m2_base)
    m2_base_p90 = _quantile(m2_base_errors, 0.9)

    # --- Mission 2 learned: Elevator frozen, M1's map ---
    world_m2_learn = build_frozen_world_fn()
    injected_map = inject_map_fn(learned_map)
    result_m2_learn, _ = run_mission_fn(world_m2_learn, traj, mode_config, injected_map; seed=seed+1)
    m2_learn_errors = compute_errors_fn(result_m2_learn)
    m2_learn_p90 = _quantile(m2_learn_errors, 0.9)

    # --- Compare ---
    ratio = m2_base_p90 > 1e-10 ? m2_learn_p90 / m2_base_p90 : 1.0
    passes = ratio <= 1.05

    return MapPoisoningResult(
        m1_p90, m2_base_p90, m2_learn_p90, ratio,
        m1_errors, m2_base_errors, m2_learn_errors,
        passes,
    )
end
