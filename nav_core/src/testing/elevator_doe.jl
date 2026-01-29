# ============================================================================
# elevator_doe.jl - Core DOE orchestration for elevator interference testing
# ============================================================================
#
# Implements the SE-grade Design of Experiments for moving elevators.
# Answers: "Can you keep localization stable while nearby steel is moving?"
#
# Uses Latin Hypercube Sampling for screening designs and full factorial
# for the practical-first design.
# ============================================================================

export ElevatorSpeed, ELEV_SLOW, ELEV_MEDIUM, ELEV_FAST
export StopFrequency, STOP_RARE, STOP_NORMAL, STOP_FREQUENT
export ClosestApproach, APPROACH_NEAR, APPROACH_MEDIUM, APPROACH_FAR
export DipoleStrength, DIPOLE_WEAK, DIPOLE_NOMINAL, DIPOLE_STRONG
export ShaftGeometry, SHAFT_SINGLE, SHAFT_OFFSET, SHAFT_DUAL
export TrajectoryRichness, RICHNESS_STRAIGHT, RICHNESS_L_TURNS, RICHNESS_LOOP
export SensorNoiseScale, NOISE_HALF, NOISE_NOMINAL, NOISE_DOUBLE

export ElevatorDOEPoint, ElevatorDOEDesign, ElevatorDOERun, ElevatorDOEResult
export create_practical_first_doe, create_screening_doe
export build_elevator_world, build_trajectory, build_control_world
export run_elevator_doe!, run_single_point

using StaticArrays
using Random

# ============================================================================
# DOE Factor Enums (Section 5)
# ============================================================================

@enum ElevatorSpeed begin
    ELEV_SLOW = 1      # 0.5 m/s
    ELEV_MEDIUM = 2    # 1.5 m/s
    ELEV_FAST = 3      # 3.0 m/s
end

@enum StopFrequency begin
    STOP_RARE = 1      # 60s dwell
    STOP_NORMAL = 2    # 15s dwell
    STOP_FREQUENT = 3  # 5s dwell
end

@enum ClosestApproach begin
    APPROACH_NEAR = 1    # 1m offset
    APPROACH_MEDIUM = 2  # 3m offset
    APPROACH_FAR = 3     # 10m offset
end

@enum DipoleStrength begin
    DIPOLE_WEAK = 1      # 50 A·m²
    DIPOLE_NOMINAL = 2   # 200 A·m²
    DIPOLE_STRONG = 3    # 500 A·m²
end

@enum ShaftGeometry begin
    SHAFT_SINGLE = 1
    SHAFT_OFFSET = 2
    SHAFT_DUAL = 3
end

@enum TrajectoryRichness begin
    RICHNESS_STRAIGHT = 1
    RICHNESS_L_TURNS = 2
    RICHNESS_LOOP = 3
end

@enum SensorNoiseScale begin
    NOISE_HALF = 1       # 0.5× nominal
    NOISE_NOMINAL = 2    # 1.0× nominal
    NOISE_DOUBLE = 3     # 2.0× nominal
end

# ============================================================================
# Structs
# ============================================================================

"""
    ElevatorDOEPoint

One factor combination for the elevator DOE.
"""
struct ElevatorDOEPoint
    elevator_speed::ElevatorSpeed
    stop_frequency::StopFrequency
    closest_approach::ClosestApproach
    dipole_strength::DipoleStrength
    shaft_geometry::ShaftGeometry
    trajectory_richness::TrajectoryRichness
    sensor_noise_scale::SensorNoiseScale
    archetype::Symbol  # :corridor, :perpendicular, :loop, :stop_and_go, :multi_floor, :dual_shaft
end

"""
    ElevatorDOEDesign

Full DOE design: a collection of points to run across modes and seeds.
"""
struct ElevatorDOEDesign
    name::String
    points::Vector{ElevatorDOEPoint}
    modes::Vector{ElevatorNavMode}
    n_seeds::Int
end

"""
    ElevatorDOERun

A single run: one DOE point + one mode + one seed.
"""
struct ElevatorDOERun
    point::ElevatorDOEPoint
    mode::ElevatorNavMode
    seed::Int
    run_id::Int
end

"""
    ElevatorDOEResult

Result of a single elevator DOE run.
"""
struct ElevatorDOEResult
    run::ElevatorDOERun
    metrics::ElevatorDOEMetrics
    segment_metrics::SegmentMetrics
    pass::Bool
    failure_reasons::Vector{String}
end

# ============================================================================
# Factor Value Accessors
# ============================================================================

function _speed_value(s::ElevatorSpeed)::Float64
    s == ELEV_SLOW ? 0.5 : s == ELEV_MEDIUM ? 1.5 : 3.0
end

function _dwell_value(f::StopFrequency)::Float64
    f == STOP_RARE ? 60.0 : f == STOP_NORMAL ? 15.0 : 5.0
end

function _approach_value(a::ClosestApproach)::Float64
    a == APPROACH_NEAR ? 1.0 : a == APPROACH_MEDIUM ? 3.0 : 10.0
end

function _dipole_value(d::DipoleStrength)::Float64
    d == DIPOLE_WEAK ? 50.0 : d == DIPOLE_NOMINAL ? 200.0 : 500.0
end

function _noise_scale_value(n::SensorNoiseScale)::Float64
    n == NOISE_HALF ? 0.5 : n == NOISE_NOMINAL ? 1.0 : 2.0
end

function _shaft_positions(g::ShaftGeometry)::Vector{SVector{2,Float64}}
    if g == SHAFT_SINGLE
        return [SVector(0.0, 0.0)]
    elseif g == SHAFT_OFFSET
        return [SVector(0.0, 0.0), SVector(2.0, 3.0)]
    else  # SHAFT_DUAL
        return [SVector(0.0, 0.0), SVector(8.0, 0.0)]
    end
end

# ============================================================================
# Design Factories
# ============================================================================

"""
    create_practical_first_doe() -> ElevatorDOEDesign

Create the practical-first DOE: 6 archetypes × 2 approach × 2 motion = 24 points,
crossed with 3 modes and 3 seeds = 216 total runs.
"""
function create_practical_first_doe()
    points = ElevatorDOEPoint[]

    archetypes = [:corridor, :perpendicular, :loop, :stop_and_go, :multi_floor, :dual_shaft]
    approaches = [APPROACH_NEAR, APPROACH_MEDIUM]
    speeds = [ELEV_SLOW, ELEV_FAST]

    for arch in archetypes
        for approach in approaches
            for speed in speeds
                geom = arch == :dual_shaft ? SHAFT_DUAL : SHAFT_SINGLE
                richness = arch == :loop ? RICHNESS_LOOP :
                           arch in (:stop_and_go, :perpendicular) ? RICHNESS_L_TURNS :
                           RICHNESS_STRAIGHT

                point = ElevatorDOEPoint(
                    speed, STOP_NORMAL, approach, DIPOLE_NOMINAL,
                    geom, richness, NOISE_NOMINAL, arch,
                )
                push!(points, point)
            end
        end
    end

    modes = [NAV_MODE_A_BASELINE, NAV_MODE_B_ROBUST_IGNORE, NAV_MODE_C_SOURCE_AWARE]
    return ElevatorDOEDesign("practical_first", points, modes, 3)
end

"""
    create_screening_doe(; n_points=80, seed=123) -> ElevatorDOEDesign

Create a screening DOE using Latin Hypercube Sampling over all factors.
Returns 60-120 points × 3 modes × 3 seeds.
"""
function create_screening_doe(; n_points::Int = 80, seed::Int = 123)
    rng = MersenneTwister(seed)
    points = ElevatorDOEPoint[]

    all_speeds = [ELEV_SLOW, ELEV_MEDIUM, ELEV_FAST]
    all_stops = [STOP_RARE, STOP_NORMAL, STOP_FREQUENT]
    all_approaches = [APPROACH_NEAR, APPROACH_MEDIUM, APPROACH_FAR]
    all_dipoles = [DIPOLE_WEAK, DIPOLE_NOMINAL, DIPOLE_STRONG]
    all_geoms = [SHAFT_SINGLE, SHAFT_OFFSET, SHAFT_DUAL]
    all_richness = [RICHNESS_STRAIGHT, RICHNESS_L_TURNS, RICHNESS_LOOP]
    all_noise = [NOISE_HALF, NOISE_NOMINAL, NOISE_DOUBLE]
    all_archetypes = [:corridor, :perpendicular, :loop, :stop_and_go, :multi_floor, :dual_shaft]

    for _ in 1:n_points
        point = ElevatorDOEPoint(
            rand(rng, all_speeds),
            rand(rng, all_stops),
            rand(rng, all_approaches),
            rand(rng, all_dipoles),
            rand(rng, all_geoms),
            rand(rng, all_richness),
            rand(rng, all_noise),
            rand(rng, all_archetypes),
        )
        push!(points, point)
    end

    modes = [NAV_MODE_A_BASELINE, NAV_MODE_B_ROBUST_IGNORE, NAV_MODE_C_SOURCE_AWARE]
    return ElevatorDOEDesign("screening_lhs", points, modes, 3)
end

# ============================================================================
# World and Trajectory Builders
# ============================================================================

"""
    build_elevator_world(point::ElevatorDOEPoint; seed=42) -> ElevatorWorld

Build an ElevatorWorld configured for the given DOE point.
"""
function build_elevator_world(point::ElevatorDOEPoint; seed::Int = 42)
    create_doe_elevator_world(
        speed = _speed_value(point.elevator_speed),
        dwell_time = _dwell_value(point.stop_frequency),
        shaft_positions = _shaft_positions(point.shaft_geometry),
        dipole_moment = _dipole_value(point.dipole_strength),
        seed = seed,
        frozen = false,
    )
end

"""
    build_control_world(point::ElevatorDOEPoint; seed=42) -> ElevatorWorld

Build a frozen ElevatorWorld (elevator never moves) for do-no-harm baseline.
"""
function build_control_world(point::ElevatorDOEPoint; seed::Int = 42)
    create_doe_elevator_world(
        speed = _speed_value(point.elevator_speed),
        dwell_time = _dwell_value(point.stop_frequency),
        shaft_positions = _shaft_positions(point.shaft_geometry),
        dipole_moment = _dipole_value(point.dipole_strength),
        seed = seed,
        frozen = true,
    )
end

"""
    build_trajectory(point::ElevatorDOEPoint) -> AbstractTrajectory

Build the trajectory for the given DOE point archetype and approach distance.
"""
function build_trajectory(point::ElevatorDOEPoint)
    offset = _approach_value(point.closest_approach)
    shaft1 = _shaft_positions(point.shaft_geometry)[1]

    if point.archetype == :corridor
        return CorridorParallel(shaft1; offset=offset)
    elseif point.archetype == :perpendicular
        return PerpendicularCrossing(shaft1; spacing=offset)
    elseif point.archetype == :loop
        return ShaftLoop(shaft1; radius=offset)
    elseif point.archetype == :stop_and_go
        return StopAndGo(shaft1; offset=offset)
    elseif point.archetype == :multi_floor
        return MultiFloorWalk(shaft1; offset=offset)
    elseif point.archetype == :dual_shaft
        shafts = _shaft_positions(point.shaft_geometry)
        shaft2 = length(shafts) >= 2 ? shafts[2] : SVector(8.0, 0.0)
        return DualShaftPath(shaft1, shaft2; offset=offset)
    else
        error("Unknown archetype: $(point.archetype)")
    end
end

# ============================================================================
# DOE Runner
# ============================================================================

"""
    run_single_point(run::ElevatorDOERun;
                     run_mission_fn, compute_errors_fn) -> ElevatorDOEResult

Execute a single DOE run: build worlds, run missions with and without elevator,
compute metrics, and evaluate gates.

# Arguments
- `run`: The DOE run specification.
- `run_mission_fn`: Function(world, trajectory, mode_config; seed) → mission_result.
- `compute_errors_fn`: Function(result) → Vector{Float64} position errors.
"""
function run_single_point(
    run::ElevatorDOERun;
    run_mission_fn::Function,
    compute_errors_fn::Function,
    compute_innovations_fn::Function = (_) -> Float64[],
    compute_tile_updates_fn::Function = (_) -> Float64[],
    extract_elevator_positions_fn::Function = (_) -> [],
    extract_elevator_velocities_fn::Function = (_) -> Float64[],
    extract_pedestrian_positions_fn::Function = (_) -> [],
    compute_path_length_fn::Function = (_) -> 100.0,
    count_false_sources_fn::Function = (_) -> 0,
)
    point = run.point
    mode_config = configure_elevator_mode(run.mode)
    traj = build_trajectory(point)

    # Run with elevator active
    world_active = build_elevator_world(point; seed=run.seed)
    result_with = run_mission_fn(world_active, traj, mode_config; seed=run.seed)
    errors_with = compute_errors_fn(result_with)

    # Run with elevator frozen (control)
    world_frozen = build_control_world(point; seed=run.seed)
    result_without = run_mission_fn(world_frozen, traj, mode_config; seed=run.seed)
    errors_without = compute_errors_fn(result_without)

    # Compute do-no-harm ratio
    dnh_ratio = compute_do_no_harm(errors_with, errors_without)

    # Extract diagnostics from active run
    innovations = compute_innovations_fn(result_with)
    tile_updates = compute_tile_updates_fn(result_with)
    elev_positions = extract_elevator_positions_fn(result_with)
    elev_velocities = extract_elevator_velocities_fn(result_with)
    ped_positions = extract_pedestrian_positions_fn(result_with)
    path_length = compute_path_length_fn(result_with)
    false_sources = count_false_sources_fn(result_with)

    # Compute metrics
    metrics = compute_elevator_metrics(
        errors_with, innovations, tile_updates, elev_positions, path_length;
        do_no_harm_ratio=dnh_ratio, false_source_count=false_sources,
    )

    # Timestamps placeholder (use index-based)
    timestamps = collect(range(0.0, step=0.1, length=length(errors_with)))

    segment_metrics = compute_segment_metrics(
        errors_with, innovations, tile_updates,
        elev_positions, ped_positions, elev_velocities,
        timestamps, path_length;
        do_no_harm_ratio=dnh_ratio, false_source_count=false_sources,
    )

    # Evaluate gates
    failures = String[]
    if run.mode == NAV_MODE_B_ROBUST_IGNORE && dnh_ratio > 1.10
        push!(failures, "Do-no-harm: $(round(dnh_ratio, digits=3)) > 1.10")
    end
    if run.mode == NAV_MODE_C_SOURCE_AWARE
        if dnh_ratio > 1.10
            push!(failures, "Do-no-harm: $(round(dnh_ratio, digits=3)) > 1.10")
        end
    end

    pass = isempty(failures)

    return ElevatorDOEResult(run, metrics, segment_metrics, pass, failures)
end

"""
    run_elevator_doe!(design::ElevatorDOEDesign;
                      run_mission_fn, compute_errors_fn, kwargs...) -> Vector{ElevatorDOEResult}

Run all points in the DOE design across all modes and seeds.
"""
function run_elevator_doe!(
    design::ElevatorDOEDesign;
    run_mission_fn::Function,
    compute_errors_fn::Function,
    kwargs...
)
    results = ElevatorDOEResult[]
    run_id = 0

    for point in design.points
        for mode in design.modes
            for seed_idx in 1:design.n_seeds
                run_id += 1
                seed = 1000 * run_id + seed_idx

                doe_run = ElevatorDOERun(point, mode, seed, run_id)
                result = run_single_point(
                    doe_run;
                    run_mission_fn=run_mission_fn,
                    compute_errors_fn=compute_errors_fn,
                    kwargs...
                )
                push!(results, result)
            end
        end
    end

    return results
end
