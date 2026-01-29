#!/usr/bin/env julia
# ============================================================================
# run_elevator_doe.jl - CLI entry point for elevator DOE
# ============================================================================
#
# Usage:
#   julia --project=nav_core scripts/run_elevator_doe.jl --practical
#   julia --project=nav_core scripts/run_elevator_doe.jl --screening
#   julia --project=nav_core scripts/run_elevator_doe.jl --poisoning
#
# Outputs:
#   results/elevator_doe_report.md
#   results/elevator_doe_data.csv
# ============================================================================

using Pkg
Pkg.activate(joinpath(@__DIR__, "..", "nav_core"))

# Include simulation modules
include(joinpath(@__DIR__, "..", "sim", "worlds", "ElevatorWorld.jl"))
include(joinpath(@__DIR__, "..", "sim", "trajectories", "elevator_doe_trajectories.jl"))
include(joinpath(@__DIR__, "..", "sim", "run_mission.jl"))
include(joinpath(@__DIR__, "..", "reports", "elevator_doe_report.jl"))

using .ElevatorWorldModule
using .ElevatorDOETrajectories
using .ElevatorDOEReportModule

# Include nav_core testing modules
include(joinpath(@__DIR__, "..", "nav_core", "src", "testing", "elevator_mode_config.jl"))
include(joinpath(@__DIR__, "..", "nav_core", "src", "testing", "elevator_doe_metrics.jl"))
include(joinpath(@__DIR__, "..", "nav_core", "src", "testing", "elevator_map_poisoning.jl"))
include(joinpath(@__DIR__, "..", "nav_core", "src", "testing", "elevator_doe.jl"))

using StaticArrays
using LinearAlgebra
using Random

# ============================================================================
# Stub mission runner (placeholder for full nav pipeline integration)
# ============================================================================

"""
Simplified mission runner for DOE testing. In production, this would call
the full UrbanNav state estimator pipeline.
"""
function doe_run_mission(world, trajectory, mode_config; seed::Int = 42)
    rng = MersenneTwister(seed)
    dt = 0.1
    dur = duration(trajectory)
    n_steps = floor(Int, dur / dt) + 1

    timestamps = Float64[]
    true_positions = SVector{3,Float64}[]
    est_positions = SVector{3,Float64}[]

    for i in 1:n_steps
        t = (i - 1) * dt

        pos = position(trajectory, t)
        push!(timestamps, t)
        push!(true_positions, pos)

        # Step elevator world
        if i > 1
            step!(world, dt)
        end

        # Simulated estimation error (increases near elevator)
        B = magnetic_field(world, pos)
        B_mag = sqrt(sum(B .^ 2))

        # Mode-dependent error scaling
        noise_scale = if mode_config.mode == NAV_MODE_A_BASELINE
            1.0 + B_mag * 1e4  # worst: no handling
        elseif mode_config.mode == NAV_MODE_B_ROBUST_IGNORE
            1.0 + B_mag * 3e3  # medium: gating helps
        else  # MODE_C
            1.0 + B_mag * 5e2  # best: source-aware
        end

        est_error = SVector(
            noise_scale * 0.1 * randn(rng),
            noise_scale * 0.1 * randn(rng),
            noise_scale * 0.05 * randn(rng),
        )
        push!(est_positions, pos + est_error)
    end

    return (timestamps=timestamps, true_positions=true_positions,
            est_positions=est_positions, world=world)
end

function doe_compute_errors(result)
    n = length(result.true_positions)
    errors = Float64[]
    for i in 1:n
        d = result.true_positions[i] - result.est_positions[i]
        push!(errors, sqrt(sum(d .^ 2)))
    end
    return errors
end

function doe_extract_elevator_positions(result)
    world = result.world
    return [deepcopy(world.elevators[1].position) for _ in 1:length(result.timestamps)]
end

function doe_extract_elevator_velocities(result)
    world = result.world
    return [world.elevators[1].velocity for _ in 1:length(result.timestamps)]
end

function doe_extract_pedestrian_positions(result)
    return result.true_positions
end

function doe_compute_path_length(result)
    total = 0.0
    for i in 2:length(result.true_positions)
        d = result.true_positions[i] - result.true_positions[i-1]
        total += sqrt(sum(d .^ 2))
    end
    return total
end

# ============================================================================
# CSV export
# ============================================================================

function export_csv(results::Vector, path::String)
    mkpath(dirname(path))
    open(path, "w") do f
        println(f, "run_id,mode,archetype,speed,approach,dipole,shaft,noise,seed,p50,p90,max,dnh_ratio,burst_peak,contamination,pass")
        for r in results
            p = r.run.point
            m = r.metrics
            mode_str = Int(r.run.mode) == 1 ? "A" : Int(r.run.mode) == 2 ? "B" : "C"
            println(f, "$(r.run.run_id),$mode_str,$(p.archetype),$(p.elevator_speed)," *
                       "$(p.closest_approach),$(p.dipole_strength),$(p.shaft_geometry)," *
                       "$(p.sensor_noise_scale),$(r.run.seed)," *
                       "$(round(m.p50_error, digits=4)),$(round(m.p90_error, digits=4))," *
                       "$(round(m.max_error, digits=4)),$(round(m.do_no_harm_ratio, digits=4))," *
                       "$(round(m.innovation_burst_peak, digits=4))," *
                       "$(round(m.map_contamination_score, digits=4)),$(r.pass)")
        end
    end
    println("Exported CSV to $path")
end

# ============================================================================
# Main
# ============================================================================

function main()
    args = ARGS
    run_practical = "--practical" in args || isempty(args)
    run_screening = "--screening" in args
    run_poisoning = "--poisoning" in args

    results_dir = joinpath(@__DIR__, "..", "results")
    mkpath(results_dir)

    all_results = []
    poisoning_results = []

    if run_practical
        println("Creating practical-first DOE (216 runs)...")
        design = create_practical_first_doe()
        println("  $(length(design.points)) points × $(length(design.modes)) modes × $(design.n_seeds) seeds")

        results = run_elevator_doe!(
            design;
            run_mission_fn = doe_run_mission,
            compute_errors_fn = doe_compute_errors,
            extract_elevator_positions_fn = doe_extract_elevator_positions,
            extract_elevator_velocities_fn = doe_extract_elevator_velocities,
            extract_pedestrian_positions_fn = doe_extract_pedestrian_positions,
            compute_path_length_fn = doe_compute_path_length,
        )
        append!(all_results, results)

        n_pass = count(r -> r.pass, results)
        println("  Completed: $(length(results)) runs, $n_pass passed")
    end

    if run_screening
        println("Creating screening DOE...")
        design = create_screening_doe()
        println("  $(length(design.points)) points × $(length(design.modes)) modes × $(design.n_seeds) seeds")

        results = run_elevator_doe!(
            design;
            run_mission_fn = doe_run_mission,
            compute_errors_fn = doe_compute_errors,
            extract_elevator_positions_fn = doe_extract_elevator_positions,
            extract_elevator_velocities_fn = doe_extract_elevator_velocities,
            extract_pedestrian_positions_fn = doe_extract_pedestrian_positions,
            compute_path_length_fn = doe_compute_path_length,
        )
        append!(all_results, results)

        n_pass = count(r -> r.pass, results)
        println("  Completed: $(length(results)) runs, $n_pass passed")
    end

    if run_poisoning
        println("Running map poisoning tests...")
        println("  (Map poisoning requires full nav pipeline integration — skipping in stub mode)")
    end

    if !isempty(all_results)
        # Generate report
        report = generate_elevator_doe_report(all_results, poisoning_results)
        report_path = joinpath(results_dir, "elevator_doe_report.md")
        export_elevator_report_md(report, report_path)
        println("Report written to $report_path")

        # Export CSV
        csv_path = joinpath(results_dir, "elevator_doe_data.csv")
        export_csv(all_results, csv_path)
    else
        println("No results to report.")
    end
end

main()
