module QualificationExecutor

using StaticArrays

export run_qualification, QualificationConfig, QualificationResult

struct QualificationConfig
    scenarios::Vector{Symbol}
    n_seeds::Int
    output_dir::String
end

function QualificationConfig(;
    scenarios::Vector{Symbol} = [:hallway_patrol, :elevator_ride, :spiral_ramp, :lobby_crossing],
    n_seeds::Int = 10,
    output_dir::String = "results/qualification"
)
    QualificationConfig(scenarios, n_seeds, output_dir)
end

struct QualificationResult
    scenario::Symbol
    seed::Int
    horizontal_rmse::Float64
    vertical_rmse::Float64
    floor_detection_accuracy::Float64
    source_detection_latency::Float64
    passed::Bool
end

function run_qualification(config::QualificationConfig)
    results = QualificationResult[]
    println("Qualification executor ready. $(length(config.scenarios)) scenarios × $(config.n_seeds) seeds.")
    return results
end

end # module
