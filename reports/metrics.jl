module UrbanNavMetrics

using StaticArrays
using LinearAlgebra
using Statistics

export compute_horizontal_rmse, compute_vertical_rmse, compute_floor_accuracy
export compute_nees, compute_detection_latency

function compute_horizontal_rmse(true_positions::Vector{SVector{3,Float64}},
                                  est_positions::Vector{SVector{3,Float64}})
    n = length(true_positions)
    @assert n == length(est_positions)
    sum_sq = 0.0
    for i in 1:n
        dx = true_positions[i][1] - est_positions[i][1]
        dy = true_positions[i][2] - est_positions[i][2]
        sum_sq += dx^2 + dy^2
    end
    return sqrt(sum_sq / n)
end

function compute_vertical_rmse(true_positions::Vector{SVector{3,Float64}},
                                est_positions::Vector{SVector{3,Float64}})
    n = length(true_positions)
    sum_sq = sum((true_positions[i][3] - est_positions[i][3])^2 for i in 1:n)
    return sqrt(sum_sq / n)
end

function compute_floor_accuracy(true_floors::Vector{Int}, detected_floors::Vector{Int})
    n = length(true_floors)
    correct = sum(true_floors[i] == detected_floors[i] for i in 1:n)
    return correct / n
end

function compute_nees(errors::Vector{SVector{3,Float64}},
                      covariances::Vector{SMatrix{3,3,Float64,9}})
    n = length(errors)
    nees_values = Float64[]
    for i in 1:n
        e = errors[i]
        P = covariances[i]
        nees = dot(e, P \ e)
        push!(nees_values, nees)
    end
    return nees_values
end

function compute_detection_latency(snr_history::Vector{Float64},
                                    threshold::Float64,
                                    detection_time::Float64,
                                    dt::Float64)
    first_above = findfirst(x -> x > threshold, snr_history)
    if first_above === nothing
        return Inf
    end
    first_above_time = (first_above - 1) * dt
    return detection_time - first_above_time
end

end # module
