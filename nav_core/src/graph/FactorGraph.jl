# ============================================================================
# FactorGraph.jl - Factor graph data structure
# ============================================================================
#
# The factor graph is the core data structure for state estimation.
# States are variable nodes, factors encode constraints between them.
# ============================================================================

export FactorGraph, Variable, add_variable!, add_factor!, get_variable
export linearize!, solve!, marginal_covariance

"""
    Variable{T}

A variable node in the factor graph.

# Fields
- `id::Int` - Unique identifier
- `dim::Int` - State dimension
- `value::Vector{T}` - Current estimate
- `timestamp::Float64` - Associated time
"""
mutable struct Variable{T<:Real}
    id::Int
    dim::Int
    value::Vector{T}
    timestamp::Float64
end

"""
    FactorGraph{T}

Factor graph for nonlinear least squares optimization.

# Fields
- `variables::Dict{Int, Variable{T}}` - Variable nodes
- `factors::Vector{AbstractFactor}` - Factor constraints
- `variable_order::Vector{Int}` - Variable ordering for linearization
- `next_var_id::Int` - Counter for variable IDs
- `next_factor_id::Int` - Counter for factor IDs
"""
mutable struct FactorGraph{T<:Real}
    variables::Dict{Int, Variable{T}}
    factors::Vector{AbstractFactor}
    variable_order::Vector{Int}
    next_var_id::Int
    next_factor_id::Int
end

"""
    FactorGraph{T}() where T

Create an empty factor graph.
"""
function FactorGraph{T}() where T
    return FactorGraph{T}(
        Dict{Int, Variable{T}}(),
        AbstractFactor[],
        Int[],
        1,
        1
    )
end

FactorGraph() = FactorGraph{Float64}()

"""
    add_variable!(graph, dim, initial_value, timestamp) -> Int

Add a variable node to the graph. Returns variable ID.
"""
function add_variable!(graph::FactorGraph{T}, dim::Int,
                       initial_value::AbstractVector{T},
                       timestamp::Float64) where T
    @assert length(initial_value) == dim

    id = graph.next_var_id
    graph.next_var_id += 1

    var = Variable{T}(id, dim, collect(initial_value), timestamp)
    graph.variables[id] = var
    push!(graph.variable_order, id)

    return id
end

"""
    add_factor!(graph, factor) -> Int

Add a factor to the graph. Returns factor ID.
"""
function add_factor!(graph::FactorGraph, factor::AbstractFactor)
    push!(graph.factors, factor)
    return length(graph.factors)
end

"""
    get_variable(graph, id) -> Variable

Retrieve a variable by ID.
"""
function get_variable(graph::FactorGraph, id::Int)
    return graph.variables[id]
end

"""
    num_variables(graph) -> Int

Return number of variables in graph.
"""
num_variables(graph::FactorGraph) = length(graph.variables)

"""
    num_factors(graph) -> Int

Return number of factors in graph.
"""
num_factors(graph::FactorGraph) = length(graph.factors)

"""
    total_state_dim(graph) -> Int

Return total dimension of all variables.
"""
function total_state_dim(graph::FactorGraph)
    return sum(v.dim for v in values(graph.variables))
end

"""
    state_vector(graph) -> Vector

Extract concatenated state vector in variable order.
"""
function state_vector(graph::FactorGraph{T}) where T
    n = total_state_dim(graph)
    x = Vector{T}(undef, n)
    idx = 1
    for var_id in graph.variable_order
        var = graph.variables[var_id]
        x[idx:idx+var.dim-1] = var.value
        idx += var.dim
    end
    return x
end

"""
    update_variables!(graph, delta)

Update all variables with delta vector.
"""
function update_variables!(graph::FactorGraph, delta::AbstractVector)
    idx = 1
    for var_id in graph.variable_order
        var = graph.variables[var_id]
        var.value .+= delta[idx:idx+var.dim-1]
        idx += var.dim
    end
end
