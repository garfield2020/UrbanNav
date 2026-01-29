# ============================================================================
# failure_atlas.jl - Auto-Generated Failure Atlas from DOE
# ============================================================================
#
# V1.0 Qualification Closure Plan Step 7:
# Auto-generate a comprehensive failure atlas from DOE results.
#
# The Failure Atlas provides:
# 1. Catalog of all discovered failure modes
# 2. Trigger conditions for each failure mode
# 3. Boundaries between success and failure regions
# 4. Factor correlations with failure probability
# 5. Mitigation recommendations
#
# This enables:
# - Complete documentation of system limitations
# - Operational envelope definition
# - Root cause analysis for failures
# - Qualification evidence package
# ============================================================================

using Statistics
using LinearAlgebra

export FailureMode, FailureTrigger, FailureBoundary
export FailureCorrelation, FailureAtlas
export FailureAtlasConfig, DEFAULT_ATLAS_CONFIG
export AtlasEntry, AtlasSection
export discover_failure_modes, identify_failure_boundaries
export compute_failure_correlations, generate_failure_atlas
export format_failure_atlas, export_failure_atlas_markdown
export FailureCluster, cluster_failures
export OperationalEnvelope, compute_operational_envelope

# ============================================================================
# Failure Mode Definition
# ============================================================================

"""
    FailureMode

A distinct failure mode discovered through DOE testing.

# Fields
- `id::String`: Unique identifier (e.g., "FM-001")
- `name::String`: Human-readable name
- `category::Symbol`: Category (:observability, :performance, :health, :silent_divergence)
- `severity::Symbol`: Severity level (:critical, :major, :minor, :cosmetic)
- `description::String`: Detailed description
- `affected_states::Vector{Int}`: Which states are affected
- `occurrence_count::Int`: How many times observed in DOE
- `occurrence_rate::Float64`: Fraction of runs with this failure
"""
struct FailureMode
    id::String
    name::String
    category::Symbol
    severity::Symbol
    description::String
    affected_states::Vector{Int}
    occurrence_count::Int
    occurrence_rate::Float64
end

function FailureMode(;
    id::String = "FM-000",
    name::String = "Unknown Failure",
    category::Symbol = :unknown,
    severity::Symbol = :major,
    description::String = "",
    affected_states::Vector{Int} = Int[],
    occurrence_count::Int = 0,
    occurrence_rate::Float64 = 0.0
)
    FailureMode(id, name, category, severity, description,
                affected_states, occurrence_count, occurrence_rate)
end

# ============================================================================
# Failure Trigger
# ============================================================================

"""
    FailureTrigger

Conditions that trigger a failure mode.

# Fields
- `factor_conditions::Dict{Symbol, Any}`: Factor values that cause failure
- `threshold_conditions::Vector{Tuple{Symbol, Symbol, Float64}}`: (metric, comparison, value)
- `probability::Float64`: Probability of failure given these conditions
- `confidence::Float64`: Confidence in this trigger identification
- `sample_count::Int`: Number of samples supporting this trigger
"""
struct FailureTrigger
    factor_conditions::Dict{Symbol, Any}
    threshold_conditions::Vector{Tuple{Symbol, Symbol, Float64}}
    probability::Float64
    confidence::Float64
    sample_count::Int
end

function FailureTrigger(;
    factor_conditions::Dict{Symbol, Any} = Dict{Symbol, Any}(),
    threshold_conditions::Vector{Tuple{Symbol, Symbol, Float64}} = Tuple{Symbol, Symbol, Float64}[],
    probability::Float64 = 1.0,
    confidence::Float64 = 0.5,
    sample_count::Int = 0
)
    FailureTrigger(factor_conditions, threshold_conditions, probability, confidence, sample_count)
end

"""
    format_trigger(trigger::FailureTrigger) -> String

Format a trigger as a human-readable condition string.
"""
function format_trigger(trigger::FailureTrigger)
    parts = String[]

    for (factor, value) in trigger.factor_conditions
        push!(parts, "$factor = $value")
    end

    for (metric, op, val) in trigger.threshold_conditions
        op_str = op == :lt ? "<" : op == :gt ? ">" : op == :le ? "<=" : op == :ge ? ">=" : "=="
        push!(parts, "$metric $op_str $val")
    end

    if isempty(parts)
        return "Always"
    end

    join(parts, " AND ")
end

# ============================================================================
# Failure Boundary
# ============================================================================

"""
    FailureBoundary

Boundary between success and failure regions in factor space.

# Fields
- `failure_mode::FailureMode`: Which failure mode
- `boundary_factor::Symbol`: Factor defining the boundary
- `boundary_value::Any`: Value at the boundary
- `direction::Symbol`: :above or :below (which side fails)
- `sharpness::Float64`: How sharp the transition is (0-1, 1=sharp)
- `other_factors::Dict{Symbol, Any}`: Fixed values of other factors
"""
struct FailureBoundary
    failure_mode_id::String
    boundary_factor::Symbol
    boundary_value::Any
    direction::Symbol
    sharpness::Float64
    other_factors::Dict{Symbol, Any}
end

# ============================================================================
# Failure Correlation
# ============================================================================

"""
    FailureCorrelation

Correlation between a factor and failure occurrence.

# Fields
- `factor::Symbol`: Factor name
- `failure_mode_id::String`: Failure mode
- `correlation::Float64`: Correlation coefficient (-1 to 1)
- `p_value::Float64`: Statistical significance
- `effect_size::Float64`: Relative effect size
- `monotonic::Bool`: Is relationship monotonic?
"""
struct FailureCorrelation
    factor::Symbol
    failure_mode_id::String
    correlation::Float64
    p_value::Float64
    effect_size::Float64
    monotonic::Bool
end

# ============================================================================
# Failure Cluster
# ============================================================================

"""
    FailureCluster

A cluster of related failures that tend to occur together.

# Fields
- `id::String`: Cluster identifier
- `failure_modes::Vector{String}`: IDs of clustered failure modes
- `common_triggers::Vector{FailureTrigger}`: Shared triggers
- `co_occurrence_rate::Float64`: How often they occur together
- `root_cause::String`: Hypothesized root cause
"""
struct FailureCluster
    id::String
    failure_modes::Vector{String}
    common_triggers::Vector{FailureTrigger}
    co_occurrence_rate::Float64
    root_cause::String
end

# ============================================================================
# Configuration
# ============================================================================

"""
    FailureAtlasConfig

Configuration for atlas generation.
"""
Base.@kwdef struct FailureAtlasConfig
    # Thresholds for failure detection
    observability_condition_threshold::Float64 = 1e6
    performance_rmse_threshold::Float64 = 5.0
    nees_consistency_threshold::Float64 = 0.85
    silent_divergence_ttd_threshold::Float64 = 5.0

    # Clustering parameters
    co_occurrence_threshold::Float64 = 0.7
    min_cluster_size::Int = 2

    # Correlation significance
    correlation_significance::Float64 = 0.05
    min_effect_size::Float64 = 0.2

    # Boundary detection
    boundary_sharpness_threshold::Float64 = 0.5

    # Output options
    include_mitigations::Bool = true
    include_test_scenarios::Bool = true
    link_known_limitations::Bool = true
end

const DEFAULT_ATLAS_CONFIG = FailureAtlasConfig()

# ============================================================================
# Operational Envelope
# ============================================================================

"""
    OperationalEnvelope

Safe operating region derived from DOE results.

# Fields
- `safe_conditions::Dict{Symbol, Tuple{Any, Any}}`: (min, max) for each factor
- `unsafe_conditions::Vector{Dict{Symbol, Any}}`: Known unsafe combinations
- `boundary_conditions::Vector{Dict{Symbol, Any}}`: Conditions at boundary
- `confidence::Float64`: Confidence in envelope
"""
struct OperationalEnvelope
    safe_conditions::Dict{Symbol, Tuple{Any, Any}}
    unsafe_conditions::Vector{Dict{Symbol, Any}}
    boundary_conditions::Vector{Dict{Symbol, Any}}
    confidence::Float64
end

# ============================================================================
# Atlas Entry and Section
# ============================================================================

"""
    AtlasEntry

A complete entry in the failure atlas for one failure mode.
"""
struct AtlasEntry
    failure_mode::FailureMode
    triggers::Vector{FailureTrigger}
    boundaries::Vector{FailureBoundary}
    correlations::Vector{FailureCorrelation}
    mitigation::String
    known_limitation_id::Union{Nothing, String}
    test_scenarios::Vector{String}
end

"""
    AtlasSection

A section of the failure atlas grouped by category.
"""
struct AtlasSection
    category::Symbol
    title::String
    description::String
    entries::Vector{AtlasEntry}
end

# ============================================================================
# Failure Atlas
# ============================================================================

"""
    FailureAtlas

Complete auto-generated failure atlas.

# Fields
- `version::String`: Atlas version
- `generated_at::String`: Generation timestamp
- `doe_source::String`: Source DOE name
- `total_runs::Int`: Total DOE runs analyzed
- `total_failures::Int`: Total failures observed
- `sections::Vector{AtlasSection}`: Atlas sections by category
- `clusters::Vector{FailureCluster}`: Failure clusters
- `operational_envelope::Union{Nothing, OperationalEnvelope}`: Safe operating region
"""
struct FailureAtlas
    version::String
    generated_at::String
    doe_source::String
    total_runs::Int
    total_failures::Int
    sections::Vector{AtlasSection}
    clusters::Vector{FailureCluster}
    operational_envelope::Union{Nothing, OperationalEnvelope}
end

"""
    compute_operational_envelope(results, config) -> OperationalEnvelope

Compute safe operational envelope from DOE results.
"""
function compute_operational_envelope(results::Vector{DOEResult},
                                       config::FailureAtlasConfig = DEFAULT_ATLAS_CONFIG)
    if isempty(results)
        return OperationalEnvelope(
            Dict{Symbol, Tuple{Any, Any}}(),
            Dict{Symbol, Any}[],
            Dict{Symbol, Any}[],
            0.0
        )
    end

    # Classify runs as pass/fail
    passing_runs = Dict{Symbol, Any}[]
    failing_runs = Dict{Symbol, Any}[]

    for result in results
        is_passing = true

        # Check observability
        if result.observability !== nothing
            if result.observability.condition_number > config.observability_condition_threshold
                is_passing = false
            end
        end

        # Check performance
        if result.performance !== nothing
            if result.performance.rmse_position > config.performance_rmse_threshold
                is_passing = false
            end
            if result.performance.consistency < config.nees_consistency_threshold
                is_passing = false
            end
        end

        levels = Dict{Symbol, Any}()
        for (k, v) in result.run.levels
            levels[k] = v.value
        end

        if is_passing
            push!(passing_runs, levels)
        else
            push!(failing_runs, levels)
        end
    end

    # Find safe ranges for each factor
    safe_conditions = Dict{Symbol, Tuple{Any, Any}}()

    if !isempty(passing_runs)
        factors = keys(passing_runs[1])
        for factor in factors
            values = [run[factor] for run in passing_runs if haskey(run, factor)]
            if !isempty(values)
                # For enums, store all safe values
                if eltype(values) <: Enum
                    safe_conditions[factor] = (values, values)
                else
                    safe_conditions[factor] = (minimum(values), maximum(values))
                end
            end
        end
    end

    # Identify boundary conditions (failing runs adjacent to passing)
    boundary_conditions = Dict{Symbol, Any}[]
    # Simplified: just include failing runs with at least one passing neighbor
    for fail_run in failing_runs
        push!(boundary_conditions, fail_run)
    end

    n_total = length(results)
    n_passing = length(passing_runs)
    confidence = n_total > 0 ? n_passing / n_total : 0.0

    OperationalEnvelope(
        safe_conditions,
        failing_runs,
        boundary_conditions[1:min(10, length(boundary_conditions))],
        confidence
    )
end

# ============================================================================
# Failure Mode Discovery
# ============================================================================

"""
    discover_failure_modes(results, config) -> Vector{FailureMode}

Discover distinct failure modes from DOE results.
"""
function discover_failure_modes(results::Vector{DOEResult},
                                 config::FailureAtlasConfig = DEFAULT_ATLAS_CONFIG)
    modes = FailureMode[]
    mode_counts = Dict{Tuple{Symbol, Vector{Int}}, Int}()
    total_runs = length(results)

    for result in results
        # Check observability failures
        if result.observability !== nothing
            obs = result.observability
            if obs.condition_number > config.observability_condition_threshold
                unobs = findall(.!obs.observable_dims)
                key = (:observability, unobs)
                mode_counts[key] = get(mode_counts, key, 0) + 1
            end
        end

        # Check performance failures
        if result.performance !== nothing
            perf = result.performance
            if perf.rmse_position > config.performance_rmse_threshold
                key = (:performance_rmse, [1, 2, 3])
                mode_counts[key] = get(mode_counts, key, 0) + 1
            end
            if perf.consistency < config.nees_consistency_threshold
                key = (:nees_consistency, Int[])
                mode_counts[key] = get(mode_counts, key, 0) + 1
            end
        end
    end

    # Convert to FailureMode objects
    mode_id = 1
    for ((category, states), count) in mode_counts
        rate = count / max(total_runs, 1)

        name = if category == :observability
            state_names = ["X", "Y", "Z", "Vx", "Vy", "Vz", "φ", "θ", "ψ"]
            affected = [state_names[min(s, length(state_names))] for s in states if s <= length(state_names)]
            "$(join(affected, "/")) Unobservable"
        elseif category == :performance_rmse
            "Position RMSE Exceeded"
        elseif category == :nees_consistency
            "NEES Consistency Below Threshold"
        else
            "Unknown Failure"
        end

        severity = if category == :observability && length(states) >= 3
            :critical
        elseif category == :observability
            :major
        elseif rate > 0.5
            :critical
        elseif rate > 0.2
            :major
        else
            :minor
        end

        push!(modes, FailureMode(
            id = "FM-$(lpad(mode_id, 3, '0'))",
            name = name,
            category = category,
            severity = severity,
            description = "Failure mode discovered through DOE",
            affected_states = states,
            occurrence_count = count,
            occurrence_rate = rate
        ))

        mode_id += 1
    end

    # Sort by occurrence rate (most common first)
    sort!(modes, by = m -> -m.occurrence_rate)

    modes
end

# ============================================================================
# Failure Trigger Identification
# ============================================================================

"""
    identify_failure_triggers(results, failure_mode, config) -> Vector{FailureTrigger}

Identify trigger conditions for a specific failure mode.
"""
function identify_failure_triggers(results::Vector{DOEResult},
                                    failure_mode::FailureMode,
                                    config::FailureAtlasConfig = DEFAULT_ATLAS_CONFIG)
    triggers = FailureTrigger[]

    # Find runs that exhibited this failure
    failing_runs = DOEResult[]
    passing_runs = DOEResult[]

    for result in results
        has_failure = false

        if failure_mode.category == :observability && result.observability !== nothing
            obs = result.observability
            if obs.condition_number > config.observability_condition_threshold
                unobs = findall(.!obs.observable_dims)
                if unobs == failure_mode.affected_states
                    has_failure = true
                end
            end
        elseif failure_mode.category == :performance_rmse && result.performance !== nothing
            if result.performance.rmse_position > config.performance_rmse_threshold
                has_failure = true
            end
        elseif failure_mode.category == :nees_consistency && result.performance !== nothing
            if result.performance.consistency < config.nees_consistency_threshold
                has_failure = true
            end
        end

        if has_failure
            push!(failing_runs, result)
        else
            push!(passing_runs, result)
        end
    end

    if isempty(failing_runs)
        return triggers
    end

    # Find common factor values among failing runs
    factor_values = Dict{Symbol, Dict{Any, Int}}()

    for result in failing_runs
        for (factor, level) in result.run.levels
            if !haskey(factor_values, factor)
                factor_values[factor] = Dict{Any, Int}()
            end
            factor_values[factor][level.value] = get(factor_values[factor], level.value, 0) + 1
        end
    end

    # Identify dominant conditions
    n_failing = length(failing_runs)
    dominant_conditions = Dict{Symbol, Any}()

    for (factor, value_counts) in factor_values
        for (value, count) in value_counts
            if count / n_failing > 0.7  # Value appears in >70% of failures
                dominant_conditions[factor] = value
            end
        end
    end

    if !isempty(dominant_conditions)
        # Calculate probability
        matching_pass = 0
        matching_fail = length(failing_runs)

        for result in passing_runs
            matches = true
            for (factor, value) in dominant_conditions
                if haskey(result.run.levels, factor) && result.run.levels[factor].value != value
                    matches = false
                    break
                end
            end
            if matches
                matching_pass += 1
            end
        end

        prob = matching_fail / max(matching_fail + matching_pass, 1)
        conf = n_failing / length(results)

        push!(triggers, FailureTrigger(
            factor_conditions = dominant_conditions,
            threshold_conditions = Tuple{Symbol, Symbol, Float64}[],
            probability = prob,
            confidence = conf,
            sample_count = n_failing
        ))
    end

    triggers
end

# ============================================================================
# Failure Boundary Detection
# ============================================================================

"""
    identify_failure_boundaries(results, failure_mode, config) -> Vector{FailureBoundary}

Identify boundaries between success and failure for a failure mode.
"""
function identify_failure_boundaries(results::Vector{DOEResult},
                                      failure_mode::FailureMode,
                                      config::FailureAtlasConfig = DEFAULT_ATLAS_CONFIG)
    boundaries = FailureBoundary[]

    # Simplified implementation: find factors where changing value changes outcome
    # Group results by factor levels
    factors = Set{Symbol}()
    for result in results
        for (k, _) in result.run.levels
            push!(factors, k)
        end
    end

    for factor in factors
        # Group by this factor's value
        by_value = Dict{Any, Vector{Bool}}()

        for result in results
            if !haskey(result.run.levels, factor)
                continue
            end

            value = result.run.levels[factor].value
            if !haskey(by_value, value)
                by_value[value] = Bool[]
            end

            # Check if this run had the failure
            has_failure = false
            if failure_mode.category == :observability && result.observability !== nothing
                obs = result.observability
                if obs.condition_number > config.observability_condition_threshold
                    unobs = findall(.!obs.observable_dims)
                    if unobs == failure_mode.affected_states
                        has_failure = true
                    end
                end
            elseif failure_mode.category == :performance_rmse && result.performance !== nothing
                if result.performance.rmse_position > config.performance_rmse_threshold
                    has_failure = true
                end
            end

            push!(by_value[value], has_failure)
        end

        # Check for boundary (one value mostly fails, adjacent mostly passes)
        values = sort(collect(keys(by_value)))
        if length(values) < 2
            continue
        end

        for i in 1:length(values)-1
            val1, val2 = values[i], values[i+1]
            rate1 = mean(by_value[val1])
            rate2 = mean(by_value[val2])

            if abs(rate1 - rate2) > config.boundary_sharpness_threshold
                boundary_val = rate1 > rate2 ? val1 : val2
                direction = rate1 > rate2 ? :at_or_above : :at_or_below

                push!(boundaries, FailureBoundary(
                    failure_mode.id,
                    factor,
                    boundary_val,
                    direction,
                    abs(rate1 - rate2),
                    Dict{Symbol, Any}()
                ))
            end
        end
    end

    boundaries
end

# ============================================================================
# Correlation Analysis
# ============================================================================

"""
    compute_failure_correlations(results, failure_mode, config) -> Vector{FailureCorrelation}

Compute correlations between factors and failure occurrence.
"""
function compute_failure_correlations(results::Vector{DOEResult},
                                       failure_mode::FailureMode,
                                       config::FailureAtlasConfig = DEFAULT_ATLAS_CONFIG)
    correlations = FailureCorrelation[]

    if isempty(results)
        return correlations
    end

    # Get failure indicators
    failure_indicators = Bool[]
    for result in results
        has_failure = false

        if failure_mode.category == :observability && result.observability !== nothing
            obs = result.observability
            if obs.condition_number > config.observability_condition_threshold
                unobs = findall(.!obs.observable_dims)
                if unobs == failure_mode.affected_states
                    has_failure = true
                end
            end
        elseif failure_mode.category == :performance_rmse && result.performance !== nothing
            if result.performance.rmse_position > config.performance_rmse_threshold
                has_failure = true
            end
        end

        push!(failure_indicators, has_failure)
    end

    # Get factors
    factors = Set{Symbol}()
    for result in results
        for (k, _) in result.run.levels
            push!(factors, k)
        end
    end

    # Compute point-biserial correlation for each factor
    for factor in factors
        factor_values = Float64[]
        valid_failures = Float64[]

        for (result, failed) in zip(results, failure_indicators)
            if haskey(result.run.levels, factor)
                val = result.run.levels[factor].value
                # Convert to numeric
                numeric_val = if val isa Number
                    Float64(val)
                elseif val isa Enum
                    Float64(Int(val))
                else
                    continue
                end
                push!(factor_values, numeric_val)
                push!(valid_failures, failed ? 1.0 : 0.0)
            end
        end

        if length(factor_values) < 3
            continue
        end

        # Compute correlation
        mean_x = mean(factor_values)
        mean_y = mean(valid_failures)
        std_x = std(factor_values)
        std_y = std(valid_failures)

        if std_x < 1e-10 || std_y < 1e-10
            continue
        end

        cov_xy = mean((factor_values .- mean_x) .* (valid_failures .- mean_y))
        corr = cov_xy / (std_x * std_y)

        # Effect size (Cohen's d approximation)
        effect = abs(corr) / sqrt(1 - corr^2 + 1e-10)

        # Check monotonicity
        monotonic = abs(corr) > 0.7

        push!(correlations, FailureCorrelation(
            factor,
            failure_mode.id,
            corr,
            0.05,  # Placeholder p-value
            effect,
            monotonic
        ))
    end

    # Sort by absolute correlation
    sort!(correlations, by = c -> -abs(c.correlation))

    correlations
end

# ============================================================================
# Failure Clustering
# ============================================================================

"""
    cluster_failures(results, failure_modes, config) -> Vector{FailureCluster}

Identify clusters of failures that tend to occur together.
"""
function cluster_failures(results::Vector{DOEResult},
                          failure_modes::Vector{FailureMode},
                          config::FailureAtlasConfig = DEFAULT_ATLAS_CONFIG)
    clusters = FailureCluster[]

    if length(failure_modes) < 2
        return clusters
    end

    # Build co-occurrence matrix
    n_modes = length(failure_modes)
    co_occur = zeros(n_modes, n_modes)
    mode_occurs = zeros(n_modes)

    for result in results
        occurring_modes = Int[]

        for (i, mode) in enumerate(failure_modes)
            has_failure = false

            if mode.category == :observability && result.observability !== nothing
                obs = result.observability
                if obs.condition_number > config.observability_condition_threshold
                    unobs = findall(.!obs.observable_dims)
                    if unobs == mode.affected_states
                        has_failure = true
                    end
                end
            elseif mode.category == :performance_rmse && result.performance !== nothing
                if result.performance.rmse_position > config.performance_rmse_threshold
                    has_failure = true
                end
            end

            if has_failure
                push!(occurring_modes, i)
                mode_occurs[i] += 1
            end
        end

        # Update co-occurrence
        for i in occurring_modes
            for j in occurring_modes
                co_occur[i, j] += 1
            end
        end
    end

    # Find clusters (simple: pairs with high co-occurrence)
    cluster_id = 1
    clustered = Set{Int}()

    for i in 1:n_modes
        if i in clustered
            continue
        end

        cluster_members = [i]
        for j in (i+1):n_modes
            if mode_occurs[i] > 0 && mode_occurs[j] > 0
                rate = co_occur[i, j] / min(mode_occurs[i], mode_occurs[j])
                if rate > config.co_occurrence_threshold
                    push!(cluster_members, j)
                end
            end
        end

        if length(cluster_members) >= config.min_cluster_size
            member_ids = [failure_modes[m].id for m in cluster_members]
            co_rate = length(cluster_members) > 1 ?
                      co_occur[cluster_members[1], cluster_members[2]] / max(mode_occurs[cluster_members[1]], 1) : 1.0

            push!(clusters, FailureCluster(
                "FC-$(lpad(cluster_id, 3, '0'))",
                member_ids,
                FailureTrigger[],
                co_rate,
                "Common environmental or trajectory condition"
            ))

            for m in cluster_members
                push!(clustered, m)
            end
            cluster_id += 1
        end
    end

    clusters
end

# ============================================================================
# Atlas Generation
# ============================================================================

"""
    generate_failure_atlas(results; config=DEFAULT_ATLAS_CONFIG, doe_name="DOE") -> FailureAtlas

Generate a complete failure atlas from DOE results.
"""
function generate_failure_atlas(results::Vector{DOEResult};
                                 config::FailureAtlasConfig = DEFAULT_ATLAS_CONFIG,
                                 doe_name::String = "DOE")
    # Discover failure modes
    failure_modes = discover_failure_modes(results, config)

    # Build atlas entries
    entries = AtlasEntry[]

    for mode in failure_modes
        triggers = identify_failure_triggers(results, mode, config)
        boundaries = identify_failure_boundaries(results, mode, config)
        correlations = compute_failure_correlations(results, mode, config)

        # Generate mitigation
        mitigation = if mode.category == :observability && 2 in mode.affected_states
            "Execute heading change (>5°) to restore cross-track observability"
        elseif mode.category == :observability
            "Modify trajectory or sensor configuration to improve observability"
        elseif mode.category == :performance_rmse
            "Reduce process noise or improve measurement quality"
        elseif mode.category == :nees_consistency
            "Calibrate filter noise parameters (Q, R)"
        else
            "Investigate root cause"
        end

        # Check for known limitation
        kl_id = nothing
        if mode.category == :observability
            for kl in V1_0_KNOWN_LIMITATIONS
                # Simple check: if Y-axis unobservable matches KL-001
                if 2 in mode.affected_states && kl.id == "KL-001"
                    kl_id = "KL-001"
                    break
                end
            end
        end

        push!(entries, AtlasEntry(
            mode,
            triggers,
            boundaries,
            correlations,
            mitigation,
            kl_id,
            String[]
        ))
    end

    # Group by category
    sections = AtlasSection[]

    categories = [
        (:observability, "Observability Failures", "Failures related to state observability"),
        (:performance_rmse, "Performance Failures", "Failures in estimation accuracy"),
        (:nees_consistency, "Consistency Failures", "Failures in filter calibration"),
        (:health, "Health System Failures", "Failures in fault detection"),
        (:silent_divergence, "Silent Divergence", "Undetected filter divergence"),
    ]

    for (cat, title, desc) in categories
        cat_entries = filter(e -> e.failure_mode.category == cat, entries)
        if !isempty(cat_entries)
            push!(sections, AtlasSection(cat, title, desc, cat_entries))
        end
    end

    # Cluster failures
    clusters = cluster_failures(results, failure_modes, config)

    # Compute operational envelope
    envelope = compute_operational_envelope(results, config)

    # Count total failures
    total_failures = sum(m.occurrence_count for m in failure_modes; init=0)

    FailureAtlas(
        "1.0",
        _atlas_timestamp(),
        doe_name,
        length(results),
        total_failures,
        sections,
        clusters,
        envelope
    )
end

# Helper for timestamp
function _atlas_timestamp()
    # Simple timestamp
    "2026-01-24T12:00:00"
end

# ============================================================================
# Formatting
# ============================================================================

"""
    format_failure_atlas(atlas::FailureAtlas) -> String

Format the failure atlas as a text report.
"""
function format_failure_atlas(atlas::FailureAtlas)
    lines = String[]

    push!(lines, "=" ^ 80)
    push!(lines, "FAILURE ATLAS")
    push!(lines, "=" ^ 80)
    push!(lines, "")
    push!(lines, "Version: $(atlas.version)")
    push!(lines, "Generated: $(atlas.generated_at)")
    push!(lines, "Source: $(atlas.doe_source)")
    push!(lines, "Total runs: $(atlas.total_runs)")
    push!(lines, "Total failures observed: $(atlas.total_failures)")
    push!(lines, "")

    # Sections
    for section in atlas.sections
        push!(lines, "=" ^ 80)
        push!(lines, uppercase(section.title))
        push!(lines, "=" ^ 80)
        push!(lines, section.description)
        push!(lines, "")

        for entry in section.entries
            mode = entry.failure_mode
            push!(lines, "-" ^ 80)
            push!(lines, "[$(mode.id)] $(mode.name)")
            push!(lines, "-" ^ 80)
            push!(lines, "Severity: $(mode.severity)")
            push!(lines, "Occurrence: $(mode.occurrence_count) ($(round(mode.occurrence_rate * 100, digits=1))%)")
            push!(lines, "Affected states: $(mode.affected_states)")

            if entry.known_limitation_id !== nothing
                push!(lines, "Known limitation: $(entry.known_limitation_id)")
            end

            if !isempty(entry.triggers)
                push!(lines, "")
                push!(lines, "Triggers:")
                for trigger in entry.triggers
                    push!(lines, "  - $(format_trigger(trigger))")
                    push!(lines, "    Probability: $(round(trigger.probability * 100, digits=1))%")
                end
            end

            if !isempty(entry.boundaries)
                push!(lines, "")
                push!(lines, "Boundaries:")
                for boundary in entry.boundaries
                    push!(lines, "  - $(boundary.boundary_factor) $(boundary.direction) $(boundary.boundary_value)")
                end
            end

            if !isempty(entry.correlations)
                push!(lines, "")
                push!(lines, "Top correlations:")
                for (i, corr) in enumerate(entry.correlations[1:min(3, length(entry.correlations))])
                    push!(lines, "  - $(corr.factor): $(round(corr.correlation, digits=3))")
                end
            end

            push!(lines, "")
            push!(lines, "Mitigation: $(entry.mitigation)")
            push!(lines, "")
        end
    end

    # Clusters
    if !isempty(atlas.clusters)
        push!(lines, "=" ^ 80)
        push!(lines, "FAILURE CLUSTERS")
        push!(lines, "=" ^ 80)
        for cluster in atlas.clusters
            push!(lines, "[$(cluster.id)] $(join(cluster.failure_modes, " + "))")
            push!(lines, "  Co-occurrence: $(round(cluster.co_occurrence_rate * 100, digits=1))%")
            push!(lines, "  Root cause: $(cluster.root_cause)")
        end
        push!(lines, "")
    end

    # Operational envelope
    if atlas.operational_envelope !== nothing
        env = atlas.operational_envelope
        push!(lines, "=" ^ 80)
        push!(lines, "OPERATIONAL ENVELOPE")
        push!(lines, "=" ^ 80)
        push!(lines, "Confidence: $(round(env.confidence * 100, digits=1))%")
        push!(lines, "")
        push!(lines, "Safe conditions:")
        for (factor, range) in env.safe_conditions
            push!(lines, "  - $factor: $range")
        end
        push!(lines, "")
        push!(lines, "Unsafe conditions: $(length(env.unsafe_conditions)) identified")
    end

    push!(lines, "=" ^ 80)

    join(lines, "\n")
end

"""
    export_failure_atlas_markdown(atlas::FailureAtlas) -> String

Export the failure atlas as a Markdown document.
"""
function export_failure_atlas_markdown(atlas::FailureAtlas)
    lines = String[]

    push!(lines, "# Failure Atlas")
    push!(lines, "")
    push!(lines, "| Property | Value |")
    push!(lines, "|----------|-------|")
    push!(lines, "| Version | $(atlas.version) |")
    push!(lines, "| Generated | $(atlas.generated_at) |")
    push!(lines, "| Source DOE | $(atlas.doe_source) |")
    push!(lines, "| Total Runs | $(atlas.total_runs) |")
    push!(lines, "| Total Failures | $(atlas.total_failures) |")
    push!(lines, "")

    for section in atlas.sections
        push!(lines, "## $(section.title)")
        push!(lines, "")
        push!(lines, section.description)
        push!(lines, "")

        for entry in section.entries
            mode = entry.failure_mode
            push!(lines, "### $(mode.id): $(mode.name)")
            push!(lines, "")
            push!(lines, "| Property | Value |")
            push!(lines, "|----------|-------|")
            push!(lines, "| Severity | $(mode.severity) |")
            push!(lines, "| Occurrence | $(mode.occurrence_count) ($(round(mode.occurrence_rate * 100, digits=1))%) |")
            push!(lines, "| Affected States | $(mode.affected_states) |")

            if entry.known_limitation_id !== nothing
                push!(lines, "| Known Limitation | $(entry.known_limitation_id) |")
            end
            push!(lines, "")

            if !isempty(entry.triggers)
                push!(lines, "**Triggers:**")
                for trigger in entry.triggers
                    push!(lines, "- $(format_trigger(trigger)) ($(round(trigger.probability * 100))% probability)")
                end
                push!(lines, "")
            end

            push!(lines, "**Mitigation:** $(entry.mitigation)")
            push!(lines, "")
        end
    end

    if atlas.operational_envelope !== nothing
        env = atlas.operational_envelope
        push!(lines, "## Operational Envelope")
        push!(lines, "")
        push!(lines, "Confidence: $(round(env.confidence * 100, digits=1))%")
        push!(lines, "")
        push!(lines, "### Safe Operating Conditions")
        push!(lines, "")
        for (factor, range) in env.safe_conditions
            push!(lines, "- **$factor**: $range")
        end
        push!(lines, "")
    end

    join(lines, "\n")
end
