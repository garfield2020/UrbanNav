# ============================================================================
# Basis Order Ladder (Phase B)
# ============================================================================
#
# Implements adaptive basis complexity selection for map representation.
#
# Purpose: "Use the simplest model that fits the data"
#
# Physics Motivation:
# ------------------
# In source-free regions, B = -∇Φ where Φ satisfies Laplace: ∇²Φ = 0.
# Solutions form a hierarchy of solid harmonics:
#
#   Order 0: Φ = Φ₀                    → B = constant (3 params)
#   Order 1: Φ = Φ₀ + a·x             → B = B₀ + G₀·x (8 params)
#   Order 2: Φ = Φ₀ + a·x + b·(x²-r²/3) → quadratic field (18+ params)
#
# The "order ladder" selects minimum complexity that:
# 1. Fits the data within measurement noise (χ²/dof ≈ 1)
# 2. Has sufficient observations to constrain parameters
# 3. Passes model selection criteria (F-test or AIC)
#
# Key Principle: Occam's Razor with Statistical Teeth
# ---------------------------------------------------
# - Start at lowest order with identifiable parameters
# - Upgrade only when data strongly supports higher complexity
# - Downgrade when data becomes sparse (multi-mission scenarios)
#
# ============================================================================

using LinearAlgebra
using StaticArrays
using Statistics: mean, var

# ============================================================================
# Basis Order Definition
# ============================================================================

"""
    BasisOrder

Enumeration of supported basis complexity levels.

# Orders
- `ORDER_CONSTANT (0)`: Uniform field, 3 parameters
- `ORDER_LINEAR (1)`: Linear gradient, 8 parameters (Phase A default)
- `ORDER_QUADRATIC (2)`: Quadratic terms, 15 parameters (Maxwell-consistent)

# Parameter Counts
The number of independent parameters follows from Maxwell's equations (∇·B=0):
- Order 0: B₀ = 3 (field components)
- Order 1: B₀ + G = 3 + 5 = 8 (gradient is traceless symmetric)
- Order 2: B₀ + G + Q = 3 + 5 + 7 = 15 (quadratic solid harmonics, divergence-free)

# Physical Validity
Higher orders capture local field variation more accurately but:
- Require more calibration data
- May overfit with sparse observations
- Extrapolate poorly beyond training region
"""
@enum BasisOrder begin
    ORDER_CONSTANT = 0  # B = B₀
    ORDER_LINEAR = 1    # B = B₀ + G·(x - x_ref)
    ORDER_QUADRATIC = 2 # B = B₀ + G·Δx + ½Δx'·H·Δx (component-wise)
end

"""
Number of independent tile coefficients for each basis order.

Canonical counts (Maxwell-consistent, ∇·B=0 enforced):
- Order 0 (CONSTANT): 3 (B₀ field components)
- Order 1 (LINEAR):   8 (3 field + 5 traceless symmetric gradient)
- Order 2 (QUADRATIC):15 (3 field + 5 gradient + 7 quadratic solid harmonics)

NOTE: Previously this returned 18 for QUADRATIC (3+5+10 unconstrained Hessian).
That was incorrect for magnetic tile background models where ∇·B=0 removes 3 DOF
from the second-order terms, leaving 7 independent quadratic coefficients.
The canonical 15 matches BasisEnrichment.COEFFICIENT_COUNTS[QUADRATIC].
"""
function n_parameters(order::BasisOrder)
    if order == ORDER_CONSTANT
        return 3
    elseif order == ORDER_LINEAR
        return 8   # 3 field + 5 gradient (traceless symmetric)
    elseif order == ORDER_QUADRATIC
        return 15  # 3 + 5 + 7 (Maxwell-consistent solid harmonics)
    else
        error("Unknown basis order: $order")
    end
end

"""
    minimum_observations(order::BasisOrder) -> Int

Minimum number of observations required to fit this order.

# Justification
For least squares: n_obs ≥ n_params for identifiability.
We require 2× the minimum for robustness:
- Provides degrees of freedom for χ² test
- Reduces sensitivity to outliers
- Enables residual-based model validation

This is a conservative choice; 1.5× would be acceptable but 2× is safer.
"""
function minimum_observations(order::BasisOrder)
    # 2× minimum for robustness (standard practice in regression)
    return 2 * n_parameters(order)
end

"""
    effective_observations(n_field::Int, n_gradient::Int) -> Int

Compute effective observation count from field and gradient measurements.

# Arguments
- `n_field`: Number of field (B) measurements
- `n_gradient`: Number of gradient (G) measurements

# Returns
Effective observation count (field = 3 DOF, gradient = 5 DOF per measurement)

# Note
Each field measurement provides 3 scalar observations.
Each gradient measurement provides 5 scalar observations (traceless symmetric).
"""
function effective_observations(n_field::Int, n_gradient::Int)
    return 3 * n_field + 5 * n_gradient
end

# ============================================================================
# Order Selection Criteria
# ============================================================================

"""
    OrderSelectionConfig

Configuration for basis order selection.

# Fields
- `f_test_alpha::Float64`: Significance level for F-test (default: 0.05)
- `aic_delta_threshold::Float64`: Minimum AIC improvement to upgrade (default: 2.0)
- `min_dof_ratio::Float64`: Minimum DOF/n_params ratio (default: 2.0)
- `max_order::BasisOrder`: Maximum allowed order (default: ORDER_LINEAR)
- `prefer_parsimony::Bool`: Tie-break toward simpler model (default: true)

# Threshold Justifications

**f_test_alpha = 0.05**:
Standard 95% confidence level for nested model comparison.
This is the probability of incorrectly upgrading when simpler model is adequate.
Lower values (e.g., 0.01) would be more conservative.

**aic_delta_threshold = 2.0**:
Burnham & Anderson (2002) criterion: ΔAIC < 2 indicates substantial support
for both models. We require ΔAIC > 2 to upgrade, meaning the complex model
must have substantially better fit.

**min_dof_ratio = 2.0**:
With dof = n_obs - n_params, requiring dof ≥ 2×n_params ensures:
- At least 50% of observations inform residual variance
- Robust χ² estimate for model validation
- Buffer for outlier rejection
"""
struct OrderSelectionConfig
    f_test_alpha::Float64
    aic_delta_threshold::Float64
    min_dof_ratio::Float64
    max_order::BasisOrder
    prefer_parsimony::Bool

    function OrderSelectionConfig(;
        f_test_alpha::Float64 = 0.05,
        aic_delta_threshold::Float64 = 2.0,
        min_dof_ratio::Float64 = 2.0,
        max_order::BasisOrder = ORDER_LINEAR,
        prefer_parsimony::Bool = true
    )
        @assert 0 < f_test_alpha < 1 "f_test_alpha must be in (0, 1)"
        @assert aic_delta_threshold > 0 "aic_delta_threshold must be positive"
        @assert min_dof_ratio >= 1.0 "min_dof_ratio must be at least 1.0"
        new(f_test_alpha, aic_delta_threshold, min_dof_ratio, max_order, prefer_parsimony)
    end
end

const DEFAULT_ORDER_SELECTION_CONFIG = OrderSelectionConfig()

# ============================================================================
# Statistical Model Selection
# ============================================================================

"""
    ModelFitStatistics

Statistics for comparing model fits.

# Fields
- `order::BasisOrder`: Which order model was fit
- `n_observations::Int`: Number of scalar observations
- `n_parameters::Int`: Number of model parameters
- `dof::Int`: Degrees of freedom (n_obs - n_params)
- `rss::Float64`: Residual sum of squares (χ² statistic)
- `sigma2::Float64`: Estimated variance (RSS/DOF)
- `aic::Float64`: Akaike Information Criterion
- `bic::Float64`: Bayesian Information Criterion
"""
struct ModelFitStatistics
    order::BasisOrder
    n_observations::Int
    n_parameters::Int
    dof::Int
    rss::Float64
    sigma2::Float64
    aic::Float64
    bic::Float64
end

"""
Compute AIC: AIC = n·log(RSS/n) + 2k

where n = observations, k = parameters, RSS = residual sum of squares.

Reference: Akaike, H. (1974). "A new look at the statistical model identification"
"""
function compute_aic(n::Int, k::Int, rss::Float64)
    if n <= 0 || rss <= 0
        return Inf
    end
    return n * log(rss / n) + 2 * k
end

"""
Compute BIC: BIC = n·log(RSS/n) + k·log(n)

BIC penalizes complexity more heavily than AIC for large samples.

Reference: Schwarz, G. (1978). "Estimating the dimension of a model"
"""
function compute_bic(n::Int, k::Int, rss::Float64)
    if n <= 0 || rss <= 0
        return Inf
    end
    return n * log(rss / n) + k * log(n)
end

"""Create ModelFitStatistics from fit results."""
function ModelFitStatistics(order::BasisOrder, n_obs::Int, chi2::Float64)
    n_params = n_parameters(order)
    dof = n_obs - n_params

    if dof <= 0
        # Underdetermined - return invalid statistics
        return ModelFitStatistics(order, n_obs, n_params, dof, chi2, Inf, Inf, Inf)
    end

    sigma2 = chi2 / dof
    aic = compute_aic(n_obs, n_params, chi2)
    bic = compute_bic(n_obs, n_params, chi2)

    return ModelFitStatistics(order, n_obs, n_params, dof, chi2, sigma2, aic, bic)
end

"""
    f_test(stats_simple::ModelFitStatistics, stats_complex::ModelFitStatistics) -> (F, p_value)

Perform F-test comparing nested models.

The F-statistic tests whether the complex model provides significantly better fit:

    F = [(RSS₁ - RSS₂) / (df₁ - df₂)] / [RSS₂ / df₂]

where model 1 is simpler (more DOF) and model 2 is complex (fewer DOF).

# Returns
- `F`: F-statistic
- `p_value`: Probability of observing this F or larger under null hypothesis
             (that simpler model is adequate)

# Interpretation
- Large F, small p-value → complex model fits significantly better
- Small F, large p-value → no evidence that complex model is needed

# Reference
Draper, N.R. & Smith, H. (1998). Applied Regression Analysis, 3rd ed.
"""
function f_test(stats_simple::ModelFitStatistics, stats_complex::ModelFitStatistics)
    rss1 = stats_simple.rss
    rss2 = stats_complex.rss
    df1 = stats_simple.dof
    df2 = stats_complex.dof

    # Sanity checks
    if df1 <= df2
        error("Simple model must have more DOF than complex model")
    end
    if rss2 <= 0 || df2 <= 0
        return (Inf, 0.0)
    end

    # F-statistic
    delta_rss = rss1 - rss2
    delta_df = df1 - df2

    if delta_rss < 0
        # Complex model has higher RSS (worse fit) - shouldn't happen but handle gracefully
        return (0.0, 1.0)
    end

    F = (delta_rss / delta_df) / (rss2 / df2)

    # Approximate p-value using F-distribution
    # For simplicity, use Snedecor's F-distribution approximation
    # In production, use Distributions.jl: ccdf(FDist(delta_df, df2), F)
    p_value = approximate_f_pvalue(F, delta_df, df2)

    return (F, p_value)
end

"""
Approximate p-value for F-distribution using Abramowitz & Stegun approximation.

This is a reasonable approximation for df1, df2 > 5.
For exact values, use Distributions.jl.
"""
function approximate_f_pvalue(F::Float64, df1::Int, df2::Int)
    if F <= 0
        return 1.0
    end
    if df1 <= 0 || df2 <= 0
        return 0.0
    end

    # Wilson-Hilferty approximation for F-distribution
    # Transform to approximate normal
    a = df1 / (df1 + df2 * F)
    b = df2 / (df1 + df2 * F)

    # Use chi-square approximation
    # P(F > f) ≈ P(χ²(df1) > df1*a / b)
    x = df1 * F / (df1 * F + df2)

    # Beta incomplete function approximation
    # For large df, approximate using normal
    if df1 > 10 && df2 > 10
        z = sqrt(2*F) - sqrt(2*df1 - 1)
        p = 0.5 * erfc(z / sqrt(2))
        return clamp(p, 0.0, 1.0)
    end

    # Crude approximation for small df
    # F > 4 is usually significant at α=0.05
    if F > 4.0
        return exp(-0.5 * F)  # Very approximate
    else
        return 1.0 - F / 10.0
    end
end

"""
    compare_aic(stats1::ModelFitStatistics, stats2::ModelFitStatistics) -> Float64

Compute AIC difference: ΔAIC = AIC(model1) - AIC(model2).

Positive ΔAIC means model2 is preferred.

# Interpretation (Burnham & Anderson, 2002)
- ΔAIC < 2: Substantial support for both models
- 2 < ΔAIC < 4: Moderate support for worse model
- 4 < ΔAIC < 7: Less support for worse model
- ΔAIC > 10: Essentially no support for worse model
"""
function compare_aic(stats1::ModelFitStatistics, stats2::ModelFitStatistics)
    return stats1.aic - stats2.aic
end

# ============================================================================
# Order Selection Logic
# ============================================================================

"""
    OrderSelectionResult

Result of automatic order selection.

# Fields
- `selected_order::BasisOrder`: Recommended basis order
- `statistics::Dict{BasisOrder, ModelFitStatistics}`: Statistics for each order tried
- `upgrade_blocked::Bool`: Whether upgrade was blocked by insufficient data
- `selection_reason::Symbol`: Why this order was selected
- `details::String`: Human-readable explanation
"""
struct OrderSelectionResult
    selected_order::BasisOrder
    statistics::Dict{BasisOrder, ModelFitStatistics}
    upgrade_blocked::Bool
    selection_reason::Symbol
    details::String
end

"""
    select_basis_order(n_observations::Int,
                       fit_chi2::Dict{BasisOrder, Float64},
                       config::OrderSelectionConfig = DEFAULT_ORDER_SELECTION_CONFIG)

Select optimal basis order given available data.

# Arguments
- `n_observations`: Number of effective scalar observations
- `fit_chi2`: χ² statistic for each order that was fit
- `config`: Selection configuration

# Returns
OrderSelectionResult with recommended order and justification.

# Algorithm
1. Filter to orders with sufficient DOF
2. Compute AIC for each valid order
3. Select minimum AIC order (with parsimony tie-break)
4. Verify selection with F-test if upgrading
"""
function select_basis_order(n_observations::Int,
                            fit_chi2::Dict{BasisOrder, Float64},
                            config::OrderSelectionConfig = DEFAULT_ORDER_SELECTION_CONFIG)
    stats = Dict{BasisOrder, ModelFitStatistics}()
    valid_orders = BasisOrder[]

    # Compute statistics for each fitted order
    for (order, chi2) in fit_chi2
        # Check if this order is allowed
        if Int(order) > Int(config.max_order)
            continue
        end

        s = ModelFitStatistics(order, n_observations, chi2)
        stats[order] = s

        # Check DOF requirement
        min_dof = config.min_dof_ratio * n_parameters(order)
        if s.dof >= min_dof
            push!(valid_orders, order)
        end
    end

    # Handle edge cases
    if isempty(valid_orders)
        # No order has sufficient DOF - use lowest available
        lowest = minimum(keys(fit_chi2))
        return OrderSelectionResult(
            lowest,
            stats,
            true,
            :insufficient_data,
            "Insufficient observations for any order; defaulting to $lowest"
        )
    end

    # Sort by AIC
    sorted_orders = sort(valid_orders, by = o -> stats[o].aic)
    best_by_aic = sorted_orders[1]

    # Check if parsimony applies
    if length(sorted_orders) > 1 && config.prefer_parsimony
        # If two orders have similar AIC (within threshold), prefer simpler
        second_best = sorted_orders[2]
        delta_aic = stats[second_best].aic - stats[best_by_aic].aic

        if delta_aic < config.aic_delta_threshold
            # Not enough evidence to prefer complex model
            simpler = min(best_by_aic, second_best)
            return OrderSelectionResult(
                simpler,
                stats,
                false,
                :parsimony,
                "AIC difference $(round(delta_aic, digits=2)) < threshold $(config.aic_delta_threshold); selecting simpler order $simpler"
            )
        end
    end

    # Check F-test if selecting higher order than constant
    if best_by_aic != ORDER_CONSTANT && ORDER_CONSTANT in valid_orders
        F, p = f_test(stats[ORDER_CONSTANT], stats[best_by_aic])
        if p > config.f_test_alpha
            # F-test doesn't support upgrade
            return OrderSelectionResult(
                ORDER_CONSTANT,
                stats,
                false,
                :f_test_fail,
                "F-test p-value $(round(p, digits=3)) > α=$(config.f_test_alpha); upgrade not justified"
            )
        end
    end

    # Return best by AIC
    return OrderSelectionResult(
        best_by_aic,
        stats,
        false,
        :aic_minimum,
        "Selected $best_by_aic with AIC=$(round(stats[best_by_aic].aic, digits=2))"
    )
end

"""
    can_upgrade_order(current::BasisOrder, n_observations::Int,
                      config::OrderSelectionConfig = DEFAULT_ORDER_SELECTION_CONFIG) -> Bool

Check if upgrade to next order is feasible given available data.

This is a quick check before attempting to fit a higher-order model.
"""
function can_upgrade_order(current::BasisOrder, n_observations::Int,
                           config::OrderSelectionConfig = DEFAULT_ORDER_SELECTION_CONFIG)
    if Int(current) >= Int(config.max_order)
        return false
    end

    next_order = BasisOrder(Int(current) + 1)
    min_obs = minimum_observations(next_order)

    return n_observations >= min_obs
end

"""
    should_downgrade_order(current::BasisOrder, n_observations::Int,
                           config::OrderSelectionConfig = DEFAULT_ORDER_SELECTION_CONFIG) -> Bool

Check if current order should be downgraded due to insufficient data.

This can happen when observations become sparse (e.g., vehicle moves to new area).
"""
function should_downgrade_order(current::BasisOrder, n_observations::Int,
                                config::OrderSelectionConfig = DEFAULT_ORDER_SELECTION_CONFIG)
    if current == ORDER_CONSTANT
        return false  # Can't go lower
    end

    # Check if we have enough DOF for current order
    n_params = n_parameters(current)
    min_dof = config.min_dof_ratio * n_params
    current_dof = n_observations - n_params

    return current_dof < min_dof
end

# ============================================================================
# Order Transition Management
# ============================================================================

"""
    OrderTransition

Record of a basis order change.

# Fields
- `from_order::BasisOrder`: Previous order
- `to_order::BasisOrder`: New order
- `direction::Symbol`: :upgrade, :downgrade, or :maintain
- `reason::Symbol`: Why transition occurred
- `n_observations::Int`: Observations at time of transition
- `aic_improvement::Float64`: AIC change (negative = improvement)
"""
struct OrderTransition
    from_order::BasisOrder
    to_order::BasisOrder
    direction::Symbol
    reason::Symbol
    n_observations::Int
    aic_improvement::Float64
end

"""
    OrderLadderState

Tracks current position on the order ladder with history.

# Fields
- `current_order::BasisOrder`: Current basis complexity
- `observation_count::Int`: Current effective observation count
- `transitions::Vector{OrderTransition}`: History of order changes
- `locked::Bool`: Whether order changes are disabled
- `lock_reason::String`: Why order is locked (if applicable)
"""
mutable struct OrderLadderState
    current_order::BasisOrder
    observation_count::Int
    transitions::Vector{OrderTransition}
    locked::Bool
    lock_reason::String
end

"""Create initial ladder state."""
function OrderLadderState(initial_order::BasisOrder = ORDER_CONSTANT)
    OrderLadderState(initial_order, 0, OrderTransition[], false, "")
end

"""Lock the order ladder (prevent automatic changes)."""
function lock_order!(state::OrderLadderState, reason::String = "Manual lock")
    state.locked = true
    state.lock_reason = reason
end

"""Unlock the order ladder (allow automatic changes)."""
function unlock_order!(state::OrderLadderState)
    state.locked = false
    state.lock_reason = ""
end

"""
    update_ladder!(state::OrderLadderState,
                   n_observations::Int,
                   fit_chi2::Dict{BasisOrder, Float64},
                   config::OrderSelectionConfig = DEFAULT_ORDER_SELECTION_CONFIG)

Update ladder state based on current data.

# Returns
OrderTransition if order changed, nothing otherwise.
"""
function update_ladder!(state::OrderLadderState,
                        n_observations::Int,
                        fit_chi2::Dict{BasisOrder, Float64},
                        config::OrderSelectionConfig = DEFAULT_ORDER_SELECTION_CONFIG)
    state.observation_count = n_observations

    if state.locked
        return nothing
    end

    # Select optimal order
    result = select_basis_order(n_observations, fit_chi2, config)

    if result.selected_order == state.current_order
        return nothing  # No change
    end

    # Record transition
    direction = if Int(result.selected_order) > Int(state.current_order)
        :upgrade
    else
        :downgrade
    end

    # Compute AIC improvement
    aic_old = get(result.statistics, state.current_order, nothing)
    aic_new = get(result.statistics, result.selected_order, nothing)
    aic_improvement = if aic_old !== nothing && aic_new !== nothing
        aic_old.aic - aic_new.aic
    else
        0.0
    end

    transition = OrderTransition(
        state.current_order,
        result.selected_order,
        direction,
        result.selection_reason,
        n_observations,
        aic_improvement
    )

    push!(state.transitions, transition)
    state.current_order = result.selected_order

    return transition
end

# ============================================================================
# Formatting and Display
# ============================================================================

"""Format order selection result as human-readable string."""
function format_selection_result(result::OrderSelectionResult)
    lines = String[]
    push!(lines, "Basis Order Selection")
    push!(lines, "=" ^ 40)
    push!(lines, "Selected: $(result.selected_order)")
    push!(lines, "Reason: $(result.selection_reason)")
    push!(lines, "")

    push!(lines, "Model Statistics:")
    for (order, stats) in sort(collect(result.statistics), by = x -> Int(x[1]))
        push!(lines, "  $order:")
        push!(lines, "    DOF: $(stats.dof)")
        push!(lines, "    χ²/DOF: $(round(stats.sigma2, digits=3))")
        push!(lines, "    AIC: $(round(stats.aic, digits=2))")
        push!(lines, "    BIC: $(round(stats.bic, digits=2))")
    end

    if result.upgrade_blocked
        push!(lines, "")
        push!(lines, "⚠ Upgrade blocked: insufficient data")
    end

    push!(lines, "")
    push!(lines, "Details: $(result.details)")

    return join(lines, "\n")
end

"""Format ladder state as human-readable string."""
function format_ladder_state(state::OrderLadderState)
    lines = String[]
    push!(lines, "Order Ladder State")
    push!(lines, "=" ^ 40)
    push!(lines, "Current order: $(state.current_order)")
    push!(lines, "Observations: $(state.observation_count)")

    if state.locked
        push!(lines, "Status: LOCKED ($(state.lock_reason))")
    else
        push!(lines, "Status: Active")
    end

    if !isempty(state.transitions)
        push!(lines, "")
        push!(lines, "Transition History:")
        for (i, t) in enumerate(state.transitions)
            push!(lines, "  $i. $(t.from_order) → $(t.to_order) ($(t.direction))")
            push!(lines, "     Reason: $(t.reason), n=$(t.n_observations)")
        end
    end

    return join(lines, "\n")
end

# ============================================================================
# Exports
# ============================================================================

export BasisOrder, ORDER_CONSTANT, ORDER_LINEAR, ORDER_QUADRATIC
export n_parameters, minimum_observations, effective_observations
export OrderSelectionConfig, DEFAULT_ORDER_SELECTION_CONFIG
export ModelFitStatistics, compute_aic, compute_bic
export f_test, compare_aic
export OrderSelectionResult, select_basis_order
export can_upgrade_order, should_downgrade_order
export OrderTransition, OrderLadderState
export lock_order!, unlock_order!, update_ladder!
export format_selection_result, format_ladder_state
