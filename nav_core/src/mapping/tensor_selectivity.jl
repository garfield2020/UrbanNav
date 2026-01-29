# ============================================================================
# Tensor Selectivity - Intelligent Measurement Mode Selection
# ============================================================================
#
# Ported from AUV-Navigation/src/tensor_selectivity.jl
#
# Intelligently selects between measurement modes:
#   - B-only (d=3): Field only, most robust at long range
#   - G-only (d=5): Gradient only, rarely used alone
#   - Full tensor (d=8): Field + gradient, best at close range
#
# Key insight: Gradients fall off as 1/r⁴ vs B as 1/r³, so gradient SNR
# degrades faster with range. At long range, adding noisy gradients HURTS.
# ============================================================================

using LinearAlgebra
using StaticArrays
using Statistics: mean

# ============================================================================
# Measurement Mode Enum
# ============================================================================

"""
    MeasurementMode

Available measurement modes for tensor estimation.
"""
@enum MeasurementMode begin
    MODE_FIELD_ONLY      # d=3, B only
    MODE_GRADIENT_ONLY   # d=5, G only (rarely used)
    MODE_FULL_TENSOR     # d=8, B + G
end

function measurement_dimension(mode::MeasurementMode)
    if mode == MODE_FIELD_ONLY
        return 3
    elseif mode == MODE_GRADIENT_ONLY
        return 5
    else
        return 8
    end
end

# ============================================================================
# SNR-Based Selection Model
# ============================================================================

"""
    TensorSelectivityConfig

Configuration for tensor measurement mode selection.

# Fields
- `noise_B::Float64`: Field measurement noise (T)
- `noise_G::Float64`: Gradient measurement noise (T/m)
- `snr_threshold_gradient::Float64`: Minimum gradient SNR to include gradients
- `snr_threshold_field::Float64`: Minimum field SNR for any detection
- `condition_number_limit::Float64`: Max condition number before mode downgrade
- `always_use_gradient::Bool`: Force full tensor mode (ignore SNR)
- `never_use_gradient::Bool`: Force field-only mode
"""
struct TensorSelectivityConfig
    noise_B::Float64
    noise_G::Float64
    snr_threshold_gradient::Float64
    snr_threshold_field::Float64
    condition_number_limit::Float64
    always_use_gradient::Bool
    never_use_gradient::Bool
end

function TensorSelectivityConfig(;
    noise_B::Float64 = 2.34e-9,         # From tetrahedron model (2.34 nT)
    noise_G::Float64 = 15.59e-9,        # From tetrahedron model (15.59 nT/m)
    snr_threshold_gradient::Float64 = 2.0,  # Min gradient SNR to use
    snr_threshold_field::Float64 = 3.0,     # Min field SNR for detection
    condition_number_limit::Float64 = 1e6,  # Max condition number
    always_use_gradient::Bool = false,
    never_use_gradient::Bool = false
)
    TensorSelectivityConfig(
        noise_B, noise_G,
        snr_threshold_gradient, snr_threshold_field,
        condition_number_limit,
        always_use_gradient, never_use_gradient
    )
end

const DEFAULT_SELECTIVITY_CONFIG = TensorSelectivityConfig()

# ============================================================================
# SNR Estimation
# ============================================================================

"""
    estimate_gradient_snr(B_magnitude, G_magnitude, config)

Estimate gradient SNR from measured magnitudes.

Returns (snr_B, snr_G, ratio).
"""
function estimate_gradient_snr(B_magnitude::Float64, G_magnitude::Float64,
                                config::TensorSelectivityConfig)
    snr_B = B_magnitude / config.noise_B
    snr_G = G_magnitude / config.noise_G
    ratio = snr_G / max(snr_B, 1e-10)
    return (snr_B = snr_B, snr_G = snr_G, ratio = ratio)
end

"""
    estimate_snr_at_range(moment, range, config)

Estimate expected SNR at given range for a dipole source.

Physics:
- |B| ∝ μ₀m / (2πr³)
- |G| ∝ 3μ₀m / (2πr⁴)
"""
function estimate_snr_at_range(moment::Float64, range::Float64,
                                config::TensorSelectivityConfig)
    μ0 = 4π * 1e-7

    # Approximate field and gradient magnitudes
    B_mag = μ0 * moment / (2π * range^3)
    G_mag = 3 * μ0 * moment / (2π * range^4)

    return estimate_gradient_snr(B_mag, G_mag, config)
end

"""
    crossover_range(moment, config)

Find range where gradient SNR equals field SNR.

Below this range, gradients are relatively more informative.
Above this range, field dominates.
"""
function crossover_range(moment::Float64, config::TensorSelectivityConfig)
    # SNR_G / SNR_B = (3 × σ_B) / (r × σ_G) = 1
    # r = 3 × σ_B / σ_G
    return 3 * config.noise_B / config.noise_G
end

# ============================================================================
# Mode Selection Logic
# ============================================================================

"""
    ModeSelectionResult

Result of measurement mode selection.
"""
struct ModeSelectionResult
    mode::MeasurementMode
    snr_B::Float64
    snr_G::Float64
    reason::String
    dimension::Int
end

"""
    select_measurement_mode(B_meas, G_meas; config)

Select optimal measurement mode based on current measurements.

Decision logic:
1. If gradient SNR < threshold → B-only
2. If combined covariance ill-conditioned → B-only
3. Otherwise → Full tensor
"""
function select_measurement_mode(B_meas::AbstractVector, G_meas::AbstractVector;
                                  config::TensorSelectivityConfig = DEFAULT_SELECTIVITY_CONFIG)
    # Handle forced modes
    if config.never_use_gradient
        return ModeSelectionResult(MODE_FIELD_ONLY, 0.0, 0.0, "forced B-only", 3)
    end

    if config.always_use_gradient
        return ModeSelectionResult(MODE_FULL_TENSOR, 0.0, 0.0, "forced full tensor", 8)
    end

    # Compute magnitudes
    B_mag = norm(B_meas)
    G_mag = norm(G_meas)

    # Estimate SNR
    snr = estimate_gradient_snr(B_mag, G_mag, config)

    # Decision logic
    if snr.snr_B < config.snr_threshold_field
        # Very low signal - still use B-only for detection attempt
        return ModeSelectionResult(MODE_FIELD_ONLY, snr.snr_B, snr.snr_G,
                              "low field SNR ($(round(snr.snr_B, digits=1)))", 3)
    end

    if snr.snr_G < config.snr_threshold_gradient
        # Gradient SNR too low - would add more noise than signal
        return ModeSelectionResult(MODE_FIELD_ONLY, snr.snr_B, snr.snr_G,
                              "low gradient SNR ($(round(snr.snr_G, digits=1)) < $(config.snr_threshold_gradient))", 3)
    end

    # Gradient SNR acceptable - use full tensor
    return ModeSelectionResult(MODE_FULL_TENSOR, snr.snr_B, snr.snr_G,
                          "gradient SNR OK ($(round(snr.snr_G, digits=1)))", 8)
end

"""Safe condition number that handles singular matrices."""
function cond_safe(A::AbstractMatrix)
    try
        return cond(A)
    catch
        return Inf
    end
end

"""
    select_measurement_mode_with_covariance(B_meas, G_meas, Σ_total; config)

Select mode considering full innovation covariance.

Checks condition number of combined covariance to avoid numerical issues.
"""
function select_measurement_mode_with_covariance(
    B_meas::AbstractVector,
    G_meas::AbstractVector,
    Σ_total::AbstractMatrix;
    config::TensorSelectivityConfig = DEFAULT_SELECTIVITY_CONFIG
)
    # First check SNR
    result = select_measurement_mode(B_meas, G_meas; config=config)

    if result.mode == MODE_FIELD_ONLY
        return result  # Already downgraded
    end

    # Check condition number of full Σ_total
    if size(Σ_total, 1) == 8
        cn = cond_safe(Σ_total)
        if cn > config.condition_number_limit
            return ModeSelectionResult(MODE_FIELD_ONLY, result.snr_B, result.snr_G,
                                  "ill-conditioned Σ_total (cond=$(round(cn, sigdigits=2)))", 3)
        end
    end

    return result
end

# ============================================================================
# Fisher Information Analysis
# ============================================================================

"""
    GradientFisherInfo

Fisher information comparison for mode selection.
"""
struct GradientFisherInfo
    fisher_B::Float64          # Trace of Fisher info for B-only
    fisher_full::Float64       # Trace of Fisher info for full tensor
    information_gain::Float64  # Ratio full/B
    recommended_mode::MeasurementMode
end

"""
    compute_gradient_fisher_gain(range, moment; config)

Compute information gain from adding gradients at given range.
"""
function compute_gradient_fisher_gain(range::Float64, moment::Float64;
                                       config::TensorSelectivityConfig = DEFAULT_SELECTIVITY_CONFIG)
    μ0 = 4π * 1e-7

    # Signal magnitudes
    B_mag = μ0 * moment / (2π * range^3)
    G_mag = 3 * μ0 * moment / (2π * range^4)

    # Fisher info (simplified: proportional to SNR²)
    fisher_B = (B_mag / config.noise_B)^2
    fisher_G = (G_mag / config.noise_G)^2

    fisher_full = fisher_B + fisher_G
    gain = fisher_full / max(fisher_B, 1e-10)

    # Recommend mode based on gain
    mode = gain > 1.1 ? MODE_FULL_TENSOR : MODE_FIELD_ONLY

    return GradientFisherInfo(fisher_B, fisher_full, gain, mode)
end

"""
    optimal_gradient_range(moment; config, gain_threshold)

Find range below which gradients provide significant information gain.
"""
function optimal_gradient_range(moment::Float64;
                                 config::TensorSelectivityConfig = DEFAULT_SELECTIVITY_CONFIG,
                                 gain_threshold::Float64 = 1.5)
    # Binary search for range where gain = threshold
    lo, hi = 0.5, 50.0

    while hi - lo > 0.1
        mid = (lo + hi) / 2
        info = compute_gradient_fisher_gain(mid, moment; config=config)
        if info.information_gain > gain_threshold
            lo = mid
        else
            hi = mid
        end
    end

    return (lo + hi) / 2
end

# ============================================================================
# Adaptive Mode Selector (Stateful)
# ============================================================================

"""
    AdaptiveModeSelector

Stateful mode selector that adapts based on recent measurements.

Tracks:
- Recent SNR history
- Mode switching frequency
- Performance metrics per mode
"""
mutable struct AdaptiveModeSelector
    config::TensorSelectivityConfig
    current_mode::MeasurementMode

    # History tracking
    snr_B_history::Vector{Float64}
    snr_G_history::Vector{Float64}
    mode_history::Vector{MeasurementMode}
    history_window::Int

    # Statistics
    mode_switches::Int
    measurements_B_only::Int
    measurements_full::Int
end

function AdaptiveModeSelector(;
    config::TensorSelectivityConfig = DEFAULT_SELECTIVITY_CONFIG,
    history_window::Int = 20
)
    AdaptiveModeSelector(
        config,
        MODE_FIELD_ONLY,  # Start conservative
        Float64[],
        Float64[],
        MeasurementMode[],
        history_window,
        0, 0, 0
    )
end

"""
    update_mode_selector!(selector, B_meas, G_meas) -> ModeSelectionResult

Update selector with new measurement and get recommended mode.

Uses hysteresis to avoid rapid mode switching.
"""
function update_mode_selector!(selector::AdaptiveModeSelector,
                                B_meas::AbstractVector,
                                G_meas::AbstractVector)
    # Get base recommendation
    result = select_measurement_mode(B_meas, G_meas; config=selector.config)

    # Update history
    push!(selector.snr_B_history, result.snr_B)
    push!(selector.snr_G_history, result.snr_G)
    push!(selector.mode_history, result.mode)

    # Trim history
    if length(selector.snr_B_history) > selector.history_window
        popfirst!(selector.snr_B_history)
        popfirst!(selector.snr_G_history)
        popfirst!(selector.mode_history)
    end

    # Hysteresis: require sustained SNR to switch modes
    if result.mode != selector.current_mode
        # Count recent recommendations for new mode
        n_recent = min(5, length(selector.mode_history))
        recent = selector.mode_history[max(1, end-n_recent+1):end]
        votes_for_new = count(m -> m == result.mode, recent)

        if votes_for_new >= 3  # 3 out of last 5
            selector.current_mode = result.mode
            selector.mode_switches += 1
        end
    end

    # Update statistics
    if selector.current_mode == MODE_FIELD_ONLY
        selector.measurements_B_only += 1
    else
        selector.measurements_full += 1
    end

    # Return result with possibly overridden mode due to hysteresis
    return ModeSelectionResult(
        selector.current_mode,
        result.snr_B,
        result.snr_G,
        result.reason * (selector.current_mode != result.mode ? " [hysteresis]" : ""),
        measurement_dimension(selector.current_mode)
    )
end

"""Get selector statistics."""
function get_selector_statistics(selector::AdaptiveModeSelector)
    total = selector.measurements_B_only + selector.measurements_full
    return (
        current_mode = selector.current_mode,
        mode_switches = selector.mode_switches,
        measurements_B_only = selector.measurements_B_only,
        measurements_full = selector.measurements_full,
        full_tensor_rate = total > 0 ? selector.measurements_full / total : 0.0,
        mean_snr_B = isempty(selector.snr_B_history) ? 0.0 : mean(selector.snr_B_history),
        mean_snr_G = isempty(selector.snr_G_history) ? 0.0 : mean(selector.snr_G_history)
    )
end

# ============================================================================
# Integration with Residual Manager
# ============================================================================

"""
    apply_mode_selection(B_residual, G_residual, mode)

Apply mode selection to residuals, zeroing out unused components.

Returns (effective_residual, effective_dimension).
"""
function apply_mode_selection(B_residual::AbstractVector,
                               G_residual::AbstractVector,
                               mode::MeasurementMode)
    if mode == MODE_FIELD_ONLY
        return (SVector{3}(B_residual...), 3)
    elseif mode == MODE_GRADIENT_ONLY
        return (SVector{5}(G_residual...), 5)
    else
        return (vcat(SVector{3}(B_residual...), SVector{5}(G_residual...)), 8)
    end
end

"""
    build_mode_covariance(mode, noise_B, noise_G)

Build measurement covariance matrix for selected mode.
"""
function build_mode_covariance(mode::MeasurementMode,
                                noise_B::Float64,
                                noise_G::Float64)
    if mode == MODE_FIELD_ONLY
        return Diagonal(SVector{3}(fill(noise_B^2, 3)...))
    elseif mode == MODE_GRADIENT_ONLY
        return Diagonal(SVector{5}(fill(noise_G^2, 5)...))
    else
        return Diagonal(vcat(
            SVector{3}(fill(noise_B^2, 3)...),
            SVector{5}(fill(noise_G^2, 5)...)
        ))
    end
end

# ============================================================================
# Exports
# ============================================================================

export MeasurementMode, MODE_FIELD_ONLY, MODE_GRADIENT_ONLY, MODE_FULL_TENSOR
export measurement_dimension
export TensorSelectivityConfig, DEFAULT_SELECTIVITY_CONFIG
export estimate_gradient_snr, estimate_snr_at_range, crossover_range
export ModeSelectionResult, select_measurement_mode, select_measurement_mode_with_covariance
export GradientFisherInfo, compute_gradient_fisher_gain, optimal_gradient_range
export AdaptiveModeSelector, update_mode_selector!, get_selector_statistics
export apply_mode_selection, build_mode_covariance
