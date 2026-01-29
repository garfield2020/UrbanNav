# ============================================================================
# elevator_mode_config.jl - A/B/C mode configuration for elevator DOE
# ============================================================================
#
# Configures three navigation modes for elevator interference testing:
# - Mode A (baseline): No dynamic handling
# - Mode B (robust ignore): Chi² gating + covariance inflation
# - Mode C (source-aware): CUSUM detection + moving dipole tracking
# ============================================================================

export ElevatorNavMode, NAV_MODE_A_BASELINE, NAV_MODE_B_ROBUST_IGNORE, NAV_MODE_C_SOURCE_AWARE
export ElevatorModeConfig
export configure_mode_a, configure_mode_b, configure_mode_c, configure_elevator_mode

"""
    ElevatorNavMode

Navigation mode for elevator DOE testing.
"""
@enum ElevatorNavMode begin
    NAV_MODE_A_BASELINE = 1        # No dynamic handling
    NAV_MODE_B_ROBUST_IGNORE = 2   # Robust ignore via gating + inflation
    NAV_MODE_C_SOURCE_AWARE = 3    # Source-aware: detect, track, subtract
end

"""
    ElevatorModeConfig

Configuration parameters for elevator interference handling.

# Fields
- `mode::ElevatorNavMode`: Active mode.
- `chi2_gating_alpha::Float64`: Chi² gating significance level (Mode B).
- `covariance_inflation_factor::Float64`: Inflation factor during innovation spikes (Mode B).
- `innovation_spike_threshold::Float64`: Normalized innovation threshold for spike detection.
- `reject_dynamic_from_learning::Bool`: Reject dynamic residuals from map learning (Mode B).
- `cusum_threshold::Float64`: CUSUM detection threshold (Mode C).
- `cusum_drift::Float64`: CUSUM drift parameter (Mode C).
- `augmented_state_dipole::Bool`: Track elevator as moving dipole in state (Mode C).
- `subtract_predicted_field::Bool`: Subtract predicted source field (Mode C).
- `freeze_tiles_near_source::Bool`: Freeze map tiles near detected source (Mode C).
- `freeze_radius::Float64`: Radius around source to freeze tiles (m) (Mode C).
"""
struct ElevatorModeConfig
    mode::ElevatorNavMode
    chi2_gating_alpha::Float64
    covariance_inflation_factor::Float64
    innovation_spike_threshold::Float64
    reject_dynamic_from_learning::Bool
    cusum_threshold::Float64
    cusum_drift::Float64
    augmented_state_dipole::Bool
    subtract_predicted_field::Bool
    freeze_tiles_near_source::Bool
    freeze_radius::Float64
end

"""
    configure_mode_a() -> ElevatorModeConfig

Mode A: Baseline — no dynamic elevator handling. The system operates as if
there are no moving magnetic sources.
"""
function configure_mode_a()
    ElevatorModeConfig(
        NAV_MODE_A_BASELINE,
        0.05,    # default chi2 gating (not specialized)
        1.0,     # no inflation
        Inf,     # no spike detection
        false,   # don't reject dynamic residuals
        Inf,     # no CUSUM
        0.0,     # no CUSUM drift
        false,   # no augmented state
        false,   # no field subtraction
        false,   # no tile freezing
        0.0,     # no freeze radius
    )
end

"""
    configure_mode_b() -> ElevatorModeConfig

Mode B: Robust Ignore — uses chi² gating (α=0.01), covariance inflation (3×)
when innovation spikes are detected, and rejects dynamic residuals from map
learning. Does not attempt to model the elevator source.
"""
function configure_mode_b()
    ElevatorModeConfig(
        NAV_MODE_B_ROBUST_IGNORE,
        0.01,    # tight chi2 gating
        3.0,     # 3× covariance inflation
        3.0,     # innovation spike at 3σ
        true,    # reject dynamic residuals from learning
        Inf,     # no CUSUM (Mode B doesn't track sources)
        0.0,
        false,   # no augmented state
        false,   # no field subtraction
        false,   # no tile freezing
        0.0,
    )
end

"""
    configure_mode_c() -> ElevatorModeConfig

Mode C: Source-Aware — detects elevator via CUSUM on innovations, tracks it
as a moving dipole in augmented state, subtracts predicted source field from
measurements, and freezes map tiles near the detected source.
"""
function configure_mode_c()
    ElevatorModeConfig(
        NAV_MODE_C_SOURCE_AWARE,
        0.01,    # tight chi2 gating
        3.0,     # inflation as fallback
        3.0,     # spike threshold
        true,    # reject dynamic from learning
        5.0,     # CUSUM threshold
        0.5,     # CUSUM drift
        true,    # augmented state dipole
        true,    # subtract predicted field
        true,    # freeze tiles near source
        8.0,     # freeze radius 8m
    )
end

"""
    configure_elevator_mode(mode::ElevatorNavMode) -> ElevatorModeConfig

Get the configuration for a given elevator navigation mode.
"""
function configure_elevator_mode(mode::ElevatorNavMode)
    if mode == NAV_MODE_A_BASELINE
        return configure_mode_a()
    elseif mode == NAV_MODE_B_ROBUST_IGNORE
        return configure_mode_b()
    elseif mode == NAV_MODE_C_SOURCE_AWARE
        return configure_mode_c()
    else
        error("Unknown elevator navigation mode: $mode")
    end
end
