#
# TetrahedronSensor.jl - Tetrahedral Full Tensor Magnetometer (FTM) Sensor Model
#
# SE Layer: B (SENSOR/FUSION)
#
# ============================================================================
# HARDWARE ARCHITECTURE
# ============================================================================
#
# 17 single-axis Hall bars on a regular tetrahedron frame:
# - 4 Hall bars per face × 4 faces = 16 face sensors
# - 1 center reference sensor = 17 total
#
# CRITICAL: Hall bars are SCALAR sensors (single-axis), not vector sensors!
# Total measurements: 17 scalars, NOT 51 (17×3)
#
# Signal Chain (per channel):
# ┌─────────────┐    ┌──────────────┐    ┌─────────┐    ┌─────────┐
# │ Hall Bar    │───▶│ 250kHz Mod   │───▶│ ADC     │───▶│ Lock-in │
# │ (graphene)  │    │ (1/f reject) │    │ 8kHz    │    │ Demod   │
# └─────────────┘    └──────────────┘    └─────────┘    └─────────┘
#                                              │
#                                              ▼
#                                    ┌─────────────────┐
#                                    │ Every 20th:     │
#                                    │ T-stable R meas │
#                                    │ (drift cal)     │
#                                    └─────────────────┘
#
# Key features:
# - 17 dedicated ADCs (no multiplexing) - parallel acquisition
# - 250 kHz modulation moves signal away from 1/f noise knee
# - Lock-in detection provides excellent noise rejection
# - Interleaved resistance measurement every 20 samples
# - T-stable resistor reference for real-time temperature calibration
# - Per-channel drift compensation from continuous T tracking
#
# Effective rates:
# - ADC: 8000 Hz per channel
# - Field measurements: 8000 × 19/20 = 7600 Hz (after T-cal interleave)
# - After LPF for navigation: typically 1-50 Hz output
#
# ============================================================================
# SENSOR ARRANGEMENT
# ============================================================================
#
# Face sensor arrangement (Wheatstone bridge config):
# - 4 bars per face measure field along face normal
# - Bridge differential extracts in-plane gradients
# - Software-defined Wheatstone bridge for noise-reduced normal component
# - Common-mode rejection of temperature drift
#
# Overdetermination: 17 measurements → 8 unknowns (B₀ + G₅)
# - 9 DOF of redundancy enables fit residual computation
# - Fit residual used for near-field source detection
#
# Maxwell constraints (built into parameterization):
# - Traceless: Gzz = -(Gxx + Gyy)
# - Symmetric: Gxy = Gyx, etc.
# - Only 5 independent gradient components
#
# ============================================================================

using LinearAlgebra
using StaticArrays
using Statistics
using Random
using Printf

# ============================================================================
# Configuration
# ============================================================================

"""
    TetrahedronConfig

Configuration for tetrahedral FTM sensor.

# Hardware Parameters
- `side_length::Float64`: Tetrahedron side length (m), default 0.150 (150mm)
- `n_sensors_per_face::Int`: Hall bars per face, default 4 (Wheatstone bridge)
- `include_center_sensor::Bool`: Include center reference sensor, default true

# Signal Chain
- `modulation_freq::Float64`: Carrier frequency for 1/f rejection (Hz), default 250e3
- `adc_rate::Float64`: Per-channel ADC rate (Hz), default 8000.0
- `tcal_interleave::Int`: T-cal every N samples, default 20
- `lpf_cutoff::Float64`: Output low-pass filter cutoff (Hz), default 50.0

# Noise Model
- `asd_per_sensor::Float64`: Amplitude spectral density per Hall bar (T/√Hz)
- `flux_concentration_factor::Float64`: Flux concentrator gain, default 15.0
- `use_flux_concentrator::Bool`: Enable flux concentrators, default true

# Temperature Calibration
- `t_stable_resistor_ppm::Float64`: T-stable resistor drift (ppm/°C), default 1.0
- `hall_tempco::Float64`: Hall bar temperature coefficient (ppm/°C), default 100.0
"""
Base.@kwdef struct TetrahedronConfig
    # Geometry
    side_length::Float64 = 0.150           # 150mm tetrahedron sides
    n_sensors_per_face::Int = 4            # 4 Hall bars per face (Wheatstone config)
    include_center_sensor::Bool = true     # Center reference sensor

    # Signal chain (actual hardware)
    modulation_freq::Float64 = 250e3       # 250 kHz modulation (1/f rejection)
    adc_rate::Float64 = 8000.0             # 8 kHz per-channel ADC (17 parallel ADCs)
    tcal_interleave::Int = 20              # T-cal every 20th sample
    lpf_cutoff::Float64 = 50.0             # Output LPF cutoff (Hz)

    # Noise (graphene van der Waals Hall bars with flux concentrators)
    # 5 nT/√Hz is WITH flux concentrators (standard configuration)
    # The 250 kHz modulation puts us well above 1/f knee
    asd_per_sensor::Float64 = 5e-9         # 5 nT/√Hz per bar (post-demod)
    flux_concentration_factor::Float64 = 15.0  # 10-20x typical
    use_flux_concentrator::Bool = true

    # Temperature calibration
    # Every 20th sample: measure Hall bar R vs T-stable reference
    t_stable_resistor_ppm::Float64 = 1.0   # ppm/°C (high quality reference)
    hall_tempco::Float64 = 100.0           # Hall bar tempco (ppm/°C)
    # Real-time T compensation removes per-channel drift
end

"""Total number of sensors (17 = 4×4 face + 1 center)."""
function n_sensors(config::TetrahedronConfig)
    n = config.n_sensors_per_face * 4  # 4 faces
    if config.include_center_sensor
        n += 1
    end
    return n
end

"""Number of scalar measurements (= n_sensors for single-axis Hall bars)."""
n_measurements(config::TetrahedronConfig) = n_sensors(config)

"""Single sensor noise density ASD (depends on concentrator)."""
function noise_density(config::TetrahedronConfig)
    if config.use_flux_concentrator
        return config.asd_per_sensor
    else
        # Without concentrator, noise is higher by concentration factor
        return config.asd_per_sensor * config.flux_concentration_factor
    end
end

"""Noise after √N averaging across all sensors (ideal, ignoring common-mode)."""
function effective_noise_density(config::TetrahedronConfig)
    return noise_density(config) / sqrt(n_sensors(config))
end

"""Effective baseline for gradient measurement."""
baseline(config::TetrahedronConfig) = config.side_length

"""Field sample rate after T-cal interleaving."""
function field_sample_rate(config::TetrahedronConfig)
    return config.adc_rate * (config.tcal_interleave - 1) / config.tcal_interleave
end

"""Number of unknowns in tensor reconstruction (B₀=3 + G₅=5 = 8)."""
n_unknowns(::TetrahedronConfig) = 8

"""Degrees of freedom for fit residual (n_measurements - n_unknowns)."""
fit_dof(config::TetrahedronConfig) = n_measurements(config) - n_unknowns(config)

# ============================================================================
# Tetrahedron Sensor Model
# ============================================================================

"""
    HallBar

Single graphene Hall bar sensor.

Graphene is a true 2D material - the Hall bar measures the B-field component
NORMAL to the graphene plane:

    m = B(r) · n̂

where n̂ is the unit normal to the Hall bar surface.

# Fields
- `position`: Location in sensor frame (m)
- `normal`: Unit normal to graphene plane (measurement direction)
- `face_id`: Which tetrahedral face (1-4), or 0 for center sensor
- `bar_id`: Index within face (1-4), or 0 for center
"""
struct HallBar
    position::SVector{3, Float64}    # Position in sensor frame (m)
    normal::SVector{3, Float64}      # Normal to graphene plane (unit vector)
    face_id::Int                     # Which face (1-4) or 0 for center
    bar_id::Int                      # Index within face (1-4) or 0 for center
end

"""
    TetrahedronSensor

17-channel tetrahedral FTM sensor for urban navigation.

Reconstructs field and gradient tensor from 17 scalar Hall bar measurements.
Computes fit residual (9 DOF) to detect near-field sources such as elevators,
HVAC motors, or other urban magnetic anomalies.

# Hardware
- 4 Hall bars per face × 4 faces = 16 face sensors
- 1 center reference sensor = 17 total
- Each bar measures scalar field along its sensitive axis
- 17 parallel ADCs at 8 kHz (no multiplexing)
- 250 kHz modulation for 1/f rejection
- Interleaved T-cal every 20 samples

# Measurement Model
Each graphene Hall bar i gives scalar measurement:
    m_i = B(r_i) · n̂_i = (B₀ + G·Δr_i) · n̂_i

Where:
- B₀: Field at sensor center (3 components)
- G: Gradient tensor (5 independent: Gxx, Gyy, Gxy, Gxz, Gyz; Gzz = -Gxx-Gyy)
- Δr_i: Position relative to center
- n̂_i: Normal to graphene plane of bar i (2D material → measures B⊥)

# Overdetermination
- 17 measurements, 8 unknowns
- 9 degrees of freedom for fit residual
- Elevated residual indicates near-field source (dipole model breaks down)

# Usage
```julia
config = TetrahedronConfig()
sensor = TetrahedronSensor(config)

# Simulate 17 scalar measurements
measurements = simulate_measurement(sensor, field_func; add_noise=true)

# Reconstruct tensor (least squares)
recon = reconstruct_tensor(sensor, measurements)

# Check fit residual for near-field detection
if recon.fit_residual_normalized > 3.0
    # Potential near-field source (elevator, HVAC motor, etc.)
end
```
"""
struct TetrahedronSensor
    config::TetrahedronConfig
    hall_bars::Vector{HallBar}                 # 17 Hall bars with positions & axes
    center::SVector{3, Float64}                # Geometric center
    design_matrix::Matrix{Float64}             # A: 17×8 matrix for reconstruction
    design_pinv::Matrix{Float64}               # A⁺: 8×17 pseudo-inverse
    face_normals::Vector{SVector{3, Float64}}  # 4 face normal vectors
    rng::MersenneTwister
end

"""
    TetrahedronSensor(config::TetrahedronConfig = TetrahedronConfig(); seed::Int = 42)

Create a tetrahedral FTM sensor with the given configuration.
"""
function TetrahedronSensor(config::TetrahedronConfig = TetrahedronConfig(); seed::Int = 42)
    hall_bars, face_normals = build_hall_bars(config)
    center = SVector{3}(0.0, 0.0, 0.0)  # Tetrahedron centered at origin
    A, A_pinv = build_design_matrix_scalar(hall_bars, center)

    TetrahedronSensor(config, hall_bars, center, A, A_pinv, face_normals, MersenneTwister(seed))
end

"""
    build_hall_bars(config::TetrahedronConfig)

Build Hall bar array for tetrahedral sensor.

# Arrangement
- 4 graphene Hall bars per face × 4 faces = 16 face sensors
- 1 center reference sensor (oriented along +Z)
- Total: 17 Hall bars

# Face Sensors
Each face has 4 Hall bars arranged in a square pattern for Wheatstone bridge operation.
The graphene planes are parallel to the tetrahedral face, so each bar measures
the B-field component normal to that face.

# Center Sensor
The center Hall bar is oriented with normal along +Z (arbitrary choice, could be
any direction since it's primarily for common-mode drift reference).

# Returns
- `hall_bars::Vector{HallBar}`: 17 Hall bar definitions
- `face_normals::Vector{SVector{3,Float64}}`: 4 face normal vectors
"""
function build_hall_bars(config::TetrahedronConfig)
    s = config.side_length

    # Regular tetrahedron vertices (centered at origin)
    # Using standard orientation with one vertex pointing up
    a = s / sqrt(2)
    vertices = [
        SVector{3}(a, 0.0, -a/sqrt(2)),
        SVector{3}(-a, 0.0, -a/sqrt(2)),
        SVector{3}(0.0, a, a/sqrt(2)),
        SVector{3}(0.0, -a, a/sqrt(2))
    ]

    # Center tetrahedron at origin
    v_center = mean(vertices)
    vertices = [v - v_center for v in vertices]

    # Four faces (vertex indices, ordered for outward normals)
    face_vertex_indices = [
        (1, 2, 3),
        (1, 4, 2),
        (1, 3, 4),
        (2, 4, 3)
    ]

    hall_bars = HallBar[]
    face_normals = SVector{3, Float64}[]
    bar_spacing = s * 0.15  # Spacing within Wheatstone bridge pattern

    for (face_id, (i1, i2, i3)) in enumerate(face_vertex_indices)
        v0, v1, v2 = vertices[i1], vertices[i2], vertices[i3]
        face_center = (v0 + v1 + v2) / 3

        # Face normal (outward pointing)
        # For graphene Hall bars lying in the face plane, this is the measurement direction
        normal = normalize(cross(v1 - v0, v2 - v0))
        push!(face_normals, normal)

        # Build orthogonal basis vectors in the face plane
        # (for positioning the 4 bars in a square pattern)
        if abs(normal[3]) < 0.9
            u = normalize(cross(normal, SVector{3}(0.0, 0.0, 1.0)))
        else
            u = normalize(cross(normal, SVector{3}(1.0, 0.0, 0.0)))
        end
        v = cross(normal, u)

        # Place 4 Hall bars per face in Wheatstone bridge configuration
        # Square pattern enables extraction of in-plane gradients
        #
        #   [2]----[4]
        #    |      |
        #   [1]----[3]
        #
        bar_positions = [
            (-0.5, -0.5),  # bar 1: bottom-left
            (-0.5,  0.5),  # bar 2: top-left
            ( 0.5, -0.5),  # bar 3: bottom-right
            ( 0.5,  0.5),  # bar 4: top-right
        ]

        for (bar_id, (du, dv)) in enumerate(bar_positions)
            pos = face_center + bar_spacing * (du * u + dv * v)
            # Graphene plane is parallel to face → normal to graphene = face normal
            push!(hall_bars, HallBar(pos, normal, face_id, bar_id))
        end
    end

    # Center reference sensor
    # Oriented with normal along +Z (measures Bz at center)
    if config.include_center_sensor
        center_normal = SVector{3}(0.0, 0.0, 1.0)
        push!(hall_bars, HallBar(SVector{3}(0.0, 0.0, 0.0), center_normal, 0, 0))
    end

    return hall_bars, face_normals
end

"""
    build_design_matrix_scalar(hall_bars, center)

Build design matrix for SCALAR Hall bar measurements.

# Measurement Model
Each graphene Hall bar measures the B-field normal to its plane:
    m_i = B(r_i) · n̂_i

where B(r) = B₀ + G·Δr (linear gradient model).

# State Vector (8 components)
    x = [Bx₀, By₀, Bz₀, Gxx, Gyy, Gxy, Gxz, Gyz]

Note: Gzz = -(Gxx + Gyy) enforced by Maxwell ∇·B = 0

# Design Matrix
    A: 17×8 matrix where A[i,:] maps state to measurement m_i
    m = A·x

# Returns
    (A, A_pinv) where A_pinv = (AᵀA)⁻¹Aᵀ is the least-squares pseudo-inverse
"""
function build_design_matrix_scalar(hall_bars::Vector{HallBar}, center::SVector{3, Float64})
    n = length(hall_bars)
    A = zeros(n, 8)

    for (i, hb) in enumerate(hall_bars)
        # Position relative to center
        Δr = hb.position - center
        Δx, Δy, Δz = Δr

        # Hall bar normal (measurement direction)
        n̂ = hb.normal
        nx, ny, nz = n̂

        # Bx₀ coefficient
        A[i, 1] = nx

        # By₀ coefficient
        A[i, 2] = ny

        # Bz₀ coefficient
        A[i, 3] = nz

        # Gxx coefficient: from Bx (nx·Δx) and Bz (-nz·Δz due to traceless)
        A[i, 4] = nx * Δx - nz * Δz

        # Gyy coefficient: from By (ny·Δy) and Bz (-nz·Δz due to traceless)
        A[i, 5] = ny * Δy - nz * Δz

        # Gxy coefficient: from Bx (nx·Δy) and By (ny·Δx)
        A[i, 6] = nx * Δy + ny * Δx

        # Gxz coefficient: from Bx (nx·Δz) and Bz (nz·Δx)
        A[i, 7] = nx * Δz + nz * Δx

        # Gyz coefficient: from By (ny·Δz) and Bz (nz·Δy)
        A[i, 8] = ny * Δz + nz * Δy
    end

    # Pseudo-inverse for least-squares solution
    A_pinv = pinv(A)

    return A, A_pinv
end

# ============================================================================
# Measurement Simulation
# ============================================================================

"""
    simulate_measurement(sensor, field_func; add_noise=true, bandwidth_hz=1.0)

Simulate 17 scalar Hall bar measurements.

# Arguments
- `sensor::TetrahedronSensor`: The sensor model
- `field_func`: Function (position::SVector{3}) -> B field vector (3-element)
- `add_noise::Bool`: Add sensor noise, default true
- `bandwidth_hz::Float64`: Effective measurement bandwidth (Hz), default 1.0

# Returns
- `measurements::Vector{Float64}`: 17 scalar measurements (one per Hall bar)

# Noise Model
Noise standard deviation per measurement:
    σ = ASD × √bandwidth
"""
function simulate_measurement(
    sensor::TetrahedronSensor,
    field_func::Function;
    add_noise::Bool = true,
    bandwidth_hz::Float64 = 1.0
)
    n = length(sensor.hall_bars)
    measurements = zeros(n)

    # Noise standard deviation: σ = ASD × √BW
    noise_std = noise_density(sensor.config) * sqrt(bandwidth_hz)

    for (i, hb) in enumerate(sensor.hall_bars)
        # Get B-field vector at Hall bar position
        B = field_func(hb.position)

        # Scalar measurement: projection onto Hall bar normal
        m = dot(B, hb.normal)

        # Add noise if requested
        if add_noise
            m += noise_std * randn(sensor.rng)
        end

        measurements[i] = m
    end

    return measurements
end

# ============================================================================
# Tensor Reconstruction
# ============================================================================

"""
    TensorReconstruction

Result of tensor reconstruction from 17 scalar Hall bar measurements.

# Fields
- `B0::SVector{3, Float64}`: Field at sensor center [Bx, By, Bz]
- `G5::SVector{5, Float64}`: Independent gradient components [Gxx, Gyy, Gxy, Gxz, Gyz]
- `G_full::SMatrix{3,3,Float64,9}`: Full 3×3 symmetric gradient tensor
- `fit_residual::Float64`: RMS residual from least-squares fit
- `fit_residual_normalized::Float64`: Residual / expected noise (χ² statistic)
- `n_measurements::Int`: Number of measurements used (17)
- `dof::Int`: Degrees of freedom for residual (17 - 8 = 9)

# Residual Interpretation
- **Low residual** (normalized < 3): Measurements consistent with uniform gradient field
- **High residual** (normalized > 3): Near-field source likely (elevator, HVAC, etc.)
"""
struct TensorReconstruction
    B0::SVector{3, Float64}
    G5::SVector{5, Float64}              # [Gxx, Gyy, Gxy, Gxz, Gyz]
    G_full::SMatrix{3, 3, Float64, 9}    # Full 3×3 gradient matrix
    fit_residual::Float64                # RMS residual
    fit_residual_normalized::Float64     # Residual / σ (chi-square like)
    n_measurements::Int
    dof::Int                             # Degrees of freedom (n - 8)
end

"""
    reconstruct_tensor(sensor, measurements; σ_meas=nothing)

Reconstruct field and gradient tensor from 17 scalar Hall bar measurements.

# Arguments
- `sensor::TetrahedronSensor`: The sensor model
- `measurements::Vector{Float64}`: 17 scalar measurements
- `σ_meas::Float64`: Measurement noise std (optional, for normalized residual)

# Returns
- `TensorReconstruction`: Contains B0, G5, G_full, residuals, and statistics
"""
function reconstruct_tensor(sensor::TetrahedronSensor, measurements::Vector{Float64};
                           σ_meas::Union{Float64, Nothing} = nothing)
    n = length(measurements)
    @assert n == length(sensor.hall_bars) "Expected $(length(sensor.hall_bars)) measurements, got $n"

    # Least-squares solve: x = A⁺ · m
    x = sensor.design_pinv * measurements

    # Extract field components
    B0 = SVector{3}(x[1], x[2], x[3])

    # Extract gradient components (5 independent)
    Gxx, Gyy, Gxy, Gxz, Gyz = x[4], x[5], x[6], x[7], x[8]

    # Compute Gzz from traceless constraint (Maxwell: ∇·B = 0)
    Gzz = -Gxx - Gyy

    G5 = SVector{5}(Gxx, Gyy, Gxy, Gxz, Gyz)

    # Build full symmetric gradient tensor
    G_full = SMatrix{3,3}(
        Gxx, Gxy, Gxz,
        Gxy, Gyy, Gyz,
        Gxz, Gyz, Gzz
    )

    # Compute fit residual
    m_predicted = sensor.design_matrix * x
    residual_vec = measurements - m_predicted
    fit_residual = norm(residual_vec)  # RMS residual

    # Normalized residual (if noise known)
    dof = n - 8  # 9 degrees of freedom
    fit_residual_normalized = if σ_meas !== nothing && σ_meas > 0
        fit_residual / (σ_meas * sqrt(dof))
    else
        NaN
    end

    return TensorReconstruction(B0, G5, G_full, fit_residual, fit_residual_normalized, n, dof)
end

"""
    tensor_magnitude(recon::TensorReconstruction)

Compute gradient tensor magnitude |G| = sqrt(sum(Gij²)).
"""
function tensor_magnitude(recon::TensorReconstruction)
    G = recon.G_full
    return sqrt(sum(G .^ 2))
end

# ============================================================================
# Wheatstone Bridge Extraction
# ============================================================================

"""
    WheatstoneExtraction

Result of Wheatstone bridge processing for one tetrahedral face.

# Fields
- `face_id::Int`: Which face (1-4)
- `B_normal::Float64`: Average normal field (from software-defined bridge)
- `dB_du::Float64`: In-plane gradient along u direction
- `dB_dv::Float64`: In-plane gradient along v direction
- `common_mode::Float64`: Common-mode signal (for drift monitoring)
"""
struct WheatstoneExtraction
    face_id::Int
    B_normal::Float64      # Average of 4 bars (noise-reduced normal)
    dB_du::Float64         # Gradient in u direction
    dB_dv::Float64         # Gradient in v direction
    common_mode::Float64   # Common-mode (for diagnostics)
end

"""
    extract_wheatstone(sensor, measurements, face_id)

Extract Wheatstone bridge outputs for one face.

# Wheatstone Bridge Arrangement
```
    [2]----[4]
     |      |
    [1]----[3]
```

# Outputs
- **Average (B_normal)**: (m1 + m2 + m3 + m4) / 4
- **U-gradient (dB_du)**: ((m3 + m4) - (m1 + m2)) / (2 × spacing)
- **V-gradient (dB_dv)**: ((m2 + m4) - (m1 + m3)) / (2 × spacing)
- **Common-mode**: (m1 - m2 + m3 - m4) / 4
"""
function extract_wheatstone(sensor::TetrahedronSensor, measurements::Vector{Float64}, face_id::Int)
    # Get measurement indices for this face
    bar_indices = [findfirst(hb -> hb.face_id == face_id && hb.bar_id == bid, sensor.hall_bars)
                   for bid in 1:4]

    m1, m2, m3, m4 = measurements[bar_indices]

    # Wheatstone bridge spacing
    spacing = sensor.config.side_length * 0.15

    # Extract outputs
    B_normal = (m1 + m2 + m3 + m4) / 4
    dB_du = ((m3 + m4) - (m1 + m2)) / (2 * spacing)
    dB_dv = ((m2 + m4) - (m1 + m3)) / (2 * spacing)
    common_mode = (m1 - m2 + m3 - m4) / 4

    return WheatstoneExtraction(face_id, B_normal, dB_du, dB_dv, common_mode)
end

"""
    extract_all_wheatstone(sensor, measurements)

Extract Wheatstone bridge outputs for all 4 faces.
"""
function extract_all_wheatstone(sensor::TetrahedronSensor, measurements::Vector{Float64})
    return [extract_wheatstone(sensor, measurements, face_id) for face_id in 1:4]
end

# ============================================================================
# Noise Budget
# ============================================================================

"""
    NoiseBudget

Physics-rigorous noise budget for the tetrahedral FTM sensor.

Follows the correct signal processing chain:
1. Start with per-sensor ASD: S_B [T/√Hz]
2. Convert to RMS using effective bandwidth: σ = S_B × √BW
3. Apply spatial averaging with proper covariance (including common-mode)
"""
struct NoiseBudget
    asd_per_sensor::Float64
    bandwidth_hz::Float64
    publish_rate_hz::Float64
    n_sensors::Int
    common_mode_asd::Float64
    baseline::Float64

    rms_per_sensor::Float64
    rms_common_mode::Float64
    rms_combined::Float64
    rms_gradient::Float64
    spatial_improvement::Float64
    measurements_correlated::Bool
end

"""
    compute_noise_budget(config; bandwidth_hz, publish_rate_hz, common_mode_asd)

Compute physics-rigorous noise budget.

The correct conversion from ASD to RMS is:
    σ_rms = ASD × √BW
"""
function compute_noise_budget(
    config::TetrahedronConfig;
    bandwidth_hz::Float64 = 1.0,
    publish_rate_hz::Float64 = 10.0,
    common_mode_asd::Float64 = 2e-9
)
    n = n_sensors(config)
    S_B = noise_density(config)
    bl = baseline(config)

    σ_sensor = S_B * sqrt(bandwidth_hz)
    σ_cm = common_mode_asd * sqrt(bandwidth_hz)

    σ_combined = sqrt(σ_sensor^2 / n + σ_cm^2)
    spatial_improvement = σ_sensor / σ_combined
    σ_gradient = σ_combined / bl

    measurements_correlated = publish_rate_hz > 2 * bandwidth_hz

    NoiseBudget(
        S_B, bandwidth_hz, publish_rate_hz, n, common_mode_asd, bl,
        σ_sensor, σ_cm, σ_combined, σ_gradient, spatial_improvement,
        measurements_correlated
    )
end

"""
    sensor_noise_floor(config; bandwidth_hz=1.0, common_mode_asd=2e-9)

Get the effective measurement noise floor (RMS per measurement).

For default config (17 sensors, BW=1 Hz, 2 nT/√Hz common-mode):
Returns ~2.3 nT per measurement
"""
function sensor_noise_floor(
    config::TetrahedronConfig;
    bandwidth_hz::Float64 = 1.0,
    common_mode_asd::Float64 = 2e-9
)
    budget = compute_noise_budget(config;
        bandwidth_hz=bandwidth_hz,
        common_mode_asd=common_mode_asd
    )
    return budget.rms_combined
end

function sensor_noise_floor(
    sensor::TetrahedronSensor;
    bandwidth_hz::Float64 = 1.0,
    common_mode_asd::Float64 = 2e-9
)
    return sensor_noise_floor(sensor.config;
        bandwidth_hz=bandwidth_hz,
        common_mode_asd=common_mode_asd
    )
end

# ============================================================================
# Integration with Estimator
# ============================================================================

"""
    MeasurementD8

Single d=8 measurement reconstructed from tetrahedron sensor.

# Fields
- `B::SVector{3, Float64}`: Field at sensor center [Bx, By, Bz]
- `G::SVector{5, Float64}`: Gradient tensor [Gxx, Gyy, Gxy, Gxz, Gyz]
- `fit_residual::Float64`: Least-squares fit residual (for near-field detection)
- `fit_residual_normalized::Float64`: Residual normalized by noise (χ² statistic)
- `timestamp::Float64`: Measurement timestamp
"""
struct MeasurementD8
    B::SVector{3, Float64}
    G::SVector{5, Float64}
    fit_residual::Float64
    fit_residual_normalized::Float64
    timestamp::Float64
end

"""
    process_hall_bars(sensor, raw_measurements; σ_meas=nothing, timestamp=0.0)

Convert 17 scalar Hall bar measurements to d=8 format (B + G).

This is the primary interface between the tetrahedron sensor and the estimator.
"""
function process_hall_bars(sensor::TetrahedronSensor, raw_measurements::Vector{Float64};
                          σ_meas::Union{Float64, Nothing} = nothing,
                          timestamp::Float64 = 0.0)
    recon = reconstruct_tensor(sensor, raw_measurements; σ_meas=σ_meas)

    return MeasurementD8(
        recon.B0,
        recon.G5,
        recon.fit_residual,
        recon.fit_residual_normalized,
        timestamp
    )
end

# ============================================================================
# Urban Noise Environment
# ============================================================================

"""
    UrbanNoiseEnvironment

Per-source urban electromagnetic noise model. Each source is individually
modeled and parameterized. The tetrahedron's 250 kHz modulation rejects
low-frequency drift, but DC and quasi-DC fields from moving/switching
sources pass through the signal chain.

All field values are in nT. RSS of all sources gives the effective
environmental noise floor.
"""
Base.@kwdef struct UrbanNoiseEnvironment
    # --- AC sources (50/60 Hz) ---
    # Largely rejected by 250 kHz mod + lock-in + 50 Hz LPF,
    # but aliased residual leaks through at ~1-5% of raw amplitude.
    hvac_field_nT::Float64 = 50.0       # HVAC motor at 3-5m, raw ~1000 nT,
                                         # after lock-in rejection: ~50 nT residual
    lighting_field_nT::Float64 = 20.0    # Fluorescent ballasts at 2-3m,
                                         # raw ~200-500 nT, residual ~20 nT
    power_wiring_nT::Float64 = 30.0      # Building power distribution at 2-5m,
                                         # raw ~500 nT, residual ~30 nT

    # --- DC / quasi-DC sources (pass through signal chain) ---
    electronic_devices_nT::Float64 = 10.0  # Phones, laptops, security gates at 1-2m.
                                            # Mostly DC from permanent magnets (speakers).
    vehicle_dc_nT::Float64 = 100.0         # Parked cars at 2-5m in garages.
                                            # DC from magnetized steel body.

    # --- Broadband / impulsive ---
    switching_transients_nT::Float64 = 15.0  # Elevator relay switching, light
                                              # switches, breakers. Brief pulses.
    pedestrian_devices_nT::Float64 = 5.0     # Phone in pocket, keys, belt buckle.
                                              # Moves with pedestrian (self-noise).
end

"""
    environmental_noise_std(env::UrbanNoiseEnvironment)

Compute RSS of all environmental noise sources. Returns σ in Tesla.

Default RSS: √(50² + 20² + 30² + 10² + 100² + 15² + 5²) ≈ 122 nT = 0.122 µT
"""
function environmental_noise_std(env::UrbanNoiseEnvironment)
    σ_nT = sqrt(
        env.hvac_field_nT^2 +
        env.lighting_field_nT^2 +
        env.power_wiring_nT^2 +
        env.electronic_devices_nT^2 +
        env.vehicle_dc_nT^2 +
        env.switching_transients_nT^2 +
        env.pedestrian_devices_nT^2
    )
    return σ_nT * 1e-9  # Convert to T
end

# ============================================================================
# Testing
# ============================================================================

function test_tetrahedron()
    println("=" ^ 70)
    println("TETRAHEDRAL FTM SENSOR TEST")
    println("17 Scalar Graphene Hall Bars (4/face × 4 + 1 center)")
    println("=" ^ 70)

    # Create sensor with default config
    config = TetrahedronConfig()
    sensor = TetrahedronSensor(config)

    println("\n1. HARDWARE CONFIGURATION:")
    println("   Side length:     $(config.side_length * 1000) mm")
    println("   Total sensors:   $(n_sensors(config))")
    println("   ASD per bar:     $(noise_density(config) * 1e9) nT/√Hz")

    # Test with a dipole source
    println("\n2. TENSOR RECONSTRUCTION TEST:")
    μ0 = 4π * 1e-7
    source_pos = SVector{3}(0.0, 0.0, -3.0)
    moment_vec = SVector{3}(0.0, 0.0, 50.0)

    function dipole_field(pos)
        r = pos - source_pos
        r_mag = norm(r)
        r_hat = r / r_mag
        B = (μ0 / (4π)) * (3 * dot(moment_vec, r_hat) * r_hat - moment_vec) / r_mag^3
        return B
    end

    B_true = dipole_field(SVector{3}(0.0, 0.0, 0.0))

    # Exact reconstruction (no noise)
    measurements = simulate_measurement(sensor, dipole_field; add_noise=false)
    recon = reconstruct_tensor(sensor, measurements)

    B_err = norm(recon.B0 - B_true)
    @printf("   B₀ error (no noise): %.2e nT\n", B_err * 1e9)

    # Noisy reconstruction
    σ_meas = noise_density(config) * sqrt(1.0)
    measurements_noisy = simulate_measurement(sensor, dipole_field; add_noise=true, bandwidth_hz=1.0)
    recon_noisy = reconstruct_tensor(sensor, measurements_noisy; σ_meas=σ_meas)
    @printf("   Normalized residual: %.2f (expect ~1 for good fit)\n", recon_noisy.fit_residual_normalized)

    # Urban noise environment
    println("\n3. URBAN NOISE ENVIRONMENT:")
    env = UrbanNoiseEnvironment()
    σ_env = environmental_noise_std(env)
    σ_sensor = sensor_noise_floor(config; bandwidth_hz=10.0)
    σ_total = sqrt(σ_sensor^2 + σ_env^2)
    @printf("   Sensor noise (10 Hz BW): %.1f nT\n", σ_sensor * 1e9)
    @printf("   Environmental noise:     %.1f nT\n", σ_env * 1e9)
    @printf("   Total noise:             %.1f nT = %.3f µT\n", σ_total * 1e9, σ_total * 1e6)

    println("\n" * "=" ^ 70)
    println("Tetrahedron sensor tests complete!")
    println("=" ^ 70)

    return true
end

# Run if executed directly
if abspath(PROGRAM_FILE) == @__FILE__
    using Printf
    test_tetrahedron()
end
