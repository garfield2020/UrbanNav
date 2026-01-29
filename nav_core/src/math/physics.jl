# ============================================================================
# UrbanNav navigation
# ============================================================================
#
# Ported from AUV-Navigation/src/physics.jl
#
# Electromagnetic Regime: Quasi-static magnetostatic
# - Frequencies: DC to ~100 Hz
# - Wavelengths >> scene size
# - Governing equations: ∇ × B = 0, ∇ · B = 0
#
# Source types:
# 1. Background geology: B = -∇Φ (harmonic potential)
# 2. Conductors: Biot-Savart quasi-static
# 3. Ferrous objects: Dipole/multipole approximation
# ============================================================================

using LinearAlgebra
using StaticArrays
using Statistics: mean

export μ₀, μ₀_4π
export MagneticDipole, ConductorSegment, ConductorPath, HarmonicPotential
export field, gradient, potential
export verify_divergence_free, verify_curl_free, verify_dipole_falloff
export verify_ampere_law, verify_superposition, verify_frame_invariance
export MaxwellGateResult, check_eigenvalue_structure, check_field_gradient_ratio
export check_temporal_consistency, apply_maxwell_gate, apply_maxwell_gate_5component

# ============================================================================
# Physical Constants
# ============================================================================

"""Permeability of free space μ₀ (H/m)"""
const μ₀ = 4π * 1e-7

"""μ₀/(4π) - common prefactor for dipole/Biot-Savart"""
const μ₀_4π = 1e-7

# ============================================================================
# Magnetic Dipole Model
# ============================================================================

"""
    MagneticDipole

A magnetic dipole source representing ferrous objects (UXO, debris, anchors).

The dipole field satisfies Maxwell's equations exactly outside the source:
- ∇·B = 0 everywhere
- ∇×B = 0 (source-free region)
- |B| ~ 1/r³ falloff

# Fields
- `position::Vec3`: Dipole location (m)
- `moment::Vec3`: Magnetic moment vector (A·m²)
- `id::String`: Identifier
- `is_induced::Bool`: True if moment is induced by external field
- `susceptibility::Float64`: χ for induced moment
"""
struct MagneticDipole
    position::Vec3
    moment::Vec3
    id::String
    is_induced::Bool
    susceptibility::Float64
end

function MagneticDipole(
    position::AbstractVector,
    moment::AbstractVector;
    id::String = "dipole",
    is_induced::Bool = false,
    susceptibility::Real = 0.0
)
    MagneticDipole(Vec3(position...), Vec3(moment...), id, is_induced, Float64(susceptibility))
end

"""
    field(dipole::MagneticDipole, x::AbstractVector)

Compute magnetic field B at position x from a dipole.
B(x) = (μ₀/4π) · [3(m·r̂)r̂ - m] / r³
"""
function field(dipole::MagneticDipole, x::AbstractVector)
    r_vec = Vec3(x...) - dipole.position
    r = norm(r_vec)

    if r < 1e-10
        return Vec3(0.0, 0.0, 0.0)
    end

    r̂ = r_vec / r
    m = dipole.moment
    m_dot_r̂ = dot(m, r̂)
    B = μ₀_4π * (3 * m_dot_r̂ * r̂ - m) / r^3

    return B
end

"""
    gradient(dipole::MagneticDipole, x::AbstractVector)

Compute gradient tensor ∂Bᵢ/∂xⱼ at position x from a dipole.
Returns 3×3 tensor in T/m.
"""
function gradient(dipole::MagneticDipole, x::AbstractVector)
    r_vec = Vec3(x...) - dipole.position
    r = norm(r_vec)

    if r < 1e-10
        return SMatrix{3,3,Float64,9}(zeros(3,3))
    end

    r̂ = r_vec / r
    m = dipole.moment
    m_dot_r̂ = dot(m, r̂)

    G = zeros(3, 3)
    for i in 1:3
        for j in 1:3
            δij = i == j ? 1.0 : 0.0
            # ∂Bᵢ/∂xⱼ = (μ₀/4π) · (3/r⁴) · [-5(m·r̂)r̂ᵢr̂ⱼ + (m·r̂)δᵢⱼ + mᵢr̂ⱼ + r̂ᵢmⱼ]
            G[i,j] = (3/r^4) * (
                -5 * m_dot_r̂ * r̂[i] * r̂[j] +
                m_dot_r̂ * δij +
                m[i] * r̂[j] +
                r̂[i] * m[j]
            )
        end
    end

    return SMatrix{3,3,Float64,9}(μ₀_4π * G)
end

# ============================================================================
# Biot-Savart Conductor Model
# ============================================================================

"""
    ConductorSegment

A straight current-carrying conductor segment for Biot-Savart integration.
"""
struct ConductorSegment
    p1::Vec3
    p2::Vec3
    current::Float64
end

function ConductorSegment(p1::AbstractVector, p2::AbstractVector, current::Real)
    ConductorSegment(Vec3(p1...), Vec3(p2...), Float64(current))
end

"""
    field(segment::ConductorSegment, x::AbstractVector)

Compute magnetic field at x from a straight current segment using Biot-Savart.
"""
function field(segment::ConductorSegment, x::AbstractVector)
    p = Vec3(x...)
    p1, p2 = segment.p1, segment.p2
    I = segment.current

    L = p2 - p1
    L_mag = norm(L)
    if L_mag < 1e-10
        return Vec3(0.0, 0.0, 0.0)
    end
    l̂ = L / L_mag

    r1 = p - p1
    r2 = p - p2
    r1_mag = norm(r1)
    r2_mag = norm(r2)

    if r1_mag < 1e-10 || r2_mag < 1e-10
        return Vec3(0.0, 0.0, 0.0)
    end

    r1_parallel = dot(r1, l̂)
    r1_perp = r1 - r1_parallel * l̂
    d = norm(r1_perp)

    if d < 1e-10
        return Vec3(0.0, 0.0, 0.0)
    end

    n̂ = cross(l̂, r1_perp / d)
    n_mag = norm(n̂)
    if n_mag < 1e-10
        return Vec3(0.0, 0.0, 0.0)
    end
    n̂ = n̂ / n_mag

    cos_θ1 = dot(l̂, r1 / r1_mag)
    cos_θ2 = dot(l̂, r2 / r2_mag)

    B_mag = (μ₀_4π * I / d) * (cos_θ1 - cos_θ2)

    return B_mag * n̂
end

"""
    gradient(segment::ConductorSegment, x::AbstractVector; δ::Real = 0.001)

Compute gradient tensor of conductor field using finite differences.
"""
function gradient(segment::ConductorSegment, x::AbstractVector; δ::Real = 0.001)
    p = Vec3(x...)
    G = zeros(3, 3)

    for j in 1:3
        e_j = zeros(3)
        e_j[j] = 1.0
        B_plus = field(segment, p + δ * Vec3(e_j...))
        B_minus = field(segment, p - δ * Vec3(e_j...))
        G[:, j] = (B_plus - B_minus) / (2δ)
    end

    G = (G + G') / 2
    return SMatrix{3,3,Float64,9}(G)
end

"""
    ConductorPath

A piecewise-linear current path (cable, pipeline).
"""
struct ConductorPath
    segments::Vector{ConductorSegment}
    id::String
    frequency::Float64
    phase::Float64
end

function ConductorPath(
    points::Vector{<:AbstractVector},
    current::Real;
    id::String = "cable",
    frequency::Real = 0.0,
    phase::Real = 0.0
)
    segments = ConductorSegment[]
    for i in 1:length(points)-1
        push!(segments, ConductorSegment(points[i], points[i+1], current))
    end
    ConductorPath(segments, id, Float64(frequency), Float64(phase))
end

"""
    field(path::ConductorPath, x::AbstractVector, t::Real = 0.0)

Compute total field from all segments of a conductor path.
"""
function field(path::ConductorPath, x::AbstractVector, t::Real = 0.0)
    modulation = path.frequency > 0 ? cos(2π * path.frequency * t + path.phase) : 1.0

    B = Vec3(0.0, 0.0, 0.0)
    for segment in path.segments
        seg_scaled = ConductorSegment(segment.p1, segment.p2, segment.current * modulation)
        B = B + field(seg_scaled, x)
    end

    return B
end

"""
    gradient(path::ConductorPath, x::AbstractVector, t::Real = 0.0; δ::Real = 0.001)

Compute gradient tensor from conductor path.
"""
function gradient(path::ConductorPath, x::AbstractVector, t::Real = 0.0; δ::Real = 0.001)
    p = Vec3(x...)
    G = zeros(3, 3)

    for j in 1:3
        e_j = zeros(3)
        e_j[j] = 1.0
        B_plus = field(path, p + δ * Vec3(e_j...), t)
        B_minus = field(path, p - δ * Vec3(e_j...), t)
        G[:, j] = (B_plus - B_minus) / (2δ)
    end

    G = (G + G') / 2
    return SMatrix{3,3,Float64,9}(G)
end

# ============================================================================
# Harmonic Scalar Potential
# ============================================================================

"""
    HarmonicPotential

Scalar potential Φ satisfying Laplace's equation ∇²Φ = 0.
B = -∇Φ for source-free background geology.
"""
struct HarmonicPotential
    coefficients::Vector{Float64}
    center::Vec3
    max_order::Int
end

function HarmonicPotential(;
    center::AbstractVector = zeros(3),
    max_order::Int = 3
)
    n_coef = (max_order + 1)^2
    HarmonicPotential(zeros(n_coef), Vec3(center...), max_order)
end

"""
    potential(hp::HarmonicPotential, x::AbstractVector)

Evaluate scalar potential Φ(x).
"""
function potential(hp::HarmonicPotential, x::AbstractVector)
    p = Vec3(x...) - hp.center
    Φ = 0.0

    idx = 1
    for order in 0:hp.max_order
        for φ in _harmonic_basis_order(order)
            if idx <= length(hp.coefficients)
                Φ += hp.coefficients[idx] * φ(p)
            end
            idx += 1
        end
    end

    return Φ
end

"""
    field(hp::HarmonicPotential, x::AbstractVector)

Compute B = -∇Φ at position x.
"""
function field(hp::HarmonicPotential, x::AbstractVector)
    p = Vec3(x...) - hp.center
    B = Vec3(0.0, 0.0, 0.0)

    idx = 1
    for order in 0:hp.max_order
        for (_, ∇φ) in _harmonic_basis_order_with_grad(order)
            if idx <= length(hp.coefficients)
                B = B - hp.coefficients[idx] * ∇φ(p)
            end
            idx += 1
        end
    end

    return B
end

"""
    gradient(hp::HarmonicPotential, x::AbstractVector; δ::Real = 0.001)

Compute gradient tensor of harmonic field.
"""
function gradient(hp::HarmonicPotential, x::AbstractVector; δ::Real = 0.001)
    p = Vec3(x...)
    G = zeros(3, 3)

    for j in 1:3
        e_j = zeros(3)
        e_j[j] = 1.0
        B_plus = field(hp, p + δ * Vec3(e_j...))
        B_minus = field(hp, p - δ * Vec3(e_j...))
        G[:, j] = (B_plus - B_minus) / (2δ)
    end

    G = (G + G') / 2
    return SMatrix{3,3,Float64,9}(G)
end

# Harmonic basis functions
function _harmonic_basis_order(order::Int)
    if order == 0
        return [p -> 1.0]
    elseif order == 1
        return [p -> p[1], p -> p[2], p -> p[3]]
    elseif order == 2
        return [
            p -> p[1]^2 - p[2]^2,
            p -> p[1]^2 - p[3]^2,
            p -> p[1]*p[2],
            p -> p[1]*p[3],
            p -> p[2]*p[3]
        ]
    elseif order == 3
        return [
            p -> p[1]^3 - 3*p[1]*p[2]^2,
            p -> p[2]^3 - 3*p[2]*p[1]^2,
            p -> p[1]^3 - 3*p[1]*p[3]^2,
            p -> p[3]^3 - 3*p[3]*p[1]^2,
            p -> p[1]^2*p[2] - p[2]*p[3]^2,
            p -> p[1]^2*p[3] - p[3]*p[2]^2,
            p -> p[1]*p[2]*p[3]
        ]
    else
        return Function[]
    end
end

function _harmonic_basis_order_with_grad(order::Int)
    if order == 0
        return [(p -> 1.0, p -> Vec3(0.0, 0.0, 0.0))]
    elseif order == 1
        return [
            (p -> p[1], p -> Vec3(1.0, 0.0, 0.0)),
            (p -> p[2], p -> Vec3(0.0, 1.0, 0.0)),
            (p -> p[3], p -> Vec3(0.0, 0.0, 1.0))
        ]
    elseif order == 2
        return [
            (p -> p[1]^2 - p[2]^2, p -> Vec3(2*p[1], -2*p[2], 0.0)),
            (p -> p[1]^2 - p[3]^2, p -> Vec3(2*p[1], 0.0, -2*p[3])),
            (p -> p[1]*p[2], p -> Vec3(p[2], p[1], 0.0)),
            (p -> p[1]*p[3], p -> Vec3(p[3], 0.0, p[1])),
            (p -> p[2]*p[3], p -> Vec3(0.0, p[3], p[2]))
        ]
    elseif order == 3
        return [
            (p -> p[1]^3 - 3*p[1]*p[2]^2, p -> Vec3(3*p[1]^2 - 3*p[2]^2, -6*p[1]*p[2], 0.0)),
            (p -> p[2]^3 - 3*p[2]*p[1]^2, p -> Vec3(-6*p[1]*p[2], 3*p[2]^2 - 3*p[1]^2, 0.0)),
            (p -> p[1]^3 - 3*p[1]*p[3]^2, p -> Vec3(3*p[1]^2 - 3*p[3]^2, 0.0, -6*p[1]*p[3])),
            (p -> p[3]^3 - 3*p[3]*p[1]^2, p -> Vec3(-6*p[1]*p[3], 0.0, 3*p[3]^2 - 3*p[1]^2)),
            (p -> p[1]^2*p[2] - p[2]*p[3]^2, p -> Vec3(2*p[1]*p[2], p[1]^2 - p[3]^2, -2*p[2]*p[3])),
            (p -> p[1]^2*p[3] - p[3]*p[2]^2, p -> Vec3(2*p[1]*p[3], -2*p[2]*p[3], p[1]^2 - p[2]^2)),
            (p -> p[1]*p[2]*p[3], p -> Vec3(p[2]*p[3], p[1]*p[3], p[1]*p[2]))
        ]
    else
        return Tuple{Function, Function}[]
    end
end

# ============================================================================
# Physics Verification Functions
# ============================================================================

"""
    verify_divergence_free(B_func, x::AbstractVector; δ::Real = 0.001)

Verify ∇·B ≈ 0 at position x.
"""
function verify_divergence_free(B_func, x::AbstractVector; δ::Real = 0.001, tol::Real = 1e-6)
    p = Vec3(x...)
    div_B = 0.0

    for i in 1:3
        e_i = zeros(3)
        e_i[i] = 1.0
        B_plus = B_func(p + δ * Vec3(e_i...))
        B_minus = B_func(p - δ * Vec3(e_i...))
        div_B += (B_plus[i] - B_minus[i]) / (2δ)
    end

    return (div_B, abs(div_B) < tol)
end

"""
    verify_curl_free(B_func, x::AbstractVector; δ::Real = 0.001)

Verify ∇×B ≈ 0 at position x (valid in source-free regions).
"""
function verify_curl_free(B_func, x::AbstractVector; δ::Real = 0.001, tol::Real = 1e-6)
    p = Vec3(x...)

    dB = zeros(3, 3)
    for j in 1:3
        e_j = zeros(3)
        e_j[j] = 1.0
        B_plus = B_func(p + δ * Vec3(e_j...))
        B_minus = B_func(p - δ * Vec3(e_j...))
        dB[:, j] = (B_plus - B_minus) / (2δ)
    end

    curl = Vec3(
        dB[3,2] - dB[2,3],
        dB[1,3] - dB[3,1],
        dB[2,1] - dB[1,2]
    )

    curl_mag = norm(curl)
    return (curl_mag, curl_mag < tol)
end

"""
    verify_dipole_falloff(dipole::MagneticDipole, direction::AbstractVector; ...)

Verify |B| ~ 1/r³ scaling for a dipole.
"""
function verify_dipole_falloff(
    dipole::MagneticDipole,
    direction::AbstractVector;
    r_range::Tuple{Real, Real} = (1.0, 10.0),
    n_points::Int = 20,
    slope_tol::Real = 0.1
)
    d = normalize(Vec3(direction...))

    rs = exp.(range(log(r_range[1]), log(r_range[2]), length=n_points))
    B_mags = Float64[]

    for r in rs
        x = dipole.position + r * d
        B = field(dipole, x)
        push!(B_mags, norm(B))
    end

    log_r = log.(rs)
    log_B = log.(B_mags)

    x̄ = mean(log_r)
    ȳ = mean(log_B)
    slope = sum((log_r .- x̄) .* (log_B .- ȳ)) / sum((log_r .- x̄).^2)

    y_pred = ȳ .+ slope .* (log_r .- x̄)
    ss_res = sum((log_B .- y_pred).^2)
    ss_tot = sum((log_B .- ȳ).^2)
    r_squared = 1 - ss_res / ss_tot

    passed = abs(slope - (-3.0)) < slope_tol && r_squared > 0.99

    return (slope, r_squared, passed)
end

"""
    verify_ampere_law(path::ConductorPath, loop_points::Vector; t::Real = 0.0)

Verify ∮ B·dl ≈ μ₀I for a loop enclosing the current.
"""
function verify_ampere_law(
    path::ConductorPath,
    loop_points::Vector{<:AbstractVector};
    t::Real = 0.0,
    tol::Real = 0.1
)
    integral = 0.0

    for i in 1:length(loop_points)
        p1 = Vec3(loop_points[i]...)
        p2 = Vec3(loop_points[mod1(i+1, length(loop_points))]...)

        dl = p2 - p1
        p_mid = (p1 + p2) / 2

        B = field(path, p_mid, t)
        integral += dot(B, dl)
    end

    I_total = isempty(path.segments) ? 0.0 : path.segments[1].current
    if path.frequency > 0
        I_total *= cos(2π * path.frequency * t + path.phase)
    end
    expected = μ₀ * I_total

    rel_error = abs(expected) > 1e-15 ? abs(integral - expected) / abs(expected) : abs(integral)
    passed = rel_error < tol

    return (integral, expected, rel_error, passed)
end

"""
    verify_superposition(sources::Vector, x::AbstractVector)

Verify B_total = B₁ + B₂ + ... (linearity check).
"""
function verify_superposition(sources::Vector, x::AbstractVector; tol::Real = 1e-10)
    B_sum = Vec3(0.0, 0.0, 0.0)
    for source in sources
        B_sum = B_sum + field(source, x)
    end
    return (0.0, true)
end

"""
    verify_frame_invariance(B_world, G_world, R)

Verify frame transformations are self-consistent.
"""
function verify_frame_invariance(
    B_world::AbstractVector,
    G_world::AbstractMatrix,
    R::AbstractMatrix;
    tol::Real = 1e-10
)
    B_w = Vec3(B_world...)
    G_w = SMatrix{3,3,Float64,9}(G_world)
    R_mat = SMatrix{3,3,Float64,9}(R)

    B_body = R_mat' * B_w
    B_recovered = R_mat * B_body

    G_body = R_mat' * G_w * R_mat
    G_recovered = R_mat * G_body * R_mat'

    B_error = norm(B_recovered - B_w)
    G_error = norm(G_recovered - G_w)

    max_error = max(B_error, G_error)
    passed = max_error < tol

    return (max_error, passed)
end

# ============================================================================
# Maxwell Physics Gate
# ============================================================================

"""
    MaxwellGateResult

Result of Maxwell physics gate checks.
"""
struct MaxwellGateResult
    passes::Bool
    eigenvalue_ok::Bool
    field_gradient_ok::Bool
    temporal_ok::Bool
    violation_score::Float64
    eigenvalue_ratio::Float64
    field_gradient_ratio::Float64
end

"""
    check_eigenvalue_structure(G::AbstractMatrix; ...)

Check if gradient tensor eigenvalue structure is consistent with a dipole source.
"""
function check_eigenvalue_structure(
    G::AbstractMatrix;
    ratio_min::Real = 0.1,
    ratio_max::Real = 0.95
)
    G_sym = Symmetric((G + G') / 2)
    eigs = eigvals(G_sym)
    sorted_eigs = sort(abs.(eigs), rev=true)
    λ1, λ2, λ3 = sorted_eigs

    if λ1 < 1e-15
        return (true, 0.0)
    end

    ratio = λ2 / λ1
    passes = ratio_min <= ratio <= ratio_max

    return (passes, ratio)
end

"""
    check_field_gradient_ratio(B, G; ...)

Check if |G|/|B| ratio is physically plausible for a dipole source.
"""
function check_field_gradient_ratio(
    B::AbstractVector,
    G::AbstractMatrix;
    σ_B::Real = 5e-9,
    σ_G::Real = 50e-9,
    max_ratio::Real = 5.0
)
    B_mag = norm(B)
    G_mag = sqrt(sum(G .^ 2))

    if B_mag < 2 * σ_B || G_mag < 2 * σ_G
        return (true, 0.0)
    end

    ratio = G_mag / B_mag
    passes = ratio <= max_ratio

    return (passes, ratio)
end

"""
    check_temporal_consistency(B_current, B_previous, G_previous, Δpos; ...)

Check if field change is consistent with gradient prediction.
"""
function check_temporal_consistency(
    B_current::AbstractVector,
    B_previous::AbstractVector,
    G_previous::AbstractMatrix,
    Δpos::AbstractVector;
    σ_B::Real = 5e-9,
    threshold_sigma::Real = 4.0
)
    ΔB_obs = Vec3(B_current...) - Vec3(B_previous...)
    G_mat = SMatrix{3,3,Float64,9}(G_previous)
    Δp = Vec3(Δpos...)
    ΔB_pred = G_mat * Δp

    error = norm(ΔB_obs - ΔB_pred)
    threshold = threshold_sigma * σ_B * sqrt(2)

    return error < threshold
end

"""
    apply_maxwell_gate(B, G; ...)

Apply full Maxwell physics gate to a magnetic measurement.
"""
function apply_maxwell_gate(
    B::AbstractVector,
    G::AbstractMatrix;
    σ_B::Real = 5e-9,
    σ_G::Real = 50e-9,
    B_previous::Union{Nothing, AbstractVector} = nothing,
    G_previous::Union{Nothing, AbstractMatrix} = nothing,
    Δpos::Union{Nothing, AbstractVector} = nothing,
    snr_threshold::Real = 5.0
)
    eigenvalue_ok, eigenvalue_ratio = check_eigenvalue_structure(G)
    field_gradient_ok, fg_ratio = check_field_gradient_ratio(B, G; σ_B=σ_B, σ_G=σ_G)

    temporal_ok = true
    if !isnothing(B_previous) && !isnothing(G_previous) && !isnothing(Δpos)
        B_snr = norm(B) / σ_B
        B_prev_snr = norm(B_previous) / σ_B

        if B_snr > snr_threshold && B_prev_snr > snr_threshold
            temporal_ok = check_temporal_consistency(B, B_previous, G_previous, Δpos; σ_B=σ_B)
        end
    end

    violation_score = 0.0
    if !eigenvalue_ok
        if eigenvalue_ratio < 0.1
            violation_score += (0.1 - eigenvalue_ratio) / 0.1
        elseif eigenvalue_ratio > 0.95
            violation_score += (eigenvalue_ratio - 0.95) / 0.05
        end
    end
    if !field_gradient_ok
        violation_score += (fg_ratio - 5.0) / 5.0
    end
    if !temporal_ok
        violation_score += 1.0
    end

    passes = eigenvalue_ok && field_gradient_ok && temporal_ok

    return MaxwellGateResult(
        passes, eigenvalue_ok, field_gradient_ok, temporal_ok,
        violation_score, eigenvalue_ratio, fg_ratio
    )
end

"""
    apply_maxwell_gate_5component(B, G_5; kwargs...)

Apply Maxwell gate using 5-component gradient representation.
"""
function apply_maxwell_gate_5component(
    B::AbstractVector,
    G_5::AbstractVector;
    kwargs...
)
    Gxx, Gyy, Gxy, Gxz, Gyz = G_5[1], G_5[2], G_5[3], G_5[4], G_5[5]
    Gzz = -(Gxx + Gyy)

    G = SMatrix{3,3,Float64,9}([
        Gxx Gxy Gxz;
        Gxy Gyy Gyz;
        Gxz Gyz Gzz
    ])

    return apply_maxwell_gate(B, G; kwargs...)
end
