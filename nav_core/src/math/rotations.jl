# ============================================================================
# rotations.jl - Rotation representations and conversions
# ============================================================================
#
# Convention: Hamilton quaternion convention, NED world frame
# q = [w, x, y, z] where w is scalar part
# ============================================================================

export quat_multiply, quat_conjugate, quat_rotate, quat_to_dcm, dcm_to_quat
export euler_to_quat, quat_to_euler, quat_normalize, euler_to_dcm

"""
    quat_multiply(q1, q2) -> SVector{4}

Quaternion multiplication (Hamilton convention).
"""
function quat_multiply(q1::SVector{4,T}, q2::SVector{4,T}) where T
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return SVector{4,T}(
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    )
end

"""
    quat_conjugate(q) -> SVector{4}

Quaternion conjugate (inverse for unit quaternions).
"""
function quat_conjugate(q::SVector{4,T}) where T
    return SVector{4,T}(q[1], -q[2], -q[3], -q[4])
end

"""
    quat_normalize(q) -> SVector{4}

Normalize quaternion to unit length.
"""
function quat_normalize(q::SVector{4,T}) where T
    n = sqrt(q[1]^2 + q[2]^2 + q[3]^2 + q[4]^2)
    return q / n
end

"""
    quat_rotate(q, v) -> SVector{3}

Rotate vector v by quaternion q.
"""
function quat_rotate(q::SVector{4,T}, v::SVector{3,T}) where T
    qv = SVector{4,T}(zero(T), v[1], v[2], v[3])
    result = quat_multiply(quat_multiply(q, qv), quat_conjugate(q))
    return SVector{3,T}(result[2], result[3], result[4])
end

"""
    quat_to_dcm(q) -> SMatrix{3,3}

Convert quaternion to direction cosine matrix (rotation matrix).
Note: SMatrix constructor is column-major, so we transpose the layout.
"""
function quat_to_dcm(q::SVector{4,T}) where T
    w, x, y, z = q
    # SMatrix is column-major: SMatrix(col1..., col2..., col3...)
    # Row 1: [1-2(y²+z²), 2(xy-wz), 2(xz+wy)]
    # Row 2: [2(xy+wz), 1-2(x²+z²), 2(yz-wx)]
    # Row 3: [2(xz-wy), 2(yz+wx), 1-2(x²+y²)]
    return SMatrix{3,3,T}(
        # Column 1
        1 - 2*(y^2 + z^2),
        2*(x*y + w*z),
        2*(x*z - w*y),
        # Column 2
        2*(x*y - w*z),
        1 - 2*(x^2 + z^2),
        2*(y*z + w*x),
        # Column 3
        2*(x*z + w*y),
        2*(y*z - w*x),
        1 - 2*(x^2 + y^2)
    )
end

"""
    dcm_to_quat(R) -> SVector{4}

Convert rotation matrix to quaternion.
"""
function dcm_to_quat(R::AbstractMatrix{T}) where T
    tr = R[1,1] + R[2,2] + R[3,3]

    if tr > 0
        s = sqrt(tr + 1) * 2
        w = s / 4
        x = (R[3,2] - R[2,3]) / s
        y = (R[1,3] - R[3,1]) / s
        z = (R[2,1] - R[1,2]) / s
    elseif R[1,1] > R[2,2] && R[1,1] > R[3,3]
        s = sqrt(1 + R[1,1] - R[2,2] - R[3,3]) * 2
        w = (R[3,2] - R[2,3]) / s
        x = s / 4
        y = (R[1,2] + R[2,1]) / s
        z = (R[1,3] + R[3,1]) / s
    elseif R[2,2] > R[3,3]
        s = sqrt(1 + R[2,2] - R[1,1] - R[3,3]) * 2
        w = (R[1,3] - R[3,1]) / s
        x = (R[1,2] + R[2,1]) / s
        y = s / 4
        z = (R[2,3] + R[3,2]) / s
    else
        s = sqrt(1 + R[3,3] - R[1,1] - R[2,2]) * 2
        w = (R[2,1] - R[1,2]) / s
        x = (R[1,3] + R[3,1]) / s
        y = (R[2,3] + R[3,2]) / s
        z = s / 4
    end

    return quat_normalize(SVector{4,T}(w, x, y, z))
end

"""
    euler_to_quat(roll, pitch, yaw) -> SVector{4}
    euler_to_quat(euler::SVector{3}) -> SVector{4}

Convert Euler angles (ZYX convention) to quaternion.
Angles in radians. Order is [roll, pitch, yaw].
"""
function euler_to_quat(roll::T, pitch::T, yaw::T) where T
    cr, sr = cos(roll/2), sin(roll/2)
    cp, sp = cos(pitch/2), sin(pitch/2)
    cy, sy = cos(yaw/2), sin(yaw/2)

    return SVector{4,T}(
        cr*cp*cy + sr*sp*sy,
        sr*cp*cy - cr*sp*sy,
        cr*sp*cy + sr*cp*sy,
        cr*cp*sy - sr*sp*cy
    )
end

# SVector version
euler_to_quat(euler::SVector{3,T}) where T = euler_to_quat(euler[1], euler[2], euler[3])

"""
    euler_to_dcm(roll, pitch, yaw) -> SMatrix{3,3}
    euler_to_dcm(euler::SVector{3}) -> SMatrix{3,3}

Convert Euler angles to Direction Cosine Matrix (rotation matrix).
Uses ZYX convention: R = Rz(yaw) * Ry(pitch) * Rx(roll)

This produces R_bn: the rotation from body frame to NED frame.
v_ned = R_bn * v_body

Convention:
- yaw = 0: body X points North
- yaw = π/2: body X points East
- yaw = π: body X points South
"""
function euler_to_dcm(roll::T, pitch::T, yaw::T) where T
    cr, sr = cos(roll), sin(roll)
    cp, sp = cos(pitch), sin(pitch)
    cy, sy = cos(yaw), sin(yaw)

    # R = Rz(yaw) * Ry(pitch) * Rx(roll)
    # This is the body-to-NED rotation matrix
    return SMatrix{3,3,T}(
        # Column 1
        cy*cp,
        sy*cp,
        -sp,
        # Column 2
        cy*sp*sr - sy*cr,
        sy*sp*sr + cy*cr,
        cp*sr,
        # Column 3
        cy*sp*cr + sy*sr,
        sy*sp*cr - cy*sr,
        cp*cr
    )
end

# SVector version
euler_to_dcm(euler::SVector{3,T}) where T = euler_to_dcm(euler[1], euler[2], euler[3])

"""
    quat_to_euler(q) -> Tuple{T, T, T}

Convert quaternion to Euler angles (roll, pitch, yaw).
Returns angles in radians.
"""
function quat_to_euler(q::SVector{4,T}) where T
    w, x, y, z = q

    # Roll (x-axis rotation)
    sinr_cosp = 2 * (w*x + y*z)
    cosr_cosp = 1 - 2 * (x^2 + y^2)
    roll = atan(sinr_cosp, cosr_cosp)

    # Pitch (y-axis rotation)
    sinp = 2 * (w*y - z*x)
    pitch = abs(sinp) >= 1 ? copysign(T(π/2), sinp) : asin(sinp)

    # Yaw (z-axis rotation)
    siny_cosp = 2 * (w*z + x*y)
    cosy_cosp = 1 - 2 * (y^2 + z^2)
    yaw = atan(siny_cosp, cosy_cosp)

    return (roll, pitch, yaw)
end
