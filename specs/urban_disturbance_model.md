# Urban Disturbance Model

## Overview

Urban environments contain magnetic disturbance sources that are fundamentally different
from underwater scenarios. Sources may be **moving** (elevators, vehicles), **transient**
(doors opening/closing), or **static** (structural steel, HVAC equipment). All produce
magnetic anomalies that follow the same dipole physics (1/r³ decay) but with different
temporal dynamics.

## Source Taxonomy

### 1. Elevators
- **Dipole moment range**: 50–500 A·m² (motor + counterweight + cab steel)
- **Motion model**: Vertical-only, constrained to shaft axis
  - Acceleration: ≤ 1.5 m/s² typical
  - Max velocity: 1–4 m/s (low-rise to high-rise)
  - Dwell time at floors: 5–30 s
  - State machine: STOPPED → ACCELERATING → CRUISING → DECELERATING → STOPPED
- **Spatial extent**: Concentrated at motor room (top) and cab (moving)
- **Signature**: Strong vertical gradient, periodic stop pattern
- **Detection range**: 10–30 m depending on moment

### 2. Vehicles (Parking Garage)
- **Dipole moment range**: 5–100 A·m² (engine block, battery, steel body)
  - Electric vehicles: 20–100 A·m² (battery + motor magnets)
  - ICE vehicles: 5–50 A·m² (engine block, exhaust)
  - Parked vehicles: static dipole at known height (~0.5 m above floor)
- **Motion model**: Ground-plane constrained (2D + heading)
  - Max velocity: 5–15 km/h in parking structures
  - Turn radius: ≥ 3 m
- **Signature**: Horizontal approach/departure, ground-plane gradient
- **Detection range**: 5–15 m

### 3. Doors and Gates
- **Dipole moment range**: 1–20 A·m² (electromagnetic locks, hinges, frame)
  - Electromagnetic locks: 5–20 A·m² (energized), < 1 A·m² (de-energized)
  - Steel fire doors: 2–10 A·m² (permanent magnetization)
- **Motion model**: Rotational (hinge axis), binary open/close
  - Transition time: 1–5 s
  - Duty cycle: sporadic, correlated with foot traffic
- **Signature**: Step change at fixed location, small spatial extent
- **Detection range**: 2–5 m

### 4. Structural Steel (Static)
- **Dipole moment range**: 10–1000 A·m² (columns, beams, rebar)
- **Motion model**: None (static, absorbed into background map)
- **Signature**: Smooth spatial variation, consistent across missions
- **Treatment**: Part of background field, learned by tile map

### 5. HVAC and Electrical
- **Dipole moment range**: 1–50 A·m² (motors, transformers, conduit)
- **Motion model**: Static position, time-varying moment (motor cycling)
  - Period: 10–60 s for compressor cycles
  - On/off duty cycle: 30–70%
- **Signature**: Fixed position, amplitude modulation
- **Detection range**: 3–10 m

## Background Field Characteristics

### Indoor Magnetic Environment
- **Earth's field**: ~50 µT (varies by latitude), provides baseline
- **Building distortion**: ±20 µT from structural steel, smoothly varying
- **Gradient magnitude**: 0.1–10 µT/m indoors (vs 0.001–0.01 µT/m open ocean)
- **Spatial correlation length**: 2–10 m (building structural grid)

### Key Differences from Underwater
| Property | Underwater (AUV) | Urban (UrbanNav) |
|----------|-----------------|-------------------|
| Source dynamics | Static (seafloor) | Moving (elevators, vehicles) |
| Source density | Sparse (1–3 per 100m) | Dense (5–20 per floor) |
| Background gradient | Weak (geological) | Strong (structural steel) |
| Sensor velocity | 1–3 m/s | 0–2 m/s (walking) |
| Altitude variation | Continuous (depth) | Discrete (floors) |
| Source lifetime | Permanent | Minutes to hours |

## Signal-to-Noise Considerations

The SNR for detecting a dipole source at range r:
  SNR = (µ₀ / 4π) · |m| / (r³ · σ_noise)

Where:
- µ₀/4π = 10⁻⁷ T·m/A
- |m| = dipole moment magnitude [A·m²]
- r = distance to source [m]
- σ_noise = sensor noise floor [T] (typically 1–10 nT for fluxgate)

### Detection Ranges (SNR = 3, σ_noise = 5 nT)
- Elevator (m=200 A·m²): r_detect ≈ 23 m
- Vehicle (m=30 A·m²): r_detect ≈ 12 m
- Door (m=10 A·m²): r_detect ≈ 8 m
- HVAC (m=5 A·m²): r_detect ≈ 7 m
