# Phase 2: Envelope SSFM and Alternative Methods

## Status: DEFERRED — do after the FDTD technical note is complete

This document is the forward-looking part of the methods analysis,
separated from the immediate work (global MR projection + paper).
Implement after the technical note is written and the carrier-resolved
FDTD is documented as a known-good, known-limited reference.

---

## Overview

The carrier-resolved FDTD is valuable for pedagogy and for
understanding the numerics of coupled Yee grids. But for quantitative
reference physics — especially OPO cavities, long crystals, ultrafast
pulses, and chirped QPM — an envelope split-step Fourier method (SSFM)
is the right tool.

The recommendation is to keep both: FDTD for "see the waves," SSFM
for "trust the numbers." Same visualization interface, same sliders,
different physics engine.

## Envelope SSFM Implementation

### Core equations

Complex envelopes A₁(z,t) and A₂(z,t) where:
```
E₁(z,t) = Re[A₁(z,t) · exp(i(k₁z - ω₁t))]
E₂(z,t) = Re[A₂(z,t) · exp(i(k₂z - ω₂t))]
```

Coupled propagation equations:
```
∂A₂/∂z + (1/v_g2)∂A₂/∂t + (i·β₂₂/2)∂²A₂/∂t² = -iκ₂·d(z)·A₁²·exp(iΔkz)
∂A₁/∂z + (1/v_g1)∂A₁/∂t + (i·β₂₁/2)∂²A₁/∂t² = -iκ₁·d(z)·A₂·A₁*·exp(-iΔkz)
```

where:
- v_g = group velocity from Sellmeier
- β₂ = GVD from Sellmeier second derivative
- κ = coupling coefficient = ω·d_eff/(n·c)
- Δk = k₂ - 2k₁ (residual mismatch after QPM)

### Split-step algorithm

For each spatial step Δz:

1. **Half-step dispersion (frequency domain):**
   ```
   Â₁(ω) = FFT[A₁(t)]
   Â₁(ω) *= exp(-i·D₁(ω)·Δz/2)
   A₁(t) = IFFT[Â₁(ω)]
   ```
   where D₁(ω) = β₂₁ω²/2 + β₃₁ω³/6 + ... (or use full Sellmeier)

2. **Full-step nonlinear coupling (RK4 in z):**
   ```
   dA₂/dz = -iκ₂·d(z)·A₁²·exp(iΔkz)
   dA₁/dz = -iκ₁·d(z)·A₂·A₁*·exp(-iΔkz)
   ```

3. **Half-step dispersion (frequency domain):**
   Same as step 1.

### Grid sizing

- Temporal grid: ~200–1000 points (resolve pulse envelope)
- Spatial steps: ~500–5000 (resolve crystal, or poling period if
  using explicit d(z))
- Memory: ~16 KB per spatial step (two complex arrays × 1000 points)
- Total: ~80 MB for 5000 steps — negligible

### Poling pattern options

**Option A: d_eff with Δk phase factor.** Use the first Fourier
coefficient of the poling pattern. Simple, standard, accurate for
uniform poling. The exp(±iΔkz) factors handle the residual mismatch.

**Option B: explicit d(z).** Keep the full square wave poling pattern.
Spatial step must resolve the poling period (~7 μm), giving Δz ≈ 0.5 μm.
This naturally captures duty cycle errors, chirped poling, aperiodic
QPM, and higher-order QPM. More spatial steps but each is cheap.

Start with Option A, add Option B for chirped/aperiodic experiments.

### Performance estimate

Per spatial step: 2 FFTs (256–1024 points) + 4 RK4 evaluations
= ~2 μs on RTX 5090.

Full crystal transit (1000 steps): ~2 ms.
Full crystal transit (5000 steps): ~10 ms.

This is fast enough to run at 60+ fps while sweeping parameters
interactively. Far faster than the carrier-resolved FDTD.

## OPO Cavity Mode

The SSFM makes OPO simulation natural:

```
for round_trip in range(N_round_trips):
    # Forward pass through crystal
    propagate_forward(A1, A2, crystal_length)

    # Right mirror: signal reflects, pump transmits
    A1 *= R_right        # signal reflection
    A2 *= T_right        # pump transmission (output coupling)

    # Backward pass through crystal
    propagate_backward(A1, A2, crystal_length)

    # Left mirror: signal reflects, pump enters
    A1 *= R_left         # signal reflection
    A2 = A2_pump         # fresh pump injected

    # Record intracavity power, output power, spectrum
```

Each round trip is two SSFM passes (~4 ms). Running 1000 round trips
to reach steady state: ~4 seconds. This is feasible for interactive
exploration of:
- OPO threshold (pump power vs cavity loss)
- Steady-state conversion efficiency
- Signal/idler spectrum evolution
- Doubly-resonant vs singly-resonant dynamics

## Side-by-Side Mode

Display both FDTD and SSFM simultaneously:

- Top panel: carrier-resolved E₁(z) and E₂(z) from FDTD
- Middle panel: SSFM envelope |A₁(z)| and |A₂(z)|, optionally
  reconstructed with carrier for visual comparison
- Bottom panel: spectra from both, overlaid

The SSFM envelope serves as the "correct answer" against which FDTD
artifacts are visible and instructive.

## Chirped QPM and Aperiodic Poling

With the SSFM using explicit d(z) (Option B), chirped poling is
trivial: vary Λ(z) along the crystal. This enables:

- Broadband SHG (each spectral component phase-matches at a different
  position)
- Pulse compression via SHG (chirped QPM can compensate GVM)
- Aperiodic QPM for engineered spectral response

These are active research topics and the simulation would be directly
useful for exploring them.

## Temperature Gradient Modeling

With the SSFM, spatial variation of temperature T(z) is easy:
evaluate Sellmeier at each spatial step with the local T. This
models oven non-uniformity, thermal lensing (in 1D: thermal
refractive index gradient), and temperature-tuned bandwidth
engineering.

## Implementation Plan

1. New class `SSFMSimulation` in `ssfm.py`, same interface as
   `FDTDSimulation` (config dict in, fields out)
2. Shared `materials.py` (Sellmeier, d(z) generation)
3. Mode selector in `main.py`: FDTD / SSFM / side-by-side
4. New `viz_ssfm.py` for envelope-specific displays (temporal
   profile, spectrum, spectrogram)
5. OPO mode with cavity round-trip loop
6. Chirped QPM with Λ(z) specification
