# SHG Coupling Magnitude Issue — Question for Opus

## Context

We have a 1D FDTD simulation of SHG in PPLN with two coupled Yee grids
(fundamental ω and second harmonic 2ω). Physical parameters: 1064→532 nm,
MgO:LiNbO₃, d₃₃ = 27 pm/V, 500 μm crystal, 1 GW/cm² peak intensity
(E₀ = 59 MV/m), Courant S = 0.5, Δz = 12 nm (20 cells per SH wavelength).

QPM is working qualitatively: SH grows linearly with crystal length,
detuning kills it, ~2.6× QPM enhancement over heavily detuned case.
Energy conservation is good (+0.04% drift over full traversal).

## The Problem

CW coupled-wave theory predicts ~91% amplitude conversion (E₂/E₁) for
these parameters. The simulation gives ~5%. That's roughly 16× too low.

## What we've tried

### 1. D-field constitutive approach
```
E₂ = (D₂ - ε₀χ⁽²⁾d(z)E₁²) / (ε₀n₂²)
```
This gives a static perturbation that doesn't accumulate. Same ~5%
regardless of pulse length (200 fs to 2 ps). Abandoned.

### 2. ΔP_NL source-term approach (current)
```
E₂ -= χ⁽²⁾ · d(z) · (E₁_new² - E₁_old²) / n₂²
```
where E₁_old is saved before the Yee update and E₁_new is after.
This is the discrete form of -∂P_NL/∂t / (ε₀n₂²).

Same ~5% result. Per-step source magnitude is CORRECT (~430 V/m RMS
at the pulse peak, matching the CW coupling coefficient). But the
accumulation over the crystal gives only 2 MV/m instead of ~28 MV/m.

### 3. Resolution test (key finding)

Conversion scales as ppw² (points per SH wavelength squared):

| ppw | E₂/E₁ | Relative |
|-----|--------|----------|
| 20  | 0.26%  | 1×       |
| 40  | 1.08%  | 4.2×     |
| 80  | 3.24%  | 12.5×    |

This is close to ppw² ∝ (1/Δz²) scaling. At 80 ppw the answer is
converging toward the CW prediction but still below it.

## Diagnosis

The discrete time derivative `E₁_new² - E₁_old²` at 20 ppw
poorly approximates the continuous `∂(E₁²)/∂t`. With ~20 samples
per carrier cycle of E₁² (which oscillates at 2ω₁), the finite
difference loses most of the coupling strength. At 80 ppw (80 samples
per cycle), the approximation improves and conversion increases ~12×.

## Questions for Opus

1. **Is the ΔP_NL source-term approach the right formulation for
   first-order Yee FDTD with two coupled grids?** The D-field
   constitutive approach failed to accumulate. The ΔP_NL approach
   has correct per-step magnitude but resolution-dependent accumulation.

2. **Would the wave equation source formulation work better?**
   Instead of `ΔE₂ = -χ⁽²⁾d·ΔE₁²/n₂²`, use:
   ```
   ΔE₂ = Δt² · ω₂² · χ⁽²⁾ · d(z) · E₁² / n₂²
   ```
   This uses E₁² directly (not its discrete time derivative) and
   includes the ω₂² factor from the wave equation. It should converge
   at coarser resolution since it doesn't rely on finite-differencing
   a rapidly oscillating quantity.

3. **Is there a better coupling scheme entirely?** For example:
   - Auxiliary differential equation (ADE) for P_NL
   - Split-step / operator splitting between linear propagation and
     nonlinear coupling
   - Coupled-mode / envelope FDTD (separate carrier from envelope)

4. **Should we just increase to 80 ppw?** At 80 ppw the 500 μm crystal
   needs ~180k cells — still trivial for the RTX 5090 at 12 μs/step.
   The main cost is the spatial FFT display (larger array).

## Numerical details

- Grid: 44,325 cells at 20 ppw, Δz = 11.96 nm, Δt = 19.94 as
- Courant S = 0.5, numerical phase velocity error < 0.2%
- Kernel: single fused CUDA launch per step, 12 μs/step
- n₁ = 2.148, n₂ = 2.225 (Gayer et al. 2008 Sellmeier for MgO:LN)
- QPM period Λ = 6.97 μm (292 cells per half-period, well-resolved)
- GVM walk-off length >> crystal length (not limiting)
- Back-conversion disabled: same result (not the issue)
