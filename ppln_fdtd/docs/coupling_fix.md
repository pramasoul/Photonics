# SHG Coupling Fix — Wave Equation Source Formulation

## Diagnosis

The ppw² scaling is the smoking gun. Here's what's happening:

The ΔP_NL approach computes `(E₁_new² - E₁_old²)/Δt` as an approximation
to `∂(E₁²)/∂t`. But E₁² oscillates at 2ω (since sin²(ωt) = ½ - ½cos(2ωt)),
so you're finite-differencing a quantity at the second harmonic frequency
with only ~20 samples per cycle.

The first-order finite difference of a sinusoid with N samples per cycle
captures `(N/π)·sin(π/N)` of the true derivative amplitude — at N=20 that's
~98.4%, which sounds fine. But the **phase error** of the finite difference
accumulates over the hundreds of carrier cycles in the crystal, causing
destructive interference. That's where the ppw² scaling comes from — it's
a phase-accuracy problem, not an amplitude-accuracy problem.

## The Fix: Wave Equation Source

Instead of differencing E₁² in time, use the wave equation formulation
where the second time derivative of P_NL is evaluated analytically.

Starting from the wave equation:

```
∂²E/∂t² - (c/n)² · ∂²E/∂z² = -1/(ε₀n²) · ∂²P_NL/∂t²
```

For a source oscillating at frequency 2ω:

```
∂²P_NL/∂t² ≈ -(2ω)² · P_NL = -ω₂² · ε₀ · χ⁽²⁾ · d(z) · E₁²
```

The key insight: the second time derivative is **analytical** — you know
the carrier frequency, so you replace ∂²/∂t² with -ω₂². No finite
differencing of the fast oscillation needed.

The Yee leapfrog E-update is already a second-order-in-time scheme
(equivalent to the wave equation discretized in time), so adding
`Δt² · ω₂² · (source)` is dimensionally and structurally consistent.
The remaining finite-difference error is only in how well the Yee grid
**propagates** the 2ω wave (numerical dispersion), which at 20 ppw
is very good — that's the standard FDTD accuracy regime.

## Implementation

Replace the coupling terms with:

```python
# Precompute (once, at init):
omega_2 = 2 * pi * c / lambda_sh       # angular frequency of SH
omega_1 = 2 * pi * c / lambda_fund      # angular frequency of fundamental
shg_coeff = dt**2 * omega_2**2 * chi2 / n2**2
back_coeff = dt**2 * omega_1**2 * chi2 * 2.0 / n1**2

# Per timestep, after standard Yee E-update:

# SHG: ω + ω → 2ω
E2 += shg_coeff * d_z * E1**2

# Back-conversion: 2ω → ω + ω (disable initially for validation)
# E1 += back_coeff * d_z * E2 * E1
```

Notes:
- Applied once per timestep, after the standard E₂ Yee update
- Uses current E₁ (post-update). No need to store E₁_old
- The factor of 2 in `back_coeff` comes from d(E₁·E₂)/dE₁ in the
  nonlinear polarization
- `chi2 = 2 * d33` (d-tensor to χ⁽²⁾ convention factor)
- `d_z` is the ±d₃₃ poling pattern array

## Validation Plan

1. **Disable back-conversion** for initial testing so you can compare
   directly against the undepleted-pump CW analytic prediction.

2. **PPW convergence test.** Run the same ppw sweep (20, 40, 80) with
   the new formulation. All three should now give approximately the same
   conversion ratio. If they do, the problem is solved. If ppw dependence
   persists, it's coming from somewhere else (numerical dispersion of the
   propagator, not the coupling).

3. **Matched vs detuned.** Verify that Λ = Λ_QPM gives strong monotonic
   SH growth and Λ = 2×Λ_QPM gives oscillatory non-growth. The ratio
   should now be dramatic (not the 2.6× seen before).

4. **CW comparison.** For a long pulse (≫ crystal transit time, effectively
   CW), compare E₂/E₁ at the crystal exit against the analytic formula:

   ```
   E₂(L) = -i · ω₂ · d_eff · E₁² · L / (n₂ · c)
   ```

   where `d_eff = (2/π) · d₃₃` for first-order QPM. At 1 GW/cm²
   (E₁ ≈ 59 MV/m) and L = 500 μm, this gives |E₂|/|E₁| in the range
   the original CW estimate predicted.

5. **Re-enable back-conversion** once the undepleted-pump case validates.
   At high pump intensity or longer crystals, you should see the
   characteristic sinusoidal energy exchange (pump depletion and
   back-conversion cycling).

## Bandwidth caveat

The approximation ∂²/∂t² → -ω₂² is exact for CW and very accurate for
pulses longer than ~10 carrier cycles. For the 200 fs pulse at 1064 nm
(~56 cycles FWHM, bandwidth ~5 THz), it's fine. It would start to matter
for few-cycle pulses, but that's well outside the current parameter range.

## Why not just increase to 80 ppw?

You could, and it would work with the ΔP_NL formulation. But that's
treating the symptom. The wave equation source should give converged
results at 20 ppw. Fix the coupling first, verify ppw-independence,
then you know the simulation is correct and 20 ppw is sufficient.
80 ppw is 64× more cells — unnecessary if the coupling is done right.
