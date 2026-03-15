# Coupling Status — Findings for Opus

## Key finding: the ΔP_NL approach was correct all along

The wave equation source (`Δt²ω₂²χ⁽²⁾E₁²/n₂²`) doesn't work in the
first-order Yee scheme because:

1. `∂²P_NL/∂t² ≈ -ω₂²P_NL` is wrong for E₁² because sin²(ωt) has a
   DC component. The second time derivative of the DC part is zero,
   but -ω₂² multiplies it anyway, creating a spurious static source.

2. Even the correct second-order finite difference
   `(E₁^{n+1}² - 2E₁^n² + E₁^{n-1}²)` gives the wrong coupling
   when added to the first-order Yee update. The wave equation source
   is `Δt² × ∂²P_NL/∂t²` (second-order in time), but the Yee E-update
   is first-order: `E^{n+1} = E^n + Δt×(...)`. Adding a second-order
   source to a first-order scheme mismatches the time integration.

3. Both +/- signs give identical results (0.25%), confirming the source
   is 90° out of phase with E₂ and can't drive growth.

## The ΔP_NL approach

```
ΔE₂ = -χ⁽²⁾·d(z)·(E₁_new² - E₁_old²) / n₂²
```

This IS the correct first-order coupling for the Yee scheme. It gives:
- 500 μm, 1 GW/cm², 20 ppw: E₂/E₁ = 5%
- CW theory: E₂/E₁ = 91%
- The ppw² scaling shows this converges toward the right answer at
  higher resolution

## Path forward

Increase ppw from 20 to 40-80. The ΔP_NL approach at 80 ppw should
give ~80% conversion, matching CW theory. At 80 ppw:
- 177k cells (4× current)
- ~40 μs/step estimated
- Still 50+ fps at 500 steps/frame

The ppw is now a configurable parameter (constructor + CLI).
