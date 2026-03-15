# Conservation Issue — Request for Opus

## Summary

Neither total EM energy nor Manley-Rowe is conserved during SHG.
The ΔP_NL source-term coupling creates both energy and photons.

## Data (ppw=100, 1 GW/cm², 500 μm PPLN, boost=1)

Reference taken at 200k steps when pulse has entered crystal.

| steps | t (ps) | E/E₀ | MR/MR₀ | E₂/E₁ |
|-------|--------|------|--------|--------|
| 200k  | 0.80   | 1.00 | 1.00   | 0.010  |
| 300k  | 1.20   | 2.03 | 0.93   | 0.095  |
| 400k  | 1.60   | 2.11 | 0.94   | 0.191  |
| 600k  | 2.39   | 2.45 | 1.05   | 0.367  |
| 800k  | 3.19   | 3.01 | 1.24   | 0.485  |
| 1000k | 3.99   | 3.92 | 1.54   | 0.567  |

Energy grows ~3.9× and MR grows ~1.5× during crystal transit.
Both should be constant (MR exactly, energy approximately for
the two-grid model where ω₂ = 2ω₁).

## Current coupling implementation

```cuda
// After standard Yee E-update:
double dE1sq = e1_new * e1_new - e1_old * e1_old;
double dE2E1 = e2_old * (e1_new - e1_old);

E2[i] -= nl_coeff2 * d * dE1sq;       // SHG: adds to E₂
E1[i] -= nl_coeff1 * d * dE2E1;       // back-conversion: modifies E₁
```

where `nl_coeff2 = χ⁽²⁾/n₂²` and `nl_coeff1 = 2χ⁽²⁾/n₁²`.

## Why it's non-conservative

The SHG source `ΔE₂ ∝ dE1sq` adds amplitude to E₂ proportional to
the change in E₁², but does NOT reduce E₁ in the same operation.
The back-conversion term is supposed to compensate, but the energy
balance equation:

```
ε₂ · 2E₂ · ΔE₂_shg + ε₁ · 2E₁ · ΔE₁_back = 0
```

is not satisfied because ΔE₂_shg depends on (E₁_new² − E₁_old²)
while ΔE₁_back depends on E₂_old·(E₁_new − E₁_old), and these
don't cancel for arbitrary field values.

## What we need

A coupling formulation for the first-order Yee leapfrog that:

1. **Correctly accumulates SH** — the ΔP_NL approach does this
   (verified: E₂/E₁ converges to CW theory with increasing ppw)

2. **Conserves Manley-Rowe** — MR/MR₀ should stay at 1.0 throughout
   the crystal transit, meaning photon conversion (not creation)

3. **Works at 20-100 ppw** — can't require extreme resolution

## Possible approaches

### A. Per-cell energy renormalization
After applying both coupling terms, compute the local energy change
and rescale E₁, E₂ to conserve MR at each cell. This is a local
operation (no global reduction), but it modifies the field phases
and might affect the SHG accumulation.

### B. Implicit coupling
Solve the coupled constitutive relation implicitly at each cell:
find E₁_new, E₂_new that simultaneously satisfy the Yee update
AND the Manley-Rowe constraint. This is a 2×2 nonlinear system
per cell, solvable by Newton iteration (1-2 iterations for weak
nonlinearity).

### C. Symplectic splitting
Alternate between linear propagation (Yee, symplectic) and
nonlinear phase rotation (coupling, canonical transformation).
The nonlinear step would rotate (E₁, E₂) on a constant-MR
manifold. This is how split-step Fourier methods handle χ⁽²⁾
in the envelope formulation.

### D. Something else entirely?

## Constraints

- Single fused CUDA kernel per step (currently 5-7 μs/step)
- Must work with the carrier-resolved two-grid Yee scheme
- Back-conversion included (needed for pump depletion physics)
- Real-valued fields (not complex envelope)

## Context

This is a 1D FDTD on an RTX 5090 via CuPy. The simulation runs
at 40 fps with 2500 steps/frame at ppw=100 (221k cells). The
coupling physics (QPM, conversion efficiency) is quantitatively
correct — the ONLY issue is energy/photon conservation.
