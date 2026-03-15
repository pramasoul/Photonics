# The Real Fix: Numerical Dispersion Correction

## Summary

The ppw² convergence problem was misdiagnosed. It's not the coupling
finite-difference — it's **numerical dispersion mismatch between the
two Yee grids** creating a parasitic phase mismatch that the physical
poling period Λ can't compensate. The fix is a one-time correction to
the permittivity on each grid at initialization. Zero runtime cost.

## The Problem

The 1D Yee numerical dispersion relation for a medium with index n:

```
sin(ω·Δt/2) = S · sin(k_num·Δz/2) / n
```

where S = c·Δt/Δz (Courant number). This gives a slightly wrong
wavenumber k_num on each grid. The error is different on each grid
because ω₁ ≠ ω₂ and n₁ ≠ n₂.

QPM requires Δk = k₂ - 2k₁ = 2π/Λ. The physical Λ is computed from
the physical k values. But the simulation uses the numerical k values,
so there's a residual mismatch:

```
δ(Δk) = (k₂_num - k₂_true) - 2·(k₁_num - k₁_true)
```

This residual creates a parasitic coherence length:

```
L_coh_parasitic = π / |δ(Δk)|
```

### Numerical estimates

At ppw=20 (Δz ≈ 12 nm, S = 0.5):

Grid 1 (λ₁=1064 nm, n₁=2.148):
- Physical: k₁ = 2π·n₁/λ₁ = 12.694 rad/μm
- Numerical: solve dispersion relation → k₁_num ≈ 12.700 rad/μm
- Error: δk₁ ≈ +0.006 rad/μm

Grid 2 (λ₂=532 nm, n₂=2.225):
- Physical: k₂ = 2π·n₂/λ₂ = 26.289 rad/μm
- Numerical: solve dispersion relation → k₂_num ≈ 26.401 rad/μm
- Error: δk₂ ≈ +0.112 rad/μm

Residual mismatch:
```
δ(Δk) = δk₂ - 2·δk₁ ≈ 0.112 - 0.012 = 0.100 rad/μm = 100 /mm
```

Parasitic coherence length:
```
L_coh_parasitic = π / 0.100 ≈ 31 μm
```

In a 500 μm crystal, the SH undergoes ~16 parasitic oscillations
(build up for 15 μm, driven back down for 15 μm, repeat). Net
conversion is the small residual — explaining the observed ~4%.

At ppw=80: δ(Δk) ≈ 100/16 ≈ 6 /mm (scales as 1/ppw²), giving
L_coh_parasitic ≈ 500 μm ≈ crystal length. The parasitic mismatch
barely completes one cycle, so most of the crystal contributes
coherently — explaining the observed ~58%.

**This is the ppw² scaling.** It was never the coupling accuracy.

### Proof the coupling was fine all along

The forward difference `(E₁_{n+1}² - E₁_n²)/Δt` as an approximation
to `∂(E₁²)/∂t` where E₁² oscillates at 2ω₁:

- Amplitude accuracy: |sin(ω₂·Δt/2)/(ω₂·Δt/2)| at ppw=20 → 99.98%
- Phase offset: half-step delay → cos(ω₂·Δt/2) ≈ 99.95%

These are negligible. The coupling term was putting the right energy
into E₂ at the right phase — but the E₂ field was then propagating
at the wrong k, so the energy drifted out of phase with subsequent
source contributions.

## The Fix

Adjust the permittivity on each grid so that the **numerical** phase
velocity exactly matches the **physical** phase velocity at the
carrier frequency.

From the Yee dispersion relation, the numerical k for a given ω and n:

```
k_num(ω, n) = (2/Δz) · arcsin(n · sin(ω·Δt/2) / S)
```

We want k_num = k_true = n·ω/c. So we need an effective index n_eff
such that the Yee grid with permittivity ε₀·n_eff² propagates the
carrier at exactly the right k:

```
k_true = (2/Δz) · arcsin(n_eff · sin(ω·Δt/2) / S)
```

Solve for n_eff:

```
n_eff = S · sin(k_true·Δz/2) / sin(ω·Δt/2)
```

where k_true = n_physical · ω / c.

Expanding: `k_true·Δz/2 = n·ω·Δz/(2c) = n·π/ppw` (since
ppw = λ_medium/Δz and λ_medium = 2πc/(nω)).

### Implementation

In materials.py or sim.py initialization:

```python
def corrected_index(n_physical, omega, dz, dt, S):
    """
    Compute the effective refractive index that makes the Yee
    numerical dispersion exact at the given frequency.

    n_physical: true refractive index at omega
    omega: angular frequency (rad/s)
    dz: spatial step (m)
    dt: temporal step (s)
    S: Courant number (c*dt/dz)
    """
    k_true = n_physical * omega / c
    n_eff = S * np.sin(k_true * dz / 2) / np.sin(omega * dt / 2)
    return n_eff
```

Then at initialization:

```python
n1_eff = corrected_index(n1, omega1, dz, dt, S)
n2_eff = corrected_index(n2, omega2, dz, dt, S)

eps1 = eps0 * n1_eff**2   # used in Yee E-update for grid 1
eps2 = eps0 * n2_eff**2   # used in Yee E-update for grid 2
```

That's it. Three lines in initialization. Zero runtime cost.

### Expected corrections

| ppw | n₁ physical | n₁_eff | Δn₁/n₁ | n₂ physical | n₂_eff | Δn₂/n₂ |
|-----|-------------|--------|---------|-------------|--------|---------|
| 20  | 2.1483      | ~2.1468 | 0.07% | 2.2246      | ~2.2142 | 0.47% |
| 40  | 2.1483      | ~2.1479 | 0.02% | 2.2246      | ~2.2220 | 0.12% |
| 80  | 2.1483      | ~2.1482 | 0.005%| 2.2246      | ~2.2240 | 0.03% |

Sub-percent corrections everywhere. Grid 2 needs a larger correction
because the SH has a shorter wavelength (fewer ppw in absolute terms).

### Important: update the poling period too

The QPM period Λ = 2π/Δk where Δk = k₂ - 2k₁. With the corrected
indices, the physical Λ is unchanged — but verify that the code
computes Λ from the physical indices (not the corrected ones). The
poling pattern is a physical structure; the index correction is purely
numerical. Λ should always use n₁_physical and n₂_physical.

Similarly, the coupling coefficients χ⁽²⁾/n₂² should use the
**physical** n values (they represent material properties), while
the Yee update coefficients Δt/(ε·Δz) use the **corrected** n_eff
values (they control propagation).

### Coupling coefficients: which n to use

To be explicit:

```python
# Yee propagation (use corrected n for exact phase velocity)
yee_coeff1 = dt / (eps0 * n1_eff**2 * dz)
yee_coeff2 = dt / (eps0 * n2_eff**2 * dz)

# Nonlinear coupling (use physical n — material property)
nl_coeff_shg  = chi2 / n2_physical**2
nl_coeff_back = chi2 / (2.0 * n1_physical**2)

# MR projection weights (use physical n and ω)
mr_c1 = eps0 * n1_physical**2 / omega1
mr_c2 = eps0 * n2_physical**2 / omega2
```

## Validation

### Primary test: ppw convergence

Re-run the ppw sweep with corrected indices:

| ppw | E₂/E₁ (before fix) | E₂/E₁ (expected after) |
|-----|---------------------|------------------------|
| 20  | 4.4%                | ~75-85%                |
| 40  | 14.2%               | ~80-88%                |
| 80  | 57.9%               | ~85-90%                |
| 100 | 78.0%               | ~88-91%                |

**All values should now be within ~10-15% of each other and close
to the CW prediction.** If ppw=20 jumps to 70%+, the fix is working
and numerical dispersion was the dominant error.

Any remaining ppw dependence after this fix is the genuine coupling
finite-difference error (~0.02% per cycle amplitude, ~0.05% phase).
This should be a very weak residual.

### Secondary test: detuning contrast

With corrected indices, the QPM-matched vs deliberately-detuned
conversion ratio should be dramatic (>>10×), even at ppw=20. If
it was ~2.6× before, it should now be 20×+ because the matched
case actually works.

### Sanity check: dispersion relation verification

Before and after the correction, measure the actual numerical phase
velocity on each grid by launching a CW tone and tracking zero
crossings:

```python
# Inject CW at omega1 into grid 1, measure phase velocity
# Should be c/n1_physical (not c/n1_eff) after correction
v_phase_measured = (distance between zero crossings) / (time between them)
assert abs(v_phase_measured - c/n1_physical) / (c/n1_physical) < 1e-5
```

This confirms the correction is doing what it claims.

## Interaction with ADE Dispersion (Change 3)

When ADE is eventually added, the dispersion correction becomes
more nuanced. The ADE pole already modifies the numerical dispersion
— you'd need to solve the full Yee+ADE dispersion relation to find
n_eff. This is straightforward (it's a transcendental equation solved
once at init) but the formula above no longer applies directly.

For now, without ADE, the formula is exact.

## What This Means

If this fix works as expected, you'll have a simulation that:

- Gives quantitatively correct SHG conversion at ppw=20
- Conserves Manley-Rowe to ±0.2%
- Correctly shows total EM energy increase during SHG
- Runs at ~5 μs/step for a 500 μm crystal (~44k cells)
- Has zero computational overhead from the fix

This is the reference-quality simulation you wanted, and it's fast
enough to be interactive at 30+ fps. The ADE dispersion (Change 3)
then adds correct group velocity and GVD on top of this foundation.
