# PPLN FDTD Fidelity Improvements — Phase 1.5

## Context

The simulation is now qualitatively correct: QPM works, conversion
efficiency approaches CW theory at high ppw, energy is conserved to
0.4%. These three changes push it to quantitative reference quality.
The RTX 5090 has ~20 GB VRAM free and the 1D grids are tiny, so
memory and compute are not constraints.

Priority order: implement and validate each change before starting
the next. Each one is independently valuable.

---

## Change 1: True Manley-Rowe Projection

### What

The current projection enforces EM energy conservation:
```
ε₁E₁² + ε₂E₂² = const
```

This is wrong for SHG. The correct conserved quantity is photon number
(Manley-Rowe):
```
ε₁E₁²/ω₁ + ε₂E₂²/ω₂ = const
```

Since ω₂ = 2ω₁, this simplifies to:
```
ε₁E₁² + ε₂E₂²/2 = const
```

With this projection, total EM energy (ε₁E₁² + ε₂E₂²) will correctly
*increase* during SHG. The increase should equal ε₂E₂²/2 — half the
SH energy — representing the work done by the nonlinear polarization.
This is real physics: two ω photons become one 2ω photon carrying
twice the energy.

### Implementation

In the per-cell projection, replace:
```cuda
// OLD (EM energy conservation)
double mr_old = c1 * e1*e1 + c2 * e2*e2;
double mr_new = c1 * e1_new*e1_new + c2 * e2_new*e2_new;
```

with:
```cuda
// NEW (Manley-Rowe conservation)
// c1 = ε₁/ω₁ = ε₀·n₁²/ω₁
// c2 = ε₂/ω₂ = ε₀·n₂²/ω₂ = ε₀·n₂²/(2ω₁)
double mr_old = c1 * e1*e1 + c2 * e2*e2;
double mr_new = c1 * e1_new*e1_new + c2 * e2_new*e2_new;
```

The rest of the projection (subtracting violation from E₁) stays
the same — only the coefficients c1, c2 change.

### Validation

1. Compute and display three quantities during transit:
   - MR = ε₁∫E₁²dz/ω₁ + ε₂∫E₂²dz/ω₂ → should be constant (±0.1%)
   - U_total = ε₁∫E₁²dz + ε₂∫E₂²dz → should INCREASE during SHG
   - ΔU = U_total - U_initial → should equal ε₂∫E₂²dz/2 (half the SH energy)

2. Verify: if you disable the nonlinear coupling entirely, MR and
   U_total should both be constant (pure linear propagation).

---

## Change 2: Second-Order Time Derivative for Coupling

### What

The current ΔP_NL coupling uses a first-order forward difference:
```
∂(E₁²)/∂t ≈ (E₁_new² - E₁_old²) / Δt
```

This has O(Δt²) phase error per cycle at 2ω, which accumulates
over hundreds of carrier cycles to give the ppw² convergence
problem. At ppw=40 the coupling captures only ~15% of the CW
prediction.

Replace with the wave equation source using a *numerical* (not
analytical) second time derivative:
```
∂²(E₁²)/∂t² ≈ (E₁_{n+1}² - 2·E₁_n² + E₁_{n-1}²) / Δt²
```

This centered second difference has O(Δt⁴) phase error per cycle —
two orders better. It correctly handles both the DC component of
E₁² (gives zero, as it should) and the 2ω component (gives the
right amplitude and phase).

The coupling source on E₂ becomes:
```
ΔE₂ = -Δt² · χ⁽²⁾ · d(z) · (E₁_{n+1}² - 2·E₁_n² + E₁_{n-1}²) / (n₂² · Δt²)
     = -χ⁽²⁾ · d(z) · (E₁_{n+1}² - 2·E₁_n² + E₁_{n-1}²) / n₂²
```

Note: the Δt² from the source and the Δt² in the denominator of
the finite difference cancel! The expression is beautifully simple.

Similarly for back-conversion on E₁:
```
∂²(E₂·E₁)/∂t² ≈ (E₂_{n+1}·E₁_{n+1} - 2·E₂_n·E₁_n + E₂_{n-1}·E₁_{n-1}) / Δt²
```

So:
```
ΔE₁ = -2·χ⁽²⁾ · d(z) · (E₂_{n+1}·E₁_{n+1} - 2·E₂_n·E₁_n + E₂_{n-1}·E₁_{n-1}) / n₁²
```

The factor of 2 is the degeneracy factor for ω + ω → 2ω.

**Wait — use the MR-matched ratio from the coupling fix:**
```
nl_coeff_shg  = χ⁽²⁾ / n₂²
nl_coeff_back = χ⁽²⁾ / (2·n₁²)    // MR-matched (the 4× fix)
```

### Storage

Need two additional arrays:
- `E1_prev`: E₁ at timestep n-1 (float64, same size as E1)
- `E2_prev`: E₂ at timestep n-1 (float64, same size as E2)

At ~200k cells × 8 bytes = 1.6 MB each. Negligible.

Each timestep, rotate:
```
E1_prev ← E1_cur (copy before Yee update)
E1_cur  ← (Yee-updated value)
```

Or better: use a three-slot ring buffer (just swap pointers, no copy):
```
E1_buffers[3]  // indexed by timestep % 3
```

### Implementation

```cuda
// Ring buffer indices (passed as kernel params each step)
int n1 = step % 3;        // current = n+1 (post-Yee)
int n0 = (step + 2) % 3;  // previous = n
int nm = (step + 1) % 3;  // two ago = n-1

double e1_np1 = E1[n1][i];  // E₁ at n+1 (post-Yee update)
double e1_n   = E1[n0][i];  // E₁ at n
double e1_nm1 = E1[nm][i];  // E₁ at n-1

double e2_np1 = E2[n1][i];
double e2_n   = E2[n0][i];
double e2_nm1 = E2[nm][i];

double d = d_z[i];

// Second time derivative of P_NL (Δt² cancels)
double d2E1sq = e1_np1*e1_np1 - 2.0*e1_n*e1_n + e1_nm1*e1_nm1;
double d2E2E1 = e2_np1*e1_np1 - 2.0*e2_n*e1_n + e2_nm1*e1_nm1;

double src2 = -nl_coeff_shg  * d * d2E1sq;
double src1 = -nl_coeff_back * d * d2E2E1;

// Tentative coupling
double e2_new = e2_np1 + src2;
double e1_new = e1_np1 + src1;

// MR projection (Change 1)
double mr_old = mr_c1 * e1_np1*e1_np1 + mr_c2 * e2_np1*e2_np1;
double mr_new = mr_c1 * e1_new*e1_new + mr_c2 * e2_new*e2_new;
double dmr = mr_new - mr_old;

double e1sq = e1_new * e1_new;
if (e1sq > 1e-30) {
    e1_new *= (1.0 - 0.5 * dmr / (mr_c1 * e1sq));
}

E1[n1][i] = e1_new;
E2[n1][i] = e2_new;
```

### Kernel structure note

The Yee update and the coupling are now conceptually separate:
1. Yee H-update (both grids)
2. Yee E-update (both grids), writing into the n+1 buffer slot
3. Source injection
4. Nonlinear coupling + MR projection (reads n+1, n, n-1 slots)
5. Boundary conditions

These can remain in a single fused kernel. The ring buffer is just
three separate allocations with pointer swapping on the host side
(or index arithmetic in the kernel).

### Initialization

The n-1 buffer must be initialized. Run 1 standard Yee step (no
coupling) to populate the n=0 and n=-1 slots before enabling
coupling. Or just zero-initialize E1_prev — the first coupling
step will be slightly wrong but it's one step out of millions.

### Validation

**Critical test: ppw convergence.** Re-run the ppw sweep:

| ppw | Expected E₂/E₁ (approx) |
|-----|--------------------------|
| 20  | Should now be ~60-80%    |
| 40  | Should be ~80-90%        |
| 80  | Should be ~85-90%        |
| 100 | Should match CW (~91%)   |

If ppw=20 and ppw=80 give similar results (within ~10%), the
ppw dependence is solved. This is the key test.

Also verify: detuned Λ should give dramatically lower conversion
than matched Λ (much more than the 2.6× seen with the old coupling).

---

## Change 3: Single-Pole ADE Dispersion

### What

Currently each grid has a constant refractive index: n₁ at ω₁ and
n₂ at ω₂. This is exact at the carrier frequencies but the grids
have no dispersion — all spectral components travel at the same
speed. For a 200 fs pulse (bandwidth ~5 THz), the group velocity
dispersion (GVD) of LiNbO₃ will:

- Chirp the pump pulse, reducing peak intensity
- Change the group velocity (which differs from c/n for dispersive
  media), affecting temporal walkoff between pump and SH
- Broaden the SH pulse differently from the pump

Adding one Lorentzian pole per grid gives each grid the correct
group velocity AND group velocity dispersion at its carrier wavelength.

### Lorentzian model

The standard ADE approach: add a polarization current J that
satisfies a damped oscillator equation. For each grid, the
relative permittivity becomes:

```
ε(ω) = ε_inf + Δε · ω_p² / (ω_p² - ω² + iγω)
```

We need to fit three parameters (ε_inf, Δε, ω_p — fixing γ = 0
for now since absorption is negligible) to match three quantities
at the carrier frequency:

1. n(ω_carrier) — refractive index (from Sellmeier)
2. n_g(ω_carrier) — group index: n_g = n - λ·dn/dλ (from Sellmeier derivative)
3. GVD(ω_carrier) — group velocity dispersion: d²k/dω² (from Sellmeier second derivative)

For MgO:LiNbO₃ at 1064 nm (extraordinary):
- n = 2.1483
- n_g ≈ 2.19 (compute from Sellmeier: n_g = n + ω·dn/dω)
- GVD ≈ 200 fs²/mm (compute from Sellmeier second derivative)

And at 532 nm:
- n = 2.2246
- n_g ≈ 2.35
- GVD ≈ 450 fs²/mm

### Fitting procedure

Write a function that takes the Sellmeier equation and a carrier
wavelength, evaluates n, dn/dω, d²n/dω² at that wavelength, and
fits (ε_inf, Δε, ω_p) to match. This is a 3×3 nonlinear system
but well-conditioned; scipy.optimize.fsolve will handle it.

**Important:** the fit only needs to be accurate *near* the carrier
frequency. It will be wildly wrong far from it. That's fine — each
grid only carries fields near its own carrier.

### ADE update equations

The Lorentzian polarization adds an auxiliary field J (polarization
current density) per grid:

```
∂J/∂t = ε₀·Δε·ω_p² · E - ω_p² · P - γ · J
∂P/∂t = J
```

Discretized (leapfrog-compatible):

```
J^{n+1/2} = (1-γΔt/2)/(1+γΔt/2) · J^{n-1/2}
           + ε₀·Δε·ω_p²·Δt/(1+γΔt/2) · E^n
           - ω_p²·Δt/(1+γΔt/2) · P^n

P^{n+1} = P^n + Δt · J^{n+1/2}
```

And the E-update becomes:
```
E^{n+1} = E^n + (Δt/ε_inf·ε₀) · (∂H/∂z - J^{n+1/2})
```

Note: the effective permittivity in the Yee update changes from
n² to ε_inf when using ADE. The Lorentzian pole accounts for the
rest of the permittivity.

### Storage

Per grid, two additional arrays:
- J (polarization current, float64)
- P (polarization, float64)

At 200k cells: 3.2 MB per grid, 6.4 MB total. Negligible.

### Implementation

Add to materials.py:
```python
def fit_lorentzian_pole(sellmeier_func, lambda_carrier_um, T_celsius):
    """
    Fit (eps_inf, delta_eps, omega_p) to match n, ng, GVD
    at the carrier wavelength from the Sellmeier equation.

    Returns dict with: eps_inf, delta_eps, omega_p, gamma (=0)

    Validate by comparing:
    - n at carrier (should match Sellmeier to 6+ digits)
    - ng at carrier (should match Sellmeier derivative to 4+ digits)
    - GVD at carrier (should match to ~10%)
    """
    ...
```

Add to the kernel: J and P updates interleaved with the Yee update.
The kernel sequence becomes:

1. H-update (both grids, standard Yee)
2. J-update (both grids, ADE oscillator)
3. P-update (both grids, integrate J)
4. E-update (both grids, Yee with J source and ε_inf)
5. Source injection
6. Nonlinear coupling + MR projection
7. Boundary conditions

### Boundary conditions

The Mur ABC needs to be updated for dispersive media. The wave
speed in the Mur condition should use the group velocity (not
c/n_phase), or better: switch to CPML (convolutional perfectly
matched layer) which handles dispersive media naturally. CPML in
1D is straightforward — about 10-20 cells of absorbing layer at
each end with exponentially graded conductivity. It's more robust
than Mur and frequency-independent.

If Mur is kept, use c/n_g instead of c/n for the boundary velocity.

### Validation

1. **Dispersion-only test** (disable nonlinearity): Launch a pulse
   on a single grid and measure the group velocity by tracking the
   envelope peak. It should match c/n_g from Sellmeier, not c/n.
   The difference is about 2% at 1064 nm and 5% at 532 nm.

2. **GVD test**: Launch a transform-limited 100 fs pulse and measure
   the pulse duration after propagating 5 mm (extend the grid
   temporarily). It should broaden according to:
   ```
   τ_out = τ_in · √(1 + (GVD·L/τ_in²)²)
   ```
   For 100 fs at 1064 nm, GVD ≈ 200 fs²/mm, L = 5 mm:
   τ_out ≈ 100 · √(1 + (1000/10000)²) ≈ 100.5 fs — very small
   broadening, which is consistent with GVD being a minor effect
   at this crystal length. The test is more dramatic at 532 nm
   with shorter pulses.

3. **SHG with dispersion**: Conversion efficiency should change
   slightly from the non-dispersive case due to corrected group
   velocity mismatch. The effect is small for 200 fs in 500 μm
   but becomes significant for longer crystals or shorter pulses.

---

## Summary

| Change | Memory cost | Compute cost | Physics gained |
|--------|-------------|-------------|----------------|
| MR projection | None | ~0 | Correct conservation law |
| 2nd-order coupling | +2 arrays (3.2 MB) | +3 FMAs/cell | ppw convergence: O(1/N⁴) not O(1/N²) |
| ADE dispersion | +4 arrays (6.4 MB) | +6 FMAs/cell | Group velocity, GVD, pulse chirp |
| **Total** | **~10 MB** | **~30% more FMAs** | **Reference-quality 1D SHG** |

Total VRAM usage stays under 50 MB. At 7 μs/step currently, the
extra FMAs might push to 10 μs/step. Still >100 fps at 2500 steps/frame.
