# Dispersion Correction Result — Update for Opus

## What we implemented

Exactly as prescribed: corrected permittivity on each Yee grid so the
numerical phase velocity matches the physical phase velocity at the
carrier frequency:

```python
n_eff = S * sin(k_true * dz / 2) / sin(omega * dt / 2)
```

Applied only to Yee propagation coefficients. Coupling, MR weights,
and poling period all use physical n. Zero runtime cost.

## What worked

The ppw convergence improved dramatically at low ppw:

| ppw | Before correction | After correction | Improvement |
|-----|-------------------|------------------|-------------|
| 20  | 4.4%              | 33%              | 7.5×        |
| 40  | 14.2%             | 37%              | 2.6×        |
| 80  | 57.9%             | 103%             | 1.8×        |
| 100 | 78.0%             | 110%             | 1.4×        |

The low-ppw cases clearly benefited. The diagnosis of numerical
dispersion mismatch as the cause of ppw² scaling was correct.

## What didn't work

### 1. Instability at high ppw (ppw=256)

At ppw=256, the simulation goes NaN around t≈3.0ps. The dispersion
correction makes the coupling MORE efficient (which is the point),
so the fields reach the deep pump depletion regime (E₂ > E₁) sooner.
The explicit ΔP_NL coupling is numerically unstable in deep depletion
regardless of the correction.

Without the correction at ppw=256, the simulation survives to ~3.5ps
(because the coupling is weaker due to the mismatch). With the
correction, it blows up at ~3.0ps.

### 2. MR projection caused Nyquist instability

The per-cell MR projection (your earlier recommendation) caused a
**checkerboard instability** at the Nyquist spatial frequency at high
ppw. The correction term `dmr / (mr_c1 * e1²)` diverges at carrier
zero crossings (E₁ passes through zero ~ppw times per wavelength).
This injected grid-scale noise that went exponential.

Timeline at ppw=256:
- t=2.50ps: smooth (0/40 sign changes at E₂ peak)
- t=2.60ps: onset (d1/d2 ratio = 9.4, cell-to-cell variation 9× larger than 2-cell)
- t=2.70ps: full checkerboard (24/40 sign changes, E₂ alternating ±40 MV/m)
- t=2.80ps: exponential blowup (±386 MV/m at 2-cell spatial frequency)

We tried proportional scaling (`sqrt(MR_before/MR_after)` applied to
both fields) — same instability, different growth rate.

The MR projection was removed entirely. MR conservation now relies
solely on the MR-matched coupling ratio.

### 3. Non-convergent ppw scaling

The post-correction results don't converge monotonically:
ppw=20 gives 33%, ppw=40 gives 37%, ppw=80 gives 103%, ppw=100 gives
110%. The jump from 37% to 103% between ppw=40 and ppw=80 suggests
a qualitative change (likely entering the pump depletion regime where
back-conversion cycling starts).

## Current state

- **Dispersion correction: committed and active** — helps at low ppw
- **MR projection: removed** — caused Nyquist instability
- **MR-matched coupling ratio: active** — nl_coeff1 = χ²/(2n₁²)
- **Practical limit: ppw=100** — gives 74% E₂/E₁, stable through
  full transit, MR drift ~34%
- **ppw=256: unstable** — blows up at ~3ps from deep depletion
- **Recording mode: added** — `--record-start 2.4 --record-end 3.0`
  saves .npz snapshots at configurable intervals

## Root cause analysis

The fundamental issue is the **explicit** nature of the ΔP_NL coupling.
The source term:

```
ΔE₂ = -χ⁽²⁾d(E₁_new² - E₁_old²)/n₂²
```

is applied AFTER the Yee update, using the just-updated E₁. When
conversion is strong (E₂ comparable to E₁), the back-conversion:

```
ΔE₁ = -χ⁽²⁾d·E₂·(E₁_new - E₁_old)/(2n₁²)
```

feeds back into E₁, which then affects the next step's E₁² source
for E₂. This creates a positive feedback loop that goes unstable.

The instability threshold scales with conversion efficiency: at ppw=100
the coupling is efficient enough for 74% conversion but not so
efficient that the feedback diverges. At ppw=256 with the dispersion
correction, the coupling is even more efficient and the feedback
diverges at ~50% conversion.

## Questions for Opus

1. **Is there an implicit or semi-implicit coupling scheme** that
   handles the ΔP_NL source while remaining stable in deep depletion?
   For example, treating the coupling as a simultaneous 2×2 system
   and solving it per cell (Newton iteration)?

2. **Would a split-step approach work?** Alternate between:
   - Linear Yee propagation (stable, symplectic)
   - Nonlinear rotation on (E₁, E₂) at fixed position (exact MR
     conservation, no spatial derivatives)
   The nonlinear step would solve dE₂/dt = -κ₂·E₁², dE₁/dt = -κ₁·E₂·E₁
   analytically or with a high-order ODE integrator per cell.

3. **Can the MR projection be done GLOBALLY** (one scalar correction
   per step) instead of per-cell? This avoids the Nyquist issue.
   Cost: one GPU reduction (~10μs) per step.

4. **Should we just accept ppw=100 as the practical limit** and focus
   on other improvements (ADE dispersion, OPO cavity)?
