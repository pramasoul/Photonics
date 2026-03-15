# Global Manley-Rowe Projection — Implementation Brief

## Context

The per-cell MR projection caused Nyquist checkerboard instability
(divergence at carrier zero crossings). This replaces it with a
global scalar projection: one GPU reduction per step, no per-cell
division by field values, no spatial instability.

This is the final stabilization step for the carrier-resolved FDTD
before we document the development in a writeup.

## Implementation

After applying BOTH coupling terms (SHG and back-conversion) to all
cells, enforce Manley-Rowe conservation globally:

```python
# Precomputed constants (at init):
#   mr_c1 = eps0 * n1_physical**2 / omega1
#   mr_c2 = eps0 * n2_physical**2 / omega2

# --- In the coupling kernel or as a post-coupling step ---

# 1. Compute global MR before coupling (from saved pre-coupling fields)
MR_before = cp.sum(mr_c1 * E1_pre**2 + mr_c2 * E2_pre**2) * dz

# 2. Apply coupling (SHG + back-conversion, as currently implemented)
#    ... existing ΔP_NL code ...

# 3. Compute global MR after coupling
MR_after = cp.sum(mr_c1 * E1**2 + mr_c2 * E2**2) * dz

# 4. Uniform correction
ratio = cp.sqrt(MR_before / MR_after)
E1 *= ratio
E2 *= ratio
```

### Why scale both fields by the same ratio?

- Preserves the relative phase of E₁ and E₂ everywhere (no phase
  corruption, no spatial artifacts)
- Preserves the E₂/E₁ ratio at every cell (the coupling's spatial
  structure is untouched)
- The correction is small (ratio ≈ 1 ± 0.001 per step), so the
  perturbation to the field evolution is negligible
- MR is exactly conserved by construction

### Where to apply the scaling

Only inside the crystal region (where the nonlinear coupling acts).
In vacuum margins the fields are uncoupled and MR is trivially
conserved. Scaling in vacuum would create a tiny artificial
discontinuity at the crystal boundary.

```python
E1[crystal_start:crystal_end] *= ratio
E2[crystal_start:crystal_end] *= ratio
```

### Kernel structure options

**Option A: separate reduction step.** Keep the existing fused
Yee+coupling kernel. After it completes, run a separate reduction
kernel for MR_before/MR_after, then a scaling kernel. Three
launches per step instead of one.

**Option B: two-pass fused kernel.** First pass: Yee + coupling +
accumulate per-block partial sums (shared memory reduction). Second
pass: finalize reduction + apply scaling. Two launches.

**Option A is simpler and the performance cost is tolerable** (~10 μs
for the reduction on 200k elements). Go with Option A unless
profiling shows it's a problem.

### Handling edge cases

If MR_after ≈ 0 (no field in crystal yet, e.g. before pulse arrives):
ratio would be 0/0. Guard with:

```python
if MR_after > 1e-30:
    ratio = cp.sqrt(MR_before / MR_after)
else:
    ratio = 1.0
```

### Pre-coupling field storage

You need E1_pre and E2_pre (the fields before coupling is applied)
to compute MR_before. Two options:

1. Save copies before coupling: `E1_pre = E1.copy()` — costs one
   extra 200k array copy per step (~2 μs).

2. Compute MR_before from the pre-Yee fields and account for the
   Yee update's MR change (which should be zero for the linear
   propagator). This avoids the copy but is more complex.

Option 1 is simpler. The copy is cheap. Or: compute MR_before
at the end of the previous step (after projection), cache it as
a scalar, and only compute MR_after in the current step. This
avoids the copy entirely:

```python
# At end of each step (after projection):
self.MR_cached = cp.sum(mr_c1 * E1**2 + mr_c2 * E2**2) * dz

# Next step, after coupling:
MR_after = cp.sum(mr_c1 * E1**2 + mr_c2 * E2**2) * dz
ratio = cp.sqrt(self.MR_cached / MR_after)
```

This works because the Yee linear update conserves MR (no energy
exchange between grids), so MR at end-of-previous-step equals MR
at pre-coupling-of-current-step.

## Validation

### MR conservation
MR/MR₀ should be 1.0000 ± noise (< 0.01%) throughout the full
crystal transit at any ppw.

### Conversion efficiency
Should be similar to current values (maybe slightly different due
to the projection modifying the field amplitudes):

| ppw | Before global MR | After (expected) |
|-----|------------------|------------------|
| 20  | 33%              | ~30-35%          |
| 40  | 37%              | ~35-40%          |
| 100 | 78-110%          | ~75-85%          |

The >100% values at high ppw should disappear (the projection
prevents E₂ from gaining more photons than E₁ loses).

### Stability at ppw=256
Should now be stable through full transit. The global MR clamp
prevents the positive feedback loop from diverging. Test this
explicitly.

### EM energy
Total EM energy should increase during SHG (real physics), by
an amount equal to half the SH field energy (from Manley-Rowe:
each 2ω photon carries 2× the energy of an ω photon).

## Performance estimate

| Component         | Time    |
|-------------------|---------|
| Yee + coupling    | 7-9 μs  |
| MR reduction      | 5-10 μs |
| Scaling kernel    | 2-3 μs  |
| **Total per step**| ~15-22 μs |

At 2500 steps/frame: 37-55 ms/frame → 18-27 fps. Still interactive.
Reduce steps/frame if needed to maintain 30 fps.
