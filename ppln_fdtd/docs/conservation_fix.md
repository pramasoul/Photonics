# Conservation Fix — Symplectic Nonlinear Sub-Step

## The Problem

The SHG source `ΔE₂ ∝ (E₁_new² - E₁_old²)` and the back-conversion
source `ΔE₁ ∝ E₂·(E₁_new - E₁_old)` are applied independently and
don't form a matched pair. Energy and Manley-Rowe are violated because
the coupling adds to E₂ without self-consistently removing from E₁ in
the same operation.

## The Fix: Operator Splitting with Analytic Nonlinear Sub-Step

### Concept

Split each timestep into two operations:

1. **Linear step**: standard Yee update on both grids (propagation,
   no coupling). This is already symplectic and energy-conserving.

2. **Nonlinear step**: at each cell, rotate energy between E₁ and E₂
   according to the χ⁽²⁾ coupling, conserving Manley-Rowe exactly.

The linear step doesn't touch the coupling. The nonlinear step doesn't
touch the spatial derivatives. This is standard Strang splitting.

### The Nonlinear Sub-Step

At each cell i, after the Yee update, we have E₁ and E₂. We need to
apply the nonlinear coupling for one Δt. The coupled equations
(time-only, no spatial derivatives) are:

```
dE₂/dt = -κ₂ · d(z) · E₁²
dE₁/dt = -κ₁ · d(z) · E₂ · E₁
```

where κ₂ = χ⁽²⁾/(2n₂²ε₀) × (appropriate dimensional factors) and
similarly for κ₁. These conserve Manley-Rowe when:

```
ε₁ · E₁ · dE₁/dt / ω₁ + ε₂ · E₂ · dE₂/dt / ω₂ = 0
```

Substituting: ε₁·κ₁·E₁²·E₂/ω₁ = ε₂·κ₂·E₁²·E₂/ω₂. This gives
the constraint on the coupling constants:

```
κ₁/κ₂ = (ε₂·ω₁)/(ε₁·ω₂) = (n₂²·ω₁)/(n₁²·ω₂) = n₂²/(2·n₁²)
```

**First check: do your current nl_coeff1 and nl_coeff2 satisfy this
ratio?** If nl_coeff2 = χ⁽²⁾/n₂² and nl_coeff1 = 2χ⁽²⁾/n₁², then
κ₁/κ₂ = (2/n₁²)/(1/n₂²) = 2n₂²/n₁², which is OFF by a factor of 4
from the Manley-Rowe requirement of n₂²/(2n₁²). This alone could
explain a big chunk of the energy non-conservation.

### Correct Coupling Constants

From the nonlinear wave equation for SHG with fields in SI units:

```
∂P_NL/∂t for the 2ω grid: ε₀·χ⁽²⁾·d(z)·∂(E₁²)/∂t
∂P_NL/∂t for the ω grid:  ε₀·χ⁽²⁾·d(z)·∂(2·E₂·E₁)/∂t
```

The factor of 2 in the back-conversion comes from the degeneracy of
ω + ω → 2ω (two identical input photons). In the Yee E-update
(from ∂D/∂t = ∇×H):

```
ΔE₂ = -(1/ε₂) · ΔP_NL_2 = -(χ⁽²⁾/n₂²) · d(z) · Δ(E₁²)
ΔE₁ = -(1/ε₁) · ΔP_NL_1 = -(2χ⁽²⁾/n₁²) · d(z) · Δ(E₂·E₁)
```

Now check Manley-Rowe. The energy change per grid per cell is:

```
ΔU₂ = ε₂ · E₂ · ΔE₂ · Δz = -ε₀·χ⁽²⁾·d(z) · E₂ · Δ(E₁²) · Δz
ΔU₁ = ε₁ · E₁ · ΔE₁ · Δz = -2ε₀·χ⁽²⁾·d(z) · E₁ · Δ(E₂·E₁) · Δz
```

For MR conservation: ΔU₁/ω₁ + ΔU₂/ω₂ = 0, i.e. ΔU₁ + ΔU₂/2 = 0:

```
-2ε₀χ⁽²⁾d · E₁ · Δ(E₂·E₁) - ½ε₀χ⁽²⁾d · E₂ · Δ(E₁²) = 0
```

Expand the deltas using the POST-update values:
```
Δ(E₁²) = E₁_new² - E₁_old²
Δ(E₂·E₁) = E₂_new·E₁_new - E₂_old·E₁_old
```

These are NOT equal to `2E₁·ΔE₁` and `E₂·ΔE₁ + E₁·ΔE₂` exactly
(because both E₁ and E₂ are changing simultaneously). This is the
root cause: **sequential application of the coupling breaks the
cancellation that would hold in the continuous equations.**

### Solution: Simultaneous Implicit Update

At each cell, solve for ΔE₁ and ΔE₂ simultaneously such that MR is
exactly conserved. Here's a practical approach:

**Step 1:** Compute the uncorrected updates (current method):
```
δE₂ = -nl2 · d · (E₁² - E₁_pre²)     // SHG drive
δE₁ = -nl1 · d · E₂ · (E₁ - E₁_pre)  // back-conversion drive
```
where E₁_pre is E₁ before the Yee update (for the ΔP_NL differencing)
and E₁, E₂ are post-Yee-update values.

**Step 2:** Apply both, then project onto the MR-conserving manifold.

After tentatively setting:
```
E₂_tent = E₂ + δE₂
E₁_tent = E₁ + δE₁
```

compute the MR violation:
```
MR_before = ε₁·E₁²/ω₁ + ε₂·E₂²/ω₂
MR_after  = ε₁·E₁_tent²/ω₁ + ε₂·E₂_tent²/ω₂
ΔMR = MR_after - MR_before
```

**Step 3:** Redistribute the violation. Scale E₁_tent and E₂_tent to
restore MR. The minimal correction that preserves the phase of both
fields is a multiplicative rescaling:

```
E₁_final = E₁_tent · √(α)
E₂_final = E₂_tent · √(β)
```

where α and β satisfy:
```
ε₁·α·E₁_tent²/ω₁ + ε₂·β·E₂_tent²/ω₂ = MR_before
```

subject to: the total EM energy change should come entirely from
the nonlinear redistribution, not from the correction. The simplest
constraint is α + β = 2 (symmetric correction) or to assign all
the correction to whichever field has more energy (effectively:
the correction is tiny and its exact distribution barely matters).

**Even simpler: single-field correction.** Since ΔMR is small (the
coupling per step is weak), just subtract the entire MR violation
from E₁ (the dominant field):

```
E₁_final = E₁_tent · √(1 - ΔMR·ω₁/(ε₁·E₁_tent²))
```

This is one extra multiply per cell. No branching, no iteration.

### Actually, the Simplest Correct Approach

I've been overcomplicating this with the projection. Here's a cleaner
path.

The problem is that the two coupling terms are applied sequentially
using stale values. The fix is to make them **use the same field values**:

```cuda
// Save pre-coupling values
double e1 = E1[i];  // post-Yee, pre-coupling
double e2 = E2[i];  // post-Yee, pre-coupling
double e1_pre = E1_old[i];  // pre-Yee (for ΔP_NL)

// Compute BOTH sources from the SAME state
double dE1sq = e1*e1 - e1_pre*e1_pre;
double src2 = -nl_coeff2 * d * dE1sq;          // SHG source

double dE2E1 = e2 * (e1 - e1_pre);             // note: uses e2, not e2+src2
double src1 = -nl_coeff1 * d * dE2E1;          // back-conv source

// Apply BOTH simultaneously
E2[i] = e2 + src2;
E1[i] = e1 + src1;
```

**This might already be what you're doing** — check whether E₂ is
modified before E₁ reads it in the coupling (if so, E₁'s back-conversion
uses the already-modified E₂, breaking the symmetry).

But even with simultaneous application, the discrete MR won't be
*exactly* zero — it'll be O(Δt²) per step, accumulating to O(Δt) over
the transit. The projection step cleans this up.

## Recommended Implementation

```cuda
// In the fused kernel, after Yee updates for both grids:

double e1 = E1[i];
double e2 = E2[i];
double e1_pre = E1_old[i];
double d = d_z[i];

// Compute both coupling sources from same pre-coupling state
double dE1sq = e1*e1 - e1_pre*e1_pre;
double src2 = -nl_coeff2 * d * dE1sq;

double dE2E1 = e2 * (e1 - e1_pre);
double src1 = -nl_coeff1 * d * dE2E1;

// Tentative update
double e1_new = e1 + src1;
double e2_new = e2 + src2;

// MR projection (local, per-cell)
double mr_old = c1 * e1*e1 + c2 * e2*e2;     // c1=ε₁/ω₁, c2=ε₂/ω₂
double mr_new = c1 * e1_new*e1_new + c2 * e2_new*e2_new;
double dmr = mr_new - mr_old;

// Subtract violation from fundamental (dominant field)
// e1² → e1² - dmr/c1, so e1 → e1·√(1 - dmr/(c1·e1²))
// For small dmr, use first-order: e1 → e1·(1 - dmr/(2·c1·e1²))
double e1sq = e1_new * e1_new;
if (e1sq > 1e-30) {  // avoid division by zero in vacuum
    e1_new *= (1.0 - 0.5 * dmr / (c1 * e1sq));
}

E1[i] = e1_new;
E2[i] = e2_new;
```

This is:
- Local (per-cell, no reductions)
- Branchless (the if is just a safety guard, always true inside crystal)
- One extra multiply and a few extra additions per cell
- Exact MR conservation to first order in the violation
- No iteration, no implicit solve

## Validation

After implementing, the MR/MR₀ column in the table should stay at
1.00 ± numerical noise (say ±0.1%) throughout the transit.

Total EM energy U₁ + U₂ should INCREASE during SHG — that's real
physics as discussed (each 2ω photon has 2× the energy). The increase
should be exactly equal to ΔU₂/2 (half the SH energy gained), which
is the work done by the nonlinear polarization.

## Priority

First verify that the coupling constants have the correct ratio for
MR (the factor-of-4 issue I flagged above). That might be the dominant
error. Then add the projection if residual violation is still significant.
