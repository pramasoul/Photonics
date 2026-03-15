# Change 2 Result: Second-Order Coupling Doesn't Work in First-Order Yee

## What we tried

Replaced the first-order ΔP_NL coupling:
```
ΔE₂ = -χ⁽²⁾d(E₁_new² - E₁_old²)/n₂²
```

with the second-order centered difference (as Opus recommended):
```
ΔE₂ = -χ⁽²⁾d(E₁²_{n+1} - 2E₁²_n + E₁²_{n-1})/n₂²
```

This was supposed to eliminate the ppw² convergence problem by using
O(Δt⁴) phase accuracy instead of O(Δt²).

## Result

| ppw | 1st-order (ΔP_NL) | 2nd-order (d²/dt²) |
|-----|--------------------|--------------------|
| 20  | 4.4%              | 0.2%               |
| 40  | 14.2%             | 0.3%               |
| 60  | —                 | 0.4%               |
| 80  | 57.9%             | 0.6%               |
| 100 | 78.0%             | 1.1%               |

The second-order coupling is **100× weaker** and still ppw-dependent.

## Why it fails

The Yee leapfrog is a **first-order-in-time** scheme for E:
```
E^{n+1} = E^n + Δt·(∂E/∂t)
```

The ΔP_NL source term represents `Δt · ∂P_NL/∂t` — a first-order
time quantity. This matches the Yee E-update.

The second-order difference `(E₁²_{n+1} - 2E₁²_n + E₁²_{n-1})`
represents `Δt² · ∂²P_NL/∂t²` — a second-order time quantity.
Adding it directly to the first-order E-update means the source
enters as `Δt² · (...)` instead of `Δt · (...)`, making it a
factor of ~1/ωΔt ≈ 30× too weak (at ppw=20, ωΔt ≈ 0.3).

The Δt² "cancellation" from the wave equation discretization only
applies when adding to the second-order wave equation update
(`E^{n+1} = 2E^n - E^{n-1} + Δt²·source`), NOT to the first-order
Yee update.

## Sign test confirmation

With the 2nd-order coupling, flipping the source sign (`+=` vs `-=`)
gives **identical results**. This means the source is exactly 90° out
of phase with E₂ — it shifts E₂'s phase but doesn't grow it. The
wave equation source and the Yee propagator are at different time
orders, creating a systematic 90° mismatch.

## Current state

Reverted to first-order ΔP_NL + MR projection (Change 1). This gives:
- E₂/E₁ = 78% at ppw=100, 1 GW/cm², 500 μm PPLN
- MR conserved to ±0.2%
- Total energy correctly increases during SHG

The ppw² convergence remains. At ppw=100 (221k cells, 9 μs/step)
the result is quantitatively good. The ppw dependence is a known
limitation of carrier-resolved FDTD with explicit nonlinear coupling.

## Question for Opus

Is there a way to add a first-order-in-time source that has better
phase accuracy than the forward difference `E₁_new² - E₁_old²`?

For example:
- A centered difference `(E₁_{n+1/2}² - E₁_{n-1/2}²)/Δt` using
  time-averaged E₁ values?
- An analytic carrier extraction: compute the envelope of E₁,
  apply the coupling to the envelope, then re-modulate?
- A frequency-domain correction factor applied to the source?

Or is the ppw dependence fundamental to carrier-resolved FDTD and
the right answer is simply "use enough ppw"?
