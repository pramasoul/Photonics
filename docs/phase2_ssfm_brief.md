# PPLN SSFM Envelope Solver — Phase 2 Implementation Brief

## Goal

A split-step Fourier method (SSFM) simulation of χ⁽²⁾ processes in
PPLN: SHG, OPA, and OPO. This is the quantitative reference engine,
complementing the carrier-resolved FDTD (Phase 1) which remains the
pedagogical visualization tool. The SSFM eliminates all of the FDTD's
numerical limitations (ppw dependence, coupling accuracy, MR drift,
deep-depletion instability) by propagating complex envelopes instead
of carrier-resolved fields.

## Physics Model

### Envelope equations

Two complex envelopes A₁(z,t) and A₂(z,t) for fundamental and SH:

```
E₁(z,t) = Re[A₁(z,t) · exp(i(k₁z - ω₁t))]
E₂(z,t) = Re[A₂(z,t) · exp(i(k₂z - ω₂t))]
```

Coupled propagation (in a frame moving at the fundamental group
velocity v_g1):

∂A₁/∂z + δ₁·∂A₁/∂t + (i·β₂₁/2)·∂²A₁/∂t²
    = -iκ₁ · d(z) · A₂ · A₁* · exp(-iΔkz)

∂A₂/∂z + δ₂·∂A₂/∂t + (i·β₂₂/2)·∂²A₂/∂t²
    = -iκ₂ · d(z) · A₁² · exp(+iΔkz)

where:
- δ₁ = 0 (moving frame), δ₂ = 1/v_g2 - 1/v_g1 (GVM)
- β₂ = GVD from Sellmeier second derivative (d²k/dω²)
- κ₁ = ω₁·d₃₃/(n₁·c), κ₂ = ω₂·d₃₃/(n₂·c)
- Δk = k₂ - 2k₁ - 2π/Λ (residual mismatch after QPM)
- d(z) = poling pattern (either ±1 square wave or d_eff = 2/π)

### Dispersion parameters from Sellmeier

Compute at initialization from the Gayer et al. (2008) Sellmeier
equation for MgO:LiNbO₃ extraordinary ray (same as Phase 1):

```python
def compute_dispersion(lambda_um, T_celsius):
    """Returns n, v_g, beta2, beta3 at the given wavelength."""
    # n from Sellmeier
    # v_g = c / n_g where n_g = n - λ·dn/dλ
    # beta2 = d²k/dω² (GVD)
    # beta3 = d³k/dω³ (TOD, optional)
    # Use numerical differentiation of Sellmeier for derivatives
```

### Two poling modes

**Option A: d_eff with Δk phase factor (default).** The poling
pattern enters as the first Fourier coefficient d_eff = (2/π)·d₃₃
and the QPM mismatch is carried by the exp(±iΔkz) factors. The
spatial step Δz can be large (~1-10 μm), limited only by the need
to resolve the nonlinear evolution, not the poling structure.

**Option B: explicit d(z).** The full ±d₃₃ square wave is kept.
Δz must resolve the poling period (~0.5 μm for Λ ≈ 7 μm). Use
this for chirped QPM, aperiodic poling, duty cycle studies. Set
Δk = 0 in the phase factors when using explicit d(z) (the QPM is
handled by the poling pattern itself).

## Algorithm

Symmetric split-step (Strang splitting) marching in z:

```
for each spatial step Δz:
    1. Half-step dispersion (frequency domain)
    2. Full-step nonlinear coupling (RK4 in z)
    3. Half-step dispersion (frequency domain)
```

### Dispersion half-step

Transform to frequency domain, apply the exact dispersion operator:

```python
A1_hat = fft(A1)
A2_hat = fft(A2)

# Full Sellmeier dispersion (not just beta2)
# D(ω) = k(ω₀ + ω) - k(ω₀) - ω/v_g  (residual dispersion)
# Or use Taylor expansion: D(ω) = β₂ω²/2 + β₃ω³/6 + ...

A1_hat *= exp(-i · D1(ω) · dz/2)
A2_hat *= exp(-i · D2(ω) · dz/2)

A1 = ifft(A1_hat)
A2 = ifft(A2_hat)
```

The dispersion operators D1(ω), D2(ω) are precomputed once at
initialization (they depend on the Sellmeier equation and the
frequency grid, but not on z or the fields).

For maximum accuracy, use the full Sellmeier to compute k(ω) at
every frequency bin, rather than truncating at β₂ or β₃. This is
free — it's just a precomputed array multiply.

### Nonlinear coupling step (RK4)

At each spatial position z, with the current A₁(t) and A₂(t),
advance by Δz using 4th-order Runge-Kutta:

```python
def coupling_rhs(z, A1, A2):
    phase = exp(i * delta_k * z)  # or 1.0 if using explicit d(z)
    d = d_pattern(z)              # d_eff or ±d33

    dA1_dz = -i * kappa1 * d * A2 * conj(A1) * conj(phase)
    dA2_dz = -i * kappa2 * d * A1**2 * phase

    return dA1_dz, dA2_dz
```

Standard RK4 with step Δz. For Option A with smooth evolution, a
single RK4 step per Δz is sufficient. For Option B where d(z) flips
sign every ~3.5 μm, Δz should be ≤ 0.5 μm so the sign flip is
resolved within the RK4 step.

### Conservation check

Manley-Rowe: |A₁|²/ω₁ + |A₂|²/ω₂ = const (integrated over t).
With RK4 at reasonable step size, MR drift should be < 10⁻⁸ per
pass. Compute and log this at each step as a sanity check. If
drift exceeds 10⁻⁴, the step size is too large.

No projection needed — the SSFM with RK4 is accurate enough that
conservation is automatic.

## Grid Parameters

### Temporal grid

The temporal grid resolves the pulse envelope, not the carrier.

- Nt = 256 to 2048 points (power of 2 for FFT)
- Temporal window T_win: must be large enough to contain the pulse
  plus any GVM walkoff plus some margin. For a 200 fs pulse in a
  500 μm crystal: walkoff = L · (1/v_g2 - 1/v_g1) ≈ 60 fs.
  T_win = 2 ps gives comfortable margin. Δt = T_win/Nt ≈ 4-8 fs.
- For OPO with CW pump: T_win should span several cavity round-trip
  times, Nt = 1024-4096.

Default: Nt = 512, T_win = 2 ps → Δt = 3.9 fs.

### Spatial grid

**Option A (d_eff):**
- Δz = 1-5 μm (resolve nonlinear evolution, not poling)
- 500 μm crystal → 100-500 steps
- 25 mm OPO crystal → 5000-25000 steps

**Option B (explicit d(z)):**
- Δz = 0.5 μm (resolve poling period)
- 500 μm crystal → 1000 steps
- 25 mm OPO crystal → 50000 steps

Default: Option A, Δz = 2 μm → 250 steps for 500 μm crystal.

## Implementation

### Directory structure

```
Photonics/
  common/                  # shared utilities (can defer, duplicate initially)
    materials.py
  ppln_fdtd/               # Phase 1 (unchanged)
    ...
  ppln_ssfm/               # Phase 2 (new)
    sim.py                 # SSFMSimulation class
    materials.py           # copy from ppln_fdtd, extend with dispersion params
    viz.py                 # matplotlib visualization
    main.py                # matplotlib entry point
    main_gl.py             # OpenGL entry point (later)
    default.yaml           # default parameters
    pixi.toml              # dependencies
```

### Dependencies

- `numpy` (primary compute — see GPU discussion below)
- `scipy` (RK4 via solve_ivp, or hand-rolled)
- `matplotlib` (visualization)
- `cupy` (optional, for GPU acceleration of long-crystal runs)

### sim.py — SSFMSimulation class

```python
class SSFMSimulation:
    def __init__(self, config):
        # Parse config: wavelength, crystal length, intensity,
        # temperature, pulse_width, Nt, dz, poling_mode ('deff' or 'explicit')

        # Compute Sellmeier parameters at both wavelengths
        # Compute dispersion operators D1(ω), D2(ω) — precomputed arrays
        # Compute coupling constants κ₁, κ₂
        # Initialize temporal grid, frequency grid
        # Initialize A1(t), A2(t) as zero arrays (complex128)
        # Set up source pulse (Gaussian envelope in A1)
        # Set up poling pattern d(z) if Option B

        # Storage for output history (for display)
        self.output_history = []  # list of (z, A1_out, A2_out) snapshots

    def propagate_pass(self, direction=+1):
        """
        Propagate one full pass through the crystal.
        direction: +1 (forward) or -1 (backward, for OPO)
        Returns: A1_exit, A2_exit (temporal profiles at crystal output)
        """
        z_start = 0 if direction == +1 else self.L_crystal
        z_end = self.L_crystal if direction == +1 else 0
        dz = self.dz * direction

        A1 = self.A1.copy()
        A2 = self.A2.copy()

        z = z_start
        for step in range(self.Nz):
            # Half-step dispersion
            A1 = self._dispersion_step(A1, self.D1, dz/2)
            A2 = self._dispersion_step(A2, self.D2, dz/2)

            # Full-step nonlinear coupling (RK4)
            A1, A2 = self._nonlinear_step(A1, A2, z, dz)

            # Half-step dispersion
            A1 = self._dispersion_step(A1, self.D1, dz/2)
            A2 = self._dispersion_step(A2, self.D2, dz/2)

            z += dz

            # Store snapshot if requested (for spatial display mode)
            if self._recording:
                self.spatial_record.append((z, A1.copy(), A2.copy()))

        self.A1 = A1
        self.A2 = A2
        return A1, A2

    def _dispersion_step(self, A, D_operator, dz):
        """Apply dispersion in frequency domain."""
        A_hat = np.fft.fft(A)
        A_hat *= np.exp(-1j * D_operator * dz)
        return np.fft.ifft(A_hat)

    def _nonlinear_step(self, A1, A2, z, dz):
        """RK4 step for the nonlinear coupling."""
        def rhs(z_local, A1_local, A2_local):
            d = self._poling(z_local)
            phase = np.exp(1j * self.delta_k * z_local)
            dA1 = -1j * self.kappa1 * d * A2_local * np.conj(A1_local) * np.conj(phase)
            dA2 = -1j * self.kappa2 * d * A1_local**2 * phase
            return dA1, dA2

        # Standard RK4
        k1a, k1b = rhs(z, A1, A2)
        k2a, k2b = rhs(z + dz/2, A1 + dz/2*k1a, A2 + dz/2*k1b)
        k3a, k3b = rhs(z + dz/2, A1 + dz/2*k2a, A2 + dz/2*k2b)
        k4a, k4b = rhs(z + dz, A1 + dz*k3a, A2 + dz*k3b)

        A1_new = A1 + (dz/6) * (k1a + 2*k2a + 2*k3a + k4a)
        A2_new = A2 + (dz/6) * (k1b + 2*k2b + 2*k3b + k4b)
        return A1_new, A2_new

    def _poling(self, z):
        """Return d(z) / d33. Option A: d_eff/d33 = 2/π ≈ 0.637.
        Option B: +1 or -1 depending on position in poling pattern."""
        if self.poling_mode == 'deff':
            return 2.0 / np.pi
        else:
            # Square wave: +1 if in even domain, -1 if odd
            domain_index = int(abs(z) / (self.Lambda / 2))
            return 1.0 if domain_index % 2 == 0 else -1.0

    def get_output_temporal(self):
        """Return |A1(t)|², |A2(t)|², t_grid for display."""
        return self.t_grid, np.abs(self.A1)**2, np.abs(self.A2)**2

    def get_output_spectral(self):
        """Return |Ã1(ω)|², |Ã2(ω)|², ω_grid for display."""
        A1_hat = np.fft.fftshift(np.fft.fft(self.A1))
        A2_hat = np.fft.fftshift(np.fft.fft(self.A2))
        return self.omega_grid, np.abs(A1_hat)**2, np.abs(A2_hat)**2

    def reset(self, config=None):
        """Reset fields to initial conditions (new pulse)."""
        ...
```

### CPU vs GPU strategy

Start with pure numpy (CPU). The SSFM for a 500 μm crystal with
Option A is ~250 steps × 2 FFTs on 512-point arrays = 500 FFTs.
On CPU with numpy, each FFT is ~1 μs, so the full pass is ~0.5 ms.
This is already 2000 fps — vastly faster than the display can use.

GPU (CuPy) becomes relevant only for:
- Option B with long crystals (50k+ steps)
- Batch parameter sweeps (many passes with different parameters)
- OPO with thousands of round trips

When GPU is needed, use CuPy's batched FFT (`cupy.fft.fft` on a
2D array where each row is one step's temporal profile). But don't
prematurely optimize — CPU will be fast enough for everything in
the initial implementation.

If profiling shows the nonlinear RK4 step is the bottleneck (it
shouldn't be — it's elementwise complex multiplies on 512 points),
it can be moved to CuPy without changing the structure.

The decision: **start pure numpy, profile, add CuPy selectively
if and when a specific bottleneck is identified.**

## Visualization

### Primary display: temporal + spectral at crystal output

Two-panel display (matplotlib initially, GL later):

**Top panel: temporal profiles.**
|A₁(t)|² (red) and |A₂(t)|² (blue) at the crystal exit.
X-axis: time in fs/ps. Y-axis: intensity (W/cm² or normalized).

**Bottom panel: spectra.**
|Ã₁(ω)|² and |Ã₂(ω)|² at the crystal exit.
X-axis: wavelength (nm) or frequency (THz). Y-axis: spectral
power density (log scale).

### OPO display: round-trip evolution

When running OPO mode, add a third panel:

**Waterfall / heatmap panel:**
Rows = round trips (most recent at top), columns = time.
Color = |A₁(t)|² (intracavity signal power). This shows the
pulse building up from noise, reaching steady state, and any
interesting dynamics (pulse breakup, self-pulsing, etc.).

Alternatively: intracavity power and output power vs round trip
number, as a simple line plot that updates in real time.

### Sliders (same concept as FDTD)

- **Λ** (poling period or temperature) — detune/retune QPM
- **T** (temperature) — shifts Sellmeier, detunes phase matching
- **Intensity** — pump peak intensity
- **Pulse width** — envelope duration
- **Crystal length** — can go much longer than FDTD (mm to cm)
- **R_signal** (OPO mode) — mirror reflectivity at signal wavelength

### Carrier reconstruction (optional diagnostic mode)

For comparison with FDTD, reconstruct the carrier-resolved field:
```python
E1_reconstructed = np.real(A1 * np.exp(1j * (k1 * z - omega1 * t)))
```
This can be overlaid on the FDTD output for side-by-side validation
but is NOT the primary display.

## Validation Plan

### Step 1: Linear propagation (no coupling)

Disable χ⁽²⁾. Launch a Gaussian pulse on grid 1. Verify:
- Group velocity matches c/n_g from Sellmeier (track envelope peak)
- Pulse broadens correctly from GVD over a known propagation
  distance. For 100 fs at 1064 nm, GVD ≈ 200 fs²/mm:
  after 5 mm, τ_out = τ_in · √(1 + (GVD·L/τ²)²) ≈ 100.5 fs
  (small but measurable)
- Pulse spectrum unchanged (dispersion doesn't create new frequencies)

### Step 2: Single-pass SHG cross-check against FDTD

Same parameters as Phase 1: 1064→532 nm, 1 GW/cm², 200 fs, 500 μm
crystal, 25°C. Compare:
- E₂/E₁ conversion efficiency (SSFM should match CW theory ~91%
  for this crystal/intensity; FDTD gives ~45% at crystal midpoint
  or ~75% at exit at ppw=100)
- SH temporal profile shape
- SH spectral bandwidth
- Group velocity walkoff between pump and SH

The FDTD's remaining ppw-dependent error means the SSFM and FDTD
won't match exactly. The SSFM should match the CW analytic theory
more closely. Document the comparison quantitatively.

### Step 3: CW limit

Use a long flat-top pulse (>>crystal transit time, effectively CW).
Compare conversion efficiency vs crystal length against the analytic
solution:
```
E₂(L) / E₁(0) = -i · ω₂ · d_eff · E₁(0) · L / (n₂ · c)
```
(undepleted pump, small-signal regime)

and the Jacobi elliptic function solution (with pump depletion):
```
E₂(L) = E₁(0) · sn(κ·L, m)
```
where κ and m depend on the coupling and phase mismatch.

### Step 4: Phase-matching bandwidth

Sweep Λ (or equivalently, Δk) and plot conversion efficiency as a
function of detuning. Compare the sinc² shape against the analytic
prediction:
```
η ∝ sinc²(ΔkL/2)
```
Verify that the FWHM narrows as 1/L.

### Step 5: OPO threshold (after cavity is implemented)

Sweep pump power at fixed cavity loss. Find the threshold where
signal power jumps from noise to macroscopic. Compare against the
analytic threshold condition:
```
g · L = ln(1/R)
```
where g is the parametric gain coefficient.

## OPO Cavity (after validation)

### Structure

```python
def run_opo(self, N_round_trips, pump_profile):
    """
    Run N round trips of a singly-resonant OPO.
    pump_profile: A2(t) injected each round trip (CW or pulsed).
    Signal (A1) recirculates; pump (A2) is single-pass.
    """
    # Initialize signal from noise
    self.A1 = noise_seed(self.Nt, amplitude=1e-6)

    for rt in range(N_round_trips):
        # Inject fresh pump
        self.A2 = pump_profile.copy()

        # Forward pass through crystal
        self.propagate_pass(direction=+1)

        # Output coupler: signal reflects, pump transmits
        A1_output = self.A1 * np.sqrt(1 - self.R_signal)
        self.A1 *= np.sqrt(self.R_signal)

        # Backward pass (if doubly-resonant or ring cavity)
        # For a linear cavity: propagate_pass(direction=-1)
        # For a ring cavity: skip backward pass

        # Record diagnostics
        self.record_round_trip(rt, A1_output)

        # Display update
        if rt % display_interval == 0:
            self.update_display(rt)
```

### Noise seed

For SPDC initiation (quantum noise analog):
```python
def noise_seed(Nt, amplitude):
    """Complex Gaussian noise with flat spectrum."""
    return amplitude * (np.random.randn(Nt) + 1j * np.random.randn(Nt))
```

The OPO should start from noise and build up to steady state when
above threshold. Below threshold, the signal stays at the noise
floor. This demonstrates the threshold phenomenon.

### Cavity configurations

Start with singly-resonant (signal recirculates, pump single-pass).
This is the simplest and most stable OPO type. Configurable as:
- **Ring cavity**: forward pass only, A₁ multiplied by R at the
  output coupler, fresh pump injected each round trip
- **Linear cavity**: forward + backward passes per round trip,
  mirrors at both ends

### Long-crystal OPO performance

For 25 mm crystal with Option A (Δz = 5 μm):
- 5000 steps per pass × 2 FFTs × ~1 μs = ~10 ms per pass
- Ring cavity: ~10 ms per round trip
- 1000 round trips: ~10 seconds — interactive-ish

For 25 mm crystal with Option B (Δz = 0.5 μm):
- 50000 steps per pass → ~100 ms per pass
- 1000 round trips → ~100 seconds — slow but acceptable as a
  "run and wait" computation. Show progress updates.

For interactive exploration (sweeping pump power, R, temperature):
use Option A. Switch to Option B only when poling structure matters.

## Deliverables (in order)

1. **sim.py**: SSFMSimulation class with propagate_pass, dispersion,
   nonlinear coupling, Sellmeier-derived parameters. Pure numpy.

2. **Validation**: run Steps 1-4 above, document results in a
   comparison table/plot. Cross-check against FDTD and analytics.

3. **main.py**: matplotlib display with temporal + spectral panels
   and interactive sliders.

4. **OPO mode**: cavity round-trip loop, noise seed, waterfall
   display, threshold demonstration.

5. **main_gl.py**: GL display (port from matplotlib once the physics
   is validated).

6. **Option B**: explicit d(z) poling for chirped QPM experiments.

## Parameters (default.yaml)

```yaml
# Wavelength and material
lambda_fund_um: 1.064
temperature_C: 25.0
crystal_length_um: 500.0

# Pulse
peak_intensity_W_cm2: 1.0e9
pulse_width_fs: 200.0

# Grid
Nt: 512
T_window_ps: 2.0

# Spatial stepping
poling_mode: deff        # 'deff' (Option A) or 'explicit' (Option B)
dz_um: 2.0              # spatial step (Option A default)
# dz_um: 0.5            # for Option B

# QPM
poling_period_um: 6.97   # auto-computed from Sellmeier if not specified

# OPO (when enabled)
opo_enabled: false
R_signal: 0.9            # signal mirror reflectivity
N_round_trips: 1000
pump_mode: cw             # 'cw' or 'pulsed'
```

## What NOT to Include in Phase 2

- No 2D/3D (never, for this project)
- No waveguide modes or transverse structure
- No third-harmonic or cascaded processes (yet)
- No thermal effects (slow timescale, irrelevant for pulse propagation)
- No quantum noise statistics (SPDC photon counting) — the noise
  seed is classical, not quantum. Quantum would require the Wigner
  function or positive-P representation.
