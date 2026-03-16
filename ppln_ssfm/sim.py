"""Split-Step Fourier Method for χ⁽²⁾ processes in PPLN.

Propagates complex envelopes A₁(z,t) and A₂(z,t) through a PPLN crystal
using symmetric Strang splitting: half-step dispersion, full-step RK4
nonlinear coupling, half-step dispersion.
"""

import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from common.materials import (
    sellmeier_n, qpm_period, D33, C,
    group_velocity, gvd, wavenumber, dispersion_operator,
)


class SSFMSimulation:
    def __init__(
        self,
        lambda_fund_um: float = 1.064,
        T_celsius: float = 25.0,
        crystal_length_um: float = 500.0,
        peak_intensity_W_cm2: float = 1e9,
        pulse_width_fs: float = 200.0,
        Nt: int = 512,
        T_window_ps: float = 2.0,
        dz_um: float = 2.0,
        poling_mode: str = 'deff',
        poling_period_um: float | None = None,
        boost: float = 1.0,
    ):
        self.lambda_fund_um = lambda_fund_um
        self.lambda_sh_um = lambda_fund_um / 2.0
        self.T = T_celsius
        self.L = crystal_length_um * 1e-6  # meters
        self.peak_intensity = peak_intensity_W_cm2
        self.pulse_width_s = pulse_width_fs * 1e-15
        self.boost = boost
        self.Nt = Nt
        self.poling_mode = poling_mode

        # Refractive indices
        self.n1 = sellmeier_n(lambda_fund_um, T_celsius)
        self.n2 = sellmeier_n(self.lambda_sh_um, T_celsius)

        # Angular frequencies
        self.omega1 = 2 * np.pi * C / (lambda_fund_um * 1e-6)
        self.omega2 = 2 * self.omega1

        # Group velocities
        self.vg1 = group_velocity(lambda_fund_um, T_celsius)
        self.vg2 = group_velocity(self.lambda_sh_um, T_celsius)
        self.gvm = 1.0 / self.vg2 - 1.0 / self.vg1  # GVM: δ₂ in co-moving frame

        # GVD
        self.beta2_1 = gvd(lambda_fund_um, T_celsius)
        self.beta2_2 = gvd(self.lambda_sh_um, T_celsius)

        # Spatial grid
        self.dz = dz_um * 1e-6
        self.Nz = int(np.ceil(self.L / self.dz))
        self.dz = self.L / self.Nz  # exact fit

        # QPM
        if poling_period_um is None:
            self.Lambda = qpm_period(lambda_fund_um, T_celsius)
        else:
            self.Lambda = poling_period_um * 1e-6

        # Phase mismatch after QPM
        k1 = wavenumber(lambda_fund_um, T_celsius)
        k2 = wavenumber(self.lambda_sh_um, T_celsius)
        K_qpm = 2 * np.pi / self.Lambda
        self.delta_k = k2 - 2 * k1 - K_qpm

        # Coupling constants: κ = ω·d₃₃·boost / (n·c)
        d33 = D33 * self.boost
        self.kappa1 = self.omega1 * d33 / (self.n1 * C)
        self.kappa2 = self.omega2 * d33 / (self.n2 * C)

        # d_eff for Option A
        self.d_eff_ratio = 2.0 / np.pi  # first Fourier coefficient of square wave

        # Temporal grid (co-moving frame at v_g1)
        # Auto-expand window to accommodate GVM walkoff + pulse width
        gvm_walkoff = abs(self.gvm) * self.L  # seconds of walkoff
        min_window = (self.pulse_width_s * 4 + gvm_walkoff * 2) * 1.5
        self.T_window = max(T_window_ps * 1e-12, min_window)
        self.dt = self.T_window / Nt
        self.t_grid = np.arange(Nt) * self.dt - self.T_window / 2  # centered

        # Frequency grid
        self.omega_grid = 2 * np.pi * np.fft.fftfreq(Nt, d=self.dt)

        # Precompute dispersion operators
        self.D1 = dispersion_operator(lambda_fund_um, T_celsius, self.omega_grid)
        self.D2 = dispersion_operator(self.lambda_sh_um, T_celsius, self.omega_grid)
        # Add GVM to D2 (co-moving frame removes v_g1, so D2 gets the mismatch)
        self.D2 += self.omega_grid * self.gvm

        # Backward operator is just negated
        self.D1_back = -self.D1
        self.D2_back = -self.D2

        # Source amplitude from intensity: I = ½nε₀c|A|² (envelope convention)
        eps0 = 8.854e-12
        I_W_m2 = peak_intensity_W_cm2 * 1e4
        self.A0 = np.sqrt(2.0 * I_W_m2 / (self.n1 * eps0 * C))

        # Initialize fields
        self.reset()

    def reset(self):
        """Initialize with Gaussian pump pulse on grid 1, empty grid 2."""
        sigma_t = self.pulse_width_s / (2 * np.sqrt(2 * np.log(2)))  # FWHM to σ
        self.A1 = self.A0 * np.exp(-self.t_grid ** 2 / (2 * sigma_t ** 2)).astype(np.complex128)
        self.A2 = np.zeros(self.Nt, dtype=np.complex128)
        self.z_current = 0.0
        self.pass_count = 0
        self.spatial_record = []

    def propagate_pass(self, direction=+1, record=False):
        """Propagate one full pass through the crystal.

        Args:
            direction: +1 (forward) or -1 (backward)
            record: if True, store (z, A1, A2) at each step

        Returns:
            A1_exit, A2_exit
        """
        dz = self.dz * direction
        D1 = self.D1 if direction == +1 else self.D1_back
        D2 = self.D2 if direction == +1 else self.D2_back

        A1 = self.A1.copy()
        A2 = self.A2.copy()
        z = 0.0 if direction == +1 else self.L

        if record:
            self.spatial_record = [(z, A1.copy(), A2.copy())]

        for step in range(self.Nz):
            # Half-step dispersion
            A1 = self._dispersion_step(A1, D1, dz / 2)
            A2 = self._dispersion_step(A2, D2, dz / 2)

            # Full-step nonlinear coupling (RK4)
            A1, A2 = self._nonlinear_step(A1, A2, z, dz)

            # Half-step dispersion
            A1 = self._dispersion_step(A1, D1, dz / 2)
            A2 = self._dispersion_step(A2, D2, dz / 2)

            z += dz

            if record:
                self.spatial_record.append((z, A1.copy(), A2.copy()))

        self.A1 = A1
        self.A2 = A2
        self.z_current = z
        self.pass_count += 1
        return A1, A2

    def _dispersion_step(self, A, D_op, dz):
        """Apply dispersion in frequency domain."""
        A_hat = np.fft.fft(A)
        A_hat *= np.exp(-1j * D_op * dz)
        return np.fft.ifft(A_hat)

    def _nonlinear_step(self, A1, A2, z, dz):
        """RK4 step for the nonlinear coupling."""
        def rhs(z_local, A1_local, A2_local):
            d = self._poling(z_local)
            if self.poling_mode == 'deff':
                phase = np.exp(1j * self.delta_k * z_local)
            else:
                phase = 1.0  # QPM handled by d(z)

            dA1 = -1j * self.kappa1 * d * A2_local * np.conj(A1_local) * np.conj(phase)
            dA2 = -1j * self.kappa2 * d * A1_local ** 2 * phase
            return dA1, dA2

        k1a, k1b = rhs(z, A1, A2)
        k2a, k2b = rhs(z + dz / 2, A1 + dz / 2 * k1a, A2 + dz / 2 * k1b)
        k3a, k3b = rhs(z + dz / 2, A1 + dz / 2 * k2a, A2 + dz / 2 * k2b)
        k4a, k4b = rhs(z + dz, A1 + dz * k3a, A2 + dz * k3b)

        A1_new = A1 + (dz / 6) * (k1a + 2 * k2a + 2 * k3a + k4a)
        A2_new = A2 + (dz / 6) * (k1b + 2 * k2b + 2 * k3b + k4b)
        return A1_new, A2_new

    def _poling(self, z):
        """Return d(z)/d₃₃."""
        if self.poling_mode == 'deff':
            return self.d_eff_ratio
        else:
            domain_index = int(abs(z) / (self.Lambda / 2))
            return 1.0 if domain_index % 2 == 0 else -1.0

    def get_manley_rowe(self):
        """Manley-Rowe invariant: ∫|A₁|²dt/ω₁ + ∫|A₂|²dt/ω₂."""
        mr1 = np.sum(np.abs(self.A1) ** 2) * self.dt / self.omega1
        mr2 = np.sum(np.abs(self.A2) ** 2) * self.dt / self.omega2
        return mr1 + mr2

    def get_conversion(self):
        """Peak |A₂|/|A₁| ratio."""
        max_a1 = np.max(np.abs(self.A1))
        max_a2 = np.max(np.abs(self.A2))
        return max_a2 / max(max_a1, 1e-30)

    def get_energy_ratio(self):
        """Energy in SH / energy in fundamental."""
        u1 = np.sum(np.abs(self.A1) ** 2) * self.dt
        u2 = np.sum(np.abs(self.A2) ** 2) * self.dt
        return u2 / max(u1, 1e-30)
