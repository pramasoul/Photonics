"""1D FDTD simulation of SHG in PPLN — two coupled Yee grids on GPU.

Uses the D-field constitutive approach for energy-conserving nonlinear coupling.
The entire FDTD step (H, D, E, P_NL, source, Mur ABC) is a single CUDA kernel.
"""

import cupy as cp
import numpy as np
from materials import sellmeier_n, make_poling_pattern, D33, qpm_period

# Physical constants
C = 2.998e8
MU0 = 4e-7 * np.pi
EPS0 = 8.854e-12

# ── Fully fused CUDA kernel ───────────────────────────────────────────
# One launch does the entire FDTD step including boundaries.
_KERNEL_SRC = r"""
extern "C" __global__
void fdtd_step(
    double* __restrict__ E1, double* __restrict__ H1, double* __restrict__ D1,
    double* __restrict__ E2, double* __restrict__ H2, double* __restrict__ D2,
    const double* __restrict__ eps1, const double* __restrict__ eps2,
    const double* __restrict__ d_z,
    double* __restrict__ mur_prev,   // [E1_left, E1_right, E2_left, E2_right]
    double ch, double cd,
    double eps0_chi2,
    double mur_coeff,
    double src_val,         // precomputed source value * eps for this step
    int src_idx,
    int cs, int ce,         // crystal region
    int Nz)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // ── H update ──
    if (i < Nz - 1) {
        H1[i] += ch * (E1[i + 1] - E1[i]);
        H2[i] += ch * (E2[i + 1] - E2[i]);
    }
    __syncthreads();

    // ── D update ──
    if (i >= 1 && i < Nz) {
        D1[i] += cd * (H1[i] - H1[i - 1]);
        D2[i] += cd * (H2[i] - H2[i - 1]);
    }
    __syncthreads();

    // ── Source injection (single thread) ──
    if (i == src_idx) {
        D1[i] += src_val;
    }
    __syncthreads();

    // ── E = (D - P_NL) / eps ──
    if (i < Nz) {
        double e1p = E1[i];
        double e2p = E2[i];
        double e1n = D1[i] / eps1[i];
        double e2n = D2[i] / eps2[i];
        if (i >= cs && i < ce) {
            double d = d_z[i];
            e2n -= eps0_chi2 * d * e1p * e1p / eps2[i];
            e1n -= eps0_chi2 * d * 2.0 * e2p * e1p / eps1[i];
        }
        E1[i] = e1n;
        E2[i] = e2n;
    }
    __syncthreads();

    // ── Mur ABC (boundary threads only) ──
    double mc = mur_coeff;
    if (i == 0) {
        // Left boundary
        double mp0 = mur_prev[0];
        double mp2 = mur_prev[2];
        E1[0] = E1[1] + mc * (E1[1] - mp0);
        E2[0] = E2[1] + mc * (E2[1] - mp2);
        D1[0] = E1[0] * eps1[0];
        D2[0] = E2[0] * eps2[0];
        mur_prev[0] = E1[1];
        mur_prev[2] = E2[1];
    }
    if (i == Nz - 1) {
        // Right boundary
        double mp1 = mur_prev[1];
        double mp3 = mur_prev[3];
        E1[Nz-1] = E1[Nz-2] + mc * (E1[Nz-2] - mp1);
        E2[Nz-1] = E2[Nz-2] + mc * (E2[Nz-2] - mp3);
        D1[Nz-1] = E1[Nz-1] * eps1[Nz-1];
        D2[Nz-1] = E2[Nz-1] * eps2[Nz-1];
        mur_prev[1] = E1[Nz-2];
        mur_prev[3] = E2[Nz-2];
    }
}
"""

_kernel = None


def _get_kernel():
    global _kernel
    if _kernel is None:
        _kernel = cp.RawKernel(_KERNEL_SRC, "fdtd_step")
    return _kernel


class FDTDSimulation:
    def __init__(
        self,
        lambda_fund_um: float = 1.064,
        T_celsius: float = 25.0,
        crystal_length_m: float = 500e-6,
        courant: float = 0.5,
        pulse_width_fs: float = 200.0,
        boost: float = 50.0,
        poling_period_m: float | None = None,
    ):
        self.lambda_fund_um = lambda_fund_um
        self.T = T_celsius
        self.crystal_length = crystal_length_m
        self.courant = courant
        self.pulse_width_s = pulse_width_fs * 1e-15
        self.boost = boost

        self.n1 = sellmeier_n(lambda_fund_um, T_celsius)
        self.n2 = sellmeier_n(lambda_fund_um / 2.0, T_celsius)

        self.omega1 = 2 * np.pi * C / (lambda_fund_um * 1e-6)
        self.omega2 = 2 * self.omega1

        lambda_min = (lambda_fund_um * 1e-6) / (2.0 * self.n2)
        self.dz = lambda_min / 20.0
        self.dt = self.dz * courant / C

        margin = 0.03 * crystal_length_m
        total_length = crystal_length_m + 2 * margin
        self.Nz = int(np.ceil(total_length / self.dz))
        self.z = cp.arange(self.Nz) * self.dz

        self.crystal_start = int(np.round(margin / self.dz))
        self.crystal_end = self.crystal_start + int(np.round(crystal_length_m / self.dz))
        self.crystal_mask = cp.zeros(self.Nz, dtype=cp.float64)
        self.crystal_mask[self.crystal_start:self.crystal_end] = 1.0

        self.source_idx = 20

        if poling_period_m is None:
            poling_period_m = qpm_period(lambda_fund_um, T_celsius)
        self.Lambda = poling_period_m

        self.ch = self.dt / (MU0 * self.dz)
        self.cd = self.dt / self.dz

        self.eps1 = cp.full(self.Nz, EPS0 * self.n1 ** 2, dtype=cp.float64)
        self.eps2 = cp.full(self.Nz, EPS0 * self.n2 ** 2, dtype=cp.float64)
        self.eps1[:self.crystal_start] = EPS0
        self.eps1[self.crystal_end:] = EPS0
        self.eps2[:self.crystal_start] = EPS0
        self.eps2[self.crystal_end:] = EPS0

        self._compute_chi2()
        self._alloc_fields()
        self.d_z = make_poling_pattern(self.z, self.Lambda, 1.0, self.crystal_mask)

        self.n_step = 0
        self._mur_prev = cp.zeros(4, dtype=cp.float64)
        self._mur_coeff = (C * self.dt - self.dz) / (C * self.dt + self.dz)

        self.E0 = 1.0
        self.t0 = 4.0 * self.pulse_width_s
        self._src_eps = float(self.eps1[self.source_idx])

        # Temporal probe (opt-in, used by matplotlib version)
        self.probe_enabled = False
        self.probe_idx = self.crystal_end + 10
        self.probe_E1 = []
        self.probe_E2 = []
        self.probe_times = []
        max_pe = int(0.5 / (self.omega2 / (2 * np.pi) * self.dt))
        self._probe_every = max(1, min(max_pe, 32))

        self._block = 256
        self._grid = (self.Nz + self._block - 1) // self._block

    def _compute_chi2(self):
        FIELD_SCALE = 1.5e3
        self._chi2_eff = 2.0 * D33 * self.boost * FIELD_SCALE ** 2
        self._eps0_chi2 = EPS0 * self._chi2_eff

    def _alloc_fields(self):
        self.E1 = cp.zeros(self.Nz, dtype=cp.float64)
        self.H1 = cp.zeros(self.Nz, dtype=cp.float64)
        self.D1 = cp.zeros(self.Nz, dtype=cp.float64)
        self.E2 = cp.zeros(self.Nz, dtype=cp.float64)
        self.H2 = cp.zeros(self.Nz, dtype=cp.float64)
        self.D2 = cp.zeros(self.Nz, dtype=cp.float64)

    def rebuild_poling(self, Lambda: float):
        self.Lambda = Lambda
        self.d_z = make_poling_pattern(self.z, Lambda, 1.0, self.crystal_mask)

    def update_boost(self, boost: float):
        self.boost = boost
        self._compute_chi2()

    def update_temperature(self, T_celsius: float):
        self.T = T_celsius
        self.n1 = sellmeier_n(self.lambda_fund_um, T_celsius)
        self.n2 = sellmeier_n(self.lambda_fund_um / 2.0, T_celsius)
        self.omega1 = 2 * np.pi * C / (self.lambda_fund_um * 1e-6)
        self.omega2 = 2 * self.omega1
        self.eps1[self.crystal_start:self.crystal_end] = EPS0 * self.n1 ** 2
        self.eps2[self.crystal_start:self.crystal_end] = EPS0 * self.n2 ** 2
        self._compute_chi2()

    def update_pulse_width(self, pw_fs: float):
        self.pulse_width_s = pw_fs * 1e-15
        self.t0 = 4.0 * self.pulse_width_s

    def reset(self):
        self._alloc_fields()
        self.n_step = 0
        self._mur_prev = cp.zeros(4, dtype=cp.float64)
        self.probe_E1.clear()
        self.probe_E2.clear()
        self.probe_times.clear()

    def step(self, n_steps: int = 1):
        """Run n_steps FDTD updates.

        Each step is a single CUDA kernel launch.  The only host work per step
        is computing the source value (two transcendentals) and the Python
        loop counter.  Probe recording syncs every _probe_every steps.
        """
        kernel = _get_kernel()
        grid, block = (self._grid,), (self._block,)
        E1, H1, D1 = self.E1, self.H1, self.D1
        E2, H2, D2 = self.E2, self.H2, self.D2
        eps1, eps2 = self.eps1, self.eps2
        d_z = self.d_z
        mp = self._mur_prev
        ch, cd = self.ch, self.cd
        eps0_chi2 = self._eps0_chi2
        mc = self._mur_coeff
        si = np.int32(self.source_idx)
        cs = np.int32(self.crystal_start)
        ce = np.int32(self.crystal_end)
        nz = np.int32(self.Nz)
        pi = self.probe_idx
        pe = self._probe_every
        dt = self.dt
        omega1 = self.omega1
        E0 = self.E0
        t0 = self.t0
        src_eps = self._src_eps
        inv_2sig2 = -1.0 / (2.0 * self.pulse_width_s ** 2)

        probe_vals = []

        for _ in range(n_steps):
            t = self.n_step * dt

            # Source value (host math — two transcendentals, ~ns)
            t_src = t - t0
            src_d = E0 * np.exp(t_src * t_src * inv_2sig2) * np.sin(omega1 * t) * src_eps

            # Single kernel launch does everything
            kernel(grid, block,
                   (E1, H1, D1, E2, H2, D2,
                    eps1, eps2, d_z, mp,
                    ch, cd, eps0_chi2, mc, src_d,
                    si, cs, ce, nz))

            self.n_step += 1

            if self.probe_enabled and self.n_step % pe == 0:
                probe_vals.append((float(E1[pi]), float(E2[pi]), self.n_step * dt))

        if probe_vals:
            for v1, v2, tp in probe_vals:
                self.probe_E1.append(v1)
                self.probe_E2.append(v2)
                self.probe_times.append(tp)

    def get_fields(self):
        return self.E1.get(), self.E2.get()

    def get_z_host(self):
        return (self.z * 1e6).get()

    def get_spectrum(self):
        if len(self.probe_E1) < 64:
            return None, None, None
        e1 = np.array(self.probe_E1)
        e2 = np.array(self.probe_E2)
        N = len(e1)
        dt_probe = self._probe_every * self.dt
        freqs = np.fft.rfftfreq(N, d=dt_probe) * 1e-12
        spec1 = np.abs(np.fft.rfft(e1))
        spec2 = np.abs(np.fft.rfft(e2))
        return freqs, spec1, spec2

    @property
    def current_time_ps(self) -> float:
        return self.n_step * self.dt * 1e12

    @property
    def ideal_qpm_period_m(self) -> float:
        return qpm_period(self.lambda_fund_um, self.T)
