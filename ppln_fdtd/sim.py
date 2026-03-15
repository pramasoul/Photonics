"""1D FDTD simulation of SHG in PPLN — two coupled Yee grids on GPU.

Uses ΔP_NL source-term coupling: the first-order time difference of the
nonlinear polarization drives SH generation. This is the correct coupling
for the first-order Yee leapfrog scheme.

  ΔE₂ = -χ⁽²⁾·d(z)·(E₁_new² - E₁_old²) / n₂²

The entire FDTD step is a single CUDA kernel.
"""

import cupy as cp
import numpy as np
from materials import sellmeier_n, make_poling_pattern, D33, qpm_period

C = 2.998e8
MU0 = 4e-7 * np.pi
EPS0 = 8.854e-12

_KERNEL_SRC = r"""
extern "C" __global__
void fdtd_step(
    double* __restrict__ E1, double* __restrict__ H1,
    double* __restrict__ E2, double* __restrict__ H2,
    const double* __restrict__ inv_eps1,
    const double* __restrict__ inv_eps2,
    const double* __restrict__ d_z,
    double* __restrict__ mur_prev,
    double ch,
    double nl_coeff2,       // chi2 / n2^2
    double nl_coeff1,       // 2 * chi2 / n1^2
    double mur_coeff,
    double src_val,
    int src_idx,
    int cs, int ce,
    int Nz)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Save E^n before updates
    double e1_old = 0.0, e2_old = 0.0;
    if (i < Nz) {
        e1_old = E1[i];
        e2_old = E2[i];
    }

    // ── H update ──
    if (i < Nz - 1) {
        H1[i] += ch * (E1[i + 1] - E1[i]);
        H2[i] += ch * (E2[i + 1] - E2[i]);
    }
    __syncthreads();

    // ── E update (standard Yee) ──
    if (i >= 1 && i < Nz) {
        E1[i] += inv_eps1[i] * (H1[i] - H1[i - 1]);
        E2[i] += inv_eps2[i] * (H2[i] - H2[i - 1]);
    }
    __syncthreads();

    // ── Source injection ──
    if (i == src_idx) {
        E1[i] += src_val;
    }
    __syncthreads();

    // ── ΔP_NL nonlinear coupling ──
    // SHG:  ΔE₂ = -χ⁽²⁾·d·(E₁_new² - E₁_old²) / n₂²
    // Back: ΔE₁ = -2χ⁽²⁾·d·E₂_old·(E₁_new - E₁_old) / n₁²
    if (i >= cs && i < ce) {
        double d = d_z[i];
        double e1_new = E1[i];
        double dE1sq = e1_new * e1_new - e1_old * e1_old;
        double dE2E1 = e2_old * (e1_new - e1_old);

        E2[i] -= nl_coeff2 * d * dE1sq;
        E1[i] -= nl_coeff1 * d * dE2E1;
    }
    __syncthreads();

    // ── Mur ABC ──
    double mc = mur_coeff;
    if (i == 0) {
        double E1_0_old = mur_prev[0];
        double E1_1_old = mur_prev[1];
        double E2_0_old = mur_prev[4];
        double E2_1_old = mur_prev[5];
        E1[0] = E1_1_old + mc * (E1[1] - E1_0_old);
        E2[0] = E2_1_old + mc * (E2[1] - E2_0_old);
        mur_prev[0] = E1[0];
        mur_prev[1] = E1[1];
        mur_prev[4] = E2[0];
        mur_prev[5] = E2[1];
    }
    if (i == Nz - 1) {
        double E1_Nm2_old = mur_prev[2];
        double E1_Nm1_old = mur_prev[3];
        double E2_Nm2_old = mur_prev[6];
        double E2_Nm1_old = mur_prev[7];
        E1[Nz-1] = E1_Nm2_old + mc * (E1[Nz-2] - E1_Nm1_old);
        E2[Nz-1] = E2_Nm2_old + mc * (E2[Nz-2] - E2_Nm1_old);
        mur_prev[2] = E1[Nz-2];
        mur_prev[3] = E1[Nz-1];
        mur_prev[6] = E2[Nz-2];
        mur_prev[7] = E2[Nz-1];
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
        boost: float = 1.0,
        peak_intensity_W_cm2: float = 1e9,
        ppw: int = 20,
        poling_period_m: float | None = None,
    ):
        self.lambda_fund_um = lambda_fund_um
        self.T = T_celsius
        self.crystal_length = crystal_length_m
        self.courant = courant
        self.pulse_width_s = pulse_width_fs * 1e-15
        self.boost = boost
        self.peak_intensity_W_cm2 = peak_intensity_W_cm2
        self.ppw = ppw

        self.n1 = sellmeier_n(lambda_fund_um, T_celsius)
        self.n2 = sellmeier_n(lambda_fund_um / 2.0, T_celsius)
        self.omega1 = 2 * np.pi * C / (lambda_fund_um * 1e-6)
        self.omega2 = 2 * self.omega1

        lambda_min = (lambda_fund_um * 1e-6) / (2.0 * self.n2)
        self.dz = lambda_min / ppw
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

        self.R_pump = [0.0, 0.0]
        self.R_sh = [0.0, 0.0]
        self.taper_cells = 200

        self._build_eps()
        self._compute_coupling()
        self._alloc_fields()
        self.d_z = make_poling_pattern(self.z, self.Lambda, 1.0, self.crystal_mask)

        self.n_step = 0
        self._mur_prev = cp.zeros(8, dtype=cp.float64)
        self._mur_coeff = (C * self.dt - self.dz) / (C * self.dt + self.dz)

        I_W_m2 = self.peak_intensity_W_cm2 * 1e4
        self.E0 = np.sqrt(2.0 * I_W_m2 / (self.n1 * EPS0 * C))
        self.t0 = 4.0 * self.pulse_width_s

        self.probe_enabled = False
        self.probe_idx = self.crystal_end + 10
        self.probe_E1 = []
        self.probe_E2 = []
        self.probe_times = []
        max_pe = int(0.5 / (self.omega2 / (2 * np.pi) * self.dt))
        self._probe_every = max(1, min(max_pe, 32))

        self._block = 256
        self._grid = (self.Nz + self._block - 1) // self._block

    def _compute_coupling(self):
        chi2 = 2.0 * D33 * self.boost
        self._nl_coeff2 = chi2 / self.n2 ** 2
        self._nl_coeff1 = 2.0 * chi2 / self.n1 ** 2

    def _build_eps(self):
        Nz = self.Nz
        cs, ce = self.crystal_start, self.crystal_end
        eps1 = np.full(Nz, EPS0, dtype=np.float64)
        eps2 = np.full(Nz, EPS0, dtype=np.float64)
        eps1_xtal = EPS0 * self.n1 ** 2
        eps2_xtal = EPS0 * self.n2 ** 2
        eps1[cs:ce] = eps1_xtal
        eps2[cs:ce] = eps2_xtal
        for eps_arr, eps_x, R_pair in [
            (eps1, eps1_xtal, self.R_pump),
            (eps2, eps2_xtal, self.R_sh),
        ]:
            for face, R in enumerate(R_pair):
                t_len = max(1, int(self.taper_cells * (1.0 - min(R, 1.0))))
                t = np.linspace(0, np.pi / 2, t_len)
                profile = EPS0 + (eps_x - EPS0) * np.sin(t) ** 2
                if face == 0:
                    n = min(t_len, ce - cs)
                    eps_arr[cs:cs + n] = profile[:n]
                else:
                    n = min(t_len, ce - cs)
                    eps_arr[ce - n:ce] = profile[:n][::-1]
        self.eps1_host = eps1
        self.eps2_host = eps2
        self.inv_eps1 = cp.asarray(self.dt / (eps1 * self.dz))
        self.inv_eps2 = cp.asarray(self.dt / (eps2 * self.dz))

    def set_R_pump(self, left=None, right=None):
        if left is not None: self.R_pump[0] = max(0.0, min(1.0, left))
        if right is not None: self.R_pump[1] = max(0.0, min(1.0, right))
        self._build_eps()

    def set_R_sh(self, left=None, right=None):
        if left is not None: self.R_sh[0] = max(0.0, min(1.0, left))
        if right is not None: self.R_sh[1] = max(0.0, min(1.0, right))
        self._build_eps()

    def _alloc_fields(self):
        self.E1 = cp.zeros(self.Nz, dtype=cp.float64)
        self.H1 = cp.zeros(self.Nz, dtype=cp.float64)
        self.E2 = cp.zeros(self.Nz, dtype=cp.float64)
        self.H2 = cp.zeros(self.Nz, dtype=cp.float64)

    def rebuild_poling(self, Lambda):
        self.Lambda = Lambda
        self.d_z = make_poling_pattern(self.z, Lambda, 1.0, self.crystal_mask)

    def update_boost(self, boost):
        self.boost = boost
        self._compute_coupling()

    def update_temperature(self, T_celsius):
        self.T = T_celsius
        self.n1 = sellmeier_n(self.lambda_fund_um, T_celsius)
        self.n2 = sellmeier_n(self.lambda_fund_um / 2.0, T_celsius)
        self.omega1 = 2 * np.pi * C / (self.lambda_fund_um * 1e-6)
        self.omega2 = 2 * self.omega1
        self._build_eps()
        self._compute_coupling()

    def update_pulse_width(self, pw_fs):
        self.pulse_width_s = pw_fs * 1e-15
        self.t0 = 4.0 * self.pulse_width_s

    def reset(self):
        self._alloc_fields()
        self.n_step = 0
        self._mur_prev = cp.zeros(8, dtype=cp.float64)
        self.probe_E1.clear()
        self.probe_E2.clear()
        self.probe_times.clear()

    def get_energy(self):
        eps1_gpu = cp.asarray(self.eps1_host)
        eps2_gpu = cp.asarray(self.eps2_host)
        u_e = float(cp.sum(eps1_gpu * self.E1 ** 2 + eps2_gpu * self.E2 ** 2))
        u_h = float(MU0 * cp.sum(self.H1 ** 2 + self.H2 ** 2))
        return 0.5 * self.dz * (u_e + u_h)

    def step(self, n_steps=1):
        kernel = _get_kernel()
        grid, block = (self._grid,), (self._block,)
        E1, H1 = self.E1, self.H1
        E2, H2 = self.E2, self.H2
        inv_eps1, inv_eps2 = self.inv_eps1, self.inv_eps2
        d_z = self.d_z
        mp = self._mur_prev
        ch = self.ch
        nl2 = self._nl_coeff2
        nl1 = self._nl_coeff1
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
        inv_2sig2 = -1.0 / (2.0 * self.pulse_width_s ** 2)

        probe_vals = []

        for _ in range(n_steps):
            t = self.n_step * dt
            t_src = t - t0
            src = E0 * np.exp(t_src * t_src * inv_2sig2) * np.sin(omega1 * t)

            kernel(grid, block,
                   (E1, H1, E2, H2,
                    inv_eps1, inv_eps2, d_z, mp,
                    ch, nl2, nl1, mc, src,
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
    def current_time_ps(self):
        return self.n_step * self.dt * 1e12

    @property
    def ideal_qpm_period_m(self):
        return qpm_period(self.lambda_fund_um, self.T)
