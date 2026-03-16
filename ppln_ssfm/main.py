"""PPLN SSFM Explorer — matplotlib interactive viewer.

Two-panel display: temporal profiles and spectra at crystal output.
Sliders for poling period, temperature, intensity, pulse width, crystal length.
Recomputes full crystal pass on each parameter change (~1 ms).
"""

import sys
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from sim import SSFMSimulation
from common.materials import qpm_period, C


def load_config():
    """Load default.yaml if present, merge with CLI args."""
    defaults = dict(
        lambda_fund_um=1.064,
        temperature_C=25.0,
        crystal_length_um=500.0,
        peak_intensity_W_cm2=1e9,
        pulse_width_fs=200.0,
        boost=1.0,
        Nt=512,
        T_window_ps=2.0,
        dz_um=2.0,
        poling_mode='deff',
        poling_period_um=None,
    )

    # Try loading YAML
    yaml_path = os.path.join(os.path.dirname(__file__), 'default.yaml')
    if os.path.exists(yaml_path):
        import yaml
        with open(yaml_path) as f:
            file_cfg = yaml.safe_load(f) or {}
        for k, v in file_cfg.items():
            if k in defaults and v is not None:
                dv = defaults[k]
                if isinstance(dv, float) and not isinstance(v, (int, float)):
                    v = float(v)
                elif isinstance(dv, int) and not isinstance(v, int):
                    v = int(v)
                defaults[k] = v

    return defaults


def build_sim(cfg):
    """Construct SSFMSimulation from config dict."""
    poling_m = cfg['poling_period_um']
    return SSFMSimulation(
        lambda_fund_um=cfg['lambda_fund_um'],
        T_celsius=cfg['temperature_C'],
        crystal_length_um=cfg['crystal_length_um'],
        peak_intensity_W_cm2=cfg['peak_intensity_W_cm2'],
        pulse_width_fs=cfg['pulse_width_fs'],
        boost=cfg['boost'],
        Nt=cfg['Nt'],
        T_window_ps=cfg['T_window_ps'],
        dz_um=cfg['dz_um'],
        poling_mode=cfg['poling_mode'],
        poling_period_um=poling_m,
    )


def main():
    cfg = load_config()
    sim = build_sim(cfg)

    # Run initial pass
    sim.propagate_pass()

    # ── Figure setup ──
    fig = plt.figure(figsize=(12, 7))
    fig.canvas.manager.set_window_title('PPLN SSFM Explorer')
    fig.subplots_adjust(left=0.08, right=0.95, top=0.93, bottom=0.38, hspace=0.35)

    ax_time = fig.add_subplot(2, 1, 1)
    ax_spec = fig.add_subplot(2, 1, 2)

    # ── Temporal panel ──
    t_ps = sim.t_grid * 1e12
    I1 = np.abs(sim.A1) ** 2
    I2 = np.abs(sim.A2) ** 2
    I_scale = max(np.max(I1), 1e-30)

    line_t1, = ax_time.plot(t_ps, I1 / I_scale, 'r-', lw=1.5, label='$|A_1|^2$ ($\\omega$)')
    line_t2, = ax_time.plot(t_ps, I2 / I_scale, 'b-', lw=1.5, label='$|A_2|^2$ ($2\\omega$)')
    ax_time.set_xlabel('Time (ps)')
    ax_time.set_ylabel('Intensity (norm.)')
    ax_time.set_title('Temporal profiles at crystal output')
    ax_time.legend(loc='upper right', fontsize=8)
    ax_time.set_xlim(t_ps[0], t_ps[-1])

    # ── Spectral panel ──
    freq_THz = np.fft.fftshift(np.fft.fftfreq(sim.Nt, d=sim.dt)) * 1e-12
    # Convert to wavelength offset from carrier
    omega_offset = np.fft.fftshift(sim.omega_grid)
    lambda1_nm = 2 * np.pi * C / (sim.omega1 + omega_offset) * 1e9
    lambda2_nm = 2 * np.pi * C / (sim.omega2 + omega_offset) * 1e9

    S1 = np.abs(np.fft.fftshift(np.fft.fft(sim.A1))) ** 2
    S2 = np.abs(np.fft.fftshift(np.fft.fft(sim.A2))) ** 2
    S_scale = max(np.max(S1), 1e-30)

    line_s1, = ax_spec.semilogy(freq_THz, S1 / S_scale + 1e-10, 'r-', lw=1.5, label='$\\omega$ spectrum')
    line_s2, = ax_spec.semilogy(freq_THz, S2 / S_scale + 1e-10, 'b-', lw=1.5, label='$2\\omega$ spectrum')
    ax_spec.set_xlabel('Frequency offset (THz)')
    ax_spec.set_ylabel('Spectral power (norm., log)')
    ax_spec.set_title('Spectra at crystal output')
    ax_spec.legend(loc='upper right', fontsize=8)
    ax_spec.set_xlim(-15, 15)
    ax_spec.set_ylim(1e-6, 2)

    # ── Info text ──
    info_text = ax_time.text(
        0.01, 0.95, '', transform=ax_time.transAxes,
        fontsize=7, verticalalignment='top', fontfamily='monospace',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7),
    )

    def update_info():
        conv = sim.get_conversion()
        mr = sim.get_manley_rowe()
        Lambda_qpm = qpm_period(sim.lambda_fund_um, sim.T) * 1e6
        info_text.set_text(
            f'|A₂|/|A₁| = {conv:.3f}  ({conv*100:.1f}%)\n'
            f'Λ = {sim.Lambda*1e6:.2f} μm  Λ_QPM = {Lambda_qpm:.2f} μm\n'
            f'T = {sim.T:.0f}°C  I = {sim.peak_intensity:.0e} W/cm²\n'
            f'L = {sim.L*1e6:.0f} μm  τ = {sim.pulse_width_s*1e15:.0f} fs\n'
            f'MR = {mr:.4e}  Nz = {sim.Nz}'
        )

    update_info()

    # ── Sliders ──
    slider_color = 'lightgoldenrodyellow'
    Lambda_qpm_um = qpm_period(cfg['lambda_fund_um'], cfg['temperature_C']) * 1e6

    ax_lam = fig.add_axes([0.15, 0.24, 0.65, 0.025])
    sl_lam = Slider(ax_lam, 'Λ (μm)', Lambda_qpm_um * 0.5, Lambda_qpm_um * 2.0,
                    valinit=sim.Lambda * 1e6, valstep=0.01, color=slider_color)

    ax_temp = fig.add_axes([0.15, 0.20, 0.65, 0.025])
    sl_temp = Slider(ax_temp, 'T (°C)', 20, 200, valinit=sim.T, valstep=1, color=slider_color)

    ax_int = fig.add_axes([0.15, 0.16, 0.65, 0.025])
    sl_int = Slider(ax_int, 'log₁₀(I)', 6, 11,
                    valinit=np.log10(sim.peak_intensity), valstep=0.1, color=slider_color)

    ax_pw = fig.add_axes([0.15, 0.12, 0.65, 0.025])
    sl_pw = Slider(ax_pw, 'Pulse (fs)', 50, 5000, valinit=sim.pulse_width_s * 1e15,
                   valstep=10, color=slider_color)

    ax_cl = fig.add_axes([0.15, 0.08, 0.65, 0.025])
    sl_cl = Slider(ax_cl, 'L (μm)', 50, 5000, valinit=sim.L * 1e6,
                   valstep=10, color=slider_color)

    ax_reset = fig.add_axes([0.85, 0.20, 0.1, 0.03])
    btn_reset = Button(ax_reset, 'Reset', color='lightsalmon')

    def rerun(_=None):
        nonlocal sim
        cfg['poling_period_um'] = sl_lam.val
        cfg['temperature_C'] = sl_temp.val
        cfg['peak_intensity_W_cm2'] = 10 ** sl_int.val
        cfg['pulse_width_fs'] = sl_pw.val
        cfg['crystal_length_um'] = sl_cl.val

        sim = build_sim(cfg)
        sim.propagate_pass()

        I1 = np.abs(sim.A1) ** 2
        I2 = np.abs(sim.A2) ** 2
        I_sc = max(np.max(I1), 1e-30)

        t_ps = sim.t_grid * 1e12
        line_t1.set_data(t_ps, I1 / I_sc)
        line_t2.set_data(t_ps, I2 / I_sc)
        ax_time.set_xlim(t_ps[0], t_ps[-1])
        ax_time.set_ylim(0, 1.15)

        S1 = np.abs(np.fft.fftshift(np.fft.fft(sim.A1))) ** 2
        S2 = np.abs(np.fft.fftshift(np.fft.fft(sim.A2))) ** 2
        S_sc = max(np.max(S1), 1e-30)
        freq = np.fft.fftshift(np.fft.fftfreq(sim.Nt, d=sim.dt)) * 1e-12
        line_s1.set_data(freq, S1 / S_sc + 1e-10)
        line_s2.set_data(freq, S2 / S_sc + 1e-10)
        ax_spec.set_xlim(-15, 15)

        update_info()
        fig.canvas.draw_idle()

    sl_lam.on_changed(rerun)
    sl_temp.on_changed(rerun)
    sl_int.on_changed(rerun)
    sl_pw.on_changed(rerun)
    sl_cl.on_changed(rerun)

    def on_reset(_):
        sl_lam.set_val(Lambda_qpm_um)
        sl_temp.set_val(25)
        sl_int.set_val(9)
        sl_pw.set_val(200)
        sl_cl.set_val(500)
        rerun()

    btn_reset.on_clicked(on_reset)

    plt.show()


if __name__ == '__main__':
    main()
