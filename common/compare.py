"""Cross-comparison tool: run both FDTD and SSFM with the same parameters,
extract envelopes from FDTD, and plot side-by-side.

Usage:
    cd ppln_fdtd && pixi run python ../common/compare.py
    cd ppln_fdtd && pixi run python ../common/compare.py --ppw 100 --crystal-length 500
"""

import sys
import os
import argparse
import time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'ppln_fdtd'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'ppln_ssfm'))

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from common.materials import (
    extract_envelope, fdtd_to_temporal, group_velocity,
    sellmeier_n, qpm_period, C,
)


def run_comparison(
    lambda_fund_um=1.064,
    crystal_length_um=500,
    peak_intensity=1e9,
    pulse_width_fs=200,
    temperature=25.0,
    ppw=100,
    boost=1.0,
):
    """Run both engines and return comparable results."""

    params = dict(
        lambda_fund_um=lambda_fund_um,
        crystal_length_um=crystal_length_um,
        peak_intensity=peak_intensity,
        pulse_width_fs=pulse_width_fs,
        temperature=temperature,
        ppw=ppw,
        boost=boost,
    )

    n1 = sellmeier_n(lambda_fund_um, temperature)
    n2 = sellmeier_n(lambda_fund_um / 2, temperature)
    omega1 = 2 * np.pi * C / (lambda_fund_um * 1e-6)
    omega2 = 2 * omega1
    vg1 = group_velocity(lambda_fund_um, temperature)
    vg2 = group_velocity(lambda_fund_um / 2, temperature)

    results = {}

    # ── SSFM ──
    print('Running SSFM...', end=' ', flush=True)
    from ppln_ssfm.sim import SSFMSimulation  # noqa

    t0 = time.perf_counter()
    ssfm = SSFMSimulation(
        lambda_fund_um=lambda_fund_um,
        T_celsius=temperature,
        crystal_length_um=crystal_length_um,
        peak_intensity_W_cm2=peak_intensity,
        pulse_width_fs=pulse_width_fs,
        boost=boost,
    )
    ssfm.propagate_pass()
    t_ssfm = time.perf_counter() - t0

    results['ssfm'] = {
        't_ps': ssfm.t_grid * 1e12,
        'I1': np.abs(ssfm.A1) ** 2,
        'I2': np.abs(ssfm.A2) ** 2,
        'A1': ssfm.A1,
        'A2': ssfm.A2,
        'dt': ssfm.dt,
        'conv': ssfm.get_conversion(),
        'mr': ssfm.get_manley_rowe(),
        'time_s': t_ssfm,
        'Nt': ssfm.Nt,
    }
    print(f'{t_ssfm*1e3:.1f} ms, conv={ssfm.get_conversion():.3f}')

    # ── FDTD ──
    print(f'Running FDTD (ppw={ppw})...', end=' ', flush=True)
    from ppln_fdtd.sim import FDTDSimulation  # noqa

    t0 = time.perf_counter()
    fdtd = FDTDSimulation(
        lambda_fund_um=lambda_fund_um,
        T_celsius=temperature,
        crystal_length_m=crystal_length_um * 1e-6,
        peak_intensity_W_cm2=peak_intensity,
        pulse_width_fs=pulse_width_fs,
        boost=boost,
        ppw=ppw,
    )

    # Run until pulse exits crystal
    margin_steps = int((fdtd.crystal_start - fdtd.source_idx) / fdtd.courant)
    xtal_steps = int((fdtd.crystal_end - fdtd.crystal_start) / (fdtd.courant / fdtd.n1))
    source_steps = int(fdtd.t0 / fdtd.dt)
    total = source_steps + margin_steps + xtal_steps + int(xtal_steps * 0.1)
    fdtd.step(total)
    t_fdtd = time.perf_counter() - t0

    E1, E2 = fdtd.get_fields()
    z_m = fdtd.get_z_host() * 1e-6

    # Extract envelopes
    A1_fdtd = extract_envelope(E1, z_m, omega1, n1, fdtd.dz)
    A2_fdtd = extract_envelope(E2, z_m, omega2, n2, fdtd.dz)

    # Convert to temporal (co-moving at vg1)
    t1, At1 = fdtd_to_temporal(A1_fdtd, vg1, fdtd.dz)
    t2, At2 = fdtd_to_temporal(A2_fdtd, vg2, fdtd.dz)

    conv_fdtd = np.max(np.abs(A2_fdtd)) / max(np.max(np.abs(A1_fdtd)), 1e-30)

    results['fdtd'] = {
        't1_ps': t1 * 1e12,
        't2_ps': t2 * 1e12,
        'I1': np.abs(At1) ** 2,
        'I2': np.abs(At2) ** 2,
        'A1': At1,
        'A2': At2,
        'dt1': t1[1] - t1[0] if len(t1) > 1 else 1e-15,
        'dt2': t2[1] - t2[0] if len(t2) > 1 else 1e-15,
        'conv': conv_fdtd,
        'time_s': t_fdtd,
        'steps': total,
        'Nz': fdtd.Nz,
    }
    print(f'{t_fdtd:.1f} s, {total:,} steps, conv={conv_fdtd:.3f}')

    results['params'] = params
    return results


def plot_comparison(results):
    """Four-panel comparison plot."""
    ssfm = results['ssfm']
    fdtd = results['fdtd']
    p = results['params']

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    fig.suptitle(
        f"SSFM vs FDTD: {p['crystal_length_um']:.0f} μm PPLN, "
        f"{p['peak_intensity']:.0e} W/cm², {p['pulse_width_fs']:.0f} fs, ppw={p['ppw']}",
        fontsize=11,
    )

    # ── Top left: SSFM temporal ──
    ax = axes[0, 0]
    I_sc = max(np.max(ssfm['I1']), np.max(ssfm['I2']), 1e-30)
    ax.plot(ssfm['t_ps'], ssfm['I1'] / I_sc, 'r-', lw=1.5, label='pump')
    ax.plot(ssfm['t_ps'], ssfm['I2'] / I_sc, 'b-', lw=1.5, label='SH')
    ax.set_title(f"SSFM ({ssfm['time_s']*1e3:.1f} ms) — |A₂|/|A₁| = {ssfm['conv']:.1%}")
    ax.set_xlabel('Time (ps)')
    ax.set_ylabel('Intensity (norm.)')
    ax.legend(fontsize=8)

    # ── Top right: FDTD temporal (envelope) ──
    ax = axes[0, 1]
    I_sc_f = max(np.max(fdtd['I1']), np.max(fdtd['I2']), 1e-30)
    # Decimate for plotting
    dec1 = max(1, len(fdtd['t1_ps']) // 2000)
    dec2 = max(1, len(fdtd['t2_ps']) // 2000)
    ax.plot(fdtd['t1_ps'][::dec1], fdtd['I1'][::dec1] / I_sc_f, 'r-', lw=1.5, label='pump')
    ax.plot(fdtd['t2_ps'][::dec2], fdtd['I2'][::dec2] / I_sc_f, 'b-', lw=1.5, label='SH')
    ax.set_title(f"FDTD ppw={p['ppw']} ({fdtd['time_s']:.1f} s) — |A₂|/|A₁| = {fdtd['conv']:.1%}")
    ax.set_xlabel('Time (ps)')
    ax.set_ylabel('Intensity (norm.)')
    ax.legend(fontsize=8)
    # Match x-limits to SSFM
    ax.set_xlim(ssfm['t_ps'][0], ssfm['t_ps'][-1])

    # ── Bottom left: SSFM spectrum ──
    ax = axes[1, 0]
    Nfft = ssfm['Nt'] * 4
    freq_s = np.fft.fftshift(np.fft.fftfreq(Nfft, d=ssfm['dt'])) * 1e-12
    S1s = np.abs(np.fft.fftshift(np.fft.fft(ssfm['A1'], n=Nfft))) ** 2
    S2s = np.abs(np.fft.fftshift(np.fft.fft(ssfm['A2'], n=Nfft))) ** 2
    Ss_sc = max(np.max(S1s), np.max(S2s), 1e-30)
    ax.semilogy(freq_s, S1s / Ss_sc + 1e-10, 'r-', lw=1.5, label='pump')
    ax.semilogy(freq_s, S2s / Ss_sc + 1e-10, 'b-', lw=1.5, label='SH')
    ax.set_title('SSFM spectrum')
    ax.set_xlabel('Freq offset (THz)')
    ax.set_ylabel('Power (log)')
    ax.set_xlim(-15, 15)
    ax.set_ylim(1e-6, 2)
    ax.legend(fontsize=8)

    # ── Bottom right: FDTD spectrum (from envelope) ──
    ax = axes[1, 1]
    # Use same FFT size as SSFM for comparable resolution
    Nfft_f = min(len(fdtd['A1']), 8192)
    freq_f = np.fft.fftshift(np.fft.fftfreq(Nfft_f, d=abs(fdtd['dt1']))) * 1e-12
    S1f = np.abs(np.fft.fftshift(np.fft.fft(fdtd['A1'][:Nfft_f], n=Nfft_f))) ** 2
    S2f = np.abs(np.fft.fftshift(np.fft.fft(fdtd['A2'][:Nfft_f], n=Nfft_f))) ** 2
    Sf_sc = max(np.max(S1f), np.max(S2f), 1e-30)
    ax.semilogy(freq_f, S1f / Sf_sc + 1e-10, 'r-', lw=1.5, label='pump')
    ax.semilogy(freq_f, S2f / Sf_sc + 1e-10, 'b-', lw=1.5, label='SH')
    ax.set_title('FDTD spectrum (from envelope)')
    ax.set_xlabel('Freq offset (THz)')
    ax.set_ylabel('Power (log)')
    ax.set_xlim(-15, 15)
    ax.set_ylim(1e-6, 2)
    ax.legend(fontsize=8)

    fig.tight_layout()
    return fig


def main():
    parser = argparse.ArgumentParser(description='FDTD vs SSFM cross-comparison')
    parser.add_argument('--crystal-length', type=float, default=500, help='Crystal length (μm)')
    parser.add_argument('--peak-intensity', type=float, default=1e9, help='Peak intensity (W/cm²)')
    parser.add_argument('--pulse-width', type=float, default=200, help='Pulse width (fs)')
    parser.add_argument('--ppw', type=int, default=100, help='FDTD points per wavelength')
    parser.add_argument('--temperature', type=float, default=25, help='Temperature (°C)')
    parser.add_argument('--boost', type=float, default=1.0, help='χ² boost')
    parser.add_argument('--save', type=str, default=None, help='Save figure to file')
    args = parser.parse_args()

    results = run_comparison(
        crystal_length_um=args.crystal_length,
        peak_intensity=args.peak_intensity,
        pulse_width_fs=args.pulse_width,
        ppw=args.ppw,
        temperature=args.temperature,
        boost=args.boost,
    )

    fig = plot_comparison(results)

    if args.save:
        fig.savefig(args.save, dpi=150, bbox_inches='tight')
        print(f'Saved to {args.save}')
    else:
        plt.show()


if __name__ == '__main__':
    main()
