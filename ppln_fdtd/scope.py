"""Matplotlib scope — displays snapshots saved by the GL simulation.

Usage:
    pixi run scope                  # display latest snapshot
    pixi run scope snapshot.npz     # display specific file
    pixi run scope --watch          # auto-reload on new snapshots
    pixi run scope -v               # display with SSFM overlay
    Press 'v' in the figure window to toggle the overlay.
"""

import sys
import os
import time
import argparse
import numpy as np
import matplotlib.pyplot as plt
from materials import coherence_length, C

# Import SSFMSimulation from sister project (avoid name conflict with local sim.py)
try:
    import importlib.util as _ilu
    _ssfm_spec = _ilu.spec_from_file_location(
        "ssfm_sim", os.path.join(os.path.dirname(__file__), '..', 'ppln_ssfm', 'sim.py'))
    _ssfm_mod = _ilu.module_from_spec(_ssfm_spec)
    _ssfm_spec.loader.exec_module(_ssfm_mod)
    SSFMSimulation = _ssfm_mod.SSFMSimulation
    _HAS_SSFM = True
except Exception:
    _HAS_SSFM = False

DEFAULT_PATH = os.path.join(os.path.dirname(__file__), 'snapshot.npz')


def load_snapshot(path):
    d = np.load(path, allow_pickle=True)
    return {k: d[k] for k in d.files}


def compute_ssfm_overlay(snap):
    """Run SSFM with snapshot parameters and map envelopes to FDTD spatial grid.

    Returns (A1_overlay, A2_overlay) arrays in V/m on the snapshot z-grid,
    directly comparable to the Hilbert envelope of the FDTD field.
    """
    if not _HAS_SSFM:
        z_um = snap['z_um']
        return np.zeros(len(z_um)), np.zeros(len(z_um))

    from scipy.signal import hilbert

    E1 = snap['E1']
    z_um = snap['z_um']
    dz = float(snap['dz'])
    cs = int(snap['crystal_start'])
    ce = int(snap['crystal_end'])
    T = float(snap['T_celsius'])
    Lambda_um = float(snap['Lambda_um'])
    boost = float(snap['boost'])
    pw_fs = float(snap['pulse_width_fs'])
    peak_I = float(snap.get('peak_intensity', 1e9))
    lambda_fund_um = float(snap.get('lambda_fund_um', 1.064))

    z_m = z_um * 1e-6
    z_entry = cs * dz
    z_exit = ce * dz
    L = z_exit - z_entry
    crystal_length_um = L * 1e6

    # Run SSFM with matching parameters
    ssfm = SSFMSimulation(
        lambda_fund_um=lambda_fund_um,
        T_celsius=T,
        crystal_length_um=crystal_length_um,
        peak_intensity_W_cm2=peak_I,
        pulse_width_fs=pw_fs,
        poling_period_um=Lambda_um,
        boost=boost,
    )
    ssfm.propagate_pass(record=True)

    # Find FDTD pulse center from fundamental Hilbert envelope
    env1 = np.abs(hilbert(E1))
    if np.max(env1) < 1e-30:
        return np.zeros(len(z_m)), np.zeros(len(z_m))
    z_peak = z_m[np.argmax(env1)]

    vg1 = ssfm.vg1

    # Time since pulse center entered crystal (via empirical z_peak)
    if z_peak > z_exit:
        # Pulse has exited: crystal traverse + vacuum flight
        delta_t = L / vg1 + (z_peak - z_exit) / C
    elif z_peak >= z_entry:
        # Pulse is inside crystal
        delta_t = (z_peak - z_entry) / vg1
    else:
        # Pulse hasn't entered crystal yet
        return np.zeros(len(z_m)), np.zeros(len(z_m))

    A1_overlay = np.zeros(len(z_m))
    A2_overlay = np.zeros(len(z_m))
    t_grid = ssfm.t_grid

    # --- Crystal region: interpolate from spatial_record ---
    mask_xtal = (z_m >= z_entry) & (z_m <= z_exit)
    if np.any(mask_xtal):
        sr = ssfm.spatial_record
        sr_z = np.array([r[0] for r in sr])  # propagation distances
        sr_z_fdtd = z_entry + sr_z

        sr_A1 = np.zeros(len(sr))
        sr_A2 = np.zeros(len(sr))
        for k, (zk, a1k, a2k) in enumerate(sr):
            # Retarded time: account for crystal propagation at vg1
            # and (if pulse exited) vacuum propagation at c
            if z_peak > z_exit:
                tau = (z_peak - z_exit) / C + (z_exit - z_entry - zk) / vg1
            else:
                tau = (z_peak - z_entry - zk) / vg1
            sr_A1[k] = np.interp(tau, t_grid, np.abs(a1k), left=0, right=0)
            sr_A2[k] = np.interp(tau, t_grid, np.abs(a2k), left=0, right=0)

        A1_overlay[mask_xtal] = np.interp(z_m[mask_xtal], sr_z_fdtd, sr_A1)
        A2_overlay[mask_xtal] = np.interp(z_m[mask_xtal], sr_z_fdtd, sr_A2)

    # --- Vacuum after crystal: use exit envelope ---
    mask_post = z_m > z_exit
    if np.any(mask_post):
        tau = delta_t - L / vg1 - (z_m[mask_post] - z_exit) / C
        A1_overlay[mask_post] = np.interp(tau, t_grid, np.abs(ssfm.A1), left=0, right=0)
        A2_overlay[mask_post] = np.interp(tau, t_grid, np.abs(ssfm.A2), left=0, right=0)

    return A1_overlay, A2_overlay


def plot_snapshot(snap, fig=None, overlay=False):
    """Create or update a 4-panel matplotlib figure from snapshot data."""
    E1 = snap['E1']
    E2 = snap['E2']
    z = snap['z_um']
    d_z = snap['d_z']
    fft1 = snap['fft1']
    fft2 = snap['fft2']
    fft_k = snap['fft_k']
    cs = int(snap['crystal_start'])
    ce = int(snap['crystal_end'])
    t_ps = float(snap['t_ps'])
    Lambda = float(snap['Lambda_um'])
    T = float(snap['T_celsius'])
    boost = float(snap['boost'])
    n1 = float(snap['n1'])
    n2 = float(snap['n2'])
    pw_fs = float(snap['pulse_width_fs'])
    R_pump = snap['R_pump']
    R_sh = snap['R_sh']

    L_coh = coherence_length(1.064, T)
    Lambda_qpm = 2 * L_coh * 1e6
    dk = np.pi / L_coh - 2 * np.pi / (Lambda * 1e-6) if Lambda > 0 else 0

    if fig is None:
        fig = plt.figure(figsize=(14, 9))
        fig.canvas.manager.set_window_title('PPLN FDTD Scope')

    fig.clf()
    fig.subplots_adjust(left=0.08, right=0.95, top=0.93, bottom=0.07, hspace=0.4)

    # ── Panel 1: E-fields ──
    ax1 = fig.add_subplot(4, 1, 1)
    dec = max(1, len(z) // 3000)
    ax1.plot(z[::dec], E1[::dec], 'r-', lw=0.4, alpha=0.6, label='E₁ (ω)')
    ax1.plot(z[::dec], E2[::dec], 'b-', lw=0.4, alpha=0.6, label='E₂ (2ω)')
    # Envelope
    from scipy.signal import hilbert
    if np.any(E1 != 0):
        env1 = np.abs(hilbert(E1[::dec]))
        ax1.plot(z[::dec], env1, 'r-', lw=1.2, label='|E₁| env')
    if np.any(E2 != 0):
        env2 = np.abs(hilbert(E2[::dec]))
        ax1.plot(z[::dec], env2, 'b-', lw=1.2, label='|E₂| env')

    # SSFM overlay
    if overlay:
        A1_ov, A2_ov = compute_ssfm_overlay(snap)
        dec_ov = max(1, len(z) // 2000)
        if np.max(A1_ov) > 0:
            ax1.plot(z[::dec_ov], A1_ov[::dec_ov], 'r--', lw=1.5, alpha=0.8,
                     label='SSFM ω')
        if np.max(A2_ov) > 0:
            ax1.plot(z[::dec_ov], A2_ov[::dec_ov], 'b--', lw=1.5, alpha=0.8,
                     label='SSFM 2ω')

    ax1.axvspan(z[cs], z[ce - 1], alpha=0.06, color='gray')
    ax1.set_ylabel('E-field')
    title = f't = {t_ps:.2f} ps'
    if overlay:
        title += '  [SSFM overlay]'
    ax1.set_title(title)
    ax1.legend(loc='upper right', fontsize=7, ncol=3)

    # ── Panel 2: Poling ──
    ax2 = fig.add_subplot(4, 1, 2, sharex=ax1)
    crystal_z = z[cs:ce]
    crystal_d = d_z[cs:ce]
    dp = max(1, len(crystal_z) // 3000)
    ax2.fill_between(crystal_z[::dp], crystal_d[::dp], alpha=0.3,
                     where=crystal_d[::dp] > 0, color='red', interpolate=True)
    ax2.fill_between(crystal_z[::dp], crystal_d[::dp], alpha=0.3,
                     where=crystal_d[::dp] < 0, color='blue', interpolate=True)
    ax2.plot(crystal_z[::dp], crystal_d[::dp], 'k-', lw=0.3)
    ax2.set_ylim(-1.5, 1.5)
    ax2.set_ylabel('d(z)/d₃₃')
    ax2.set_xlabel('z (μm)')

    # ── Panel 3: Spatial power spectrum ──
    ax3 = fig.add_subplot(4, 1, 3)
    power1 = fft1 ** 2
    power2 = fft2 ** 2
    k_um = fft_k * 1e-6  # cycles/m → cycles/μm
    k1 = n1 / 1.064  # expected k in cycles/μm
    k2 = n2 / 0.532
    k_max = k2 * 1.5
    mask = k_um <= k_max
    ax3.semilogy(k_um[mask], power1[mask], 'r-', lw=0.8, label='ω')
    ax3.semilogy(k_um[mask], power2[mask], 'b-', lw=0.8, label='2ω')
    ax3.axvline(k1, color='r', ls='--', alpha=0.3, label=f'k₁={k1:.3f}/μm')
    ax3.axvline(k2, color='b', ls='--', alpha=0.3, label=f'k₂={k2:.3f}/μm')
    ax3.set_xlabel('Spatial frequency (cycles/μm)')
    ax3.set_ylabel('|FFT|²')
    ax3.legend(loc='upper right', fontsize=7)

    # ── Panel 4: Parameters (matching terminal status format) ──
    ax4 = fig.add_subplot(4, 1, 4)
    ax4.axis('off')

    # Extract optional fields (may be absent in old snapshots)
    ppw = snap.get('ppw', '?')
    peak_I = snap.get('peak_intensity', '?')
    n_step = snap.get('n_step', '?')
    E0 = snap.get('E0', '?')
    energy = snap.get('energy', '?')
    mr = snap.get('mr', '?')
    n_domains = int(float(snap.get('dz', 1e-9)) * (ce - cs) / (Lambda * 1e-6 / 2)) if Lambda > 0 else '?'

    lines = []
    lines.append(f"t = {t_ps:.2f} ps     total = {n_step}     ppw = {ppw}")
    lines.append(f"Λ = {Lambda:.2f} μm      Λ_QPM = {Lambda_qpm:.2f} μm     Δk = {dk * 1e-6:.1f} /mm     domains = {n_domains}")
    lines.append(f"T = {T:.0f} °C       boost = {boost:.0f}×            pulse = {pw_fs:.0f} fs     I = {peak_I} W/cm²")
    lines.append(f"E₀ = {E0} V/m    max|E₁| = {np.max(np.abs(E1)):.2e}   max|E₂| = {np.max(np.abs(E2)):.2e}")
    lines.append(f"R_pump = {R_pump}    R_sh = {R_sh}")
    lines.append(f"energy = {energy}     MR = {mr}")

    ax4.text(0.02, 0.95, '\n'.join(lines), transform=ax4.transAxes, fontsize=8,
             fontfamily='monospace', verticalalignment='top')

    return fig


def main():
    parser = argparse.ArgumentParser(description='PPLN FDTD Scope — snapshot viewer')
    parser.add_argument('file', nargs='?', default=DEFAULT_PATH,
                        help='Snapshot .npz file')
    parser.add_argument('--watch', '-w', action='store_true',
                        help='Auto-reload when snapshot file changes')
    parser.add_argument('--overlay', '-v', action='store_true',
                        help='Show SSFM envelope overlay (toggle with v key)')
    args = parser.parse_args()

    if not os.path.exists(args.file):
        print(f"No snapshot found at {args.file}")
        print("Press 's' in the GL simulation to save one.")
        sys.exit(1)

    show_overlay = args.overlay
    snap = load_snapshot(args.file)
    fig = plot_snapshot(snap, overlay=show_overlay)

    def on_key(event):
        nonlocal show_overlay, snap
        if event.key == 'v':
            show_overlay = not show_overlay
            plot_snapshot(snap, fig, overlay=show_overlay)
            fig.canvas.draw_idle()

    fig.canvas.mpl_connect('key_press_event', on_key)

    if args.watch:
        last_mtime = os.path.getmtime(args.file)
        plt.ion()
        plt.show(block=False)
        print("Watching for snapshot updates... (Ctrl+C to stop)")
        print("Press 'v' in the figure window to toggle SSFM overlay")
        try:
            while True:
                # Poll without raising window — avoid plt.pause() which grabs focus
                fig.canvas.flush_events()
                time.sleep(0.3)
                if not plt.fignum_exists(fig.number):
                    break
                mtime = os.path.getmtime(args.file)
                if mtime != last_mtime:
                    last_mtime = mtime
                    time.sleep(0.1)
                    try:
                        snap = load_snapshot(args.file)
                        plot_snapshot(snap, fig, overlay=show_overlay)
                        fig.canvas.draw_idle()
                        fig.canvas.flush_events()
                    except Exception as e:
                        print(f"  reload error: {e}")
        except KeyboardInterrupt:
            print()
    else:
        plt.show()


if __name__ == '__main__':
    main()
