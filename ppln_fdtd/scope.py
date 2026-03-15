"""Matplotlib scope — displays snapshots saved by the GL simulation.

Usage:
    pixi run scope                  # display latest snapshot
    pixi run scope snapshot.npz     # display specific file
    pixi run scope --watch          # auto-reload on new snapshots
"""

import sys
import os
import time
import argparse
import numpy as np
import matplotlib.pyplot as plt
from materials import coherence_length

DEFAULT_PATH = os.path.join(os.path.dirname(__file__), 'snapshot.npz')


def load_snapshot(path):
    d = np.load(path, allow_pickle=True)
    return {k: d[k] for k in d.files}


def plot_snapshot(snap, fig=None):
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
    ax1.axvspan(z[cs], z[ce - 1], alpha=0.06, color='gray')
    ax1.set_ylabel('E-field')
    ax1.set_title(f't = {t_ps:.2f} ps')
    ax1.legend(loc='upper right', fontsize=7, ncol=2)

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

    # ── Panel 4: Parameters ──
    ax4 = fig.add_subplot(4, 1, 4)
    ax4.axis('off')
    info = (
        f"Λ = {Lambda:.2f} μm    Λ_QPM = {Lambda_qpm:.2f} μm    Δk = {dk * 1e-6:.1f} /mm\n"
        f"T = {T:.0f} °C    boost = {boost:.0f}×    pulse = {pw_fs:.0f} fs\n"
        f"n₁ = {n1:.4f}    n₂ = {n2:.4f}\n"
        f"R_pump = {R_pump}    R_sh = {R_sh}\n"
        f"max|E₁| = {np.max(np.abs(E1)):.4f}    max|E₂| = {np.max(np.abs(E2)):.5f}"
    )
    ax4.text(0.05, 0.9, info, transform=ax4.transAxes, fontsize=9,
             fontfamily='monospace', verticalalignment='top')

    return fig


def main():
    parser = argparse.ArgumentParser(description='PPLN FDTD Scope — snapshot viewer')
    parser.add_argument('file', nargs='?', default=DEFAULT_PATH,
                        help='Snapshot .npz file')
    parser.add_argument('--watch', '-w', action='store_true',
                        help='Auto-reload when snapshot file changes')
    args = parser.parse_args()

    if not os.path.exists(args.file):
        print(f"No snapshot found at {args.file}")
        print("Press 's' in the GL simulation to save one.")
        sys.exit(1)

    snap = load_snapshot(args.file)
    fig = plot_snapshot(snap)

    if args.watch:
        last_mtime = os.path.getmtime(args.file)
        plt.ion()
        plt.show(block=False)
        print("Watching for snapshot updates... (Ctrl+C to stop)")
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
                        plot_snapshot(snap, fig)
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
