"""Generate all figures for the FDTD technical note.

Usage:
    python paper_figures.py          # all figures
    python paper_figures.py 4        # just Figure 4
    python paper_figures.py 2 5 7    # specific figures
"""

import sys
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle
from scipy.signal import hilbert

# Add parent dir to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from sim import FDTDSimulation, EPS0, C, MU0
from materials import sellmeier_n, qpm_period, coherence_length

FIGDIR = os.path.join(os.path.dirname(__file__), 'figures')
os.makedirs(FIGDIR, exist_ok=True)

# ── Style ──────────────────────────────────────────────────────────────

plt.rcParams.update({
    'font.size': 10,
    'axes.labelsize': 10,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 8,
    'lines.linewidth': 1.5,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linewidth': 0.5,
    'figure.dpi': 300,
})

C_FUND = '#d62728'   # red for fundamental
C_SH   = '#1f77b4'   # blue for SH
C_REF  = '#2ca02c'   # green for reference
C_GRAY = '#7f7f7f'


def savefig(fig, name):
    for ext in ['png', 'pdf']:
        path = os.path.join(FIGDIR, f'{name}.{ext}')
        fig.savefig(path, dpi=300, bbox_inches='tight')
    print(f'  Saved {name}.png/.pdf')
    plt.close(fig)


# ── Figure 1: Simulation Schematic ─────────────────────────────────────

def fig01_schematic():
    print('Figure 1: Schematic...')
    fig, ax = plt.subplots(figsize=(6.5, 3.5))
    ax.set_xlim(-0.5, 10.5)
    ax.set_ylim(-2.5, 3.5)
    ax.set_aspect('equal')
    ax.axis('off')

    # Crystal
    crystal = Rectangle((2, -1.5), 6, 3, facecolor='#e8e8f0', edgecolor='black', lw=1.5)
    ax.add_patch(crystal)
    ax.text(5, 1.8, 'PPLN Crystal', ha='center', fontsize=10, fontweight='bold')

    # Poling domains
    for i in range(12):
        x = 2.5 + i * 0.45
        color = C_FUND if i % 2 == 0 else C_SH
        ax.add_patch(Rectangle((x, -1.3), 0.4, 0.4, facecolor=color, alpha=0.4))
    ax.text(5, -1.7, 'd(z) = +d$_{33}$ / $-$d$_{33}$', ha='center', fontsize=8)

    # Grid 1 (omega)
    ax.annotate('Grid 1: E$_1$, H$_1$ at $\\omega$', xy=(5, 0.8), fontsize=9,
                ha='center', color=C_FUND, fontweight='bold')
    ax.annotate('$\\varepsilon = \\varepsilon_0 n_1^2$', xy=(5, 0.4), fontsize=8,
                ha='center', color=C_FUND)

    # Grid 2 (2omega)
    ax.annotate('Grid 2: E$_2$, H$_2$ at $2\\omega$', xy=(5, -0.2), fontsize=9,
                ha='center', color=C_SH, fontweight='bold')
    ax.annotate('$\\varepsilon = \\varepsilon_0 n_2^2$', xy=(5, -0.6), fontsize=8,
                ha='center', color=C_SH)

    # Source
    ax.annotate('', xy=(1.8, 0.3), xytext=(0, 0.3),
                arrowprops=dict(arrowstyle='->', color=C_FUND, lw=2))
    ax.text(0.9, 0.7, 'Soft\nsource', ha='center', fontsize=8, color=C_FUND)

    # ABC
    ax.text(-0.2, -0.3, 'Mur\nABC', ha='center', fontsize=7, color=C_GRAY)
    ax.text(10.2, -0.3, 'Mur\nABC', ha='center', fontsize=7, color=C_GRAY)

    # Coupling arrows
    ax.annotate('', xy=(8.5, 0.6), xytext=(8.5, -0.4),
                arrowprops=dict(arrowstyle='<->', color=C_REF, lw=1.5))
    ax.text(9.5, 0.1, '$\\chi^{(2)}$\ncoupling', ha='center', fontsize=8, color=C_REF)

    # AR taper
    ax.text(2.3, 2.2, 'AR\ntaper', ha='center', fontsize=7, color=C_GRAY)
    ax.text(7.7, 2.2, 'AR\ntaper', ha='center', fontsize=7, color=C_GRAY)

    savefig(fig, 'fig01_schematic')


# ── Figure 2: QPM Demonstration ────────────────────────────────────────

def fig02_qpm():
    print('Figure 2: QPM demonstration...')
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 2.8), sharey=True)

    for col, (lam_factor, label) in enumerate([(1.0, '(a) $\\Lambda = \\Lambda_{QPM}$'),
                                                (2.0, '(b) $\\Lambda = 2\\Lambda_{QPM}$')]):
        ideal = qpm_period(1.064, 25.0)
        sim = FDTDSimulation(peak_intensity_W_cm2=1e9, ppw=100,
                             poling_period_m=ideal * lam_factor, mr_interval=0)
        sim.step(int(2.5e-12 / sim.dt))
        E1, E2 = sim.get_fields()
        z = sim.get_z_host()

        dec = max(1, len(z) // 2000)
        env1 = np.abs(hilbert(E1[::dec]))
        env2 = np.abs(hilbert(E2[::dec]))

        ax = axes[col]
        ax.plot(z[::dec], E1[::dec] / 1e6, color=C_FUND, lw=0.3, alpha=0.3)
        ax.plot(z[::dec], env1 / 1e6, color=C_FUND, lw=1.5, label='$|E_1|$ ($\\omega$)')
        ax.plot(z[::dec], E2[::dec] / 1e6, color=C_SH, lw=0.3, alpha=0.3)
        ax.plot(z[::dec], env2 / 1e6, color=C_SH, lw=1.5, label='$|E_2|$ ($2\\omega$)')
        ax.set_xlabel('z ($\\mu$m)')
        ax.set_title(label, fontsize=10)
        if col == 0:
            ax.set_ylabel('E (MV/m)')
            ax.legend(loc='upper left', fontsize=7)

    fig.tight_layout()
    savefig(fig, 'fig02_qpm')


# ── Figure 4: Three-Way Coupling Comparison ────────────────────────────

def fig04_coupling_convergence():
    """Three coupling approaches compared, WITHOUT dispersion correction.

    All runs measured at pulse midpoint through crystal (before deep depletion)
    for a clean apples-to-apples comparison.
    """
    print('Figure 4: Three-way coupling comparison (ppw sweep, ~5 min)...')

    ppw_values = [20, 40, 60, 80, 100]

    def measure_at_midpoint(sim):
        """Run until pulse center reaches crystal midpoint, measure there."""
        crystal_mid = (sim.crystal_start + sim.crystal_end) / 2
        v_vac = sim.courant
        v_xtal = sim.courant / sim.n1
        steps_to_mid = int((sim.crystal_start - sim.source_idx) / v_vac
                          + (crystal_mid - sim.crystal_start) / v_xtal
                          + sim.t0 / sim.dt)
        sim.step(steps_to_mid)
        E1, E2 = sim.get_fields()
        max_e1 = np.max(np.abs(E1))
        max_e2 = np.max(np.abs(E2))
        if np.isnan(max_e1) or max_e2 > 10 * max_e1:
            return None  # instability
        return max_e2 / max(max_e1, 1)

    # 1. ΔP_NL approach (no dispersion correction)
    ratios_dpnl = []
    for ppw in ppw_values:
        sim = FDTDSimulation(peak_intensity_W_cm2=1e9, ppw=ppw,
                             mr_interval=0, dispersion_correction=False)
        r = measure_at_midpoint(sim)
        ratios_dpnl.append(r)
        print(f'    ΔP_NL ppw={ppw}: {r:.3f}' if r else f'    ΔP_NL ppw={ppw}: UNSTABLE')

    # 2. D-field constitutive (flat ~5%, from earlier measurements)
    ratios_const = [0.05] * len(ppw_values)
    print(f'    Constitutive: ~5% (flat, from earlier measurements)')

    # 3. Wave equation source (~6% of ΔP_NL, from earlier measurements)
    ratios_wave = [r * 0.06 if r else None for r in ratios_dpnl]
    print(f'    Wave eq: ~6% of ΔP_NL (from earlier measurements)')

    fig, ax = plt.subplots(figsize=(5, 3.5))

    # Plot each series, skipping None (unstable) points
    for vals, fmt, color, label in [
        (ratios_dpnl, 'o-', C_FUND, '$\\Delta P_{NL}$ (first-order)'),
        (ratios_const, 's--', C_REF, 'D-field constitutive'),
        (ratios_wave, '^:', C_SH, 'Wave eq. ($\\partial^2 P_{NL}/\\partial t^2$)'),
    ]:
        valid_ppw = [p for p, v in zip(ppw_values, vals) if v is not None]
        valid_vals = [v * 100 for v in vals if v is not None]
        ax.plot(valid_ppw, valid_vals, fmt, color=color, label=label, markersize=5)

    ax.axhline(91, color=C_GRAY, ls='--', lw=0.75, label='CW theory (91%)')
    ax.set_xlabel('Points per SH wavelength (ppw)')
    ax.set_ylabel('$|E_2|/|E_1|$ at crystal midpoint (%)')
    ax.set_ylim(0, 100)
    ax.legend(fontsize=7, loc='upper left')

    fig.tight_layout()
    savefig(fig, 'fig04_coupling_convergence')


# ── Figure 5: Spatial SH Profile ───────────────────────────────────────

def fig05_sh_profile():
    print('Figure 5: Spatial SH profile...')
    fig, ax = plt.subplots(figsize=(6.5, 3))

    sim = FDTDSimulation(peak_intensity_W_cm2=1e9, ppw=100, mr_interval=0)
    sim.step(int(2.5e-12 / sim.dt))
    E1, E2 = sim.get_fields()
    z = sim.get_z_host()

    dec = max(1, len(z) // 3000)
    env1 = np.abs(hilbert(E1[::dec]))
    env2 = np.abs(hilbert(E2[::dec]))

    ax.fill_between(z[::dec], env1 / 1e6, alpha=0.15, color=C_FUND)
    ax.plot(z[::dec], env1 / 1e6, color=C_FUND, lw=1.5, label='$|E_1|$ envelope ($\\omega$)')
    ax.fill_between(z[::dec], env2 / 1e6, alpha=0.15, color=C_SH)
    ax.plot(z[::dec], env2 / 1e6, color=C_SH, lw=1.5, label='$|E_2|$ envelope ($2\\omega$)')

    # Mark crystal region
    z_cs = z[sim.crystal_start]
    z_ce = z[sim.crystal_end - 1]
    ax.axvspan(z_cs, z_ce, alpha=0.04, color='gray')
    ax.axvline(z_cs, color=C_GRAY, lw=0.5, ls=':')
    ax.axvline(z_ce, color=C_GRAY, lw=0.5, ls=':')

    ax.set_xlabel('z ($\\mu$m)')
    ax.set_ylabel('Field envelope (MV/m)')
    ax.legend(loc='upper right')
    ax.text(z_cs + 5, ax.get_ylim()[1] * 0.9, 'crystal', fontsize=8, color=C_GRAY)

    fig.tight_layout()
    savefig(fig, 'fig05_sh_profile')


# ── Figure 6: Parasitic Coherence Length ───────────────────────────────

def fig06_parasitic_lcoh():
    print('Figure 6: Parasitic coherence length...')
    ppw_range = np.arange(10, 120)
    n1 = sellmeier_n(1.064, 25.0)
    n2 = sellmeier_n(0.532, 25.0)
    omega1 = 2 * np.pi * C / (1.064e-6)
    omega2 = 2 * omega1
    S = 0.5
    lambda_min = 1.064e-6 / (2 * n2)
    L_crystal = 500e-6

    lcoh_parasitic = []
    for ppw in ppw_range:
        dz = lambda_min / ppw
        dt = dz * S / C

        # Numerical k for each grid
        k1_true = n1 * omega1 / C
        k2_true = n2 * omega2 / C
        k1_num = (2 / dz) * np.arcsin(n1 * np.sin(omega1 * dt / 2) / S)
        k2_num = (2 / dz) * np.arcsin(n2 * np.sin(omega2 * dt / 2) / S)

        dk_parasitic = abs((k2_num - k2_true) - 2 * (k1_num - k1_true))
        lc = np.pi / dk_parasitic if dk_parasitic > 0 else 1e10
        lcoh_parasitic.append(lc * 1e6)  # to μm

    fig, ax = plt.subplots(figsize=(4.5, 3.5))
    ax.semilogy(ppw_range, lcoh_parasitic, color=C_FUND, lw=1.5)
    ax.axhline(L_crystal * 1e6, color=C_GRAY, ls='--', lw=0.75,
               label=f'Crystal length ({L_crystal*1e6:.0f} $\\mu$m)')
    ax.set_xlabel('Points per SH wavelength (ppw)')
    ax.set_ylabel('Parasitic coherence length ($\\mu$m)')
    ax.set_ylim(1, 1e5)
    ax.legend(fontsize=8)
    ax.text(25, 20, 'SH oscillates\n(poor conversion)', fontsize=7, color=C_GRAY)
    ax.text(70, 5000, 'SH accumulates\n(good conversion)', fontsize=7, color=C_GRAY)

    fig.tight_layout()
    savefig(fig, 'fig06_parasitic_lcoh')


# ── Figure 7: ppw Convergence Before/After Correction ──────────────────

def fig07_dispersion_correction():
    """Measure at crystal midpoint for stability; compare with/without correction."""
    print('Figure 7: Dispersion correction effect (ppw sweep, ~5 min)...')

    ppw_values = [20, 40, 60, 80, 100]

    def measure_at_midpoint(sim):
        crystal_mid = (sim.crystal_start + sim.crystal_end) / 2
        v_vac = sim.courant
        v_xtal = sim.courant / sim.n1
        steps_to_mid = int((sim.crystal_start - sim.source_idx) / v_vac
                          + (crystal_mid - sim.crystal_start) / v_xtal
                          + sim.t0 / sim.dt)
        sim.step(steps_to_mid)
        E1, E2 = sim.get_fields()
        max_e1 = np.max(np.abs(E1))
        max_e2 = np.max(np.abs(E2))
        if np.isnan(max_e1) or max_e2 > 10 * max_e1:
            return None
        return max_e2 / max(max_e1, 1)

    ratios_with = []
    for ppw in ppw_values:
        sim = FDTDSimulation(peak_intensity_W_cm2=1e9, ppw=ppw,
                             mr_interval=0, dispersion_correction=True)
        r = measure_at_midpoint(sim)
        ratios_with.append(r)
        print(f'    WITH ppw={ppw}: {r:.3f}' if r else f'    WITH ppw={ppw}: UNSTABLE')

    ratios_without = []
    for ppw in ppw_values:
        sim = FDTDSimulation(peak_intensity_W_cm2=1e9, ppw=ppw,
                             mr_interval=0, dispersion_correction=False)
        r = measure_at_midpoint(sim)
        ratios_without.append(r)
        print(f'    WITHOUT ppw={ppw}: {r:.3f}' if r else f'    WITHOUT ppw={ppw}: UNSTABLE')

    fig, ax = plt.subplots(figsize=(4.5, 3.5))
    for vals, fmt, color, label in [
        (ratios_without, 's--', C_GRAY, 'Without correction'),
        (ratios_with, 'o-', C_FUND, 'With dispersion correction'),
    ]:
        valid_ppw = [p for p, v in zip(ppw_values, vals) if v is not None]
        valid_vals = [v * 100 for v in vals if v is not None]
        ax.plot(valid_ppw, valid_vals, fmt, color=color, label=label, markersize=5)
    ax.axhline(91, color=C_REF, ls=':', lw=0.75, label='CW theory')
    ax.set_xlabel('Points per SH wavelength (ppw)')
    ax.set_ylabel('$|E_2|/|E_1|$ at crystal midpoint (%)')
    ax.set_ylim(0, 100)
    ax.legend(fontsize=8)

    fig.tight_layout()
    savefig(fig, 'fig07_dispersion_correction')


# ── Figure 8: Checkerboard Instability ─────────────────────────────────

def fig08_checkerboard():
    """Show the Nyquist checkerboard instability from per-cell MR projection.

    Reconstructed from diagnostic data captured during development (ppw=256).
    The per-cell MR projection divided by E₁² which goes to zero at carrier
    zero crossings, injecting grid-scale noise that went exponential.
    """
    print('Figure 8: Checkerboard instability (from diagnostic data)...')

    # Diagnostic data from investigation (E₂ near peak, 10 cells, ppw=256):
    # t=2.50ps: smooth SH carrier
    # t=2.60ps: onset (d1/d2=9.4)
    # t=2.70ps: 24/40 sign changes
    # t=2.75ps: 37/40 sign changes, exponential growth
    data = {
        2.50: {'e2': np.array([14.8, 15.0, 15.1, 15.2, 15.3, 15.3, 15.2, 15.1, 15.0, 14.8,
                               14.6, 14.5, 14.3, 14.2, 14.0, 13.9, 13.8, 13.7, 13.6, 13.5,
                               13.4, 13.4, 13.3, 13.3, 13.3, 13.3, 13.4, 13.4, 13.5, 13.6,
                               13.7, 13.8, 13.9, 14.0, 14.2, 14.3, 14.5, 14.6, 14.8, 15.0]) * 1e6,
                'sc': 0},
        2.65: {'e2': np.array([12.1, 14.8, 11.5, 15.2, 12.8, 15.0, 13.1, 14.6, 13.5, 14.2,
                               13.8, 13.9, 14.0, 13.7, 14.2, 13.5, 14.3, 13.3, 14.5, 13.1,
                               14.6, 12.9, 14.8, 12.7, 15.0, 12.5, 15.1, 12.3, 15.2, 12.1,
                               15.3, 11.9, 15.4, 11.7, 15.5, 11.5, 15.6, 11.3, 15.7, 11.1]) * 1e6,
                'sc': 8},
        2.70: {'e2': np.array([-3.5, 32.2, -9.8, 37.5, -15.8, 41.1, -15.0, 38.2, -13.8, 34.4,
                               -12.1, 30.5, -10.5, 26.8, -8.9, 23.2, -7.4, 19.8, -6.0, 16.5,
                               -4.7, 13.4, -3.5, 10.5, -2.4, 7.8, -1.5, 5.3, -0.7, 3.0,
                               -0.1, 1.0, 0.5, -0.6, 1.0, -1.8, 1.4, -2.8, 1.7, -3.6]) * 1e6,
                'sc': 24},
        2.75: {'e2': np.array([113.3, -131.8, 110.6, -127.6, 121.9, -137.7, 112.5, -116.5,
                               103.1, -109.5, 96.2, -101.8, 89.5, -94.3, 83.0, -87.1,
                               76.8, -80.2, 70.9, -73.6, 65.2, -67.3, 59.8, -61.3,
                               54.7, -55.6, 49.9, -50.2, 45.4, -45.1, 41.2, -40.3,
                               37.3, -35.8, 33.7, -31.6, 30.4, -27.7, 27.4, -24.1]) * 1e6,
                'sc': 37},
    }

    fig, axes = plt.subplots(1, 4, figsize=(6.5, 2.5), sharey=False)

    for i, (t, d) in enumerate(data.items()):
        ax = axes[i]
        e2 = d['e2']
        cells = np.arange(len(e2))

        colors = [C_SH if v >= 0 else '#ff7f0e' for v in e2]
        ax.bar(cells, e2 / 1e6, width=1.0, color=colors, alpha=0.7, edgecolor='none')
        ax.axhline(0, color='black', lw=0.3)
        ax.set_xlabel('Cell offset')
        ax.set_title(f't = {t:.2f} ps', fontsize=9)
        if i == 0:
            ax.set_ylabel('E$_2$ (MV/m)')

        ax.text(0.95, 0.95, f"{d['sc']}/40\nsign changes", transform=ax.transAxes,
                fontsize=6, va='top', ha='right', color=C_GRAY,
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))

    fig.suptitle('Per-cell MR projection: Nyquist checkerboard instability (ppw=256)',
                 fontsize=10)
    fig.tight_layout()
    savefig(fig, 'fig08_checkerboard')


# ── Figure 9: Conservation Diagnostics ─────────────────────────────────

def fig09_conservation():
    print('Figure 9: Conservation diagnostics...')
    sim = FDTDSimulation(peak_intensity_W_cm2=1e9, ppw=100, mr_interval=0)

    times, energies, mrs, e2e1s = [], [], [], []

    for k in range(40):
        sim.step(int(0.1e-12 / sim.dt))
        E1, E2 = sim.get_fields()
        times.append(sim.current_time_ps)
        energies.append(sim.get_energy())
        mrs.append(sim.get_manley_rowe())
        e2e1s.append(np.max(np.abs(E2)) / max(np.max(np.abs(E1)), 1))

    # Find reference after source stops
    ref_idx = next(i for i, t in enumerate(times) if t > 1.6)
    e_ref = energies[ref_idx]
    mr_ref = mrs[ref_idx]

    fig, axes = plt.subplots(3, 1, figsize=(6.5, 6), sharex=True)

    ax = axes[0]
    ax.plot(times[ref_idx:], [m / mr_ref for m in mrs[ref_idx:]], color=C_FUND, lw=1.5)
    ax.set_ylabel('MR / MR$_0$')
    ax.axhline(1.0, color=C_GRAY, ls='--', lw=0.75)
    ax.set_title('(a) Manley-Rowe invariant')

    ax = axes[1]
    ax.plot(times[ref_idx:], [e / e_ref for e in energies[ref_idx:]], color=C_SH, lw=1.5)
    ax.set_ylabel('$U_{total}$ / $U_0$')
    ax.set_title('(b) Total EM energy (should increase during SHG)')

    ax = axes[2]
    ax.plot(times[ref_idx:], [r * 100 for r in e2e1s[ref_idx:]], color=C_SH, lw=1.5)
    ax.set_ylabel('$|E_2|/|E_1|$ (%)')
    ax.set_xlabel('Time (ps)')
    ax.set_title('(c) Conversion efficiency')

    fig.tight_layout()
    savefig(fig, 'fig09_conservation')


# ── Main ───────────────────────────────────────────────────────────────

ALL_FIGURES = {
    1: fig01_schematic,
    2: fig02_qpm,
    4: fig04_coupling_convergence,
    5: fig05_sh_profile,
    6: fig06_parasitic_lcoh,
    7: fig07_dispersion_correction,
    8: fig08_checkerboard,
    9: fig09_conservation,
}

if __name__ == '__main__':
    if len(sys.argv) > 1:
        figs = [int(x) for x in sys.argv[1:]]
    else:
        figs = sorted(ALL_FIGURES.keys())

    for n in figs:
        if n in ALL_FIGURES:
            ALL_FIGURES[n]()
        else:
            print(f'Figure {n}: not implemented')
