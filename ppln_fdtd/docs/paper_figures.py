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


# ── Figure 4: Coupling Convergence ─────────────────────────────────────

def fig04_coupling_convergence():
    print('Figure 4: Coupling convergence (ppw sweep, ~5 min)...')
    ppw_values = [20, 30, 40, 60, 80, 100]

    # ΔP_NL approach (current, with dispersion correction)
    ratios_dpnl = []
    for ppw in ppw_values:
        sim = FDTDSimulation(peak_intensity_W_cm2=1e9, ppw=ppw, mr_interval=0)
        steps = int(250000 * ppw / 20)
        sim.step(min(steps, 1200000))
        E1, E2 = sim.get_fields()
        r = np.max(np.abs(E2)) / max(np.max(np.abs(E1)), 1)
        ratios_dpnl.append(r)
        print(f'    ppw={ppw}: E2/E1={r:.3f}')

    fig, ax = plt.subplots(figsize=(4.5, 3.5))
    ax.plot(ppw_values, [r * 100 for r in ratios_dpnl], 'o-', color=C_FUND,
            label='$\\Delta P_{NL}$ source + dispersion corr.', markersize=5)
    ax.axhline(91, color=C_GRAY, ls='--', lw=0.75, label='CW theory (91%)')
    ax.set_xlabel('Points per SH wavelength (ppw)')
    ax.set_ylabel('$|E_2|/|E_1|$ (%)')
    ax.set_ylim(0, 120)
    ax.legend(fontsize=8)

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
    print('Figure 7: Dispersion correction effect (ppw sweep, ~5 min)...')
    import cupy as cp

    ppw_values = [20, 40, 60, 80, 100]

    # WITH correction (current default)
    ratios_with = []
    for ppw in ppw_values:
        sim = FDTDSimulation(peak_intensity_W_cm2=1e9, ppw=ppw, mr_interval=0)
        steps = int(250000 * ppw / 20)
        sim.step(min(steps, 1200000))
        E1, E2 = sim.get_fields()
        ratios_with.append(np.max(np.abs(E2)) / max(np.max(np.abs(E1)), 1))
        print(f'    WITH ppw={ppw}: {ratios_with[-1]:.3f}')

    # WITHOUT correction
    ratios_without = []
    for ppw in ppw_values:
        sim = FDTDSimulation(peak_intensity_W_cm2=1e9, ppw=ppw, mr_interval=0)
        # Override with physical n
        eps1_phys = EPS0 * sim.n1 ** 2
        eps2_phys = EPS0 * sim.n2 ** 2
        inv1 = np.full(sim.Nz, sim.dt / (eps1_phys * sim.dz))
        inv2 = np.full(sim.Nz, sim.dt / (eps2_phys * sim.dz))
        inv1[:sim.crystal_start] = sim.dt / (EPS0 * sim.dz)
        inv1[sim.crystal_end:] = sim.dt / (EPS0 * sim.dz)
        inv2[:sim.crystal_start] = sim.dt / (EPS0 * sim.dz)
        inv2[sim.crystal_end:] = sim.dt / (EPS0 * sim.dz)
        sim.inv_eps1 = cp.asarray(inv1)
        sim.inv_eps2 = cp.asarray(inv2)
        steps = int(250000 * ppw / 20)
        sim.step(min(steps, 1200000))
        E1, E2 = sim.get_fields()
        ratios_without.append(np.max(np.abs(E2)) / max(np.max(np.abs(E1)), 1))
        print(f'    WITHOUT ppw={ppw}: {ratios_without[-1]:.3f}')

    fig, ax = plt.subplots(figsize=(4.5, 3.5))
    ax.plot(ppw_values, [r * 100 for r in ratios_without], 's--', color=C_GRAY,
            label='Without correction', markersize=5)
    ax.plot(ppw_values, [r * 100 for r in ratios_with], 'o-', color=C_FUND,
            label='With dispersion correction', markersize=5)
    ax.axhline(91, color=C_REF, ls=':', lw=0.75, label='CW theory')
    ax.set_xlabel('Points per SH wavelength (ppw)')
    ax.set_ylabel('$|E_2|/|E_1|$ (%)')
    ax.set_ylim(0, 120)
    ax.legend(fontsize=8)

    fig.tight_layout()
    savefig(fig, 'fig07_dispersion_correction')


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
