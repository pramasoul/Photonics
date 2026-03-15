"""Visualization: 3-panel matplotlib figure with sliders for PPLN FDTD."""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from scipy.signal import hilbert


def setup_figure(sim):
    """Create the figure, axes, lines, sliders, and button. Returns a dict of handles."""

    z_um = sim.get_z_host()
    crystal_start_um = z_um[sim.crystal_start]
    crystal_end_um = z_um[sim.crystal_end - 1]

    fig = plt.figure(figsize=(14, 9))
    fig.canvas.manager.set_window_title("PPLN 1D FDTD — SHG Explorer")
    fig.subplots_adjust(left=0.08, right=0.95, top=0.94, bottom=0.35, hspace=0.35)

    # --- Top panel: E-fields ---
    ax_field = fig.add_subplot(3, 1, 1)
    ax_field.set_xlim(z_um[0], z_um[-1])
    ax_field.set_ylim(-1.2, 1.2)
    ax_field.set_ylabel("E-field (norm.)")
    ax_field.set_title("Fundamental (red) and Second Harmonic (blue)")
    # Decimate for plotting (show every Nth point)
    decimate = max(1, len(z_um) // 2000)
    z_dec = z_um[::decimate]
    line_e1, = ax_field.plot(z_dec, np.zeros_like(z_dec), 'r-', lw=0.5, alpha=0.6, label="ω (carrier)")
    line_e2, = ax_field.plot(z_dec, np.zeros_like(z_dec), 'b-', lw=0.5, alpha=0.6, label="2ω (carrier)")
    line_env1, = ax_field.plot(z_dec, np.zeros_like(z_dec), 'r-', lw=1.5, label="ω (envelope)")
    line_env2, = ax_field.plot(z_dec, np.zeros_like(z_dec), 'b-', lw=1.5, label="2ω (envelope)")
    # Crystal region shading
    ax_field.axvspan(crystal_start_um, crystal_end_um, alpha=0.07, color='gray')
    # Spectrum probe marker
    probe_um = z_um[sim.probe_idx]
    ax_field.axvline(probe_um, color='green', ls='--', lw=1, alpha=0.6)
    ax_field.text(probe_um, 0.98, " probe", color='green', fontsize=7,
                  transform=ax_field.get_xaxis_transform(), va='top')
    ax_field.legend(loc='upper right', fontsize=7, ncol=2)

    # --- Middle panel: Poling pattern ---
    ax_pol = fig.add_subplot(3, 1, 2, sharex=ax_field)
    d_z_host = sim.d_z.get()
    # Show only crystal region for clarity, decimate
    crystal_z = z_um[sim.crystal_start:sim.crystal_end]
    crystal_d = d_z_host[sim.crystal_start:sim.crystal_end]
    dec_pol = max(1, len(crystal_z) // 2000)
    z_pol_dec = crystal_z[::dec_pol]
    d_pol_dec = crystal_d[::dec_pol]
    line_pol, = ax_pol.plot(z_pol_dec, d_pol_dec, 'k-', lw=0.5)
    ax_pol.fill_between(z_pol_dec, d_pol_dec, alpha=0.3,
                        where=d_pol_dec > 0, color='red', interpolate=True)
    ax_pol.fill_between(z_pol_dec, d_pol_dec, alpha=0.3,
                        where=d_pol_dec < 0, color='blue', interpolate=True)
    ax_pol.set_ylim(-1.5, 1.5)
    ax_pol.set_ylabel("d(z) / d₃₃")
    ax_pol.set_xlabel("z (μm)")
    ax_pol.set_title("Poling Pattern")

    # --- Bottom panel: Spectrum ---
    ax_spec = fig.add_subplot(3, 1, 3)
    f2_THz = 2 * 2.998e8 / (sim.lambda_fund_um * 1e-6) * 1e-12
    ax_spec.set_xlim(0, f2_THz * 1.4)  # show well past 2ω
    ax_spec.set_ylim(1e-6, 1)
    ax_spec.set_yscale('log')
    ax_spec.set_xlabel("Frequency (THz)")
    ax_spec.set_ylabel("|FFT| (a.u.)")
    ax_spec.set_title("Spectrum at Probe")
    line_sp1, = ax_spec.plot([], [], 'r-', lw=1, label="ω")
    line_sp2, = ax_spec.plot([], [], 'b-', lw=1, label="2ω")
    # Mark expected fundamental and SH frequencies
    f1_THz = 2.998e8 / (sim.lambda_fund_um * 1e-6) * 1e-12
    ax_spec.axvline(f1_THz, color='r', ls='--', alpha=0.3, label=f"ω = {f1_THz:.0f} THz")
    ax_spec.axvline(2 * f1_THz, color='b', ls='--', alpha=0.3, label=f"2ω = {2*f1_THz:.0f} THz")
    ax_spec.legend(loc='upper right', fontsize=7)

    # --- Info text ---
    info_text = ax_field.text(
        0.01, 0.95, "", transform=ax_field.transAxes,
        fontsize=8, verticalalignment='top', fontfamily='monospace',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
    )

    # --- Sliders ---
    slider_color = 'lightgoldenrodyellow'
    ideal_Lambda = sim.ideal_qpm_period_m * 1e6  # to μm

    ax_lambda = fig.add_axes([0.15, 0.20, 0.55, 0.025])
    sl_lambda = Slider(ax_lambda, "Λ (μm)", ideal_Lambda * 0.5, ideal_Lambda * 2.0,
                       valinit=ideal_Lambda, valstep=0.01, color=slider_color)

    ax_temp = fig.add_axes([0.15, 0.16, 0.55, 0.025])
    sl_temp = Slider(ax_temp, "T (°C)", 20.0, 200.0,
                     valinit=sim.T, valstep=1.0, color=slider_color)

    ax_boost = fig.add_axes([0.15, 0.12, 0.55, 0.025])
    sl_boost = Slider(ax_boost, "χ⁽²⁾ boost", 0, np.log10(200), valinit=np.log10(sim.boost),
                      valstep=0.01, color=slider_color)
    # Override label to show actual value
    sl_boost.valtext.set_text(f"{sim.boost:.0f}×")

    ax_pw = fig.add_axes([0.15, 0.08, 0.55, 0.025])
    sl_pw = Slider(ax_pw, "Pulse (fs)", 50.0, 2000.0,
                   valinit=sim.pulse_width_s * 1e15, valstep=10.0, color=slider_color)

    # Steps/frame slider with gamma curve: more resolution at the low end
    # t in [0,1] → steps = 10 + 4990 * t^gamma
    _spf_gamma = 2.5
    _spf_min, _spf_max = 10, 100000
    _spf_range = _spf_max - _spf_min

    def _spf_to_t(spf):
        return ((spf - _spf_min) / _spf_range) ** (1.0 / _spf_gamma)

    def _t_to_spf(t):
        return int(_spf_min + _spf_range * t ** _spf_gamma)

    ax_spf = fig.add_axes([0.15, 0.04, 0.55, 0.025])
    sl_spf = Slider(ax_spf, "Steps/frame", 0.0, 1.0,
                    valinit=_spf_to_t(50), valstep=0.005, color=slider_color)
    sl_spf.valtext.set_text(f"{50}")

    # --- Envelope toggle button ---
    ax_env_btn = fig.add_axes([0.82, 0.20, 0.12, 0.03])
    btn_env = Button(ax_env_btn, "Envelope: ON", color=slider_color)

    # --- Reset button ---
    ax_reset = fig.add_axes([0.82, 0.16, 0.12, 0.03])
    btn_reset = Button(ax_reset, "Reset", color='lightsalmon')

    # --- Ideal Λ annotation on slider ---
    ax_lambda.axvline(ideal_Lambda, color='green', ls='--', lw=1.5)
    lambda_annotation = ax_lambda.text(
        ideal_Lambda, 1.3, f"QPM={ideal_Lambda:.2f}μm",
        fontsize=7, ha='center', color='green', transform=ax_lambda.get_xaxis_transform(),
    )

    return {
        'fig': fig,
        'ax_field': ax_field, 'ax_pol': ax_pol, 'ax_spec': ax_spec,
        'line_e1': line_e1, 'line_e2': line_e2,
        'line_env1': line_env1, 'line_env2': line_env2,
        'line_pol': line_pol,
        'line_sp1': line_sp1, 'line_sp2': line_sp2,
        'info_text': info_text,
        'sl_lambda': sl_lambda, 'sl_temp': sl_temp,
        'sl_boost': sl_boost, 'sl_pw': sl_pw,
        'sl_spf': sl_spf, '_t_to_spf': _t_to_spf, '_spf_to_t': _spf_to_t,
        'btn_reset': btn_reset, 'btn_env': btn_env,
        'lambda_annotation': lambda_annotation,
        'decimate': decimate, 'dec_pol': dec_pol,
        'z_dec': z_dec, 'z_pol_dec': z_pol_dec,
        'show_envelope': [True],  # mutable container for toggle state
    }


def compute_envelope(signal):
    """Compute envelope of a real signal via Hilbert transform."""
    if len(signal) < 4:
        return np.abs(signal)
    analytic = hilbert(signal)
    return np.abs(analytic)


def update_frame(sim, h):
    """Update all plot elements from current simulation state."""
    E1, E2 = sim.get_fields()
    dec = h['decimate']

    E1_dec = E1[::dec]
    E2_dec = E2[::dec]

    h['line_e1'].set_ydata(E1_dec)
    h['line_e2'].set_ydata(E2_dec)

    if h['show_envelope'][0]:
        env1 = compute_envelope(E1_dec)
        env2 = compute_envelope(E2_dec)
        h['line_env1'].set_ydata(env1)
        h['line_env2'].set_ydata(env2)
        h['line_env1'].set_visible(True)
        h['line_env2'].set_visible(True)
        h['line_e1'].set_alpha(0.3)
        h['line_e2'].set_alpha(0.3)
    else:
        h['line_env1'].set_visible(False)
        h['line_env2'].set_visible(False)
        h['line_e1'].set_alpha(0.8)
        h['line_e2'].set_alpha(0.8)

    # Auto-scale y-axis for fields
    max_val = max(np.max(np.abs(E1_dec)), np.max(np.abs(E2_dec)), 0.01)
    h['ax_field'].set_ylim(-1.3 * max_val, 1.3 * max_val)

    # Spectrum
    freqs, sp1, sp2 = sim.get_spectrum()
    if freqs is not None:
        norm = max(sp1.max(), 1e-30)
        h['line_sp1'].set_data(freqs, sp1 / norm)
        h['line_sp2'].set_data(freqs, sp2 / norm)

    # Info text
    from materials import coherence_length
    L_coh = coherence_length(sim.lambda_fund_um, sim.T)
    Lambda_qpm = 2 * L_coh
    delta_k = np.pi / L_coh - 2 * np.pi / sim.Lambda if sim.Lambda > 0 else 0
    h['info_text'].set_text(
        f"t = {sim.current_time_ps:.2f} ps | "
        f"Δk = {delta_k * 1e-6:.1f} /mm | "
        f"L_coh = {L_coh * 1e6:.2f} μm | "
        f"Λ_QPM = {Lambda_qpm * 1e6:.2f} μm | "
        f"boost = {sim.boost:.0f}×"
    )


def update_poling_plot(sim, h):
    """Redraw the poling pattern panel."""
    d_z_host = sim.d_z.get()
    crystal_d = d_z_host[sim.crystal_start:sim.crystal_end]
    d_pol_dec = crystal_d[::h['dec_pol']]
    z_pol_dec = h['z_pol_dec']

    h['line_pol'].set_ydata(d_pol_dec)

    # Redo fill_between (no way to update in place)
    ax = h['ax_pol']
    # Remove old collections
    while len(ax.collections) > 0:
        ax.collections[0].remove()
    ax.fill_between(z_pol_dec, d_pol_dec, alpha=0.3,
                    where=d_pol_dec > 0, color='red', interpolate=True)
    ax.fill_between(z_pol_dec, d_pol_dec, alpha=0.3,
                    where=d_pol_dec < 0, color='blue', interpolate=True)
