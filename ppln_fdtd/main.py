"""PPLN 1D FDTD Explorer — real-time SHG simulation with interactive controls."""

import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from sim import FDTDSimulation
from viz import setup_figure, update_frame, update_poling_plot
from materials import qpm_period, coherence_length


def main():
    sim = FDTDSimulation()
    sim.probe_enabled = True

    print(f"Grid: {sim.Nz} cells, dz = {sim.dz*1e9:.2f} nm, dt = {sim.dt*1e18:.2f} as")
    print(f"n(ω) = {sim.n1:.4f}, n(2ω) = {sim.n2:.4f}")
    print(f"Ideal QPM period = {sim.ideal_qpm_period_m*1e6:.2f} μm")
    print(f"Crystal: {sim.crystal_length*1e6:.0f} μm, cells {sim.crystal_start}–{sim.crystal_end}")

    h = setup_figure(sim)

    # --- Steps per frame (user-adjustable via slider, seeded by auto-calibration) ---
    sim.step(100)
    t0 = time.perf_counter()
    sim.step(500)
    t1 = time.perf_counter()
    step_time = (t1 - t0) / 500
    frame_budget = 1.0 / 30
    steps_per_frame = [max(50, int(frame_budget * 0.6 / step_time))]
    sim.reset()
    h['sl_spf'].set_val(h['_spf_to_t'](steps_per_frame[0]))
    h['sl_spf'].valtext.set_text(f"{steps_per_frame[0]}")
    print(f"Steps/frame: {steps_per_frame[0]} ({step_time*1e6:.1f} μs/step)")

    # --- Slider callbacks ---
    def on_lambda(val):
        sim.rebuild_poling(val * 1e-6)  # μm -> m
        update_poling_plot(sim, h)
        # Update ideal QPM marker
        ideal = sim.ideal_qpm_period_m * 1e6
        h['lambda_annotation'].set_text(f"QPM={ideal:.2f}μm")
        h['lambda_annotation'].set_x(ideal)

    def on_temp(val):
        sim.update_temperature(val)
        # Update ideal QPM on lambda slider
        ideal = sim.ideal_qpm_period_m * 1e6
        h['lambda_annotation'].set_text(f"QPM={ideal:.2f}μm")
        h['lambda_annotation'].set_x(ideal)
        # Move the green marker
        for line in h['sl_lambda'].ax.get_lines():
            if line.get_color() == 'green':
                line.set_xdata([ideal, ideal])

    def on_boost(val):
        actual = 10 ** val
        sim.update_boost(actual)
        h['sl_boost'].valtext.set_text(f"{actual:.0f}×")

    def on_pw(val):
        sim.update_pulse_width(val)

    def on_spf(val):
        steps_per_frame[0] = h['_t_to_spf'](val)
        h['sl_spf'].valtext.set_text(f"{steps_per_frame[0]}")

    def on_reset(event):
        sim.reset()

    def on_envelope(event):
        h['show_envelope'][0] = not h['show_envelope'][0]
        state = "ON" if h['show_envelope'][0] else "OFF"
        h['btn_env'].label.set_text(f"Envelope: {state}")

    h['sl_lambda'].on_changed(on_lambda)
    h['sl_temp'].on_changed(on_temp)
    h['sl_boost'].on_changed(on_boost)
    h['sl_pw'].on_changed(on_pw)
    h['sl_spf'].on_changed(on_spf)
    h['btn_reset'].on_clicked(on_reset)
    h['btn_env'].on_clicked(on_envelope)

    # --- Animation ---
    def animate(_frame):
        sim.step(steps_per_frame[0])
        update_frame(sim, h)
        return []

    _anim = FuncAnimation(h['fig'], animate, interval=33, blit=False, cache_frame_data=False)
    plt.show()


if __name__ == "__main__":
    main()
