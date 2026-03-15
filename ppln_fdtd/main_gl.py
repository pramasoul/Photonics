"""PPLN 1D FDTD Explorer — GPU-rendered version (GLFW + ModernGL).

Keyboard controls (press H for help):
  Space     pause / resume
  R         reset simulation
  Up/Down   steps per frame (simulation speed)
  L/;       poling period Λ  (decrease / increase)
  T/Y       temperature  (decrease / increase)
  B/N       χ⁽²⁾ boost  (decrease / increase)
  P/O       pulse width  (decrease / increase)
  H         toggle help overlay
  Q/Esc     quit
"""

import time
import numpy as np
import glfw
import moderngl

from sim import FDTDSimulation
from render import Renderer
from materials import coherence_length

HELP_TEXT = """\
 PPLN 1D FDTD — Keyboard Controls

 Space      pause / resume
 R          reset (re-inject pulse)
 Up/Down    simulation speed (steps/frame)

 L / ;      poling period  Lambda  -/+
 T / Y      temperature  -/+ 5 C
 B / N      chi2 boost  -/+ (log scale)
 P / O      pulse width  -/+ 50 fs

 H          toggle this help
 Q / Esc    quit

 Color key:
   Red   = fundamental (omega)
   Cyan  = fundamental (negative)
   Blue  = second harmonic (2omega)
   Yellow = second harmonic (negative)
   Bottom bar = poling pattern (+d33 / -d33)
"""


def main():
    sim = FDTDSimulation()

    print(f"Grid: {sim.Nz} cells, dz = {sim.dz*1e9:.2f} nm")
    print(f"n(ω) = {sim.n1:.4f}, n(2ω) = {sim.n2:.4f}")
    print(f"Ideal QPM period = {sim.ideal_qpm_period_m*1e6:.2f} μm")
    print("Press H for keyboard controls")

    # ── Window setup ──
    if not glfw.init():
        raise RuntimeError("Failed to initialize GLFW")

    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
    glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
    glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, True)
    glfw.window_hint(glfw.RESIZABLE, True)

    width, height = 1400, 700
    window = glfw.create_window(width, height, "PPLN 1D FDTD — SHG Explorer", None, None)
    if not window:
        glfw.terminate()
        raise RuntimeError("Failed to create GLFW window")

    glfw.make_context_current(window)
    glfw.swap_interval(1)  # vsync

    ctx = moderngl.create_context()
    renderer = Renderer(ctx, width, height, sim)

    # ── State ──
    paused = False
    show_help = False
    steps_per_frame = 200
    spf_min, spf_max = 10, 100000

    # ── Resize callback ──
    def on_resize(win, w, h):
        nonlocal width, height
        if w > 0 and h > 0:
            width, height = w, h
            renderer.resize(w, h)

    glfw.set_framebuffer_size_callback(window, on_resize)

    # ── Key callback ──
    def on_key(win, key, scancode, action, mods):
        nonlocal paused, show_help, steps_per_frame

        if action not in (glfw.PRESS, glfw.REPEAT):
            return

        if key == glfw.KEY_ESCAPE or key == glfw.KEY_Q:
            glfw.set_window_should_close(win, True)
        elif key == glfw.KEY_SPACE:
            paused = not paused
        elif key == glfw.KEY_H:
            show_help = not show_help
        elif key == glfw.KEY_R:
            sim.reset()

        # Simulation speed
        elif key == glfw.KEY_UP:
            steps_per_frame = min(int(steps_per_frame * 1.5), spf_max)
        elif key == glfw.KEY_DOWN:
            steps_per_frame = max(int(steps_per_frame / 1.5), spf_min)

        # Poling period
        elif key == glfw.KEY_L:
            new_L = sim.Lambda - 0.1e-6
            if new_L > 1e-6:
                sim.rebuild_poling(new_L)
        elif key == glfw.KEY_SEMICOLON:
            sim.rebuild_poling(sim.Lambda + 0.1e-6)

        # Temperature
        elif key == glfw.KEY_T:
            sim.update_temperature(max(20, sim.T - 5))
        elif key == glfw.KEY_Y:
            sim.update_temperature(min(200, sim.T + 5))

        # Boost
        elif key == glfw.KEY_B:
            sim.update_boost(max(1, sim.boost / 1.5))
        elif key == glfw.KEY_N:
            sim.update_boost(min(500, sim.boost * 1.5))

        # Pulse width
        elif key == glfw.KEY_P:
            pw = sim.pulse_width_s * 1e15
            sim.update_pulse_width(max(50, pw - 50))
        elif key == glfw.KEY_O:
            pw = sim.pulse_width_s * 1e15
            sim.update_pulse_width(min(2000, pw + 50))

    glfw.set_key_callback(window, on_key)

    # ── Main loop ──
    frame_count = 0
    fps_time = time.perf_counter()
    fps_display = 0.0

    while not glfw.window_should_close(window):
        glfw.poll_events()

        # ── Simulate ──
        if not paused:
            sim.step(steps_per_frame)

        # ── Get fields ──
        E1, E2 = sim.get_fields()

        # ── Build info lines ──
        L_coh = coherence_length(sim.lambda_fund_um, sim.T)
        Lambda_qpm = 2 * L_coh
        delta_k = np.pi / L_coh - 2 * np.pi / sim.Lambda if sim.Lambda > 0 else 0

        info = [
            f"t = {sim.current_time_ps:.2f} ps    "
            f"steps/frame = {steps_per_frame}    "
            f"fps = {fps_display:.0f}"
            f"{'  [PAUSED]' if paused else ''}",

            f"Lambda = {sim.Lambda*1e6:.2f} um    "
            f"QPM = {Lambda_qpm*1e6:.2f} um    "
            f"Dk = {delta_k*1e-6:.1f} /mm",

            f"T = {sim.T:.0f} C    "
            f"boost = {sim.boost:.0f}x    "
            f"pulse = {sim.pulse_width_s*1e15:.0f} fs",

            f"max|E1| = {np.max(np.abs(E1)):.3f}    "
            f"max|E2| = {np.max(np.abs(E2)):.4f}",

            "H = help" if not show_help else "",
        ]

        # ── Render ──
        renderer.render(E1, E2, info, help_active=show_help, help_text=HELP_TEXT)

        glfw.swap_buffers(window)

        # FPS counter
        frame_count += 1
        now = time.perf_counter()
        if now - fps_time >= 0.5:
            fps_display = frame_count / (now - fps_time)
            frame_count = 0
            fps_time = now

    glfw.terminate()


if __name__ == "__main__":
    main()
