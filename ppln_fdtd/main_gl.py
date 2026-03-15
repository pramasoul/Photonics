"""PPLN 1D FDTD Explorer — GPU-rendered display, terminal controls.

GL window is pure visual display. All text and keyboard input via terminal.
"""

import sys
import os
import time
import tty
import termios
import select
import numpy as np
import glfw
import moderngl

from sim import FDTDSimulation
from render import Renderer
from materials import coherence_length

HELP = """\
  PPLN 1D FDTD — Keyboard Controls
  ─────────────────────────────────
  space     pause / resume
  r         reset (re-inject pulse)
  + / -     simulation speed (steps/frame)
  l / ;     poling period Λ  −/+
  t / y     temperature  −/+ 5°C
  b / n     χ⁽²⁾ boost  −/+ (log)
  p / o     pulse width  −/+ 50 fs
  h         toggle this help
  q / ESC   quit

  Color key:
    Red/Cyan    = fundamental ω  (+/−)
    Blue/Yellow = second harmonic 2ω  (+/−)
    Bottom bar  = poling domains (red +d₃₃ / blue −d₃₃)
"""


class Terminal:
    """Raw-mode terminal for non-blocking single-char reads.

    Falls back gracefully when stdin is not a tty (e.g. backgrounded).
    """

    def __init__(self):
        self.active = os.isatty(sys.stdin.fileno())
        if self.active:
            self.fd = sys.stdin.fileno()
            self.old_settings = termios.tcgetattr(self.fd)
            tty.setraw(self.fd)
            new = termios.tcgetattr(self.fd)
            new[1] |= termios.OPOST
            termios.tcsetattr(self.fd, termios.TCSANOW, new)

    def read_key(self) -> str | None:
        if not self.active:
            return None
        if select.select([sys.stdin], [], [], 0)[0]:
            ch = sys.stdin.read(1)
            if ch == '\x1b':
                # Read any buffered continuation bytes (escape sequence)
                seq = ''
                while select.select([sys.stdin], [], [], 0.05)[0]:
                    seq += sys.stdin.read(1)
                    if len(seq) > 4:
                        break
                if seq.startswith('['):
                    code = seq[1:2]
                    return {'A': 'UP', 'B': 'DOWN', 'C': 'RIGHT', 'D': 'LEFT'}.get(code)
                # Bare escape (no sequence followed)
                return 'ESC'
            return ch
        return None

    def restore(self):
        if self.active:
            termios.tcsetattr(self.fd, termios.TCSADRAIN, self.old_settings)

    def clear_and_home(self):
        if self.active:
            sys.stdout.write('\x1b[2J\x1b[H')
            sys.stdout.flush()


def print_status(sim, steps_per_frame, fps, paused, show_help, energy_info):
    """Print status block to terminal, overwriting previous output."""
    sys.stdout.write('\x1b[H')  # cursor home

    L_coh = coherence_length(sim.lambda_fund_um, sim.T)
    Lambda_qpm = 2 * L_coh
    dk = np.pi / L_coh - 2 * np.pi / sim.Lambda if sim.Lambda > 0 else 0

    E1, E2 = sim.E1, sim.E2
    max_e1 = float(E1.max()) if hasattr(E1, 'max') else 0
    max_e2 = float(E2.max()) if hasattr(E2, 'max') else 0

    e_now, e_ref, e_drift = energy_info

    lines = [
        f"  PPLN 1D FDTD Explorer          {'[PAUSED]' if paused else ''}",
        f"  ────────────────────────────────────────────",
        f"  t = {sim.current_time_ps:8.2f} ps     fps = {fps:5.1f}     steps/frame = {steps_per_frame}",
        f"  Λ = {sim.Lambda*1e6:7.2f} μm      Λ_QPM = {Lambda_qpm*1e6:.2f} μm     Δk = {dk*1e-6:.1f} /mm",
        f"  T = {sim.T:5.0f} °C       boost = {sim.boost:.0f}×            pulse = {sim.pulse_width_s*1e15:.0f} fs",
        f"  max|E₁| = {max_e1:.4f}   max|E₂| = {max_e2:.5f}",
        f"  energy = {e_now:.6e}   drift = {e_drift:+.2e}%  (Verlet/leapfrog)",
        f"",
    ]

    if show_help:
        lines.extend(HELP.split('\n'))

    # Pad and write with ANSI clear-to-end-of-line
    for line in lines:
        sys.stdout.write(f"{line}\x1b[K\n")

    # Clear any leftover lines from previous (longer) output
    sys.stdout.write('\x1b[J')
    sys.stdout.flush()


def main():
    sim = FDTDSimulation()

    # ── Window ──
    if not glfw.init():
        raise RuntimeError("GLFW init failed")

    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
    glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
    glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, True)

    width, height = 1600, 800
    window = glfw.create_window(width, height, "PPLN FDTD", None, None)
    if not window:
        glfw.terminate()
        raise RuntimeError("Window creation failed")

    glfw.make_context_current(window)
    glfw.swap_interval(1)

    ctx = moderngl.create_context()
    renderer = Renderer(ctx, width, height, sim)

    def on_resize(win, w, h):
        nonlocal width, height
        if w > 0 and h > 0:
            width, height = w, h
            renderer.resize(w, h)

    glfw.set_framebuffer_size_callback(window, on_resize)

    # ── Terminal ──
    term = Terminal()
    term.clear_and_home()

    # ── State ──
    paused = False
    show_help = False
    steps_per_frame = 500
    spf_min, spf_max = 10, 100000
    pending_keys = []  # for GLFW fallback

    # ── GLFW key fallback (when no tty) ──
    if not term.active:
        KEY_MAP = {
            glfw.KEY_SPACE: ' ', glfw.KEY_R: 'r', glfw.KEY_H: 'h',
            glfw.KEY_Q: 'q', glfw.KEY_ESCAPE: 'ESC',
            glfw.KEY_UP: 'UP', glfw.KEY_DOWN: 'DOWN',
            glfw.KEY_L: 'l', glfw.KEY_SEMICOLON: ';',
            glfw.KEY_T: 't', glfw.KEY_Y: 'y',
            glfw.KEY_B: 'b', glfw.KEY_N: 'n',
            glfw.KEY_P: 'p', glfw.KEY_O: 'o',
        }

        def on_key(win, key, scancode, action, mods):
            if action in (glfw.PRESS, glfw.REPEAT) and key in KEY_MAP:
                pending_keys.append(KEY_MAP[key])

        glfw.set_key_callback(window, on_key)

    frame_count = 0
    fps_time = time.perf_counter()
    fps = 0.0
    energy_ref = 0.0  # set after source has fired
    energy_now = 0.0
    energy_drift = 0.0
    energy_ref_set = False

    try:
        while not glfw.window_should_close(window):
            glfw.poll_events()

            # ── Input (terminal or GLFW fallback) ──
            key = term.read_key()
            if key is None and pending_keys:
                key = pending_keys.pop(0)
            if key:
                if key in ('q', 'ESC'):
                    break
                elif key == ' ':
                    paused = not paused
                elif key == 'h':
                    show_help = not show_help
                elif key == 'r':
                    sim.reset()
                    energy_ref_set = False
                elif key == '+' or key == '=':
                    steps_per_frame = min(int(steps_per_frame * 1.5), spf_max)
                elif key == '-' or key == '_':
                    steps_per_frame = max(int(steps_per_frame / 1.5), spf_min)
                elif key == 'l':
                    new_L = sim.Lambda - 0.1e-6
                    if new_L > 1e-6:
                        sim.rebuild_poling(new_L)
                elif key == ';':
                    sim.rebuild_poling(sim.Lambda + 0.1e-6)
                elif key == 't':
                    sim.update_temperature(max(20, sim.T - 5))
                elif key == 'y':
                    sim.update_temperature(min(200, sim.T + 5))
                elif key == 'b':
                    sim.update_boost(max(1, sim.boost / 1.5))
                elif key == 'n':
                    sim.update_boost(min(500, sim.boost * 1.5))
                elif key == 'p':
                    pw = sim.pulse_width_s * 1e15
                    sim.update_pulse_width(max(50, pw - 50))
                elif key == 'o':
                    pw = sim.pulse_width_s * 1e15
                    sim.update_pulse_width(min(2000, pw + 50))

            # ── Simulate ──
            if not paused:
                sim.step(steps_per_frame)

            # ── Render ──
            E1, E2 = sim.get_fields()
            renderer.render(E1, E2)
            glfw.swap_buffers(window)

            # ── Terminal status ──
            frame_count += 1
            now = time.perf_counter()
            if now - fps_time >= 0.25:
                fps = frame_count / (now - fps_time)
                frame_count = 0
                fps_time = now

                # Energy check (GPU reduction, ~1x per status update)
                energy_now = sim.get_energy()
                # Set reference after source pulse has mostly fired (t > 5*t0)
                if not energy_ref_set and sim.current_time_ps > sim.t0 * 1e12 * 1.2 and energy_now > 0:
                    energy_ref = energy_now
                    energy_ref_set = True
                if energy_ref > 0:
                    energy_drift = (energy_now - energy_ref) / energy_ref * 100
                else:
                    energy_drift = 0.0

                print_status(sim, steps_per_frame, fps, paused, show_help,
                             (energy_now, energy_ref, energy_drift))

    finally:
        term.restore()
        glfw.terminate()
        print()  # clean line after restore


if __name__ == "__main__":
    main()
