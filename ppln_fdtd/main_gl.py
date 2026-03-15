"""PPLN 1D FDTD Explorer — GPU-rendered display, terminal controls.

GL window is pure visual display. All text and keyboard input via terminal.
Parameters settable via YAML config file and/or command-line switches.
"""

import sys
import os
import time
import argparse
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
  1 / 2     R_pump (ω face reflectivity)  −/+
  3 / 4     R_sh (2ω face reflectivity)  −/+
  [ / ]     spectrum ref level  −/+ 10 dB
  { / }     spectrum range  −/+ 1 decade
  s         save snapshot (view with: pixi run scope)
  f         toggle interpolation (linear / nearest)
  h         toggle this help
  q / ESC   quit

  Color key:
    Red/Cyan    = fundamental ω  (+/−)
    Blue/Yellow = second harmonic 2ω  (+/−)
    Bottom bar  = poling domains (red +d₃₃ / blue −d₃₃)
"""


# ── Defaults ──────────────────────────────────────────────────────────

DEFAULTS = dict(
    lambda_fund=1.064,
    crystal_length=500.0,
    temperature=25.0,
    boost=1.0,
    peak_intensity=1e9,       # W/cm²
    ppw=20,                   # points per SH wavelength
    pulse_width=200.0,
    poling_period=None,
    steps_per_frame=500,
    R_pump=0.0,
    R_sh=0.0,
    spec_ref=4.0,
    spec_decades=8.0,
    width=1600,
    height=800,
    record_start=None,      # ps, start recording snapshots
    record_end=None,        # ps, stop recording
    record_interval=0.01,   # ps between snapshots (default 10 fs)
    record_dir='recordings',
    mr_interval=0,          # global MR projection every N steps (0=off)
)


def load_config(args):
    """Merge YAML config (if any) with CLI overrides. CLI wins."""
    cfg = dict(DEFAULTS)

    config_path = args.config
    if config_path is None:
        # Auto-load default.yaml from script directory if present
        default_path = os.path.join(os.path.dirname(__file__), 'default.yaml')
        if os.path.exists(default_path):
            config_path = default_path

    if config_path:
        import yaml
        with open(config_path) as f:
            file_cfg = yaml.safe_load(f) or {}
        for k, v in file_cfg.items():
            if k in cfg and v is not None:
                # Coerce to match default's type
                default_val = DEFAULTS[k]
                if isinstance(default_val, float) and not isinstance(v, (int, float)):
                    v = float(v)
                elif isinstance(default_val, int) and not isinstance(v, int):
                    v = int(v)
                cfg[k] = v

    # CLI overrides (only if explicitly set)
    for key in DEFAULTS:
        cli_val = getattr(args, key, None)
        if cli_val is not None:
            cfg[key] = cli_val

    return cfg


def build_parser():
    p = argparse.ArgumentParser(
        description="PPLN 1D FDTD SHG Explorer (GPU-rendered)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument('-c', '--config', type=str, default=None,
                   help='YAML config file path')
    p.add_argument('--lambda-fund', type=float, default=None,
                   help='Fundamental wavelength (μm)')
    p.add_argument('--crystal-length', type=float, default=None,
                   help='Crystal length (μm)')
    p.add_argument('--temperature', type=float, default=None,
                   help='Temperature (°C)')
    p.add_argument('--boost', type=float, default=None,
                   help='χ⁽²⁾ boost factor')
    p.add_argument('--peak-intensity', type=float, default=None,
                   help='Peak intensity (W/cm²)')
    p.add_argument('--ppw', type=int, default=None,
                   help='Points per SH wavelength (resolution)')
    p.add_argument('--pulse-width', type=float, default=None,
                   help='Pulse width (fs)')
    p.add_argument('--poling-period', type=float, default=None,
                   help='Poling period (μm), default=QPM')
    p.add_argument('--steps-per-frame', type=int, default=None,
                   help='Simulation steps per display frame')
    p.add_argument('--R-pump', type=float, default=None,
                   help='Pump face reflectivity (0=AR, 1=bare)')
    p.add_argument('--R-sh', type=float, default=None,
                   help='SH face reflectivity (0=AR, 1=bare)')
    p.add_argument('--spec-ref', type=float, default=None,
                   help='Spectrum reference level (log10 power, top of display)')
    p.add_argument('--spec-decades', type=float, default=None,
                   help='Spectrum display range (decades)')
    p.add_argument('--width', type=int, default=None,
                   help='Window width (pixels)')
    p.add_argument('--height', type=int, default=None,
                   help='Window height (pixels)')
    p.add_argument('--record-start', type=float, default=None,
                   help='Start recording snapshots at this time (ps)')
    p.add_argument('--record-end', type=float, default=None,
                   help='Stop recording at this time (ps)')
    p.add_argument('--record-interval', type=float, default=None,
                   help='Interval between snapshots (ps, default 0.01=10fs)')
    p.add_argument('--record-dir', type=str, default=None,
                   help='Directory for snapshot recordings')
    p.add_argument('--mr-interval', type=int, default=None,
                   help='Global MR projection every N steps (default 100)')
    return p


class Terminal:
    """Raw-mode terminal for non-blocking single-char reads."""

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
                seq = ''
                while select.select([sys.stdin], [], [], 0.05)[0]:
                    seq += sys.stdin.read(1)
                    if len(seq) > 4:
                        break
                if seq.startswith('['):
                    code = seq[1:2]
                    return {'A': 'UP', 'B': 'DOWN', 'C': 'RIGHT', 'D': 'LEFT'}.get(code)
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


def print_status(sim, steps_per_frame, fps, paused, show_help, energy_info, renderer=None):
    sys.stdout.write('\x1b[H')

    L_coh = coherence_length(sim.lambda_fund_um, sim.T)
    Lambda_qpm = 2 * L_coh
    dk = np.pi / L_coh - 2 * np.pi / sim.Lambda if sim.Lambda > 0 else 0

    E1, E2 = sim.E1, sim.E2
    max_e1 = float(E1.max()) if hasattr(E1, 'max') else 0
    max_e2 = float(E2.max()) if hasattr(E2, 'max') else 0

    e_now, e_ref, mr_now, mr_ref = energy_info
    e_ratio = f"{e_now / e_ref:.10f}" if e_ref > 0 else "—"
    mr_ratio = f"{mr_now / mr_ref:.10f}" if mr_ref > 0 else "—"

    n_domains = int(sim.crystal_length / (sim.Lambda / 2)) if sim.Lambda > 0 else 0

    spec_info = ""
    if renderer:
        spec_info = f"  spec: ref={renderer.spec_ref:.0f} range={renderer.spec_decades:.0f} decades"

    lines = [
        f"  PPLN 1D FDTD Explorer          {'[PAUSED]' if paused else ''}",
        f"  ────────────────────────────────────────────────────────────────",
        f"  t = {sim.current_time_ps:8.2f} ps     fps = {fps:5.1f}     steps/frame = {steps_per_frame}"
        f"     total = {sim.n_step:,}",
        f"  Λ = {sim.Lambda*1e6:7.2f} μm      Λ_QPM = {Lambda_qpm*1e6:.2f} μm     Δk = {dk*1e-6:.1f} /mm"
        f"     domains = {n_domains}",
        f"  T = {sim.T:5.0f} °C       boost = {sim.boost:.0f}×            pulse = {sim.pulse_width_s*1e15:.0f} fs"
        f"     I = {sim.peak_intensity_W_cm2:.0e} W/cm²     ppw = {sim.ppw}",
        f"  E₀ = {sim.E0:.2e} V/m    max|E₁| = {max_e1:.2e}   max|E₂| = {max_e2:.2e}"
        f"     R_pump={sim.R_pump[0]:.2f}  R_sh={sim.R_sh[0]:.2f}",
        f"  energy = {e_now:.6e}   E/E₀ = {e_ratio}"
        f"     MR/MR₀ = {mr_ratio}"
        f"{'   [NEAREST]' if renderer and renderer.nearest_mode else ''}",
        spec_info,
        f"",
    ]

    if show_help:
        lines.extend(HELP.split('\n'))

    for line in lines:
        sys.stdout.write(f"{line}\x1b[K\n")
    sys.stdout.write('\x1b[J')
    sys.stdout.flush()


SNAPSHOT_PATH = os.path.join(os.path.dirname(__file__), 'snapshot.npz')


def save_snapshot(sim):
    """Dump current simulation state to snapshot.npz."""
    import cupy as cp
    E1, E2 = sim.get_fields()
    z_um = sim.get_z_host()
    d_z = sim.d_z.get()

    # Spatial FFT (on GPU, bring to host)
    fft1 = cp.abs(cp.fft.rfft(sim.E1)).get()
    fft2 = cp.abs(cp.fft.rfft(sim.E2)).get()
    dk = 1.0 / (sim.Nz * sim.dz)
    fft_k = np.arange(len(fft1)) * dk

    np.savez(SNAPSHOT_PATH,
             E1=E1, E2=E2, z_um=z_um, d_z=d_z,
             fft1=fft1, fft2=fft2, fft_k=fft_k,
             t_ps=sim.current_time_ps,
             Lambda_um=sim.Lambda * 1e6,
             T_celsius=sim.T,
             boost=sim.boost,
             n1=sim.n1, n2=sim.n2,
             crystal_start=sim.crystal_start,
             crystal_end=sim.crystal_end,
             dz=sim.dz,
             R_pump=sim.R_pump,
             R_sh=sim.R_sh,
             pulse_width_fs=sim.pulse_width_s * 1e15)
    sys.stdout.write(f"\r  >> Snapshot saved to {SNAPSHOT_PATH}\x1b[K\n")
    sys.stdout.flush()


def main():
    parser = build_parser()
    args = parser.parse_args()
    cfg = load_config(args)

    # Build sim
    poling_m = cfg['poling_period'] * 1e-6 if cfg['poling_period'] else None
    sim = FDTDSimulation(
        lambda_fund_um=cfg['lambda_fund'],
        crystal_length_m=cfg['crystal_length'] * 1e-6,
        T_celsius=cfg['temperature'],
        boost=cfg['boost'],
        peak_intensity_W_cm2=cfg['peak_intensity'],
        ppw=cfg['ppw'],
        mr_interval=cfg['mr_interval'],
        pulse_width_fs=cfg['pulse_width'],
        poling_period_m=poling_m,
    )
    if cfg['R_pump'] > 0:
        sim.set_R_pump(left=cfg['R_pump'], right=cfg['R_pump'])
    if cfg['R_sh'] > 0:
        sim.set_R_sh(left=cfg['R_sh'], right=cfg['R_sh'])

    # ── Recording setup ──
    rec_start = cfg.get('record_start')
    rec_end = cfg.get('record_end')
    rec_interval = cfg.get('record_interval', 0.01)
    rec_dir = cfg.get('record_dir', 'recordings')
    recording = rec_start is not None
    rec_next_ps = rec_start if recording else None
    rec_count = 0

    if recording:
        os.makedirs(rec_dir, exist_ok=True)
        if rec_end is None:
            # Default: record for 2ps after start
            rec_end = rec_start + 2.0
        sys.stderr.write(f"Recording: {rec_start}–{rec_end} ps every {rec_interval} ps → {rec_dir}/\n")

    # ── Window ──
    if not glfw.init():
        raise RuntimeError("GLFW init failed")

    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
    glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
    glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, True)

    width, height = cfg['width'], cfg['height']
    window = glfw.create_window(width, height, "PPLN FDTD", None, None)
    if not window:
        glfw.terminate()
        raise RuntimeError("Window creation failed")

    glfw.make_context_current(window)
    glfw.swap_interval(1)

    ctx = moderngl.create_context()
    renderer = Renderer(ctx, width, height, sim,
                        spec_ref=cfg['spec_ref'],
                        spec_decades=cfg['spec_decades'])

    def on_resize(win, w, h):
        nonlocal width, height
        if w > 0 and h > 0:
            width, height = w, h
            renderer.resize(w, h)

    glfw.set_framebuffer_size_callback(window, on_resize)

    needs_redraw = [False]
    dragging = [False]
    drag_last_x = [0.0]

    def on_scroll(win, xoff, yoff):
        renderer.zoom(yoff)
        needs_redraw[0] = True

    def on_mouse_button(win, button, action, mods):
        if button == glfw.MOUSE_BUTTON_LEFT:
            if action == glfw.PRESS:
                dragging[0] = True
                x, _ = glfw.get_cursor_pos(win)
                drag_last_x[0] = x
            else:
                dragging[0] = False

    def on_cursor_pos(win, x, y):
        if dragging[0] and renderer.zoom_hi - renderer.zoom_lo < 0.99:
            dx_pixels = x - drag_last_x[0]
            drag_last_x[0] = x
            renderer.pan(-dx_pixels / width)
            needs_redraw[0] = True

    glfw.set_scroll_callback(window, on_scroll)
    glfw.set_mouse_button_callback(window, on_mouse_button)
    glfw.set_cursor_pos_callback(window, on_cursor_pos)

    # ── Terminal ──
    term = Terminal()
    term.clear_and_home()

    # ── State ──
    paused = False
    show_help = False
    steps_per_frame = cfg['steps_per_frame']
    spf_min, spf_max = 10, 100000
    pending_keys = []

    # ── GLFW key input (always active) ──
    GLFW_KEY_MAP = {
        glfw.KEY_SPACE: ' ', glfw.KEY_R: 'r', glfw.KEY_H: 'h',
        glfw.KEY_Q: 'q', glfw.KEY_ESCAPE: 'q',
        glfw.KEY_L: 'l', glfw.KEY_SEMICOLON: ';',
        glfw.KEY_T: 't', glfw.KEY_Y: 'y',
        glfw.KEY_B: 'b', glfw.KEY_N: 'n',
        glfw.KEY_P: 'p', glfw.KEY_O: 'o', glfw.KEY_F: 'f', glfw.KEY_S: 's',
        glfw.KEY_EQUAL: '+', glfw.KEY_MINUS: '-',
        glfw.KEY_KP_ADD: '+', glfw.KEY_KP_SUBTRACT: '-',
        glfw.KEY_1: '1', glfw.KEY_2: '2', glfw.KEY_3: '3', glfw.KEY_4: '4',
        glfw.KEY_LEFT_BRACKET: '[', glfw.KEY_RIGHT_BRACKET: ']',
    }

    def on_key(win, key, scancode, action, mods):
        if action in (glfw.PRESS, glfw.REPEAT):
            if key in GLFW_KEY_MAP:
                ch = GLFW_KEY_MAP[key]
                # Shift+[ = {, Shift+] = }
                if mods & glfw.MOD_SHIFT:
                    ch = {'[': '{', ']': '}'}.get(ch, ch)
                pending_keys.append(ch)

    glfw.set_key_callback(window, on_key)

    frame_count = 0
    fps_time = time.perf_counter()
    fps = 0.0
    energy_ref = 0.0
    energy_now = 0.0
    mr_ref = 0.0
    mr_now = 0.0
    energy_ref_set = False
    crystal_mid = (sim.crystal_start + sim.crystal_end) / 2
    v_vac = sim.courant
    v_xtal = sim.courant / sim.n1
    steps_to_mid = ((sim.crystal_start - sim.source_idx) / v_vac
                    + (crystal_mid - sim.crystal_start) / v_xtal)
    energy_ref_time_ps = (sim.t0 + steps_to_mid * sim.dt) * 1e12

    try:
        while not glfw.window_should_close(window):
            glfw.poll_events()

            # ── Input ──
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
                    renderer.reset_spectrum_scale()
                elif key == 'f':
                    renderer.toggle_nearest()
                    needs_redraw[0] = True
                elif key == '+' or key == '=':
                    steps_per_frame = min(int(steps_per_frame * 1.5), spf_max)
                elif key == '-' or key == '_':
                    steps_per_frame = max(int(steps_per_frame / 1.5), spf_min)
                elif key == 'l':
                    new_L = sim.Lambda - 10e-9
                    if new_L > 1e-6:
                        sim.rebuild_poling(new_L)
                        renderer.invalidate_poling_cache()
                elif key == ';':
                    sim.rebuild_poling(sim.Lambda + 10e-9)
                    renderer.invalidate_poling_cache()
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
                elif key == '1':
                    r = max(0.0, sim.R_pump[0] - 0.05)
                    sim.set_R_pump(left=r, right=r)
                elif key == '2':
                    r = min(1.0, sim.R_pump[0] + 0.05)
                    sim.set_R_pump(left=r, right=r)
                elif key == '3':
                    r = max(0.0, sim.R_sh[0] - 0.05)
                    sim.set_R_sh(left=r, right=r)
                elif key == '4':
                    r = min(1.0, sim.R_sh[0] + 0.05)
                    sim.set_R_sh(left=r, right=r)
                elif key == 's':
                    save_snapshot(sim)
                elif key == '[':
                    renderer.adjust_spec_ref(-1.0)
                    needs_redraw[0] = True
                elif key == ']':
                    renderer.adjust_spec_ref(1.0)
                    needs_redraw[0] = True
                elif key == '{':
                    renderer.adjust_spec_decades(-1.0)
                    needs_redraw[0] = True
                elif key == '}':
                    renderer.adjust_spec_decades(1.0)
                    needs_redraw[0] = True

            # ── Simulate ──
            if not paused:
                sim.step(steps_per_frame)
                E1, E2 = sim.get_fields()
                renderer.render(E1, E2)
                glfw.swap_buffers(window)

                # ── Recording ──
                if recording and rec_next_ps is not None:
                    t = sim.current_time_ps
                    if t >= rec_next_ps and t <= rec_end:
                        import cupy as cp
                        fft1 = cp.abs(cp.fft.rfft(sim.E1)).get()
                        fft2 = cp.abs(cp.fft.rfft(sim.E2)).get()
                        dk = 1.0 / (sim.Nz * sim.dz)
                        fft_k = np.arange(len(fft1)) * dk
                        fname = os.path.join(rec_dir, f'snap_{rec_count:05d}_{t:.4f}ps.npz')
                        np.savez(fname, E1=E1, E2=E2, z_um=sim.get_z_host(),
                                 d_z=sim.d_z.get(), fft1=fft1, fft2=fft2, fft_k=fft_k,
                                 t_ps=t, Lambda_um=sim.Lambda*1e6, T_celsius=sim.T,
                                 boost=sim.boost, n1=sim.n1, n2=sim.n2,
                                 crystal_start=sim.crystal_start,
                                 crystal_end=sim.crystal_end, dz=sim.dz,
                                 R_pump=sim.R_pump, R_sh=sim.R_sh,
                                 pulse_width_fs=sim.pulse_width_s*1e15,
                                 ppw=sim.ppw, peak_intensity=sim.peak_intensity_W_cm2)
                        rec_count += 1
                        rec_next_ps += rec_interval
                        sys.stderr.write(f"\r  rec #{rec_count}: t={t:.4f}ps → {fname}\033[K")
                        sys.stderr.flush()
                    elif t > rec_end:
                        rec_next_ps = None  # done recording
                        sys.stderr.write(f"\n  Recording complete: {rec_count} snapshots in {rec_dir}/\n")
                        sys.stderr.flush()
            else:
                if key or needs_redraw[0]:
                    E1, E2 = sim.get_fields()
                    renderer.render(E1, E2)
                    glfw.swap_buffers(window)
                    needs_redraw[0] = False
                time.sleep(0.05)

            # ── Terminal status ──
            frame_count += 1
            now = time.perf_counter()
            if now - fps_time >= 0.25:
                fps = frame_count / (now - fps_time)
                frame_count = 0
                fps_time = now

                energy_now = sim.get_energy()
                mr_now = sim.get_manley_rowe()
                if not energy_ref_set and sim.current_time_ps >= energy_ref_time_ps and energy_now > 0:
                    energy_ref = energy_now
                    mr_ref = mr_now
                    energy_ref_set = True

                print_status(sim, steps_per_frame, fps, paused, show_help,
                             (energy_now, energy_ref, mr_now, mr_ref), renderer)

    finally:
        term.restore()
        glfw.terminate()
        print()


if __name__ == "__main__":
    main()
