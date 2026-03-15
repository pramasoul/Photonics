"""GPU-native OpenGL renderer for PPLN FDTD simulation.

Pure visual display — no text rendering. All text goes to the terminal.

Top panel:  spatial FFT spectrum (line plot, log scale)
Bottom panel: E-field color intensity + poling pattern
"""

import numpy as np
import cupy as cp
import moderngl


# ── Shared vertex shader ───────────────────────────────────────────────

VERT_SHADER = """
#version 330
in vec2 in_pos;
in vec2 in_uv;
out vec2 uv;
void main() {
    gl_Position = vec4(in_pos, 0.0, 1.0);
    uv = in_uv;
}
"""

# ── Field display (textured quad) ──────────────────────────────────────

FIELD_FRAG = """
#version 330
uniform sampler2D field_tex;
uniform float crystal_lo;  // crystal edges in screen [0,1] space (adjusted for zoom)
uniform float crystal_hi;
in vec2 uv;
out vec4 fragColor;

void main() {
    vec4 f = texture(field_tex, vec2(uv.x, 0.5));
    float e1 = f.r;
    float e2 = f.b;
    float poling = f.a;

    bool in_crystal = uv.x >= crystal_lo && uv.x <= crystal_hi;
    vec3 bg = in_crystal ? vec3(0.08) : vec3(0.03);

    // Poling strip (bottom 8%)
    if (uv.y < 0.08) {
        vec3 pol = poling > 0.0 ? vec3(0.6, 0.15, 0.1) : vec3(0.1, 0.15, 0.6);
        if (!in_crystal) pol = bg;
        fragColor = vec4(pol, 1.0);
        return;
    }
    if (uv.y < 0.085) {
        fragColor = vec4(0.25, 0.25, 0.25, 1.0);
        return;
    }

    vec3 c = bg;
    if (e1 > 0.0) c.r += e1;
    else { c.g += -e1 * 0.5; c.b += -e1 * 0.5; }
    if (e2 > 0.0) c.b += e2;
    else { c.r += -e2 * 0.4; c.g += -e2 * 0.4; }

    fragColor = vec4(clamp(c, 0.0, 1.0), 1.0);
}
"""

# ── Spectrum background ───────────────────────────────────────────────

SPEC_BG_FRAG = """
#version 330
in vec2 uv;
out vec4 fragColor;
void main() {
    float g = mix(0.05, 0.07, uv.y);
    fragColor = vec4(vec3(g), 1.0);
}
"""

# ── Spectrum line (simple colored line) ────────────────────────────────

LINE_VERT = """
#version 330
in vec2 in_pos;
void main() {
    gl_Position = vec4(in_pos, 0.0, 1.0);
}
"""

LINE_FRAG = """
#version 330
uniform vec3 line_color;
out vec4 fragColor;
void main() {
    fragColor = vec4(line_color, 1.0);
}
"""


class Renderer:
    # Layout: spectrum takes most of the window, field is a thin strip at bottom
    SPEC_TOP = 1.0      # NDC
    SPEC_BOT = -0.6
    FIELD_TOP = -0.62
    FIELD_BOT = -1.0

    def __init__(self, ctx: moderngl.Context, width: int, height: int, sim,
                 spec_ref: float = 4.0, spec_decades: float = 8.0):
        self.ctx = ctx
        self.width = width
        self.height = height
        self.sim = sim

        # Zoom state: [0,1] range in UV space
        self.zoom_lo = 0.0
        self.zoom_hi = 1.0

        Nz = sim.Nz
        self.crystal_lo = sim.crystal_start / Nz
        self.crystal_hi = sim.crystal_end / Nz

        # ── Field texture (rebuilt each frame to match visible range) ──
        self.max_tex = ctx.info['GL_MAX_TEXTURE_SIZE']
        self.nearest_mode = False
        self._tex_width_current = 0
        self.field_tex = None
        self.field_data = None
        self._d_z_cache = None  # cached poling data on host

        # ── Field shader + quad ──
        self.field_prog = ctx.program(vertex_shader=VERT_SHADER, fragment_shader=FIELD_FRAG)
        self.field_prog['field_tex'] = 0

        fb, ft = self.FIELD_BOT, self.FIELD_TOP
        fv = np.array([
            -1, fb, 0, 0,  1, fb, 1, 0,  1, ft, 1, 1,
            -1, fb, 0, 0,  1, ft, 1, 1, -1, ft, 0, 1,
        ], dtype='f4')
        self.field_vao = ctx.vertex_array(self.field_prog,
                                          [(ctx.buffer(fv), '2f 2f', 'in_pos', 'in_uv')])

        # ── Spectrum background quad ──
        self.spec_bg_prog = ctx.program(vertex_shader=VERT_SHADER, fragment_shader=SPEC_BG_FRAG)
        sb, st = self.SPEC_BOT, self.SPEC_TOP
        sv = np.array([
            -1, sb, 0, 0,  1, sb, 1, 0,  1, st, 1, 1,
            -1, sb, 0, 0,  1, st, 1, 1, -1, st, 0, 1,
        ], dtype='f4')
        self.spec_bg_vao = ctx.vertex_array(self.spec_bg_prog,
                                            [(ctx.buffer(sv), '2f 2f', 'in_pos', 'in_uv')])

        # ── Spectrum line shader ──
        self.line_prog = ctx.program(vertex_shader=LINE_VERT, fragment_shader=LINE_FRAG)

        # Spectrum line buffers (updated each frame)
        self.spec_npts = min(width, 2048)
        self._spec_vbo1 = ctx.buffer(reserve=self.spec_npts * 8)  # 2 floats * 4 bytes
        self._spec_vbo2 = ctx.buffer(reserve=self.spec_npts * 8)
        self._spec_vao1 = ctx.vertex_array(self.line_prog,
                                           [(self._spec_vbo1, '2f', 'in_pos')])
        self._spec_vao2 = ctx.vertex_array(self.line_prog,
                                           [(self._spec_vbo2, '2f', 'in_pos')])

        # ── Separator line between panels ──
        sep_y = (self.SPEC_BOT + self.FIELD_TOP) / 2
        sep_v = np.array([-1, sep_y, 1, sep_y], dtype='f4')
        self._sep_vbo = ctx.buffer(sep_v)
        self._sep_vao = ctx.vertex_array(self.line_prog,
                                         [(self._sep_vbo, '2f', 'in_pos')])

        # ── Precompute FFT frequency axis ──
        # Spatial FFT of E-field: freq in cycles/meter
        dk = 1.0 / (Nz * sim.dz)  # frequency resolution
        self._fft_freqs_full = cp.arange(Nz // 2) * dk  # positive frequencies
        # Expected k-vectors for reference lines
        self._k1 = sim.n1 / (sim.lambda_fund_um * 1e-6)  # cycles/m
        self._k2 = sim.n2 / (sim.lambda_fund_um * 0.5e-6)
        # Display range: 0 to 1.5 * k2
        self._k_max = 1.5 * self._k2
        # Spectrum scale (adjustable at runtime)
        self.spec_ref = spec_ref      # top of display in log10(power)
        self.spec_decades = spec_decades
        self._spec_vmax = self.spec_ref
        self._spec_vmin = self.spec_ref - self.spec_decades

    def adjust_spec_ref(self, delta: float):
        """Shift spectrum reference level by delta decades."""
        self.spec_ref += delta
        self._spec_vmax = self.spec_ref
        self._spec_vmin = self.spec_ref - self.spec_decades

    def adjust_spec_decades(self, delta: float):
        """Adjust spectrum range by delta decades (min 2)."""
        self.spec_decades = max(2.0, self.spec_decades + delta)
        self._spec_vmin = self.spec_ref - self.spec_decades

    def _compute_spectrum(self):
        """Compute spatial FFT of E1 and E2 on GPU, return downsampled log magnitudes."""
        E1_gpu = self.sim.E1
        E2_gpu = self.sim.E2

        # FFT on GPU
        fft1 = cp.abs(cp.fft.rfft(E1_gpu))
        fft2 = cp.abs(cp.fft.rfft(E2_gpu))

        # Trim to display range
        Nf = len(fft1)
        freqs = self._fft_freqs_full[:Nf]
        mask = freqs <= self._k_max
        n_keep = int(cp.sum(mask))
        if n_keep < 2:
            return None, None

        s1 = fft1[:n_keep].get()
        s2 = fft2[:n_keep].get()

        # Power spectrum (|FFT|²), log10 scale
        floor = 1e-20
        s1 = np.log10(np.maximum(s1 ** 2, floor))
        s2 = np.log10(np.maximum(s2 ** 2, floor))

        vmax = self._spec_vmax
        vmin = self._spec_vmin
        s1 = np.clip((s1 - vmin) / (vmax - vmin), 0, 1)
        s2 = np.clip((s2 - vmin) / (vmax - vmin), 0, 1)

        # Downsample to spec_npts
        npts = self.spec_npts
        if len(s1) > npts:
            idx = np.linspace(0, len(s1) - 1, npts).astype(int)
            s1 = s1[idx]
            s2 = s2[idx]

        return s1, s2

    def _build_line_verts(self, spectrum: np.ndarray) -> np.ndarray:
        """Convert normalized spectrum [0,1] to NDC vertex positions in spectrum panel."""
        n = len(spectrum)
        x = np.linspace(-1.0, 1.0, n).astype(np.float32)
        # Map spectrum [0,1] to NDC y in [SPEC_BOT, SPEC_TOP] with margin
        margin = 0.05 * (self.SPEC_TOP - self.SPEC_BOT)
        y_lo = self.SPEC_BOT + margin
        y_hi = self.SPEC_TOP - margin
        y = (y_lo + spectrum * (y_hi - y_lo)).astype(np.float32)
        return np.column_stack([x, y]).astype(np.float32)

    def _ensure_d_z_cache(self):
        if self._d_z_cache is None:
            self._d_z_cache = self.sim.d_z.get()

    def invalidate_poling_cache(self):
        self._d_z_cache = None

    def reset_spectrum_scale(self):
        pass  # scale is user-controlled now

    def update_field_texture(self, E1: np.ndarray, E2: np.ndarray):
        """Extract visible zoom range and upload at up to 1:1 resolution."""
        Nz = len(E1)
        self._ensure_d_z_cache()

        # Visible cell range
        i_lo = int(self.zoom_lo * Nz)
        i_hi = min(int(np.ceil(self.zoom_hi * Nz)), Nz)
        n_visible = i_hi - i_lo

        # Texture width: use full cell count if it fits, else downsample to screen width
        tw = min(n_visible, self.max_tex, max(self.width, 1024))

        # Reallocate texture if size changed
        if tw != self._tex_width_current:
            if self.field_tex is not None:
                self.field_tex.release()
            self.field_tex = self.ctx.texture((tw, 1), 4, dtype='f4')
            mode = moderngl.NEAREST if self.nearest_mode else moderngl.LINEAR
            self.field_tex.filter = (mode, mode)
            self.field_data = np.zeros((1, tw, 4), dtype=np.float32)
            self._tex_width_current = tw

        # Extract visible slice, downsample if needed
        if n_visible > tw:
            idx = np.linspace(i_lo, i_hi - 1, tw).astype(int)
        else:
            idx = np.arange(i_lo, i_hi)
            # Pad if rounding gave fewer cells than texture width
            if len(idx) < tw:
                idx = np.linspace(i_lo, i_hi - 1, tw).astype(int)

        e1 = E1[idx]
        e2 = E2[idx]
        d_z = self._d_z_cache[idx]

        max_e1 = max(np.max(np.abs(e1)), 1e-10)
        max_e2 = max(np.max(np.abs(e2)), 1e-10)
        scale1 = min(max_e1, 2.0)
        scale2 = max(scale1 * 0.3, max_e2)

        data = self.field_data
        data[0, :len(idx), 0] = e1 / scale1
        data[0, :len(idx), 1] = 0
        data[0, :len(idx), 2] = e2 / scale2
        data[0, :len(idx), 3] = d_z

        self.field_tex.write(data.tobytes())

        # Update crystal markers relative to current zoom window
        zoom_span = self.zoom_hi - self.zoom_lo
        self.field_prog['crystal_lo'] = (self.crystal_lo - self.zoom_lo) / zoom_span
        self.field_prog['crystal_hi'] = (self.crystal_hi - self.zoom_lo) / zoom_span

    def render(self, E1: np.ndarray, E2: np.ndarray):
        """Render one frame: spectrum panel + field panel."""
        self.ctx.clear(0.05, 0.05, 0.05)

        # ── Field panel ──
        self.update_field_texture(E1, E2)
        self.field_tex.use(0)
        self.field_vao.render()

        # ── Spectrum panel background ──
        self.spec_bg_vao.render()

        # ── Separator ──
        self.line_prog['line_color'] = (0.3, 0.3, 0.3)
        self._sep_vao.render(moderngl.LINES)

        # ── 10 dB grid lines ──
        self._draw_db_grid()

        # ── Spectrum lines ──
        s1, s2 = self._compute_spectrum()
        if s1 is not None:
            # E1 spectrum (red)
            verts1 = self._build_line_verts(s1)
            self._spec_vbo1.orphan(len(verts1.tobytes()))
            self._spec_vbo1.write(verts1.tobytes())
            self.line_prog['line_color'] = (0.9, 0.2, 0.2)
            self._spec_vao1.render(moderngl.LINE_STRIP, vertices=len(verts1))

            # E2 spectrum (blue)
            verts2 = self._build_line_verts(s2)
            self._spec_vbo2.orphan(len(verts2.tobytes()))
            self._spec_vbo2.write(verts2.tobytes())
            self.line_prog['line_color'] = (0.3, 0.4, 1.0)
            self._spec_vao2.render(moderngl.LINE_STRIP, vertices=len(verts2))

            # Reference lines for k₁ and k₂
            self._draw_ref_line(self._k1, (0.5, 0.2, 0.2))
            self._draw_ref_line(self._k2, (0.2, 0.2, 0.5))

    def _draw_db_grid(self):
        """Draw horizontal 1px lines at 10 dB intervals in the spectrum panel."""
        vmin, vmax = self._spec_vmin, self._spec_vmax
        if vmax <= vmin:
            return
        margin = 0.05 * (self.SPEC_TOP - self.SPEC_BOT)
        y_lo = self.SPEC_BOT + margin
        y_hi = self.SPEC_TOP - margin
        span = vmax - vmin

        # 10 dB = 1.0 in log10(power) units
        # Find grid lines at integer multiples of 1.0 (= 10 dB)
        db_step = 1.0  # 10 dB in log10 power
        first = np.ceil(vmin / db_step) * db_step
        verts = []
        for level in np.arange(first, vmax, db_step):
            frac = (level - vmin) / span
            if frac < 0.01 or frac > 0.99:
                continue
            y_ndc = y_lo + frac * (y_hi - y_lo)
            verts.extend([-1.0, y_ndc, 1.0, y_ndc])

        if verts:
            v = np.array(verts, dtype='f4')
            vbo = self.ctx.buffer(v)
            vao = self.ctx.vertex_array(self.line_prog, [(vbo, '2f', 'in_pos')])
            self.line_prog['line_color'] = (0.2, 0.2, 0.2)
            vao.render(moderngl.LINES)
            vao.release()
            vbo.release()

    def _draw_ref_line(self, k_val: float, color: tuple):
        """Draw a vertical dashed reference line at wavenumber k_val."""
        x_ndc = -1.0 + 2.0 * (k_val / self._k_max)
        if x_ndc < -1 or x_ndc > 1:
            return
        # Draw as a simple vertical line
        verts = np.array([x_ndc, self.SPEC_BOT, x_ndc, self.SPEC_TOP], dtype='f4')
        vbo = self.ctx.buffer(verts)
        vao = self.ctx.vertex_array(self.line_prog, [(vbo, '2f', 'in_pos')])
        self.line_prog['line_color'] = color
        vao.render(moderngl.LINES)
        vao.release()
        vbo.release()

    def zoom(self, scroll_y: float):
        """Zoom in/out centered on the current view midpoint."""
        factor = 0.85 if scroll_y > 0 else 1.0 / 0.85
        center = (self.zoom_lo + self.zoom_hi) / 2
        half = (self.zoom_hi - self.zoom_lo) / 2 * factor
        self.zoom_lo = max(0.0, center - half)
        self.zoom_hi = min(1.0, center + half)
        if self.zoom_hi - self.zoom_lo > 0.99:
            self.zoom_lo = 0.0
            self.zoom_hi = 1.0

    def toggle_nearest(self):
        """Toggle between linear interpolation and nearest-neighbor (hard cell boundaries)."""
        self.nearest_mode = not self.nearest_mode
        if self.field_tex is not None:
            mode = moderngl.NEAREST if self.nearest_mode else moderngl.LINEAR
            self.field_tex.filter = (mode, mode)

    def pan(self, dx_frac: float):
        """Pan the view by dx_frac of the current view width."""
        span = self.zoom_hi - self.zoom_lo
        dx = dx_frac * span
        if self.zoom_lo + dx < 0:
            dx = -self.zoom_lo
        elif self.zoom_hi + dx > 1:
            dx = 1.0 - self.zoom_hi
        self.zoom_lo += dx
        self.zoom_hi += dx

    def resize(self, width: int, height: int):
        self.width = width
        self.height = height
        self.ctx.viewport = (0, 0, width, height)
