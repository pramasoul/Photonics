"""GPU-native OpenGL renderer for PPLN FDTD simulation.

Renders E-fields as color-intensity strips:
  - E₁ (fundamental) → red channel
  - E₂ (second harmonic) → blue channel
  - Positive = bright, negative = dark complement
  - Poling pattern as a thin alternating red/blue band
  - Crystal/vacuum regions distinguished by background
"""

import numpy as np
import moderngl


# ── Shaders ────────────────────────────────────────────────────────────

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

# Fragment shader for the field strip.
# Reads a 1D texture (Nx1 RGBA float) and maps to color.
FIELD_FRAG = """
#version 330
uniform sampler2D field_tex;
uniform float y_lo;       // bottom of this strip in NDC [0,1]
uniform float y_hi;       // top of this strip in NDC [0,1]
uniform float crystal_lo; // crystal left edge in UV space [0,1]
uniform float crystal_hi; // crystal right edge in UV space [0,1]
in vec2 uv;
out vec4 fragColor;

void main() {
    // Sample field texture: R=E1, G=0, B=E2, A=poling
    vec4 f = texture(field_tex, vec2(uv.x, 0.5));
    float e1 = f.r;
    float e2 = f.b;
    float poling = f.a;

    // Vertical position within the strip [0=bottom, 1=top]
    float vy = (uv.y - y_lo) / (y_hi - y_lo);

    // Background: dark gray in crystal, darker outside
    bool in_crystal = uv.x >= crystal_lo && uv.x <= crystal_hi;
    vec3 bg = in_crystal ? vec3(0.08) : vec3(0.03);

    // Poling strip (bottom 8% of field area)
    if (vy < 0.08) {
        vec3 pol_color = poling > 0.0 ? vec3(0.6, 0.15, 0.1) : vec3(0.1, 0.15, 0.6);
        if (!in_crystal) pol_color = bg;
        fragColor = vec4(pol_color, 1.0);
        return;
    }

    // Thin separator line
    if (vy < 0.09) {
        fragColor = vec4(0.2, 0.2, 0.2, 1.0);
        return;
    }

    // Field visualization: map E to color intensity
    // E1 → red/cyan:  positive = red, negative = cyan
    // E2 → blue/yellow: positive = blue, negative = yellow
    vec3 c = bg;

    // Fundamental (ω) — red/cyan
    if (e1 > 0.0) {
        c.r += e1;
    } else {
        c.g += -e1 * 0.5;
        c.b += -e1 * 0.5;
    }

    // Second harmonic (2ω) — blue/yellow
    if (e2 > 0.0) {
        c.b += e2;
    } else {
        c.r += -e2 * 0.4;
        c.g += -e2 * 0.4;
    }

    fragColor = vec4(clamp(c, 0.0, 1.0), 1.0);
}
"""

# Simple passthrough for the info bar background
INFO_FRAG = """
#version 330
uniform vec3 bg_color;
in vec2 uv;
out vec4 fragColor;
void main() {
    // subtle vertical gradient using uv.y
    float grad = mix(0.9, 1.0, uv.y);
    fragColor = vec4(bg_color * grad, 1.0);
}
"""

# Text rendering: textured quad with alpha from font atlas
TEXT_FRAG = """
#version 330
uniform sampler2D text_tex;
uniform vec3 text_color;
in vec2 uv;
out vec4 fragColor;
void main() {
    float a = texture(text_tex, uv).r;
    fragColor = vec4(text_color, a);
}
"""


class Renderer:
    def __init__(self, ctx: moderngl.Context, width: int, height: int, sim):
        self.ctx = ctx
        self.width = width
        self.height = height
        self.sim = sim

        # Field strip occupies top 75% of window
        self.field_top = 1.0
        self.field_bottom = -0.5  # in NDC [-1, 1]

        # Layout in UV y-space (0=bottom of strip, 1=top)
        self.field_y_lo = 0.0
        self.field_y_hi = 1.0

        # Crystal position in UV x-space [0, 1]
        Nz = sim.Nz
        self.crystal_lo = sim.crystal_start / Nz
        self.crystal_hi = sim.crystal_end / Nz

        # ── Field texture (downsample to fit GL_MAX_TEXTURE_SIZE) ──
        max_tex = ctx.info['GL_MAX_TEXTURE_SIZE']
        self.tex_width = min(Nz, max_tex, width * 2)  # 2x window width is plenty
        self.field_data = np.zeros((1, self.tex_width, 4), dtype=np.float32)
        self.field_tex = ctx.texture((self.tex_width, 1), 4, dtype='f4')
        self.field_tex.filter = (moderngl.LINEAR, moderngl.LINEAR)

        # ── Shader programs ──
        self.field_prog = ctx.program(
            vertex_shader=VERT_SHADER,
            fragment_shader=FIELD_FRAG,
        )
        self.field_prog['field_tex'] = 0
        self.field_prog['y_lo'] = 0.0
        self.field_prog['y_hi'] = 1.0
        self.field_prog['crystal_lo'] = self.crystal_lo
        self.field_prog['crystal_hi'] = self.crystal_hi

        self.info_prog = ctx.program(
            vertex_shader=VERT_SHADER,
            fragment_shader=INFO_FRAG,
        )

        self.text_prog = ctx.program(
            vertex_shader=VERT_SHADER,
            fragment_shader=TEXT_FRAG,
        )
        self.text_prog['text_tex'] = 1
        self.text_prog['text_color'] = (0.9, 0.9, 0.9)

        # ── Fullscreen quad for field ──
        fb = self.field_bottom
        ft = self.field_top
        verts = np.array([
            # pos (x,y)      uv (u,v)
            -1, fb,   0, 0,
             1, fb,   1, 0,
             1, ft,   1, 1,
            -1, fb,   0, 0,
             1, ft,   1, 1,
            -1, ft,   0, 1,
        ], dtype='f4')
        vbo = ctx.buffer(verts)
        self.field_vao = ctx.vertex_array(self.field_prog, [(vbo, '2f 2f', 'in_pos', 'in_uv')])

        # ── Info bar quad (bottom 25%) ──
        ib = -1.0
        it = fb
        info_verts = np.array([
            -1, ib, 0, 0,
             1, ib, 1, 0,
             1, it, 1, 1,
            -1, ib, 0, 0,
             1, it, 1, 1,
            -1, it, 0, 1,
        ], dtype='f4')
        info_vbo = ctx.buffer(info_verts)
        self.info_vao = ctx.vertex_array(self.info_prog, [(info_vbo, '2f 2f', 'in_pos', 'in_uv')])
        self.info_prog['bg_color'] = (0.12, 0.12, 0.12)

        # ── Text rendering setup ──
        self._text_textures = {}  # cache rendered text
        self._text_vao_cache = {}

        # ── Probe marker position ──
        self.probe_x_ndc = 2.0 * (sim.probe_idx / Nz) - 1.0

        ctx.enable(moderngl.BLEND)
        ctx.blend_func = (moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA)

    def update_field_texture(self, E1: np.ndarray, E2: np.ndarray):
        """Downsample fields and upload to GPU texture."""
        Nz = len(E1)
        tw = self.tex_width
        data = self.field_data

        # Downsample by reshaping into bins and taking max-abs per bin
        # (preserves carrier oscillation peaks better than averaging)
        if Nz != tw:
            indices = np.linspace(0, Nz - 1, tw).astype(int)
            e1 = E1[indices]
            e2 = E2[indices]
            d_z = self.sim.d_z.get()[indices]
        else:
            e1, e2 = E1, E2
            d_z = self.sim.d_z.get()

        # Normalize fields for display
        max_e1 = max(np.max(np.abs(e1)), 1e-10)
        max_e2 = max(np.max(np.abs(e2)), 1e-10)
        scale1 = min(max_e1, 2.0)
        scale2 = max(scale1 * 0.3, max_e2)  # E2 gets more gain

        data[0, :, 0] = e1 / scale1
        data[0, :, 1] = 0
        data[0, :, 2] = e2 / scale2
        data[0, :, 3] = d_z

        self.field_tex.write(data.tobytes())

    def render_text(self, text: str, x: float, y: float, scale: float = 1.0,
                    color: tuple = (0.9, 0.9, 0.9)):
        """Render text at NDC position (x, y) using Pillow for rasterization."""
        key = (text, scale)
        if key not in self._text_textures:
            self._rasterize_text(text, scale, key)

        tex, tw, th = self._text_textures[key]

        # Convert pixel size to NDC
        w_ndc = 2.0 * tw / self.width * scale
        h_ndc = 2.0 * th / self.height * scale

        verts = np.array([
            x,         y,         0, 1,
            x + w_ndc, y,         1, 1,
            x + w_ndc, y + h_ndc, 1, 0,
            x,         y,         0, 1,
            x + w_ndc, y + h_ndc, 1, 0,
            x,         y + h_ndc, 0, 0,
        ], dtype='f4')

        vao_key = (text, scale, x, y)
        if vao_key in self._text_vao_cache:
            vao = self._text_vao_cache[vao_key]
            vao.release()
        vbo = self.ctx.buffer(verts)
        vao = self.ctx.vertex_array(self.text_prog, [(vbo, '2f 2f', 'in_pos', 'in_uv')])
        self._text_vao_cache[vao_key] = vao

        self.text_prog['text_color'] = color
        tex.use(1)
        vao.render()

    def _rasterize_text(self, text: str, scale: float, key):
        """Render text string to a GL texture using Pillow."""
        from PIL import Image, ImageDraw, ImageFont
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
                                      int(14 * scale))
        except (OSError, IOError):
            font = ImageFont.load_default()

        # Measure text
        bbox = font.getbbox(text)
        tw = bbox[2] - bbox[0] + 4
        th = bbox[3] - bbox[1] + 4

        img = Image.new('L', (tw, th), 0)
        draw = ImageDraw.Draw(img)
        draw.text((2 - bbox[0], 2 - bbox[1]), text, fill=255, font=font)

        tex = self.ctx.texture((tw, th), 1, img.tobytes())
        tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self._text_textures[key] = (tex, tw, th)

    def render(self, E1, E2, info_lines: list[str], help_active: bool = False,
               help_text: str = ""):
        """Render one frame."""
        self.ctx.clear(0.05, 0.05, 0.05)

        # ── Field strip ──
        self.update_field_texture(E1, E2)
        self.field_tex.use(0)
        self.field_vao.render()

        # ── Info bar background ──
        self.info_vao.render()

        # ── Info text ──
        y_start = -0.55
        line_h = 0.09
        for i, line in enumerate(info_lines):
            self.render_text(line, -0.98, y_start - i * line_h, scale=1.0)

        # ── Probe marker (thin green line in field area) ──
        # Draw as a thin colored text character at the probe position
        probe_uv = self.sim.probe_idx / self.sim.Nz
        probe_ndc_x = -1.0 + 2.0 * probe_uv
        self.render_text("|", probe_ndc_x - 0.003, self.field_bottom + 0.02,
                         scale=1.0, color=(0.0, 0.8, 0.0))

        # ── Help overlay ──
        if help_active:
            # Semi-transparent background would need another quad;
            # just render text over the field area
            lines = help_text.split('\n')
            for i, line in enumerate(lines):
                self.render_text(line, -0.95, 0.85 - i * 0.07,
                                 scale=1.0, color=(1.0, 1.0, 0.6))

    def resize(self, width: int, height: int):
        self.width = width
        self.height = height
        self.ctx.viewport = (0, 0, width, height)
        # Clear text caches on resize
        self._text_textures.clear()
        for vao in self._text_vao_cache.values():
            vao.release()
        self._text_vao_cache.clear()
