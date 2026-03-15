"""GPU-native OpenGL renderer for PPLN FDTD simulation.

Pure visual display — no text rendering. All text goes to the terminal.

Renders E-fields as color-intensity strips:
  - E₁ (fundamental) → red/cyan (positive/negative)
  - E₂ (second harmonic) → blue/yellow (positive/negative)
  - Poling pattern as alternating red/blue bands
  - Crystal vs vacuum distinguished by background brightness
"""

import numpy as np
import moderngl


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

FIELD_FRAG = """
#version 330
uniform sampler2D field_tex;
uniform float crystal_lo;
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

    // Poling strip (bottom 6%)
    if (uv.y < 0.06) {
        vec3 pol = poling > 0.0 ? vec3(0.6, 0.15, 0.1) : vec3(0.1, 0.15, 0.6);
        if (!in_crystal) pol = bg;
        fragColor = vec4(pol, 1.0);
        return;
    }
    // Separator
    if (uv.y < 0.065) {
        fragColor = vec4(0.25, 0.25, 0.25, 1.0);
        return;
    }

    // Field color mapping
    vec3 c = bg;

    // Fundamental: positive=red, negative=cyan
    if (e1 > 0.0) c.r += e1;
    else { c.g += -e1 * 0.5; c.b += -e1 * 0.5; }

    // Second harmonic: positive=blue, negative=yellow
    if (e2 > 0.0) c.b += e2;
    else { c.r += -e2 * 0.4; c.g += -e2 * 0.4; }

    fragColor = vec4(clamp(c, 0.0, 1.0), 1.0);
}
"""


class Renderer:
    def __init__(self, ctx: moderngl.Context, width: int, height: int, sim):
        self.ctx = ctx
        self.width = width
        self.height = height
        self.sim = sim

        Nz = sim.Nz
        self.crystal_lo = sim.crystal_start / Nz
        self.crystal_hi = sim.crystal_end / Nz

        # Field texture — downsample to fit GL limits and window
        max_tex = ctx.info['GL_MAX_TEXTURE_SIZE']
        self.tex_width = min(Nz, max_tex, max(width * 2, 4096))
        self.field_data = np.zeros((1, self.tex_width, 4), dtype=np.float32)
        self.field_tex = ctx.texture((self.tex_width, 1), 4, dtype='f4')
        self.field_tex.filter = (moderngl.LINEAR, moderngl.LINEAR)

        # Shader
        self.prog = ctx.program(vertex_shader=VERT_SHADER, fragment_shader=FIELD_FRAG)
        self.prog['field_tex'] = 0
        self.prog['crystal_lo'] = self.crystal_lo
        self.prog['crystal_hi'] = self.crystal_hi

        # Fullscreen quad
        verts = np.array([
            -1, -1,  0, 0,
             1, -1,  1, 0,
             1,  1,  1, 1,
            -1, -1,  0, 0,
             1,  1,  1, 1,
            -1,  1,  0, 1,
        ], dtype='f4')
        vbo = ctx.buffer(verts)
        self.vao = ctx.vertex_array(self.prog, [(vbo, '2f 2f', 'in_pos', 'in_uv')])

    def update_field_texture(self, E1: np.ndarray, E2: np.ndarray):
        """Downsample fields and upload to GPU texture."""
        Nz = len(E1)
        tw = self.tex_width
        data = self.field_data

        if Nz != tw:
            idx = np.linspace(0, Nz - 1, tw).astype(int)
            e1 = E1[idx]
            e2 = E2[idx]
            d_z = self.sim.d_z.get()[idx]
        else:
            e1, e2 = E1, E2
            d_z = self.sim.d_z.get()

        max_e1 = max(np.max(np.abs(e1)), 1e-10)
        max_e2 = max(np.max(np.abs(e2)), 1e-10)
        scale1 = min(max_e1, 2.0)
        scale2 = max(scale1 * 0.3, max_e2)

        data[0, :, 0] = e1 / scale1
        data[0, :, 1] = 0
        data[0, :, 2] = e2 / scale2
        data[0, :, 3] = d_z

        self.field_tex.write(data.tobytes())

    def render(self, E1: np.ndarray, E2: np.ndarray):
        """Render one frame."""
        self.ctx.clear(0.05, 0.05, 0.05)
        self.update_field_texture(E1, E2)
        self.field_tex.use(0)
        self.vao.render()

    def resize(self, width: int, height: int):
        self.width = width
        self.height = height
        self.ctx.viewport = (0, 0, width, height)
