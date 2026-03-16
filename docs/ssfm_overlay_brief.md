# SSFM ↔ FDTD Overlay — Brief for CC

## Goal

Overlay the SSFM-predicted envelopes on the FDTD GL scope view, so
you can see the SSFM "correct answer" directly on top of the
carrier-resolved fields in real time.

## How

1. **Run the SSFM once at startup** (or when parameters change) for
   the current configuration. This takes <1 ms. Store A1_exit(t)
   and A2_exit(t) — the complex envelopes at the crystal exit face.

2. **Convert temporal envelopes to spatial envelopes.** In vacuum
   the mapping is just a coordinate transform:

   ```python
   # z_exit = position of crystal exit face
   # For each point on the FDTD z-axis (in vacuum, z > z_exit):
   t_retarded = (z - z_exit) / (c / n)  # in crystal: c/n, in vacuum: c
   A1_spatial[z] = A1_exit(t_exit - t_retarded)  # interpolate
   A2_spatial[z] = A2_exit(t_exit - t_retarded)
   ```

   Inside the crystal, the mapping uses the group velocity:
   ```python
   t_retarded = (z - z_entry) / v_g
   ```

   Use `np.interp` on the SSFM time grid to get values at the
   FDTD z positions.

3. **Plot |A1_spatial| and |A2_spatial| as dashed lines** in the
   top panel of the scope, overlaid on the solid FDTD envelopes.
   Match colors: red dashed for SSFM pump, blue dashed for SSFM SH.
   The SSFM curves should track the FDTD envelopes closely where
   the FDTD is accurate, and diverge where it isn't.

4. **Track the pulse position.** The SSFM envelope needs to move
   with the pulse. At simulation time t, the pulse leading edge is
   at z ≈ z_source + (c/n₁) · t (inside crystal) or
   z_exit + c · (t - t_exit) (in vacuum). Shift the SSFM spatial
   envelope accordingly each frame.

## Implementation notes

- The SSFM propagation is one-shot: run it once when parameters
  change, cache the exit envelopes. Don't re-run every frame.
- Recompute only when a slider changes (Λ, T, intensity, pulse
  width, crystal length).
- The SSFM also gives the envelope *inside* the crystal at every
  spatial step (from the propagation loop). Cache the full
  A1(z), A2(z) spatial record for overlay while the FDTD pulse
  is still inside the crystal.
- Toggle the overlay with 'v' (validate/overlay).

## What this shows

- Where FDTD and SSFM agree: the FDTD is accurate there.
- Where they disagree: visible as the dashed line diverging from
  the solid line. The gap is the FDTD's numerical error.
- The SSFM envelope moves at the correct group velocity; if the
  FDTD pulse drifts ahead or behind, that's a group velocity
  error (absent with the dispersion correction, visible without).
