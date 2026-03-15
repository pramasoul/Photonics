"""Re-export shared materials from common package."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from common.materials import (
    sellmeier_n, coherence_length, qpm_period, make_poling_pattern,
    D33, C, wavenumber, group_index, group_velocity, gvd,
    dispersion_operator,
)
