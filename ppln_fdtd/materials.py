"""Sellmeier equation and poling pattern for MgO:LiNbO3 (PPLN)."""

import numpy as np

# Gayer et al. 2008 coefficients for 5% MgO:LiNbO3 extraordinary ray
# Full equation: n² = (a1+b1f) + (a2+b2f)/(λ²-(a3+b3f)²) + (a4+b4f)/(λ²-a5²) - a6λ²
_A1 = 5.756
_A2 = 0.0983
_A3 = 0.2020
_A4 = 189.32
_A5 = 12.52
_A6 = 1.32e-2
_B1 = 2.860e-6
_B2 = 4.700e-8
_B3 = 6.113e-8
_B4 = 1.516e-4

# Nonlinear coefficient
D33 = 27.0e-12  # pm/V -> m/V


def sellmeier_n(wavelength_um: float, T_celsius: float = 25.0) -> float:
    """Refractive index of MgO:LiNbO3 (extraordinary) from Gayer/Jundt equation.

    Args:
        wavelength_um: Wavelength in micrometers.
        T_celsius: Temperature in degrees Celsius.
    """
    lam2 = wavelength_um ** 2
    f = (T_celsius + 273.15) ** 2 - (24.5 + 273.15) ** 2
    n2 = (
        _A1 + _B1 * f
        + (_A2 + _B2 * f) / (lam2 - (_A3 + _B3 * f) ** 2)
        + (_A4 + _B4 * f) / (lam2 - _A5 ** 2)
        - _A6 * lam2
    )
    return np.sqrt(n2)


def coherence_length(lambda_fund_um: float, T_celsius: float = 25.0) -> float:
    """Coherence length L_coh = lambda_fund / (4 * (n_2w - n_w)) in meters."""
    n_w = sellmeier_n(lambda_fund_um, T_celsius)
    n_2w = sellmeier_n(lambda_fund_um / 2.0, T_celsius)
    return (lambda_fund_um * 1e-6) / (4.0 * (n_2w - n_w))


def qpm_period(lambda_fund_um: float, T_celsius: float = 25.0) -> float:
    """First-order QPM poling period Lambda = 2 * L_coh in meters."""
    return 2.0 * coherence_length(lambda_fund_um, T_celsius)


def make_poling_pattern(z_array, Lambda: float, d33: float, crystal_mask=None):
    """Generate poling pattern array of ±d33 values.

    Args:
        z_array: CuPy array of z positions (meters).
        Lambda: Poling period in meters.
        d33: Nonlinear coefficient magnitude (m/V).
        crystal_mask: Boolean CuPy array, True inside crystal. If None, entire grid is crystal.

    Returns:
        CuPy array of d(z) values (±d33 inside crystal, 0 outside).
    """
    import cupy as cp

    half_period = Lambda / 2.0
    domain_index = cp.floor(z_array / half_period).astype(cp.int64)
    sign = cp.where(domain_index % 2 == 0, 1.0, -1.0)
    d_z = sign * d33
    if crystal_mask is not None:
        d_z *= crystal_mask
    return d_z
