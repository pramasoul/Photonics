"""Settled physics for MgO:LiNbO₃ (PPLN).

Sellmeier equation (Gayer et al. 2008), nonlinear coefficient,
QPM period, and dispersion parameters. Shared by FDTD and SSFM codes.
"""

import numpy as np

# Speed of light
C = 2.998e8  # m/s

# ── Gayer et al. 2008 Sellmeier for 5% MgO:LiNbO₃ extraordinary ray ──
# n²(λ,T) = (a1+b1f) + (a2+b2f)/(λ²-(a3+b3f)²) + (a4+b4f)/(λ²-a5²) - a6λ²
# where f(T) = (T+273.15)² - (24.5+273.15)², λ in μm, T in °C

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
D33 = 27.0e-12  # m/V


def sellmeier_n(wavelength_um: float, T_celsius: float = 25.0) -> float:
    """Refractive index n(λ, T) for MgO:LiNbO₃ extraordinary ray."""
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
    """Coherence length L_coh = λ / (4(n₂ω − nω)) in meters."""
    n_w = sellmeier_n(lambda_fund_um, T_celsius)
    n_2w = sellmeier_n(lambda_fund_um / 2.0, T_celsius)
    return (lambda_fund_um * 1e-6) / (4.0 * (n_2w - n_w))


def qpm_period(lambda_fund_um: float, T_celsius: float = 25.0) -> float:
    """First-order QPM poling period Λ = 2L_coh in meters."""
    return 2.0 * coherence_length(lambda_fund_um, T_celsius)


# ── Dispersion parameters from Sellmeier derivatives ──────────────────

def wavenumber(wavelength_um: float, T_celsius: float = 25.0) -> float:
    """Wavenumber k = 2πn/λ in rad/m."""
    n = sellmeier_n(wavelength_um, T_celsius)
    return 2 * np.pi * n / (wavelength_um * 1e-6)


def group_index(wavelength_um: float, T_celsius: float = 25.0) -> float:
    """Group index n_g = n - λ·dn/dλ."""
    dlam = 1e-4  # small step in μm
    n = sellmeier_n(wavelength_um, T_celsius)
    n_plus = sellmeier_n(wavelength_um + dlam, T_celsius)
    n_minus = sellmeier_n(wavelength_um - dlam, T_celsius)
    dn_dlam = (n_plus - n_minus) / (2 * dlam)
    return n - wavelength_um * dn_dlam


def group_velocity(wavelength_um: float, T_celsius: float = 25.0) -> float:
    """Group velocity v_g = c/n_g in m/s."""
    return C / group_index(wavelength_um, T_celsius)


def gvd(wavelength_um: float, T_celsius: float = 25.0) -> float:
    """Group velocity dispersion β₂ = d²k/dω² in s²/m.

    Computed from the Sellmeier via numerical differentiation of k(ω).
    """
    omega = 2 * np.pi * C / (wavelength_um * 1e-6)
    domega = omega * 1e-4

    k_plus = wavenumber(_omega_to_lambda_um(omega + domega), T_celsius)
    k_center = wavenumber(wavelength_um, T_celsius)
    k_minus = wavenumber(_omega_to_lambda_um(omega - domega), T_celsius)

    return (k_plus - 2 * k_center + k_minus) / domega ** 2


def dispersion_operator(wavelength_um: float, T_celsius: float,
                        omega_grid: np.ndarray) -> np.ndarray:
    """Full Sellmeier dispersion operator D(ω) for the SSFM.

    D(ω) = k(ω₀+ω) - k(ω₀) - ω·dk/dω|₀

    This is the residual dispersion after removing the carrier phase
    and the group velocity (which is handled by the co-moving frame).

    Args:
        wavelength_um: carrier wavelength
        T_celsius: temperature
        omega_grid: angular frequency offsets from carrier (rad/s)

    Returns:
        D(ω) array in rad/m
    """
    omega0 = 2 * np.pi * C / (wavelength_um * 1e-6)
    k0 = wavenumber(wavelength_um, T_celsius)
    vg = group_velocity(wavelength_um, T_celsius)
    dk_domega = 1.0 / vg  # = n_g / c

    D = np.zeros_like(omega_grid)
    for i, dw in enumerate(omega_grid):
        if abs(dw) < 1e-6:
            D[i] = 0.0
        else:
            lam = _omega_to_lambda_um(omega0 + dw)
            if lam > 0.2 and lam < 10.0:  # physical range
                k = wavenumber(lam, T_celsius)
                D[i] = k - k0 - dw * dk_domega
            else:
                D[i] = 0.0  # outside Sellmeier validity

    return D


def _omega_to_lambda_um(omega: float) -> float:
    """Convert angular frequency (rad/s) to wavelength (μm)."""
    return 2 * np.pi * C / omega * 1e6


# ── CuPy-dependent utilities (for FDTD only) ──────────────────────────

def make_poling_pattern(z_array, Lambda: float, d33: float, crystal_mask=None):
    """Generate poling pattern array of ±d33 values (CuPy arrays)."""
    import cupy as cp

    half_period = Lambda / 2.0
    domain_index = cp.floor(z_array / half_period).astype(cp.int64)
    sign = cp.where(domain_index % 2 == 0, 1.0, -1.0)
    d_z = sign * d33
    if crystal_mask is not None:
        d_z *= crystal_mask
    return d_z
