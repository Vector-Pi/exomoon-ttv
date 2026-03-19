"""
utils.py
--------
Physical calculations, unit conversions, and TESS noise model
for the exomoon TTV population study.
"""

from __future__ import annotations
import numpy as np

# Physical constants
_G        = 6.674e-11    # m^3 kg^-1 s^-2
_MSUN     = 1.989e30     # kg
_RSUN     = 6.957e8      # m
_MJUP     = 1.898e27     # kg
_MEARTH   = 5.972e24     # kg
_AU       = 1.496e11     # m
_DAY_S    = 86400.0
_DAY_MIN  = 1440.0


def orbital_velocity_ms(period_days: float, a_au: float) -> float:
    """
    Circular orbital velocity in m/s.
        v = 2π a / P
    """
    return 2 * np.pi * (a_au * _AU) / (period_days * _DAY_S)


def a_au_from_rstar(a_rstar: float, R_star_rsun: float) -> float:
    """Convert a/R* to AU."""
    return (a_rstar * R_star_rsun * _RSUN) / _AU


def a_rstar_from_period(period_days: float,
                         M_star_msun: float,
                         R_star_rsun: float) -> float:
    """Kepler's third law: a/R* from period and stellar parameters."""
    P_s = period_days * _DAY_S
    M_s = M_star_msun * _MSUN
    R_s = R_star_rsun * _RSUN
    a_m = ((_G * M_s * P_s**2) / (4 * np.pi**2)) ** (1.0 / 3.0)
    return a_m / R_s


def moon_ttv_amplitude_s(M_moon_mearth: float,
                          M_planet_mjup: float,
                          a_moon_au: float,
                          v_planet_ms: float) -> float:
    """
    TTV amplitude in seconds from an exomoon.

        delta_t = (M_m / (M_p + M_m)) * (a_m / v_p)

    Parameters
    ----------
    M_moon_mearth : float   Moon mass in Earth masses.
    M_planet_mjup : float   Planet mass in Jupiter masses.
    a_moon_au     : float   Moon semi-major axis in AU.
    v_planet_ms   : float   Planet orbital velocity in m/s.
    """
    M_m = M_moon_mearth * _MEARTH
    M_p = M_planet_mjup * _MJUP
    a_m = a_moon_au * _AU
    return (M_m / (M_p + M_m)) * (a_m / v_planet_ms)


def min_detectable_moon_mearth(sigma_t_s: float,
                                M_planet_mjup: float,
                                a_moon_au: float,
                                v_planet_ms: float,
                                snr_threshold: float = 3.0) -> float:
    """
    Minimum detectable moon mass in Earth masses at a given separation,
    given a per-transit timing precision sigma_t_s (seconds).

    Inverts  delta_t = (M_m / M_p) * (a_m / v_p)  for M_m.
    """
    M_p = M_planet_mjup * _MJUP
    a_m = a_moon_au * _AU
    # delta_t_min = snr_threshold * sigma_t_s
    # M_m_min / M_p = delta_t_min * v_p / a_m  (for M_m << M_p)
    M_m_min_kg = snr_threshold * sigma_t_s * M_p * v_planet_ms / a_m
    return M_m_min_kg / _MEARTH


def hill_radius_au(a_planet_au: float,
                   M_planet_mjup: float,
                   M_star_msun: float) -> float:
    """
    Hill radius of the planet in AU.
        r_H = a_p * (M_p / (3 M_*))^{1/3}
    """
    M_p = M_planet_mjup * _MJUP
    M_s = M_star_msun * _MSUN
    return a_planet_au * (M_p / (3 * M_s)) ** (1.0 / 3.0)


def stable_moon_region(hill_radius_au: float,
                        stability_fraction: float = 0.5) -> float:
    """
    Outer edge of the stable moon region (Domingos et al. 2006).
    Prograde moons are stable out to ~0.5 r_H for circular orbits.
    """
    return stability_fraction * hill_radius_au


def tess_timing_precision_s(tmag: float,
                              duration_days: float,
                              cadence_min: float = 2.0) -> float:
    """
    Estimated per-transit central time precision in seconds.

    Based on:  sigma_t ≈ sigma_flux * duration / sqrt(N_in_transit)
    where sigma_flux is the per-cadence photometric precision from the
    TESS noise model (Sullivan et al. 2015).

    Parameters
    ----------
    tmag         : float  TESS magnitude.
    duration_days: float  Transit duration in days.
    cadence_min  : float  Cadence in minutes (2 or 20).
    """
    # Per-cadence noise (ppm), Sullivan et al. 2015 approximation
    photon_noise_ppm = 10 ** (0.2 * (tmag - 10.5)) * 1000.0
    sys_floor_ppm    = 100.0
    sigma_ppm        = np.sqrt(photon_noise_ppm**2 + sys_floor_ppm**2)
    sigma_flux       = sigma_ppm * 1e-6

    n_in_transit = duration_days * _DAY_MIN / cadence_min
    # Timing precision from Doyle & Deeg (2004)
    sigma_t_days = sigma_flux * duration_days / np.sqrt(n_in_transit)
    return sigma_t_days * _DAY_S


def transit_snr(depth: float,
                duration_days: float,
                period_days: float,
                n_sectors: int,
                tmag: float,
                cadence_min: float = 2.0) -> float:
    """Expected BLS transit detection SNR."""
    photon_ppm = 10 ** (0.2 * (tmag - 10.5)) * 1000.0
    sigma_ppm  = np.sqrt(photon_ppm**2 + 100.0**2)
    sigma_1    = sigma_ppm * 1e-6

    n_transits   = (n_sectors * 27.0) / period_days
    n_in_transit = duration_days * _DAY_MIN / cadence_min
    return (depth / sigma_1) * np.sqrt(n_in_transit * n_transits)
