"""
sensitivity.py
--------------
Per-system exomoon mass sensitivity curves and population-level
exclusion map.

For each planet:
  - Compute minimum detectable moon mass as a function of moon
    semi-major axis, given the measured per-transit timing precision.
  - Clip at the Hill stability boundary.

Population exclusion map:
  - At each (a_moon, M_moon) grid point, count the fraction of systems
    for which a moon of that mass and separation would be detectable at
    >= 3 sigma.  Contours show 50% and 90% completeness.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from .sample import PlanetRecord
from .utils  import min_detectable_moon_mearth, orbital_velocity_ms

_MEARTH = 5.972e24   # kg
_MJUP   = 1.898e27   # kg


# ---------------------------------------------------------------------------
# SensitivityCalculator
# ---------------------------------------------------------------------------

class SensitivityCalculator:
    """
    Compute exomoon mass sensitivity for a single system or a population.

    Parameters
    ----------
    n_a_points : int
        Number of moon semi-major axis grid points.
    snr_threshold : float
        Minimum TTV SNR for detection (default 3.0).
    a_moon_min_au : float
        Inner edge of the moon separation grid in AU.
    """

    def __init__(
        self,
        n_a_points:    int   = 300,
        snr_threshold: float = 3.0,
        a_moon_min_au: float = 0.0005,
    ) -> None:
        self.n_a_points    = n_a_points
        self.snr_threshold = snr_threshold
        self.a_moon_min_au = a_moon_min_au

    # ------------------------------------------------------------------
    # Single system
    # ------------------------------------------------------------------

    def sensitivity_curve(
        self,
        record:       PlanetRecord,
        sigma_t_s:    Optional[float] = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Minimum detectable moon mass vs. moon orbital separation
        for a single planetary system.

        Parameters
        ----------
        record    : PlanetRecord
        sigma_t_s : float, optional
            Per-transit timing precision in seconds.  If None, uses the
            analytic TESS noise model from record.sigma_t_s.

        Returns
        -------
        a_moon_au    : array   Moon separations in AU.
        m_min_mearth : array   Minimum detectable moon mass in Earth masses.
        """
        if sigma_t_s is None:
            sigma_t_s = record.sigma_t_s

        # Grid from a_moon_min to the Hill stability boundary
        a_max = record.stable_limit_au
        a_min = max(self.a_moon_min_au, a_max * 0.01)
        a_moon_au = np.linspace(a_min, a_max, self.n_a_points)

        m_min = np.array([
            min_detectable_moon_mearth(
                sigma_t_s      = sigma_t_s,
                M_planet_mjup  = record.planet_mass_mjup,
                a_moon_au      = a,
                v_planet_ms    = record.v_planet_ms,
                snr_threshold  = self.snr_threshold,
            )
            for a in a_moon_au
        ])
        return a_moon_au, m_min

    # ------------------------------------------------------------------
    # Population exclusion map
    # ------------------------------------------------------------------

    def exclusion_map(
        self,
        records:     list[PlanetRecord],
        sigma_t_s_list: Optional[list[float]] = None,
        n_m_points:  int   = 200,
        m_min_mearth: float = 0.01,
        m_max_mearth: float = 100.0,
        a_max_au:    float  = 0.05,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Build the population-level exclusion map.

        At each (a_moon, M_moon) grid point, compute the fraction of systems
        in the sample for which a moon of that mass at that separation would
        produce a TTV signal detectable at >= snr_threshold sigma.

        Parameters
        ----------
        records       : list[PlanetRecord]
        sigma_t_s_list: list[float], optional
            Measured per-transit precisions.  If None, uses analytic estimates.
        n_m_points    : int     Number of mass grid points (log scale).
        m_min_mearth  : float   Minimum moon mass in Earth masses.
        m_max_mearth  : float   Maximum moon mass in Earth masses.
        a_max_au      : float   Maximum moon separation in AU.

        Returns
        -------
        A_grid : 2D array   Moon separations (AU).
        M_grid : 2D array   Moon masses (Earth masses).
        frac   : 2D array   Detection fraction [0, 1] at each grid point.
        """
        if sigma_t_s_list is None:
            sigma_t_s_list = [r.sigma_t_s for r in records]

        a_arr = np.linspace(self.a_moon_min_au, a_max_au, self.n_a_points)
        m_arr = np.logspace(np.log10(m_min_mearth), np.log10(m_max_mearth), n_m_points)
        A_grid, M_grid = np.meshgrid(a_arr, m_arr)

        frac = np.zeros_like(A_grid)
        n_sys = len(records)

        for i, (rec, sig) in enumerate(zip(records, sigma_t_s_list)):
            # For each grid point, check if this system can detect the moon
            m_min_grid = np.array([
                [
                    min_detectable_moon_mearth(
                        sigma_t_s     = sig,
                        M_planet_mjup = rec.planet_mass_mjup,
                        a_moon_au     = a,
                        v_planet_ms   = rec.v_planet_ms,
                        snr_threshold = self.snr_threshold,
                    )
                    for a in a_arr
                ]
                for _ in [None]  # single row, will broadcast
            ])[0]  # shape (n_a,)

            # Moon is detectable if M_moon > m_min at this separation
            # and a_moon < stable limit for this planet
            detectable = (M_grid >= m_min_grid[np.newaxis, :]) & \
                         (A_grid <= rec.stable_limit_au)
            frac += detectable.astype(float)

        frac /= n_sys
        return A_grid, M_grid, frac

    # ------------------------------------------------------------------
    # Plots
    # ------------------------------------------------------------------

    def plot_sensitivity_curve(
        self,
        record:       PlanetRecord,
        sigma_t_s:    Optional[float] = None,
        ax:           Optional[plt.Axes] = None,
        color:        str = "steelblue",
        label:        Optional[str] = None,
        savepath:     Optional[str] = None,
    ) -> plt.Axes:
        """Plot the sensitivity curve for a single system."""
        a_moon_au, m_min = self.sensitivity_curve(record, sigma_t_s)

        if ax is None:
            _, ax = plt.subplots(figsize=(9, 5))

        lbl = label if label else record.name
        ax.plot(a_moon_au, m_min, color=color, lw=2, label=lbl)
        ax.fill_between(a_moon_au, m_min, m_min.max(),
                        alpha=0.08, color=color)

        # Reference lines
        ax.axhline(1.0,   color="gray", ls="--", lw=1, alpha=0.7,
                   label=r"1 $M_\oplus$")
        ax.axhline(17.15, color="gray", ls=":",  lw=1, alpha=0.7,
                   label=r"$M_\text{Neptune}$")
        ax.axvline(record.stable_limit_au, color="red", ls="--", lw=1,
                   alpha=0.6, label=r"0.5 $r_H$ (stability)")

        ax.set_xlabel("Moon Semi-major Axis (AU)")
        ax.set_ylabel(r"Min. Detectable Moon Mass ($M_\oplus$)")
        ax.set_title(
            rf"{record.name}  |  "
            rf"$M_p = {record.planet_mass_mjup:.2f}\,M_J$  |  "
            rf"$\sigma_t = {(sigma_t_s or record.sigma_t_s):.0f}$ s"
        )
        ax.set_yscale("log")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.25, which="both")

        if savepath:
            plt.savefig(savepath, dpi=150, bbox_inches="tight")
        return ax

    def plot_exclusion_map(
        self,
        records:        list[PlanetRecord],
        sigma_t_s_list: Optional[list[float]] = None,
        contour_levels: list[float] = [0.5, 0.9],
        savepath:       Optional[str] = None,
    ) -> plt.Figure:
        """
        Population exclusion map: fraction of systems that could detect
        a moon of given mass and separation at >= 3 sigma.
        """
        A_grid, M_grid, frac = self.exclusion_map(records, sigma_t_s_list)

        fig, ax = plt.subplots(figsize=(11, 7))

        # Filled colour map
        im = ax.pcolormesh(
            A_grid, M_grid, frac,
            cmap="Blues", vmin=0, vmax=1,
            shading="auto",
        )
        cbar = fig.colorbar(im, ax=ax, label="Detection fraction")

        # Contours
        CS = ax.contour(
            A_grid, M_grid, frac,
            levels=contour_levels,
            colors=["orange", "red"],
            linewidths=[1.5, 2.0],
        )
        ax.clabel(CS, fmt={0.5: "50%", 0.9: "90%"}, fontsize=10)

        # Reference lines
        ax.axhline(1.0,   color="white", ls="--", lw=1, alpha=0.8,
                   label=r"1 $M_\oplus$")
        ax.axhline(17.15, color="white", ls=":",  lw=1, alpha=0.8,
                   label=r"$M_\text{Neptune}$")

        ax.set_xlabel("Moon Semi-major Axis (AU)")
        ax.set_ylabel(r"Moon Mass ($M_\oplus$)")
        ax.set_yscale("log")
        ax.set_title(
            f"Population Exclusion Map  ({len(records)} TESS planets, "
            rf"SNR $\geq$ {self.snr_threshold})"
        )
        ax.legend(fontsize=9, loc="upper left")

        if savepath:
            fig.savefig(savepath, dpi=150, bbox_inches="tight")
        return fig
