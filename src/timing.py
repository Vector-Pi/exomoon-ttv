"""
timing.py
---------
Per-transit central time extraction and O-C computation for a single
planetary system. Downloads TESS data, detrends, identifies transit
windows, fits each transit independently, and returns O-C residuals.
"""

from __future__ import annotations

import warnings
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar
import batman
import lightkurve as lk
from astropy.stats import sigma_clip
import astropy.units as u

from .sample import PlanetRecord

warnings.filterwarnings("ignore", category=lk.LightkurveWarning)
warnings.filterwarnings("ignore", message=".*tpfmodel.*")
warnings.filterwarnings("ignore", message=".*oktopus.*")

_DAY_MIN = 1440.0
_DAY_S   = 86400.0


# ---------------------------------------------------------------------------
# TransitTimer
# ---------------------------------------------------------------------------

class TransitTimer:
    """
    Extract per-transit central times for a single planetary system.

    Parameters
    ----------
    record : PlanetRecord
        Planet parameters from the sample.
    window_factor : float
        Half-width of each transit window = window_factor × duration.
    sg_window : int
        Savitzky-Golay detrending window in cadences (must be odd).

    Examples
    --------
    >>> from src.sample import PlanetSample
    >>> sample = PlanetSample().load()
    >>> rec = sample["WASP-18 b"]
    >>> timer = TransitTimer(rec)
    >>> t_obs, t_err = timer.run()
    >>> oc, oc_err   = timer.compute_oc(t_obs, t_err)
    """

    def __init__(
        self,
        record:        PlanetRecord,
        window_factor: float = 2.5,
        sg_window:     int   = 301,
    ) -> None:
        self.rec           = record
        self.window_factor = window_factor
        self.sg_window     = sg_window if sg_window % 2 == 1 else sg_window + 1

        self._lc_flat: Optional[lk.LightCurve] = None
        self._transit_times: Optional[np.ndarray] = None
        self._transit_errs:  Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # Download and detrend
    # ------------------------------------------------------------------

    def _download(self) -> lk.LightCurve:
        """Download SPOC PDCSAP data for the specified sectors."""
        search = lk.search_lightcurve(
            f"TIC {self.rec.tic_id}",
            mission="TESS",
            author="SPOC",
            sector=self.rec.sectors,
        )
        if len(search) == 0:
            search = lk.search_lightcurve(
                f"TIC {self.rec.tic_id}",
                mission="TESS",
                author="TESS-SPOC",
                sector=self.rec.sectors,
            )
        if len(search) == 0:
            raise RuntimeError(
                f"No TESS data found for TIC {self.rec.tic_id} "
                f"in sectors {self.rec.sectors}."
            )

        col = search.download_all(quality_bitmask="default")
        lc  = col.stitch(corrector_func=lambda x: x.normalize())
        lc  = lc.remove_nans()

        flux_plain = np.array(lc.flux.value, dtype=float)
        clipped    = sigma_clip(flux_plain, sigma=4.0, masked=True)
        lc         = lc[~clipped.mask]
        return lc

    def _detrend(self, lc: lk.LightCurve) -> lk.LightCurve:
        """Savitzky-Golay detrending."""
        return lc.flatten(window_length=self.sg_window, polyorder=3)

    # ------------------------------------------------------------------
    # Transit window identification
    # ------------------------------------------------------------------

    def _get_windows(
        self, time: np.ndarray
    ) -> list[tuple[float, np.ndarray]]:
        """Return (expected_t0, boolean_mask) for each transit in time."""
        half_w  = self.window_factor * self.rec.duration_days
        t_start = time.min()
        t_end   = time.max()

        n_start = int(np.ceil((t_start - self.rec.t0_btjd) / self.rec.period_days))
        n_end   = int(np.floor((t_end   - self.rec.t0_btjd) / self.rec.period_days))

        windows = []
        for n in range(n_start, n_end + 1):
            t_exp = self.rec.t0_btjd + n * self.rec.period_days
            mask  = np.abs(time - t_exp) <= half_w
            if mask.sum() >= 5:
                windows.append((t_exp, mask))
        return windows

    # ------------------------------------------------------------------
    # Single transit fit
    # ------------------------------------------------------------------

    def _batman_lc(self, t0_val: float, time_seg: np.ndarray) -> np.ndarray:
        p = batman.TransitParams()
        p.t0        = t0_val
        p.per       = self.rec.period_days
        p.rp        = self.rec.rp_rstar
        p.a         = self.rec.a_rstar
        p.inc       = self.rec.inc_deg
        p.ecc       = 0.0
        p.w         = 90.0
        p.u         = [0.3, 0.2]
        p.limb_dark = "quadratic"
        m = batman.TransitModel(p, time_seg,
                                supersample_factor=7,
                                exp_time=2.0 / _DAY_MIN)
        return m.light_curve(p)

    def _fit_single(
        self,
        t_exp:   float,
        time_s:  np.ndarray,
        flux_s:  np.ndarray,
        ferr_s:  np.ndarray,
    ) -> tuple[float, float]:
        """
        Fit a single transit window to find the central time.

        Returns (t0_best, t0_err) in BTJD days.
        """
        half_d = self.rec.duration_days * 0.6

        def neg_ll(t0_val):
            try:
                model = self._batman_lc(t0_val, time_s)
                return 0.5 * np.sum(((flux_s - model) / ferr_s) ** 2)
            except Exception:
                return 1e12

        result = minimize_scalar(
            neg_ll,
            bounds=(t_exp - half_d, t_exp + half_d),
            method="bounded",
            options={"xatol": 1e-6},
        )
        t_best = result.x

        # Uncertainty from likelihood curvature
        dt = 1e-5
        h  = (neg_ll(t_best + dt) + neg_ll(t_best - dt) - 2 * neg_ll(t_best)) / dt**2
        t_err = (1.0 / np.sqrt(h)) if h > 0 else self.rec.duration_days / np.sqrt(len(time_s))

        return t_best, t_err

    # ------------------------------------------------------------------
    # O-C computation
    # ------------------------------------------------------------------

    def compute_oc(
        self,
        transit_times: np.ndarray,
        transit_errs:  np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Weighted linear fit to transit times; return O-C residuals in minutes.

        Parameters
        ----------
        transit_times : array   Observed central times in BTJD.
        transit_errs  : array   1-sigma uncertainties in BTJD.

        Returns
        -------
        oc_min  : array   O-C residuals in minutes.
        err_min : array   Uncertainties in minutes.
        """
        epochs  = np.round(
            (transit_times - self.rec.t0_btjd) / self.rec.period_days
        ).astype(int)
        weights = 1.0 / transit_errs**2

        A   = np.column_stack([np.ones_like(epochs), epochs])
        W   = np.diag(weights)
        cov = np.linalg.inv(A.T @ W @ A)
        p   = cov @ A.T @ W @ transit_times

        t_calc  = p[0] + epochs * p[1]
        oc_min  = (transit_times - t_calc) * _DAY_MIN
        err_min = transit_errs * _DAY_MIN
        return oc_min, err_min

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def run(self, lc: Optional[lk.LightCurve] = None) -> tuple[np.ndarray, np.ndarray]:
        """
        Download data, detrend, and extract per-transit times.

        Parameters
        ----------
        lc : LightCurve, optional
            Pre-downloaded light curve.  If None, downloads from MAST.

        Returns
        -------
        transit_times : array   Central transit times in BTJD.
        transit_errs  : array   1-sigma uncertainties in BTJD.
        """
        if lc is None:
            print(f"  [{self.rec.name}] Downloading sectors {self.rec.sectors} ...")
            lc = self._download()

        print(f"  [{self.rec.name}] Detrending ...")
        lc_flat = self._detrend(lc)
        self._lc_flat = lc_flat

        time = lc_flat.time.value
        flux = lc_flat.flux.value
        ferr = (lc_flat.flux_err.value
                if lc_flat.flux_err is not None
                else np.full_like(flux, 1e-4))

        windows = self._get_windows(time)
        if not windows:
            raise RuntimeError(
                f"No transit windows found for {self.rec.name}. "
                f"Check t0={self.rec.t0_btjd} and period={self.rec.period_days}."
            )

        print(f"  [{self.rec.name}] Fitting {len(windows)} transits ...")
        t_out, e_out = [], []
        for t_exp, mask in windows:
            t_b, t_e = self._fit_single(t_exp, time[mask], flux[mask], ferr[mask])
            t_out.append(t_b)
            e_out.append(t_e)

        transit_times = np.array(t_out)
        transit_errs  = np.array(e_out)
        self._transit_times = transit_times
        self._transit_errs  = transit_errs

        print(f"  [{self.rec.name}] Done. "
              f"Median precision: {np.median(transit_errs)*_DAY_MIN:.2f} min  "
              f"({np.median(transit_errs)*_DAY_S:.1f} s)")
        return transit_times, transit_errs

    # ------------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------------

    def plot_oc(
        self,
        oc_min:   np.ndarray,
        err_min:  np.ndarray,
        ax:       Optional[plt.Axes] = None,
        savepath: Optional[str]      = None,
    ) -> plt.Axes:
        """O-C diagram for this system."""
        n      = len(oc_min)
        epochs = np.arange(n)

        if ax is None:
            _, ax = plt.subplots(figsize=(10, 4))

        ax.errorbar(epochs, oc_min, yerr=err_min,
                    fmt="ko", ms=4, lw=1.2, capsize=2)
        ax.axhline(0, color="gray", ls="--", lw=0.8)
        med_err = np.median(err_min)
        ax.fill_between(epochs, -med_err, med_err,
                        alpha=0.15, color="steelblue",
                        label=rf"Median $\sigma_t$ = {med_err:.2f} min")
        ax.set_xlabel("Transit Epoch")
        ax.set_ylabel("O – C (minutes)")
        ax.set_title(f"{self.rec.name}  (TIC {self.rec.tic_id})  O-C Diagram")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.25)

        if savepath:
            plt.savefig(savepath, dpi=150, bbox_inches="tight")
        return ax
