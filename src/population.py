"""
population.py
-------------
Orchestrates the full exomoon TTV population study:
  1. Load planet sample
  2. For each system: download TESS data, extract transit times, compute O-C
  3. Compute per-system sensitivity curves
  4. Build and plot the population exclusion map
  5. Generate LaTeX summary table
"""

from __future__ import annotations

import os
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from tqdm import tqdm

from .sample      import PlanetSample, PlanetRecord
from .timing      import TransitTimer
from .sensitivity import SensitivityCalculator
from .utils       import hill_radius_au, stable_moon_region


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

class SystemResult:
    """Stores TTV results for a single planetary system."""

    def __init__(self, record: PlanetRecord) -> None:
        self.record         = record
        self.transit_times: Optional[np.ndarray] = None
        self.transit_errs:  Optional[np.ndarray] = None
        self.oc_min:        Optional[np.ndarray] = None
        self.oc_err_min:    Optional[np.ndarray] = None
        self.sigma_t_s:     Optional[float]      = None   # measured
        self.rms_oc_min:    Optional[float]      = None
        self.n_transits:    int                  = 0
        self.error:         Optional[str]        = None   # if run failed

    @property
    def success(self) -> bool:
        return self.error is None and self.transit_times is not None

    def __repr__(self) -> str:
        if not self.success:
            return f"SystemResult({self.record.name}, FAILED: {self.error})"
        return (
            f"SystemResult({self.record.name}, "
            f"n={self.n_transits}, "
            f"sigma_t={self.sigma_t_s:.1f} s, "
            f"rms={self.rms_oc_min:.2f} min)"
        )


# ---------------------------------------------------------------------------
# ExomoonPopulation
# ---------------------------------------------------------------------------

class ExomoonPopulation:
    """
    Full population-level exomoon TTV pipeline.

    Parameters
    ----------
    csv_path : str or Path, optional
        Path to the sample CSV.  Defaults to data/sample.csv.
    output_dir : str or Path, optional
        Directory to save plots and tables.  Defaults to 'results/'.
    snr_threshold : float
        TTV detection SNR threshold (default 3.0).
    window_factor : float
        Transit window half-width in units of transit duration.

    Examples
    --------
    >>> pop = ExomoonPopulation()
    >>> pop.load_sample()
    >>> pop.run(n_jobs=1)
    >>> pop.plot_exclusion_map()
    >>> pop.summary_table()
    """

    def __init__(
        self,
        csv_path:      Optional[str | Path] = None,
        output_dir:    str | Path           = "results",
        snr_threshold: float                = 3.0,
        window_factor: float                = 2.5,
    ) -> None:
        self.csv_path      = csv_path
        self.output_dir    = Path(output_dir)
        self.snr_threshold = snr_threshold
        self.window_factor = window_factor

        self._sample:  Optional[PlanetSample]       = None
        self._results: dict[str, SystemResult]       = {}
        self._sens     = SensitivityCalculator(snr_threshold=snr_threshold)

        self.output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Load
    # ------------------------------------------------------------------

    def load_sample(self, csv_path: Optional[str | Path] = None) -> "ExomoonPopulation":
        """Load the planet sample from CSV."""
        path = csv_path or self.csv_path
        self._sample = PlanetSample(csv_path=path).load()
        self._sample.summary()
        return self

    # ------------------------------------------------------------------
    # Run pipeline on one system
    # ------------------------------------------------------------------

    def _run_one(self, record: PlanetRecord) -> SystemResult:
        result = SystemResult(record)
        try:
            timer = TransitTimer(record, window_factor=self.window_factor)
            t_obs, t_err = timer.run()
            oc_min, oc_err_min = timer.compute_oc(t_obs, t_err)

            result.transit_times = t_obs
            result.transit_errs  = t_err
            result.oc_min        = oc_min
            result.oc_err_min    = oc_err_min
            result.n_transits    = len(t_obs)
            result.sigma_t_s     = float(np.median(t_err) * 86400.0)
            result.rms_oc_min    = float(np.std(oc_min))

            # Save O-C plot
            oc_path = str(self.output_dir / f"{record.name.replace(' ', '_')}_oc.png")
            timer.plot_oc(oc_min, oc_err_min, savepath=oc_path)
            plt.close("all")

            # Save sensitivity curve plot
            sens_path = str(self.output_dir / f"{record.name.replace(' ', '_')}_sens.png")
            self._sens.plot_sensitivity_curve(
                record,
                sigma_t_s = result.sigma_t_s,
                savepath  = sens_path,
            )
            plt.close("all")

        except Exception as e:
            result.error = str(e)
            warnings.warn(f"Failed for {record.name}: {e}")

        return result

    # ------------------------------------------------------------------
    # Run full population
    # ------------------------------------------------------------------

    def run(self, n_jobs: int = 1) -> "ExomoonPopulation":
        """
        Run the TTV extraction pipeline for all systems in the sample.

        Parameters
        ----------
        n_jobs : int
            Number of parallel jobs.  Currently only n_jobs=1 is supported
            (sequential). Set > 1 as a reminder for future parallelisation.
        """
        if self._sample is None:
            raise RuntimeError("Call load_sample() first.")

        records = list(self._sample)
        print(f"\n[ExomoonPopulation] Running TTV pipeline on "
              f"{len(records)} systems ...\n")

        for rec in tqdm(records, desc="Systems", unit="planet"):
            result = self._run_one(rec)
            self._results[rec.name] = result
            status = "OK" if result.success else f"FAILED ({result.error})"
            tqdm.write(f"  {rec.name:20s} → {status}")

        n_ok   = sum(1 for r in self._results.values() if r.success)
        n_fail = len(self._results) - n_ok
        print(f"\n[ExomoonPopulation] Done: {n_ok} succeeded, {n_fail} failed.")
        return self

    # ------------------------------------------------------------------
    # Population exclusion map
    # ------------------------------------------------------------------

    def plot_exclusion_map(self, savepath: Optional[str] = None) -> plt.Figure:
        """Build and plot the population-level exclusion map."""
        successful = [r for r in self._results.values() if r.success]
        if not successful:
            raise RuntimeError("No successful results to plot.")

        records     = [r.record    for r in successful]
        sigma_t_s   = [r.sigma_t_s for r in successful]

        sp = savepath or str(self.output_dir / "population_exclusion_map.png")
        fig = self._sens.plot_exclusion_map(
            records        = records,
            sigma_t_s_list = sigma_t_s,
            savepath       = sp,
        )
        print(f"[ExomoonPopulation] Exclusion map saved to {sp}")
        return fig

    def plot_all_sensitivity_curves(self, savepath: Optional[str] = None) -> plt.Figure:
        """Overlay sensitivity curves for all successful systems."""
        successful = [r for r in self._results.values() if r.success]
        if not successful:
            raise RuntimeError("No successful results.")

        fig, ax = plt.subplots(figsize=(12, 6))
        cmap    = plt.cm.get_cmap("tab20", len(successful))

        for i, res in enumerate(successful):
            a_au, m_min = self._sens.sensitivity_curve(
                res.record, sigma_t_s=res.sigma_t_s
            )
            ax.plot(a_au, m_min, color=cmap(i), lw=1.5,
                    label=res.record.name, alpha=0.85)

        ax.axhline(1.0,   color="k", ls="--", lw=1,
                   label=r"1 $M_\oplus$")
        ax.axhline(17.15, color="k", ls=":",  lw=1,
                   label=r"$M_\text{Neptune}$")
        ax.set_xlabel("Moon Semi-major Axis (AU)")
        ax.set_ylabel(r"Min. Detectable Moon Mass ($M_\oplus$)")
        ax.set_title("Per-System Exomoon Sensitivity Curves")
        ax.set_yscale("log")
        ax.legend(fontsize=7, ncol=2, loc="upper right")
        ax.grid(True, alpha=0.25, which="both")

        sp = savepath or str(self.output_dir / "all_sensitivity_curves.png")
        fig.savefig(sp, dpi=150, bbox_inches="tight")
        print(f"[ExomoonPopulation] Sensitivity curves saved to {sp}")
        return fig

    # ------------------------------------------------------------------
    # Summary table
    # ------------------------------------------------------------------

    def summary_table(self, savepath: Optional[str] = None) -> str:
        """
        Generate a LaTeX table summarising per-system TTV results and
        moon mass upper limits.
        """
        lines = [
            r"\begin{table*}[htbp]",
            r"  \centering",
            r"  \caption{Per-system TTV results and exomoon upper limits}",
            r"  \label{tab:ttv_results}",
            r"  \begin{tabular}{lrrrrrrr}",
            r"    \hline\hline",
            r"    Planet & $N_\mathrm{tr}$ & $\sigma_t$ (s) & "
            r"RMS O-C (min) & $r_H$ (AU) & "
            r"$M_\mathrm{lim}^\mathrm{inner}$ ($M_\oplus$) & "
            r"$M_\mathrm{lim}^\mathrm{outer}$ ($M_\oplus$) \\",
            r"    \hline",
        ]

        for name, res in self._results.items():
            if not res.success:
                lines.append(
                    rf"    {name} & \multicolumn{{6}}{{c}}{{(failed)}} \\"
                )
                continue

            rec = res.record
            # Sensitivity at inner (0.002 AU) and outer (stable limit) edge
            a_au, m_min = self._sens.sensitivity_curve(
                rec, sigma_t_s=res.sigma_t_s
            )
            m_inner = m_min[0]       # smallest separation
            m_outer = m_min[-1]      # Hill stability boundary

            lines.append(
                rf"    {rec.name} & "
                rf"{res.n_transits} & "
                rf"{res.sigma_t_s:.1f} & "
                rf"{res.rms_oc_min:.2f} & "
                rf"{rec.hill_radius_au:.4f} & "
                rf"{m_inner:.1f} & "
                rf"{m_outer:.1f} \\"
            )

        lines += [
            r"    \hline",
            r"  \end{tabular}",
            r"\end{table*}",
        ]
        table = "\n".join(lines)

        sp = savepath or str(self.output_dir / "summary_table.tex")
        with open(sp, "w") as f:
            f.write(table)

        print(f"\n[ExomoonPopulation] LaTeX table saved to {sp}")
        print(table)
        return table

    # ------------------------------------------------------------------
    # Text summary
    # ------------------------------------------------------------------

    def print_summary(self) -> None:
        """Print a plain-text summary of results."""
        print("\n" + "=" * 70)
        print("Exomoon TTV Population Study — Results Summary")
        print("=" * 70)
        for name, res in self._results.items():
            if res.success:
                print(
                    f"  {name:20s}  "
                    f"N={res.n_transits:3d}  "
                    f"sigma_t={res.sigma_t_s:6.1f} s  "
                    f"RMS={res.rms_oc_min:.3f} min"
                )
            else:
                print(f"  {name:20s}  FAILED: {res.error}")
        print("=" * 70)
