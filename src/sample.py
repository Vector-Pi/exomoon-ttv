"""
sample.py
---------
Load, validate, and query the planet sample for the exomoon TTV study.
Reads data/sample.csv and optionally cross-matches with the NASA
Exoplanet Archive to fill missing parameters.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from .utils import (
    a_au_from_rstar,
    a_rstar_from_period,
    hill_radius_au,
    stable_moon_region,
    orbital_velocity_ms,
    tess_timing_precision_s,
    transit_snr,
)

_DEFAULT_CSV = Path(__file__).parent.parent / "data" / "sample.csv"


# ---------------------------------------------------------------------------
# Data container
# ---------------------------------------------------------------------------

@dataclass
class PlanetRecord:
    """
    All parameters needed for TTV analysis of a single planet.
    """
    tic_id:           int
    name:             str
    period_days:      float
    t0_btjd:          float
    rp_rstar:         float
    a_rstar:          float
    inc_deg:          float
    duration_days:    float
    planet_mass_mjup: float
    star_mass_msun:   float
    star_radius_rsun: float
    tmag:             float
    sectors:          list[int]

    # Derived quantities set after init
    a_au:             float = field(init=False)
    hill_radius_au:   float = field(init=False)
    stable_limit_au:  float = field(init=False)
    v_planet_ms:      float = field(init=False)
    sigma_t_s:        float = field(init=False)

    def __post_init__(self):
        self.a_au           = a_au_from_rstar(self.a_rstar, self.star_radius_rsun)
        self.hill_radius_au = hill_radius_au(
            self.a_au, self.planet_mass_mjup, self.star_mass_msun
        )
        self.stable_limit_au = stable_moon_region(self.hill_radius_au)
        self.v_planet_ms    = orbital_velocity_ms(self.period_days, self.a_au)
        self.sigma_t_s      = tess_timing_precision_s(
            self.tmag, self.duration_days
        )

    def snr(self, n_sectors: Optional[int] = None) -> float:
        ns = n_sectors if n_sectors is not None else len(self.sectors)
        return transit_snr(
            depth         = self.rp_rstar**2,
            duration_days = self.duration_days,
            period_days   = self.period_days,
            n_sectors     = ns,
            tmag          = self.tmag,
        )

    def __repr__(self) -> str:
        return (
            f"PlanetRecord({self.name}, P={self.period_days:.3f} d, "
            f"Mp={self.planet_mass_mjup:.3f} MJ, "
            f"sigma_t={self.sigma_t_s:.1f} s, "
            f"r_H={self.hill_radius_au:.4f} AU)"
        )


# ---------------------------------------------------------------------------
# Sample loader
# ---------------------------------------------------------------------------

class PlanetSample:
    """
    Load and manage the planet sample for the TTV population study.

    Parameters
    ----------
    csv_path : str or Path, optional
        Path to the sample CSV.  Defaults to data/sample.csv.
    min_snr : float
        Minimum expected transit SNR to include a planet.  Default 10.

    Examples
    --------
    >>> sample = PlanetSample()
    >>> sample.load()
    >>> print(len(sample))
    12
    >>> pi_men = sample["Pi Men c"]
    >>> print(pi_men.sigma_t_s)
    """

    def __init__(
        self,
        csv_path: Optional[str | Path] = None,
        min_snr:  float = 10.0,
    ) -> None:
        self.csv_path = Path(csv_path) if csv_path else _DEFAULT_CSV
        self.min_snr  = min_snr
        self._records: list[PlanetRecord] = []

    def load(self) -> "PlanetSample":
        """
        Load the sample CSV and build PlanetRecord objects.

        Returns self for method chaining.
        """
        if not self.csv_path.exists():
            raise FileNotFoundError(
                f"Sample CSV not found: {self.csv_path}\n"
                f"Expected at: {_DEFAULT_CSV}"
            )

        df = pd.read_csv(self.csv_path)
        required_cols = [
            "tic_id", "name", "period_days", "t0_btjd", "rp_rstar",
            "a_rstar", "inc_deg", "duration_days", "planet_mass_mjup",
            "star_mass_msun", "star_radius_rsun", "tmag", "sectors",
        ]
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            raise ValueError(f"Sample CSV missing columns: {missing}")

        self._records = []
        skipped = []

        for _, row in df.iterrows():
            # Parse sector list from comma-separated string
            try:
                sectors = [int(s.strip()) for s in str(row["sectors"]).split(",")]
            except ValueError:
                warnings.warn(f"Could not parse sectors for {row['name']}; skipping.")
                continue

            rec = PlanetRecord(
                tic_id           = int(row["tic_id"]),
                name             = str(row["name"]),
                period_days      = float(row["period_days"]),
                t0_btjd          = float(row["t0_btjd"]),
                rp_rstar         = float(row["rp_rstar"]),
                a_rstar          = float(row["a_rstar"]),
                inc_deg          = float(row["inc_deg"]),
                duration_days    = float(row["duration_days"]),
                planet_mass_mjup = float(row["planet_mass_mjup"]),
                star_mass_msun   = float(row["star_mass_msun"]),
                star_radius_rsun = float(row["star_radius_rsun"]),
                tmag             = float(row["tmag"]),
                sectors          = sectors,
            )

            if rec.snr() < self.min_snr:
                skipped.append(rec.name)
                continue

            self._records.append(rec)

        if skipped:
            warnings.warn(
                f"Skipped {len(skipped)} planets with SNR < {self.min_snr}: "
                f"{skipped}"
            )

        print(f"[PlanetSample] Loaded {len(self._records)} planets from "
              f"{self.csv_path.name}.")
        return self

    def __len__(self) -> int:
        return len(self._records)

    def __iter__(self):
        return iter(self._records)

    def __getitem__(self, name_or_idx):
        if isinstance(name_or_idx, int):
            return self._records[name_or_idx]
        for r in self._records:
            if r.name == name_or_idx:
                return r
        raise KeyError(f"Planet '{name_or_idx}' not in sample.")

    def to_dataframe(self) -> pd.DataFrame:
        """Return a summary DataFrame of the sample."""
        rows = []
        for r in self._records:
            rows.append({
                "name":          r.name,
                "tic_id":        r.tic_id,
                "period_days":   r.period_days,
                "planet_mass_mjup": r.planet_mass_mjup,
                "tmag":          r.tmag,
                "sigma_t_s":     round(r.sigma_t_s, 1),
                "hill_radius_au": round(r.hill_radius_au, 5),
                "stable_limit_au": round(r.stable_limit_au, 5),
                "n_sectors":     len(r.sectors),
                "expected_snr":  round(r.snr(), 1),
            })
        return pd.DataFrame(rows)

    def summary(self) -> None:
        """Print a formatted sample summary."""
        df = self.to_dataframe()
        print("\n" + "=" * 80)
        print("Planet Sample Summary")
        print("=" * 80)
        print(df.to_string(index=False))
        print(f"\n{len(self._records)} planets total.")
        print(f"Median timing precision: "
              f"{np.median([r.sigma_t_s for r in self._records]):.1f} s")
        print(f"Best timing precision:   "
              f"{min(r.sigma_t_s for r in self._records):.1f} s "
              f"({min(self._records, key=lambda r: r.sigma_t_s).name})")
