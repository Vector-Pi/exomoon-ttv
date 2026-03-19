"""
example_single_system.py
------------------------
Complete walkthrough on a single system: WASP-18 b (TIC 100100827).

WASP-18 b is a hot Jupiter with M_p = 10.2 M_Jup — one of the most
massive known transiting planets.  Its large mass means the TTV amplitude
from any moon would be very small (delta_t ~ M_m/M_p * a_m/v_p), making
it a useful test of the sensitivity floor.  We expect a null TTV result.

Expected timing precision: ~30-60 s (bright host star, Tmag = 8.7)
Expected minimum detectable moon: ~several M_earth at 0.001 AU
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import matplotlib.pyplot as plt

from src.sample      import PlanetSample
from src.timing      import TransitTimer
from src.sensitivity import SensitivityCalculator

# ---------------------------------------------------------------------------
# 1. Load planet record
# ---------------------------------------------------------------------------
print("=" * 60)
print("WASP-18 b — Single System TTV Analysis")
print("=" * 60)

sample = PlanetSample().load()
rec    = sample["WASP-18 b"]
print(f"\nSystem parameters:")
print(f"  Period:         {rec.period_days:.5f} d")
print(f"  Planet mass:    {rec.planet_mass_mjup:.2f} M_Jup")
print(f"  Hill radius:    {rec.hill_radius_au:.5f} AU")
print(f"  Stable limit:   {rec.stable_limit_au:.5f} AU")
print(f"  v_planet:       {rec.v_planet_ms/1000:.1f} km/s")
print(f"  Analytic sigma_t: {rec.sigma_t_s:.1f} s")

# ---------------------------------------------------------------------------
# 2. Extract transit times
# ---------------------------------------------------------------------------
print("\n[1] Extracting transit times ...")
timer = TransitTimer(rec, window_factor=2.5)
t_obs, t_err = timer.run()

oc_min, oc_err_min = timer.compute_oc(t_obs, t_err)

sigma_t_measured = float(np.median(t_err) * 86400.0)   # BTJD -> seconds
print(f"\nResults:")
print(f"  Transits extracted:       {len(t_obs)}")
print(f"  Measured sigma_t:         {sigma_t_measured:.1f} s")
print(f"  Analytic sigma_t:         {rec.sigma_t_s:.1f} s")
print(f"  RMS O-C:                  {np.std(oc_min):.3f} min")
print(f"  Max |O-C|:                {np.max(np.abs(oc_min)):.3f} min")

# ---------------------------------------------------------------------------
# 3. Sensitivity curve
# ---------------------------------------------------------------------------
print("\n[2] Computing sensitivity curve ...")
sens = SensitivityCalculator(snr_threshold=3.0)
a_moon_au, m_min_mearth = sens.sensitivity_curve(rec, sigma_t_s=sigma_t_measured)

# Key limits
idx_001  = np.argmin(np.abs(a_moon_au - 0.001))
idx_hill = len(a_moon_au) - 1
print(f"  At a_moon = 0.001 AU:  M_lim = {m_min_mearth[idx_001]:.1f} M_earth")
print(f"  At Hill boundary:      M_lim = {m_min_mearth[idx_hill]:.1f} M_earth")

# ---------------------------------------------------------------------------
# 4. Plots
# ---------------------------------------------------------------------------
print("\n[3] Generating plots ...")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# O-C diagram
n = len(oc_min)
axes[0].errorbar(np.arange(n), oc_min, yerr=oc_err_min,
                 fmt="ko", ms=4, lw=1.2, capsize=2)
axes[0].axhline(0, color="gray", ls="--", lw=0.8)
med_err = np.median(oc_err_min)
axes[0].fill_between(np.arange(n), -med_err, med_err,
                      alpha=0.2, color="steelblue",
                      label=rf"$\sigma_t$ = {sigma_t_measured:.0f} s")
axes[0].set_xlabel("Transit Epoch")
axes[0].set_ylabel("O – C (minutes)")
axes[0].set_title("WASP-18 b  |  O-C Diagram")
axes[0].legend(fontsize=9)
axes[0].grid(True, alpha=0.25)

# Sensitivity curve
axes[1].plot(a_moon_au, m_min_mearth, "k-", lw=2)
axes[1].fill_between(a_moon_au, m_min_mearth, m_min_mearth.max(),
                      alpha=0.15, color="red", label="Undetectable")
axes[1].fill_between(a_moon_au, 0, m_min_mearth,
                      alpha=0.15, color="green", label="Detectable (SNR ≥ 3)")
axes[1].axhline(1.0,   color="gray", ls="--", lw=1, label=r"1 $M_\oplus$")
axes[1].axhline(17.15, color="gray", ls=":",  lw=1, label=r"$M_\text{Neptune}$")
axes[1].axvline(rec.stable_limit_au, color="red", ls="--", lw=1,
                alpha=0.7, label=r"0.5 $r_H$")
axes[1].set_xlabel("Moon Semi-major Axis (AU)")
axes[1].set_ylabel(r"Min. Detectable Moon Mass ($M_\oplus$)")
axes[1].set_title(
    rf"WASP-18 b  |  $M_p = {rec.planet_mass_mjup:.1f}\,M_J$  |  "
    rf"$\sigma_t = {sigma_t_measured:.0f}$ s"
)
axes[1].set_yscale("log")
axes[1].legend(fontsize=8)
axes[1].grid(True, alpha=0.25, which="both")

plt.tight_layout()
plt.savefig("wasp18b_single_system.png", dpi=150, bbox_inches="tight")
print("  Saved: wasp18b_single_system.png")

print("\nDone.")
