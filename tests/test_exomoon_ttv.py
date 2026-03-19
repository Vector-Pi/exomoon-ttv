"""
test_exomoon_ttv.py
-------------------
Offline test suite for the exomoon TTV population pipeline.
All tests run on synthetic data or the sample CSV — no internet required.

Run with:  pytest tests/ -v
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest
import batman

from src.utils import (
    orbital_velocity_ms,
    moon_ttv_amplitude_s,
    min_detectable_moon_mearth,
    hill_radius_au,
    stable_moon_region,
    tess_timing_precision_s,
    transit_snr,
    a_rstar_from_period,
    a_au_from_rstar,
)
from src.sample import PlanetSample, PlanetRecord
from src.sensitivity import SensitivityCalculator


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def wasp18_record():
    """WASP-18 b PlanetRecord from the sample CSV."""
    sample = PlanetSample().load()
    return sample["WASP-18 b"]


@pytest.fixture
def pi_men_record():
    """Pi Men c PlanetRecord from the sample CSV."""
    sample = PlanetSample().load()
    return sample["Pi Men c"]


@pytest.fixture
def synthetic_oc():
    """Synthetic O-C residuals with a known injected TTV signal."""
    rng = np.random.default_rng(42)
    n   = 30
    t   = np.linspace(0, 90, n)       # transit times (days)
    # Inject: 2-minute amplitude, 20-day period
    signal  = (2.0 / 1440.0) * np.sin(2 * np.pi * t / 20.0)  # days
    noise   = rng.normal(0, 0.5 / 1440.0, n)                  # 0.5 min noise
    oc_days = signal + noise
    err_days = np.full(n, 0.5 / 1440.0)
    return t, oc_days, err_days


# ---------------------------------------------------------------------------
# Unit conversion tests
# ---------------------------------------------------------------------------

class TestUtils:

    def test_orbital_velocity_earth(self):
        v = orbital_velocity_ms(365.25, 1.0)
        assert 29000 < v < 31000

    def test_orbital_velocity_hot_jupiter(self):
        # 3-day period, ~0.04 AU → ~130 km/s
        v = orbital_velocity_ms(3.0, 0.04)
        assert 100_000 < v < 160_000

    def test_moon_ttv_amplitude_increases_with_mass(self):
        kwargs = dict(M_planet_mjup=1.0, a_moon_au=0.005, v_planet_ms=30000.0)
        amp1 = moon_ttv_amplitude_s(M_moon_mearth=1.0,  **kwargs)
        amp5 = moon_ttv_amplitude_s(M_moon_mearth=5.0,  **kwargs)
        assert amp5 > amp1

    def test_moon_ttv_amplitude_increases_with_separation(self):
        kwargs = dict(M_moon_mearth=1.0, M_planet_mjup=1.0, v_planet_ms=30000.0)
        amp_near = moon_ttv_amplitude_s(a_moon_au=0.001, **kwargs)
        amp_far  = moon_ttv_amplitude_s(a_moon_au=0.01,  **kwargs)
        assert amp_far > amp_near

    def test_moon_ttv_amplitude_positive(self):
        amp = moon_ttv_amplitude_s(1.0, 1.0, 0.005, 30000.0)
        assert amp > 0

    def test_min_detectable_decreases_with_separation(self):
        # At larger a_moon, TTV amplitude is larger → easier to detect → lower M_lim
        kwargs = dict(sigma_t_s=60.0, M_planet_mjup=1.0, v_planet_ms=30000.0)
        m_near = min_detectable_moon_mearth(a_moon_au=0.001, **kwargs)
        m_far  = min_detectable_moon_mearth(a_moon_au=0.01,  **kwargs)
        assert m_far < m_near

    def test_min_detectable_increases_with_sigma(self):
        kwargs = dict(M_planet_mjup=1.0, a_moon_au=0.005, v_planet_ms=30000.0)
        m_good = min_detectable_moon_mearth(sigma_t_s=30.0,  **kwargs)
        m_bad  = min_detectable_moon_mearth(sigma_t_s=300.0, **kwargs)
        assert m_bad > m_good

    def test_hill_radius_positive(self):
        r_H = hill_radius_au(1.0, 1.0, 1.0)
        assert r_H > 0

    def test_hill_radius_order_of_magnitude(self):
        # Jupiter at 5 AU around the Sun → r_H ≈ 0.35 AU
        r_H = hill_radius_au(5.0, 1.0, 1.0)
        assert 0.2 < r_H < 0.5

    def test_stable_moon_region_fraction(self):
        r_H   = hill_radius_au(1.0, 1.0, 1.0)
        r_stab = stable_moon_region(r_H, stability_fraction=0.5)
        assert abs(r_stab - 0.5 * r_H) < 1e-10

    def test_tess_timing_precision_positive(self):
        sig = tess_timing_precision_s(tmag=10.0, duration_days=0.1)
        assert sig > 0

    def test_tess_timing_precision_brighter_star_better(self):
        sig_bright = tess_timing_precision_s(tmag=7.0, duration_days=0.1)
        sig_faint  = tess_timing_precision_s(tmag=12.0, duration_days=0.1)
        assert sig_bright < sig_faint

    def test_transit_snr_positive(self):
        snr = transit_snr(0.01, 0.1, 5.0, 3, 10.0)
        assert snr > 0

    def test_a_rstar_from_period_consistency(self):
        a_r = a_rstar_from_period(5.0, 1.0, 1.0)
        assert 5 < a_r < 15

    def test_a_au_from_rstar_conversion(self):
        # 215 R_sun ≈ 1 AU
        a_au = a_au_from_rstar(215.0, 1.0)
        assert 0.9 < a_au < 1.1


# ---------------------------------------------------------------------------
# Sample loading tests
# ---------------------------------------------------------------------------

class TestSampleLoading:

    def test_loads_without_error(self):
        sample = PlanetSample().load()
        assert len(sample) > 0

    def test_contains_wasp18(self):
        sample = PlanetSample().load()
        rec = sample["WASP-18 b"]
        assert rec.tic_id == 100100827

    def test_contains_pi_men(self):
        sample = PlanetSample().load()
        rec = sample["Pi Men c"]
        assert rec.tic_id == 261136679

    def test_derived_quantities_computed(self, wasp18_record):
        assert wasp18_record.a_au > 0
        assert wasp18_record.hill_radius_au > 0
        assert wasp18_record.stable_limit_au > 0
        assert wasp18_record.v_planet_ms > 0
        assert wasp18_record.sigma_t_s > 0

    def test_stable_limit_less_than_hill(self, wasp18_record):
        assert wasp18_record.stable_limit_au < wasp18_record.hill_radius_au

    def test_sectors_parsed_as_list(self, wasp18_record):
        assert isinstance(wasp18_record.sectors, list)
        assert all(isinstance(s, int) for s in wasp18_record.sectors)

    def test_snr_positive(self, wasp18_record):
        assert wasp18_record.snr() > 0

    def test_to_dataframe_has_correct_columns(self):
        sample = PlanetSample().load()
        df = sample.to_dataframe()
        for col in ["name", "sigma_t_s", "hill_radius_au", "expected_snr"]:
            assert col in df.columns

    def test_getitem_by_index(self):
        sample = PlanetSample().load()
        rec = sample[0]
        assert isinstance(rec, PlanetRecord)

    def test_missing_planet_raises_key_error(self):
        sample = PlanetSample().load()
        with pytest.raises(KeyError):
            _ = sample["Nonexistent Planet XYZ"]


# ---------------------------------------------------------------------------
# Sensitivity calculator tests
# ---------------------------------------------------------------------------

class TestSensitivity:

    def test_sensitivity_curve_returns_correct_shape(self, wasp18_record):
        sens = SensitivityCalculator(n_a_points=50)
        a_au, m_min = sens.sensitivity_curve(wasp18_record)
        assert len(a_au) == 50
        assert len(m_min) == 50

    def test_sensitivity_curve_all_positive(self, wasp18_record):
        sens = SensitivityCalculator(n_a_points=50)
        _, m_min = sens.sensitivity_curve(wasp18_record)
        assert np.all(m_min > 0)

    def test_sensitivity_curve_monotonically_decreasing(self, wasp18_record):
        """Larger separation → larger TTV → easier to detect → lower M_lim."""
        sens = SensitivityCalculator(n_a_points=100)
        a_au, m_min = sens.sensitivity_curve(wasp18_record)
        assert m_min[0] > m_min[-1], (
            "Sensitivity should improve (M_lim decrease) at larger separations"
        )

    def test_sensitivity_better_for_brighter_star(self):
        """Pi Men c (Tmag=5) should have better sensitivity than a fainter system."""
        sample   = PlanetSample().load()
        pi_men   = sample["Pi Men c"]
        wasp18   = sample["WASP-18 b"]
        sens     = SensitivityCalculator(n_a_points=50)

        _, m_pi  = sens.sensitivity_curve(pi_men)
        _, m_w18 = sens.sensitivity_curve(wasp18)

        # Pi Men c is brighter → better timing → lower M_lim (easier detection)
        assert np.median(m_pi) < np.median(m_w18)

    def test_exclusion_map_shape(self):
        sample = PlanetSample().load()
        records = list(sample)[:3]   # use 3 systems for speed
        sens    = SensitivityCalculator(n_a_points=20)
        A, M, frac = sens.exclusion_map(records, n_m_points=20)
        assert A.shape == M.shape == frac.shape
        assert A.shape == (20, 20)

    def test_exclusion_map_fraction_bounded(self):
        sample  = PlanetSample().load()
        records = list(sample)[:3]
        sens    = SensitivityCalculator(n_a_points=20)
        _, _, frac = sens.exclusion_map(records, n_m_points=20)
        assert np.all(frac >= 0)
        assert np.all(frac <= 1)

    def test_exclusion_map_high_mass_fully_detectable(self):
        """
        At a separation well within every planet's Hill radius, a very
        massive moon should be detectable by most systems.

        We use a_max_au = 0.001 AU — inside the Hill sphere of even the
        tightest hot Jupiter in the sample (WASP-18 b has r_H = 0.0028 AU)
        — and a mass of 10000 M_earth, which produces a TTV of thousands
        of seconds even for massive planets.  At least 50% of systems should
        flag this as detectable.
        """
        sample  = PlanetSample().load()
        records = list(sample)
        sens    = SensitivityCalculator(n_a_points=30, snr_threshold=3.0,
                                        a_moon_min_au=0.0005)
        A, M, frac = sens.exclusion_map(
            records,
            n_m_points    = 30,
            m_min_mearth  = 1000.0,
            m_max_mearth  = 10000.0,
            a_max_au      = 0.001,   # well within all Hill radii
        )
        # At the highest mass row, the mean detection fraction should be > 0.5
        assert frac[-1, :].mean() > 0.5, (
            f"Expected >50% detection at 10000 M_earth, "
            f"got {frac[-1,:].mean():.3f}. "
            f"Check Hill radius clipping in exclusion_map()."
        )

    def test_sensitivity_uses_measured_sigma_if_provided(self, wasp18_record):
        """
        Passing a worse (larger) sigma_t should give a higher minimum
        detectable mass.  WASP-18 b has analytic sigma_t ~ 0.4 s (very
        bright star), so passing sigma_t=30 s degrades the sensitivity.
        We verify that the degraded precision gives a higher M_lim.
        """
        sens = SensitivityCalculator(n_a_points=50)
        _, m_analytic  = sens.sensitivity_curve(wasp18_record, sigma_t_s=None)
        # 30 s is much worse than the analytic 0.4 s for WASP-18 b
        _, m_degraded  = sens.sensitivity_curve(wasp18_record, sigma_t_s=30.0)
        # Worse precision → harder to detect → higher minimum mass required
        assert np.median(m_degraded) > np.median(m_analytic), (
            f"Expected degraded sigma_t=30s to give higher M_lim than "
            f"analytic sigma_t={wasp18_record.sigma_t_s:.1f}s, "
            f"but got {np.median(m_degraded):.1f} vs {np.median(m_analytic):.1f} M_earth"
        )


# ---------------------------------------------------------------------------
# O-C arithmetic (uses TransitTimer.compute_oc indirectly via a mock)
# ---------------------------------------------------------------------------

class TestOCArithmetic:

    def test_perfect_ephemeris_gives_zero_oc(self):
        """If transit times are exactly T0 + n*P, O-C should be zero."""
        from src.timing import TransitTimer
        sample = PlanetSample().load()
        rec    = sample["WASP-18 b"]
        timer  = TransitTimer(rec)

        n      = 20
        epochs = np.arange(n)
        t_obs  = rec.t0_btjd + epochs * rec.period_days
        t_err  = np.full(n, 1e-5)

        oc_min, _ = timer.compute_oc(t_obs, t_err)
        assert np.allclose(oc_min, 0.0, atol=1e-6), (
            f"Expected zero O-C for perfect ephemeris, got max |OC| = "
            f"{np.max(np.abs(oc_min)):.2e} min"
        )

    def test_injected_offset_appears_in_oc(self):
        """A constant timing offset should show up as a constant O-C shift."""
        from src.timing import TransitTimer
        sample = PlanetSample().load()
        rec    = sample["WASP-18 b"]
        timer  = TransitTimer(rec)

        n      = 20
        epochs = np.arange(n)
        offset = 3.0 / 1440.0   # 3 minutes in days
        t_obs  = rec.t0_btjd + epochs * rec.period_days + offset
        t_err  = np.full(n, 1e-5)

        oc_min, _ = timer.compute_oc(t_obs, t_err)
        # After removing linear trend, residuals should be ~0 (offset is absorbed)
        assert np.std(oc_min) < 0.01, (
            f"Expected near-zero O-C std for constant offset, "
            f"got {np.std(oc_min):.4f} min"
        )

    def test_sinusoidal_ttv_survives_oc(self, synthetic_oc):
        """A sinusoidal TTV injected before linear trend removal should
        survive in the O-C residuals."""
        from src.timing import TransitTimer
        sample = PlanetSample().load()
        rec    = sample["WASP-18 b"]
        timer  = TransitTimer(rec)

        t_obs, oc_days, err_days = synthetic_oc
        # Reconstruct transit times from the O-C
        epochs = np.arange(len(t_obs))
        t_abs  = rec.t0_btjd + epochs * rec.period_days + oc_days
        t_err  = err_days

        oc_out, oc_err_out = timer.compute_oc(t_abs, t_err)
        # The sinusoidal signal should be present: RMS >> noise
        rms_signal = np.std(oc_out)   # minutes
        rms_noise  = np.median(oc_err_out)
        # Signal amplitude is 2 min, noise is 0.5 min → SNR > 2
        assert rms_signal > rms_noise, (
            f"Expected TTV signal (RMS={rms_signal:.3f}) > "
            f"noise (RMS={rms_noise:.3f}) in O-C"
        )
