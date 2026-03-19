# Exomoon TTV Population Study

A Python pipeline for placing population-level upper limits on exomoon masses using Transit Timing Variations (TTVs) from TESS photometry.

Rather than searching for a moon around a single planet, this pipeline processes a sample of known TESS planets, extracts per-transit central times for each, and produces a **population-level sensitivity map**: for what moon masses and orbital separations can we say moons are absent, across the observed sample?

Built and maintained by [Om Arora](https://vector-pi.github.io/omarora/).

---

## Scientific Background

No exomoon has been definitively confirmed. The best candidate, Kepler-1625b-i (Teachey & Kipping 2018), remains contested. The indirect detection method used here is **Transit Timing Variations (TTVs)**: a moon orbiting a transiting planet causes the planet to wobble around the planet-moon barycentre, shifting each transit time by

$$\delta t \approx \frac{M_m}{M_p + M_m} \cdot \frac{a_m}{v_p}$$

where $M_m$ is the moon mass, $M_p$ the planet mass, $a_m$ the moon semi-major axis, and $v_p$ the planet's orbital velocity. For an Earth-mass moon around a Jupiter-mass planet at 0.01 AU separation, $\delta t \sim 10$–100 s — at the edge of TESS timing precision for bright stars.

### Key references
- Kálmán et al. (2025) — CHEOPS exomoon search: [arXiv:2507.15318](https://arxiv.org/abs/2507.15318)
- Winterhalder et al. (2025) — Astrometric exomoon detection: [arXiv:2509.15304](https://arxiv.org/abs/2509.15304)
- Teachey & Kipping (2018) — Kepler-1625b-i candidate: [arXiv:1810.02730](https://arxiv.org/abs/1810.02730)
- Kreidberg et al. (2019) — Contested reanalysis: [arXiv:1908.10526](https://arxiv.org/abs/1908.10526)

---

## What This Does

| Module | Purpose |
|--------|---------|
| `sample.py` | Define and validate the planet sample; fetch parameters from NASA Exoplanet Archive |
| `timing.py` | Per-transit time extraction for a single system; O-C computation |
| `sensitivity.py` | Moon mass sensitivity curve per system; population exclusion map |
| `population.py` | Run the full pipeline over the sample; aggregate results |
| `plots.py` | All publication-quality figures |
| `utils.py` | Physical calculations, unit conversions, TESS noise model |

---

## Installation

```bash
git clone https://github.com/Vector-Pi/exomoon-ttv.git
cd exomoon-ttv
pip install -e .
```

Requires Python ≥ 3.9.

---

## Quick Start

```python
from src.population import ExomoonPopulation

pop = ExomoonPopulation()
pop.load_sample()           # loads data/sample.csv
pop.run(n_jobs=1)           # extracts TTVs for all systems
pop.plot_exclusion_map()    # population-level sensitivity figure
pop.summary_table()         # LaTeX table of per-system limits
```

---

## The Default Sample

`data/sample.csv` contains 12 well-characterised TESS planets chosen for bright host stars (Tmag < 11), multiple sectors, known planet mass from RV, and periods 1–20 days.

| TIC | Planet | Period (d) | Mp (MJ) | Tmag |
|-----|--------|-----------|---------|------|
| 261136679 | Pi Men c | 6.268 | 0.014 | 5.0 |
| 100100827 | WASP-18 b | 0.941 | 10.20 | 8.7 |
| 149603524 | WASP-126 b | 3.289 | 0.28 | 10.0 |
| 259592689 | TOI-270 b | 3.360 | 0.019 | 9.1 |
| 207141131 | WASP-121 b | 1.275 | 1.18 | 9.4 |
| 350618622 | LTT 9779 b | 0.792 | 0.55 | 8.4 |
| 460205581 | TOI-1431 b | 2.650 | 3.12 | 7.7 |
| 167418898 | WASP-69 b | 3.868 | 0.26 | 9.5 |
| 139375960 | KELT-9 b | 1.481 | 2.88 | 7.5 |
| 271893367 | TOI-132 b | 2.995 | 0.19 | 9.7 |
| 279741379 | HD 213885 b | 1.008 | 0.066 | 7.9 |
| 394050135 | TOI-700 d | 37.42 | 0.006 | 10.5 |

---

## Output

1. **Per-system O-C diagrams** — transit timing residuals with 1σ band
2. **Per-system sensitivity curves** — minimum detectable moon mass vs. separation
3. **Population exclusion map** — combined sensitivity across all systems
4. **Summary LaTeX table** — per-system timing precision and moon mass limits

---

## Tests

```bash
pytest tests/ -v
```

All tests run offline. Covers unit conversions, O-C arithmetic, sensitivity monotonicity, population aggregation logic, and noise model.

---

## Dependencies

| Package | Purpose |
|---------|---------|
| `lightkurve ≥2.4` | TESS data access |
| `astropy ≥5.3` | Units, BLS, timeseries |
| `batman-package ≥2.4.9` | Transit model |
| `numpy ≥1.24` | Arrays |
| `scipy ≥1.11` | Optimisation |
| `matplotlib ≥3.7` | Plotting |
| `astroquery ≥0.4.6` | NASA Exoplanet Archive |
| `pandas ≥2.0` | Sample table |
| `tqdm ≥4.65` | Progress bars |

---

## License

MIT. See `LICENSE`.

## Author

**Om Arora** · [omarora.netlify.app](https://omarora.netlify.app) · [Vector-Pi](https://github.com/Vector-Pi)
