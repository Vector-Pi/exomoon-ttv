"""
exomoon-ttv
===========
Population-level exomoon upper limits via TESS transit timing variations.

Author: Om Arora  (https://vector-pi.github.io/omarora/)
"""

from .sample      import PlanetSample, PlanetRecord
from .timing      import TransitTimer
from .sensitivity import SensitivityCalculator
from .population  import ExomoonPopulation
from .utils       import (
    moon_ttv_amplitude_s,
    orbital_velocity_ms,
    hill_radius_au,
    tess_timing_precision_s,
    transit_snr,
)

__version__ = "1.0.0"
__all__ = [
    "PlanetSample", "PlanetRecord",
    "TransitTimer",
    "SensitivityCalculator",
    "ExomoonPopulation",
    "moon_ttv_amplitude_s",
    "orbital_velocity_ms",
    "hill_radius_au",
    "tess_timing_precision_s",
    "transit_snr",
]
