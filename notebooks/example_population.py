"""
example_population.py
---------------------
Run the full exomoon TTV population study on the default 12-planet sample.

This script:
  1. Loads the sample from data/sample.csv
  2. Downloads TESS data and extracts transit times for all systems
  3. Computes per-system sensitivity curves
  4. Generates the population exclusion map
  5. Produces the LaTeX summary table

Runtime: approximately 30-90 minutes depending on MAST download speed
         and the number of transits per system.

Output files are saved to results/.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import matplotlib
matplotlib.use("Agg")   # non-interactive backend for batch runs

from src.population import ExomoonPopulation

print("=" * 60)
print("Exomoon TTV Population Study")
print("=" * 60)

# Initialise
pop = ExomoonPopulation(
    output_dir    = "results",
    snr_threshold = 3.0,
    window_factor = 2.5,
)

# Load sample
pop.load_sample()

# Run TTV extraction for all systems
# n_jobs=1 runs sequentially; change to n_jobs>1 when parallelism is added
pop.run(n_jobs=1)

# Print plain-text summary
pop.print_summary()

# Population exclusion map
pop.plot_exclusion_map(savepath="results/exclusion_map.png")

# All sensitivity curves overlaid
pop.plot_all_sensitivity_curves(savepath="results/all_sensitivity_curves.png")

# LaTeX table
pop.summary_table(savepath="results/summary_table.tex")

print("\nAll outputs saved to results/")
print("Done.")
