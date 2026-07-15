"""
gpcr_energy_landscapes
=======================

Tool 4 in the GPCR pipeline: turns a *diverse conformational ensemble* (tool 2,
already protonated per-pH/pKa by tool 1) together with a *per-structure Gibbs
free energy* (tool 3) into the free-energy conformational landscapes seen in
Fleetwood, Carlsson & Delemotte, "Identification of ligand-specific GPCR
states and prediction of downstream efficacy via data-driven modeling",
eLife 2021;10:e60715 -- i.e. 1D/2D landscapes along key microswitch distances
(Figure 2) and dimensionality-reduction embeddings colored by free energy
(Figure 3).

This package does not assume a specific implementation of tools 1-3. It only
assumes their *outputs* look like:

  * an ensemble of protonated conformer structures (PDB files, one per
    conformer, in a directory) -- see :mod:`gpcr_energy_landscapes.io`
  * a table (CSV/DataFrame) with one Gibbs free energy value per structure,
    keyed by the same structure id -- see :mod:`gpcr_energy_landscapes.io`

See ``examples/synthetic_demo.py`` for a runnable end-to-end example using
synthetic data, and README.md for the interface contract expected from tools
1-3.
"""

from . import collective_variables, dimensionality_reduction, energy_landscape, io, pipeline, plotting

__all__ = [
    "collective_variables",
    "dimensionality_reduction",
    "energy_landscape",
    "io",
    "pipeline",
    "plotting",
]
