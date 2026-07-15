# python-potato

Four tools for GPCR conformational free-energy landscapes, each built in a
separate session and merged here:

| # | Tool | Location | Docs |
|---|------|----------|------|
| 1 | Protonation from pKa + pH (WSME model, pH-linkage) | `linkage_pka/`, `wsme_gpcr/` | [README_wsme_gpcr.md](README_wsme_gpcr.md) |
| 2 | Diverse conformational ensemble from AlphaFold/ColabFold | `gpcr_ensemble/` | [README_gpcr_ensemble.md](README_gpcr_ensemble.md) |
| 3 | Gibbs free energy per structure | `gibbs/` | [gibbs/README.md](gibbs/README.md) |
| 4 | Energy conformational landscapes from ensemble + Gibbs energies | `gpcr_energy_landscapes/` | below |

Note: tool 1's branch (`wsme_gpcr`/`linkage_pka`) also independently computes
its own multi-basin free-energy landscapes directly from a structure via a
statistical-mechanical (WSME) model with built-in pH-dependence -- it doesn't
require tools 2/3 to produce a landscape on its own. Tool 4 below takes the
complementary approach described in Fleetwood et al. 2021: landscapes
projected from an explicit conformational *ensemble* (tool 2) with
per-structure Gibbs energies (tool 3), along key distances or in PCA/MDS/t-SNE
space.

## gpcr_energy_landscapes (tool 4)

Tool 4 of the GPCR pipeline: turns a protonated, diverse conformational
ensemble and per-structure Gibbs free energies into the free-energy
conformational landscapes described in Fleetwood, Carlsson & Delemotte,
*"Identification of ligand-specific GPCR states and prediction of downstream
efficacy via data-driven modeling"*, eLife 2021;10:e60715 -- 1D/2D landscapes
along key microswitch distances, and PCA/MDS/t-SNE embeddings colored by
free energy.

### Interface contract with tools 1-3

This tool doesn't assume a specific implementation of tools 1-3, only that
their combined output looks like:

* an **ensemble directory**: one PDB (or mmCIF) file per conformer, named
  `<structure_id>.pdb`, already protonated by tool 1.
* an **energies table** (CSV or DataFrame): a `structure_id` column joining
  to the ensemble, plus a `gibbs_kcal_mol` column from tool 3 (column names
  are configurable).

If those don't match tools 1-3's actual output format, only `gpcr_energy_landscapes/io.py`
needs to change -- everything downstream operates on Biopython `Structure`
objects and plain per-structure free energies.

### Usage

```python
from gpcr_energy_landscapes import io, pipeline, plotting
from gpcr_energy_landscapes.collective_variables import BETA2AR_MICROSWITCHES

ensemble = io.load_ensemble("conformers/")          # tool 2's output
energies = io.load_energies("gibbs_energies.csv")   # tool 3's output
refs = {"active": io.load_structure("active_ref.pdb"),
        "inactive": io.load_structure("inactive_ref.pdb")}

cv_table = pipeline.compute_cv_table(ensemble, BETA2AR_MICROSWITCHES, refs=refs)
merged = pipeline.merge_with_energies(cv_table, energies)

# 1D free-energy landscape along one collective variable (Figure 2a style)
landscape = pipeline.build_1d_landscape(merged, "tm5_bulge")

# 2D free-energy landscape along two collective variables (Figure 2b style)
landscape2d = pipeline.build_2d_landscape(merged, "tm5_bulge", "ionic_lock")

# Dimensionality-reduction embedding colored by free energy (Figure 3 style)
embedding_df, model = pipeline.build_embedding_landscape(
    merged, feature_cols=["tm5_bulge", "ionic_lock", "y_y_motif"], method="pca"
)
```

Or from the command line:

```bash
python -m gpcr_energy_landscapes landscape1d \
    --ensemble conformers/ --energies gibbs.csv --cv tm5_bulge --out tm5_bulge.png

python -m gpcr_energy_landscapes landscape2d \
    --ensemble conformers/ --energies gibbs.csv \
    --cv-x tm5_bulge --cv-y ionic_lock --out landscape2d.png

python -m gpcr_energy_landscapes embed \
    --ensemble conformers/ --energies gibbs.csv --embed-method pca --out embedding.png
```

Collective variables (`gpcr_energy_landscapes/collective_variables.py`)
reproduce the paper's microswitches (TM5 bulge, ionic lock, Y-Y motif,
connector ΔRMSD) using generic chain/residue selectors, so the same code
works for any GPCR ensemble -- `BETA2AR_MICROSWITCHES` is provided as a
usable default matching the paper's β2AR residue numbering.

Free energy landscapes (`gpcr_energy_landscapes/energy_landscape.py`) support
two estimation modes depending on what tool 3 actually outputs:
* `gibbs`: each conformer carries its own Gibbs free energy; energies of
  conformers landing in the same region of CV space are combined via the
  Boltzmann-weighted partition function.
* `counts` / `weighted`: conformers are (optionally weighted) samples from a
  Boltzmann ensemble; the landscape is the usual `-RT ln(density)` estimate.

Both a smooth (Gaussian KDE, better for sparse AlphaFold-style ensembles) and
a raw weighted-histogram estimator are available.

### Real end-to-end run (tools 1, 3, 4 real; tool 2 substituted)

`examples/gpcr_pipeline_real_demo.py` runs the actual tool 1 (PROPKA3
pKa prediction) and tool 3 (AMBER ff14SB + GBn2 + RRHO Gibbs energy via
OpenMM/PDBFixer) code against a real structure, with a lightweight
Calpha-ANM ensemble standing in for tool 2 (GPU-bound ColabFold folding
isn't available in this environment). Results, plots, and a full writeup
of what's real vs. substituted are committed under
[`examples/output/real_demo/`](examples/output/real_demo/README.md).

### Setup

```bash
pip install -r requirements.txt
```

### Tests & demo

```bash
python -m pytest tests/ -q
PYTHONPATH=. python examples/synthetic_demo.py   # generates a synthetic ensemble end-to-end
```

`examples/synthetic_demo.py` fabricates a ~100-conformer ensemble across an
active-like and inactive-like basin (with a Gibbs energy bias favoring the
active basin) and writes example landscape plots to `examples/output/`.

test repo
this is a test in 2026
