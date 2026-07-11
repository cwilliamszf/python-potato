# wsme-gpcr

A Python port of the blocked WSME (bWSME) statistical mechanical model for
protein / GPCR conformational free-energy landscapes, reimplementing:

- [AthiNaganathan/WSMEmodel](https://github.com/AthiNaganathan/WSMEmodel) (SSA + DSA + DSAw/L approximations)
- [AthiNaganathan/GPCR-Landscapes](https://github.com/AthiNaganathan/GPCR-Landscapes) (the same model applied to GPCR active/inactive structures)

Reference: Gopi S, Aranganathan A, Naganathan AN. *"Thermodynamics and
folding landscapes of large proteins from a statistical mechanical
model."* Curr Res Struct Biol. 2019 Oct 23;1:6-12.

## What it computes

Given a protein structure, the WSME model treats folding as an
order-disorder transition along contiguous stretches of residues
("blocks"). It enumerates microstates where one (SSA) or two (DSA /
DSAw/L) contiguous blocks are folded and everything else is unfolded,
weighting each by its native-contact stabilization energy, electrostatics,
and conformational entropy cost, then sums them into a partition
function. From that you get:

- **1D free-energy profile** vs. number of structured blocks
- **2D free-energy landscape** (N-terminal vs. C-terminal structured
  block count) — this is what reveals multiple basins/an intermediate,
  which is the point of the "GPCR landscape" analysis (inactive vs.
  active-like conformational states)
- **Residue folding probability** as a function of the reaction coordinate
- **DSC thermogram** (heat capacity vs. temperature)

## What's different from the MATLAB original

- **No STRIDE dependency.** The MATLAB tool requires running the external
  STRIDE program and hand-formatting its output as `struct.txt`. This port
  computes an approximate helix/strand/3-10-helix assignment directly from
  backbone coordinates (Ramachandran phi/psi classification with run-length
  smoothing) — no external binary needed. On the bundled CI2 test case this
  agrees with the real STRIDE assignment on 82% of residues (correctly
  identifies the main helix and most strands; some turn/loop residues are
  misclassified, as expected from torsion angles alone without H-bond
  geometry). If you have real STRIDE/DSSP output, pass it via
  `--ss-codes`/`--ss-file` (CLI) or `secondary_structure_from_codes()`
  (API) for exact fidelity to the original tool.
- **Vectorized enumeration.** The MATLAB code recomputes contact-map
  submatrix sums from scratch in 2-4 nested loops per microstate, which
  is only practical for small proteins (their own docs: ~3 minutes for a
  65-residue protein). This port uses 2D prefix sums ("summed area
  tables") so each microstate's energy is an O(1) lookup, making
  GPCR-sized proteins (250-350 residues, tens of thousands to a few
  million microstates) run in seconds. The vectorized engine is checked
  against a literal brute-force translation of the original nested loops
  on random small systems in `tests/test_wsme_engine.py`.
- **Scope**: this port covers the free-energy landscape (1D/2D) + residue
  folding probability + DSC thermogram. It does not implement the
  residue-residue coupling free-energy / phi-value machinery from
  `FesCalc_Block_full.m` (used for allosteric-pathway analysis in some of
  the GPCR papers) — that's a separate, larger addition if you need it.
- PDB **and mmCIF** input (via Biopython), not just fixed-column PDB text.

## Install

```bash
pip install -e ".[dev]"
```

## Usage

### GUI

A Streamlit GUI exposes every option (structure upload, chain/model, pH,
secondary-structure source, block size, parameter preset and individual
overrides, DSC sweep range):

```bash
pip install -e ".[gui]"
streamlit run wsme_gpcr/app.py
```

Then open the printed local URL, upload a PDB/mmCIF file (try
`examples/data/CI2.pdb` first), and click **Run**. Results (1D profile, 2D
landscape, residue folding probability, and DSC if enabled) render inline
with download buttons for the underlying data files.

Check **Run for all pH values** to get the full analysis at pH 7, 5, 3.5,
and 2 from a single upload -- each pH is a fully independent run (pH
changes which atoms carry a titratable charge, which feeds back into the
contact map itself, not just electrostatic screening), shown as a
pH-overlaid 1D profile comparison, a summary table, and a per-pH tab with
the full breakdown.

### CLI

```bash
wsme-gpcr examples/data/CI2.pdb --preset soluble --out-dir out/
wsme-gpcr my_gpcr.pdb --preset membrane --block-size 4 --dsc --out-dir out/
wsme-gpcr my_gpcr.pdb --preset membrane --all-ph --out-dir out/   # pH 7/5/3.5/2 in one run
```

`--preset membrane` (default) uses dielectric=4 and the GPCR-tuned energy
parameters from GPCR-Landscapes; `--preset soluble` uses dielectric=29 and
the water-soluble-protein parameters from the base WSMEmodel repo (matches
the CI2 example in that repo). Any individual parameter (`--ene`, `--ds`,
`--dcp`, `--ionic-strength`, `--dielectric`, `--temp`) can be overridden.
Run `wsme-gpcr --help` for the full option list.

This writes `1D_FreeEnergyProfile.txt`, `2D_FreeEnergySurface.txt`,
`ResFoldProb_vs_RC.txt`, `summary.png` (and `DSC_Thermogram.txt` with
`--dsc`) to the output directory. With `--all-ph`, each pH gets its own
`pH_<value>/` subdirectory plus a top-level `pH_comparison.png` overlaying
the four 1D profiles.

### Library

```python
from wsme_gpcr import (
    load_structure, assign_secondary_structure, compute_contact_map,
    build_blocks, WSMEParams, run_wsme,
)
from wsme_gpcr.plotting import plot_summary

structure = load_structure("my_gpcr.pdb")          # or .cif
ss_mask = assign_secondary_structure(structure)     # or supply your own
contacts = compute_contact_map(structure)
blocks = build_blocks(ss_mask, contacts, block_size=4)

params = WSMEParams()  # membrane/GPCR preset; see WSMEParams.soluble_protein_defaults()
result = run_wsme(structure, blocks, ss_mask, params)

print(result.zfin, result.stats)
plot_summary(result, save_path="landscape.png")
```

For a GPCR active-vs-inactive comparison (the actual point of the
GPCR-Landscapes repo), run this on both conformational structures and
compare the resulting `fes2D` landscapes and `fes` profiles — a
multi-basin 2D landscape or a shift in the dominant basin is the signal
of interest, not a single scalar.

## Performance notes

DSA/DSAw/L enumeration scales roughly as `nblocks^4`. For a ~300-residue
GPCR with `block_size=4` (~75 blocks) this is a few million microstates
and runs in seconds; going much below `block_size=3` on large receptors
will get slow and memory-heavy (a warning is emitted above 120 blocks).
Increasing `block_size` trades landscape resolution for speed, matching
the original tool's own guidance (block sizes above ~6 are untested
upstream).

## Tests

```bash
pytest
```

`tests/test_wsme_engine.py` validates the vectorized SSA/DSA/DSAw-L engine
against a literal brute-force translation of the original MATLAB nested
loops on random small synthetic systems. `tests/test_blocking.py` checks
the residue-to-block partitioning, including a MATLAB quirk preserved on
purpose: a leftover single-residue chunk merges into the previously
formed block rather than becoming its own singleton block, for every run
in the protein after the first full block has formed anywhere.
