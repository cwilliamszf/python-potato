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
- **Residue-residue coupling free energy** (block j vs. block k: do they
  tend to fold together, independently, or against each other?) — the
  `CouplingMat` from `FesCalc_Block_full.m` / `Plot_Imp_Variables.m`,
  computed from co-occurrence statistics in the equilibrium ensemble.
  Entries where a block is folded in essentially every (or almost no)
  populated microstate are reported as undefined (NaN) rather than a
  numerically noisy value — there's genuinely no resolvable partial-folding
  signal there, not just a precision limit. Expect a sparse, mostly-NaN
  matrix for small, highly cooperative single-domain proteins (there's
  little partially-folded population to measure coupling from), and a
  richer matrix for large, multi-basin receptors like GPCRs.
- **In silico alanine-scanning mutagenesis** — computationally truncate
  each residue's side chain to Ala (backbone + CB only) one at a time,
  recompute the coupling free-energy matrix (`chi_plus`, "ΔG+" in the
  reference below), and diff it against the wild type. Averaging the
  per-mutant difference over one axis gives a per-block "mutational
  response" vector; stacking many mutants' vectors gives their mean/std
  across the structure — replicating Fig. 7 of Anantakrishnan &
  Naganathan, *"Thermodynamic architecture and conformational plasticity
  of GPCRs,"* Nat Commun 14, 128 (2023). This is a general-purpose
  workflow: it runs on any structure and, by default, every eligible
  residue in it (Ala/Gly/Pro are skipped, matching the paper) — not
  hardcoded to a particular receptor or mutation list. Mutating a residue
  never changes secondary structure, so the block partition — and hence
  the WT/mutant `chi_plus` matrices' shape and block indexing — is always
  identical, letting every mutant be compared to the wild type directly,
  element-wise, with no realignment step. See
  `alanine_scan.estimate_scan_seconds` for a time estimate before running
  a full receptor-wide scan (tens of minutes for ~250-300 residues).

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
- **Coupling analysis, made tractable.** The MATLAB coupling code frames
  itself as a per-residue-perturbation calculation, but as shipped the
  perturbation is a no-op (`pert = nres+1`), so the only quantity it ever
  actually produces is the single unperturbed (wild-type) coupling matrix
  — computed from co-occurrence statistics in the equilibrium ensemble.
  Even that alone was impractical to run at GPCR scale in MATLAB. Here
  it's computed with the same 2D-difference-array trick as the landscape
  itself (all four joint-probability quadrants accumulated directly, not
  derived by subtracting large aggregates — the naive approach catastrophically
  cancels whenever a block is folded in nearly every microstate, which is
  common), making it a few extra seconds on top of the landscape
  calculation rather than a separate, much slower run.
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
and 3D landscape, residue folding probability, and DSC/coupling if enabled)
render inline with download buttons for the underlying data files.

Check **Run for all pH values** to get the full analysis at pH 7, 5, 3.5,
and 2 from a single upload -- each pH is a fully independent run (pH
changes which atoms carry a titratable charge, which feeds back into the
contact map itself, not just electrostatic screening), shown as a
pH-overlaid 1D profile comparison, a summary table, and a per-pH tab with
the full breakdown.

Check **Run in silico alanine scan** to also run the mutational-response
workflow above, at the single pH selected in the sidebar (a mutational
scan is not repeated across the pH sweep). Choose an evenly-spaced
subsample (fast, still covers the whole structure), every eligible
residue (slow, full receptor-wide scan), or a specific comma-separated
list of residue numbers. Results show the mutational-response plot, a
top-hits table ranked by total perturbation magnitude, and, per top hit,
a ΔΔG+ vs. distance plot and a 3D structure map — with download buttons
for the underlying data.

### CLI

```bash
wsme-gpcr examples/data/CI2.pdb --preset soluble --out-dir out/
wsme-gpcr my_gpcr.pdb --preset membrane --block-size 4 --dsc --coupling --out-dir out/
wsme-gpcr my_gpcr.pdb --preset membrane --all-ph --out-dir out/   # pH 7/5/3.5/2 in one run
wsme-gpcr my_gpcr.pdb --preset membrane --alanine-scan --out-dir out/   # full receptor-wide Ala scan
```

`--preset membrane` (default) uses dielectric=4 and the GPCR-tuned energy
parameters from GPCR-Landscapes; `--preset soluble` uses dielectric=29 and
the water-soluble-protein parameters from the base WSMEmodel repo (matches
the CI2 example in that repo). Any individual parameter (`--ene`, `--ds`,
`--dcp`, `--ionic-strength`, `--dielectric`, `--temp`) can be overridden.
`--coupling` adds the residue-residue coupling free-energy matrix (roughly
doubles run time). Run `wsme-gpcr --help` for the full option list.

This writes `1D_FreeEnergyProfile.txt`, `2D_FreeEnergySurface.txt`,
`2D_FreeEnergyLandscape_3D.png`, `ResFoldProb_vs_RC.txt`, `summary.png`
(plus `DSC_Thermogram.txt` with `--dsc`, and `CouplingMatrix.txt` /
`CouplingMatrix.png` with `--coupling`) to the output directory. With
`--all-ph`, each pH gets its own `pH_<value>/` subdirectory plus
top-level `pH_comparison.png` / `pH_comparison_3D.png` overlaying the
four pH values.

`--alanine-scan` runs the mutational-response workflow described above
and writes an `alanine_scan/` subdirectory: `MutationalResponse.txt`/`.png`
(per-block mean ± std across all scanned mutants), `DeltaDeltaG.csv` (one
row per mutation × block), `TopHits.txt` (mutation sites ranked by total
perturbation magnitude), and per-top-hit `DistanceDependence_<resnum>.png`
/ `StructureMap_<resnum>.png`. It applies to *any* structure/receptor, not
just GPCRs, and by default scans every eligible residue — pass
`--ala-max-n N` to evenly subsample N sites instead (prints a time
estimate before running either way), or `--ala-positions 45,102,150` to
target specific author residue numbers. `--ala-top-n` controls how many
top hits get their own distance/structure-map plots (default 5). It runs
at the single `--ph` value even when combined with `--all-ph`/`--ph-values`
(a mutational scan is not repeated across a pH sweep).

### Library

The lowest-level pieces compose explicitly:

```python
from wsme_gpcr import (
    load_structure, assign_secondary_structure, compute_contact_map,
    build_blocks, WSMEParams, run_wsme, compute_coupling,
)
from wsme_gpcr.plotting import plot_summary

structure = load_structure("my_gpcr.pdb")          # or .cif
ss_mask = assign_secondary_structure(structure)     # or supply your own
contacts = compute_contact_map(structure)
blocks = build_blocks(ss_mask, contacts, block_size=4)

params = WSMEParams()  # membrane/GPCR preset; see WSMEParams.soluble_protein_defaults()
result = run_wsme(structure, blocks, ss_mask, params)
coupling = compute_coupling(structure, blocks, ss_mask, params)  # optional

print(result.zfin, result.stats)
plot_summary(result, coupling_result=coupling, save_path="landscape.png")
```

Or use `run_pipeline`/`run_pipeline_multi_ph` (what the CLI and GUI call
under the hood) to get all of the above, plus optional DSC/coupling, in
one call:

```python
from wsme_gpcr import run_pipeline

pr = run_pipeline("my_gpcr.pdb", with_dsc=True, with_coupling=True)
print(pr.result.zfin, pr.coupling_result.coupling_free_energy)
```

For a GPCR active-vs-inactive comparison (the actual point of the
GPCR-Landscapes repo), run this on both conformational structures and
compare the resulting `fes2D` landscapes and `fes` profiles — a
multi-basin 2D landscape or a shift in the dominant basin is the signal
of interest, not a single scalar.

`run_alanine_scan_pipeline` runs the mutational-response workflow on any
structure. It defaults to every eligible residue; pass `positions` for
specific sites or `max_positions` to evenly subsample:

```python
from wsme_gpcr import run_alanine_scan_pipeline

scan_pr = run_alanine_scan_pipeline("my_gpcr.pdb", max_positions=40)
print(scan_pr.scan.top_hits(10))          # [(resnum, perturbation_magnitude), ...]
dist, ddg = scan_pr.scan.ddg_vs_distance(102)   # one mutant's spatial decay profile
```

The lower-level building blocks (`alanine_scan.scannable_positions`,
`alanine_scan.run_alanine_scan`, `alanine_scan.estimate_scan_seconds`)
compose the same way as the rest of the library if you need finer control
— e.g. reusing an already-loaded `Structure`/`BlockModel` across many
scans, or wiring scan progress into your own UI via `progress_callback`.

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
loops on random small synthetic systems. `tests/test_coupling.py` does the
same for the coupling matrix, checking all four joint-probability
quadrants (not just the folded/folded one) sum to 1 and match brute force
exactly. `tests/test_blocking.py` checks the residue-to-block
partitioning, including a MATLAB quirk preserved on purpose: a leftover
single-residue chunk merges into the previously formed block rather than
becoming its own singleton block, for every run in the protein after the
first full block has formed anywhere.
