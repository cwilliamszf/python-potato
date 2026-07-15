# GPCR conformational ensemble generation with ColabFold/AlphaFold2

Pipeline for using ColabFold (AlphaFold2) to generate a diverse ensemble of GPCR
conformations — active, inactive, and intermediate states — rather than a single static
structure, and to organize/classify the resulting models.

## Why AlphaFold2 needs help to sample multiple states

Standard AlphaFold2/ColabFold is trained and run to predict *one* structure per sequence,
and it is very good at collapsing onto the single most co-evolutionarily-supported
conformation (usually whichever state dominates the PDB training set for that fold family).
GPCRs are functionally defined by switching between at least three basins — inactive
(ground state), active (agonist- and transducer-coupled), and various intermediate/meta-
stable states — so getting the ensemble out requires deliberately perturbing the standard
single-structure recipe. This pipeline combines three published, complementary strategies:

1. **Shallow-MSA subsampling** ([Del Alamo et al. 2022, *eLife*](https://elifesciences.org/articles/75751)) —
   randomly subsetting the input MSA to low depth removes much of the co-evolutionary
   signal that locks the model onto one state, and repeating this at many depths/replicates
   with dropout left on at inference and few recycles produces a spread of models across
   the conformational landscape. This was demonstrated directly on transporters and GPCRs.
2. **Sequence-cluster subsampling / AF-Cluster** ([Wayment-Steele et al. 2024, *Nature*](https://www.nature.com/articles/s41586-023-06832-9)) —
   splitting the full MSA into clusters of similar sequences (representing different
   ortholog/paralog sub-families that may prefer different conformations) and folding each
   cluster's shallow MSA independently. This recovered multiple experimentally validated
   states for fold-switching proteins.
3. **Dropout + multi-seed inference** (also from Del Alamo et al. 2022 / Wallner-style
   "AF2 ensembles") — keeping the structure-module dropout active at inference time and
   sampling many random seeds injects stochasticity into every individual prediction, so
   even a single MSA yields a distribution of models rather than one point estimate.

Optionally, custom structural templates from known active/inactive reference structures
of the receptor (or a close homolog) can be supplied to `colabfold_batch --custom-template-path`
to bias a subset of runs toward specific known basins, complementing the unbiased sampling
above.

None of this guarantees *complete* coverage of "all potential stable folded conformations"
— that is an open research problem, and metadynamics/MD-based approaches remain the
gold standard for exhaustive conformational sampling with a physical energy function. What
this pipeline gives you is a strong, well-attested AF2-based diversity generator plus the
bookkeeping to filter, deduplicate, and label the output.

## Pipeline stages (`gpcr_ensemble/`)

1. `msa_subsample.py` — turns one full a3m MSA into dozens–hundreds of shallow/clustered
   MSA variants (strategies 1 and 2 above).
2. `run_colabfold.py` — thin wrapper that runs `colabfold_batch` over the whole variant
   directory with dropout, multiple seeds, and low recycle count (strategy 3), and collects
   the resulting PDBs + confidence scores into a manifest.
3. `activation_state.py` — filters out unfolded/low-confidence junk (mean pLDDT + CA-CA
   bond-length sanity check), then classifies each surviving model as active / inactive /
   intermediate using the class-A "ionic lock" TM3–TM6 cytoplasmic distance microswitch,
   optionally calibrated per-receptor from your own reference PDBs.
4. `cluster.py` — Kabsch-RMSD-based structural clustering of the surviving models (over the
   TM core, to avoid loop noise) so you get a small set of *structurally distinct*
   representative conformations instead of hundreds of near-duplicates, plus an MDS
   projection for visualizing the sampled conformational landscape.
5. `pipeline.py` — CLI gluing all of the above into one command.

## Prerequisites

- ColabFold installed with a GPU available to actually fold structures — this repo only
  orchestrates it, it does not reimplement AlphaFold2. Use
  [LocalColabFold](https://github.com/YoshitakaMo/localcolabfold) for a local install, or
  the hosted ColabFold Colab notebook if you don't have a local GPU.
- `pip install -r requirements.txt` for the orchestration/analysis dependencies
  (numpy, scipy, biopython).
- A full MSA (`.a3m`) for your target GPCR. Generate one with:
  `colabfold_batch --msa-only receptor.fasta msa_out/`, which queries the ColabFold
  MMseqs2 server and writes `msa_out/receptor.a3m`.
- The residue numbers of the DRY-motif arginine (Ballesteros-Weinstein 3.50, TM3) and the
  cytoplasmic TM6 reference residue (~6.30–6.34) in *your* receptor's numbering. Look these
  up via [GPCRdb](https://gpcrdb.org) generic residue numbering for your receptor, or by
  aligning to a homolog of known structure.

## Usage

```bash
pip install -r requirements.txt

# 1. Get a full MSA for your receptor (needs network access to the ColabFold MSA server)
colabfold_batch --msa-only receptor.fasta msa_out/

# 2. Run the full ensemble pipeline (needs a GPU + colabfold_batch on PATH)
python -m gpcr_ensemble.pipeline \
    --a3m msa_out/receptor.a3m \
    --out results/ \
    --tm3-resnum 131 --tm6-resnum 272 \
    --inactive-ref inactive_template.pdb --active-ref active_template.pdb \
    --num-seeds 8 --num-recycle 3
```

This produces `results/ensemble_report.csv` — one row per structurally distinct cluster
representative, with its pLDDT, TM3–TM6 distance, and active/inactive/intermediate label —
plus every raw model under `results/models/`.

If reference active/inactive PDBs for your receptor (or a close homolog, renumbered to
match) aren't available, omit `--inactive-ref`/`--active-ref` and the pipeline falls back
to generic class-A default thresholds (`activation_state.DEFAULT_THRESHOLDS`); calibrating
per-receptor is strongly recommended since the absolute TM3–TM6 distances in active vs.
inactive states vary noticeably across receptors.

### Tuning for maximum diversity vs. compute budget

- `--num-seeds`: more seeds per MSA variant = more stochastic diversity per unit of MSA
  engineering, but scales compute linearly. 8–16 is a reasonable starting point.
- `--num-recycle`: lower values (1–3) trade per-model confidence for ensemble breadth;
  higher values tend to converge different starts back onto the same basin.
- MSA variant counts/fractions are set in `msa_subsample.generate_diverse_msas` — widen
  `random_fractions` / `n_random_replicates` / `cluster_distance_thresholds` for broader
  (and more expensive) coverage of the landscape.
- `--rmsd-cluster-cutoff` controls how finely the final ensemble is de-duplicated; smaller
  values keep more, finer-grained conformational states as separate representatives.

## Tests

The MSA-subsampling, structural-clustering, and activation-state-classification logic is
unit tested with synthetic data and does not require ColabFold, JAX, or a GPU:

```bash
pip install pytest
pytest tests/ -v
```

`run_colabfold.py`'s `run()`/`collect_manifest()` (the actual `colabfold_batch` invocation)
is not covered by these tests since it requires GPU compute and cannot run in this
environment — smoke-test it against your ColabFold install directly.
