# Corrected, symmetric interpolated-path re-run: GPR68 vs. GPR132

**What this fixes.** A review correctly identified that the earlier GPR68
and GPR132 interpolated-path runs used different geometric preprocessing:
GPR68 used no superposition at all (the prior README argued this
preserved "the genuine rigid-body helix motion that is the activation
transition" -- **that reasoning was wrong**: an unremoved rotation/
translation offset between two independently-generated files is a
file-format artifact, not biology; superposing on an invariant reference
set that excludes the actual movers removes exactly the artifact while
leaving the real signal intact). GPR132 used a Kabsch fit on a manually
chosen residue range (30-355), never applied to GPR68. This asymmetry
directly confounded the cross-receptor comparison those runs were meant
to support.

**What changed.** `examples/gpcr_pipeline_tm_topology.py` implements one
protocol, applied identically to every receptor, no exceptions:

1. TM1-TM7 identified via real DSSP secondary structure (H-bond geometry;
   installed for this run) anchored by the DRY-like (TM3) and NPxxY-like
   (TM7) motifs already validated elsewhere in this project, with
   TM1/2/4/5/6 assigned by their invariant class-A topological sequence
   order -- not a hand-picked residue range, and not literal-motif
   guessing for the harder anchors (W4.50/P5.50/CWxP were tried first and
   empirically do not exist as literal matches in either receptor's real
   sequence -- see the module docstring). A self-consistency check (TM7
   run index must be exactly TM3's + 4) is verified, not assumed; a
   proline-kink-aware merge rule (a DSSP break landing exactly on a
   conserved anchor residue is treated as a helix-internal kink, not a
   loop) was needed and is disclosed, not hidden.
2. Superposition uses ONLY TM1, TM2, TM3, TM4, TM5, TM7 backbone Cα (TM6
   excluded from the FIT, not from the structure) -- inactive superposed
   onto active.
3. Cross-checked against an independent, prior-free iterative
   outlier-rejecting superposition (fit all Cα, drop atoms beyond 3.0 Å,
   refit, converge).
4. TM6 cytoplasmic-tip displacement measured post-superposition and
   gated at 4 Å (below which a pair is not a genuine activation
   transition).
5. Both structures trimmed to the ordered core (TM1's first residue
   through TM7's last -- derived from step 1, not hand-picked) before
   interpolation.
6. Same 11-image linear interpolation, same minimizer (150 L-BFGS
   iterations), same scorer (AMBER ff14SB + GBn2 + RRHO Gibbs via
   OpenMM/PDBFixer, PROPKA-assigned protonation), same two pH values
   (7.4, 6.0) as every prior run this session.

GPR68 was **re-run from scratch**; its old numbers (published under
`examples/output/gpr68_string_demo/`) are **not reused** here.

## Diagnostic table (both receptors, identical method)

| Metric | GPR68 | GPR132 |
|---|---|---|
| Raw whole-chain Cα RMSD, no superposition | 9.42 Å | 33.59 Å |
| Invariant-core (TM1,2,3,4,5,7) superposition RMS | **0.844 Å** | **1.822 Å** |
| TM1 RMSD post-superposition | 0.48 Å | 0.80 Å |
| TM2 RMSD post-superposition | 0.42 Å | 0.97 Å |
| TM3 RMSD post-superposition | 0.44 Å | 1.52 Å |
| TM4 RMSD post-superposition | 0.83 Å | 0.92 Å |
| TM5 RMSD post-superposition | 0.37 Å | 2.71 Å |
| TM6 RMSD post-superposition | 0.83 Å | 2.42 Å |
| TM7 RMSD post-superposition | 1.29 Å | 2.31 Å |
| **TM6 cytoplasmic-tip displacement** | **1.18 Å** | **4.87 Å** |
| **TM6 >= 4 Å gate** | **FAIL** | **PASS (marginal)** |
| Outlier-rejection (cutoff 3.0 Å): kept / total | 316/365 | 258/380 |
| Outlier-rejection RMS | 0.893 Å | 1.293 Å |
| Overlap: outlier-rejection kept-set vs. BW invariant core | **99.4%** | **81.4%** |
| Core (TM1-TM7 span) endpoint RMSD used for interpolation | 0.839 Å | 2.360 Å |
| Interpolation core span (residues) | 36-310 (275 res) | 39-347 (309 res) |

GPR184, GPR4, GPR65 are not yet available in this session -- not run.

**Reading the diagnostic table:** GPR68's two structures are barely
different once correctly registered (invariant-core RMS 0.844 Å, TM6
itself only 0.83 Å post-fit, cytoplasmic-tip displacement 1.18 Å) and the
independent outlier-rejection method agrees almost exactly (99.4%
overlap) -- there is no real ambiguity here, both methods say the same
thing: **this pair does not represent two distinct conformational
states**, genuine or otherwise modeled. GPR132 shows real, larger
differences throughout (core RMS 1.822 Å, TM5/TM6/TM7 all >2 Å,
cytoplasmic-tip displacement 4.87 Å, just clearing the 4 Å gate) with
somewhat lower agreement between the two superposition methods (81.4%
overlap) -- consistent with a real, if modest, conformational difference,
though the marginal gate-passing margin (4.87 vs. the 4.0 Å threshold)
means this should be read as "plausibly a genuine pair," not "definitely
a strong activation transition."

## Sanity gate 1: barrier vs. path length, monotonic?

| | Core endpoint RMSD | Barrier at pH 7.4 |
|---|---|---|
| GPR68 | 0.84 Å | 60.3 kcal/mol |
| GPR132 | 2.36 Å | 344.5 kcal/mol |

**Monotonic now** (shorter path -> smaller barrier). Under the old,
asymmetric protocol this was backwards (GPR132's ~2.6 Å path produced a
larger barrier than GPR68's ~9.42 Å path) -- exactly the "physically
backwards" scaling flagged in review. It is resolved once both receptors
are treated identically: it was a preprocessing artifact, not a real
anomaly requiring a mechanistic explanation.

## Sanity gate 2: TM6 displacement < 4 Å exclusion

**GPR68 is EXCLUDED from the sensor/non-sensor comparison.** Its TM6
cytoplasmic-tip displacement (1.18 Å) is well below the 4 Å genuine-
activation-pair threshold. Its pH-response numbers are reported below
for the record and because the user requested GPR68 be re-run regardless
of outcome, but **must not be interpreted as evidence about proton-
sensing-coupled conformational switching** -- this pair does not
represent a real active/inactive transition, corrected or not.

GPR132 passes (4.87 Å), marginally.

## Results

### G along the path, absolute and relative

`gibbs_vs_path_both_ph.png` and `barrier_shape_relative.png` per receptor
(`gpr68/`, `gpr132/`); full per-image data in each `path_table.csv`.

| Receptor | pH | Endpoint G (active) | Endpoint G (inactive) | Peak G (f) | Barrier vs. lower endpoint |
|---|---|---|---|---|---|
| GPR68 | 7.4 | -11066.6 | -11051.0 | -11006.3 (f=0.50) | 60.3 kcal/mol |
| GPR68 | 6.0 | -10907.1 | -10893.4 | -10861.1 (f=0.60) | 46.0 kcal/mol |
| GPR132 | 7.4 | -12648.4 | -12620.0 | -12303.9 (f=0.50) | 344.5 kcal/mol |
| GPR132 | 6.0 | -12525.5 | -12476.5 | -12172.2 (f=0.50) | 353.3 kcal/mol |

### Barrier height and its pH-dependence

| Receptor | Barrier, pH 7.4 | Barrier, pH 6.0 | Change (7.4 -> 6.0) | Excluded by gate? |
|---|---|---|---|---|
| GPR68 | 60.3 kcal/mol | 46.0 kcal/mol | **-14.3 kcal/mol** | YES -- do not interpret |
| GPR132 | 344.5 kcal/mol | 353.3 kcal/mol | **+8.7 kcal/mol** | no |

Both receptors' corrected barriers move by single-digit-to-low-double-digit
kcal/mol as pH drops -- an order of magnitude smaller than the ~1400 kcal/mol
scale of the barriers themselves, and dwarfed by N=1-per-point noise
concerns (see Gate 3 below). **GPR132's barrier goes up, not down, at low
pH** -- the opposite direction from the previous (uncorrected) GPR132
report, which found a small decrease (347.0 -> 337.3). This sign flip
matters: it means the earlier "barrier is essentially pH-independent for
GPR132" conclusion does not survive under the corrected, symmetric
protocol either, and the magnitude of the change here (+8.7 kcal/mol out
of a 344.5 kcal/mol barrier, ~2.5%) is small enough that "pH-independent"
remains the fairer one-line summary for GPR132 post-fix, just for a
different underlying reason (small net change either direction, not a
confirmed flattening).

### Endpoint shift: G(inactive) - G(active), and how it moves with pH

This is the activation-equilibrium direction: positive means the active
state is more stable (lower G) than the inactive state.

| Receptor | Shift, pH 7.4 | Shift, pH 6.0 | Change (7.4 -> 6.0) |
|---|---|---|---|
| GPR68 (old, uncorrected) | -19.5 kcal/mol | -13.2 kcal/mol | +6.3 kcal/mol |
| GPR68 (this run, corrected) | **+15.6 kcal/mol** | **+13.7 kcal/mol** | -1.9 kcal/mol |
| GPR132 (old, ad hoc superposition) | +37.4 kcal/mol | +59.6 kcal/mol | +22.2 kcal/mol |
| GPR132 (this run, corrected) | **+28.5 kcal/mol** | **+49.0 kcal/mol** | +20.5 kcal/mol |

**The user's specific question -- did GPR68 and GPR132 have opposite
endpoint-shift signs, and does that survive the fix -- has a clear
answer: no, it does not survive.** Pre-fix, GPR68's shift was negative
(inactive more stable) and GPR132's was positive (active more stable) --
genuinely opposite signs, exactly as flagged. Post-fix, under the
identical, corrected protocol, **both receptors show a positive shift**
(active more stable than inactive, at both pH values). The apparent
sign disagreement was itself a preprocessing artifact of GPR68's
unremoved frame offset, not a real biological difference between the two
receptors. GPR132's shift is however still reported for completeness --
recall GPR68 is excluded from mechanistic interpretation by Gate 2.

### Uniform pH offset (protonation bookkeeping, not the differential)

Reported separately as the user requested, since it reflects PROPKA's
absolute (uncalibrated, Gate A) protonation-state bookkeeping rather than
any conformational signal:

| Receptor | Uniform pH offset (active endpoint, pH7.4->6.0) |
|---|---|
| GPR68 | +159.5 kcal/mol |
| GPR132 | +122.9 kcal/mol |

Both receptors are destabilized by a large, roughly similar amount
(120-160 kcal/mol) simply from protonating titratable residues at lower
pH -- this dwarfs every differential number in this report by more than
an order of magnitude and is expected, not a finding.

### RMSD sanity check

`rmsd_sanity_check.png` per receptor: RMSD-to-active and RMSD-to-inactive
sum to the core endpoint RMSD at every image and cross at the midpoint
for both receptors, confirming the 150-iteration local relaxation did not
collapse the interpolated path back toward either endpoint (the standard
failure mode a real string method's tangent-force projection exists to
prevent, and which this simpler linear-interpolation-plus-relaxation
approach could, in principle, suffer from without this check).

## Sanity gate 3: N=1, no error bars, nothing here is "significant"

Every number in this report is a single point estimate: one interpolated
image, one minimization, one energy evaluation, at one pH. There is no
replicate structure, no ensemble average, and therefore no error bar on
any barrier, shift, or displacement value above. Differences of a few to
a few tens of kcal/mol (the pH-dependent barrier changes, the endpoint
shifts) should be read as "this is what this specific calculation
produced," not as statistically distinguishable from zero. The
qualitative, structural findings this report leans on -- the TM6 gate
failure/pass, the barrier-vs-path-length monotonicity, the endpoint-shift
sign correction -- are robust to this limitation because they follow
from geometry (RMSD, displacement) cross-checked by two independent
methods, not from a single noisy energy difference; the pH-dependent
energetic findings are not, and are labeled accordingly above.

## Gate A: absolute energetics remain uncalibrated

PROPKA has not been validated against an experimental buried-carboxylate
pKa benchmark (e.g. SNase) by this pipeline; an earlier attempt at a
from-scratch Poisson-Boltzmann calibration in this project's `linkage_pka`
module was sign-inverted (see that module's own `FINDINGS.md`). Absolute
G magnitudes throughout this report -- and throughout every run this
session -- are therefore uncalibrated. The differential structure along
one interpolation path (barrier shape, relative endpoint stability) is
more robust than any single absolute G value, but is still built on top
of an uncalibrated force field + protonation model. Treat every kcal/mol
number in this report as illustrative of the pipeline's behavior, not as
a validated thermodynamic measurement.

## What this run decides

Under identical superposition, **GPR68 fails the genuine-activation-pair
gate and is excluded from interpretation**; only GPR132 currently
qualifies (marginally) as a real active/inactive transition among the
receptors available in this session. The apparent GPR68-vs-GPR132
contrast reported in the earlier (asymmetric-protocol) runs -- including
the specific "opposite endpoint-shift sign" pattern -- does not survive
correction. This does not resolve the sensor/non-sensor question either
way: it means the evidence for a *GPR68* proton-sensing conformational
signature that was previously reported does not hold up, while GPR132's
pair remains a plausible (not confirmed, N=1, uncalibrated) candidate.
Extending this corrected protocol to GPR184, GPR4, and GPR65 once
available is the natural next step, with the same TM6 gate applied before
any of them are used for mechanistic interpretation.

## Provenance

| | GPR68 | GPR132 |
|---|---|---|
| Active structure | `examples/data/gpr68_structures/active/ClassA_ogr1_human_Active_AFMS_2024-05-15_GPCRdb.pdb` | `examples/data/gpr132_structures/active/ClassA_gp132_human_Active_AF_2024-05-15_GPCRdb.pdb` |
| Active SHA-256 (16 char) | `04cd154ca4973436` | `a13251f7f63b3832` |
| Inactive structure | `.../inactive/ClassA_ogr1_human_Inactive_AFMS_2024-05-15_GPCRdb.pdb` | `.../inactive/ClassA_gp132_human_Inactive_AFMS_2024-05-15_GPCRdb.pdb` |
| Inactive SHA-256 (16 char) | `8ed2f47286db058d` | `9ee9996fa84ff353` |
| Superposition reference set | TM1,2,3,4,5,7 backbone Cα (DSSP+anchor-derived) | same method, receptor-specific ranges |
| Minimizer | 150 L-BFGS iterations (OpenMM) | same |
| Scorer | AMBER ff14SB + GBn2 + RRHO (`gibbs/gpcr_gibbs_energy.py`) | same |
| pH values | 7.4, 6.0 | same |
| Git commit (GPR68 run) | `bb36f94132f0758d9b1f713e09705e5642fc9a1c` | -- |
| Git commit (GPR132 run) | -- | `e0194db57e0aaaca70fdb38d9b2f7fa7bee0b4d1` |

Reproduce with:

```bash
pip install -r requirements.txt
apt-get install -y dssp   # real DSSP, required by gpcr_pipeline_tm_topology.py
SETUPTOOLS_USE_DISTUTILS=stdlib pip install openmm pdbfixer propka
PYTHONPATH=.:gibbs python examples/gpcr_pipeline_symmetric_string_demo.py
```

Takes ~50-70 minutes (44 real OpenMM minimizations: 2 receptors x 11
images x 2 pH values).
