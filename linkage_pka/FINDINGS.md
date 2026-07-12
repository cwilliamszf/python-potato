# GPR68 smoke test: findings on conformational sampling and PB pKa accuracy

Status: exploratory pipeline-validation work, not a validated result. **No
number in this document should be read as a real prediction about GPR68's
proton-sensing behavior.** Gate A (SNase buried-ionizable calibration) has
now been run against the real experimental dataset and **fails**, and the
failure is now understood precisely. The original runs failed by a wide
margin (RMSE 7.9-11.0 pKa units against a 1.0-unit threshold) largely
because of a single undocumented error -- the protein interior dielectric
was hardcoded to `pdie=2.0`, indefensibly low for rigid single-structure
PB; fixing it to a literature-standard value drops aggregate RMSE to
~1.4. But at any benchmark-independent dielectric the gate still fails,
and metric decomposition shows why: the pipeline's computed pKa's for
SNase's buried carboxylates are *anti-correlated* with experiment
(negative Pearson/Spearman within the Asp/Glu set), so a low aggregate
RMSE at very high pdie is a range-compression artifact, not real
predictive skill (see "Gate A, path (a) executed" below). Per the
pipeline spec's own acceptance criterion, this means **no PB pKa produced
by this pipeline, on GPR68 or anywhere else, should be treated as
quantitatively calibrated** — only as pipeline-mechanics validation. This
document exists to record what was learned about the *pipeline's own
behavior* at real-protein scale, since that surfaced a genuine,
previously-undetected methodological gap, and now a genuine, decisive,
and specifically-diagnosed calibration failure.

## Context

The proton-linkage pipeline (`linkage_pka/`) had been built and unit-tested
against a small, simple validation protein (CI2, 65 residues) but never
run end-to-end on a real target GPCR. This was the first such run: the
real active/inactive GPR68 structures (GPCRdb, WT, already hydrogenated),
taken through `structure_prep` → `membrane_frame` → PDB2PQR →
per-site/per-cluster titration → `multisite` solver.

## The pipeline ran end to end

`run_structure_prep` (protonation, rotamer optimization, restrained
minimization) completed on both the active and inactive structures
(~211-213 s each, 68 ionizable residues optimized, 2 and 9 CA-displacement-
flagged residues respectively). `compute_membrane_frame` correctly located
R3.50 (resnum 119, DRY motif) and Y7.53 (resnum 286, DPVLY motif) on both,
falling back to the secondary-structure TM mask as expected (this repo's
GPCRdb B-factor column is not real pLDDT — confirmed by range check,
values include negatives). This is the first time every module built this
session had been exercised together on real, non-toy data.

## The anomaly

A 4-residue cluster was selected for the smoke test: Glu164, Asp165,
Glu166, His169 — extracellular of the membrane slab, all within ~4-12 Å of
each other by real CA-CA distance on both conformers (a genuine tight
cluster, not cherry-picked for the finding). Running the existing
intrinsic-pKa + pairwise-coupling pipeline on this cluster (30 Å residue-
complete truncation around the cluster centroid, dime=33³/glen=65 Å APBS
grid, pdie=2/sdie=78.54/0.15 M ionic strength) gave:

| site | active pKa | inactive pKa | model pKa |
|---|---|---|---|
| Glu164 | **-16.4** | -3.3 | 4.1 |
| Asp165 | 1.5 | -0.6 | 3.9 |
| Glu166 | -0.7 | -3.8 | 4.1 |
| His169 | 1.5 | **-15.5** | 6.0 |

The bolded values are physically impossible: no buried-ionizable pKa shift
in the literature exceeds ~5 units (the most extreme published case,
engineered cavity Lys in staphylococcal nuclease, Isom et al. PNAS 2011).
Shifts here exceed 20 units.

## Diagnostic chain (each step ruled out one hypothesis)

1. **Truncation radius** (30 Å → 40 Å, ~90 → 168-249 residues): barely
   moved the anomaly (His169 inactive: -15.5 → -15.0). In hindsight this
   doesn't actually distinguish "real physics" from "a bug" either way —
   electrostatics is Debye-screened to ~8 Å at this ionic strength, so
   truncation-radius-independence is expected regardless of the true
   cause. Ruled out truncation as the *dominant* driver, nothing more.

2. **Reduced-site (Bashford-Karplus) approximation vs. exact joint
   enumeration**: `compute_intrinsic_pka` freezes every *other* titratable
   site at a fixed reference state while computing one site's own pKa —
   a fine approximation for isolated sites, suspect for a tight cluster.
   Built `titration.compute_cluster_joint_energies` +
   `multisite.solve_cluster_titration_exact` to compute all 2⁴=16 joint
   microstate energies directly, with no per-site decomposition. Result:
   **the anomaly persisted at essentially the same magnitude**
   (His169 inactive single-site energy gap: 93.6 kJ/mol reduced-site vs.
   93.6 kJ/mol exact-joint) — ruling out the reduced-site approximation as
   the cause. This solver is real, tested, permanent infrastructure
   regardless (it's also cheaper for small clusters, since it subsumes
   the separate pairwise-coupling step), but it didn't fix this case.

3. **Grid convergence** (dime 33→65→97, same 65 Å box, His169 inactive):
   -15.52 → -11.76 → -12.16. The 33→65 step moved it 3.76 units; the
   65→97 step moved it only 0.40 units — classic diminishing-returns
   convergence, not divergence. **The calculation is numerically sound
   and converges cleanly to ≈-12** — not a grid bug. The problem is that
   the converged answer is still ~18 units from anything documented.

4. **Per-microstate rotamer relaxation (Coulomb-only)**: real PB-pKa
   methods (e.g. MCCE) address exactly this class of failure — a single
   rigid structure is a worse approximation exactly when nearby charges
   are close enough to matter, which is when different protonation states
   most favor different side-chain packing. Built
   `titration.optimize_rotamer_for_microstate`: re-selects one residue's
   chi1/chi2 rotamer for its *current* charge state (staggered
   gauche-/gauche+/trans candidates, reusing `structure_prep.py`'s
   geometry, scored by a new cheap pairwise-Coulomb proxy since this
   module has no OpenMM context). Wired as an opt-in `optimize_rotamer`
   flag through every PB energy function (default False; per the
   pipeline spec's own guardrail, "Report ... with-rotamer-relaxation and
   without, every time" — a comparable sensitivity axis, not a silent
   behavior change).

   Result, rerun on the same cluster: **His169's active-conformer pKa was
   completely fixed** — single-site gap dropped from 93.6 to -2.5 kJ/mol,
   yielding a physically normal sigmoidal titration curve (apparent pKa ≈
   6.7, close to the model value of 6.0). **The inactive conformer was
   essentially unchanged** (93.6 → 91.1 kJ/mol; θ(pH) still ≡ 0 across
   pH 5-8). The three carboxylates (164/165/166) stayed fully deprotonated
   in every variant on both conformers — at least internally consistent.

5. **Added steric repulsion**: pure Coulomb scoring has no way to penalize
   two atoms physically overlapping, and can even *reward* close approach
   between opposite charges. Built `titration._pairwise_repulsion_energy`
   (soft-sphere, `(sigma_ij/dist)¹²` with `sigma_ij` = sum of AMBER
   radii, repulsive-only by design) and combined it with the Coulomb term
   in `optimize_rotamer_for_microstate`'s scoring.

   Result: **active conformer's His169 was fixed again, but landed on a
   different answer** (single-site gap -10.0 kJ/mol; apparent pKa now
   <5.0 instead of ≈6.7 — the single-residue rotamer search is finding
   different local minima depending on the scoring function, not
   converging to one clear answer). **The inactive conformer remained
   unchanged** (95.2 kJ/mol, if anything marginally worse than
   Coulomb-only).

6. **Multi-residue relaxation (real geometric neighborhood, 8 Å, titratable
   or not)**: extended `CHI_ATOMS` to `EXTRA_CHI_ATOMS`, covering all 12
   remaining standard rotatable side chains (verified against PDB2PQR's
   own AMBER.DAT, not assumed), kept strictly separate from
   `IONIZABLE_RESNAMES` so non-titratable residues can't leak into
   structure_prep's protonation-relevant residue selection. Built
   `titration.find_relaxation_neighbors` +
   `titration.optimize_rotamers_with_neighbors`: relaxes the target site,
   then every real geometric neighbor within radius, titratable or not.

   Result: **active conformer's His169 swing dropped to +5.2 kJ/mol** —
   small, near-neutral, physically unremarkable. **The inactive conformer
   remained unchanged again** (93.9 kJ/mol — statistically indistinguishable
   from every earlier variant: 93.6 with no relaxation, 91.1 Coulomb-only,
   95.2 Coulomb+repulsion single-residue). An 8 Å real geometric
   neighborhood, covering every standard rotatable side chain and not just
   the titratable cluster, still did not move it.

## Current conclusion

Two things are now well-supported by direct evidence across six
independent variants, not guesswork:

- The active-conformer anomaly is a real, local rotamer-packing artifact,
  resolvable by per-microstate conformational relaxation — but the
  *specific* relaxed geometry (and hence the resulting pKa) is sensitive
  to which scoring function/neighborhood is used (apparent pKa moved
  ≈6.7 → <5.0 → ≈unremarkable across three relaxation variants that all
  "fixed" it). This sensitivity should itself be reported as an
  uncertainty band, not resolved by picking one variant and reporting its
  answer as ground truth.
- The inactive-conformer anomaly is **not resolvable by any local
  chi-angle relaxation tried** — target-only, Coulomb-only, Coulomb+
  repulsion, or a full real 8 Å neighborhood of every rotatable residue
  type, titratable or not. All six variants land within ~4 kJ/mol of each
  other (91-96 kJ/mol), a genuinely stable, unmoved number. This rules
  out "wrong scoring function" and "neighborhood too narrow" as the
  explanation. What's left, in decreasing order of plausibility:
  (a) the real constraint is a **backbone** degree of freedom, not a
  side-chain rotamer — outside what any chi-angle search can reach;
  (b) a genuinely large relaxation radius (>8 Å, e.g. transmitted through
  a longer packing network) is needed; (c) this is a real, if extreme,
  structural feature of this specific inactive GPCRdb model. None of
  these are quick fixes, and distinguishing them meaningfully requires
  either MD-scale sampling (outside this pipeline's no-MD design) or Gate
  A calibration data to know whether the distinction even matters for the
  pipeline's actual accuracy. **This thread is a reasonable place to
  stop** rather than continue iterating on local rotamer search.

## Gate A dataset sourcing (resolved via user upload)

Before the GPR68 smoke test, real effort went into sourcing the SNase
buried-ionizable experimental dataset for Gate A calibration. WebFetch and
direct network access are blocked for every external host tried in this
sandbox (PubMed, PNAS, JHU, Clemson — confirmed via repeated 403s, not
assumed). WebSearch (a separate, allowed tool) confirmed the real citable
sources exist — Isom et al. PNAS 2011 (10.1073/pnas.1010750108), Harms et
al. J Mol Biol 2009, Castañeda et al. Proteins 2009, and the PKAD-R
curated database (Clemson/JHU) — but their actual numeric tables could not
be fetched. `pip download`-ing PyPKa and PROPKA (candidates for bundled
benchmark data) confirmed neither ships one; both only include force-field
parameter tables. This was resolved by the user directly downloading the
PKAD-2 wild-type dataset (compbio.clemson.edu/PKAD-2/) and the real 1STN
mmCIF structure and uploading both into this session — see
`linkage_pka/gate_a.py` for the transcribed dataset (24 experimental
pKa's, provenance and citations in its module docstring) and "Gate A
calibration: FAIL" below for the result.

## Gate A calibration: FAIL

Ran the real Gate A test: `titration.compute_intrinsic_pka` (soluble
protein, `frame=None`, no membrane) on the real 1STN structure for every
experimental site the structure actually resolves (21 of 24 — ASP143,
ASP146, GLU142 fall outside the crystal's resolved range of resnum 6-141,
most likely disordered C-terminal tail, excluded rather than guessed at),
then `gate_a.compute_gate_a_rmse` against the real PKAD-2/Castañeda 2009
values. Same pipeline conventions as the GPR68 work throughout:
`structure_prep.run_structure_prep` (PDBFixer + rotamer optimization) →
`pdb2pqr30 --ff AMBER --with-ph 7.0 --titration-state-method propka` →
`GridParams(dime=(33,33,33), glen=(65,65,65), ...)` for the protein side,
`glen=(25,25,25)` for the model compound. Four variants run, matching
the relaxation levers built for the GPR68 work plus the new MCCE-style
ensemble (`titration.select_rotamer_ensemble` +
`compute_environment_energies_ensemble`, `ensemble_size=4`: top-4
classically-ranked rotamers per microstate, Boltzmann-averaged over their
*real* PB energies via log-sum-exp rather than picking a single winner):

| Variant | RMSE (pKa units) | MAE | n compared | Pass (<1.0)? |
|---|---|---|---|---|
| Rigid (no relaxation) | 10.99 | 8.83 | 17 | **No** |
| Single-residue rotamer relaxation | 10.09 | 7.60 | 17 | **No** |
| Neighbor relaxation (8 Å radius) | 7.86 | 6.12 | 17 | **No** |
| MCCE-style ensemble (K=4, target site only) | 11.32 | 8.77 | 17 | **No** |

(`n=17`: excludes the 2 biphasic entries — Asp19, Asp21, no principled
single-value comparison — and the 2 upper-bound entries — Asp77, Asp83,
`<2.2` — per `compute_gate_a_rmse`'s documented defaults;
`include_upper_bounds=True` makes every variant's RMSE worse, since both
excluded sites come back in at +18 to +24 computed against a <2.2 bound.)

Per-residue detail (`expt` = real experimental pKa):

| Site | expt | rigid | rotamer | neighbor | ensemble(K=4) |
|---|---|---|---|---|---|
| His8 | 6.52 | 1.63 | 3.26 | 10.02 | 4.59 |
| Glu10 | 2.82 | 20.45 | 23.02 | 18.93 | 11.53 |
| Asp40 | 3.87 | 3.80 | -3.72 | -3.68 | -6.34 |
| Glu43 | 4.32 | -16.14 | -15.25 | -5.55 | -16.49 |
| His46 | 5.86 | -1.34 | -0.26 | 12.98 | -9.10 |
| Glu52 | 3.93 | 15.28 | 9.43 | 8.94 | 8.20 |
| Glu57 | 3.49 | 6.73 | 2.02 | 2.12 | 1.90 |
| Glu67 | 3.76 | 19.73 | 3.42 | 0.81 | 5.75 |
| Glu73 | 3.31 | 9.35 | 9.14 | 7.54 | 15.09 |
| Glu75 | 3.26 | 21.11 | 24.10 | 22.27 | 20.95 |
| Asp95 | 2.16 | 17.11 | 10.70 | 9.90 | 21.73 |
| Glu101 | 3.81 | 10.35 | 12.03 | 10.94 | 5.14 |
| His121 | 5.30 | 0.66 | 4.76 | 7.21 | 1.58 |
| Glu122 | 3.89 | 7.62 | 5.41 | 9.06 | 7.24 |
| His124 | 5.73 | 4.49 | -0.37 | 1.58 | 3.69 |
| Glu129 | 3.75 | 17.66 | 16.03 | 2.99 | 24.41 |
| Glu135 | 3.76 | 4.07 | 5.00 | 3.23 | -0.69 |

Key observations:

1. **Relaxation helps on average but the effect is not monotonic or
   reliable per-site.** Going rigid → rotamer → neighbor lowers RMSE
   (11.0 → 10.1 → 7.9), consistent with the GPR68 diagnosis that rigid
   single-structure PB exaggerates buried-charge electrostatics. But
   individual sites move the *wrong* direction under relaxation just as
   often as the right one: Asp40 was nearly perfect unrelaxed (diff
   -0.07) and got *worse* under every relaxed variant, monotonically so
   (-7.59 → -7.55 → -10.21 rigid→rotamer→neighbor→ensemble); His46 and
   His8 get progressively *worse* going from rigid to neighbor relaxation
   (His46: -7.20 → -6.12 → +7.12 magnitude).
2. **No clean burial (%SASA) correlation.** Asp40 (71% exposed) is
   nearly exact; Glu67 (76% exposed, comparably solvent-exposed) is off
   by 16 units unrelaxed. Deeply buried and moderately exposed sites both
   appear among the best- and worst-performing residues. This rules out
   "just a burial/dielectric-boundary problem" as the sole explanation —
   something more specific to each local geometry (rotamer packing,
   nearby H-bond partners, possibly propka's starting protonation-state
   assignment) is driving the site-to-site variance.
3. **The MCCE-style ensemble (K=4) is a genuine negative result — it did
   not improve on the best existing variant, and is in fact the worst of
   the four (RMSE 11.32, edging out even the rigid baseline's 10.99).**
   This is the honest, unexpected outcome of actually testing the
   approach the earlier "next step" writeup (below) predicted would help,
   not a result to explain away. Two real, non-mutually-exclusive reasons
   this pipeline's implementation likely under-delivers relative to the
   literature's MCCE-style methods:
   - **The candidate pool comes from the same crude classical (Coulomb +
     soft-repulsion) prescreen used for single-rotamer selection**
     (`select_rotamer_ensemble` calls the same
     `_enumerate_rotamer_candidates` core as
     `optimize_rotamer_for_microstate`, just keeps the top 4 instead of
     the top 1). If that classical score ranks candidates in a way that
     correlates poorly with their *actual* PB energy (plausible — it has
     no desolvation/dielectric-boundary term at all, only vacuum-like
     Coulomb + steric repulsion), the ensemble's log-sum-exp average is
     built from 4 samples of a possibly mis-ranked distribution rather
     than a representative one — worse, this can happen *asymmetrically*
     between the deprotonated and protonated microstates (different
     charge distributions favor different candidates under the same
     classical proxy), directly corrupting the dG_ion = E_deprot - E_prot
     difference the whole calculation hinges on. A real fix would need
     either a better/PB-informed prescreen or a larger K approaching
     exhaustive enumeration (all 9 chi1/chi2 combinations) to reduce this
     sampling bias.
   - **This ensemble only varies the target residue's own rotamer** —
     unlike the neighbor-relaxation variant (RMSE 7.86, still the best of
     the four), which also relaxes real geometric neighbors within 8 Å.
     The neighbor variant's advantage most plausibly comes from resolving
     steric/electrostatic clashes involving *other* side chains, a
     completely different degree of freedom the target-only ensemble
     never touches — so the two levers are not measuring the same thing,
     and stacking them (ensemble target + relaxed neighbors) is an
     obvious next experiment this run does not answer.
4. **Even the best variant (neighbor relaxation, RMSE 7.86) is ~8x over
   the 1.0-unit threshold.** This is not a borderline result nudged over
   the line by one or two outliers — nearly every buried Glu is off by
   6-20 units in every variant.
5. **This matches, and now sharply confirms, the literature's own
   framing of SNase as a hard benchmark** — but the naive top-K ensemble
   tried here is not, by itself, the fix. Castañeda et al. 2009 (the
   source of this data) and the broader Garcia-Moreno lab literature
   built this dataset specifically because naive single-structure
   continuum electrostatics is known to struggle on SNase's buried
   ionizable cluster; real published methods that do well on this
   benchmark generally use multi-conformer continuum electrostatics
   (MCCE-style) with either much larger, PB-validated rotamer ensembles
   or self-consistent multi-site sampling, not a 4-candidate,
   classically-prescreened, single-residue ensemble on top of an
   otherwise-unchanged pipeline. The negative result above is evidence
   *for* that gap, not against the general MCCE approach.

**Conclusion:** Gate A fails in every variant tried, including the new
MCCE-style ensemble, which performed worse than the simpler neighbor-
relaxation variant. Per the pipeline spec's own acceptance rule ("no
ancestral-node number may be reported" before Gate A passes), **no
absolute pKa or Δn_H(pH) number this pipeline has produced — for SNase or
for GPR68 — should be treated as quantitatively calibrated.** The GPR68
results earlier in this document (ECL2 cluster, D2.50/Asp282/Glu103
cluster, Na+ ion effect) remain useful as *pipeline-mechanics* validation
(the code runs correctly, responds to physically sensible perturbations
in the right direction, e.g. Na+ raising a nearby Asp's pKa) but not as
validated predictions of GPR68's real proton-sensing thermodynamics. A
multi-conformer/MCCE-style ensemble extension has now been built
(`select_rotamer_ensemble`, `compute_environment_energies_ensemble`) and
run against Gate A -- it did not close the gap at K=4, target-site-only
(see the negative result above). Closing this gap further most plausibly
needs either a larger/PB-validated candidate pool (not just a bigger
classically-prescreened K) or combining the ensemble with neighbor
relaxation (the two levers address different degrees of freedom and have
not yet been tried together) -- both are concrete, in-scope follow-ups,
not parameter tweaks to what exists today.

## Code artifacts produced (all tested, all in `linkage_pka/`)

- `titration.compute_cluster_joint_energies` — exact whole-cluster joint
  microstate PB energies (§2 above).
- `multisite.solve_cluster_titration_exact` — consumes those energies
  into θ(pH)/ln(Z) without the pairwise-additive assumption; proven
  equivalent to `solve_cluster_titration` when the input happens to be
  pairwise-additive (`tests/test_multisite.py`).
- `titration.optimize_rotamer_for_microstate` — per-microstate chi1/chi2
  relaxation (§4 above), `optimize_rotamer` flag threaded through
  `compute_environment_energies`, `compute_intrinsic_pka`,
  `compute_pairwise_coupling`, `compute_cluster_joint_energies`.
- `titration._pairwise_coulomb_energy`, `titration._pairwise_repulsion_energy`
  — the two scoring terms (§4-5 above).
- `structure_prep.EXTRA_CHI_ATOMS`, `titration.ALL_CHI_ATOMS`,
  `titration.find_relaxation_neighbors`,
  `titration.optimize_rotamers_with_neighbors` — multi-residue real-
  neighborhood relaxation (§6 above), `neighbor_radius_ang` parameter
  threaded through the same four PB energy functions.
- `membrane_frame._segment_tm_helices`, `membrane_frame.find_d250` —
  geometric (not sequence-motif) location of D2.50, the conserved
  sodium-pocket Asp: segments the TM mask into real helices (length +
  fraction-in-slab filtering, verified to recover exactly the canonical 7
  on real GPR68 data), anchors TM2 as the helix immediately N-terminal to
  the one containing R3.50, then picks the Asp/Glu closest to the
  membrane's geometric center. Confirmed on both real GPR68 conformers:
  Asp67, decisively separated from the runner-up (0.3 A vs 15.3 A from
  center).
- `titration.load_na_ion_parameters`, `titration.place_na_ion`,
  `titration.build_na_ion_atom` — explicit Na+ ion modeling (pipeline
  spec step 4): real ion charge/radius parsed from OpenMM's bundled
  amber14/tip3p.xml (PDB2PQR's own AMBER.DAT has no ion entries at all),
  positioned via the standard LJ combining-rule contact distance from
  D2.50's carboxylate oxygens. No changes needed to the PB energy
  functions themselves -- the ion is just another atom in `protein_atoms`
  from their perspective; "with/without" is running the same calculation
  with and without one atom appended. **Validated on the real GPR68
  inactive structure** (Asp67, local 30 A neighborhood): intrinsic pKa
  5.98 without the ion -> 9.35 with it, a +3.4 unit increase -- the
  physically correct direction (a nearby +1 charge stabilizes the
  protonated/neutral carboxylate over the deprotonated/anionic form) and
  a sensible magnitude for direct ion contact, unlike the earlier ECL2
  cluster's implausible >20-unit shifts. The clearest sign yet that this
  pipeline's core PB machinery behaves correctly when the local
  environment isn't a tightly-packed multi-charge cluster.
- `gate_a.py` — Gate A calibration scaffolding: `SNASE_1STN_EXPERIMENTAL_PKA`
  (24 real PKAD-2/Castañeda 2009 entries, provenance in the module
  docstring), `compute_gate_a_rmse` (biphasic/upper-bound handling,
  per-residue breakdown, `skipped` list with reasons). Run against the
  real 1STN structure — see "Gate A calibration: FAIL" above.
- `titration.select_rotamer_ensemble`, `titration._enumerate_rotamer_candidates`
  (shared refactor with `optimize_rotamer_for_microstate`, no behavior
  change to the existing single-best-candidate path — verified by the
  existing rotamer test suite passing unmodified),
  `titration.compute_environment_energies_ensemble` — MCCE-style top-K
  Boltzmann-averaged rotamer ensemble (log-sum-exp over real PB energies,
  not the classical proxy used only to prune the candidate pool), threaded
  through `compute_intrinsic_pka` via `ensemble_size` (mutually exclusive
  with `optimize_rotamer`/`neighbor_radius_ang`). Run against Gate A at
  K=4: did not improve on the existing best variant — see "Gate A
  calibration: FAIL" above for the full negative-result writeup.

Full test suite: 203 passed as of this writing (`pytest` from the repo
root).

## Na+ ion comparison across the full pH grid (superseded -- see below)

*(Initial pass, kept for the record of what changed and why.)* Extended
the single-pH validation above to both conformers and the full pH 5-8
grid using `linkage.compute_linkage`/`protonation_fraction` on the four
intrinsic pKa's already obtained, treating D2.50 as an **isolated**
titratable site:

| structure | without ion | with ion |
|---|---|---|
| active   | 9.509  | 15.392 |
| inactive | 5.977  | 9.346  |

This gave Delta_n_H(pH) peaking near +0.96 (no ion) collapsing to
near-zero (with ion) -- i.e. "the ion switches off D2.50's contribution
to the proton-linkage signal." **This conclusion does not survive
checking the coupling it was flagged as skipping** (see next section) --
the real pairwise couplings to Asp282/Glu103 turned out to be 13-27
kJ/mol (5-11 kT), 5-11x the multisite solver's own clustering threshold,
so isolated-site treatment was never a valid approximation here. Left in
this document as the concrete illustration of why the caveat mattered.

## Coupled 3-site cluster (D2.50 + Asp282 + Glu103): the corrected result

Checked the coupling the isolated-site treatment skipped:
`compute_pairwise_coupling` between D2.50 (Asp67) and its two titratable
neighbors within ~9-11 A, on both conformers:

| pair | active W_ij | inactive W_ij | CA distance |
|---|---|---|---|
| D2.50-Asp282 | -27.0 kJ/mol | -19.2 kJ/mol | ~9.0 A |
| D2.50-Glu103 | -13.7 kJ/mol | -13.0 kJ/mol | ~9.3-10.8 A |

All four couplings are 5-11x `multisite.DEFAULT_COUPLING_THRESHOLD_KJ_MOL`
(2.5 kJ/mol ~ 1 kT) -- real, substantial, not negligible. Negative sign
makes physical sense: when one carboxylate protonates, it relieves
electrostatic repulsion for its neighbors, making their protonation
easier too (cooperative protonation among nearby acidic groups).

Redid the calculation properly: full 2^3=8 joint-microstate cluster
(`compute_cluster_joint_energies` + `solve_cluster_titration_exact`),
with and without the ion, on both conformers. Result is qualitatively
different from the isolated-site version, not just quantitatively:

- **D2.50 itself**: coupling pulls its apparent pKa down substantially
  from the isolated-site estimate (crosses theta=0.5 around pH 6.0-6.5,
  not staying protonated to pH 8) -- the negative coupling terms mean
  nearby deprotonation makes D2.50's own deprotonation easier, the
  opposite pull from what the isolated-site number implied.
- **Glu103 is a striking, genuine conformer-differentiator**: titrates
  normally in the active conformer (theta 1.0->0.0 across pH 5-8, apparent
  pKa~6.4) but is **completely, uniformly deprotonated (theta=0.0) across
  the entire pH 5-8 range in the inactive conformer, with or without the
  ion**. That is a conformational difference, not a pH effect.
- **With the ion**, D2.50 and Asp282 both get pinned to theta~1.0 in
  *both* conformers identically (matching the isolated-site finding for
  D2.50 alone) -- but this means **Glu103 becomes essentially the entire
  Delta_n_H(pH) signal for the cluster once the ion is bound** (its
  per-residue contribution at pH 7.0 is +0.9999 out of a cluster total of
  +1.004, with D2.50 and Asp282 contributing ~0).

**Revised mechanistic picture**: the ion does not silence the cluster's
proton-linkage signal (the isolated-site conclusion above). It silences
D2.50 and Asp282 specifically, while the conformational proton-sensing
signal routes entirely through Glu103. Delta_n_H(pH) summed over the
cluster:

| pH  | without ion | with ion |
|---|---|---|
| 5.0 | +1.127 | +1.000 |
| 6.0 | +1.975 | +1.000 |
| 6.5 | +0.568 | +1.001 |
| 7.0 | -0.026 | +1.004 |
| 7.5 | -0.016 | +1.014 |
| 8.0 | -0.005 | +1.043 |

Without the ion, the total signal is non-monotonic -- large and positive
at low pH (dominated by all three sites protonating together), crossing
through ~0 and going slightly *negative* around neutral-to-basic pH. With
the ion, it locks to a clean, nearly pH-independent +1.0 (essentially
"Glu103 alone, fully switched between conformers"). Both remain
unvalidated pipeline output (no Gate A calibration), and (as with the
ECL2 cluster) only one grid/rotamer-relaxation variant was run here -- no
convergence or relaxation-sensitivity check has been done for this
cluster specifically.

## Open questions / next steps

1. **Local rotamer relaxation is exhausted as a lever for the inactive
   conformer's anomaly** (see "Current conclusion" above) — six variants,
   all landing within ~4 kJ/mol of each other. Further progress here
   needs a categorically different approach (backbone sampling, or MD-
   scale conformational search), not another scoring-function tweak.
   Recommend treating this as a documented, stable finding rather than
   continuing to iterate.
2. Report the relaxation-variant sensitivity as an explicit band (matching
   `linkage.sensitivity_band`'s existing convention) for any site where
   different relaxation variants disagree, rather than picking one and
   reporting it as ground truth (the active conformer's His169 is a
   concrete example: three variants gave three different apparent pKa's,
   all "fixed" relative to the unrelaxed case).
3. ~~Gate A calibration remains blocked on dataset access~~ — resolved
   (user supplied PKAD-2 CSV + real 1STN structure) and **run: FAILS**
   in every variant tried (RMSE 7.9-11.0 vs 1.0-unit threshold). See
   "Gate A calibration: FAIL" above. This is now the dominant open issue:
   every GPR68 number in this document is pipeline-mechanics validation
   only, not a calibrated prediction, until this gap is closed.
4. ~~Explicit Na⁺ ion modeling at D2.50~~ — done. ~~pH-grid with/without
   comparison~~ — done, then corrected: the initial isolated-site version
   was superseded after checking (and confirming significant) coupling to
   Asp282/Glu103 (see "Coupled 3-site cluster" above).
5. The corrected 3-site cluster result has not itself been through a
   grid-convergence or rotamer-relaxation-sensitivity check (unlike the
   ECL2 cluster, which got both) — Glu103's striking active/inactive
   difference (θ=0 across the *entire* pH range in inactive) is exactly
   the kind of large, clean-looking result that should be stress-tested
   the same way before treating it as more than a first-pass hypothesis.
6. Neither the ECL2 cluster nor the D2.50/Asp282/Glu103 cluster has been
   checked for coupling *to each other* or to any other titratable site
   beyond the ~9-12 Å searched so far — a genuinely complete treatment
   would need a wider coupling search across the whole receptor, which is
   real-production-run territory, not a smoke test.
7. ~~Given the Gate A failure, the highest-value next step is almost
   certainly closing the calibration gap~~ — built and run
   (`select_rotamer_ensemble`, `compute_environment_energies_ensemble`,
   MCCE-style top-K Boltzmann-averaged ensemble): **did not close the gap
   at K=4, target-site-only** (RMSE 11.32, worse than every other
   variant — see "Gate A calibration: FAIL" above for the full negative-
   result writeup and the two likely reasons: a classically-prescreened
   candidate pool that may rank poorly against real PB energy, and no
   overlap with the neighbor-relaxation lever that gave the best result
   so far). Concrete untried follow-ups: (a) combine the ensemble with
   neighbor relaxation in one run: Boltzmann-average the target site
   while relaxing real geometric neighbors, rather than either lever
   alone; (b) increase K toward the full 9-candidate chi1/chi2 space to
   test whether the negative result at K=4 was a sampling-size artifact
   or a prescreen-ranking-quality problem; (c) a PB-informed (rather than
   purely classical) candidate ranking, which would need a cheap PB
   proxy or a way to batch multiple candidates per APBS call.

## Double-funnel landscape: built, run, and a new WSME limitation found

Following up on Gate A: built `linkage_pka.double_funnel`
(`build_double_funnel_landscape`, `plot_double_funnel`) to visualize a
genuine two-basin, pH-dependent free-energy landscape for GPR68
activation -- stitching `wsme_gpcr`'s exact per-conformer WSME
free-energy profile G(n, pH) (no MD, no sampling) with `linkage_pka`'s
real inter-conformer offset ΔG_activation(pH) (from
`linkage.delta_g_act_from_ln_z` on the D2.50/Asp282/Glu103 cluster's
already-computed `ln_z_total`, reusing the exact joint-microstate data
from the "Coupled 3-site cluster" section above -- no new PB/APBS runs
needed). See the module docstring for the full anchoring derivation: each
conformer's WSME curve is pinned at its own reference structure
(n=nblocks, not wherever WSME's own free energy happens to be lowest)
using the real PB-derived offset, and Q (the stitched coordinate) places
each conformer's actual reference structure adjacent to Q=0, with
disorder spreading outward to Q=-1/+1. 11 new tests (synthetic arrays,
no APBS dependency) verify the anchoring math directly.

**Run against real data**: ΔG_activation(pH) from the D2.50 cluster
ranges from -13.02 kJ/mol at pH 5.0 to +0.01 kJ/mol at pH 8.0 --
consistent with this session's earlier finding that acidification favors
the active conformer, now expressed as a real free-energy number rather
than just Δn_H(pH).

**But a new, unrelated limitation surfaced in the process**: `wsme_gpcr`
has never been run on GPR68 before this. Running it revealed that its
default `WSMEParams` -- validated only against CI2 (65 residues, 18
blocks) -- produce a *physically inverted* landscape on GPR68 (365
residues, 97-101 blocks): the fully-disordered end of each conformer's
WSME curve sits 235-257 kJ/mol *below* (more stable than) the actual
reference structure. Direct comparison confirms this is GPR68-specific,
not a general WSME problem: on CI2, `run_pipeline(CI2.pdb, ph=7.0)` gives
the physically correct pattern (folded reference near the global minimum,
fes=5.8 vs true min 1.2 at n=17/18; disorder costs +43 kJ/mol) -- the
opposite sign/direction from GPR68's result.

~~Most likely cause: WSME's entropic/energetic parameters scale with
block count~~ -- **root cause identified and it is more specific than
that**: see "xi calibration (Prompt 1)" below. `ene` (xi, the vdW energy
per native contact -- the model's one real free parameter per the
original paper) had been left at a single fixed default
(`WSMEParams.ene = -48.2e-3` kJ/mol) applied to every structure, when it
must be calibrated per structure so the model's own heat-capacity peak
lands at the real Tm=333 K. That default is not an arbitrary guess --
it is rhodopsin's (1U19's) own paper-reported calibrated value,
confirmed by directly cross-referencing the paper's reference repository
(see below): using one receptor's calibrated packing energy universally
for every other receptor is exactly the kind of error that would produce
a structure-specific, not general, anomaly.

This is a real, previously-undiscovered limitation of `wsme_gpcr` itself
(separate from Gate A, which is about `linkage_pka`'s PB pKa's) -- it
means the double-funnel plot's *within-basin shape far from Q=0* cannot
be trusted for GPR68 as computed with the old default; only the region
near the real PB-anchored reference states (Q≈0) carries
validated-as-far-as-Gate-A-allows information. Two plots were generated
and delivered to the user: the full-range landscape (visibly dominated
by this artifact) and a version zoomed to the region near Q=0 (where the
real anchor signal lives). **This finding is superseded by the direct
xi-calibration investigation below, which found that correcting xi alone
does not fully resolve GPR68 inactive's folded-minimum problem either --
read the next section before treating either explanation as complete.**

**Compounding caveat**: the anchor itself is the same PB-based
`ΔG_activation(pH)` that failed Gate A, so even the "real" near-Q=0
region is a pipeline-mechanics demonstration, not a calibrated
prediction -- this landscape currently carries multiple independent,
unresolved uncertainty sources (Gate A's PB pKa failure, and the xi
calibration issues documented below), not one.

## xi calibration (Prompt 1): built, validated against the paper's own data, and a deeper GPR68 finding

A follow-up task ("Prompt 1") named the likely root cause directly: xi
was left at a fixed default instead of being calibrated per structure so
the model's own heat-capacity peak (Tm) lands at 333 K, per
Anantakrishnan & Naganathan, Nat Commun 14:128 (2023) (the same paper
`wsme_gpcr`'s GPCR preset already cites). Built `wsme_gpcr/calibration.py`:

- `find_cp_peaks_and_tm` -- locates the Cp(T) peak(s) from `compute_dsc`'s
  already-implemented excess heat capacity (that module already ports the
  paper's `Cp = 2RT dlnZ/dT + RT^2 d^2lnZ/dT^2` expression via
  spline-smoothed finite differences -- verified line-by-line identical
  to the reference `DSCcalc_Block.m`'s own `Cpd=2*R*T.*der1df+R*T.^2.*der2df`
  and spline-grid construction). Implements the paper's bimodal rule
  (Tm = the trough between two peaks, not either peak).
- `calibrate_xi_tm_mode` -- Brent's method root-finds xi so Tm(xi)=333 K,
  bracketed at -80/-20 J/mol per the paper, with a required post-condition
  (the 310 K profile's global minimum must fall in the top 15% of the
  reaction coordinate) enforced by raising `CalibrationError` -- **never
  returns a number that fails either check**, per the task's own explicit
  instruction.
- `calibrate_xi_isostability_mode` -- solves a companion structure's xi
  so its folded-minus-unfolded free energy matches an already-calibrated
  reference's, explicitly flagged (in the result's own `.warning` field)
  as imposing relative stability, not predicting it.
- `compute_fc` -- fraction of residues in a "strongly coupled" block pair
  (default threshold 1 RT at 310 K -- the paper's own exact threshold
  definition could not be verified, network access is blocked).

**Regression validation against the paper's own real data**: the
reference implementation's own repository
(`github.com/AthiNaganathan/GPCR-Landscapes`, named in the task) bundles
its 45-receptor dataset's real structures AND the paper's own reported
per-receptor PDBID/xi/Tm as `.mat` files -- extracted directly (not
downloaded from RCSB/PDB, which remains blocked in this sandbox like
every other external host tried this session). This directly confirmed
the bug: rhodopsin's (1U19's) paper-reported calibrated xi is exactly
-48.2 J/mol, bit-for-bit identical to this codebase's previous universal
`WSMEParams` default -- the default was never a generic placeholder, it
was rhodopsin's own calibrated value, silently applied to every other
receptor including GPR68.

Two-tier check against 5 real receptors (1U19/rhodopsin, 2LNL, 5LWE,
4DKL, 6OS9): **Tier 1** (does this port's Cp/Tm machinery reproduce the
paper's own Tm when run at the paper's own reported xi?) gave 1/5 exact
matches (4DKL: 0.0 K delta) and the rest off by 5-16 K in *both*
directions (no consistent sign) -- consistent with this port's
independently-built, DSSP-free secondary-structure/blocking logic
differing somewhat from the original MATLAB code's, not a formula bug
(confirmed by the line-by-line Cp-formula comparison above). **Tier 2**
(does `calibrate_xi_tm_mode`, run blind, independently recover a
comparable xi?) surfaced a real robustness bug in the first version of
the solver -- an unhandled crash when a bracket edge has no resolvable
Cp peak within the search grid -- fixed to raise a clear, diagnostic
`CalibrationError` instead (see `calibrate_xi_tm_mode`'s own docstring).

**GPR68 inactive, recalibrated**: confirmed the bug report's exact
symptom first -- at the old default (xi=-48.2 J/mol), the 310 K profile's
minimum sits at n=76/101 (75.2%), fes ranging up to +177.3 kJ/mol,
matching the report's "climbs to +178 kJ/mol, minimum near 76%
structured" almost exactly. `calibrate_xi_tm_mode` (bracket narrowed to
[-50.0, -48.2] J/mol after directly scanning this structure's own
Tm(xi) -- monotonic and well-behaved here, unlike some of the bimodal
reference receptors above) found **xi=-49.15 J/mol achieves Tm=333.0 K
exactly** -- a real, in-bracket, paper-typical solution (z=-0.09 vs the
population mean).

**But the folded-minimum post-condition still fails at that xi**: the
310 K profile's global minimum remains at n=76/101 (75.2%) -- unchanged
from the broken default. Hitting the correct Tm does not, by itself,
restore a folded free-energy minimum for this specific structure. Per
the task's own explicit instruction ("If no xi in the bracket yields
Tm=333 K, or the folded minimum still does not appear, raise... do not
return a number that fails the post-condition"), `calibrate_xi_tm_mode`
correctly refused to return this as a valid calibration -- this is the
implementation working as designed, not a bug to paper over.

**Open finding, not yet resolved**: xi calibration alone (via Tm-matching)
is necessary but evidently not sufficient to restore GPR68 inactive's
folded state in this model. Since -49.15 J/mol is barely different from
the old universal default (-48.2 J/mol) -- both land in nearly the same
place on the Tm(xi) curve for this specific structure -- the remaining
~75%-structured local minimum may reflect something else about this
structure's own contact map/blocking (e.g. a genuinely floppy or
disorder-prone region, consistent with real GPCRs' often-unresolved
loops/termini) rather than a pure xi-selection problem. Investigating
that further would mean touching the block definition or contact map,
which this task's own guardrails explicitly put out of scope ("Only xi
selection is broken"). Flagged here as the honest, direct result of
running the prescribed calibration procedure on the real structure, not
resolved further within this task's stated scope.

## Sodium-pocket hypothesis test: built rigorously, real negative result

Direct follow-up (user-directed): does the bWSME model's total lack of
any representation of the conserved D2.50 sodium pocket explain why
GPR68 inactive's folded minimum won't restore even at the correctly
Tm-calibrated xi? The bWSME model is a pure protein-contact model with
zero concept of a bound ion; this session's own `linkage_pka`
Poisson-Boltzmann work already confirmed, on these exact structures,
that an explicit Na+ ion at D2.50 (Asp67) produces a real stabilizing
shift (see "Na+ ion modeling" above) -- a real physical feature the
folding model cannot see at all.

Built `wsme_gpcr/ion_pocket.py`: places the ion at D2.50's real geometry
(reusing `linkage_pka.titration.place_na_ion`'s exact convention --
bisector of the carboxylate oxygens, LJ contact distance from sourced
AMBER/OpenMM radii), searches for every other charged atom within a
real, data-driven radius (not an assumed canonical-motif position), and
appends the resulting ion-mediated stabilization as a new block-block
electrostatic pair to `BlockModel.block_elec` -- pure additional data
fed into `wsme.py`'s existing, unmodified pairwise-electrostatic
machinery. 10 tests, including the required "zero partners = bit-for-bit
no-op" control.

**A real geometry bug surfaced and was fixed during this work**: the
naive single-residue ion placement landed only 1.33 Angstrom from a real
partner's oxygen -- shorter than a covalent bond, physically impossible
for two non-bonded heavy atoms. Root cause: placing the ion from D2.50's
own local geometry alone doesn't account for a SECOND residue also
coordinating the same ion. Fixed with `place_na_ion_multi_coordinate`:
centroids the real coordinating oxygens (D2.50's own OD1/OD2 plus any
nearby partner's) rather than extrapolating from one residue alone --
moves the pathological case to 2.76 Angstrom, squarely in the real Na-O
coordination range (~2.2-2.6 A). A naive two-point average of "estimate"
and "partner position" was tried first and rejected: it only halves an
already-too-short distance, it doesn't fix it -- worth recording as a
real dead end, not silently dropped.

**Real partner found, independently corroborating earlier PB work**: at
every cutoff tried (4-10 A), the only real partner is **Asp282** (OD1 at
4.42 A, OD2 at 2.39 A from the refined ion position) -- the same residue
`linkage_pka`'s real coupling calculation (`compute_pairwise_coupling`,
an entirely different, independent method: Poisson-Boltzmann double-
difference, not geometric distance) already found substantially coupled
to D2.50 earlier this session. Two independent methods agreeing on the
same real partner is a meaningful cross-check, not a coincidence.
Glu103 -- also coupled to D2.50 in the PB work -- is NOT geometrically
close enough to be a direct ion-coordination partner (not found at any
cutoff up to 10 A), consistent with that earlier coupling being a
longer-range electrostatic effect rather than direct ion coordination.

**But the hypothesis test result is a real negative**: adding the ion
term (combined ion-Asp282 stabilization -111.8 kJ/mol at the D250-Asp282
block pair) shifts the whole 310 K profile down slightly (fes range
[3.5,173.8] -> [3.0,173.2] kJ/mol) but **does not move the global
minimum at all** -- still exactly n=76/101 (75.2%), identical to the
no-ion baseline, at every cutoff tried. Mechanistic reason, checked
directly: D2.50 sits in block 19/101, Asp282 in block 76/101 -- 57
blocks apart. For the ion bonus to apply, a folded segment must span
essentially the entire middle of the sequence (>=58 blocks, roughly
blocks 19-76). Many, but not all, of the model's n=76 microstate
arrangements already satisfy this (segments of length 76 can start
anywhere from block 0 to block 25 and still cover both 19 and 76), so
the bonus doesn't preferentially reward n=101 (full fold) over n=76 --
it smears across a broad range of n instead of acting as the specific
"missing piece" that would tip the balance toward full folding.

**Conclusion**: the D2.50 sodium pocket is real, its coordinating
partner (Asp282) is independently confirmed by two different methods,
and the ion-bridge interaction was modeled rigorously (not a quick
guess -- a real geometry bug was caught and fixed along the way). But it
does not explain GPR68 inactive's residual folded-minimum anomaly. The
long sequence separation between D2.50 and its real ion-bridge partner
means this particular interaction cannot discriminate between the
model's current 76%-folded local minimum and full folding. Whatever
keeps the remaining ~25 blocks from folding is a separate, still-open
question -- most likely something about that specific region's own
local contact density/entropic balance, not a missing long-range
electrostatic bridge. This is a genuine, rigorously-obtained negative
result, not an inconclusive one.

## Root-caused: the ~25 non-folding blocks are a construct-scope
## mismatch, not a physics bug -- and the active-state model is far worse

Follow-up investigation (block-level decomposition, not a code change)
into exactly *which* blocks stay unfolded at GPR68 inactive's n=76/101
global minimum, using `WSMEResult.fpath` (P(folded) per block at each
n). The 25 non-folding blocks are not scattered noise -- they fall into
three specific, contiguous regions:

- blocks 0-4 = residues 1-13: the extracellular N-terminal tail.
- block 46 = residues 166-168: the tip of ECL2 (flanked by clearly
  non-helical, extended CA(i,i+4) geometry on both sides, res 160-179).
- blocks 82-100 = residues ~303-365: helix 8 and the intracellular
  C-terminal tail (19 of the 25 blocks -- the dominant contributor).

Structural evidence that these are genuinely disordered/low-confidence
regions in the GPCRdb model, not an artifact of the blocking or contact
code: contact density (`block_cmap` row-sum) in blocks 76-101 is
124 vs. 244 for blocks 0-75 (nearly half); mean CA B-factor jumps from
~7-8 across the ordered bundle to 30-70+ starting at block 82
(residue ~303); and the CA(i,i+4) distance for residues 311-365 is
11-13.5 A throughout -- categorically non-helical (real alpha helices
run 5.4-6.5 A), i.e. an extended/coil conformation, not just "loosely
packed helix."

Critically, the paper's own 5 real reference structures (the ones the
Tier-1/Tier-2 regression gate validates against) do **not** show this
cliff. `gpcr1i` (rhodopsin, 1U19, 348 residues, no gaps) tapers only
mildly toward the C-terminus (contacts: Q1=291, Q4=186 -- rhodopsin is
unusual in having a lipid-anchored, unusually ordered tail via
palmitoylation at Cys322/323). `gpcr9i` (delta-opioid, 4DKL, 288
residues, no gaps) is essentially flat (Q1=283, Q4=243). Neither shows
GPR68's ~50% cliff. This is because real deposited GPCR structures are
either naturally ordered to the end (rhodopsin) or are truncated
crystallization constructs that simply never include the disordered
H8/C-tail region in the model at all (very common -- ICL3 and the
C-tail are the two most commonly truncated/fusion-replaced regions in
GPCR crystallography). GPR68's `WT_Inactive/Active_GPCRdb.pdb` files are
full-length homology models that explicitly build out the entire
365-residue native sequence, including the ~55-residue disordered
H8+tail that no crystallographer would normally feed into a folding
calculation.

**Diagnosis**: this is a construct-scope mismatch, not a ξ, entropy,
electrostatics, or missing-interaction bug. The model is being asked to
compute one cooperative 1D folding coordinate spanning both a rigid,
densely-packed 7TM core and a ~55-residue intrinsically disordered tail
that was never part of the paper's own calibration domain. Demanding
the global minimum sit in the top 15% of that coordinate is
mathematically close to demanding the disordered tail also "fold" --
which no real GPCR C-tail does. This directly explains why neither the
ξ recalibration nor the sodium-pocket bridge (both real, both correctly
implemented) could fix it: they don't touch the part of the problem
that's actually wrong (scope), only the parts that were fine
(calibration, missing interactions).

**Active-state comparison** (the other GPCRdb model, `WT_Active_GPCRdb.pdb`,
run through the identical untouched pipeline at the same default ξ):
dramatically worse, not better -- global minimum at n=6/97 (6.2%
folded), fes range [4.8, 239.7] kJ/mol, 92 of 97 blocks unfolded at the
minimum. This is *not* explained by reduced core-bundle packing: the
transmembrane-bundle contact density (blocks spanning the first three
quarters) is essentially identical active vs. inactive (245 vs. 244).
So the active-state collapse is either (a) a stronger version of the
same disorder-scope problem (its C-terminal quarter contact density is
even lower than inactive's, 90.5 vs. 124.1), (b) a real, if
exaggerated, reflection of genuine GPCR biophysics -- active-state
conformations are well documented to be intrinsically less stable
without a bound intracellular partner (G protein/arrestin/nanobody),
which is exactly why they're hard to purify/crystallize on their own --
or (c) an artifact of the active model's block partition differing from
inactive's (97 vs. 101 blocks, a ~14-residue/4-block discrepancy, most
likely in how ICL3 is packed by the homology-modeling pipeline for the
two states). These are not mutually exclusive and have not yet been
disentangled.

**Proposed tests** (none require touching model physics):

1. Truncation test: re-run both GPR68 structures with the N-terminal
   tail (res 1-13) and H8+C-tail (res >~305) excluded from
   `run_pipeline`'s modeled range, matching the scope convention the
   paper's own reference structures already have by construction.
   Prediction if the scope-mismatch diagnosis is right: the folded
   minimum lands near the top of the now-shorter coordinate for both
   states.
2. Disorder-aware order parameter: keep all residues in the contact map
   / energetics, but exclude the same three flagged regions from the
   reaction coordinate used by the top-15% post-condition check, and
   see whether a well-defined core-folded basin was already present
   underneath the tail-dominated 1D profile.
3. Active-state collapse triage: diff `block_elec` block-by-block
   active vs. inactive for any anomalous repulsive term unique to the
   active geometry, and diff the two block partitions directly (101 vs.
   97 blocks) to localize the ICL3-region discrepancy and test whether
   forcing matched block boundaries changes the result.
4. Cross-check against the paper's own dataset: check whether any of
   the 45 reference receptors in the bundled `.mat` files have both an
   active and inactive structure, and whether real published pairs also
   show this magnitude of active-state collapse -- if not, that's
   further evidence the GPR68 active homology model specifically has a
   construct-quality problem rather than this being generic model
   behavior for any active GPCR conformation.

None of this has been implemented yet -- it is a diagnosis and a set of
proposed next steps, not a fix, pending direction on which test to run
first.

## Test 1 result: truncation fixes inactive almost completely; active is
## a separate, unresolved problem

Ran test 1 as proposed above: built truncated copies of both
`WT_Inactive_GPCRdb.pdb` and `WT_Active_GPCRdb.pdb` keeping only
residues 14-302 (author numbering; drops the N-terminal tail and the
H8+C-tail flagged above), no other change -- same untouched pipeline,
same contact/electrostatics/entropy code, same default ξ.

**Inactive: essentially fixed.** Global minimum moves from n=76/101
(75.2%) to n=77/79 (**97.5%**) after truncation, with only 2 residual
soft blocks (down from 25), fes range shrinks from [4.4,177.3] to
[3.5,28.1] kJ/mol. Holds at both the untouched default ξ (-48.2 J/mol)
and the Tm-calibrated ξ (-49.15 J/mol) -- the fix comes from scope, not
from ξ, exactly as the diagnosis predicted. This is strong direct
confirmation that the disorder-scope mismatch, not a physics bug, was
the cause of inactive's stuck minimum.

**Active: not fixed.** Same truncation moves the minimum only from
n=6/97 (6.2%) to n=6/73 (8.2%) -- still totally collapsed, 68 of 73
blocks unfolded at the "minimum." The disorder-scope explanation is
therefore *not* sufficient for the active-state model; something else,
localized to the core 7TM bundle itself (residues 14-302, i.e. the same
range that folds essentially perfectly for inactive), is broken for the
active homology model specifically.

Follow-up comparison of the two truncated structures (same 289-residue
range in both) narrows it down, but doesn't close it: total contacts
(19153 vs. 18015) and total electrostatic energy (-37.0 vs. -25.7
kJ/mol) are comparable, not wildly different -- this isn't one giant
repulsive outlier or a bulk contact-count collapse. Two smaller, real
differences were found: active has more residues classified as coil
(38/289, 13.1%) vs. inactive (26/289, 9.0%), and a lower fraction of
long-range (tertiary, block-index separation >=10) contacts (0.152 vs.
0.179). Both point toward active's secondary-structure geometry/contact
topology being somewhat less regular/cooperative than inactive's, but
neither is dramatic enough by itself to obviously explain a drop from
97.5% to 8.2% folded -- this still needs test 3 (block-partition diff,
block_elec block-by-block comparison) to actually localize the cause,
or test 4 (checking whether the paper's own active/inactive receptor
pairs, if any exist in the bundled dataset, show anything like this
magnitude of asymmetry).

**Net effect on the original bug report**: the inactive-state folded-minimum
anomaly that motivated this entire investigation (ξ calibration, then
the sodium-pocket hypothesis, then this scope diagnosis) is resolved --
truncating the two flagged disordered termini, with zero physics
changes, restores a proper folded minimum at 97.5%. The active-state
model surfaced a second, distinct, still-open problem in the course of
testing this fix; it was not part of the original bug report and has
not been fixed.

## Does active need its own ξ recalibration? Tried it -- no meaningful
## melting transition exists to calibrate against

Ran `calibrate_xi_tm_mode` on the truncated active structure (residues
14-302, same range that fixed inactive), full bracket [-80,-20] J/mol,
target Tm=333K. Result: `CalibrationError`, correctly raised rather than
returning a number. At the *most stabilizing* edge of the bracket
(xi=-80 J/mol -- more stabilizing than any of the paper's real 45
receptors), the partition function's Cp(T) curve has no resolvable peak
at all: "too flat/monotonic to define a melting transition." At the
other edge (xi=-20 J/mol) Tm=346K, 13K off target, and there's no
evidence a valid two-state-like transition exists anywhere between the
two edges either -- the function's own edge-check is what caught this,
not an internal search failure.

Conclusion: this isn't a case where the right ξ hasn't been found yet;
within the entire physically-defensible range, there is no cooperative
two-state melting transition to Tm-match against for this structure.
Recalibrating ξ for active is not the fix and was correctly refused by
the calibration machinery's own guardrail, consistent with (and adding
independent evidence for) the test-1 finding that active's problem is
structural/topological in the core bundle, not an energy-scale
parameter. Whatever is wrong with active (test 3, still open) has to be
resolved before a meaningful ξ calibration for it is even possible.

## Test 3: block-partition ruled out; active has a genuinely sparser
## tertiary contact network, broadly distributed, not one hotspot

**Sub-test A -- is it just how residues get chopped into blocks?** No.
Rebuilt a block model using active's real, untouched 3D contact
geometry (`ra.contact_map`, from its actual atom positions) but
partitioned with *inactive's* secondary-structure assignment instead of
active's own (`build_blocks(ri.ss_mask, ra.contact_map)`). If block
fragmentation from the 28/289 residues where the two structures'
geometric SS-assignment disagrees (see below) were driving the
collapse, forcing inactive's cleaner partition onto active's geometry
should have helped. It didn't -- global minimum actually got worse
(3.8% folded vs. 8.2% with active's own partition). The collapse lives
in the atomic contact geometry itself, not the blocking choice.

**Where the SS assignment disagrees (28/289 residues, ~10 short runs):**
most are 1-4 residue blips, but one stands out: residues 254-257,
inactive=helix / active=coil, a four-residue stretch right at the
cytoplasmic end of TM6 -- precisely where real class-A GPCRs are known
to locally unwind/kink upon activation (the conserved Pro-containing
"toggle switch" region). This is plausibly a real activation-associated
signal being correctly picked up by the geometric SS assignment, not
noise. But sub-test A already shows this isn't what's driving the
overall collapse by itself.

**Sub-test B -- direct residue-residue contact topology diff** (bypasses
blocking entirely; sequences and author numbering are identical between
the two structures, confirmed, so residue-pair contacts can be diffed
1:1). Restricted to long-range pairs (sequence separation >=10, i.e.
tertiary packing, not local helix turns): inactive has 401 such
contacts, active has only 302 -- a **25% net reduction**, with 167 pairs
present only in inactive vs. just 68 gained in active. This loss is not
localized to one region -- it's spread across essentially the whole
receptor (N-term/TM1-TM2, TM2-TM3, ECL2/TM4-TM5, TM5-TM6, TM6-TM7 all
show net losses of 6-25 contacts per 10-residue window). This is a
genuinely global loosening of tertiary packing in the active homology
model, not a single hotspot.

**Sub-test C -- is this a modeling-confidence artifact or a real signal?**
Checked whether active's core bundle (res 14-302, i.e. excluding the
already-flagged disordered termini) shows elevated B-factor relative to
inactive's, as it would if the active homology model were simply
lower-confidence in this region. It doesn't: mean CA B-factor 10.02
(active) vs. 12.47 (inactive), identical median (5.44), identical
fraction >20 (0.135). No confidence-artifact signal in the one proxy
available in this sandbox -- inconclusive rather than a clean
distinguisher, since this B-factor column's actual meaning for these
GPCRdb files (true per-residue confidence vs. carried-through template
values) is not independently known here.

**Where this leaves it**: the active-state collapse is real, broad
(not one fixable hotspot), survives block-partition control, and isn't
explained by an energy-scale (ξ) problem. It is consistent with real
GPCR biophysics (active-state conformations are well documented to be
intrinsically less stable/more dynamic without a bound intracellular
partner locking them in place -- exactly why nanobodies/mini-G proteins
are needed to trap them for structural work), but the magnitude (75%+
loss of a stable fold) is large enough that homology-model quality for
the active state can't be ruled out either, and the one confidence
proxy available (B-factor) doesn't clearly settle it either way.
Distinguishing "real, if exaggerated, biology" from "active-state
homology-model quality" would need something not available in this
sandbox -- either an experimentally solved GPR68 active structure, or
the GPCRdb model's own template/confidence metadata. This is where the
investigation currently stops without new external data.

## The user supplied exactly that missing data: a real solved GPR68
## active structure (9BHM). It settles the artifact-vs-biology question
## -- and overturns the "sparser packing" explanation

9BHM (PDB, deposited; user-supplied `.cif`) is a real 2.9 A cryo-EM
structure: "Human proton sensing receptor GPR68 in complex with
miniGs" -- GPR68 (chain R) bound to a mini-Gs heterotrimer mimic
(miniGs + Gβ1 + Nanobody-35), the standard construct used to trap an
active-state GPCR for structural work. This is real, independent,
experimentally-determined ground truth for the active conformation --
exactly the kind of data the previous entry flagged as missing.

**It resolves residues 13-294, no gaps.** No fusion-replaced ICL3, no
missing loops. And critically: it does **not** resolve anything past
residue 294 -- the H8+C-terminal tail (res ~295-365) simply isn't in
the model at all, even with a bound G protein that typically helps
order the intracellular face. This independently confirms, with real
data, the disorder-scope diagnosis from earlier in this investigation:
real GPR68's H8+tail is genuinely not part of the ordered fold, in
either state, and the truncation boundary chosen for test 1 (res 14)
practically coincides with where the real structure's density actually
begins (res 13).

**Ran it through the identical, untouched pipeline.** Result: still
totally collapsed -- global minimum at n=3/73 (**4.1%** folded), every
single block unfolded at that minimum. This is *worse* than the
GPCRdb active homology model's 8.2% (truncated) / 6.2% (full-length),
not better.

**This falsifies the "active homology model is just lower quality/
looser packed" explanation.** 9BHM's own contact density (block_cmap
row-sum mean 260.7) and long-range/tertiary contact fraction (0.179)
are *not* depressed relative to inactive -- they're comparable to or
higher than the GPCRdb inactive model's own numbers (244 / 0.179). A
real, properly-refined, high-resolution active structure is at least as
densely packed as inactive by these metrics, yet the model still can't
find a fold for it. So the earlier test-3 finding (GPCRdb active has
25% fewer long-range contacts than GPCRdb inactive) was real for that
specific homology model, but is now shown to be a symptom of that
model's quality, not the general reason WSME fails on GPR68's active
state -- the real structure fails for a different, deeper reason that
plain contact density/fraction doesn't capture (most likely the
specific topology of which blocks bridge to which, not how many
bridges exist -- not yet isolated).

**ξ recalibration fails identically on the real structure.** Ran
`calibrate_xi_tm_mode` on 9BHM directly: `CalibrationError`, same
failure mode as the homology model -- even at xi=-80 J/mol (the most
stabilizing edge of the physically valid bracket), no Cp(T) peak
exists at all. No cooperative two-state melting transition to
Tm-match against, full stop, using real experimental coordinates.

**Conclusion**: the active-state collapse is real GPR68 biology (or at
minimum, a real and robust property of how this specific WSME
formulation treats GPR68's true active conformation), not a homology-
model-quality artifact -- confirmed with real experimental data, not
inferred. It is consistent with the well-documented practical fact that
active-state GPCR-G-protein complexes are often less thermally
stable/harder to purify than the apo inactive receptor (this exact
structure required cholesterol hemisuccinate and a stabilizing
nanobody just to solve). What specifically in the real contact
topology breaks cooperativity, given that raw density/fraction numbers
look fine, remains open and would be the next question if pursued
further.

## Double-funnel landscape v2: the inactive-basin fix wired back in

The double-funnel plot (see "Double-funnel landscape" above) predates
everything discovered in this document from "Root-caused: the ~25
non-folding blocks..." onward. Re-ran it (`double_funnel_v2`, script in
scratchpad, not committed -- same convention as the original driver)
with two inputs updated and everything else, including the Gate A
caveat, held fixed:

- **Inactive**: now the truncated ordered core (res 14-302) instead of
  the full 365-residue GPCRdb model -- the disorder-scope fix from test
  1, wired into the actual plot for the first time.
- **Active**: now the real 9BHM cryo-EM structure (chain R, res 13-294)
  instead of the GPCRdb active homology model -- a strict upgrade in
  input quality, not a fix for the active-basin pathology (which is
  still open, see above).
- **Unchanged**: the D2.50/Asp282/Glu103 cluster's PB-derived
  ΔG_activation(pH) anchor (same already-computed `ln_z` data reused
  as-is) -- still fails Gate A, still pipeline-mechanics only, not
  recalibrated by anything in this update.

**Result, and why it's a real improvement, not just a re-run**: the
inactive basin now has the physically correct shape for the first time
-- free energy rises monotonically from the reference structure (Q=0,
anchored at 0 kJ/mol) out to full disorder (Q=-1, ~+25 kJ/mol at pH
5), i.e. a real folded well, matching the fixed 97.5%-folded landscape
underneath it. The old v1 plot's inactive side did not have this
shape (dominated by the pre-truncation artifact). The active basin, by
contrast, now visibly displays the exact pathology documented above as
a feature of the plot itself rather than only a caveat in prose: free
energy falls monotonically and steeply from the seam (Q=0+, near
9BHM's own reference structure) out to full disorder (Q=+1, -60 to -78
kJ/mol) -- the model treating near-total unfolding as dramatically more
favorable than the real, solved active structure, for every pH in the
grid. Both basins' pH-dependence still comes through (color gradient
top-to-bottom within each basin), driven by the unchanged, uncalibrated
ΔG_activation(pH) anchor plus each conformer's own pH-dependent
electrostatics.

**Net assessment against the ultimate goal**: this is a real, wired-in
improvement (the inactive basin is now trustworthy in shape, not just
in principle), and it makes the active-basin problem legible directly
in the plot rather than only in a text caveat -- useful in its own
right for explaining the current state to a reader. It does not change
the two remaining blockers: Gate A still fails (the pH axis itself is
uncalibrated), and the active-basin fold-order pathology is still
unresolved (its shape in this plot is a real, reproducible model output,
not yet a trustworthy free-energy landscape for the active conformer).
Both must still be closed before this plot is a validated scientific
result rather than a pipeline-mechanics demonstration.

## Gate A revisited: pdie was the dominant, undocumented lever --
## real, large improvement, but a circularity caveat means this is
## not yet an honest "pass"

Applied today's diagnostic playbook (falsify with real, cheap tests
before touching parameters; real ground truth over inference) to Gate
A, which had been stuck at RMSE 7.86-11.3 across four relaxation
variants (see "Gate A calibration: FAIL" above).

**First, ruled out grid resolution.** The protein-side grid
(`dime=(33,33,33)`, `glen=(65,65,65)`) has ~2.03 A spacing, coarser
than the model-compound grid's ~0.78 A -- a real, unexplained mismatch,
and a plausible source of systematic error. Tested directly: reran
Asp40 (near-exact, diff -0.07) and Glu67 (catastrophic, diff +15.97) at
a 3x finer, spacing-matched protein grid (`dime=(97,97,97)`, same
`glen`, ~0.68 A). Result: Asp40 unchanged (diff -0.14), Glu67
*unchanged* (diff +17.32, slightly worse). Grid resolution is not the
cause -- a clean, decisive negative result before moving on.

**Then found the real lever: `pdie` (protein interior dielectric) was
set to 2.0 in the driver script with no documented justification
anywhere in the codebase** (`grep pdie` across `linkage_pka/` turns up
only the parameter's plumbing, never a rationale or citation for the
value 2.0 -- inconsistent with this codebase's usual discipline of
citing every physical constant). pdie=2.0 is on the extreme low end of
what's used in the PB-pKa literature; classic work (Antosiewicz,
McCammon & Gilson, J Mol Biol 1994) established that *rigid*,
single-structure PB pKa calculations need a substantially elevated
*effective* pdie (they found ~20 worked well across several proteins)
to phenomenologically compensate for the conformational
reorganization/polarization a static structure can't otherwise
capture -- exactly the gap this pipeline's own relaxation levers were
built to (partially) address a different way.

Tested directly on the same two sites plus two more catastrophic ones
(Glu75, Asp95), sweeping pdie = 2/4/8/20 at the fast (33^3) grid,
everything else held fixed: Asp40 (already good) stays fine across the
whole range (diff -0.07 to +0.64). Every catastrophic site improves
*monotonically and substantially*:

| Site | expt | pdie=2 | pdie=4 | pdie=8 | pdie=20 |
|---|---|---|---|---|---|
| Glu67 | 3.76 | +15.97 | +8.69 | +4.72 | +2.00 |
| Glu75 | 3.26 | +17.85 | +9.97 | +5.64 | +2.67 |
| Asp95 | 2.16 | +14.95 | +8.85 | +5.51 | +3.22 |

**Full 17-site Gate A RMSE, rigid geometry (no relaxation), by pdie:**
pdie=2 -> 10.99 (original baseline), pdie=8 -> 3.28, pdie=20 -> 1.62.
pdie alone, with zero conformational relaxation, already beats the
previous *best* variant (neighbor relaxation at pdie=2: RMSE 7.86) by
a wide margin -- pdie was the dominant lever all along, not relaxation.

**Combined with the existing best relaxation lever** (neighbor
relaxation, 8 A radius, `optimize_rotamer=True`):
- pdie=20 + neighbor relaxation: RMSE=1.40, MAE=1.08 -- still fails
  the <1.0 threshold, but every remaining diff is <=+2.93 (nothing
  catastrophic left).
- pdie=40 + neighbor relaxation: **RMSE=0.95, MAE=0.75 -- passes the
  <1.0 threshold** for the first time this project has run Gate A.

**Why this is not being reported as an honest, unconditional pass:**
pdie=40 was found by direct grid search (2/4/8/20/40) against this
exact 17-site RMSE -- the same numbers Gate A's pass/fail criterion is
computed from. That is circular: freely tuning one parameter until a
benchmark's own error metric crosses a threshold is not the same as
independently validating the pipeline against that benchmark, even
though the *direction and mechanism* of the fix (pdie=2 was
indefensibly low; literature-precedented elevated pdie for rigid PB
pKa calculations is a real, well-established compensation, not an
invented fudge factor) is genuine and well-grounded. pdie=20 -- the
literature's own commonly-cited value, chosen independently of this
specific dataset's outcome -- still fails (RMSE 1.40). Only pushing
further, specifically because 20 wasn't enough *on this benchmark*,
crosses the line.

**Honest current status**: Gate A's RMSE has been reduced from a
catastrophic 10.99 to 0.95-1.62 depending on exact pdie choice, via a
real, mechanistically well-understood, literature-precedented fix
(not a parameter hack) -- a large, genuine improvement, not resolved
alignment error. But it should not yet be treated as a validated pass
per the pipeline's own acceptance rule ("no ancestral-node number may
be reported" before Gate A passes) without addressing the circularity:
either (a) commit to a pdie value chosen independently of this exact
benchmark (e.g. the literature's pdie=20) and accept that it still
fails here, prompting further real investigation of what's specific to
SNase/this pipeline beyond dielectric choice, or (b) do a proper
train/held-out split of the 17 sites -- pick pdie on a subset, report
RMSE on the rest -- so a threshold-crossing claim isn't self-referential.
Neither has been done yet. This is real, substantial progress and a
genuine, well-evidenced mechanistic finding, but "Gate A passes" is not
yet a claim this document is prepared to stand behind.

## Gate A, path (a) executed: committed to literature pdie=20, reframed
## the metric -- and found the "pass" at pdie=40 is an artifact, while
## the pipeline's buried-carboxylate physics is actually anti-correlated

Chose path (a) over the train/test split, because with a single scalar
parameter across ~17 near-consensus sites a held-out split is
statistical theatre (the fitted pdie is stable across folds by
construction -> held-out RMSE approximately equals the full-set RMSE, so
it would rubber-stamp whatever pdie search produced, not test it). Path
(a): commit to a pdie chosen for physical reasons independent of this
benchmark (the literature's ~20 for rigid single-structure PB), and
judge the result on correlation + per-class residuals, not RMSE alone.

**Reframed metric across the pdie arc** (comparable sites, same
exclusions `compute_gate_a_rmse` uses; Pearson r and Spearman rho of
computed vs. experimental, plus computed dynamic range relative to the
experimental range):

| Variant | RMSE | Pearson r | Spearman | comp.range / expt.range |
|---|---|---|---|---|
| pdie=2 rigid (orig) | 10.99 | -0.584 | -0.754 | 8.54 |
| pdie=8 rigid | 3.28 | -0.419 | -0.587 | 1.87 |
| pdie=20 rigid | 1.62 | +0.299 | +0.161 | 0.59 |
| pdie=20 +neighbors | 1.40 | +0.593 | +0.362 | 0.65 |
| pdie=40 +neighbors | 0.95 | +0.834 | +0.414 | 0.52 |

At first glance pdie=40 looks genuinely good (Pearson 0.834). **But
decomposing by residue class destroys that reading.** Restricting to the
13 carboxylates (Asp/Glu -- the buried, strongly-shifted, scientifically
interesting sites this benchmark exists to test), His excluded:

| Variant | Pearson r (carboxylates) | Spearman (carboxylates) | comp.range |
|---|---|---|---|
| pdie=2 rigid | -0.570 | -0.600 | 37.25 |
| pdie=20 rigid | -0.510 | -0.600 | 2.39 |
| pdie=20 +neighbors | -0.482 | -0.426 | 2.12 |
| pdie=40 +neighbors | -0.339 | -0.308 | 0.78 |

The carboxylate-only correlation is **negative at every pdie** -- the
pipeline ranks SNase's buried carboxylates *backwards* relative to
experiment (experimental carboxylate range 2.33 pKa units; computed
range at pdie=40 collapses to 0.78, about a third). The apparently good
full-set Pearson of 0.834 was carried **entirely** by the His-vs-
carboxylate separation (the 4 His sit high, ~6, and the carboxylates
low, ~4, on both axes -- so a linear fit through the two clusters looks
correlated even though neither cluster is internally ordered correctly).
Raising pdie doesn't fix the physics; it compresses a wrong-signed
prediction toward the mean until the squared error stops mattering.
That is exactly how the RMSE crosses 1.0 at pdie=40 without any real
gain in buried-charge predictive skill -- a benchmark artifact, now
demonstrated, not just suspected.

**Path (a) verdict**: at the physically-principled, benchmark-independent
pdie=20, Gate A **fails honestly** -- RMSE 1.40-1.62 (a near miss on the
aggregate), but more tellingly a *negative* carboxylate rank
correlation. This is the real scientific finding: this single-structure
PB pipeline, even with a literature-standard elevated dielectric and the
neighbor-relaxation lever, does not capture the determinants of SNase's
buried carboxylate pKa's -- it gets their relative order wrong. That is
consistent with the Garcia-Moreno lab's own framing of why this dataset
is hard (buried-charge pKa's are set by specific local
desolvation/H-bonding/reorganization that a single static structure with
a uniform interior dielectric cannot represent), and it means the honest
path to a real Gate A pass is not more dielectric tuning but a
categorically better treatment of buried-site conformational response
(explicit reorganization/multi-conformer with PB-validated -- not
classically-prescreened -- ensembles), which remains future work.

**Net for the ultimate goal**: the pH-titration axis of the double-funnel
landscape depends on this same PB machinery. The honest status is
unchanged in kind but now much more precisely characterized: pdie=2 was
a real, large, previously-undocumented error (fixing it drops aggregate
RMSE from 11 to ~1.4 at a principled dielectric), but the pipeline still
cannot be said to have *passed* Gate A, because at any benchmark-
independent setting it fails, and the failure is now understood
specifically -- wrong-signed buried-carboxylate ordering, not just large
scatter. Every pH-dependent number downstream (delta_g_activation(pH),
the double-funnel colour axis) continues to carry the "pipeline-mechanics
demonstration, not calibrated prediction" caveat, but the reason is now
concrete and diagnostic rather than a blanket disclaimer.

## Anti-correlation root cause: the model gets the SIGN of the buried-
## carboxylate physics wrong (desolvation vs. specific stabilization)

Ran the diagnostic that decides whether a targeted fix is viable or the
full conformational-sampling hammer is required. For the 13 comparable
carboxylates, compared three quantities per site: burial (100 - SASA%,
from the experimental dataset's own SASA column), the MODEL's pKa shift
(computed - model_pKa, at pdie=40+neighbors, the pipeline's best
variant), and the EXPERIMENTAL pKa shift (expt - model_pKa). Result is
unambiguous:

| Correlation | Pearson | Spearman |
|---|---|---|
| burial vs. MODEL shift | **+0.669** | +0.692 |
| burial vs. EXPERIMENTAL shift | **+0.011** | -0.127 |
| MODEL shift vs. EXPERIMENTAL shift | -0.587 | -0.567 |

And the directions:
- **MODEL shifts are ALL positive** (every one of the 13; mean +0.31,
  range +0.09..+0.67). The pipeline *always* raises a buried
  carboxylate's pKa -- the signature of a pure desolvation penalty
  (removing a charge from water into a low-dielectric interior
  destabilizes the ionized state, so pKa goes up), scaling with burial
  (+0.67 correlation).
- **EXPERIMENTAL shifts are mostly negative** (mean -0.52), and have
  essentially *zero* correlation with burial (+0.01). The two most
  strongly-shifted real sites go the OPPOSITE way from the model:
  Asp95 (66% buried) real shift **-1.74**, model +0.43; Glu10 (83%
  buried) real shift **-1.28**, model +0.61. Reality *lowers* these
  pKa's -- the ionized (charged) state is *stabilized*, not
  destabilized, by specific local interactions (H-bond networks, salt
  bridges, favorable backbone/polar contacts) that hold the buried
  charge and are entirely invisible to a uniform-interior-dielectric
  continuum on a single static structure.

**This is a sign error, not a magnitude error.** The one carboxylate
whose real pKa actually *is* desolvation-dominated (Glu43, the only
site with a positive experimental shift, +0.22) is the one the model
gets essentially right (+0.16) -- confirming the model's mechanism is
real, just not the *dominant* mechanism for most of SNase's buried
carboxylates. No dielectric value, and no monotone correction term, can
convert a systematically wrong-signed prediction into a right one,
because whether a given buried carboxylate ends up shifted up (pure
desolvation) or down (specific stabilization wins) is exactly the
per-site question the continuum model cannot answer from burial alone.

**Decision implication (directly answers the "targeted fix vs. full
hammer" question)**: a cheap targeted patch is OFF the table. Getting
these right requires explicitly representing the charge-stabilizing
local interactions and their pH-coupled reorganization -- i.e.
constant-pH MD, or a genuine multi-conformer treatment with PB-validated
(not classically-prescreened) ensembles that can find the
charge-stabilized states. That is a real method-development project with
a genuinely uncertain outcome (SNase buried carboxylates sit near the
edge of what even specialized published methods achieve), not a tuning
step.

**Asset implication**: this diagnostic is itself a clean, quantitative,
publishable result -- it doesn't just say "the pipeline fails," it
localizes the failure to a specific, well-understood physical mechanism
(bulk desolvation vs. specific charge stabilization) with a decisive
sign-level signature. For a methods/negative-results framing this
strengthens the contribution; for a biology-prediction framing it
confirms calibration is blocked behind conformational sampling, not
parameter choice.

## Ancestral-reconstruction cooperativity pilot: mechanically works,
## but the fc fidelity gate fails -- run the tree only after fixing it

The user supplied four real AlphaFold-modeled ancestral nodes from the
larger proton-sensing clade (node_20, node_148, node_80, node_34; ~320
residues each, single chain, mean pLDDT 84-87 -- genuine per-residue
confidence, unlike the GPCRdb homology models). Ran the WSME
cooperativity pilot proposed in the previous section, to test whether a
comparative cooperativity signal is real before committing to the full
tree.

**Structural QC + truncation (disorder-scope lesson applied):** all four
share the same pLDDT profile as GPR68 -- low-confidence N-tail (res
1-15, pLDDT ~63), well-ordered 7TM core (res ~16-285, pLDDT 85-97), and
a low-confidence H8/C-tail (res ~300+, pLDDT dropping to 30-60). Applied
the same fix from test 1: truncated every node to the common confident
core res 16-285 (a strictly consistent protocol across nodes, the
control the previous section flagged as mandatory). This is cleaner than
GPR68's case -- here the truncation boundary is set by real pLDDT, not a
geometric proxy.

**What worked (mechanics are sound):**
- All four load, truncate, and run through the identical untouched
  pipeline (~68-72 blocks each).
- 3 of 4 fold properly at the default xi (node_20/148/80 at 92-94%
  folded); the truncation transfers the disorder-scope fix to these AF
  models cleanly. (node_34 collapses to 9.7% at default xi -- see below.)
- Per-node xi calibration to Tm=333K succeeds and lands near the paper's
  own value (-48.9 +/- 2.76 J/mol): node_20 -46.6, node_80 -49.2 (both
  within ~1 sigma), node_34 -52.8 (~1.4 sigma), node_148 -41.6 (~2.6
  sigma). So the packing-energy scale is in the right regime.
- The nodes genuinely differ in raw coupling scale (at matched default
  xi): mean|coupling free energy| ranges 2.2 (node_148) to 10.0
  (node_80), strong-pair counts 240 to 2060. There is a real,
  substantial cross-node signal in the continuous coupling matrix.

**What failed (the fidelity gate -- why the tree must not be run yet):**
- **fc is 97-100% for every node, calibrated or not**, vs. the paper's
  13.0 +/- 4.5% target. This is the same fc the regression gate
  (examples/calibration_regression.py) was built to check, with the
  paper's own instruction "do not proceed to receptor results until it
  passes." It fails by ~7x, and per-node xi calibration does NOT fix it
  (node_20 default fc 94% -> calibrated 97%; all others already at/near
  100%). Because every node saturates at the ceiling, **fc as currently
  computed cannot discriminate between nodes** -- it is useless as the
  comparative cooperativity metric for exactly this application.
- Two likely, non-exclusive causes, neither yet resolved: (1) the fc
  threshold. `DEFAULT_FC_THRESHOLD_KJ_MOL` (1 RT at 310K, ~2.58 kJ/mol)
  is documented in calibration.py itself as "a documented choice, not a
  verified transcription of the paper's own threshold" -- with mean
  |coupling| of 2-10 kJ/mol over a ~70x70 matrix, almost every block has
  at least one partner above 2.58, forcing fc toward 100%. (2) A
  truncation confound that is intrinsic to this application: cutting to
  the ordered core (necessary for folding) removes the weakly-coupled
  disordered periphery, which mechanically *raises* the coupled fraction
  relative to the paper's whole-structure fc. So the truncation that
  FIXES folding simultaneously BREAKS the apples-to-apples fc comparison
  to the paper -- two requirements in direct tension, not yet reconciled.
- node_34's collapse at default xi (9.7%) and the non-monotonic Tm(xi)
  relationship for node_148 and node_34 (e.g. node_34: xi=-48 -> 336K,
  -46 -> 382K, -44 -> 336K -- not monotonic, signalling messy/bimodal
  thermograms) mean those two nodes' calibrations sit on noisier ground
  than the single reported number suggests. A real run would need the
  full-resolution bimodal-aware Tm treatment per node, not the coarse
  interpolation used for this pilot.

**Pilot verdict (this is what the pilot was for):** the cooperativity-
evolution application is mechanically viable -- structures load,
truncate, fold, and calibrate to near-paper xi, and there IS a real
cross-node coupling signal -- but it is NOT yet trustworthy, because the
one quantitative fidelity gate available (fc vs. the paper's 13%) fails
by ~7x and, being saturated, cannot even rank the nodes. Running the
full ancestral tree now would be building comparative evolutionary
claims on an unvalidated, saturated metric. The de-risking succeeded
exactly as intended: it stopped the tree before it started and named the
specific blocker. **Required before any tree run:** (a) pin the fc
definition/threshold to the paper's actual one and validate it
reproduces fc≈13% on the paper's OWN 45 receptors (the reference .mat
files are already in the scratchpad) -- not tuned to hit 13% on an
ancestor, which would be the same circularity trap as the Gate A pdie;
(b) resolve the truncation-vs-fc tension (either a coupling metric that
is robust to periphery truncation, or a whole-structure fc with a
disorder-aware coupling coordinate); (c) only then, with a validated
discriminating metric, compare across nodes -- and even then as
hypothesis generation with AltAll posterior-uncertainty controls, since
no one has validated WSME cooperativity on ancestral sequences.

## fc-fidelity validation, item (a) executed: NOT a threshold problem --
## a real block-partition and coupling-scale mismatch against the
## paper's own ground truth

Item (a) above named the requirement precisely: pin fc to the paper's
real definition and validate against their own 45 (53 available)
reference receptors, not tune it on an ancestor -- the same discipline
already used for Gate A's pdie. Executed that directly using the real
`DeltaGc_310_<gpcr>` and `CouplingMat_310_<gpcr>` variables already
bundled in the reference repo's `.mat` files (real per-receptor,
paper-computed ground truth, not our own numbers).

**Step 1 -- what threshold does the paper's OWN real data imply?** Swept
a threshold against `|DeltaGc_310|` across all 53 usable receptors and
found the value whose *mean* fraction-exceeding matches the paper's
reported 13.0%: **~16 kJ/mol** (interpolated between 16.00 kJ/mol ->
13.47% and 16.25 kJ/mol -> 12.77%), roughly **6.2x RT(310K)** -- not the
1x RT (2.58 kJ/mol) `DEFAULT_FC_THRESHOLD_KJ_MOL` was guessing. That
guess was off by ~6x. But the cross-receptor spread at this threshold
(std ~17.5%) is far larger than the paper's reported ±4.5%, meaning a
single fixed-kJ/mol threshold on `|DeltaGc|` reproduces their *mean* but
not their *precision* -- this simple reconstruction is at best a proxy
for their real fc methodology, not an exact transcription of it (their
actual method, not included in this repo's 4 bundled `.m` files, may
normalize per-receptor or use a different aggregation).

**Step 2 -- does correcting the threshold fix OUR pipeline?** No. Ran
`compute_coupling` on the real `gpcr9i.pdb` (4DKL) at the paper's own
reported ene/Tm (-49.9 J/mol, 333 K) and computed fc at the corrected
~16 kJ/mol threshold: **fc = 92.4%**, barely moved from the old
threshold's 100%, nowhere near 13%. The threshold was never the
dominant problem.

**Step 3 -- root cause, found by direct comparison to the paper's own
real numbers on their own reference structure:**
- **Coupling-value scale mismatch.** Our `coupling_free_energy` on
  gpcr9i has mean|value|=11.87 kJ/mol and max=37.18; the paper's real
  `DeltaGc_310_gpcr9i` has mean=4.14, max=18.4 -- ours runs roughly 3x
  hotter. A row-max-based reconstruction of `DeltaGc` from the paper's
  own `CouplingMat` correlates only r=0.86 with their real `DeltaGc`,
  confirming even the *aggregation* (not just the scale) isn't an exact
  match to whatever their real formula does.
- **Block-partition mismatch, found while trying to align the two
  matrices for direct comparison.** Our own geometric (DSSP-free)
  blocking produces 75 blocks on gpcr9i; the paper's own bundled
  `BlockDet_gpcr9i` has 76, and the two partitions diverge as early as
  block 9 (paper: sizes 3,3; ours: 2,4). This is a real, previously
  undetected discrepancy in this pipeline's own block-partition logic
  relative to the paper's own reference blocking, on their own
  structure -- discovered as a side effect of this validation, not
  something this pipeline's own Tier-1 Tm regression would have caught
  (that only checks the resulting Tm, not block-by-block agreement).

**Verdict**: the fc fidelity gate failure is not a tunable-constant
problem and cannot be closed by adjusting `DEFAULT_FC_THRESHOLD_KJ_MOL`
alone (verified directly, not assumed). It traces to two compounding,
real mismatches against the paper's own ground truth: this pipeline's
block partition doesn't exactly reproduce the paper's own reference
blocking even on their own structure, and `coupling_free_energy` itself
runs at roughly 3x the paper's real coupling scale with imperfect
correlation. Both would need real, open-ended fixes (reconciling the
geometric SS/blocking heuristic against the paper's real block
boundaries; auditing the coupling free-energy formula itself, which the
paper's methods this repo doesn't include the source for) -- not a
threshold tune. This is a legitimate, decisive, non-circular negative
result, obtained the same way the Gate A pdie finding was: by testing
directly against real, independent ground truth rather than curve-fitting
a knob. Combined with the previous section's truncation-vs-fc tension,
**fc as currently implemented remains unusable as a cross-node
cooperativity metric**, and the ancestral tree should still not be run
on it. The path forward is either (a) a real block-partition and
coupling-formula audit against the paper's bundled reference data (real
method-development work, comparable in scope to the Gate A buried-
carboxylate problem), or (b) dropping fc entirely in favor of the raw
continuous coupling matrix / mean|coupling free energy| for cross-node
comparison, which this pilot and the previous section both already show
carries a real, substantial, non-saturated cross-node signal on its own
-- unvalidated against the paper's absolute scale, but usable for
*relative* comparison across consistently-processed nodes, which is
what the evolutionary-cooperativity question actually needs.

## Block-partition and coupling-formula audit: DSSP is a real, adoptable
## fix for blocking; the coupling formula itself is provably correct;
## the residual gap is upstream, in contact-map/parameter fidelity

Executed the audit the previous section's verdict called for. Two
independent halves, both resolved with hard evidence rather than
inference.

**Block partition: real DSSP is a genuine, measurable improvement.**
`mkdssp` (v4.2.2) is installable in this sandbox (`apt-get install
dssp`) -- not something noted as available before. Ran it on the real
`gpcr9i.pdb` (needed a synthetic `HEADER` line; the file's CASP-style
header isn't valid legacy PDB, and Biopython's own mmCIF re-export
lacked the chain/entity metadata modern `mkdssp` requires -- both
workarounds documented in scratchpad). Compared three block partitions
against the paper's own real `BlockDet_gpcr9i` ground truth:
- Our geometric (DSSP-free) heuristic: 75 blocks (paper: 76), 71/287
  (24.7%) adjacent-residue block-boundary disagreements.
- Real DSSP (H/G/E only, matching this module's own documented HEG
  convention): **76 blocks -- exact match** to the paper's count, 50/287
  (17.4%) boundary disagreements -- a real ~30% relative reduction, not
  a wash.
- The residual 17.4% gap even with real DSSP most likely reflects
  STRIDE-vs-DSSP algorithmic differences (the original MATLAB pipeline
  used STRIDE, not DSSP -- different H-bond geometry criteria, a known,
  real source of disagreement on ambiguous helix caps/3-10 helices) and
  possibly small structure-preparation differences from the paper's
  exact original file, not a bug in either program.

This is a genuine, concrete, adoptable finding independent of the fc
question: this pipeline should offer a real-DSSP-backed SS assignment
path when `mkdssp` is available (already partly supported via
`secondary_structure_from_codes`, just needs a DSSP-output parser and
its own dependency plumbing), rather than relying solely on the
geometric heuristic. Not yet wired into the pipeline itself -- this
audit ran it standalone against gpcr9i only, as a diagnostic.

**But DSSP-based blocking does NOT fix the coupling-value mismatch --
if anything it looked worse under direct comparison.** Recomputed
`coupling_free_energy` using the DSSP-derived block partition (now
exactly 76 blocks, matching the paper's own count) and compared directly
against the paper's real `CouplingMat_310_gpcr9i` (aggregated to
block-level via their own `BlockDet`, now a valid 1:1 comparison since
block counts finally match): mean|value| = 13.77 kJ/mol (paper's real
mean = 3.82) and Pearson r = 0.384 -- *worse* than the earlier, cruder
row-max-based comparison under the mismatched geometric-heuristic
partition (r=0.86, though that number was already flagged as not a
confirmed match to the paper's real aggregation method, so may not be
directly comparable). Either way, exact block-count agreement did not
bring the coupling values into line. **The block-partition mismatch was
not the dominant driver of the coupling-scale/correlation problem.**

**Coupling formula: verified correct, two independent ways.**
1. Direct comparison against the real MATLAB source (`FesCalc_Block_full.m`,
   bundled in the reference repo -- the actual formula, not reconstructed):
   `chipluswt=RT*log(pjfkf./pjukf)`, `chiminuswt=RT*log(pjfku./pjuku)`,
   `dGwtcp=RT*log((pjuku.*pjfkf)./(pjfku.*pjukf))`. This pipeline's
   `coupling.py` computes `chi_plus=RT*log(FF/UF)`,
   `chi_minus=RT*log(FU/UU)`,
   `coupling_free_energy=RT*log((UU*FF)/(FU*UF))` -- an exact structural
   match (FF=pjfkf, UF=pjukf, FU=pjfku, UU=pjuku), confirmed by reading
   the real source, not assumed.
2. Independent bit-exact cross-check: the coupling module's own marginal
   `p_folded` (derived from its joint-probability engine, `diag(FF)`) was
   compared against an ensemble-averaged P(folded) computed independently
   from `run_wsme`'s own (already brute-force-validated) `fpath`/`fes`
   output, via `sum_n P(n) * fpath[n, block]`. **max|diff| = 0.000000** --
   bit-exact agreement. This rules out a bug in the O(nblocks^2)
   rectangle-region joint-probability accumulation (`_accumulate`,
   `diff_ff`/`diff_fu`/`diff_uu`): if that combinatorial engine were
   wrong, it's extremely unlikely its marginal would still exactly match
   a completely independently-computed quantity.

**Verdict**: both audited components -- the coupling formula and the
joint-probability engine that feeds it -- are now proven correct against
real, independent ground truth, not merely assumed. The block partition
has a real, adoptable improvement available (DSSP) but fixing it alone
does not close the coupling-value gap. By elimination, the remaining
~3-4x scale mismatch and imperfect correlation against the paper's real
numbers must trace to **upstream inputs feeding the (correct) formula**
-- most likely the contact map construction (vdW cutoff, electrostatics
treatment) and/or the entropy/energy parameters (DS, DDS, DCp,
dielectric) that determine `compute_block_zvec`'s microstate weights,
which this pipeline built independently rather than porting from the
paper's full original source (only the final coupling-formula step and
the DSC/Cp formula were available to port directly; the paper's own
contact-map and blocking preprocessing pipeline is not included in this
repo's 4 bundled `.m` files). This is consistent with, and now directly
implicates the same category of gap as, every other cross-validation
finding this session (block-partition fidelity, Tier-1 Tm mismatches on
4 of 5 reference receptors, the fc saturation itself) -- this pipeline's
independently-built geometry-to-energetics pipeline is close to but not
bit-identical to the paper's original, and the coupling matrix is simply
the most sensitive downstream consumer of that accumulated gap. Closing
it fully would require either the paper's original contact-map/blocking
source (not available) or a systematic parameter-by-parameter
reconciliation against real reference outputs -- real, open-ended
work, appropriately out of scope for this audit's mandate (confirm
whether the coupling module itself has a bug -- it does not).

**Practical recommendation, given this**: do not keep chasing fc's
absolute scale. Use the raw coupling matrix / mean|coupling free energy|
for cross-node *relative* comparison, exactly as the previous section
already concluded, now with stronger justification -- the module
computing those numbers is verified correct, so the numbers reflect real
model output, not a software bug; they are just not on the paper's exact
absolute scale, for reasons now well characterized (upstream input
fidelity) rather than mysterious.

## DSSP wired into the pipeline as an opt-in SS-assignment source

The audit above found real DSSP a genuine, adoptable improvement,
independent of the fc question -- wired it in rather than leaving it a
standalone scratchpad diagnostic.

`wsme_gpcr/secondary_structure.py` gained `secondary_structure_from_dssp
(pdb_path, structure)`: runs `mkdssp`, parses its legacy-format output
into `{author_resnum: ss_code}` filtered to the structure's own
`chain_id`, and maps to the same H/G/E-structured boolean convention as
`secondary_structure_from_codes`. Two real, non-hypothetical robustness
issues handled, both hit while building this: (1) `mkdssp` 4.x rejects
legacy PDB files without a valid `HEADER` record (the GPCR-Landscapes
reference PDBs are CASP-style and lack one) -- a synthetic `HEADER` is
prepended in a temp copy, original file untouched; real mmCIF input
(mkdssp's native format) passes through unmodified. (2) Chain-break
sentinel rows (`!`) are skipped rather than parsed as residues. Raises
`DsspNotAvailableError` (never silently falls back to the geometric
heuristic) if `mkdssp`/`dssp` isn't on PATH.

Threaded `use_dssp: bool = False` through `run_pipeline`,
`run_pipeline_multi_ph`, `run_alanine_scan_pipeline`, and
`run_alanine_scan_pipeline_multi_ph` (same priority order everywhere:
explicit `ss_codes` > `use_dssp` > geometric default), and added a
`--use-dssp` CLI flag (errors clearly if combined with
`--ss-codes`/`--ss-file` rather than silently picking one). Verified via
CLI smoke test on the real gpcr9i structure: reports "76 blocks" --
the paper's own exact count, matching the audit's finding.

11 new tests (`tests/test_dssp.py`): DSSP-output parsing (chain
filtering, chain-break exclusion, missing-header-row error) against a
synthetic DSSP text block, so most of the suite doesn't require
`mkdssp` to be installed; the `HEADER`-patching logic (cif passthrough,
already-valid-PDB passthrough, missing-header gets fixed without
mutating the original file); `DsspNotAvailableError` when the binary is
missing (mocked, not gated on absence); and `@pytest.mark.skipif`-gated
real-`mkdssp` end-to-end tests confirming the exact 76-block/242-
structured-residue result the audit found on gpcr9i, plus that
`use_dssp=True` produces a different partition than the geometric
default and that `use_dssp=False` never even probes for `mkdssp`. Full
suite: 254 passed (243 before this session's DSSP work + 11 new).

README updated to point at `use_dssp`/`--use-dssp` right next to the
existing "No STRIDE dependency" section, with the audit's measured
numbers (76 vs. 75 blocks, 17.4% vs. 24.7% boundary disagreement)
instead of a vague "prefer if available."

## fc definition corrected: it was never a scale-mismatch problem --
## `compute_fc` was implementing the wrong definition entirely

The user asked to resolve the "3-4x scale mismatch" via the paper
itself. Found and read the real paper (`598026cf-s4146702335790z.pdf`,
uploaded by the user; text-extracted via `pdftotext -layout` after
`apt-get install poppler-utils`): Anantakrishnan & Naganathan,
"Thermodynamic architecture and conformational plasticity of GPCRs,"
Nat Commun 14:128 (2023), doi:10.1038/s41467-023-35790-z -- the exact
paper `wsme_gpcr`'s presets already cite. This resolved the question
directly rather than by further inference from bundled `.mat` data, and
the answer was not what the framing assumed.

**The paper's real fc procedure (Methods/Results, verbatim formulas
transcribed from the PDF):**
```
DG+ = RT ln( sum(p_if_jf) / sum(p_iu_jf) )
DG- = RT ln( sum(p_if_ju) / sum(p_iu_ju) )
DGc = DG+ - DG-
```
"The mean effective coupling free energy... residue-averaged coupling
free energies <DeltaGc>, i.e., averaging along the dimensions of the
symmetric matrices... were Z-scored to account for intrinsic
differences in the range of coupling free energies, and residues that
exhibit a Z-score greater than one were labeled as strongly coupled.
The fraction of strongly coupled residues (fc)..."

This is **not** "does any block pair's raw coupling free energy exceed
an absolute threshold" (what the old `compute_fc` implemented, guessing
1 RT ~= 2.58 kJ/mol since the real definition wasn't known at the time).
It is: (1) average each block's coupling to every partner (a per-block
scalar, <DeltaGc>_i); (2) Z-score that vector **within each receptor,
using that receptor's own mean/std**; (3) threshold the Z-score at >1.
Step 2 is the part that was missing entirely, and it's the part that
matters: a Z-score is invariant to uniform rescaling, so it doesn't
care whether a given receptor's raw coupling values run 3x hotter or
colder than another's. **The "3-4x scale mismatch" was never actually
the problem for fc** -- fc was just computing the wrong statistic, one
that was never going to be scale-invariant regardless of how well the
upstream contact map/energy parameters were reconciled to the paper's
original pipeline.

**Independent confirmation the formula reading is correct, before
touching any code:** the bundled `.mat` files' `DeltaGc_310_<tag>`
variable was confirmed to be exactly the row-mean (not row-max, which
had been assumed earlier and only gave r=0.86) of `CouplingMat_310_<tag>`
-- direct comparison on gpcr9i gives r=0.9998, mean|diff|=0.34 kJ/mol
(residual likely from a minor weighting difference, e.g. per-residue vs.
per-block averaging). Applying the Z-score/threshold-at-1 procedure to
the paper's own real `DeltaGc_310` across all 53 usable receptors
reproduces their reported 13.0 +/- 4.5% almost exactly: **13.29 +/-
4.27%** measured. This is not circular -- it uses the paper's own
ground-truth numbers, not anything this pipeline computed.

**Fixed `compute_fc` in `calibration.py`** to implement this real
procedure: per-block row-mean of `coupling_free_energy` -> Z-score
(population std, i.e. `ddof=0`, matching the 13.29/4.27% reproduction
above) -> threshold at `z_threshold` (default `DEFAULT_FC_Z_THRESHOLD =
1.0`, replacing the old `DEFAULT_FC_THRESHOLD_KJ_MOL` absolute-kJ/mol
guess). A block whose every partner is NaN is excluded from the
Z-scoring population entirely (not coerced to Z=0), and a receptor with
zero coupling-value variance (every block equally coupled) correctly
gives fc=0.0 -- Z-scoring against zero variance is undefined, which is
a real, different answer than a fixed-threshold definition would give
("everyone qualifies" -> 1.0), now covered by
`test_compute_fc_uniform_coupling_gives_zero_not_one`.

**Result on real reference receptors** (`use_dssp=True`, paper's own
reported ene, corrected `compute_fc`, no threshold tuning):

| Receptor | our fc | paper's real fc (this receptor) | diff |
|---|---|---|---|
| gpcr9i (4DKL) | 12.15% | 12.15% | 0.00 |
| gpcr1i (1U19) | 11.21% | 9.77% | +1.44 |
| gpcr20i (5LWE) | 13.42% | 15.10% | -1.68 |
| gpcr13a (6OS9) | 18.40% | 15.95% | +2.45 |
| gpcr2i (2LNL) | 14.86% | 9.46% | +5.40 |

Compare to the old implementation's 92-100% on every single receptor
(a 7-8x error). The residual few-percentage-point gaps are consistent
with, and fully explained by, the already-characterized block-partition
audit (DSSP vs. the paper's own STRIDE, ~17% residual boundary
disagreement) -- there is no remaining mystery-scale problem to chase.

**5 new tests** (`tests/test_calibration.py`) validate this directly
against the paper's own real per-receptor fc values on the 5 already-
committed reference structures (`@pytest.mark.parametrize`, tolerance
7 percentage points -- the measured real spread, not a loosened margin),
plus 4 rewritten synthetic tests for the Z-score semantics (including
the zero-variance edge case above). Full suite: 259 passed (254 before
this entry + 5 new).

**The ancestral pilot's actual blocker is resolved.** Re-ran fc (now
corrected, `use_dssp=True`) on the three ancestral nodes that fold
properly (node_20/148/80; node_34 still excluded, still collapsed, per
the existing guardrail): **12.10%, 10.85%, 8.30%** -- a real ~4-point
spread, comparable in scale to genuine cross-receptor variation in the
paper's own 45-receptor dataset (std 4.5%), not saturated at 97-100%.
fc can now actually discriminate between nodes. This does not by itself
answer the evolutionary-cooperativity question, but it removes the
specific, concrete blocker ("fc cannot rank nodes") that stopped the
tree from running.

**One separate, unrelated issue surfaced while re-running
`examples/calibration_regression.py` to validate this end-to-end**:
Tier 2's blind Brent-search calibration (`calibrate_xi_tm_mode`) now
raises `CalibrationError` for both `gpcr1i` and `gpcr9i` under the
geometric-heuristic default blocking (no resolvable Cp(T) peak at one
bracket edge). This is unrelated to fc -- confirmed directly, since the
fc validation above bypasses Brent search entirely by using the paper's
own already-known ene values -- and the regression script wasn't
catching `CalibrationError` per-receptor, so it crashed instead of
reporting a clean FAIL (fixed minimally, just the exception handling,
so Tier 2 now reports rather than crashes). Whether this Brent-search
bracket fragility is new or pre-existing wasn't established; it's a
separate, real issue for a future session, not part of this fc fix.

**Net effect on the "study the paper" ask**: reading the actual Methods
section resolved this in one pass, in a way that continued reverse-
engineering against bundled `.mat` outputs could not have -- the paper
states the Z-score procedure explicitly, in words, and no amount of
threshold-sweeping against `DeltaGc` alone (which is what the "solve the
scale mismatch" framing was aimed at) would have discovered that the
real fix was a missing normalization step, not a missing calibration.
The Methods section also independently confirmed several already-correct
parameter choices (vdW cutoff 5 Å, dielectric 4, DS=-10 J/mol/K,
DCp=-0.36 J/mol/K/contact all match this port already) and flagged one
apparent discrepancy: the paper's prose states DDS (excess coil entropic
penalty) as a clean -6.1 J/mol/K, while this port's `WSMEParams.DDS` is
6.0606e-3 kJ/mol/K.

**Chased and resolved: a false alarm, confirmed directly from the real
MATLAB source rather than the paper's rounded prose.**
`FesCalc_Block_full.m` line 31 (bundled in the reference repo): `DDS=
6.0606;` -- the actual, active constant the authors' own code runs,
byte-identical to this port's value. Even more tellingly, line 70 of the
same file has a **commented-out** (disabled) alternative:
`%zjj(disr2)=exp((DS-(6.1/1000))./R);` -- direct evidence the authors
themselves rounded their real 6.0606 constant to 6.1 for the paper's
prose (and apparently considered, then didn't ship, a version using the
rounded value). This port already matches the authoritative source
exactly; no bug, no fix needed.

## Ran the ancestral tree with both fixes in place: only 1 of 4 nodes
## survives -- a real result, not the multi-node comparison hoped for

Re-ran the four ancestral nodes end to end through the now-fixed
pipeline (real DSSP blocking + corrected Z-score fc), reusing the
shared-truncation-range (res 16-285) core structures already built by
the earlier concurrent-session pilot (`asr_pilot/node_*_core.cif`,
predates this entry's fixes) rather than rebuilding, but recomputing
every downstream quantity fresh -- the old `.npy` coupling matrices in
that directory predate both the DSSP wiring and the fc fix and were not
reused.

**Result: dramatically different from the earlier pilot's (geometric-
heuristic) numbers.**

| Node | nblocks (DSSP) | default-xi min | folded_ok (>=85%) |
|---|---|---|---|
| node_20 | 71 | 42/71 (59.2%) | **NO** |
| node_148 | 73 | 66/73 (90.4%) | yes |
| node_80 | 71 | 12/71 (16.9%) | **NO** |
| node_34 | 70 | 7/70 (10.0%) | **NO** (already known) |

Under the geometric heuristic, node_20 and node_80 had folded at 92-94%;
under real DSSP blocking, they collapse to 59.2% and 16.9%. Only
node_148 still folds properly. This is not a DSSP-parsing artifact --
checked directly: DSSP's structured-residue counts on all four nodes
(206-213/270) are sane and consistently, modestly lower than the
geometric heuristic's (238-241/270), the same direction and rough
magnitude of difference already characterized on real reference
receptors, not an outlier or a sign DSSP is failing on these AlphaFold-
predicted structures.

**Per the existing guardrail (same one applied to GPR68-active), coupling/
fc was only computed for node_148**, the one node whose landscape found
its own reference fold: fc=13.70%, mean|coupling|=2.67 kJ/mol (at the
default, uncalibrated xi=-48.2 J/mol -- node_148's own Tm-calibration
also failed to resolve a Cp(T) peak in the physically valid bracket, a
separate CalibrationError, so this is not a Tm-matched number). fc=13.70%
sits squarely inside the paper's own real 13.0+/-4.5% receptor
distribution -- by itself a reassuring plumbing-fidelity signal for this
one node, not a comparative result.

**Honest bottom line: this is not the multi-node cooperativity
comparison the research question needs.** With only one node clearing
the fold-quality bar, there is nothing to compare across nodes from
this specific 4-node example set -- fc/coupling values exist for
exactly one of them. This is a real, if sobering, consequence of doing
the validation properly: the more accurate the blocking (DSSP over the
geometric heuristic) and the more correct the fidelity gate (Z-scored fc
over the old saturated absolute threshold), the smaller the set of
ancestral nodes that can currently support a trustworthy comparison
shrinks from 3/4 to 1/4. That is the honest result of running the tree
with the fixes in place, not a new bug to chase.

**What this means going forward, options, none pursued yet without
further direction:**
1. This 4-node example set is almost certainly too small to draw any
   real evolutionary conclusion from regardless -- a genuine ancestral-
   tree run needs many more nodes so a 1-in-4 fold-quality attrition rate
   doesn't zero out the comparison entirely by chance.
2. Whether node_20/80's collapse under DSSP blocking is itself
   biologically meaningful (these specific ancestral reconstructions
   being genuinely less thermodynamically robust/cooperative under this
   model) or an artifact of AlphaFold prediction quality at these nodes
   specifically has not been investigated -- would need the same kind of
   real-structure cross-check that resolved the GPR68 active-state
   question (not available here; there is no experimentally solved
   structure for any internal ancestral node, by definition).
3. A real run would need this fold-quality attrition rate itself
   characterized across many more nodes before treating any single
   node's coupling matrix as informative -- if most ancestral
   reconstructions fail this bar, that ratio is itself a finding worth
   reporting, not just a nuisance to filter past.

## Real IQ-TREE2 ASR posterior data checked against the fold-quality
## pattern -- partially consistent, but does not cleanly explain it

User supplied the real IQ-TREE2 outputs: `alignment_iqtree_asr.state`
(per-site posterior probabilities, all 20 amino acids, for every one of
162 internal nodes -- confirms Node20/148/80/34 naming matches this
session's node_20/148/80/34 directly) and `alignment_iqtree_asr.iqtree`
(164 sequences, 323 sites, model Q.PLANT+R7). This is exactly the data
the previous section's caveat asked for.

**Per-node MAP-posterior confidence (mean over all 323 sites):**

| Node | mean MAP posterior | frac sites <0.8 | frac sites <0.5 | fold outcome (DSSP) |
|---|---|---|---|---|
| node_20 | **0.942** (highest) | 10.5% | 2.2% | **collapsed** (59.2%) |
| node_148 | 0.865 | 24.1% | 11.5% | **folded** (90.4%) |
| node_80 | 0.829 | 33.4% | 14.6% | collapsed (16.9%) |
| node_34 | 0.821 (lowest) | 34.4% | 13.3% | collapsed (10.0%) |

**Partially consistent, but not a clean explanation.** node_80 and
node_34 -- the two most severely collapsed nodes -- do have the lowest
reconstruction confidence, consistent with the "low-confidence chimeric
sequence" hypothesis. But node_20 breaks the pattern outright: it has
the *highest* confidence of all four (94.2% mean posterior, fewest
ambiguous sites) yet is the node whose fold quality moved the most
between blocking methods (92-94% under the geometric heuristic -> 59.2%
under real DSSP). If reconstruction uncertainty alone explained the
fold-quality pattern, node_20 should have been the most robust, not a
collapse case. It isn't.

**Follow-up check: do ambiguous sites concentrate at structurally
important (high-contact) positions?** Cross-referenced each node's
per-site MAP posterior against its own folded structure's per-residue
contact density (DSSP-blocked core, res 16-285). No clear enrichment:
ambiguous-site (posterior <0.8) mean contact count is essentially the
same as confident-site mean contact count for every node (e.g. node_20:
273.3 vs 273.1; node_148: 232.9 vs 269.6, if anything slightly lower for
ambiguous sites, the opposite of "ambiguity concentrated at critical
packing positions"). This check does not support a simple "uncertain
sites happen to sit at structurally critical spots" story either.

**Honest interpretation**: ASR uncertainty is a real, partial
contributor (it's consistent with node_80/34's collapse) but is not,
by itself, a sufficient explanation for the full pattern -- node_20's
behavior looks more like genuine sensitivity of that specific sequence's
folding cooperativity to the exact block-partition choice than an
artifact of a chimeric reconstruction. These are not mutually exclusive:
a real ancestral protein can still be genuinely more marginal/boundary-
sensitive in its cooperativity than others, same as real extant
proteins are.

**What would actually settle this -- and a real tool limitation**: the
standard, decisive test is an "AltAll" reconstruction (swap ambiguous
sites to their second-most-likely residue, re-fold, re-run WSME; if the
collapse persists under AltAll too, that's evidence toward real
biology, not reconstruction artifact). **This cannot be completed in
this sandbox**: building an AltAll sequence from the `.state` file is
straightforward, but there is no local AlphaFold/ColabFold access here
to actually fold it into a new structure -- the four structures in hand
were generated externally by the user, not by this session. Stating
this plainly rather than approximating around it: any true AltAll test
needs external folding.

**A real, achievable partial substitute, not yet built**: an in-silico
point-mutation-on-fixed-backbone sensitivity test at the ambiguous
sites, reusing/generalizing the existing `alanine_scan.py` machinery
(already built for exactly this kind of "mutate a residue, keep the
backbone, recompute the contact map/WSME landscape" analysis, just
currently hardcoded to alanine substitutions). This would not capture a
genuine backbone conformational response to the alternate sequence (a
real limitation relative to true AltAll refolding), but would directly
test whether swapping in the second-most-likely residue at node_20's 18
core-region ambiguous sites (or node_80/34's 71/75) measurably shifts
the WSME fold outcome -- a real, if partial, robustness signal,
achievable with tools already in this repo. Not built yet, pending
direction.

## Point-mutation sensitivity proxy: built and run -- node_20's collapse
## is substantially an ASR-uncertainty artifact; node_148's success is
## robust to it

Built the proposed proxy: for each node, every ambiguous core-region
site (`.state` MAP posterior <0.8, ALA/GLY/PRO excluded per the existing
`EXCLUDED_FROM_SCAN` convention) is truncated to alanine
*simultaneously* (reusing `alanine_scan.py`'s `alanine_exclude_mask` +
`compute_contact_map(exclude_atoms=...)` + `build_blocks`, unmodified),
and the resulting WSME landscape's fold fraction is compared to the
unperturbed baseline. This tests "how much do specifically the
positions ASR is uncertain about matter to the fold outcome" -- not a
true identity swap (no rotamer placement, still no real refolding), but
a real, honest, directly informative signal, explicitly scoped as such.

**Results** (default xi=-48.2 J/mol, DSSP blocking, same core structures
as the tree run):

| Node | n ambiguous positions truncated | WT fold | mutant fold | delta |
|---|---|---|---|---|
| node_20 | 16 (of 270, 5.9%) | 59.2% | 15.5% | **-43.7 pp** |
| node_148 | 44 (16.3%) | 90.4% | 90.4% | **+0.0 pp** |
| node_80 | 66 (24.4%) | 16.9% | 12.7% | -4.2 pp |
| node_34 | 71 (26.3%) | 10.0% | 4.3% | -5.7 pp |

**This resolves the node_20 puzzle from the previous section, and
flips the earlier tentative read.** node_20 has the *highest* average
ASR confidence of the four (0.942 mean posterior) but only 16 truly
ambiguous core positions -- and truncating exactly those 16 collapses
its fold fraction by 43.7 percentage points, by far the largest effect
of any node, on the fewest perturbed positions. High *average*
confidence was the wrong summary statistic to look at; what matters is
whether the *few* uncertain positions happen to be load-bearing, and for
node_20 they clearly are. This is real, direct evidence that node_20's
collapse under DSSP blocking is substantially entangled with genuine
sequence-identity uncertainty at a small number of specific sites --
not necessarily a stable property of "this ancestral protein," and not
safe to report as real evolved biology without an actual AltAll refold
(still not possible in this sandbox, see previous section).

**node_148, by contrast, is completely insensitive to its own 44
ambiguous positions** (zero change). Its "folds properly, fc=13.70%"
result does not depend on how ASR called its uncertain sites -- this is
real, positive evidence that node_148's result is robust and can be
treated as the more trustworthy of the four, not merely the one that
happened to clear an arbitrary threshold.

**node_80/34's collapse looks more diffuse/structural, not concentrated
in a few uncertain sites** -- truncating a much larger fraction of their
sequence (24-26% of positions, vs. node_20's 6%) produces comparatively
modest further degradation (-4 to -6 pp) on top of an already-severe
baseline collapse (10-17%). Their poor fold quality is not something a
handful of different ASR calls would likely rescue; whatever is wrong
looks more pervasive across the sequence than node_20's problem does.
This doesn't rule out reconstruction quality as a contributor (they also
have the lowest average confidence of the four), but the mechanism looks
different in kind from node_20's -- broadly weak rather than a few
critical positions.

**Updated bottom line for a future full tree run**: node_148-style
robustness (fc/fold outcome insensitive to the node's own ambiguous
sites) should be treated as a real requirement before trusting any
node's coupling result, not just the fold-fraction threshold already in
use -- this sensitivity check is cheap (no new folding needed) and
directly separates "real signal" (node_148) from "sequence-uncertainty
artifact" (node_20) in a way the fold-fraction number and the average
posterior confidence number both individually missed.

## Promoted the sensitivity check to reusable library code: `wsme_gpcr/asr.py`

Moved the scratchpad script into a real, tested module rather than
leaving it as a one-off. New public API (exported from `wsme_gpcr`):

- `parse_iqtree_state_file(path) -> dict[str, NodePosteriors]` -- pure-
  Python TSV parsing of an IQ-TREE2 `.state` file (no pandas dependency
  added; the file is small and simply structured). Also computes each
  site's second-most-likely state/posterior, needed for a real future
  AltAll build even though this module doesn't refold.
- `ambiguous_core_resnums(node_posteriors, structure, posterior_threshold=0.8) -> list[int]`
  -- maps alignment sites to a structure's own `author_resnum` (via
  `site_to_resnum`'s cumulative-non-gap-count convention) and returns
  the mutable (non-ALA/GLY/PRO), in-structure, low-confidence positions.
- `run_asr_sensitivity_check(structure, block_model, ss_mask, params, ambiguous_resnums, delta_tolerance_frac=0.10) -> AsrSensitivityResult`
  -- the core computation: truncate the given positions to alanine
  simultaneously, compare WSME fold fraction to baseline. Reusable with
  any resnum list, not tied to ASR (e.g. usable for any "how sensitive
  is this fold to these specific positions" question).
- `evaluate_node_trustworthiness(pdb_path, node_posteriors, params=None, use_dssp=True, ...) -> AsrSensitivityResult`
  -- single-call convenience wrapper (load, fold, identify ambiguous
  sites, run the check) for evaluating one node end to end.
- `AsrSensitivityResult.trustworthy` (bool) = `fold_ok AND sensitivity_ok`,
  plus `.reason()` for a one-line human-readable classification.

10 new tests (`tests/test_asr.py`): `.state` parsing (including sorting
by site and a missing-header error), the resnum-mapping convention
(explicitly covers the case a structure's own numbering doesn't start
at 1 -- CI2 starts at resnum 19, which the first draft of this test
initially got wrong before being caught and fixed), threshold behavior,
the trivial-when-no-ambiguous-positions case, real CI2 plumbing, and the
three `.reason()` message variants. Full suite: 269 passed (259 before +
10 new).

**Validated the promotion is faithful, not just "looks similar"**: reran
all four real nodes through `evaluate_node_trustworthiness` and diffed
against the original scratchpad script's numbers directly, position-set
included, not just the final percentages -- exact match (WT/mutant fold
fractions, deltas, and the full ambiguous-resnum sets are identical; an
apparent small count discrepancy noticed in one node during this check
turned out to be noise from an earlier, less careful scratch comparison,
not a real difference, confirmed by a direct side-by-side rerun of both
code paths against the same inputs).

## Final trustworthiness list for the four available ancestral nodes

Using the now-reusable `evaluate_node_trustworthiness` (real DSSP
blocking, default xi=-48.2 J/mol, posterior_threshold=0.8,
delta_tolerance_frac=0.10):

| Node | fold_ok | sensitivity_ok | **trustworthy** | Reason |
|---|---|---|---|---|
| **node_148** | yes (90.4%) | yes (0.0pp / 44 positions) | **YES** | Folds properly and is robust to its own ASR ambiguity. The only node whose fc (13.70%, computed earlier) should be treated as real signal. |
| node_20 | no (59.2%, below the 85% bar) | no (-43.7pp / 16 positions) | NO | Fails to fold outright, and the failure is substantially attributable to a handful of specific ASR-ambiguous positions -- likely a reconstruction-uncertainty artifact rather than a stable property of the ancestral protein. Would need a real AltAll refold (unavailable in this sandbox) before this node could be used at all. |
| node_80 | no (16.9%) | yes (-4.2pp / 64 positions, i.e. its collapse is NOT concentrated in the ambiguous sites) | NO (fails the fold gate regardless) | Fails to fold; unlike node_20, the failure is not explained by its ambiguous positions specifically -- it looks like a more diffuse/structural problem, not primarily an ASR artifact. Still carries the lowest-tier average reconstruction confidence of the four (0.829 mean posterior), so real biology vs. reconstruction quality remains open. |
| node_34 | no (10.0%) | yes (-5.7pp / 70 positions) | NO (fails the fold gate regardless) | Same profile as node_80: diffuse collapse, not concentrated in ambiguous sites, lowest average confidence of the four (0.821). |

**One node out of four (node_148) is currently usable.** This is the
honest deliverable of the whole investigation this session: not a
cooperativity comparison across ancestral nodes (which needs at least
two trustworthy nodes and this 4-node example set provides one), but a
validated, reusable, two-gate method for determining *which* nodes in a
larger tree would be usable, with a concrete, real demonstration that
the naive fold-fraction number alone is not sufficient (node_20 shows
why) and that average ASR confidence alone is not sufficient either
(node_20 again -- highest confidence, still an artifact-driven failure).
Running this against the full 162-node tree the `.state` file covers,
not just these four example structures, is the direct next step toward
an actual evolutionary-cooperativity comparison, and would need
AlphaFold structures for whichever additional nodes are worth folding
(prioritized by this same two-gate method once a first-pass triage is
possible, e.g. from sequence-level ASR confidence alone before spending
compute on structure prediction).

## Parsed the real tree topology: our one "trustworthy" node is actually
## the least load-bearing one in the entire tree

The `.iqtree` report's ML tree (`Tree in newick format:`, line 638) has
internal node labels in `NodeN/SH-aLRT/UFBoot` format, matching the
`.state` file's node names exactly and carrying real branch-support data
(SH-aLRT and ultrafast-bootstrap, 1000 replicates). Parsed with
`Bio.Phylo` (already a dependency): 164 tips, 162 internal nodes
(161 parsed with support values; the root/trifurcation point has none),
confirming this is the full tree the 4 supplied AlphaFold structures
were drawn from, not a different/partial one.

**Computed clade size (descendant tip count) for every internal node --
the direct measure of how "load-bearing" it is** (how much of the
tree's downstream diversity depends on this ancestor being correctly
reconstructed) **-- and looked up where our four already-evaluated
nodes rank:**

| Node | n_tips (of 164) | rank by clade size (of 161) | SH-aLRT | UFBoot | trustworthy? (prior entry) |
|---|---|---|---|---|---|
| node_20 | 137 | **3rd** | 84.9 | 56 | NO -- ASR-artifact-driven collapse |
| node_80 | 40 | **8th** | 99.8 | 100 | NO -- diffuse collapse |
| node_34 | 37 | **12th** | 64.1 | 95 | NO -- diffuse collapse |
| node_148 | 2 | **155th of 161** | 99.6 | 100 | YES |

**This reframes the whole result.** node_148 -- the only node that
passed both trustworthiness gates -- is the ancestor of just 2 closely
related extant sequences, essentially a shallow, recent split barely
distinguishable from an extant protein, not a deep ancestral
reconstruction. The three nodes that actually matter for an
evolutionary-cooperativity question -- node_20, node_80, node_34, which
between them are ancestral to 137, 40, and 37 of the tree's 164 taxa --
all failed. The "1 of 4 nodes usable" finding from the previous entry is
real, but it does not yet say anything about deep ancestral cooperativity;
it says something about a shallow, low-stakes node.

**Cross-referenced clade size against per-site ASR confidence
(`parse_iqtree_state_file`) for the top 20 nodes by clade size**, to
identify which large/important clades are worth prioritizing for future
AlphaFold folding:

| Node | n_tips | SH-aLRT | UFBoot | mean posterior | frac sites <0.8 |
|---|---|---|---|---|---|
| Node2 (near-root) | 162 | 70.7 | 86 | **0.971** | **5.3%** |
| Node19 | 145 | 83.3 | 58 | **0.967** | **7.1%** |
| Node20 (have) | 137 | 84.9 | 56 | 0.942 | 10.5% |
| Node21 | 136 | **100.0** | **100** | 0.779 | 41.8% |
| Node79 | 78 | 65.2 | 97 | 0.787 | 39.9% |
| Node22 | 58 | 99.8 | 100 | 0.750 | 45.2% |
| Node32 | 48 | 98.9 | 100 | 0.741 | 44.9% |
| Node80 (have) | 40 | 99.8 | 100 | 0.829 | 33.4% |
| Node34 (have) | 37 | 64.1 | 95 | 0.821 | 34.4% |

**A real, important, separate signal surfaced by this table: branch
support and per-site reconstruction confidence measure two different
things and can diverge sharply.** Node21 has *perfect* topological
support (SH-aLRT=100, UFBoot=100 -- IQ-TREE is completely certain this
clade is real) but 41.8% of its own reconstructed sites are ambiguous
(mean posterior only 0.779) -- a well-supported branch point does not
imply a confidently reconstructed ancestral sequence at that point.
Several other large, "well-supported" nodes (22, 32, 79) show the same
pattern (~40-45% ambiguous sites despite UFBoot>=97). Given node_20's
own demonstrated failure mode (a small number of ambiguous sites, not
even a large fraction, was enough to flip its fold outcome), nodes with
40%+ ambiguous sites should be expected, on priors, to fail the
sensitivity gate at least as often as node_20 did, if not more.

**Selected load-bearing candidates for future folding, prioritized by
the combination that predicts both scientific value and a real chance
of passing both trustworthiness gates:**

1. **Node2** -- the single best candidate: largest clade in the tree
   (162/164 tips, essentially the root of the whole receptor family) AND
   unusually high reconstruction confidence for a node this deep (97.1%
   mean posterior, only 5.3% ambiguous sites) -- a real, rare
   combination of maximal evolutionary importance and low expected
   reconstruction-artifact risk.
2. **Node19** -- second choice, same profile (145 tips, 96.7% mean
   posterior, 7.1% ambiguous).
3. Everything else in the top 20 by clade size shows a sharp confidence
   drop (mean posterior 0.74-0.87, 25-45% ambiguous sites) -- these
   remain scientifically important (they're the deep, large-clade nodes
   an evolutionary-cooperativity story actually needs) but should be
   treated as likely to need the full sensitivity-check workflow (and a
   real chance of failing it) rather than assumed foldable-and-trustworthy.

**No new structures were folded in this entry** -- Node2/Node19 are
recommendations for what to prioritize if/when more AlphaFold folding is
done, not new results. The concrete, actionable takeaway: of the four
nodes actually tested, the three that matter evolutionarily all failed,
and the sequence-only triage this entry performed (which needs no new
folding) identifies exactly two candidates in the entire 162-node tree
worth folding next with real expectation of a trustworthy result --
a far more targeted use of any future structure-prediction compute than
picking nodes without this check.

## Node2 and Node19 folded and evaluated: both fail too, and NOT as an
## ASR artifact -- 5 of 6 nodes now fail, a broader pattern than hoped

User supplied real AlphaFold Server output for the two recommended
candidates (`fold_2026_07_11_21_40_node_2.zip`, `fold_2026_07_11_21_41_node_19.zip`
-- full job packages: 5 ranked models + confidence JSONs + MSAs +
templates, not just a bare structure). Picked each node's own top-ranked
model by `ranking_score` (model_0 for both: 0.89 and 0.88 respectively,
consistent with AlphaFold's own ranking convention), confirmed same
323-residue/resnum-1-323 convention as the original four nodes, and
truncated to the identical shared core (res 16-285) for direct
comparability.

**Result, run through the same `evaluate_node_trustworthiness` used for
all prior nodes:**

| Node | nblocks | WT fold | sensitivity delta | fold_ok | sensitivity_ok | trustworthy |
|---|---|---|---|---|---|---|
| node_19 (Node19) | 71 | **5.6%** | +0.0pp (14 ambiguous) | NO | yes | NO |
| node_2 (Node2) | 71 | **57.7%** | +0.0pp (10 ambiguous) | NO | yes | NO |

Both fail the fold-quality gate -- node_19 severely (5.6%, close to
total collapse), node_2 less so (57.7%, still well short of the 85%
bar). **Neither failure is an ASR-uncertainty artifact**: both show
exactly 0.0 percentage points of sensitivity to their own ambiguous
positions, the same "robust, but robustly wrong" signature node_80/34
showed, not node_20's "artifact" signature. This is a real, different,
important result: these two nodes were specifically selected *because*
they combined the largest clade sizes in the tree with unusually high
per-site reconstruction confidence (97.1%, 96.7% mean posterior) --
exactly the profile that should have predicted success if reconstruction
quality were the dominant factor. It wasn't. High sequence-level
confidence did not translate into a foldable WSME cooperative landscape.

**Checked whether Tm-mode xi calibration rescues either -- no.** Same
`CalibrationError` failure mode as node_148 and the original 9BHM/GPR68
active-state finding: no resolvable Cp(T) peak anywhere in the
physically valid xi bracket, for both nodes. Recalibration is not the
fix here either.

**Updated full picture across all six evaluated nodes:**

| Node | clade rank (of 161) | fold_ok | sensitivity_ok | trustworthy |
|---|---|---|---|---|
| Node2 | **1st** (162 tips) | NO | yes (diffuse) | NO |
| Node19 | 2nd (145 tips) | NO | yes (diffuse) | NO |
| node_20 (Node20) | 3rd (137 tips) | NO | **NO (artifact)** | NO |
| node_80 (Node80) | 8th (40 tips) | NO | yes (diffuse) | NO |
| node_34 (Node34) | 12th (37 tips) | NO | yes (diffuse) | NO |
| node_148 (Node148) | 155th (2 tips) | **yes** | **yes** | **YES** |

**5 of 6 nodes fail. The one success remains the least evolutionarily
significant node tested.** This is a materially different, more
sobering picture than "just pick high-confidence, load-bearing nodes"
would have predicted -- the two nodes chosen specifically for that
combination both failed, and did so in the "diffuse, not an ASR
artifact" way that 3 of the other 4 failures also show. Only node_20's
failure has been shown to trace substantially to reconstruction
uncertainty; the other four failures (Node2, Node19, node_80, node_34)
look like a broader, still-unexplained pattern -- possibly something
about how this WSME pipeline (as currently parameterized, at the
untouched default xi, with Tm-calibration separately unable to resolve
an alternative) handles this specific GPR4-clade ancestral sequence
family generally, not a per-node reconstruction-quality issue. This
would need real investigation (e.g. checking whether ANY real, solved
GPR4-clade extant structure folds properly under this same pipeline, as
a control -- analogous to what the real 9BHM cryo-EM structure did for
the GPR68 active-state question) before drawing further conclusions.
Not yet done. The honest state of the evolutionary-cooperativity
question remains: one usable node, now confirmed via six real
data points spanning the tree, not four.

## The real-structure control ran -- and it substantially walks back
## the "5 of 6 ancestral nodes fail" conclusion, not confirms it

User supplied the full node_148 AlphaFold job package (previously only
had its `model_0.cif`). Its bundled template hits are real PDB
structures used by AlphaFold as structural templates -- and template
hit 0 is **4XNV**, which is not just any real GPCR structure: it is
literally one of the paper's own 45 reference receptors
(`gpcr14i` in the bundled `.mat` data, real reported ene=-46.6 J/mol,
Tm=333K). This is exactly the control proposed in the previous entry --
a real, experimentally solved, paper-calibrated structure from a
closely related lineage, run through this exact pipeline -- and it
surfaced something more important than a pass/fail verdict.

**4XNV folds excellently at the same default xi used throughout every
ancestral-node test** (-48.2 J/mol, rhodopsin's value): 97.4% (76/78
blocks). This is real evidence the pipeline itself is not fundamentally
broken for this receptor family -- a real, solved structure behaves
exactly as a well-folding structure should under the identical
untouched parameterization that failed for 5 of 6 ancestral nodes.

**But at the paper's own real calibrated xi for this exact structure
(-46.6 J/mol), the same real structure collapses to 5.1% (4/78).** A
fine xi sweep shows this is not a gradual, sensitive-but-continuous
relationship -- it is a genuinely sharp, near-discontinuous switch:

| xi (J/mol) | fold fraction |
|---|---|
| -58 to -48.2 | 97.4% (stable) |
| -47.5 to -44 | 3.8-5.1% (collapsed) |

The entire transition happens somewhere in a window smaller than 0.7
J/mol, between -48.2 and -47.5. **The paper's own reported calibrated
value for this receptor (-46.6) sits on the collapsed side of that
transition under this port's current (DSSP-based) blocking and
energetics** -- meaning this port does not precisely reproduce the
original MATLAB pipeline's fold behavior for this real receptor at its
own reported xi, a residual discrepancy beyond the already-characterized
~17% STRIDE-vs-DSSP boundary disagreement. Tm-mode calibration for
4XNV was also checked directly and **fails with the identical
CalibrationError as node_148, Node2, and Node19** ("no Cp(T) peak found
at all" at one bracket edge) -- the same failure mode, on a real,
solved, paper-validated structure, not just ancestral reconstructions.

**Why this matters more than a single control pass/fail:** it shows
this WSME implementation, post-DSSP-blocking, exhibits sharp,
near-discontinuous xi-dependent fold transitions for GPCR-scale block
counts -- a real, general property, not something specific to ancestral
sequences. A single fixed reference xi (this investigation used -48.2
throughout, since Tm-calibration systematically fails to resolve an
alternative) is therefore not a reliable, structure-independent test of
"does this fold" for *any* structure, real or ancestral -- whether a
given structure lands on the folded or collapsed side of -48.2 may
depend on exactly where its own sharp transition point happens to sit,
which this pipeline currently has no reliable way to locate (Tm-mode
calibration, the tool built for exactly this, fails broadly under DSSP
blocking, now confirmed on a real reference structure too, not just an
ancestral-node edge case).

**This substantially qualifies, without fully overturning, the "5 of 6
ancestral nodes fail" finding from the previous two entries.** The six
fold_ok results themselves are real, reproducible pipeline outputs, not
retracted. But the *interpretation* -- that this reflects something
about ancestral reconstruction quality or a GPR4-clade-specific pipeline
problem -- is now on much shakier ground. It may instead be substantially
explained by this more general, structure-independent phenomenon: many
GPCR-scale structures (real and ancestral alike) apparently sit close to
a sharp fold/collapse transition, and testing at one arbitrary fixed
point (because the tool meant to locate each structure's own transition
doesn't work reliably post-DSSP) will misclassify some fraction of
genuinely foldable structures as failures, essentially by chance of
which side of the switch -48.2 happens to land on for them.

**Concrete methodological fix identified, not yet implemented**: replace
single-point xi testing (`fold_ok` at xi=-48.2 alone) with a small
xi-window sweep -- e.g. across the paper's real inter-receptor spread
(-48.9+/-2.76 J/mol, or the wider bracket already used elsewhere in this
codebase) -- and classify a structure as capable of folding if it folds
*anywhere* physically plausible, not only at one fixed reference point.
This would need re-running all six already-evaluated ancestral nodes
(and ideally 4XNV/gpcr14i and a couple more real reference receptors as
controls) before the trustworthiness table can be treated as settled.
Not done in this entry -- flagging it as the necessary next step before
drawing further conclusions from the six-node table, rather than
quietly leaving that table standing as more authoritative than it now
deserves to be treated.
