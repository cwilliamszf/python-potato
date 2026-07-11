# GPR68 smoke test: findings on conformational sampling and PB pKa accuracy

Status: exploratory pipeline-validation work, not a validated result. **No
number in this document should be read as a real prediction about GPR68's
proton-sensing behavior.** Gate A (SNase buried-ionizable calibration) has
not been run — its dataset could not be sourced in this sandbox (see
"Gate A dataset sourcing" below) — so no PB pKa produced by this pipeline
is calibration-checked yet. This document exists to record what was
learned about the *pipeline's own behavior* at real-protein scale, since
that surfaced a genuine, previously-undetected methodological gap.

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

## Gate A dataset sourcing (separately blocked)

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
parameter tables. **This remains blocked on the user supplying the
dataset directly** (paste or upload) — it is not resolvable from within
this sandbox.

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

Full test suite: 181 passed as of this writing (`pytest` from the repo
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
3. Gate A calibration remains blocked on dataset access (see above).
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
