# GPCR Gibbs free energy estimator

`gpcr_gibbs_energy.py` estimates the Gibbs free energy of a protein structure
(e.g. a GPCR) directly from its 3D coordinates, using a real molecular
mechanics force field for the intramolecular forces plus a standard
statistical-thermodynamics treatment for entropy.

## What it computes

1. **Structure prep** (via [PDBFixer](https://github.com/openmm/pdbfixer)):
   converts nonstandard residues (e.g. MSE -> MET), strips heteroatoms
   (waters/ligands/lipids/ions) by default, completes truncated/disordered
   side chains and short missing loops, and adds hydrogens for the chosen pH.
2. **Force field**: AMBER `ff14SB` (bonds, angles, torsions, van der Waals,
   electrostatics) + `GBn2` implicit solvent (polar + nonpolar solvation),
   via OpenMM. This is what "accounting for all intramolecular forces" means
   concretely here: every bonded and non-bonded MM term is evaluated and
   reported separately.
3. **Minimization**: local (L-BFGS) energy minimization to a nearby potential
   energy minimum before scoring.
4. **Entropy** (RRHO -- rigid-rotor/harmonic-oscillator, the same approach
   used in MM-PBSA/MM-GBSA "normal-mode entropy" workflows):
   - Translational entropy: Sackur-Tetrode equation from the total mass.
   - Rotational entropy: rigid-rotor formula from the moment-of-inertia
     tensor of the minimized structure.
   - Vibrational entropy: harmonic-oscillator sum over normal-mode
     frequencies. By default these come from a fast Cα **anisotropic network
     model (ANM)**; a full atomistic finite-difference Hessian is available
     for small systems via `--entropy-method full-hessian`.
5. **G = H - T·S**, with `H = U_MM + E_thermal + ZPE + RT` (ideal-gas PV term)
   and `S = S_trans + S_rot + S_vib`, at the chosen temperature.
6. **Hydrogen-bond breakdown** (optional, on by default): ff14SB has no
   explicit hydrogen-bond term -- H-bonds are already fully present in the
   Coulomb/vdW totals above, emerging from the partial charges and LJ
   parameters on the donor/acceptor atoms. This step geometrically detects
   donor-H...acceptor contacts (Baker-Hubbard-style: H...acceptor <= 2.5 Å,
   angle >= 120°), classifies each by whether the two residues sit in the
   same backbone-helical segment, in two different ones (e.g. bridging two
   different transmembrane helices), or a loop, and reports the full
   residue-residue Coulomb+LJ interaction energy for each contact. This does
   not add anything to `G` -- it's a breakdown of energy already counted
   elsewhere, aimed at answering "how much of the electrostatics is
   hydrogen bonding, and is it holding different parts of the protein
   together." Disable with `--no-hbond-analysis`.

## Usage

```bash
pip install -r requirements.txt
python gpcr_gibbs_energy.py structure.pdb
python gpcr_gibbs_energy.py structure.pdb --chains R --temperature 310.15
python gpcr_gibbs_energy.py structure.pdb --entropy-method none      # potential energy only
python gpcr_gibbs_energy.py structure.pdb --entropy-method full-hessian  # small systems only
```

`structure.pdb` can also be a 4-character PDB ID, in which case the script
tries to download it from RCSB -- this requires outbound network access,
which not every environment allows (in a sandboxed environment, download the
file yourself and pass a local path).

GPCR structures solved by X-ray/cryo-EM are frequently crystallized with a
fusion partner (T4 lysozyme, BRIL) or a stabilizing nanobody/Fab in a
separate chain. Use `--chains` to restrict the calculation to the receptor
chain(s) only -- the script prints the chains it finds (with residue/hetero
summaries) before running so you can check.

## Approximations & limitations (read this before trusting a number)

This is a **single-structure estimate**, not a converged ensemble average.
A handful of things this does *not* capture, in rough order of impact for a
GPCR specifically:

- **No lipid bilayer.** GPCRs are membrane proteins; the transmembrane
  helices are normally stabilized by a highly anisotropic lipid environment
  (hydrophobic core, headgroup region, hydration boundary). This script
  replaces that with an isotropic implicit solvent (GBn2), which is a
  significant simplification for the parts of the receptor embedded in
  bilayer. A more faithful treatment would embed the receptor in an explicit
  POPC/cholesterol bilayer and run molecular dynamics (OpenMM ships lipid
  parameters, e.g. `amber14/lipid17.xml`, for exactly this, but that is an
  MD-timescale calculation, not a single-structure one).
- **No conformational ensemble / no sampling.** A true Gibbs free energy is
  an ensemble average over the Boltzmann-weighted conformational landscape
  (via free-energy perturbation, thermodynamic integration, or extensive MD
  + reweighting). Scoring one static, minimized structure with an RRHO
  entropy correction is the standard MM-PBSA/GBSA-with-normal-modes shortcut,
  and is known to have multi-kcal/mol uncertainty -- treat the reported `G`
  as an order-of-magnitude/relative quantity, not a thermodynamically exact
  number. It is most meaningful when *comparing* two structures/states
  computed the same way (e.g. active vs. inactive conformation, apo vs.
  mutant), where systematic errors partially cancel.
- **Vibrational entropy from a coarse-grained elastic network (default).**
  The Cα ANM used for `S_vib` is fast enough to run on a full receptor, but
  it is a heuristic proxy (uniform spring constant, no real force-field
  curvature) -- it gives a physically reasonable order of magnitude for
  `T*S_vib`, not a rigorous value. `--entropy-method full-hessian` uses the
  real force-field second derivatives instead, but its O((3N)^3) cost only
  scales to small systems/domains, not a full multi-hundred-residue GPCR.
- **Protonation states from simple pH rules**, not a structure-aware pKa
  predictor (e.g. PROPKA/H++). Histidine tautomers, unusual buried
  ionizable residues, and pH-shifted pKas near the binding pocket are not
  specifically checked.
- **Missing loops are rebuilt with idealized geometry** by PDBFixer if the
  input is missing internal residues -- fine for a few residues, unreliable
  for long disordered loops (e.g. ICL3, which many GPCR constructs replace
  with a fusion protein or truncate entirely; if so, that's already handled
  by chain selection rather than loop building).
- **No explicit ligand/cofactor energetics.** Bound ligands, ions, and
  lipids resolved in the crystal structure are stripped by default (`
  --keep-hetero` keeps them, but they need force-field parameters the
  built-in AMBER/GBn2 combination does not provide, e.g. GAFF2 + AM1-BCC
  charges for a small-molecule ligand -- the script will raise a clear error
  from OpenMM's `ForceField.createSystem` if you pass `--keep-hetero` without
  extending the force field yourself).
- **Helix segments are a Ramachandran-box heuristic, not DSSP.** Residues are
  labeled "helical" from backbone phi/psi falling in a fixed alpha-helix box,
  then grouped into contiguous runs -- good enough to tell "same helix" from
  "different helix" for the H-bond breakdown, but it will occasionally
  mislabel a residue at a helix boundary or a 3-10/pi-helix turn that DSSP
  would call differently.

In short: this is a legitimate, physically grounded MM-GBSA/RRHO estimate of
Gibbs free energy for a static structure, useful for sanity checks and
relative comparisons, and it does correctly account for every bonded and
non-bonded intramolecular force in the AMBER force field -- but it is not a
substitute for MD-based free-energy methods when you need
publication-quality absolute numbers.

## Output

The script prints an energy decomposition (bonds, angles, torsions, van der
Waals, electrostatics, implicit solvation) in kcal/mol, followed by the RRHO
thermochemistry breakdown and the final `G` value, and returns the same
numbers as a Python dict from `run()` for programmatic use.
