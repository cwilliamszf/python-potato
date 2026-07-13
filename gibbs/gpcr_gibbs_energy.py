#!/usr/bin/env python3
"""
Estimate the Gibbs free energy of a protein structure (e.g. a GPCR) from its
atomic coordinates.

Method
------
1.  The structure is protonated and parameterized with the AMBER ff14SB force
    field (bonds, angles, torsions, van der Waals, electrostatics -- i.e. the
    "intramolecular forces") plus a GB-Neck2 implicit-solvent term.
2.  The structure is locally energy-minimized (0 K potential-energy minimum).
3.  Entropy is estimated with the standard rigid-rotor / harmonic-oscillator
    (RRHO) treatment used in MM-PBSA/MM-GBSA "normal-mode entropy" workflows:
    translational + rotational entropy from the mass/geometry of the
    structure, and vibrational entropy from a normal-mode analysis.
4.  G = H - T*S, with H = U_MM + G_solvation + E_thermal(T) (RRHO thermal
    correction), reported at the chosen temperature.

This is a single-structure *approximation* to the Gibbs free energy, not an
ensemble average. See the "Approximations & limitations" section printed at
the end of the report (or gibbs/README.md) for what this does and does not
capture -- most importantly: no explicit lipid bilayer (GPCRs are membrane
proteins; here the membrane is replaced by an isotropic implicit solvent),
and vibrational entropy from a coarse-grained elastic network model by
default (fast, but only an order-of-magnitude estimate of S_vib).

Usage
-----
    python gpcr_gibbs_energy.py structure.pdb
    python gpcr_gibbs_energy.py structure.pdb --chains R --temperature 310.15
    python gpcr_gibbs_energy.py structure.pdb --entropy-method none
    python gpcr_gibbs_energy.py structure.pdb --entropy-method full-hessian
"""

import argparse
import io
import math

import numpy as np
import openmm as mm
from openmm import app, unit
from pdbfixer import PDBFixer

# ----------------------------------------------------------------------------
# Physical constants (SI, plus mol-based gas constant)
# ----------------------------------------------------------------------------
KB = 1.380649e-23          # J/K
H_PLANCK = 6.62607015e-34  # J*s
NA = 6.02214076e23         # 1/mol
R_GAS = KB * NA            # J/(mol K)
AMU_TO_KG = 1.66053906660e-27
ANGSTROM_TO_M = 1e-10
KCAL_TO_J = 4184.0
KJ_TO_KCAL = 1.0 / 4.184
ATM_TO_PA = 101325.0

STANDARD_AA = {
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
    "HID", "HIE", "HIP", "CYX", "CYM", "ASH", "GLH", "LYN",
}


# ----------------------------------------------------------------------------
# Structure loading / cleanup
# ----------------------------------------------------------------------------
def load_pdb_text(pdb_source):
    """Load PDB text from a local file path or a raw 4-character PDB ID.

    Fetching by ID requires outbound network access to files.rcsb.org, which
    is not available in every environment (e.g. this sandbox blocks it) --
    in that case, download the file yourself and pass a local path instead.
    """
    import os

    if os.path.isfile(pdb_source):
        with open(pdb_source) as fh:
            return fh.read()

    if len(pdb_source) == 4 and pdb_source.isalnum():
        import urllib.request

        url = f"https://files.rcsb.org/download/{pdb_source.upper()}.pdb"
        try:
            with urllib.request.urlopen(url, timeout=30) as resp:
                return resp.read().decode()
        except Exception as exc:
            raise SystemExit(
                f"Could not fetch {pdb_source!r} from RCSB ({exc}). "
                "Download the PDB file yourself and pass a local path instead."
            )

    raise SystemExit(f"'{pdb_source}' is neither an existing file nor a 4-character PDB ID.")


def clean_pdb_text(text, chains=None, keep_hetero=False, first_model_only=True):
    """Keep only the first model, optionally restrict to given chains, and
    drop heteroatoms (waters/ligands/lipids/fusion-partner cofactors) by
    default. Nonstandard residues (e.g. MSE selenomethionine) are handled
    later by PDBFixer.replaceNonstandardResidues(), not here.
    """
    chains = set(chains) if chains else None
    out_lines = []
    in_first_model = True
    seen_model = False

    for line in text.splitlines():
        record = line[:6].strip()

        if record == "MODEL":
            seen_model = True
            model_num = line.split()[1] if len(line.split()) > 1 else "1"
            in_first_model = model_num.strip() == "1"
            continue
        if record == "ENDMDL":
            if first_model_only and in_first_model:
                break
            continue
        if first_model_only and seen_model and not in_first_model:
            continue

        if record in ("ATOM", "HETATM"):
            chain_id = line[21]
            if chains and chain_id not in chains:
                continue
            if record == "HETATM" and not keep_hetero:
                resname = line[17:20].strip()
                if resname != "MSE":  # let PDBFixer convert MSE -> MET rather than dropping it
                    continue
            out_lines.append(line)
        elif record in ("TER", "END", "CRYST1"):
            out_lines.append(line)

    out_lines.append("END")
    return "\n".join(out_lines) + "\n"


def summarize_chains(text):
    chains = {}
    for line in text.splitlines():
        if line[:6].strip() in ("ATOM", "HETATM"):
            chain_id = line[21]
            resname = line[17:20].strip()
            resnum = line[22:26]
            chains.setdefault(chain_id, {"residues": set(), "hetero": set()})
            key = (resnum, resname)
            if line[:6].strip() == "ATOM" or resname in STANDARD_AA:
                chains[chain_id]["residues"].add(key)
            else:
                chains[chain_id]["hetero"].add(resname)
    return chains


# ----------------------------------------------------------------------------
# Force-field system construction
# ----------------------------------------------------------------------------
FORCE_GROUPS = {
    "HarmonicBondForce": 0,
    "HarmonicAngleForce": 1,
    "PeriodicTorsionForce": 2,
    "NonbondedForce": 3,
    "CustomGBForce": 4,
    "GBSAOBCForce": 4,
    "CMMotionRemover": 5,
}

TERM_LABELS = {
    0: "Bond stretching",
    1: "Angle bending",
    2: "Torsions (proper + improper)",
    3: "Nonbonded (van der Waals + electrostatics, raw sum)",
    4: "Implicit solvation (GB polar + nonpolar SA)",
}


def build_forcefield():
    return app.ForceField("amber14/protein.ff14SB.xml", "implicit/gbn2.xml")


def prepare_modeller(pdb_text, ph, keep_hetero=False):
    """Run the structure through PDBFixer to replace nonstandard residues,
    complete any truncated/disordered side chains (a common feature of real
    crystal structures -- surface Asp/Glu/Lys/Arg side chains are frequently
    only partially resolved), then add hydrogens for the target pH.
    """
    fixer = PDBFixer(pdbfile=io.StringIO(pdb_text))
    fixer.findNonstandardResidues()
    fixer.replaceNonstandardResidues()
    if not keep_hetero:
        fixer.removeHeterogens(keepWater=False)
    fixer.findMissingResidues()
    n_missing_residues = sum(len(v) for v in fixer.missingResidues.values())
    if n_missing_residues:
        print(f"  NOTE: PDBFixer will build {n_missing_residues} missing residue(s) "
              "(gaps in the deposited structure) using idealized geometry.")
    fixer.findMissingAtoms()
    n_missing_atoms = sum(len(v) for v in fixer.missingAtoms.values())
    n_missing_terminals = sum(len(v) for v in fixer.missingTerminals.values())
    if n_missing_atoms or n_missing_terminals:
        print(f"  NOTE: completing {n_missing_atoms} missing heavy atom(s) and "
              f"{n_missing_terminals} missing terminal atom(s) (truncated/disordered "
              "side chains in the deposited structure).")
    fixer.addMissingAtoms()
    fixer.addMissingHydrogens(ph)

    forcefield = build_forcefield()
    modeller = app.Modeller(fixer.topology, fixer.positions)
    return modeller, forcefield


def create_system(modeller, forcefield, nonbonded_cutoff_nm=1.6):
    # A finite cutoff (standard practice for GBSA) keeps nonbonded evaluation
    # O(N) instead of O(N^2) -- NoCutoff is impractically slow to minimize
    # for a full-atom, multi-thousand-atom GPCR on CPU.
    system = forcefield.createSystem(
        modeller.topology,
        nonbondedMethod=app.CutoffNonPeriodic,
        nonbondedCutoff=nonbonded_cutoff_nm * unit.nanometer,
        constraints=None,          # keep real bonds so bonded energy can be decomposed
        removeCMMotion=False,
    )
    for force in system.getForces():
        cls = force.__class__.__name__
        if cls in FORCE_GROUPS:
            force.setForceGroup(FORCE_GROUPS[cls])
    return system


def minimize(system, positions, platform_name="CPU", tolerance_kj_per_mol_nm=1.0, max_iterations=0):
    integrator = mm.VerletIntegrator(1.0 * unit.femtoseconds)
    try:
        platform = mm.Platform.getPlatformByName(platform_name)
        context = mm.Context(system, integrator, platform)
    except Exception:
        context = mm.Context(system, integrator)
    context.setPositions(positions)
    mm.LocalEnergyMinimizer.minimize(
        context, tolerance=tolerance_kj_per_mol_nm * unit.kilojoule_per_mole / unit.nanometer,
        maxIterations=max_iterations,
    )
    state = context.getState(getPositions=True, getEnergy=True)
    return context, state


def energy_by_group(context, group):
    state = context.getState(getEnergy=True, groups={group})
    return state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole) * KJ_TO_KCAL


def clone_system_nonbonded_only(system, zero="none"):
    """Return a stripped copy of `system` containing only its NonbondedForce,
    with charges or LJ well-depths optionally zeroed out, so the LJ and
    Coulomb contributions to the total nonbonded energy can be isolated.
    """
    xml = mm.XmlSerializer.serialize(system)
    clone = mm.XmlSerializer.deserialize(xml)

    # Remove every force except NonbondedForce, by index and in reverse order,
    # so indices of not-yet-processed forces stay valid as forces are removed.
    for i in reversed(range(clone.getNumForces())):
        if clone.getForce(i).__class__.__name__ != "NonbondedForce":
            clone.removeForce(i)

    nb_force = None
    for i in range(clone.getNumForces()):
        if clone.getForce(i).__class__.__name__ == "NonbondedForce":
            nb_force = clone.getForce(i)
            break
    if nb_force is None:
        raise RuntimeError("System has no NonbondedForce to decompose.")

    if zero == "charge":
        for i in range(nb_force.getNumParticles()):
            charge, sigma, epsilon = nb_force.getParticleParameters(i)
            nb_force.setParticleParameters(i, 0.0 * unit.elementary_charge, sigma, epsilon)
        for i in range(nb_force.getNumExceptions()):
            p1, p2, chargeProd, sigma, epsilon = nb_force.getExceptionParameters(i)
            nb_force.setExceptionParameters(i, p1, p2, 0.0 * unit.elementary_charge**2, sigma, epsilon)
    elif zero == "lj":
        for i in range(nb_force.getNumParticles()):
            charge, sigma, epsilon = nb_force.getParticleParameters(i)
            nb_force.setParticleParameters(i, charge, sigma, 0.0 * unit.kilojoule_per_mole)
        for i in range(nb_force.getNumExceptions()):
            p1, p2, chargeProd, sigma, epsilon = nb_force.getExceptionParameters(i)
            nb_force.setExceptionParameters(i, p1, p2, chargeProd, sigma, 0.0 * unit.kilojoule_per_mole)

    return clone


def split_lj_coulomb(system, positions):
    lj_only = clone_system_nonbonded_only(system, zero="charge")
    coul_only = clone_system_nonbonded_only(system, zero="lj")

    results = {}
    for label, sub_system in (("vdW (Lennard-Jones)", lj_only), ("Electrostatics (Coulomb)", coul_only)):
        integrator = mm.VerletIntegrator(1.0 * unit.femtoseconds)
        context = mm.Context(sub_system, integrator)
        context.setPositions(positions)
        e = context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
        results[label] = e * KJ_TO_KCAL
    return results


# ----------------------------------------------------------------------------
# RRHO thermochemistry
# ----------------------------------------------------------------------------
def translational_entropy(total_mass_amu, temperature_k, pressure_atm=1.0):
    m = total_mass_amu * AMU_TO_KG
    p = pressure_atm * ATM_TO_PA
    q = (2 * math.pi * m * KB * temperature_k / H_PLANCK**2) ** 1.5 * KB * temperature_k / p
    s = R_GAS * (math.log(q) + 2.5)
    e_thermal = 1.5 * R_GAS * temperature_k
    return s, e_thermal  # J/(mol K), J/mol


def rotational_entropy(positions_ang, masses_amu, temperature_k, symmetry_number=1):
    positions = np.asarray(positions_ang)
    masses = np.asarray(masses_amu)
    com = np.average(positions, axis=0, weights=masses)
    rel = (positions - com) * ANGSTROM_TO_M
    m_kg = masses * AMU_TO_KG

    inertia = np.zeros((3, 3))
    for r, m in zip(rel, m_kg):
        inertia += m * (np.dot(r, r) * np.eye(3) - np.outer(r, r))
    moments = np.sort(np.linalg.eigvalsh(inertia))
    moments = np.clip(moments, a_min=1e-60, a_max=None)

    prefactor = (math.pi ** 0.5 / symmetry_number) * (8 * math.pi**2 * KB * temperature_k / H_PLANCK**2) ** 1.5
    s = R_GAS * (math.log(prefactor * math.sqrt(np.prod(moments))) + 1.5)
    e_thermal = 1.5 * R_GAS * temperature_k
    return s, e_thermal, moments  # J/(mol K), J/mol, kg*m^2


def vibrational_thermo(frequencies_hz, temperature_k):
    """Quantum harmonic-oscillator entropy, ZPE and thermal energy for a list
    of (positive, non-zero) vibrational frequencies."""
    s_total = 0.0
    zpe_total = 0.0
    e_thermal_total = 0.0
    for nu in frequencies_hz:
        if nu <= 0:
            continue
        x = H_PLANCK * nu / (KB * temperature_k)
        if x > 500:  # exp overflow guard; contribution is ~0 anyway
            continue
        zpe_total += 0.5 * H_PLANCK * nu
        e_thermal_total += H_PLANCK * nu / (math.exp(x) - 1.0)
        s_total += R_GAS * (x / (math.exp(x) - 1.0) - math.log(1.0 - math.exp(-x)))
    return s_total, zpe_total * NA, e_thermal_total * NA  # J/(mol K), J/mol, J/mol


def anm_frequencies(ca_positions_ang, ca_masses_amu, cutoff_ang=15.0, gamma_kcal_per_mol_ang2=1.0):
    """Anisotropic Network Model normal modes on CA atoms only: a coarse-
    grained elastic-network proxy for the full-atom Hessian. Fast (scales
    with residue count, not atom count) but only an order-of-magnitude
    estimate of vibrational entropy -- see README for caveats.
    """
    pos = np.asarray(ca_positions_ang)
    n = len(pos)
    gamma = gamma_kcal_per_mol_ang2 * KCAL_TO_J / (ANGSTROM_TO_M**2) / NA  # J/m^2 per node pair

    hessian = np.zeros((3 * n, 3 * n))
    for i in range(n):
        for j in range(i + 1, n):
            d_vec = (pos[i] - pos[j]) * ANGSTROM_TO_M
            d2 = np.dot(d_vec, d_vec)
            d = math.sqrt(d2)
            if d > cutoff_ang * ANGSTROM_TO_M:
                continue
            block = -gamma * np.outer(d_vec, d_vec) / d2
            hessian[3 * i:3 * i + 3, 3 * j:3 * j + 3] += block
            hessian[3 * j:3 * j + 3, 3 * i:3 * i + 3] += block
            hessian[3 * i:3 * i + 3, 3 * i:3 * i + 3] -= block
            hessian[3 * j:3 * j + 3, 3 * j:3 * j + 3] -= block

    masses_kg = np.asarray(ca_masses_amu) * AMU_TO_KG
    inv_sqrt_m = 1.0 / np.sqrt(np.repeat(masses_kg, 3))
    mw_hessian = hessian * np.outer(inv_sqrt_m, inv_sqrt_m)

    eigvals = np.linalg.eigvalsh(mw_hessian)
    eigvals = np.sort(eigvals)[6:]  # drop 6 lowest (trans + rot) modes
    eigvals = np.clip(eigvals, a_min=0.0, a_max=None)
    omega = np.sqrt(eigvals)  # rad/s
    freq_hz = omega / (2 * math.pi)
    return freq_hz


def full_hessian_frequencies(system, positions_nm, masses_amu, platform_name="CPU", step_nm=1e-4):
    """Numerical (central-difference) full-atom Hessian from the actual MM
    force field, for small systems only -- O(6N) force evaluations plus an
    O((3N)^3) diagonalization, so this does not scale to a full GPCR.
    """
    n = len(masses_amu)
    integrator = mm.VerletIntegrator(1.0 * unit.femtoseconds)
    try:
        platform = mm.Platform.getPlatformByName(platform_name)
        context = mm.Context(system, integrator, platform)
    except Exception:
        context = mm.Context(system, integrator)

    base = np.array(positions_nm)
    hessian = np.zeros((3 * n, 3 * n))

    def forces_at(pos):
        context.setPositions(pos * unit.nanometer)
        state = context.getState(getForces=True)
        return state.getForces(asNumpy=True).value_in_unit(unit.kilojoule_per_mole / unit.nanometer)

    for i in range(n):
        for d in range(3):
            disp = np.zeros_like(base)
            disp[i, d] = step_nm
            f_plus = forces_at(base + disp)
            f_minus = forces_at(base - disp)
            # Hessian column = -dF/dx
            hessian[:, 3 * i + d] = (-(f_plus - f_minus) / (2 * step_nm)).flatten()

    # symmetrize, convert kJ/mol/nm^2 -> J/m^2
    hessian = 0.5 * (hessian + hessian.T)
    hessian_si = hessian * 1000.0 / NA / (1e-9 ** 2)

    masses_kg = np.asarray(masses_amu) * AMU_TO_KG
    inv_sqrt_m = 1.0 / np.sqrt(np.repeat(masses_kg, 3))
    mw_hessian = hessian_si * np.outer(inv_sqrt_m, inv_sqrt_m)

    eigvals = np.linalg.eigvalsh(mw_hessian)
    eigvals = np.sort(eigvals)[6:]
    eigvals = np.clip(eigvals, a_min=0.0, a_max=None)
    omega = np.sqrt(eigvals)
    return omega / (2 * math.pi)


# ----------------------------------------------------------------------------
# Report
# ----------------------------------------------------------------------------
def format_kcal(x):
    return f"{x:10.2f} kcal/mol"


def run(args):
    text = load_pdb_text(args.pdb)
    chain_summary = summarize_chains(text)

    print("=" * 78)
    print("Chains found in input structure:")
    for cid, info in sorted(chain_summary.items()):
        hetero = f", hetero: {sorted(info['hetero'])}" if info["hetero"] else ""
        print(f"  chain {cid!r}: {len(info['residues'])} standard residues{hetero}")
    if args.chains is None:
        print(
            "  NOTE: no --chains filter given -- ALL chains above will be modeled.\n"
            "  GPCR crystal/cryo-EM structures often include fusion partners\n"
            "  (T4 lysozyme, BRIL, nanobodies) or non-receptor chains; use\n"
            "  --chains to restrict to the receptor chain(s) if that's not\n"
            "  what you want energetically."
        )
    print("=" * 78)

    cleaned = clean_pdb_text(text, chains=args.chains, keep_hetero=args.keep_hetero)
    modeller, forcefield = prepare_modeller(cleaned, ph=args.ph, keep_hetero=args.keep_hetero)
    system = create_system(modeller, forcefield, nonbonded_cutoff_nm=args.nonbonded_cutoff)

    n_atoms = system.getNumParticles()
    masses_amu = np.array([
        system.getParticleMass(i).value_in_unit(unit.dalton) for i in range(n_atoms)
    ])
    total_mass = masses_amu.sum()
    print(f"Prepared structure: {n_atoms} atoms (with hydrogens), total mass {total_mass:.1f} Da")

    context, state = minimize(
        system, modeller.positions, platform_name=args.platform,
        max_iterations=args.minimize_iterations,
    )
    positions_nm = state.getPositions(asNumpy=True).value_in_unit(unit.nanometer)
    positions_ang = positions_nm * 10.0

    print("\nPotential-energy decomposition (minimized structure):")
    bonded_total = 0.0
    for group in (0, 1, 2):
        e = energy_by_group(context, group)
        bonded_total += e
        print(f"  {TERM_LABELS[group]:<45s} {format_kcal(e)}")

    nb_total = energy_by_group(context, 3)
    lj_coul = split_lj_coulomb(system, state.getPositions())
    print(f"  {TERM_LABELS[3]:<45s} {format_kcal(nb_total)}")
    for label, e in lj_coul.items():
        print(f"      - {label:<41s} {format_kcal(e)}")
    residual = nb_total - sum(lj_coul.values())
    print(f"      - (LJ + Coulomb) - raw nonbonded residual  {residual:10.4f} kcal/mol  (sanity check, should be ~0)")

    solvation = energy_by_group(context, 4)
    print(f"  {TERM_LABELS[4]:<45s} {format_kcal(solvation)}")

    u_mm = bonded_total + nb_total + solvation
    print(f"\n  TOTAL potential energy U_MM (0 K, includes solvation) {format_kcal(u_mm)}")

    result = {
        "n_atoms": n_atoms,
        "bonded_kcal": bonded_total,
        "nonbonded_kcal": nb_total,
        "vdw_kcal": lj_coul["vdW (Lennard-Jones)"],
        "coulomb_kcal": lj_coul["Electrostatics (Coulomb)"],
        "solvation_kcal": solvation,
        "u_mm_kcal": u_mm,
    }

    if args.entropy_method == "none":
        print("\nEntropy estimation skipped (--entropy-method none); "
              "reporting potential energy only, not a full Gibbs free energy.")
        result["G_kcal"] = None
        return result

    T = args.temperature
    print(f"\nRRHO thermochemistry at T = {T:.2f} K:")

    s_trans, e_trans = translational_entropy(total_mass, T)
    s_rot, e_rot, moments = rotational_entropy(positions_ang, masses_amu, T)

    if args.entropy_method == "full-hessian":
        if n_atoms > args.full_hessian_atom_limit:
            raise SystemExit(
                f"--entropy-method full-hessian requested but system has {n_atoms} atoms "
                f"(limit {args.full_hessian_atom_limit}); a full finite-difference Hessian "
                "does not scale to a full GPCR in reasonable time. Raise "
                "--full-hessian-atom-limit only if you understand the O((3N)^3) cost, "
                "or use --entropy-method ca-anm."
            )
        freqs = full_hessian_frequencies(system, positions_nm, masses_amu, platform_name=args.platform)
        method_desc = "full-atom finite-difference Hessian"
    else:
        ca_indices = [
            atom.index for atom in modeller.topology.atoms() if atom.name == "CA"
        ]
        if len(ca_indices) < 4:
            raise SystemExit("Could not find enough CA atoms for the coarse-grained ANM entropy model.")
        ca_pos = positions_ang[ca_indices]
        ca_masses = np.array([110.0] * len(ca_indices))  # average residue mass lumped at CA
        freqs = anm_frequencies(ca_pos, ca_masses, cutoff_ang=args.anm_cutoff, gamma_kcal_per_mol_ang2=args.anm_gamma)
        method_desc = f"Cα anisotropic network model (ANM), cutoff={args.anm_cutoff} Å, γ={args.anm_gamma} kcal/mol/Å²"

    s_vib, zpe, e_vib_thermal = vibrational_thermo(freqs, T)

    s_total = s_trans + s_rot + s_vib          # J/(mol K)
    e_thermal_total = e_trans + e_rot + e_vib_thermal + zpe  # J/mol
    pv = R_GAS * T  # ideal-gas PV term, J/mol

    h_total_kcal = u_mm + (e_thermal_total + pv) * KJ_TO_KCAL / 1000.0
    ts_kcal = s_total * T * KJ_TO_KCAL / 1000.0
    g_kcal = h_total_kcal - ts_kcal

    print(f"  Vibrational modes from: {method_desc}")
    print(f"  {len(freqs)} vibrational modes retained (after removing 6 trans/rot modes)")
    print(f"  S_trans = {s_trans:8.2f} J/mol/K   S_rot = {s_rot:8.2f} J/mol/K   S_vib = {s_vib:8.2f} J/mol/K")
    print(f"  S_total = {s_total:8.2f} J/mol/K  ({s_total * KJ_TO_KCAL / 1000.0 * 1000:.4f} cal/mol/K)")
    print(f"  Zero-point energy       {zpe * KJ_TO_KCAL / 1000.0:10.2f} kcal/mol")
    print(f"  Thermal energy (T,R,V)  {(e_trans + e_rot + e_vib_thermal) * KJ_TO_KCAL / 1000.0:10.2f} kcal/mol")
    print(f"  H = U_MM + E_thermal + ZPE + PV        {format_kcal(h_total_kcal)}")
    print(f"  T*S_total                              {format_kcal(ts_kcal)}")
    print(f"\n  G = H - T*S                            {format_kcal(g_kcal)}")

    result.update({
        "s_trans_J_per_molK": s_trans,
        "s_rot_J_per_molK": s_rot,
        "s_vib_J_per_molK": s_vib,
        "s_total_J_per_molK": s_total,
        "H_kcal": h_total_kcal,
        "TS_kcal": ts_kcal,
        "G_kcal": g_kcal,
        "temperature_K": T,
    })
    return result


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("pdb", help="Path to a local PDB file, or a 4-character PDB ID (requires network access).")
    p.add_argument("--chains", type=lambda s: s.split(","), default=None,
                   help="Comma-separated chain IDs to keep (e.g. --chains R). "
                        "Recommended for GPCR structures solved with fusion partners/nanobodies/antibodies.")
    p.add_argument("--keep-hetero", action="store_true",
                   help="Keep HETATM records (waters, ligands, lipids, ions) instead of stripping them. "
                        "These usually lack force-field parameters and will make system creation fail "
                        "unless you extend the force field yourself.")
    p.add_argument("--ph", type=float, default=7.4, help="pH used to choose protonation states (default 7.4).")
    p.add_argument("--temperature", type=float, default=310.15, help="Temperature in Kelvin (default 310.15 = 37 C).")
    p.add_argument("--platform", default="CPU", choices=["CPU", "Reference", "CUDA", "OpenCL"],
                   help="OpenMM platform to run on (default CPU).")
    p.add_argument("--minimize-iterations", type=int, default=500,
                   help="Max L-BFGS minimizer iterations (0 = run to convergence, can be very slow "
                        "on a full-atom GPCR; default 500 is enough to relax steric clashes from "
                        "added hydrogens/side chains without an open-ended runtime).")
    p.add_argument("--nonbonded-cutoff", type=float, default=1.6,
                   help="Nonbonded interaction cutoff in nm (default 1.6). Using a finite cutoff "
                        "instead of NoCutoff keeps evaluation near-linear in atom count, which "
                        "matters for a full-atom GPCR; raise it (or use a very large value to "
                        "approximate NoCutoff) if you have the compute budget for it.")
    p.add_argument("--entropy-method", choices=["ca-anm", "full-hessian", "none"], default="ca-anm",
                   help="How to estimate vibrational entropy (default: fast Cα elastic-network model). "
                        "'none' reports only the potential energy, not a full Gibbs free energy.")
    p.add_argument("--full-hessian-atom-limit", type=int, default=500,
                   help="Safety limit on atom count for --entropy-method full-hessian.")
    p.add_argument("--anm-cutoff", type=float, default=15.0, help="ANM interaction cutoff distance in Angstrom.")
    p.add_argument("--anm-gamma", type=float, default=1.0, help="ANM uniform spring constant in kcal/mol/Å^2.")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    run(args)


if __name__ == "__main__":
    main()
