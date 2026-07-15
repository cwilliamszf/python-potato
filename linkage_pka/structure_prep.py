"""Structure preparation for the proton-linkage pipeline: protonation,
side-chain rotamer optimization for ionizable residues, and a restrained
energy minimization -- pipeline spec step 1.

Tools/versions (recorded per-call in ``PrepResult.tool_versions``):
  - PDBFixer: completes missing residues/heavy atoms.
  - OpenMM ``Modeller.addHydrogens`` with the amber14-all.xml force field:
    hydrogen placement and protonation-state assignment at a given pH via
    standard pKa cutoffs (this is a *starting* protonation state for
    packing sterics, not the rigorous per-site result -- that comes from
    the Poisson-Boltzmann titration step elsewhere in the pipeline).
  - OpenMM ``LocalEnergyMinimizer``: backbone-restrained whole-structure
    minimization (a scoring-function search to a local minimum, not an MD
    trajectory).

Rotamer optimization here is deliberately NOT the empirical Dunbrack/
Lovell-Richardson rotamer library. That library's published chi-angle and
population tables could not be retrieved and verified over the network in
this environment (WebFetch was blocked for every domain tried, including
non-paywalled ones; see chat history for what was checked: PyRosetta,
PyMOL's mutagenesis wizard, MODELLER, OSPREY, FASPR/EvoEF all ruled out).
Hardcoding remembered numbers under that citation would be exactly the
"don't trust unverified remembered values" failure this pipeline is
designed to avoid.

Instead: candidate chi1/chi2 values are the three canonical staggered
conformations (gauche+, trans, gauche-; ~+60/180/-60 degrees) -- the
physical basis every empirical rotamer library is itself built on -- and
the 3x3=9 combinations are enumerated and scored by single-point OpenMM
potential energy, substituting a physics-based selection for the missing
empirical population weighting. More distal chi angles (chi3 for Glu,
chi3/chi4 for Lys/Arg) are held at their as-modeled value: the pipeline
spec only requires reporting chi1/chi2, and those two dominate how the
titratable group is positioned relative to its environment. This is
documented here as exactly what it is, not mislabeled as a literature
rotamer library.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# ---------------------------------------------------------------- chi atoms --
# For residue X, chi index k: (4-atom tuple defining the dihedral, atom-name
# set that moves when chi_k is changed). The moving set is everything
# downstream of the rotatable bond (the bond between the 2nd and 3rd
# defining atom) -- standard PDB/AMBER heavy+hydrogen connectivity, not
# empirical rotamer data.
CHI_ATOMS = {
    "ASP": [
        (("N", "CA", "CB", "CG"), ("CG", "OD1", "OD2")),
        (("CA", "CB", "CG", "OD1"), ("OD1", "OD2")),
    ],
    "GLU": [
        (("N", "CA", "CB", "CG"), ("CG", "HG2", "HG3", "CD", "OE1", "OE2")),
        (("CA", "CB", "CG", "CD"), ("CD", "OE1", "OE2")),
        (("CB", "CG", "CD", "OE1"), ("OE1", "OE2")),
    ],
    "HIS": [
        (("N", "CA", "CB", "CG"), ("CG", "CD2", "HD2", "ND1", "HD1", "CE1", "HE1", "NE2")),
        (("CA", "CB", "CG", "ND1"), ("CD2", "HD2", "ND1", "HD1", "CE1", "HE1", "NE2")),
    ],
    "LYS": [
        (("N", "CA", "CB", "CG"), ("CG", "HG2", "HG3", "CD", "HD2", "HD3", "CE", "HE2", "HE3", "NZ", "HZ1", "HZ2", "HZ3")),
        (("CA", "CB", "CG", "CD"), ("CD", "HD2", "HD3", "CE", "HE2", "HE3", "NZ", "HZ1", "HZ2", "HZ3")),
        (("CB", "CG", "CD", "CE"), ("CE", "HE2", "HE3", "NZ", "HZ1", "HZ2", "HZ3")),
        (("CG", "CD", "CE", "NZ"), ("NZ", "HZ1", "HZ2", "HZ3")),
    ],
    "ARG": [
        (("N", "CA", "CB", "CG"), ("CG", "HG2", "HG3", "CD", "HD2", "HD3", "NE", "HE", "CZ", "NH1", "HH11", "HH12", "NH2", "HH21", "HH22")),
        (("CA", "CB", "CG", "CD"), ("CD", "HD2", "HD3", "NE", "HE", "CZ", "NH1", "HH11", "HH12", "NH2", "HH21", "HH22")),
        (("CB", "CG", "CD", "NE"), ("NE", "HE", "CZ", "NH1", "HH11", "HH12", "NH2", "HH21", "HH22")),
        (("CG", "CD", "NE", "CZ"), ("CZ", "NH1", "HH11", "HH12", "NH2", "HH21", "HH22")),
    ],
}

IONIZABLE_RESNAMES = frozenset(CHI_ATOMS)  # {"ASP", "GLU", "HIS", "LYS", "ARG"}

# Chi-angle geometry for the remaining standard residues with a rotatable
# side chain (i.e. every residue except ALA/GLY, which have no heavy atom
# beyond CB to rotate, and PRO, whose ring closure back to N makes a
# simple independent-bond rotation geometrically invalid). Kept separate
# from CHI_ATOMS/IONIZABLE_RESNAMES deliberately: IONIZABLE_RESNAMES is
# derived directly from CHI_ATOMS' keys and drives structure_prep's own
# protonation/titration-relevant residue selection -- these residues are
# not titratable, and must never leak into that set. Built for
# linkage_pka's per-microstate *neighbor* rotamer relaxation (see
# titration.optimize_rotamers_with_neighbors and
# linkage_pka/FINDINGS.md), where a real finding showed that relaxing
# only the titratable site(s) themselves was insufficient. Same
# moving-atom-set convention as CHI_ATOMS above (see that comment): the
# axis-end heavy atom's own directly-attached hydrogens stay fixed, every
# atom downstream of the pivot heavy atom (including its own hydrogens)
# moves. Atom names verified directly against PDB2PQR's bundled AMBER.DAT
# (the same file load_amber_charges() parses), not assumed from memory.
EXTRA_CHI_ATOMS = {
    "SER": [
        (("N", "CA", "CB", "OG"), ("OG", "HG")),
    ],
    "THR": [
        (("N", "CA", "CB", "OG1"), ("OG1", "HG1", "CG2", "HG21", "HG22", "HG23")),
    ],
    "CYS": [
        (("N", "CA", "CB", "SG"), ("SG", "HG")),
    ],
    "VAL": [
        (("N", "CA", "CB", "CG1"), ("CG1", "HG11", "HG12", "HG13", "CG2", "HG21", "HG22", "HG23")),
    ],
    "LEU": [
        (("N", "CA", "CB", "CG"), ("CG", "HG", "CD1", "HD11", "HD12", "HD13", "CD2", "HD21", "HD22", "HD23")),
        (("CA", "CB", "CG", "CD1"), ("CD1", "HD11", "HD12", "HD13", "CD2", "HD21", "HD22", "HD23")),
    ],
    "ILE": [
        (("N", "CA", "CB", "CG1"), ("CG1", "HG12", "HG13", "CD1", "HD11", "HD12", "HD13", "CG2", "HG21", "HG22", "HG23")),
        (("CA", "CB", "CG1", "CD1"), ("CD1", "HD11", "HD12", "HD13")),
    ],
    "MET": [
        (("N", "CA", "CB", "CG"), ("CG", "HG2", "HG3", "SD", "CE", "HE1", "HE2", "HE3")),
        (("CA", "CB", "CG", "SD"), ("SD", "CE", "HE1", "HE2", "HE3")),
        (("CB", "CG", "SD", "CE"), ("CE", "HE1", "HE2", "HE3")),
    ],
    "PHE": [
        (("N", "CA", "CB", "CG"), ("CG", "CD1", "HD1", "CD2", "HD2", "CE1", "HE1", "CE2", "HE2", "CZ", "HZ")),
        (("CA", "CB", "CG", "CD1"), ("CD1", "HD1", "CE1", "HE1", "CZ", "HZ", "CD2", "HD2", "CE2", "HE2")),
    ],
    "TYR": [
        (("N", "CA", "CB", "CG"), ("CG", "CD1", "HD1", "CD2", "HD2", "CE1", "HE1", "CE2", "HE2", "CZ", "OH", "HH")),
        (("CA", "CB", "CG", "CD1"), ("CD1", "HD1", "CE1", "HE1", "CZ", "OH", "HH", "CD2", "HD2", "CE2", "HE2")),
    ],
    "TRP": [
        (("N", "CA", "CB", "CG"),
         ("CG", "CD1", "HD1", "CD2", "NE1", "HE1", "CE2", "CE3", "HE3", "CZ2", "HZ2", "CZ3", "HZ3", "CH2", "HH2")),
        (("CA", "CB", "CG", "CD1"),
         ("CD1", "HD1", "NE1", "HE1", "CE2", "CZ2", "HZ2", "CH2", "HH2", "CZ3", "HZ3", "CE3", "HE3", "CD2")),
    ],
    "ASN": [
        (("N", "CA", "CB", "CG"), ("CG", "OD1", "ND2", "HD21", "HD22")),
        (("CA", "CB", "CG", "OD1"), ("OD1", "ND2", "HD21", "HD22")),
    ],
    "GLN": [
        (("N", "CA", "CB", "CG"), ("CG", "HG2", "HG3", "CD", "OE1", "NE2", "HE21", "HE22")),
        (("CA", "CB", "CG", "CD"), ("CD", "OE1", "NE2", "HE21", "HE22")),
        (("CB", "CG", "CD", "OE1"), ("OE1", "NE2", "HE21", "HE22")),
    ],
}


def _package_version(name: str) -> str:
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version(name)
    except PackageNotFoundError:
        return "unknown"

N_REPORTED_CHI = 2  # spec only requires chi1/chi2 to be reported/optimized
STAGGERED_ANGLES_DEG = (-60.0, 60.0, 180.0)  # gauche-, gauche+, trans


def _dihedral_deg(p0, p1, p2, p3) -> float:
    """Dihedral angle in degrees defined by four points (standard formula,
    sign convention matching IUPAC/PDB chi-angle definitions)."""
    b0 = p0 - p1
    b1 = p2 - p1
    b2 = p3 - p2
    b1 = b1 / np.linalg.norm(b1)
    v = b0 - np.dot(b0, b1) * b1
    w = b2 - np.dot(b2, b1) * b1
    x = np.dot(v, w)
    y = np.dot(np.cross(b1, v), w)
    return float(np.degrees(np.arctan2(y, x)))


def _rotate_about_axis(points: np.ndarray, axis_point: np.ndarray, axis_dir: np.ndarray, angle_deg: float) -> np.ndarray:
    """Rodrigues' rotation formula: rotate ``points`` (N,3) about the axis
    through ``axis_point`` with direction ``axis_dir``, by ``angle_deg``."""
    axis_dir = axis_dir / np.linalg.norm(axis_dir)
    theta = np.radians(angle_deg)
    p = points - axis_point
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    rotated = (
        p * cos_t
        + np.cross(axis_dir, p) * sin_t
        + axis_dir * (p @ axis_dir)[:, None] * (1 - cos_t)
    )
    return rotated + axis_point


@dataclass
class _ResidueAtomIndex:
    resname: str
    resnum: int
    name_to_index: dict  # atom name -> global atom index (into the positions array)


def _index_residues(topology) -> dict:
    """Map author resnum (as int, from PDB numbering preserved by
    PDBFixer/Modeller) -> _ResidueAtomIndex, for every ionizable residue."""
    out = {}
    for res in topology.residues():
        if res.name not in IONIZABLE_RESNAMES:
            continue
        name_to_index = {atom.name: atom.index for atom in res.atoms()}
        resnum = int(res.id)
        out[resnum] = _ResidueAtomIndex(resname=res.name, resnum=resnum, name_to_index=name_to_index)
    return out


def measure_chi(positions_ang: np.ndarray, res_index: _ResidueAtomIndex, chi_i: int) -> float:
    """Current value (degrees) of chi_{chi_i+1} for one residue."""
    defining_atoms, _ = CHI_ATOMS[res_index.resname][chi_i]
    pts = [positions_ang[res_index.name_to_index[n]] for n in defining_atoms]
    return _dihedral_deg(*pts)


def _set_chi(positions_ang: np.ndarray, res_index: _ResidueAtomIndex, chi_i: int, target_deg: float) -> np.ndarray:
    """Return a copy of ``positions_ang`` with chi_{chi_i+1} of one residue
    rotated to ``target_deg``, by rotating its moving-atom set about the
    existing bond axis (preserves all bond lengths/angles -- this edits the
    dihedral only, it does not rebuild idealized geometry)."""
    defining_atoms, moving_names = CHI_ATOMS[res_index.resname][chi_i]
    current = measure_chi(positions_ang, res_index, chi_i)
    delta = target_deg - current

    axis_point = positions_ang[res_index.name_to_index[defining_atoms[1]]]
    axis_end = positions_ang[res_index.name_to_index[defining_atoms[2]]]
    axis_dir = axis_end - axis_point

    moving_idx = [res_index.name_to_index[n] for n in moving_names if n in res_index.name_to_index]
    out = positions_ang.copy()
    out[moving_idx] = _rotate_about_axis(positions_ang[moving_idx], axis_point, axis_dir, delta)
    return out


@dataclass
class RotamerChoice:
    resnum: int
    resname: str
    chi_as_modeled: list      # degrees, length = n chi angles this residue has (reported ones only)
    chi_chosen: list          # degrees, same length
    energy_kj_per_mol: dict   # {(chi1_deg, chi2_deg): energy} for every candidate scored


@dataclass
class PrepResult:
    topology: object
    positions_ang: np.ndarray          # (n_atoms, 3) Angstrom, final (post rotamer opt + minimization)
    positions_ang_pre_minimization: np.ndarray
    rotamer_choices: dict              # resnum -> RotamerChoice
    ca_displacement_ang: dict          # resnum -> float, |CA after minimization - CA before|
    strained_residues: list            # resnums whose CA displacement exceeded the tolerance
    ca_tolerance_ang: float
    tool_versions: dict


def _make_context(topology, forcefield, cutoff_nm: float = 1.2, threads: str = "0"):
    """Build a scoring Context on the compiled CPU platform (the pure-Python
    'Reference' platform is unusably slow for a system this size) with a
    finite nonbonded cutoff. Rotamer selection only needs *relative* energies
    among local candidates for one side chain at a time -- interactions
    beyond ~1.2 nm change negligibly between candidates and dominate the cost
    under NoCutoff (which is exact but O(N^2) and was untimeoutably slow here
    even for 5 residues on this ~5770-atom system)."""
    import openmm
    from openmm import unit

    system = forcefield.createSystem(
        topology, nonbondedMethod=openmm.app.CutoffNonPeriodic,
        nonbondedCutoff=cutoff_nm * unit.nanometer, constraints=None,
    )
    integrator = openmm.VerletIntegrator(1.0 * unit.femtoseconds)
    platform = openmm.Platform.getPlatformByName("CPU")
    properties = {"Threads": threads}  # "0" = let OpenMM pick
    context = openmm.Context(system, integrator, platform, properties)
    return system, context


def _energy_kj_mol(context, topology, positions_ang: np.ndarray) -> float:
    from openmm import unit

    context.setPositions((positions_ang * unit.angstrom).value_in_unit(unit.nanometer) * unit.nanometer)
    state = context.getState(getEnergy=True)
    return state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)


def optimize_rotamers(topology, positions_ang: np.ndarray, resnums, context) -> dict:
    """For each requested ionizable residue, enumerate the 3x3 staggered
    chi1/chi2 combinations, score each by single-point OpenMM potential
    energy of the *whole system* with that residue's side chain swapped in,
    and keep the lowest-energy combination. Residues are optimized in a
    single sequential pass in the order given (each accepted choice is
    visible to subsequent residues' scoring) -- a documented simplification
    relative to a fully self-consistent multi-pass packing algorithm.
    """
    residues = _index_residues(topology)
    positions_ang = positions_ang.copy()
    choices = {}

    for resnum in resnums:
        resnum = int(resnum)  # OpenMM residue.id (and hence _index_residues' keys) are ints;
        # coerce here so str/int callers both work, and so a genuinely wrong
        # resnum raises via the KeyError below instead of silently no-op'ing.
        if resnum not in residues:
            raise KeyError(
                f"resnum {resnum} is not an ionizable ({sorted(IONIZABLE_RESNAMES)}) residue "
                f"in this structure -- check ionizable_resnums against the actual residue list"
            )
        res_index = residues[resnum]
        n_chi = min(N_REPORTED_CHI, len(CHI_ATOMS[res_index.resname]))
        chi_as_modeled = [measure_chi(positions_ang, res_index, k) for k in range(n_chi)]

        if n_chi == 1:
            candidates = [(a,) for a in STAGGERED_ANGLES_DEG]
        else:
            candidates = [(a, b) for a in STAGGERED_ANGLES_DEG for b in STAGGERED_ANGLES_DEG]

        energies = {}
        best_energy = np.inf
        best_positions = positions_ang
        best_chi = tuple(chi_as_modeled)
        for cand in candidates:
            trial = positions_ang
            for k, target in enumerate(cand):
                trial = _set_chi(trial, res_index, k, target)
            e = _energy_kj_mol(context, topology, trial)
            energies[cand] = e
            if e < best_energy:
                best_energy = e
                best_positions = trial
                best_chi = cand

        positions_ang = best_positions
        choices[resnum] = RotamerChoice(
            resnum=resnum, resname=res_index.resname,
            chi_as_modeled=chi_as_modeled, chi_chosen=list(best_chi),
            energy_kj_per_mol=energies,
        )

    return positions_ang, choices


def minimize_structure(topology, positions_ang: np.ndarray, system, context,
                        restrain_backbone: bool = True, restraint_k_kj_mol_nm2: float = 1000.0,
                        max_iterations: int = 200):
    """Short backbone-restrained energy minimization (a local-minimum
    scoring-function search, not an MD trajectory). Returns final positions
    in Angstrom."""
    import openmm
    from openmm import unit

    if restrain_backbone:
        force = openmm.CustomExternalForce("0.5*k*((x-x0)^2+(y-y0)^2+(z-z0)^2)")
        force.addGlobalParameter("k", restraint_k_kj_mol_nm2 * unit.kilojoule_per_mole / unit.nanometer ** 2)
        force.addPerParticleParameter("x0")
        force.addPerParticleParameter("y0")
        force.addPerParticleParameter("z0")
        backbone_names = {"N", "CA", "C", "O"}
        for atom in topology.atoms():
            if atom.name in backbone_names:
                pos_nm = (positions_ang[atom.index] * unit.angstrom).value_in_unit(unit.nanometer)
                force.addParticle(atom.index, pos_nm)
        system.addForce(force)
        context.reinitialize()

    context.setPositions((positions_ang * unit.angstrom).value_in_unit(unit.nanometer) * unit.nanometer)
    openmm.LocalEnergyMinimizer.minimize(context, maxIterations=max_iterations)
    state = context.getState(getPositions=True)
    positions_nm = state.getPositions(asNumpy=True).value_in_unit(unit.nanometer)
    return np.asarray(positions_nm) * 10.0  # nm -> Angstrom


def run_structure_prep(pdb_path: str, ionizable_resnums=None, ph: float = 7.0,
                        ca_tolerance_ang: float = 0.5, minimize: bool = True) -> PrepResult:
    """Full step-1 pipeline: PDBFixer completion -> OpenMM protonation at
    ``ph`` -> per-site chi1/chi2 rotamer optimization (ionizable residues
    only, or every ionizable residue if ``ionizable_resnums`` is None) ->
    restrained whole-structure minimization -> per-residue CA-displacement
    strain flagging.
    """
    import openmm
    import pdbfixer
    from openmm import app, unit

    fixer = pdbfixer.PDBFixer(filename=str(pdb_path))
    fixer.findMissingResidues()
    fixer.findMissingAtoms()
    fixer.addMissingAtoms()

    modeller = app.Modeller(fixer.topology, fixer.positions)
    modeller.delete([a for a in modeller.topology.atoms() if a.element is not None and a.element.symbol == "H"])
    forcefield = app.ForceField("amber14-all.xml", "implicit/gbn2.xml")
    modeller.addHydrogens(forcefield, pH=ph)

    positions_ang = np.asarray(
        modeller.positions.value_in_unit(unit.angstrom)
        if hasattr(modeller.positions, "value_in_unit")
        else [[c.x, c.y, c.z] for c in modeller.positions]
    )
    if not hasattr(modeller.positions, "value_in_unit"):
        positions_ang = positions_ang * 10.0  # nm -> Angstrom fallback

    if ionizable_resnums is None:
        ionizable_resnums = sorted(_index_residues(modeller.topology).keys())

    system, context = _make_context(modeller.topology, forcefield)

    positions_pre_min, rotamer_choices = optimize_rotamers(modeller.topology, positions_ang, ionizable_resnums, context)

    ca_before = {
        int(res.id): positions_pre_min[[a.index for a in res.atoms() if a.name == "CA"][0]]
        for res in modeller.topology.residues()
        if any(a.name == "CA" for a in res.atoms())
    }

    if minimize:
        final_positions = minimize_structure(modeller.topology, positions_pre_min, system, context)
    else:
        final_positions = positions_pre_min

    ca_displacement = {}
    strained = []
    for res in modeller.topology.residues():
        ca_atoms = [a.index for a in res.atoms() if a.name == "CA"]
        if not ca_atoms:
            continue
        resnum = int(res.id)
        disp = float(np.linalg.norm(final_positions[ca_atoms[0]] - ca_before[resnum]))
        ca_displacement[resnum] = disp
        if disp > ca_tolerance_ang:
            strained.append(resnum)

    return PrepResult(
        topology=modeller.topology,
        positions_ang=final_positions,
        positions_ang_pre_minimization=positions_pre_min,
        rotamer_choices=rotamer_choices,
        ca_displacement_ang=ca_displacement,
        strained_residues=strained,
        ca_tolerance_ang=ca_tolerance_ang,
        tool_versions={
            "pdbfixer": _package_version("pdbfixer"),
            "openmm": openmm.version.full_version,
            "forcefield": "amber14-all.xml + implicit/gbn2.xml",
            "protonation_ph": ph,
            "rotamer_method": "staggered chi1/chi2 (+-60/180 deg) x OpenMM single-point scoring, "
                               "NOT the Dunbrack/Lovell-Richardson empirical library (unverifiable "
                               "in this environment -- see module docstring)",
        },
    )
