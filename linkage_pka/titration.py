"""Per-site pKa via Poisson-Boltzmann, with pairwise site-site coupling --
pipeline spec step 3.

Do NOT use model-compound pKa as the answer (per the pipeline spec) --
it is only the reference state a residue's *shift* is measured against.
The actual per-site intrinsic pKa is

    pKa_i = model_pKa_i - DeltaDeltaG_i / (RT ln10)

where DeltaDeltaG_i is the difference between (a) the electrostatic
energy of turning that site's charge from deprotonated to protonated
*in the protein/membrane environment* (all other titratable sites fixed
at their AMBER-default reference protonation state -- the standard
Bashford-Karplus "background" approximation) and (b) the same
protonation-energy difference for the isolated residue in a small model
compound in pure water. Site-site coupling W_ij (needed because the
histidine shell and buried carboxylates interact) is the interaction
energy between two sites' charge differences, isolated via a standard
double-difference of four whole-system PB energies -- this cancels each
site's own self-energy and any energy shared with the fixed background,
leaving only the i<->j interaction.

Charges for the protonated/deprotonated (or, for His, HIP-vs-HIE)
microstates come from PDB2PQR's own bundled AMBER ff99 charge set
(AMBER.DAT) -- real, versioned force-field data, not remembered/
fabricated numbers (see ``load_amber_charges``). Model-compound pKa
values reuse ``wsme_gpcr.structure.DEFAULT_PKA`` (already the working
reference table elsewhere in this repository) rather than introducing a
second, potentially inconsistent table.

Arg is excluded from active titration: PDB2PQR's AMBER.DAT has no neutral
guanidinium (ARN) variant (confirmed by inspection -- zero matching
entries), consistent with Arg's very high model pKa (~12.5) meaning it is
essentially never deprotonated across the pH 5-8 range this pipeline
scans. Its contribution to Delta_n_H is therefore always ~0 and it is
reported at its fixed model pKa rather than computed.

Per-microstate conformational sampling
---------------------------------------
Every PB energy function below takes an ``optimize_rotamer`` flag (default
False, preserving the original rigid-geometry behavior). When True, each
microstate's target residue gets its own chi1/chi2 rotamer re-selected
(see ``optimize_rotamer_for_microstate``) for *that* charge state, before
the PB solve -- rather than reusing one fixed geometry (from
``structure_prep``'s single, pre-titration rotamer pass) across every
protonation state. This exists because different protonation states often
favor different side-chain packing, and a single rigid structure is a
worse approximation exactly when nearby charges are close enough for that
to matter -- which is also when coupling matters most. Directly motivated
by a real finding in this pipeline's development: a tightly-clustered
4-residue GPCR loop's intrinsic pKa's, computed on ``structure_prep``'s
single fixed geometry, converged (on PB grid refinement, ruling out a
numerical artifact) to shifts of order -12 to -16 pKa units -- far beyond
anything documented for buried ionizable groups (the most extreme
published shifts, engineered cavity Lys in staphylococcal nuclease, are
~5 units) -- consistent with the known failure mode of single-rigid-
structure continuum PB for closely-packed charged clusters that
conformational-sampling methods (e.g. MCCE's multi-rotamer treatment)
exist specifically to avoid. Per the pipeline spec's own guardrail
("Report ... with-rotamer-relaxation and without, every time"), this is
implemented as an explicit, comparable sensitivity axis, not a silent
default change -- existing rigid-geometry behavior and every existing
test are unaffected unless a caller opts in.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from wsme_gpcr.structure import DEFAULT_PKA
from .structure_prep import CHI_ATOMS, STAGGERED_ANGLES_DEG, _dihedral_deg, _rotate_about_axis

R_KJ_PER_MOL_K = 8.31446261815324e-3
LN10 = np.log(10.0)

# Real AMBER ff99 protonated/deprotonated residue-name pairs (verified
# against PDB2PQR's own bundled AMBER.DAT -- see module docstring). His is
# treated as a 2-state HIP (protonated, both ring N's carry H) <-> HIE
# (neutral tautomer, H on NE2 only) system; the HID tautomer (H on ND1
# only) is a documented simplification not modeled here -- see
# ``compute_site_energy`` for how this affects the reported pKa.
TITRATABLE_RESIDUES = {
    "ASP": {"protonated_resname": "ASH", "deprotonated_resname": "ASP", "titratable_h": "HD2"},
    "GLU": {"protonated_resname": "GLH", "deprotonated_resname": "GLU", "titratable_h": "HE2"},
    "HIS": {"protonated_resname": "HIP", "deprotonated_resname": "HIE", "titratable_h": "HD1"},
    "LYS": {"protonated_resname": "LYS", "deprotonated_resname": "LYN", "titratable_h": "HZ1"},
}


def _amber_dat_path() -> Path:
    import pdb2pqr
    return Path(pdb2pqr.__file__).parent / "dat" / "AMBER.DAT"


def load_amber_charges() -> dict:
    """Parse PDB2PQR's bundled AMBER.DAT (AMBER ff99 partial charges) into
    {resname: {atomname: (charge, radius)}}."""
    charges = {}
    for line in _amber_dat_path().read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 5:
            continue
        resname, atomname, charge, radius = parts[0], parts[1], parts[2], parts[3]
        charges.setdefault(resname, {})[atomname] = (float(charge), float(radius))
    return charges


@dataclass
class PqrAtom:
    serial: int
    name: str
    resname: str
    resnum: int
    x: float
    y: float
    z: float
    charge: float
    radius: float
    chain: str = "A"


def read_pqr(path) -> list:
    """Parse a PDB2PQR-format .pqr file (whitespace-delimited; PDB2PQR's
    own output omits the chain column, so both 9- and 10-field ATOM lines
    are accepted)."""
    atoms = []
    for line in Path(path).read_text().splitlines():
        if not (line.startswith("ATOM") or line.startswith("HETATM")):
            continue
        parts = line.split()
        # parts[0]=ATOM, [1]=serial, [2]=name, [3]=resname, then either
        # [4]=resnum (no chain) or [4]=chain,[5]=resnum
        if parts[4].lstrip("-").isdigit():
            chain = "A"
            resnum, x, y, z, charge, radius = parts[4:10]
        else:
            chain = parts[4]
            resnum, x, y, z, charge, radius = parts[5:11]
        atoms.append(PqrAtom(
            serial=int(parts[1]), name=parts[2], resname=parts[3], chain=chain,
            resnum=int(resnum), x=float(x), y=float(y), z=float(z),
            charge=float(charge), radius=float(radius),
        ))
    return atoms


def write_pqr(path, atoms: list):
    lines = []
    for i, a in enumerate(atoms, start=1):
        lines.append(
            f"ATOM  {i:>5d} {a.name:<4s} {a.resname:<4s} {a.resnum:>4d}    "
            f"{a.x:8.3f}{a.y:8.3f}{a.z:8.3f} {a.charge:7.4f} {a.radius:6.4f}"
        )
    lines.append("TER")
    lines.append("END")
    Path(path).write_text("\n".join(lines) + "\n")


def build_microstate(base_atoms: list, resnum: int, resname: str, protonated: bool,
                      amber_charges: dict, extra_h_position=None) -> list:
    """Return a new atom list with residue ``resnum``'s charges/radii
    swapped to the AMBER protonated or deprotonated variant of ``resname``.
    All other residues are left exactly as given (the fixed-background
    approximation).

    ``extra_h_position`` supplies the 3D coordinate for the titratable H
    when building the *protonated* microstate and ``base_atoms`` doesn't
    already contain it (e.g. because the reference structure was prepared
    at a pH where this site defaults to deprotonated) -- typically taken
    from a structure prepared at low pH via ``structure_prep``.
    """
    info = TITRATABLE_RESIDUES[resname]
    target_resname = info["protonated_resname"] if protonated else info["deprotonated_resname"]
    h_name = info["titratable_h"]
    charge_table = amber_charges[target_resname]

    out = []
    h_atom = None
    last_atom_this_residue = None
    for atom in base_atoms:
        if atom.resnum != resnum:
            out.append(atom)
            continue
        last_atom_this_residue = atom
        if atom.name == h_name:
            h_atom = atom
            if not protonated:
                continue  # drop the titratable H for the deprotonated microstate
        if atom.name not in charge_table:
            raise KeyError(f"{target_resname} has no AMBER charge entry for atom {atom.name!r} "
                            f"(resnum {resnum}) -- base_atoms may not match the expected AMBER atom naming")
        charge, radius = charge_table[atom.name]
        out.append(replace(atom, charge=charge, radius=radius, resname=target_resname))

    if protonated and h_atom is None:
        if extra_h_position is None:
            raise ValueError(f"resnum {resnum} ({resname}) is missing its titratable H ({h_name}) "
                              f"in base_atoms and no extra_h_position was given to add it")
        if last_atom_this_residue is None:
            raise KeyError(f"resnum {resnum} not found in base_atoms")
        charge, radius = charge_table[h_name]
        out.append(PqrAtom(
            serial=-1, name=h_name, resname=target_resname, resnum=resnum,
            x=float(extra_h_position[0]), y=float(extra_h_position[1]), z=float(extra_h_position[2]),
            charge=charge, radius=radius, chain=last_atom_this_residue.chain,
        ))

    return out


# Geometric construction for the titratable hydrogen added when building a
# protonated microstate from a base structure that doesn't already carry it:
# place it a standard bond length from its parent heavy atom, pointing away
# from the average position of that parent's other nearby bonded neighbors
# (a simple, defensible heuristic -- avoids gross steric clashes and points
# in a chemically reasonable direction -- not a substitute for real QM/MM
# geometry optimization, which is unnecessary here since PB electrostatics
# is dominated by which side of the molecule the charge sits on, not
# sub-degree bond-angle precision).
_TITRATABLE_H_GEOMETRY = {
    "ASP": {"parent": "OD2", "away_from": ("CG", "OD1"), "bond_length": 0.96},
    "GLU": {"parent": "OE2", "away_from": ("CD", "OE1"), "bond_length": 0.96},
    "HIS": {"parent": "ND1", "away_from": ("CG", "CE1"), "bond_length": 1.01},
    "LYS": {"parent": "NZ", "away_from": ("CE", "HZ2", "HZ3"), "bond_length": 1.01},
}


def place_titratable_hydrogen(atoms: list, resnum: int, resname: str) -> np.ndarray:
    """3D position for the titratable hydrogen of ``resname`` at ``resnum``,
    for use as ``build_microstate``'s ``extra_h_position`` when the base
    structure doesn't already carry it."""
    geom = _TITRATABLE_H_GEOMETRY[resname]
    res_atoms = {a.name: a for a in atoms if a.resnum == resnum}
    if geom["parent"] not in res_atoms:
        raise KeyError(f"resnum {resnum} ({resname}) is missing its titratable-H parent atom {geom['parent']!r}")
    parent_pos = np.array([res_atoms[geom["parent"]].x, res_atoms[geom["parent"]].y, res_atoms[geom["parent"]].z])

    away_positions = [
        np.array([res_atoms[name].x, res_atoms[name].y, res_atoms[name].z])
        for name in geom["away_from"] if name in res_atoms
    ]
    if not away_positions:
        raise KeyError(f"resnum {resnum} ({resname}) has none of its expected neighbor atoms "
                        f"{geom['away_from']} to orient the titratable H away from")

    direction = parent_pos - np.mean(away_positions, axis=0)
    norm = np.linalg.norm(direction)
    if norm < 1e-6:
        raise ValueError(f"resnum {resnum} ({resname}): degenerate geometry, parent atom coincides "
                          f"with the centroid of its neighbors -- cannot orient the titratable H")
    return parent_pos + geom["bond_length"] * (direction / norm)


# Coulomb's constant, ke^2, in units of kJ*Angstrom/(mol*elementary_charge^2)
# -- i.e. E[kJ/mol] = COULOMB_CONSTANT_KJ_ANG_PER_MOL_E2 * q1*q2 / (dielectric * r[Angstrom]).
# Derived from 1/(4*pi*eps0) = 8.9875517923e9 N*m^2/C^2, converted via
# elementary charge e=1.602176634e-19 C, N_A=6.02214076e23 /mol, 1 m=1e10 A,
# 1 J = 1e-3 kJ: ke^2 * N_A * 1e10 * 1e-3 = 1389.354...
COULOMB_CONSTANT_KJ_ANG_PER_MOL_E2 = 1389.35457696


def _pairwise_coulomb_energy(moving_coords: np.ndarray, moving_charges: np.ndarray,
                              other_coords: np.ndarray, other_charges: np.ndarray,
                              dielectric: float, distance_cutoff_ang: float = None) -> float:
    """Cheap, non-APBS pairwise Coulomb energy (kJ/mol) between a moving
    atom set and the rest of the system -- a fast proxy scoring function
    for *ranking* candidate rotamers (not a substitute for the real
    Poisson-Boltzmann solvation energy computed afterward on the geometry
    it selects). O(n_moving * n_other); a distance cutoff keeps this cheap
    for large truncated environments since only relative ranking among
    candidates for one residue's local geometry is needed."""
    if len(other_coords) == 0:
        return 0.0  # nothing to interact with (e.g. an isolated single-residue test fixture)
    diff = moving_coords[:, None, :] - other_coords[None, :, :]  # (n_moving, n_other, 3)
    dist = np.maximum(np.linalg.norm(diff, axis=2), 1e-3)  # avoid singularities
    mask = dist <= distance_cutoff_ang if distance_cutoff_ang is not None else np.ones_like(dist, dtype=bool)
    qq = moving_charges[:, None] * other_charges[None, :]
    return float(np.sum(np.where(mask, COULOMB_CONSTANT_KJ_ANG_PER_MOL_E2 * qq / (dielectric * dist), 0.0)))


def optimize_rotamer_for_microstate(atoms: list, resnum: int, resname: str,
                                     candidate_angles=STAGGERED_ANGLES_DEG,
                                     dielectric: float = 2.0, distance_cutoff_ang: float = 15.0) -> tuple:
    """Re-select one residue's chi1/chi2 rotamer for its CURRENT charge
    state -- call this after ``build_microstate`` has already swapped
    ``resnum`` to its protonated/deprotonated AMBER charges, so the
    scoring reflects that specific microstate's electrostatics, not
    whatever state the structure happened to be prepared in (see the
    module docstring's "Per-microstate conformational sampling" section
    for why this matters).

    Reuses ``structure_prep.py``'s chi-angle geometry (``CHI_ATOMS``) and
    the same staggered gauche-/gauche+/trans candidate set (for the same
    reason documented there: the empirical Dunbrack/Lovell-Richardson
    rotamer library could not be verified/fetched in this environment) --
    but scores candidates by cheap pairwise Coulomb energy
    (``_pairwise_coulomb_energy``) rather than OpenMM single-point energy,
    since this module's ``PqrAtom`` representation carries AMBER partial
    charges directly and has no OpenMM force-field context of its own.

    ASP/GLU fixup: their titratable hydrogen (HD2/HE2) is not part of
    ``CHI_ATOMS``' moving-atom sets (that table predates per-microstate
    titration and was built for ``structure_prep``'s single pass, where
    these residues are typically still deprotonated) -- so if this
    microstate is protonated and already carries that H, its position is
    recomputed fresh via ``place_titratable_hydrogen`` after rotation
    rather than left desynced from the newly-rotated OD2/OE2. HIS
    (chi1/chi2) and LYS (chi4, beyond the reported chi1/chi2 but included
    structurally) already carry their titratable H inside ``CHI_ATOMS``'
    own moving sets and need no such fixup.

    Returns ``(new_atoms, chosen_chi_deg)``.
    """
    if resname not in CHI_ATOMS:
        raise KeyError(f"no chi-angle geometry defined for {resname!r}")
    n_chi = min(2, len(CHI_ATOMS[resname]))

    target_idx = [i for i, a in enumerate(atoms) if a.resnum == resnum]
    if not target_idx:
        raise KeyError(f"resnum {resnum} not found in atoms")
    name_to_local = {atoms[i].name: j for j, i in enumerate(target_idx)}
    base_coords = np.array([[atoms[i].x, atoms[i].y, atoms[i].z] for i in target_idx])
    target_charges = np.array([atoms[i].charge for i in target_idx])

    other_idx = [i for i, a in enumerate(atoms) if a.resnum != resnum]
    other_coords = np.array([[atoms[i].x, atoms[i].y, atoms[i].z] for i in other_idx])
    other_charges = np.array([atoms[i].charge for i in other_idx])

    candidates = [(a,) for a in candidate_angles] if n_chi == 1 else \
        [(a, b) for a in candidate_angles for b in candidate_angles]

    best_energy = np.inf
    best_coords = base_coords
    best_chi = None
    for cand in candidates:
        trial = base_coords.copy()
        for k, target_deg in enumerate(cand):
            defining_atoms, moving_names = CHI_ATOMS[resname][k]
            if not all(n in name_to_local for n in defining_atoms):
                continue  # a chi-defining atom is missing from this residue's atom set (e.g. truncated cap); skip
            pts = [trial[name_to_local[n]] for n in defining_atoms]
            current = _dihedral_deg(*pts)
            delta = target_deg - current
            axis_point, axis_end = trial[name_to_local[defining_atoms[1]]], trial[name_to_local[defining_atoms[2]]]
            moving_local = [name_to_local[n] for n in moving_names if n in name_to_local]
            if moving_local:
                trial[moving_local] = _rotate_about_axis(trial[moving_local], axis_point, axis_end - axis_point, delta)

        energy = _pairwise_coulomb_energy(trial, target_charges, other_coords, other_charges,
                                           dielectric, distance_cutoff_ang)
        if energy < best_energy:
            best_energy, best_coords, best_chi = energy, trial, cand

    new_atoms = list(atoms)
    for local_j, global_i in enumerate(target_idx):
        new_atoms[global_i] = replace(new_atoms[global_i], x=float(best_coords[local_j, 0]),
                                       y=float(best_coords[local_j, 1]), z=float(best_coords[local_j, 2]))

    titratable_h_name = TITRATABLE_RESIDUES.get(resname, {}).get("titratable_h")
    if resname in ("ASP", "GLU") and titratable_h_name is not None:
        h_idx = next((i for i in target_idx if atoms[i].name == titratable_h_name), None)
        if h_idx is not None:
            new_pos = place_titratable_hydrogen(new_atoms, resnum, resname)
            new_atoms[h_idx] = replace(new_atoms[h_idx], x=float(new_pos[0]), y=float(new_pos[1]), z=float(new_pos[2]))

    return new_atoms, list(best_chi)


def charge_delta(resname: str, amber_charges: dict) -> float:
    """Net charge change (protonated minus deprotonated) for one titratable
    residue type -- +1 for acids (Asp/Glu: deprotonated is anionic) and
    His/Lys (protonated is cationic), by construction of the AMBER charge
    sets; returned as the actual sum for a direct sanity check rather than
    assumed."""
    info = TITRATABLE_RESIDUES[resname]
    prot_total = sum(c for c, _ in amber_charges[info["protonated_resname"]].values())
    deprot_total = sum(c for c, _ in amber_charges[info["deprotonated_resname"]].values())
    return prot_total - deprot_total


# --------------------------------------------------------- PB energy calc --
# Sign convention (derived, not assumed -- see also the module docstring):
#
#   Ka = [H+][deprotonated]/[protonated]   (standard acid-dissociation
#   constant; applies uniformly whether "protonated" is neutral, e.g.
#   Asp-COOH, or cationic, e.g. Lys-NH3+ -- it's simply the form carrying
#   the titratable proton)
#   pKa = -log10(Ka),  dG_ionization = -RT ln(Ka) = RT ln(10) * pKa
#
# Moving a residue from a model compound into the protein/membrane
# environment changes only the electrostatic part of dG_ionization:
#
#   pKa_protein = pKa_model + [dG_ion,protein - dG_ion,model] / (RT ln 10)
#   dG_ion(env) = E_env(deprotonated) - E_env(protonated)
#
# where E_env(state) is a *solvation* energy (see compute_solvation_energy),
# not APBS's raw "Total electrostatic energy" -- that raw quantity includes
# each point charge's own Coulombic self-energy, which diverges as the grid
# is refined instead of converging, and does not cancel between a
# deprotonated/protonated pair (they have different charge sets on every
# atom of the residue, not just the titratable proton, since AMBER
# reparameterizes the whole residue per protonation state). Discovered
# during development on a real test case: the raw-energy version of this
# calculation gave a nonsensical intrinsic pKa around -30 that got *worse*
# with finer grids; subtracting a same-spacing reference calculation with
# sdie=pdie (uniform low dielectric, no solvent boundary at all -- the
# standard Born-cycle reference) fixed both problems at once.
#
# Checked against the literature phenomenon this pipeline exists to detect
# (Garcia-Moreno/Isom buried-ionizable series in staphylococcal nuclease,
# Gate A): a buried Lys resists staying charged in a low-dielectric
# environment, pushing E_protein(protonated) up relative to the model, so
# dG_ion,protein becomes more negative than dG_ion,model, giving
# pKa_protein < pKa_model -- a buried Lys should become *more acidic*
# (pKa drops toward neutral), which is exactly the published effect.


@dataclass
class GridParams:
    dime: tuple
    glen: tuple
    gcent: str
    pdie: float
    sdie: float
    ion_strength_m: float
    srad: float = 1.4
    swin: float = 0.3


def _solve_microstate_energy(atoms: list, grid_params: GridParams, work_dir, frame=None,
                              membrane_dielectric: float = 2.0) -> float:
    """PQR -> APBS mg-dummy maps -> (optional membrane splice) -> real
    usemap solve, for one fixed set of atom charges. ``frame=None`` means
    no membrane slab (the model-compound / pure-aqueous case)."""
    from .dielectric_map import compute_dummy_maps, compute_energy_with_maps, splice_membrane_slab, write_maps

    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    pqr_path = work_dir / "state.pqr"
    write_pqr(pqr_path, atoms)

    maps = compute_dummy_maps(
        pqr_path, grid_params.dime, grid_params.glen, grid_params.gcent,
        grid_params.pdie, grid_params.sdie, grid_params.ion_strength_m,
        work_dir / "dummy", srad=grid_params.srad, swin=grid_params.swin,
    )
    if frame is not None:
        maps = splice_membrane_slab(maps, frame, membrane_dielectric=membrane_dielectric, sdie=grid_params.sdie)
    map_paths = write_maps(maps, work_dir / "maps")
    return compute_energy_with_maps(
        pqr_path, map_paths, grid_params.dime, grid_params.glen, grid_params.gcent,
        grid_params.pdie, grid_params.sdie, grid_params.ion_strength_m,
        work_dir / "solve", srad=grid_params.srad, swin=grid_params.swin,
    )


def compute_solvation_energy(atoms: list, grid_params: GridParams, work_dir, frame=None,
                              membrane_dielectric: float = 2.0) -> float:
    """Born-cycle electrostatic solvation energy for one fixed set of atom
    charges: E(solvated, sdie=grid_params.sdie, with the membrane slab if
    ``frame`` is given) minus E(reference, sdie=pdie -- i.e. no dielectric
    boundary anywhere, a uniform low-dielectric medium). This reference
    subtraction is not optional bookkeeping: the raw "Total electrostatic
    energy" APBS reports includes each point charge's own Coulombic
    self-energy, which *diverges* as the grid is refined (confirmed
    empirically during development -- on a small test system, the raw
    ionization energy grew from 202 to 731 to 1405 kJ/mol as grid spacing
    was refined from 0.78 to 0.31 to 0.17 A, instead of converging). That
    self-energy is identical whether or not solvent surrounds the charges,
    so it cancels exactly against this reference at any shared grid
    spacing, leaving the physically meaningful (and grid-convergent)
    solvation contribution -- verified: the same test system's dG_solv went
    from -56.4 to -49.6 kJ/mol (about 12%) over the same spacing range,
    not a divergence.
    """
    e_solv = _solve_microstate_energy(atoms, grid_params, Path(work_dir) / "solv", frame, membrane_dielectric)
    ref_grid_params = replace(grid_params, sdie=grid_params.pdie)
    e_ref = _solve_microstate_energy(atoms, ref_grid_params, Path(work_dir) / "ref", frame=None)
    return e_solv - e_ref


def compute_environment_energies(atoms: list, resnum: int, resname: str, amber_charges: dict,
                                  grid_params: GridParams, work_dir, frame=None,
                                  membrane_dielectric: float = 2.0, extra_h_position=None,
                                  optimize_rotamer: bool = False) -> tuple:
    """Return (E_deprotonated, E_protonated): Born-cycle solvation energy
    (see ``compute_solvation_energy``) of the whole given atom set
    (protein, or an isolated single-residue model compound) with site
    ``resnum`` swapped between its two AMBER microstates, all other atoms
    held fixed.

    ``optimize_rotamer=True`` re-selects ``resnum``'s chi1/chi2 rotamer
    separately for each of the two microstates (see
    ``optimize_rotamer_for_microstate`` and the module docstring's
    "Per-microstate conformational sampling" section) before scoring --
    default False preserves the original single-fixed-geometry behavior.
    """
    work_dir = Path(work_dir)
    deprot_atoms = build_microstate(atoms, resnum, resname, protonated=False, amber_charges=amber_charges)
    prot_atoms = build_microstate(atoms, resnum, resname, protonated=True, amber_charges=amber_charges,
                                   extra_h_position=extra_h_position)
    if optimize_rotamer:
        deprot_atoms, _ = optimize_rotamer_for_microstate(deprot_atoms, resnum, resname)
        prot_atoms, _ = optimize_rotamer_for_microstate(prot_atoms, resnum, resname)

    e_deprot = compute_solvation_energy(deprot_atoms, grid_params, work_dir / "deprot", frame, membrane_dielectric)
    e_prot = compute_solvation_energy(prot_atoms, grid_params, work_dir / "prot", frame, membrane_dielectric)
    return e_deprot, e_prot


@dataclass
class SiteEnergyResult:
    resnum: int
    resname: str
    model_pka: float
    intrinsic_pka: float
    dg_ion_protein: float  # kJ/mol
    dg_ion_model: float    # kJ/mol
    e_protein_deprot: float
    e_protein_prot: float
    e_model_deprot: float
    e_model_prot: float


def build_model_compound_atoms(protein_atoms: list, resnum: int) -> list:
    """Model-compound fragment for one residue: its own atoms, plus the
    real backbone C/O of residue ``resnum - 1`` and N/H of
    ``resnum + 1`` -- i.e. the two peptide bonds this residue actually
    makes, closed off with real, correctly AMBER-parameterized neighbor
    atoms taken directly from the folded structure's own coordinates.

    This matters more than it might look: a residue's own AMBER charges
    differ substantially between its protonated/deprotonated templates
    across the *whole residue*, not just the titratable proton (e.g. ASH's
    backbone N carries -0.4157 e vs ASP's -0.5163 e) -- a fully isolated,
    uncapped residue has nothing to electrostatically stabilize that
    backbone charge redistribution, which was measured directly in this
    module's development to produce a >20 pKa-unit artifact (dG_ion for a
    bare-residue "model" reached +170 kJ/mol on a simple test case,
    physically implausible for what should be a small reference shift).
    Borrowing 2 real neighbor atoms per side is a much simpler fix than
    synthesizing ACE/NME cap geometry from scratch, reuses real coordinates
    instead of invented ones, and closes both dangling amide-like ends.
    Residues at a chain terminus simply get a one-sided cap (whichever
    neighbor exists).
    """
    model_atoms = [a for a in protein_atoms if a.resnum == resnum]
    if not model_atoms:
        raise KeyError(f"resnum {resnum} not found in protein_atoms")
    prev_cap = [a for a in protein_atoms if a.resnum == resnum - 1 and a.name in ("C", "O")]
    next_cap = [a for a in protein_atoms if a.resnum == resnum + 1 and a.name in ("N", "H")]
    return prev_cap + model_atoms + next_cap


def compute_intrinsic_pka(protein_atoms: list, resnum: int, resname: str, frame,
                           protein_grid_params: GridParams, model_grid_params: GridParams, work_dir,
                           amber_charges: dict = None, membrane_dielectric: float = 2.0,
                           temp_k: float = 298.15, extra_h_position=None,
                           optimize_rotamer: bool = False) -> SiteEnergyResult:
    """One site's intrinsic pKa: PB energies in the full protein/membrane
    environment (``frame`` gives the membrane slab; pass ``frame=None`` for
    a soluble-protein calculation with no membrane) minus the same in an
    isolated single-residue model compound (see
    ``build_model_compound_atoms``), referenced against
    ``wsme_gpcr.structure.DEFAULT_PKA``.

    ``optimize_rotamer=True`` re-optimizes this site's rotamer per
    microstate on both the protein and model-compound sides -- see
    ``compute_environment_energies``.
    """
    amber_charges = amber_charges or load_amber_charges()
    work_dir = Path(work_dir)

    e_prot_deprot, e_prot_prot = compute_environment_energies(
        protein_atoms, resnum, resname, amber_charges, protein_grid_params, work_dir / "protein",
        frame=frame, membrane_dielectric=membrane_dielectric, extra_h_position=extra_h_position,
        optimize_rotamer=optimize_rotamer,
    )

    model_atoms = build_model_compound_atoms(protein_atoms, resnum)
    e_model_deprot, e_model_prot = compute_environment_energies(
        model_atoms, resnum, resname, amber_charges, model_grid_params, work_dir / "model",
        frame=None, extra_h_position=extra_h_position, optimize_rotamer=optimize_rotamer,
    )

    RT = R_KJ_PER_MOL_K * temp_k
    dg_ion_protein = e_prot_deprot - e_prot_prot
    dg_ion_model = e_model_deprot - e_model_prot
    model_pka = DEFAULT_PKA[resname]
    intrinsic_pka = model_pka + (dg_ion_protein - dg_ion_model) / (RT * LN10)

    return SiteEnergyResult(
        resnum=resnum, resname=resname, model_pka=model_pka, intrinsic_pka=intrinsic_pka,
        dg_ion_protein=dg_ion_protein, dg_ion_model=dg_ion_model,
        e_protein_deprot=e_prot_deprot, e_protein_prot=e_prot_prot,
        e_model_deprot=e_model_deprot, e_model_prot=e_model_prot,
    )


def compute_pairwise_coupling(protein_atoms: list, resnum_i: int, resname_i: str,
                               resnum_j: int, resname_j: str, frame, grid_params: GridParams, work_dir,
                               amber_charges: dict = None, membrane_dielectric: float = 2.0,
                               extra_h_position_i=None, extra_h_position_j=None,
                               optimize_rotamer: bool = False) -> float:
    """W_ij (kJ/mol): the standard double-difference of four whole-system
    PB energies, isolating the pure electrostatic interaction between site
    i's and site j's charge differences (protonated - deprotonated) --
    cancels each site's own self-energy and any energy shared with the
    fixed background of every other site (the reduced-site/Bashford-Karplus
    coupling term the pipeline spec requires for the coupled histidine-
    shell + buried-carboxylate cluster).

    ``optimize_rotamer=True`` re-optimizes both i's and j's rotamers for
    each of the 4 joint microstates before scoring -- see
    ``optimize_rotamer_for_microstate``.
    """
    amber_charges = amber_charges or load_amber_charges()
    work_dir = Path(work_dir)
    energies = {}
    for i_prot in (False, True):
        for j_prot in (False, True):
            atoms = build_microstate(protein_atoms, resnum_i, resname_i, i_prot, amber_charges,
                                      extra_h_position=extra_h_position_i if i_prot else None)
            atoms = build_microstate(atoms, resnum_j, resname_j, j_prot, amber_charges,
                                      extra_h_position=extra_h_position_j if j_prot else None)
            if optimize_rotamer:
                atoms, _ = optimize_rotamer_for_microstate(atoms, resnum_i, resname_i)
                atoms, _ = optimize_rotamer_for_microstate(atoms, resnum_j, resname_j)
            label = f"i{'p' if i_prot else 'd'}_j{'p' if j_prot else 'd'}"
            energies[(i_prot, j_prot)] = compute_solvation_energy(
                atoms, grid_params, work_dir / label, frame, membrane_dielectric,
            )

    return (energies[(True, True)] - energies[(True, False)]
            - energies[(False, True)] + energies[(False, False)])


def compute_cluster_joint_energies(protein_atoms: list, sites: list, frame, grid_params: GridParams, work_dir,
                                    amber_charges: dict = None, membrane_dielectric: float = 2.0,
                                    extra_h_positions: dict = None, optimize_rotamer: bool = False) -> dict:
    """Whole-system Born-cycle solvation energy E_protein(x) for every one
    of a small cluster's 2^n joint protonation microstates x -- the exact,
    non-perturbative alternative to ``compute_intrinsic_pka`` (which freezes
    every *other* titratable site at a fixed reference charge state while
    computing one site's own pKa -- the standard Bashford-Karplus
    reduced-site approximation) plus ``compute_pairwise_coupling``.

    That reduced-site approximation is known to degrade for *tightly
    interacting* clusters (multiple charged residues within a few Angstrom
    of each other): freezing every other cluster member at its default
    reference state is a poor description of the local electrostatic
    environment exactly when those neighbors are close enough to matter
    most. Discovered directly in this pipeline's development: a real
    4-site GPCR loop cluster (Glu/Asp/Glu/His, all within ~4-12 A of each
    other) gave individual intrinsic pKa's shifted by >20 units from their
    model values -- far beyond anything in the buried-ionizable literature
    (the most extreme published shifts, for engineered cavity Lys in
    staphylococcal nuclease, are ~5 units) -- and the anomaly barely moved
    when the surrounding truncated environment was doubled (30 A -> 40 A
    radius), ruling out truncation as the cause, while still-unconverged
    grid refinement (dime 33->65) moved it by several pKa units without
    resolving it. This function sidesteps the reduced-site approximation
    entirely for small clusters by computing every joint microstate's whole-
    system energy directly, rather than decomposing it into per-site
    intrinsic terms plus pairwise corrections.

    Tested on that same real cluster: removing the reduced-site
    approximation did NOT resolve the anomaly (the joint energies showed
    the same ~90 kJ/mol single-site swing), and further grid refinement
    (dime 65->97) converged cleanly to a still-implausible ~-12 pKa-unit
    shift rather than continuing to move -- ruling out both the
    approximation and (to first order) an unconverged grid as the sole
    cause. See ``optimize_rotamer`` below and the module docstring's
    "Per-microstate conformational sampling" section for the next
    hypothesis this motivated: a single rigid geometry per microstate.

    Returns ``{occupancy: energy}`` where ``occupancy`` is a tuple of bools
    (True=protonated) in the same order as ``sites``, and ``energy`` is the
    Born-cycle solvation energy (kJ/mol, see ``compute_solvation_energy``)
    of the whole given atom set with every site in ``sites`` simultaneously
    set to that occupancy, all other atoms held fixed.

    ``optimize_rotamer=True`` re-optimizes every site's rotamer for each
    joint microstate before scoring -- see
    ``optimize_rotamer_for_microstate``.

    Cost: 2^n calls to ``compute_solvation_energy`` (each itself 2 APBS
    solves: solvated + reference), i.e. 2^(n+1) solves total -- tractable
    only for small n, matching ``multisite.MAX_EXACT_CLUSTER_SIZE``.
    """
    amber_charges = amber_charges or load_amber_charges()
    extra_h_positions = extra_h_positions or {}
    work_dir = Path(work_dir)
    n = len(sites)

    energies = {}
    for state_idx in range(2 ** n):
        occupancy = tuple(bool((state_idx >> i) & 1) for i in range(n))
        atoms = protein_atoms
        for (resnum, resname), protonated in zip(sites, occupancy):
            atoms = build_microstate(atoms, resnum, resname, protonated, amber_charges,
                                      extra_h_position=extra_h_positions.get(resnum) if protonated else None)
        if optimize_rotamer:
            for resnum, resname in sites:
                atoms, _ = optimize_rotamer_for_microstate(atoms, resnum, resname)
        label = "".join("p" if b else "d" for b in occupancy)
        energies[occupancy] = compute_solvation_energy(atoms, grid_params, work_dir / label, frame, membrane_dielectric)

    return energies
