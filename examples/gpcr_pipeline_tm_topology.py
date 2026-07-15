"""
TM helix boundary detection, invariant-core superposition, and TM6
displacement measurement -- built in response to a methodological review
that correctly identified an asymmetry between the GPR68 and GPR132
interpolated-path runs: GPR68 used no superposition, GPR132 used a
Kabsch fit on a manually-chosen residue range (30-355). Both are replaced
here with one symmetric, receptor-agnostic protocol applied identically
to every receptor.

Locating TM helix boundaries
-----------------------------
A literal search for the textbook class-A anchor motifs (D2.50, DRY at
3.49-3.51, W4.50, P5.50, the CWxP tetrad at 6.47-6.50, D/P/Y at 7.49-7.53)
was attempted first and empirically FAILED for W4.50/P5.50/CWxP on both
GPR68 and GPR132 -- neither receptor's raw sequence contains a literal
`CW.P` or even `W.P` match anywhere (checked directly; see chat record).
This is a real property of these divergent, non-canonical class-A GPCRs,
not a search bug -- forcing a fit would mean silently mis-assigning a
residue.

What DOES reliably match in both receptors is the DRY-like motif (TM3,
3.49-3.51) and an NPxxY-like motif (TM7, 7.49-7.53) via the class-A
variant sets already used elsewhere in this project. Rather than guess
the other four anchors from sequence alone, this module locates the
actual TM1-TM7 helical segments geometrically (reusing
wsme_gpcr.secondary_structure.assign_secondary_structure, an
already-implemented and validated -- 82% agreement with real STRIDE --
phi/psi-based helix detector), then uses the two reliable sequence
anchors to identify which detected helix is TM3 and which is TM7. TM1,
TM2, TM4, TM5, and TM6 are then assigned by their INVARIANT topological
sequence order relative to TM3 and TM7 (TM1-TM2-TM3-TM4-TM5-TM6-TM7 in
strict N-to-C sequence order is universal to the class-A 7TM fold) --
not by further motif guessing, and not by any hardcoded residue range.
The self-consistency check (TM7's index must be exactly 4 helices after
TM3's) is verified, not assumed, and the run aborts loudly if it fails.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
from Bio.PDB import PDBParser
from Bio.PDB.Superimposer import Superimposer
from Bio.SeqUtils import seq1

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from wsme_gpcr.secondary_structure import (  # noqa: E402
    DsspNotAvailableError, assign_secondary_structure, secondary_structure_from_dssp,
)
from wsme_gpcr.structure import load_structure as load_wsme_structure  # noqa: E402

MIN_HELIX_RUN = 15  # residues; TM helices are ~20-30, loop helical turns are much shorter
N_TM_HELICES = 7


class TopologyError(RuntimeError):
    """Raised when TM helix identification fails or fails a consistency
    check -- never silently guessed."""


def load_seq_resnums(pdb_path, chain_id="A"):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("x", str(pdb_path))
    residues = [r for r in structure[0][chain_id] if r.id[0] == " "]
    resnums = [r.id[1] for r in residues]
    seq = "".join(seq1(r.get_resname()) for r in residues)
    return seq, resnums, structure


def find_dry_anchor(seq, resnums):
    """3.49-3.51 (D/E-R-x), the class-A variant set already used and
    verified elsewhere in this project (GPR68: DRY, GPR132: DRF)."""
    m = re.search("[DE]R[YFWCH]", seq)
    if m is None:
        raise TopologyError("No D/E-R-x (DRY-like) motif found -- cannot anchor TM3")
    return {"3.49": resnums[m.start()], "3.50": resnums[m.start() + 1], "3.51": resnums[m.start() + 2]}


def find_npxxy_anchor(seq, resnums):
    """7.49-7.53 ([N/D/S]-P-x-x-[Y/F/C/H]), the class-A variant set already
    used and verified elsewhere in this project (GPR68: DPVLY, GPR132: DPIIY)."""
    m = re.search("[DNS]P..[YFCH]", seq)
    if m is None:
        raise TopologyError("No [NDS]Pxx[YFCH] (NPxxY-like) motif found -- cannot anchor TM7")
    return {"7.49": resnums[m.start()], "7.50": resnums[m.start() + 1], "7.53": resnums[m.start() + 4]}


def detect_structured_runs(pdb_path, chain_id="A", min_run=MIN_HELIX_RUN, use_dssp=True):
    """Contiguous structured (H/G/E) runs >= min_run residues, in sequence
    order, as (start_resnum, end_resnum) tuples.

    Prefers real DSSP (H-bond-geometry-based, installed via `apt-get
    install dssp` for this run) over the geometric phi/psi heuristic --
    empirically, the geometric heuristic cannot distinguish a genuine
    short loop from a 1-2 residue helix-internal kink (both look like a
    brief break in the phi/psi Ramachandran box), which made automated
    TM-helix counting unreliable (see module docstring). Falls back to
    the geometric heuristic with a warning if mkdssp isn't available.
    """
    wsme_structure = load_wsme_structure(str(pdb_path), chain=chain_id)
    if use_dssp:
        try:
            mask = secondary_structure_from_dssp(str(pdb_path), wsme_structure)
        except DsspNotAvailableError:
            import warnings

            warnings.warn("mkdssp not available, falling back to the less reliable geometric heuristic")
            mask = assign_secondary_structure(wsme_structure)
    else:
        mask = assign_secondary_structure(wsme_structure)
    resnums = wsme_structure.author_resnum

    runs = []
    start = None
    for i, structured in enumerate(mask):
        if structured and start is None:
            start = i
        elif not structured and start is not None:
            if i - start >= min_run:
                runs.append((int(resnums[start]), int(resnums[i - 1])))
            start = None
    if start is not None and len(mask) - start >= min_run:
        runs.append((int(resnums[start]), int(resnums[len(mask) - 1])))
    return runs


def _merge_runs_split_by_anchor(runs, anchor_resnum):
    """If `anchor_resnum` falls in the GAP between two adjacent runs, merge
    those two runs into one. Conserved class-A motifs (DRY's R3.50,
    NPxxY's P7.50/Y7.53) are documented to induce local backbone
    distortion/kinks at exactly their own position -- a real DSSP break
    right at the anchor is evidence of a kink internal to one helix, not
    a genuine loop, so merging here is motif-justified, not a blind
    gap-length heuristic."""
    for i in range(len(runs) - 1):
        lo, hi = runs[i]
        next_lo, next_hi = runs[i + 1]
        if hi < anchor_resnum < next_lo:
            return runs[:i] + [(lo, next_hi)] + runs[i + 2 :]
    return runs


def _run_containing_or_nearest(runs, resnum):
    for lo, hi in runs:
        if lo <= resnum <= hi:
            return runs.index((lo, hi))
    # not strictly inside any run (e.g. anchor sits a residue or two into
    # the following loop) -- fall back to nearest run by boundary distance
    dists = [min(abs(resnum - lo), abs(resnum - hi)) for lo, hi in runs]
    return int(np.argmin(dists))


def identify_tm_helices(pdb_path, chain_id="A"):
    """Returns (tm_ranges, anchors) where tm_ranges is {1: (lo,hi), ..., 7: (lo,hi)}
    and anchors records the DRY/NPxxY positions used and the consistency
    check result. Raises TopologyError if TM3/TM7 can't be anchored or if
    the self-consistency check (TM7 index == TM3 index + 4) fails."""
    seq, resnums, _ = load_seq_resnums(pdb_path, chain_id)
    dry = find_dry_anchor(seq, resnums)
    npxxy = find_npxxy_anchor(seq, resnums)

    runs = detect_structured_runs(pdb_path, chain_id)
    runs = _merge_runs_split_by_anchor(runs, dry["3.50"])
    runs = _merge_runs_split_by_anchor(runs, npxxy["7.53"])
    if len(runs) < N_TM_HELICES:
        raise TopologyError(
            f"Only {len(runs)} structured run(s) >= {MIN_HELIX_RUN} residues found; "
            f"need at least {N_TM_HELICES} to identify a 7TM bundle. Runs: {runs}"
        )

    i3 = _run_containing_or_nearest(runs, dry["3.50"])
    i7 = _run_containing_or_nearest(runs, npxxy["7.53"])

    if i7 - i3 != 4:
        raise TopologyError(
            f"Consistency check FAILED: TM3 anchored to run index {i3} ({runs[i3]}), "
            f"TM7 anchored to run index {i7} ({runs[i7]}), difference is {i7 - i3}, expected 4 "
            f"(TM4, TM5, TM6 in between). Detected runs (after anchor-kink merging): {runs}. "
            f"Refusing to guess -- this receptor's helix detection or motif anchoring needs "
            f"manual review before proceeding."
        )
    if i3 < 2:
        raise TopologyError(
            f"TM3 anchored to run index {i3}, but need at least 2 preceding runs for TM1/TM2. "
            f"Detected runs: {runs}"
        )

    tm_ranges = {
        1: runs[i3 - 2],
        2: runs[i3 - 1],
        3: runs[i3],
        4: runs[i3 + 1],
        5: runs[i3 + 2],
        6: runs[i3 + 3],
        7: runs[i3 + 4],
    }
    anchors = {
        "dry": dry, "npxxy": npxxy,
        "tm3_run_index": i3, "tm7_run_index": i7,
        "all_detected_runs": runs,
    }
    return tm_ranges, anchors


def invariant_core_ca_atoms(bio_structure, chain_id, tm_ranges, exclude=(6,)):
    """CA atoms for every residue in TM1-7 except the excluded helix
    numbers (default: TM6, the principal activation-associated mover),
    in ascending resnum order."""
    chain = bio_structure[0][chain_id]
    keep_ranges = [rng for tm, rng in tm_ranges.items() if tm not in exclude]
    atoms = []
    for res in chain:
        if res.id[0] != " ":
            continue
        resnum = res.id[1]
        if any(lo <= resnum <= hi for lo, hi in keep_ranges) and "CA" in res:
            atoms.append((resnum, res["CA"]))
    atoms.sort(key=lambda t: t[0])
    return [a for _, a in atoms]


def superpose_invariant_core(active_structure, inactive_structure, chain_id, tm_ranges, exclude=(6,)):
    """Kabsch-superposes inactive_structure onto active_structure using
    only the invariant-core CA atoms (TM1-7 minus `exclude`), then applies
    the resulting rotation+translation to EVERY atom of inactive_structure
    in place. Returns the Superimposer (².rms is the fit RMSD)."""
    active_atoms = invariant_core_ca_atoms(active_structure, chain_id, tm_ranges, exclude)
    inactive_atoms = invariant_core_ca_atoms(inactive_structure, chain_id, tm_ranges, exclude)
    if len(active_atoms) != len(inactive_atoms):
        raise TopologyError(
            f"Invariant-core atom count mismatch: active={len(active_atoms)}, inactive={len(inactive_atoms)}"
        )
    sup = Superimposer()
    sup.set_atoms(active_atoms, inactive_atoms)
    sup.apply(list(inactive_structure.get_atoms()))
    return sup


def per_helix_rmsd(active_structure, inactive_structure, chain_id, tm_ranges):
    """CA RMSD for each TM1-7 individually, AFTER whatever superposition
    has already been applied to inactive_structure -- reveals which
    helices actually differ between the two endpoints post-alignment."""
    chain_a = active_structure[0][chain_id]
    chain_i = inactive_structure[0][chain_id]
    a_ca = {r.id[1]: r["CA"].coord for r in chain_a if r.id[0] == " " and "CA" in r}
    i_ca = {r.id[1]: r["CA"].coord for r in chain_i if r.id[0] == " " and "CA" in r}

    result = {}
    for tm, (lo, hi) in tm_ranges.items():
        diffs = [a_ca[rn] - i_ca[rn] for rn in a_ca if lo <= rn <= hi and rn in i_ca]
        if not diffs:
            result[tm] = float("nan")
            continue
        diffs = np.array(diffs)
        result[tm] = float(np.sqrt((diffs**2).sum(axis=1).mean()))
    return result


def tm6_cytoplasmic_displacement(active_structure, inactive_structure, chain_id, tm_ranges, n_residues=3):
    """Displacement of TM6's cytoplasmic-end CA atoms between the two
    endpoint structures, AFTER whatever superposition has already been
    applied -- the hallmark class-A activation metric (outward swing of
    TM6's intracellular tip). By standard GPCR topology (odd TMs run
    extracellular-to-intracellular; TM6 runs intracellular-to-extracellular,
    i.e. its LOWER-resnum end, right after ICL3, is the cytoplasmic one),
    this is the first n_residues of the TM6 range."""
    lo, hi = tm_ranges[6]
    tip_resnums = list(range(lo, min(lo + n_residues, hi + 1)))
    chain_a = active_structure[0][chain_id]
    chain_i = inactive_structure[0][chain_id]
    a_ca = {r.id[1]: r["CA"].coord for r in chain_a if r.id[0] == " "}
    i_ca = {r.id[1]: r["CA"].coord for r in chain_i if r.id[0] == " "}
    diffs = np.array([a_ca[rn] - i_ca[rn] for rn in tip_resnums if rn in a_ca and rn in i_ca])
    if len(diffs) == 0:
        return float("nan"), tip_resnums
    return float(np.sqrt((diffs**2).sum(axis=1).mean())), tip_resnums


def iterative_outlier_rejecting_superposition(active_structure, inactive_structure, chain_id, cutoff_ang=3.0, max_iter=20):
    """Independent cross-check with NO prior about which helices move:
    start from all CA atoms, Kabsch-fit, drop any atom whose post-fit
    distance exceeds cutoff_ang, refit on the survivors, repeat until the
    kept-atom set stops changing. Returns (kept_resnums, final_rms).

    NOTE: operates on a COPY of inactive_structure's coordinates (does not
    mutate the structures passed in) so it can be run purely as a
    diagnostic cross-check without side effects.
    """
    import copy

    active_copy = active_structure.copy()
    inactive_copy = inactive_structure.copy()
    chain_a = active_copy[0][chain_id]
    chain_i = inactive_copy[0][chain_id]
    a_res = {r.id[1]: r for r in chain_a if r.id[0] == " " and "CA" in r}
    i_res = {r.id[1]: r for r in chain_i if r.id[0] == " " and "CA" in r}
    common_resnums = sorted(set(a_res) & set(i_res))

    kept = set(common_resnums)
    for _ in range(max_iter):
        a_atoms = [a_res[rn]["CA"] for rn in sorted(kept)]
        i_atoms = [i_res[rn]["CA"] for rn in sorted(kept)]
        sup = Superimposer()
        sup.set_atoms(a_atoms, i_atoms)
        sup.apply(list(inactive_copy.get_atoms()))

        diffs = {rn: np.linalg.norm(a_res[rn]["CA"].coord - i_res[rn]["CA"].coord) for rn in common_resnums}
        new_kept = {rn for rn in common_resnums if diffs[rn] <= cutoff_ang}
        if new_kept == kept:
            return sorted(kept), sup.rms
        if len(new_kept) < 10:
            # degenerate -- cutoff too strict for this structure pair
            return sorted(kept), sup.rms
        kept = new_kept
    return sorted(kept), sup.rms
