"""Membrane frame and dielectric slab geometry -- pipeline spec step 2.

Computes a membrane-normal axis via PCA over the transmembrane-bundle C-alpha
atoms, oriented (sign-disambiguated) using the two conserved class-A GPCR
anchors R3.50 (DRY motif, TM3 intracellular end) and Y7.53 (NPxxY motif, TM7
intracellular end), then defines a planar low-dielectric slab along that
axis for use by the Poisson-Boltzmann membrane dielectric map.

A note on what "reuse the existing PCA membrane-frame routine" and BW
numbering mean here
-------------------------------------------------------------------------
The pipeline spec calls for reusing an existing PCA membrane-frame routine
and locating R3.50/Y7.53 via Ballesteros-Weinstein (BW) numbering from "the
canonical 323-column trim alignment." Neither that routine nor that
alignment file is present in this repository or environment (confirmed:
`/mnt/project` does not exist here at all, only `/mnt/skills`). Guessing a
BW mapping from a degapped sequence alone is explicitly disallowed by the
pipeline's own guardrails.

What this module does instead, and why it's still defensible: DRY and
NPxxY are among the most conserved motifs in class A GPCRs, and R3.50/
Y7.53 are *defined* as the Arg/Tyr within them -- so a direct regex search
for these motifs in the structure's own sequence locates the same two
residues a BW lookup table would, for these two positions specifically,
without needing the external alignment file. This does NOT substitute for
full BW numbering (used elsewhere in the pipeline spec to label every
ionizable residue in the output tables) -- that still requires the
alignment file and is left unavailable/"BW: not resolved" until it's
provided. The motif search allows for the well-documented DRY->xRY and
NPxxY->DPxxY natural variants (checked against this repository's actual
GPR68 structures, which carry the DPxxY variant, not literal NPxxY).

A note on pLDDT
----------------
The spec's "high-pLDDT C-alpha" selection assumes an AlphaFold-derived
model (pLDDT is an AlphaFold-specific per-residue confidence score, 0-100).
The GPR68 structures used here are GPCRdb homology models whose B-factor
column is not pLDDT (confirmed: it contains physically-invalid values for
a confidence score, e.g. negative entries and short monotonically-
increasing runs consistent with a placeholder/interpolation artifact, not
per-residue confidence). `compute_membrane_frame` detects this
automatically (any B-factor outside [0, 100] disqualifies the column as
pLDDT) and falls back to a secondary-structure helix mask -- transmembrane
segments in a 7TM bundle are, with the minor exception of the short
intracellular helix 8, exactly the alpha-helical stretches. Which method
was used, and why, is recorded in the returned ``MembraneFrame``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np

THREE_TO_ONE = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q",
    "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K",
    "MET": "M", "PHE": "F", "PRO": "P", "SER": "S", "THR": "T", "TRP": "W",
    "TYR": "Y", "VAL": "V",
}

# Permissive but still tightly-anchored motif patterns: the residue that
# BW-numbering would call R3.50/Y7.53 is *defined* as the conserved
# Arg/Tyr in these motifs, so requiring the flanking pattern (not just a
# lone Arg/Tyr) is what makes this a legitimate identification rather than
# a guess. [DE] and [ND] cover the most common natural substitutions at
# the degenerate first position (documented in GPCR sequence literature;
# the exact variant found in each structure's own sequence is recorded in
# MembraneFrame.dry_motif / npxxy_motif, not silently assumed).
DRY_PATTERN = re.compile(r"[DE]R[YFWH]")
NPXXY_PATTERN = re.compile(r"[ND]P..Y")


def _one_letter_sequence(structure) -> str:
    return "".join(THREE_TO_ONE.get(rn, "X") for rn in structure.resname)


def find_r350(structure) -> tuple:
    """Locate R3.50 (the conserved Arg of the DRY/xRY motif). Returns
    (author_resnum, matched_motif_string). Raises ValueError if no motif
    is found (should not happen for an intact class A GPCR TM3-ICL2
    region; a real failure here likely means the DRY-containing loop is
    missing/unmodeled in this structure)."""
    seq = _one_letter_sequence(structure)
    matches = list(DRY_PATTERN.finditer(seq))
    if not matches:
        raise ValueError("no DRY/xRY-like motif found in this structure's sequence -- "
                          "TM3-ICL2 region may be missing or unmodeled")
    m = matches[0]
    if len(matches) > 1:
        import warnings
        warnings.warn(f"{len(matches)} DRY/xRY-like motifs found; using the first "
                       f"({m.group()} at sequence index {m.start()}) -- verify this is TM3, not a coincidental match")
    r_resnum = int(structure.author_resnum[m.start() + 1])
    return r_resnum, m.group()


def find_y753(structure) -> tuple:
    """Locate Y7.53 (the conserved Tyr of the NPxxY/DPxxY motif). Returns
    (author_resnum, matched_motif_string). Raises ValueError if not found."""
    seq = _one_letter_sequence(structure)
    matches = list(NPXXY_PATTERN.finditer(seq))
    if not matches:
        raise ValueError("no NPxxY/DPxxY-like motif found in this structure's sequence -- "
                          "TM7 C-terminal region may be missing or unmodeled")
    m = matches[0]
    if len(matches) > 1:
        import warnings
        warnings.warn(f"{len(matches)} NPxxY/DPxxY-like motifs found; using the first "
                       f"({m.group()} at sequence index {m.start()}) -- verify this is TM7, not a coincidental match")
    y_resnum = int(structure.author_resnum[m.start() + 4])
    return y_resnum, m.group()


def _ca_coord(structure, resnum: int) -> np.ndarray:
    ridx = int(np.where(structure.author_resnum == resnum)[0][0])
    mask = (structure.atom_resindex == ridx) & (np.array(structure.atom_name) == "CA")
    return structure.coord[mask][0]


def _segment_tm_helices(structure, frame: "MembraneFrame", min_length: int = 15,
                         min_frac_in_slab: float = 0.45) -> list:
    """Contiguous stretches of ``frame.tm_mask_resnums`` classified as real
    membrane-spanning TM helices -- filters out the short helical
    loop-caps the same per-residue mask sometimes includes (e.g. a short
    intracellular-loop helical turn right after a TM ends), which a naive
    "just take contiguous runs" segmentation cannot distinguish from a
    genuine TM span. A candidate segment is kept only if it is at least
    ``min_length`` residues long AND at least ``min_frac_in_slab`` of its
    Cα projections fall within the membrane slab (``frame.in_slab``).

    Thresholds were tuned against the real GPR68 active/inactive
    structures this pipeline targets (not assumed): this combination
    gives exactly the canonical 7 TM segments on both, with R3.50
    correctly falling in the 3rd -- see ``find_d250``, which is the
    reason this function exists.

    Returns segments in N->C sequence order, each a list of author
    resnums.
    """
    ca_mask = np.array(structure.atom_name) == "CA"
    ca_resindex = structure.atom_resindex[ca_mask]
    ca_coord = structure.coord[ca_mask]
    proj_by_resnum = {
        int(structure.author_resnum[ridx]): float(frame.project(c[None, :])[0])
        for ridx, c in zip(ca_resindex, ca_coord)
    }

    resnums = sorted(frame.tm_mask_resnums)
    if not resnums:
        return []
    raw_segments = []
    current = [resnums[0]]
    for r in resnums[1:]:
        if r == current[-1] + 1:
            current.append(r)
        else:
            raw_segments.append(current)
            current = [r]
    raw_segments.append(current)

    real_tm = []
    for seg in raw_segments:
        projs = [proj_by_resnum[r] for r in seg if r in proj_by_resnum]
        if not projs:
            continue
        frac_in_slab = float(np.mean([abs(p) <= frame.half_thickness_ang for p in projs]))
        if len(seg) >= min_length and frac_in_slab >= min_frac_in_slab:
            real_tm.append(seg)
    return real_tm


def find_d250(structure, frame: "MembraneFrame", r350_resnum: int = None) -> tuple:
    """Locate D2.50 (or, rarely, E2.50) -- the conserved sodium-pocket
    Asp/Glu on TM2 -- geometrically rather than by sequence motif.

    Unlike R3.50 (DRY) and Y7.53 (NPxxY), D2.50 has no short, family-
    general local sequence signature reliable enough for a simple regex
    (even GPCRdb's own generic-numbering algorithm uses profile/
    structural alignment for this position, not a motif search), and this
    pipeline's guardrails explicitly forbid reading the canonical BW-
    numbered alignment file (see the module docstring). Instead:

      1. Segment the membrane frame's TM mask into discrete real TM
         helices (``_segment_tm_helices``), filtering out short helical
         loop-caps the raw per-residue mask can include.
      2. Take the helix immediately N-terminal (by resnum) of the one
         containing R3.50 (found via ``find_r350`` if not supplied) as
         TM2 -- anchoring to R3.50 rather than counting "the 2nd helix
         from the N-terminus" is robust to a poorly-resolved or
         miscounted TM1.
      3. Within TM2, find every Asp/Glu and choose the one whose
         membrane-normal projection is closest to 0 -- the geometric
         center of the membrane slab, where the sodium pocket actually
         sits (it coordinates an ion within the 7TM bundle's core, not
         near either membrane face). Verified directly on the real GPR68
         structures this pipeline targets, not assumed: the correct
         candidate (Asp67) projects to 0.33 A (center), while the
         runner-up (Glu55) projects to -15.3 A -- right at the slab
         boundary, i.e. loop-proximal, not membrane-embedded. A decisive
         separation on real data, not a coin flip.

    Raises ValueError if R3.50's own helix can't be located among the
    segmented TM helices, if there is no helix N-terminal to it, or if
    that helix has no Asp/Glu at all -- ambiguous or missing cases are
    surfaced, never guessed. Warns (does not silently pick) if the two
    closest candidates are within 2 A of each other in projection -- too
    close to call decisively by this criterion alone.

    Returns ``(resnum, candidates)`` where ``candidates`` is every
    (resnum, resname, projection_ang) considered in TM2, sorted by
    |projection| ascending, so a caller can see exactly what else was in
    contention -- the chosen residue is ``candidates[0]``.
    """
    if r350_resnum is None:
        r350_resnum, _ = find_r350(structure)

    tm_helices = _segment_tm_helices(structure, frame)
    tm3_idx = next((i for i, seg in enumerate(tm_helices) if r350_resnum in seg), None)
    if tm3_idx is None:
        raise ValueError(f"R3.50 (resnum {r350_resnum}) was not found in any segmented TM helix -- "
                          "cannot anchor TM2 relative to TM3")
    if tm3_idx == 0:
        raise ValueError("R3.50's helix is the first segmented TM helix -- no helix N-terminal to it "
                          "to identify as TM2 (TM1 may be missing/unmodeled or mis-segmented)")
    tm2 = tm_helices[tm3_idx - 1]

    candidates = []
    for resnum in tm2:
        ridx = int(np.where(structure.author_resnum == resnum)[0][0])
        resname = structure.resname[ridx]
        if resname in ("ASP", "GLU"):
            proj = float(frame.project(_ca_coord(structure, resnum)[None, :])[0])
            candidates.append((resnum, resname, proj))

    if not candidates:
        raise ValueError(f"no Asp/Glu found in the TM2 helix (resnums {tm2[0]}-{tm2[-1]}) -- "
                          "D2.50 could not be located")

    candidates.sort(key=lambda c: abs(c[2]))
    if len(candidates) > 1 and abs(abs(candidates[0][2]) - abs(candidates[1][2])) < 2.0:
        import warnings
        warnings.warn(
            f"D2.50 candidates {candidates[0][:2]} and {candidates[1][:2]} are within 2 A of each "
            f"other in membrane-depth projection ({candidates[0][2]:.2f} vs {candidates[1][2]:.2f} A) "
            f"-- too close to call decisively; using {candidates[0][:2]} (closer to center)"
        )

    return candidates[0][0], candidates


def _looks_like_plddt(bfactors: np.ndarray) -> bool:
    """A real pLDDT column is a percentage: every value in [0, 100]. Any
    value outside that range (as found in this repo's GPCRdb B-factor
    columns, e.g. -5.46) disqualifies it -- not a full statistical test,
    just the minimum physical sanity check for "is this actually pLDDT."
    """
    return bool(np.all((bfactors >= 0.0) & (bfactors <= 100.0)))


@dataclass
class MembraneFrame:
    origin: np.ndarray             # (3,) Angstrom -- centroid of the TM-mask Cα atoms used for PCA
    axis: np.ndarray               # (3,) unit vector, membrane normal; +axis points toward the extracellular side
    half_thickness_ang: float
    tm_mask_method: str            # "plddt" or "secondary_structure_helix"
    tm_mask_resnums: list          # author resnums included in the PCA fit
    plddt_threshold: float         # only meaningful if tm_mask_method == "plddt"
    r350_resnum: int
    dry_motif: str
    y753_resnum: int
    npxxy_motif: str
    half_thickness_fitted: bool    # True if fit from the hydrophobic band, False if the default was used
    explained_variance_ratio: float  # fraction of TM-mask Cα positional variance along `axis` (fit quality)

    def project(self, coords: np.ndarray) -> np.ndarray:
        """Signed distance(s) along the membrane normal from `origin`, for
        one (3,) point or an (N,3) array of points -- positive is toward
        the extracellular side, by this frame's orientation convention."""
        coords = np.asarray(coords)
        return (coords - self.origin) @ self.axis

    def in_slab(self, coords: np.ndarray) -> np.ndarray:
        """Boolean mask: True where `coords` fall inside the membrane
        (low-dielectric) slab, i.e. |projection| <= half_thickness."""
        return np.abs(self.project(coords)) <= self.half_thickness_ang


def compute_membrane_frame(
    structure,
    ss_mask: np.ndarray = None,
    plddt_threshold: float = 70.0,
    half_thickness_ang: float = 15.0,
    fit_half_thickness: bool = False,
    fit_percentile: float = 10.0,
) -> MembraneFrame:
    """Compute the membrane normal axis and slab geometry for `structure`.

    TM-bundle Cα selection: uses `structure`'s B-factor column as pLDDT if
    it passes a physical sanity check (every value in [0, 100]) and
    `plddt_threshold` then selects Cα atoms with B-factor >= threshold;
    otherwise falls back to a secondary-structure helix mask (computed via
    `wsme_gpcr.secondary_structure.assign_secondary_structure` if
    `ss_mask` is not supplied).

    Axis orientation: PCA (via SVD) on the selected Cα coordinates gives an
    axis with ambiguous sign; it's oriented using R3.50 and Y7.53 (both
    conserved intracellular-side anchors -- see module docstring for how
    they're located without BW numbering): the axis sign is flipped, if
    needed, so the R3.50/Y7.53 midpoint has a *more negative* projection
    than the TM-bundle centroid, establishing the convention "+axis points
    extracellular, -axis points intracellular."

    ``half_thickness_ang``: the membrane slab's half-thickness (default 15
    Angstrom per the pipeline spec). If ``fit_half_thickness`` is True,
    this default is instead estimated from the TM-mask Cα distribution
    itself: the ``fit_percentile``-to-``100-fit_percentile`` span of Cα
    projections onto the fitted axis (a hydrophobic-band proxy -- Cα's,
    not side chains, so this is a structural not a chemical estimate;
    treat it as a starting point to sanity-check against the receptor's
    known TM boundaries, not a substitute for one).
    """
    r350_resnum, dry_motif = find_r350(structure)
    y753_resnum, npxxy_motif = find_y753(structure)

    ca_atom_mask = np.array(structure.atom_name) == "CA"
    ca_resindex = structure.atom_resindex[ca_atom_mask]
    ca_coord = structure.coord[ca_atom_mask]
    ca_bfactor = structure.bfactor[ca_atom_mask] if hasattr(structure, "bfactor") else None

    if ca_bfactor is not None and _looks_like_plddt(ca_bfactor):
        tm_mask_method = "plddt"
        selected = ca_bfactor >= plddt_threshold
    else:
        tm_mask_method = "secondary_structure_helix"
        if ss_mask is None:
            from wsme_gpcr.secondary_structure import assign_secondary_structure
            ss_mask = assign_secondary_structure(structure)
        selected = ss_mask[ca_resindex]

    if selected.sum() < 10:
        raise ValueError(
            f"only {int(selected.sum())} Cα atoms selected for the membrane-frame PCA fit "
            f"(method={tm_mask_method}) -- too few to define a reliable membrane normal"
        )

    tm_coord = ca_coord[selected]
    tm_resnums = sorted(int(r) for r in structure.author_resnum[ca_resindex[selected]])

    origin = tm_coord.mean(axis=0)
    centered = tm_coord - origin
    # PCA via SVD: the membrane normal is the principal axis of the
    # elongated TM bundle (helices run roughly parallel to it).
    _, s, vt = np.linalg.svd(centered, full_matrices=False)
    axis = vt[0]
    explained_variance_ratio = float((s[0] ** 2) / np.sum(s ** 2))

    r350_ca = _ca_coord(structure, r350_resnum)
    y753_ca = _ca_coord(structure, y753_resnum)
    intracellular_ref = 0.5 * (r350_ca + y753_ca)
    if np.dot(intracellular_ref - origin, axis) > 0:
        axis = -axis

    frame = MembraneFrame(
        origin=origin, axis=axis, half_thickness_ang=half_thickness_ang,
        tm_mask_method=tm_mask_method, tm_mask_resnums=tm_resnums,
        plddt_threshold=plddt_threshold if tm_mask_method == "plddt" else float("nan"),
        r350_resnum=r350_resnum, dry_motif=dry_motif,
        y753_resnum=y753_resnum, npxxy_motif=npxxy_motif,
        half_thickness_fitted=fit_half_thickness,
        explained_variance_ratio=explained_variance_ratio,
    )

    if fit_half_thickness:
        z = frame.project(tm_coord)
        lo, hi = np.percentile(z, [fit_percentile, 100.0 - fit_percentile])
        frame.half_thickness_ang = float((hi - lo) / 2.0)

    return frame
