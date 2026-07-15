"""Structure-aware pKa prediction via PROPKA3, for feeding as
``pka_overrides`` into ``load_structure``/``run_pipeline``.

wsme_gpcr's default charge model (``structure.DEFAULT_PKA``) assigns every
residue of a given type the same free-amino-acid solution pKa (Asp 3.9,
Glu 4.1, His 6.0, ...), independent of structural context. Real buried
ionizable residues -- especially ones lining a desolvated pocket or a
bound-cation site -- can have pKa values shifted by several units from
that free-solution baseline. This matters directly for this codebase:
Rowe & Isom (2021, JBC), using their pHinder structural-informatics
platform, identify a buried acidic triad (D2.50/E4.53/D7.49, their
"DyaD"+"apEx" sites) as the primary proton sensor in GPR4/GPR65/GPR68 --
a claim that requires those residues to actually titrate near
physiological pH. The flat default-pKa model cannot reproduce that: at
pH 5-8, Asp/Glu at their solution pKa (~3.9-4.1) stay >88% charged
throughout, essentially never titrating (see FINDINGS.md's Node148
mechanistic audit). PROPKA3 (Jensen lab; Olsson et al. 2011, J. Chem.
Theory Comput.) is a real, widely-used empirical structure-based pKa
predictor that accounts for desolvation, burial, and nearby
charged/hydrogen-bonding groups -- this module runs it and converts its
output into the ``pka_overrides`` format the rest of the pipeline already
consumes, the same integration pattern used for real DSSP secondary
structure elsewhere in this codebase.
"""

from __future__ import annotations

import tempfile
import warnings
from pathlib import Path

from .structure import CHARGED_RESIDUES


class PropkaNotAvailableError(RuntimeError):
    """Raised when the ``propka`` package isn't importable. Never falls
    back silently to the flat default-pKa model -- that would swap in a
    different charge assignment without the caller noticing."""


def predict_pka_propka(pdb_or_cif_path, chain: str | None = None) -> dict:
    """Run PROPKA3 on a structure and return ``{author_resnum: predicted_pKa}``
    for every Asp/Glu/His/Lys/Arg sidechain group it scores -- directly
    usable as ``pka_overrides`` for ``load_structure``/``run_pipeline``.

    PROPKA3 only reads PDB files; mmCIF input is transparently converted
    to a temporary PDB first via Biopython. If ``chain`` is given, only
    that chain's groups are returned (matching ``load_structure``'s own
    chain selection); otherwise all chains' groups are returned keyed by
    resnum, which is fine for the single-chain structures used throughout
    this codebase but would collide across chains for a true multi-chain
    complex.

    Raises ``PropkaNotAvailableError`` if ``propka`` isn't installed
    (``pip install propka``), rather than silently falling back to the
    default fixed-pKa model.
    """
    try:
        import propka.run
    except ImportError as e:
        raise PropkaNotAvailableError(
            "propka is not installed (`pip install propka`). Use explicit "
            "pka_overrides, or the default fixed-pKa model, instead."
        ) from e

    path = Path(pdb_or_cif_path)
    tmp_pdb_path = None
    run_path = str(path)
    if path.suffix.lower() in (".cif", ".mmcif"):
        from Bio.PDB import MMCIFParser, PDBIO

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            bio_structure = MMCIFParser(QUIET=True).get_structure(path.stem, str(path))
        with tempfile.NamedTemporaryFile(suffix=".pdb", delete=False) as tmp:
            tmp_pdb_path = tmp.name
        io = PDBIO()
        io.set_structure(bio_structure)
        io.save(tmp_pdb_path)
        run_path = tmp_pdb_path

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mol = propka.run.single(run_path, write_pka=False)
    finally:
        if tmp_pdb_path is not None:
            Path(tmp_pdb_path).unlink(missing_ok=True)

    conformation_name = mol.conformation_names[0] if mol.conformation_names else "AVR"
    conf = mol.conformations[conformation_name]

    overrides = {}
    for group in conf.groups:
        if group.residue_type not in CHARGED_RESIDUES:
            continue
        if chain is not None and group.atom.chain_id != chain:
            continue
        overrides[int(group.atom.res_num)] = float(group.pka_value)
    return overrides
