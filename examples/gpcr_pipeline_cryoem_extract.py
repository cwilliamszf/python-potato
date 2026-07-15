"""
Task #15: extract clean, single-chain, correctly-numbered receptor
structures from the 11 real cryo-EM CIF files the user uploaded (8 GPR4,
1 GPR65, 2 GPR68), using `gpcr_pipeline_cryoem_common`.

Reference numbering for GPR4 is taken from 8ZCF (arbitrary choice among
the 7 active-state structures -- verified in the prior session, via direct
pairwise comparison, that all 7 share identical native numbering, so any
of them would give the same result). 9JFU (the sole inactive structure)
carries a BRIL fusion in ICL3 with continuous renumbering downstream of
it, so it alone needs marker-based excision + sequence-alignment-based
renumbering onto the 8ZCF reference before it can be used in the same
frame/numbering as the active structures. GPR65 (9BHL) and GPR68 cryo-EM
(9BI6, 9BHM) have no fusion and no cross-structure numbering issue
(single structure each, active-state only) -- straight chain-R extraction.

Every renumbering is verified with an explicit residue-identity check
(mapped positions must be the same amino acid in both structures) before
being trusted -- silent corruption at a wrong alignment position is
exactly the failure mode this module was built to avoid.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from gpcr_pipeline_cryoem_common import (
    align_and_map_resnums,
    build_structure_from_residues,
    excise_fusion_by_marker,
    load_chain_residues,
    write_structure,
)
from Bio.SeqUtils import seq1

DATA_ROOT = Path(__file__).parent / "data"

GPR4_ACTIVE = {
    "9LGM_pH8.0": DATA_ROOT / "gpr4_structures" / "active" / "9LGM_pH8.0.cif",
    "9JFX_pH7.5": DATA_ROOT / "gpr4_structures" / "active" / "9JFX_pH7.5.cif",
    "9JFZ_intermediate_pH7.5": DATA_ROOT / "gpr4_structures" / "active" / "9JFZ_intermediate_pH7.5.cif",
    "9JFV_pH6.8": DATA_ROOT / "gpr4_structures" / "active" / "9JFV_pH6.8.cif",
    "8ZCF_pH7.5": DATA_ROOT / "gpr4_structures" / "active" / "8ZCF_pH7.5.cif",
    "8ZCE_pH6.0": DATA_ROOT / "gpr4_structures" / "active" / "8ZCE_pH6.0.cif",
    "9BIP_pHunspecified": DATA_ROOT / "gpr4_structures" / "active" / "9BIP_pHunspecified.cif",
}
GPR4_INACTIVE = DATA_ROOT / "gpr4_structures" / "inactive" / "9JFU.cif"
GPR4_REFERENCE_KEY = "8ZCF_pH7.5"

GPR65_ACTIVE = {"9BHL": DATA_ROOT / "gpr65_structures" / "active" / "9BHL.cif"}
GPR68_CRYOEM_ACTIVE = {
    "9BI6": DATA_ROOT / "gpr68_cryoem_structures" / "active" / "9BI6.cif",
    "9BHM": DATA_ROOT / "gpr68_cryoem_structures" / "active" / "9BHM.cif",
}

CLEAN_ROOT = DATA_ROOT
CHAIN_ID = "R"


def extract_plain(cif_path, out_pdb_path):
    """Straight chain-R extraction, no renumbering -- for structures with
    no fusion partner and no cross-structure numbering mismatch to fix."""
    residues, _ = load_chain_residues(cif_path, chain_id=CHAIN_ID)
    structure = build_structure_from_residues(out_pdb_path.stem, CHAIN_ID, residues)
    out_pdb_path.parent.mkdir(parents=True, exist_ok=True)
    write_structure(structure, out_pdb_path)
    return residues


def verify_identity(residues_a, residues_b, label_a, label_b):
    """Assert every shared resnum is the same amino acid in both
    structures -- the same style of sanity check used throughout the
    GPR68/GPR132 phases before trusting a structure for downstream use."""
    by_resnum_a = {r.id[1]: seq1(r.get_resname()) for r in residues_a}
    by_resnum_b = {r.id[1]: seq1(r.get_resname()) for r in residues_b}
    shared = sorted(set(by_resnum_a) & set(by_resnum_b))
    mismatches = [rn for rn in shared if by_resnum_a[rn] != by_resnum_b[rn]]
    print(f"  {label_a} vs {label_b}: {len(shared)} shared resnums, {len(mismatches)} mismatches")
    if mismatches:
        print(f"    mismatched resnums (first 10): {mismatches[:10]}")
    assert not mismatches, f"{label_a} vs {label_b}: identity check failed at {mismatches}"
    return shared


def main():
    print("=== GPR4 active structures (plain chain-R extraction) ===")
    active_residues = {}
    for name, cif_path in GPR4_ACTIVE.items():
        out_path = CLEAN_ROOT / "gpr4_structures" / "clean_active" / f"{name}.pdb"
        residues = extract_plain(cif_path, out_path)
        active_residues[name] = residues
        print(f"{name}: {len(residues)} residues -> {out_path}")

    reference_residues = active_residues[GPR4_REFERENCE_KEY]
    reference_seq_by_resnum = {r.id[1]: seq1(r.get_resname()) for r in reference_residues}

    print(f"\n=== Cross-checking all GPR4 active structures share native numbering (ref={GPR4_REFERENCE_KEY}) ===")
    for name, residues in active_residues.items():
        if name == GPR4_REFERENCE_KEY:
            continue
        verify_identity(residues, reference_residues, name, GPR4_REFERENCE_KEY)

    print("\n=== GPR4 inactive structure 9JFU: BRIL excision + alignment-based renumbering ===")
    raw_residues, _ = load_chain_residues(GPR4_INACTIVE, chain_id=CHAIN_ID)
    print(f"9JFU raw chain R: {len(raw_residues)} residues, "
          f"resnum range {raw_residues[0].id[1]}-{raw_residues[-1].id[1]}")

    excised_residues, cut_range = excise_fusion_by_marker(raw_residues)
    assert cut_range is not None, "BRIL marker not found in 9JFU -- expected fusion not detected"
    print(f"BRIL marker found; excised resnum range {cut_range[0]}-{cut_range[1]} "
          f"({len(raw_residues) - len(excised_residues)} residues removed)")
    print(f"9JFU post-excision: {len(excised_residues)} residues")

    mapping = align_and_map_resnums(excised_residues, reference_seq_by_resnum)
    print(f"Alignment-based mapping: {len(mapping)} residues confidently mapped "
          f"(exact-match positions only) out of {len(excised_residues)} post-excision residues")

    renumbered_structure = build_structure_from_residues("9JFU_clean", CHAIN_ID, excised_residues, resnum_map=mapping)
    renumbered_residues = [r for r in renumbered_structure[0][CHAIN_ID] if r.id[0] == " "]
    out_path = CLEAN_ROOT / "gpr4_structures" / "clean_inactive" / "9JFU_clean.pdb"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_structure(renumbered_structure, out_path)
    print(f"9JFU renumbered/clean: {len(renumbered_residues)} residues -> {out_path}")

    print("\n=== Verifying 9JFU renumbering against reference ===")
    verify_identity(renumbered_residues, reference_residues, "9JFU_clean", GPR4_REFERENCE_KEY)
    for name, residues in active_residues.items():
        if name == GPR4_REFERENCE_KEY:
            continue
        verify_identity(renumbered_residues, residues, "9JFU_clean", name)

    print("\n=== GPR65 (9BHL) and GPR68 cryo-EM (9BI6, 9BHM): plain chain-R extraction ===")
    for name, cif_path in GPR65_ACTIVE.items():
        out_path = CLEAN_ROOT / "gpr65_structures" / "clean_active" / f"{name}.pdb"
        residues = extract_plain(cif_path, out_path)
        print(f"{name}: {len(residues)} residues -> {out_path}")

    for name, cif_path in GPR68_CRYOEM_ACTIVE.items():
        out_path = CLEAN_ROOT / "gpr68_cryoem_structures" / "clean_active" / f"{name}.pdb"
        residues = extract_plain(cif_path, out_path)
        print(f"{name}: {len(residues)} residues -> {out_path}")

    print("\nDone. All 11 structures extracted/cleaned; GPR4 set verified self-consistent.")


if __name__ == "__main__":
    main()
