"""
Utilities for working with real cryo-EM structures (as opposed to the
AlphaFold-based homology models used for GPR68/GPR132): extracting the
receptor-only chain out of a multi-component complex (G protein subunits,
nanobodies, Fabs), detecting and excising an engineered fusion partner
(e.g. BRIL/apocytochrome b562, commonly inserted into ICL3 to stabilize
inactive-state small GPCRs for cryo-EM), and renumbering onto a reference
structure's native numbering via real sequence alignment rather than
assuming numbering is directly comparable across independently-deposited
structures (it often isn't -- see the GPR4 9JFU case this module was
built for, where the inactive structure's post-fusion residues are number
BRIL-length residues higher than the same residues in the unfused active
structures).
"""

from __future__ import annotations

from Bio import Align
from Bio.PDB import PDBIO
from Bio.PDB.StructureBuilder import StructureBuilder
from Bio.SeqUtils import seq1

# Apocytochrome b562RIL ("BRIL"), the standard thermostabilized fusion
# partner used to stabilize small/flexible GPCR constructs for structural
# biology. Matched by substring, not full-length, since expression
# constructs sometimes truncate a few residues at either end.
BRIL_MARKER = "ADLEDNWETLNDNLKVIEKADNAAQVKDALTKMRAAALDAQKAT"


def load_chain_residues(cif_path, chain_id="R"):
    from Bio.PDB import MMCIFParser

    parser = MMCIFParser(QUIET=True)
    structure = parser.get_structure(str(cif_path), str(cif_path))
    chain = structure[0][chain_id]
    residues = [r for r in chain if r.id[0] == " "]
    return residues, structure


def build_structure_from_residues(structure_id, chain_id, residues, resnum_map=None):
    """Build a fresh single-chain Structure from a list of Biopython
    residues, optionally renumbering via `resnum_map` ({old_resnum:
    new_resnum}); residues not in the map (if given) are dropped."""
    builder = StructureBuilder()
    builder.init_structure(structure_id)
    builder.init_model(0)
    builder.init_chain(chain_id)
    builder.init_seg(" ")
    for res in residues:
        old_resnum = res.id[1]
        if resnum_map is not None:
            if old_resnum not in resnum_map:
                continue
            new_resnum = resnum_map[old_resnum]
        else:
            new_resnum = old_resnum
        builder.init_residue(res.get_resname(), " ", new_resnum, " ")
        for atom in res:
            if atom.is_disordered():
                atom = atom.selected_child
            builder.init_atom(
                atom.get_name(), atom.coord, atom.get_bfactor(), atom.get_occupancy(),
                atom.get_altloc(), atom.get_fullname(), element=atom.element,
            )
    return builder.get_structure()


def excise_fusion_by_marker(residues, marker=BRIL_MARKER, context=15):
    """Find `marker` in the residue sequence and drop every residue from
    the start of that match through the point where the sequence resumes
    matching a plausible receptor context (heuristically: `context`
    residues after the marker's own end, since the exact fusion/receptor
    splice point can be off by a few residues at each junction and this
    module's caller re-aligns properly against a reference afterward
    anyway -- this step only needs to remove the bulk of the ~90-100
    residue fusion domain, not find the exact splice residue)."""
    seq = "".join(seq1(r.get_resname()) for r in residues)
    start = seq.find(marker)
    if start == -1:
        return residues, None  # no fusion detected
    # BRIL is ~106 residues; scan forward from marker end for a generous
    # window and cut at marker_start .. marker_start+130 (covers the full
    # BRIL domain plus its own short flanking linker residues on both
    # sides); the caller's alignment step cleans up any residual mismatch.
    end = min(start + 130, len(residues))
    kept = residues[:start] + residues[end:]
    return kept, (residues[start].id[1], residues[end - 1].id[1] if end > 0 else None)


def align_and_map_resnums(query_residues, reference_seq_by_resnum):
    """Global-aligns `query_residues`' sequence against a reference
    {resnum: one-letter-code} dict (typically built from an unfused,
    natively-numbered structure) and returns {query_resnum: reference_resnum}
    for every position where the two sequences agree EXACTLY (mismatches
    and gap-adjacent positions are dropped rather than guessed, since
    those are exactly the positions -- fusion splice junctions, disordered
    loop edges -- where a wrong guess would silently corrupt the
    downstream atom-correspondence-dependent pipeline)."""
    ref_resnums = sorted(reference_seq_by_resnum)
    ref_seq = "".join(reference_seq_by_resnum[rn] for rn in ref_resnums)
    query_seq = "".join(seq1(r.get_resname()) for r in query_residues)
    query_resnums = [r.id[1] for r in query_residues]

    aligner = Align.PairwiseAligner()
    aligner.mode = "global"
    aligner.open_gap_score = -10
    aligner.extend_gap_score = -0.5
    aligner.mismatch_score = -1
    aligner.match_score = 2
    alignment = aligner.align(ref_seq, query_seq)[0]
    ref_aligned, query_aligned = alignment.aligned

    mapping = {}
    for (r_start, r_end), (q_start, q_end) in zip(ref_aligned, query_aligned):
        # aligned blocks are gap-free and equal length by construction
        for offset in range(r_end - r_start):
            ref_idx = r_start + offset
            query_idx = q_start + offset
            if ref_seq[ref_idx] == query_seq[query_idx]:
                mapping[query_resnums[query_idx]] = ref_resnums[ref_idx]
    return mapping


def write_structure(structure, path):
    writer = PDBIO()
    writer.set_structure(structure)
    writer.save(str(path))
