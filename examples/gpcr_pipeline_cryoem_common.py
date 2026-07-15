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


def excise_fusion_by_marker(residues, marker=BRIL_MARKER):
    """Find `marker` in the residue sequence and drop exactly the matched
    marker residues (nothing more).

    An earlier version of this function also cut a large fixed-size
    "context" window (marker_start+130 residues by list index) past the
    marker on the theory that this would conservatively cover "the whole
    ~90-106 residue BRIL domain plus flanking linkers." That was wrong in
    a way that silently corrupted downstream results: verified against
    GPR4's 9JFU, the true splice back into native receptor sequence was
    only ~99 residues past marker start (matching real, receptor-specific
    context immediately after "...NAYIQKYL", not a hardcoded distance),
    while the fixed 130-residue window cut 31 residues too far -- deleting
    the first third of real TM6 and leaving it silently absent from the
    "cleaned" structure with no error raised.

    The fix: excise ONLY the exact marker match (whose length and
    position are known with certainty), and leave any remaining
    non-marker fusion-domain residues on either side in place. The
    caller's `align_and_map_resnums` step (affine-gap global alignment
    against a reference) is relied on to correctly gap around whatever
    fusion-domain residues remain -- a large gap is cheap under its
    scoring scheme (open=-10, extend=-0.5) relative to the essentially
    certain run of mismatches an ~50-100 residue non-receptor insertion
    would produce if forced to align 1:1, so it reliably resolves to one
    contiguous gap rather than accidentally consuming real receptor
    residues. This makes no assumption about the fusion domain's total
    length or the marker's position within it, unlike the fixed window."""
    seq = "".join(seq1(r.get_resname()) for r in residues)
    start = seq.find(marker)
    if start == -1:
        return residues, None  # no fusion detected
    end = start + len(marker)
    kept = residues[:start] + residues[end:]
    return kept, (residues[start].id[1], residues[end - 1].id[1])


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
