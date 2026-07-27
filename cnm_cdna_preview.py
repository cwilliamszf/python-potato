#!/usr/bin/env python3
"""
CNM guide-design PRELIMINARY PREVIEW from cDNA/CDS-only input.

*** NOT A SUBSTITUTE FOR THE GENOMIC-BASED cnm_guide_designer.py ***

This exists because the only sequence available for bin1b, myf6, and ryr1b
was spliced cDNA (and for bin1a/myf6/ryr1b, actually bare CDS with no UTR
at all - see README in the source data bundle). Real genomic DNA (with
introns) was not available.

Consequences of that gap, spelled out because they matter before ordering
anything:
  - There is no intron information, so a candidate guide's 20nt spacer may
    in reality straddle a real exon-exon junction. Such a guide does not
    exist as a contiguous sequence in genomic DNA and WILL NOT CUT in vivo.
    This script cannot detect or exclude that failure mode.
  - "True exon numbers" and canonical splice-site verification are not
    possible from this input and are not attempted here.
  - "Uniqueness" is only checked against the single transcript/CDS record
    itself (a 1-15kb fragment), not the genome - it is essentially
    uninformative as an off-target signal and is labeled as such below.
  - Genotyping primers need real genomic flanking sequence and are not
    designed here.

Every candidate below MUST be re-verified against real genomic coordinates
(Ensembl GRCz11 gene page, ZFIN, and CRISPOR) before synthesis/ordering.

Reuses the scoring building blocks (GC%, poly-T, Bae OOF microhomology
score, indel/PTC consequence prediction, paralog cross-reactivity) from
cnm_guide_designer.py, which is exact and should be used once genomic
sequence is available.
"""

import sys, argparse
from cnm_guide_designer import (
    rc, tr, gc, read_fasta, enumerate_guides, mh_out_of_frame,
    consequence, spacer_hits_genome, seed_hits_genome,
)

def find_longest_orf(seq):
    """Longest ATG->stop ORF in any forward frame. Used to recover the CDS
    from a transcript record when no separate CDS file is provided."""
    n = len(seq)
    best = None
    for i in range(n - 2):
        if seq[i:i+3] != 'ATG':
            continue
        j = i
        while j < n - 2:
            codon = seq[j:j+3]
            if codon in ('TAA', 'TAG', 'TGA'):
                if best is None or (j+3-i) > (best[1]-best[0]):
                    best = (i, j+3)
                break
            j += 3
    return best

def score_guide(g, gc_lo=40, gc_hi=70, para_known=None):
    s = 0.0
    s += 30 if g['uniq_local'] else -50
    s += 20 if g['seed_uniq_local'] else -30
    s += -60 if 'TTTT' in g['spacer'] else 0
    s += 25 if gc_lo <= gc(g['spacer']) <= gc_hi else 0
    s += g['oof'] * 0.5
    if para_known:
        s += 40 if g.get('para_hits', 0) == 0 else -120
    # modest preference for an earlier predicted PTC (cleaner truncation)
    ptc_vals = [v for v in g['ptc'] if v is not None]
    if ptc_vals:
        s += max(0, 15 - min(ptc_vals) / 20.0)
    return s

def design(gene_name, cdna_path, window=(60, 660), paralog_path=None,
           paralog_name=None, top_n=2, min_sep=10):
    seq = read_fasta(cdna_path)
    orf = find_longest_orf(seq)
    if orf is None:
        print(f"# {gene_name}: no ORF found - skipping"); return []
    cds_s, cds_e = orf
    cds = seq[cds_s:cds_e]
    pep = tr(cds[:-3])
    print(f"\n{'='*78}\n# {gene_name}  transcript {len(seq)} nt, "
          f"CDS {cds_s}-{cds_e} ({len(cds)} nt, {len(pep)} aa)")
    if cds_s == 0 and cds_e == len(seq):
        print("#   NOTE: this record has zero flanking sequence (starts at "
              "ATG, ends at stop) - it is CDS-only, not full cDNA.")

    para_seq = read_fasta(paralog_path) if paralog_path else None
    if para_seq is not None:
        print(f"#   paralog cross-check vs {paralog_name} "
              f"({len(para_seq)} nt) enabled")

    win_s = cds_s + min(window[0], max(0, len(cds)-6))
    win_e = min(cds_e - 3, cds_s + window[1])   # stay clear of the stop codon
    if win_e - win_s < 20:
        win_s, win_e = cds_s, cds_e - 3
    guides = enumerate_guides(seq, win_s, win_e)
    if not guides:
        print(f"#   no NGG protospacers found in the target window "
              f"{win_s-cds_s}-{win_e-cds_s} (relative to CDS start)")
        return []

    for g in guides:
        L, R = max(0, g['cut']-40), min(len(seq), g['cut']+40)
        winseq = seq[L:R]
        _, oof = mh_out_of_frame(winseq, g['cut']-L)
        g['oof'] = oof
        g['uniq_local'] = (seq.count(g['spacer']) + seq.count(rc(g['spacer']))) == 1
        seed = g['spacer'][-12:]
        g['seed_uniq_local'] = seed_hits_genome(seed, seq) == 1
        if para_seq is not None:
            g['para_hits'] = spacer_hits_genome(g['spacer'], para_seq)
        rel = g['cut'] - cds_s
        ptcs = []
        for ind in (-1, -2, +1, +2):
            _, _, ptc = consequence(cds, 0, rel, ind)
            ptcs.append(ptc)
        g['ptc'] = ptcs
        g['score'] = score_guide(g, para_known=(para_seq is not None))

    ranked = sorted(guides, key=lambda g: -g['score'])
    print(f"\n#   {len(guides)} candidate protospacers in window "
          f"(CDS-relative {win_s-cds_s}-{win_e-cds_s}):")
    hdr = (f"   {'#':3} {'spacer':22} {'PAM':4} {'str':4} {'GC%':5} "
           f"{'polyT':6} {'OOF%':6} {'localU':7} {'seedU':6} "
           + (f"{'para-hit':9} " if para_seq is not None else "")
           + f"{'score':6}")
    print(hdr)
    for n, g in enumerate(ranked, 1):
        para_col = (f"{(str(g['para_hits'])+'x' if g['para_hits'] else 'none'):9} "
                    if para_seq is not None else "")
        print(f"   {n:<3} {g['spacer']:22} {g['pam']:4} {g['strand']:4} "
              f"{gc(g['spacer']):5.0f} "
              f"{'YES' if 'TTTT' in g['spacer'] else '-':6} {g['oof']:6.1f} "
              f"{'ok' if g['uniq_local'] else 'DUP':7} "
              f"{'ok' if g['seed_uniq_local'] else 'DUP':6} "
              + para_col + f"{g['score']:6.0f}")

    picked = []
    for g in ranked:
        if all(abs(g['cut']-p['cut']) >= min_sep for p in picked):
            picked.append(g)
        if len(picked) == top_n:
            break

    print(f"\n#   >>> TOP {len(picked)} PICKS for {gene_name} "
          f"(preliminary / UNVERIFIED vs genomic structure) <<<")
    for n, g in enumerate(picked, 1):
        print(f"     guide {n}: 5'-{g['spacer']}-3'  PAM {g['pam']}  "
              f"strand {g['strand']}  cut@CDS+{g['cut']-cds_s}  "
              f"GC {gc(g['spacer']):.0f}%  OOF {g['oof']:.1f}%  "
              f"indel consequences (-1/-2/+1/+2): {g['ptc']}")
    return picked

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--gene-name', required=True)
    ap.add_argument('--cdna', required=True)
    ap.add_argument('--paralog-cdna')
    ap.add_argument('--paralog-name', default='paralog')
    args = ap.parse_args()
    design(args.gene_name, args.cdna, paralog_path=args.paralog_cdna,
           paralog_name=args.paralog_name)
