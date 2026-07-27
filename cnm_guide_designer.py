#!/usr/bin/env python3
"""
CNM knockout guide designer  (Williams Lab / MSU)

Generalized from the verified mtm1 design. Given a genomic sequence and
(optionally) the CDS, it will:
  1. map coding exons by CDS->genomic alignment  (gives TRUE exon numbers)
  2. verify canonical AG/GT splice boundaries and exon phase
  3. enumerate every SpCas9 NGG protospacer cutting inside a chosen exon
  4. score: GC, poly-T, Bae microhomology out-of-frame score
  5. predict frameshift consequence + premature stop position
  6. flag dual-guide pairings that are in-frame (the g1+g2 trap)
  7. check local uniqueness of spacer + 12nt (PAM-proximal) seed
  8. flag cross-reactivity against a supplied paralog/ohnolog sequence
     (zebrafish teleost genome duplication - see --paralog-* below)
  9. design genotyping primers flanking the cut sites

NOT done here (must go to CRISPOR against GRCz11):
  genome-wide off-target scanning.

Zebrafish note: the teleost-specific genome duplication (TGD) left many
human disease-gene orthologs as two paralogs (often suffixed a/b, e.g.
bin1a/bin1b, ryr1a/ryr1b). A single knockout may be phenotypically masked
by its co-ortholog. This script analyzes ONE gene per run - run it again
on the paralog's own genomic+CDS FASTA to get an independent guide set,
and optionally pass --paralog-genomic/--paralog-cds so candidate guides
here are checked for accidental cross-reactivity against that paralog.

Usage:
    python3 cnm_guide_designer.py --genomic gene.fa [--cds gene_cds.fa]
                                  [--exon N] [--gene-name bin1a]
                                  [--paralog-genomic bin1b.fa]
                                  [--paralog-cds bin1b_cds.fa]
                                  [--paralog-name bin1b]
"""

import argparse, math, re, sys, itertools

COD = {
 'TTT':'F','TTC':'F','TTA':'L','TTG':'L','CTT':'L','CTC':'L','CTA':'L','CTG':'L',
 'ATT':'I','ATC':'I','ATA':'I','ATG':'M','GTT':'V','GTC':'V','GTA':'V','GTG':'V',
 'TCT':'S','TCC':'S','TCA':'S','TCG':'S','CCT':'P','CCC':'P','CCA':'P','CCG':'P',
 'ACT':'T','ACC':'T','ACA':'T','ACG':'T','GCT':'A','GCC':'A','GCA':'A','GCG':'A',
 'TAT':'Y','TAC':'Y','TAA':'*','TAG':'*','CAT':'H','CAC':'H','CAA':'Q','CAG':'Q',
 'AAT':'N','AAC':'N','AAA':'K','AAG':'K','GAT':'D','GAC':'D','GAA':'E','GAG':'E',
 'TGT':'C','TGC':'C','TGA':'*','TGG':'W','CGT':'R','CGC':'R','CGA':'R','CGG':'R',
 'AGT':'S','AGC':'S','AGA':'R','AGG':'R','GGT':'G','GGC':'G','GGA':'G','GGG':'G'}
COMP = {'A':'T','T':'A','G':'C','C':'G','N':'N'}

def rc(s):  return ''.join(COMP[b] for b in reversed(s))
def tr(s):  return ''.join(COD.get(s[i:i+3],'X') for i in range(0,len(s)-2,3))
def gc(s):  return 100.0*sum(b in 'GC' for b in s)/len(s)

def read_fasta(path):
    txt = open(path).read()
    seq = ''.join(l.strip() for l in txt.splitlines() if not l.startswith('>'))
    return re.sub(r'[^ACGTNacgtn]','',seq).upper()

# ---------------------------------------------------------------- exon mapping
def map_exons_from_cds(genomic, cds, min_block=20):
    """Greedy CDS->genomic exon mapping. Returns ([(start,end,phase_in)], warnings).

    Two failure modes of a naive "extend while bases match" approach are
    corrected here:
      - the match can run 1-few bases PAST the true exon end by coincidence
        (the next intron base happens to equal the next CDS base); this is
        trimmed back to the nearest canonical GT donor.
      - a short/hard-to-anchor exon (or a genomic/CDS mismatch) used to abort
        mapping for the ENTIRE rest of the gene; now it is logged as a
        warning and mapping continues so downstream exons are still found.
    """
    exons, warnings, gpos, cpos = [], [], 0, 0
    while cpos < len(cds) - 3:
        hit = -1
        for probe_len in (min_block, 12, 8):
            if cpos + probe_len > len(cds):
                continue
            hit = genomic.find(cds[cpos:cpos+probe_len], gpos)
            if hit != -1:
                break
        if hit == -1:
            warnings.append(cpos)
            cpos += 1
            continue

        n = 0
        while (cpos+n < len(cds) and hit+n < len(genomic)
               and cds[cpos+n] == genomic[hit+n]):
            n += 1

        # trim a coincidental overrun back to the nearest GT donor (unless
        # this block runs to the end of the CDS, i.e. the terminal exon)
        if cpos + n < len(cds):
            lo = max(1, n - 15)
            n_trim = n
            while n_trim > lo and genomic[hit+n_trim:hit+n_trim+2] != 'GT':
                n_trim -= 1
            if genomic[hit+n_trim:hit+n_trim+2] == 'GT':
                n = n_trim

        if n == 0:
            warnings.append(cpos)
            cpos += 1
            continue

        exons.append((hit, hit+n, cpos % 3))
        cpos += n
        gpos = hit + n
    return exons, warnings

# ------------------------------------------------------- microhomology scoring
def mh_out_of_frame(seq, left):
    """Bae et al. 2014 microhomology / out-of-frame score."""
    lw, dup, s3, sn3 = 20.0, [], 0.0, 0.0
    for k in reversed(range(2, left+1)):
        for j in range(left, len(seq)-k+1):
            for i in range(0, left-k+1):
                if seq[i:i+k] != seq[j:j+k]:
                    continue
                if any(i>=d[0] and i<=d[1] and j>=d[2] and j<=d[3] for d in dup):
                    continue
                length = j - i
                mh = seq[i:i+k]
                g  = mh.count('G') + mh.count('C')
                sc = 100*(1.0/math.exp(length/lw))*((len(mh)-g) + g*2)
                if length % 3 == 0: s3  += sc
                else:               sn3 += sc
                dup.append((i, i+k-1, j, j+k-1))
    tot = s3 + sn3
    return tot, (sn3*100.0/tot if tot else 0.0)

# ------------------------------------------------------------ guide enumeration
def enumerate_guides(genomic, ex_s, ex_e, flank=30):
    guides = []
    for s in range(max(0, ex_s-flank), min(len(genomic)-23, ex_e+flank)):
        pam = genomic[s+20:s+23]
        if len(pam) == 3 and pam[1:] == 'GG':
            cut = s + 17
            if ex_s+1 <= cut <= ex_e-1:
                guides.append(dict(strand='+', proto_start=s,
                                   spacer=genomic[s:s+20], pam=pam, cut=cut))
        pam2 = genomic[s-3:s] if s >= 3 else ''
        if len(pam2) == 3 and pam2[:2] == 'CC':
            cut = s + 3
            if ex_s+1 <= cut <= ex_e-1:
                guides.append(dict(strand='-', proto_start=s,
                                   spacer=rc(genomic[s:s+20]), pam=rc(pam2),
                                   cut=cut))
    # dedupe
    seen, out = set(), []
    for g in guides:
        key = (g['spacer'], g['cut'])
        if key not in seen:
            seen.add(key); out.append(g)
    return sorted(out, key=lambda g: g['cut'])

def consequence(exon, phase_in, cut_rel, indel):
    """Return (frameshift?, peptide, PTC codon within exon or None)."""
    if indel < 0:
        mut = exon[:cut_rel+indel] + exon[cut_rel:]
    else:
        mut = exon[:cut_rel] + 'A'*indel + exon[cut_rel:]
    off = (3 - phase_in) % 3
    pep = tr(mut[off:])
    fs  = (len(mut) - len(exon)) % 3 != 0
    ptc = pep.index('*')+1 if '*' in pep else None
    return fs, pep, ptc

# --------------------------------------------------------- paralog cross-check
def spacer_hits_genome(spacer, genome):
    """Count loci in `genome` where this exact 20nt spacer would cut
    (valid NGG PAM adjacency, either strand). Exact-match heuristic only -
    NOT a substitute for CRISPOR's mismatch-tolerant off-target search."""
    if genome is None:
        return None
    hits = 0
    for m in re.finditer(re.escape(spacer), genome):
        pam = genome[m.end():m.end()+3]
        if len(pam) == 3 and pam[1:] == 'GG':
            hits += 1
    rcs = rc(spacer)
    for m in re.finditer(re.escape(rcs), genome):
        s = m.start()
        pam2 = genome[s-3:s] if s >= 3 else ''
        if len(pam2) == 3 and pam2[:2] == 'CC':
            hits += 1
    return hits

def seed_hits_genome(seed, genome):
    """Local occurrence count of a bare PAM-proximal seed (no PAM check) -
    a coarser, more sensitive flag than spacer_hits_genome."""
    if genome is None:
        return None
    return genome.count(seed) + genome.count(rc(seed))

# ------------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--genomic', required=True)
    ap.add_argument('--cds')
    ap.add_argument('--exon', type=int, default=None,
                    help='1-based coding exon to target (default: auto-pick early)')
    ap.add_argument('--gene-name', default='gene')
    ap.add_argument('--paralog-genomic',
                    help='genomic FASTA of a co-ortholog/paralog (e.g. the '
                         'sister ohnolog from the teleost genome duplication) '
                         'to screen candidate guides for cross-reactivity')
    ap.add_argument('--paralog-cds')
    ap.add_argument('--paralog-name', default='paralog')
    args = ap.parse_args()

    gen = read_fasta(args.genomic)
    print(f"# {args.gene_name}: genomic {len(gen)} bp")

    para_gen = read_fasta(args.paralog_genomic) if args.paralog_genomic else None
    if para_gen is not None:
        print(f"# paralog check enabled: {args.paralog_name} "
              f"genomic {len(para_gen)} bp")

    if args.cds:
        cds = read_fasta(args.cds)
        exons, ex_warnings = map_exons_from_cds(gen, cds)
        print(f"# mapped {len(exons)} coding exons from CDS ({len(cds)} bp)")
        if ex_warnings:
            print(f"# WARNING: could not anchor {len(ex_warnings)} CDS "
                  f"position(s) to the genomic sequence (near CDS offsets "
                  f"{ex_warnings[:5]}{'...' if len(ex_warnings) > 5 else ''}); "
                  f"likely a micro-exon or a genomic/CDS mismatch. Exon "
                  f"numbers after a gap should be treated as approximate.")
        if not exons:
            sys.exit("No exons could be mapped - check --genomic and --cds "
                      "are the same gene, assembly, and strand.")
        covered = sum(e[1]-e[0] for e in exons)
        print(f"# CDS coverage: {covered}/{len(cds)} bp "
              f"({100*covered/len(cds):.1f}%)")
        if covered < 0.9*len(cds):
            print("# WARNING: poor CDS coverage - check the genomic region spans "
                  "the whole gene, and that both are the same assembly/strand")
    else:
        print("# no CDS supplied: exon numbering UNAVAILABLE. "
              "Provide --cds for true exon numbers.")
        sys.exit("Supply --cds, or edit this script with a target exon interval.")

    # choose target exon: early, and ideally length %3 != 0 (skip-proof)
    if args.exon:
        idx = args.exon - 1
        if not (0 <= idx < len(exons)):
            sys.exit(f"--exon {args.exon} is out of range: only "
                      f"{len(exons)} coding exon(s) were mapped.")
    else:
        cand = [(i,e) for i,e in enumerate(exons[1:6], start=1)]
        cand.sort(key=lambda t: (0 if (t[1][1]-t[1][0]) % 3 else 1, t[0]))
        idx = cand[0][0] if cand else 0
        print(f"# auto-selected coding exon {idx+1} "
              f"(early + skip-proof preferred)")

    ex_s, ex_e, phase = exons[idx]
    exon = gen[ex_s:ex_e]
    print(f"\n## Target: coding exon {idx+1}  genomic {ex_s}-{ex_e}  "
          f"len {len(exon)}  phase_in {phase}")
    print(f"   acceptor ...{gen[max(0,ex_s-12):ex_s]}  ends AG: {gen[ex_s-2:ex_s]=='AG'}")
    print(f"   donor    {gen[ex_e:ex_e+12]}...  starts GT: {gen[ex_e:ex_e+2]=='GT'}")
    print(f"   len%3 = {len(exon)%3}  -> exon skipping "
          f"{'ALSO frameshifts (good)' if len(exon)%3 else 'stays in frame (weak)'}")
    print(f"   peptide: {tr(exon[(3-phase)%3:])}")

    guides = enumerate_guides(gen, ex_s, ex_e)
    if not guides:
        sys.exit(f"No NGG protospacers found cutting inside coding exon "
                  f"{idx+1}. Try a different --exon.")
    print(f"\n## {len(guides)} protospacers cutting inside this exon\n")
    hdr = (f"{'#':3} {'spacer':22} {'PAM':4} {'str':4} {'relCut':7} "
           f"{'GC%':5} {'polyT':6} {'OOF%':6} {'uniq':5} {'seed':5}"
           + (f" {args.paralog_name+'-hit':12}" if para_gen is not None else ""))
    print(hdr); print('-'*len(hdr))
    for n, g in enumerate(guides, 1):
        L, R = max(0, g['cut']-40), min(len(gen), g['cut']+40)
        win = gen[L:R]
        _, oof = mh_out_of_frame(win, g['cut']-L)
        uniq = (gen.count(g['spacer']) + gen.count(rc(g['spacer']))) == 1
        seed = g['spacer'][-12:]           # PAM-proximal 12nt seed
        seed_uniq = seed_hits_genome(seed, gen) == 1
        g['oof'] = oof
        para_col = ""
        if para_gen is not None:
            n_para = spacer_hits_genome(g['spacer'], para_gen)
            g['para_hits'] = n_para
            para_col = f" {(str(n_para)+' cut' if n_para else 'none'):12}"
        print(f"{n:<3} {g['spacer']:22} {g['pam']:4} {g['strand']:4} "
              f"{g['cut']-ex_s:<7} {gc(g['spacer']):5.0f} "
              f"{'YES' if 'TTTT' in g['spacer'] else '-':6} {oof:6.1f} "
              f"{'ok' if uniq else 'DUP':5} {'ok' if seed_uniq else 'DUP':5}"
              + para_col)

    print("\n## dual-guide pairings (deletion size must NOT be divisible by 3)")
    for a, b in itertools.combinations(range(len(guides)), 2):
        d = abs(guides[b]['cut'] - guides[a]['cut'])
        if d < 6:
            continue
        tag = 'FRAMESHIFT' if d % 3 else 'IN-FRAME  <-- avoid'
        print(f"   g{a+1}+g{b+1}: Δ{d:<4} {tag}")

    print("\n## single-guide indel consequences (PTC = premature stop codon)")
    for n, g in enumerate(guides, 1):
        rel = g['cut'] - ex_s
        res = []
        for ind in (-1, -2, +1, +2):
            fs, pep, ptc = consequence(exon, phase, rel, ind)
            res.append(f"{ind:+d}:{'FS' if fs else 'if'}"
                       f"{'/PTC'+str(ptc) if ptc else ''}")
        print(f"   g{n}: " + '  '.join(res))

    try:
        import primer3
        L, R = max(0, ex_s-260), min(len(gen), ex_e+260)
        tmpl = gen[L:R]
        target_start = max(0, ex_s-L-10)
        res = primer3.bindings.design_primers(
            {'SEQUENCE_ID': args.gene_name, 'SEQUENCE_TEMPLATE': tmpl,
             'SEQUENCE_TARGET': [target_start, (ex_e-ex_s)+20]},
            {'PRIMER_OPT_SIZE':21,'PRIMER_MIN_SIZE':18,'PRIMER_MAX_SIZE':25,
             'PRIMER_OPT_TM':60.0,'PRIMER_MIN_TM':57.0,'PRIMER_MAX_TM':63.0,
             'PRIMER_MIN_GC':35.0,'PRIMER_MAX_GC':65.0,
             'PRIMER_PRODUCT_SIZE_RANGE':[[150,450]],'PRIMER_NUM_RETURN':2})
        print("\n## genotyping primers")
        for k in range(res.get('PRIMER_PAIR_NUM_RETURNED', 0)):
            print(f"   pair {k+1}: {res[f'PRIMER_PAIR_{k}_PRODUCT_SIZE']} bp")
            print(f"     F {res[f'PRIMER_LEFT_{k}_SEQUENCE']}  "
                  f"Tm {res[f'PRIMER_LEFT_{k}_TM']:.1f}")
            print(f"     R {res[f'PRIMER_RIGHT_{k}_SEQUENCE']}  "
                  f"Tm {res[f'PRIMER_RIGHT_{k}_TM']:.1f}")
    except ImportError:
        print("\n## primer3 not installed; skipping primer design")
    except Exception as e:
        print(f"\n## primer3 primer design failed ({e}); skipping")

    print("\n## REMINDERS")
    print("   - run final spacers through CRISPOR (GRCz11) for genome-wide "
          "off-target + paralog check")
    print("   - Sanger-sequence spacer+PAM in YOUR background strain "
          "before ordering")
    print("   - order as synthetic sgRNA for RNP; 5' base is unconstrained")
    print("   - zebrafish TGD: if this gene has a co-ortholog (a/b paralog), "
          "run this script again on the paralog's own genomic+CDS to get an "
          "independent guide set - a single-paralog knockout may be "
          "phenotypically masked by the other copy")
    if para_gen is None:
        print("   - pass --paralog-genomic (and optionally --paralog-cds) to "
              "screen the guides above for cross-reactivity against a known "
              "paralog")

if __name__ == '__main__':
    main()
