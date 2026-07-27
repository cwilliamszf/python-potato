Zebrafish muscle-gene cDNA sequences and direct human orthologues
Generated: 2026-07-27

CONTENTS
- zebrafish_representative_cdna.fa: one selected protein-coding transcript cDNA for each requested zebrafish gene.
- human_orthologue_representative_cdna.fa: one representative human orthologue cDNA for BIN1, RYR1, MYF6, and SPEG.
- all_representative_cdna.fa: all 10 unique records in one FASTA file.
- zebrafish_human_orthologue_map.tsv: gene/transcript mapping and sequence lengths.
- zebrafish/ and human/: individual FASTA files.
- SHA256SUMS.txt: sequence-file integrity checksums.

INTERPRETATION
“Orthologues” was interpreted as direct human orthologues. The zebrafish duplicate pairs bin1a/bin1b and spega/spegb therefore map to a single human BIN1 and SPEG record, respectively. Human orthologue records are included once in the human and combined FASTA files.

SEQUENCE TYPE
These are transcript cDNA sequences (spliced transcript DNA equivalents), not genomic loci and not CDS-only sequences. A single representative protein-coding transcript was selected per gene to avoid mixing splice isoforms.

ASSEMBLIES / ANNOTATION
- Danio rerio: GRCz11, versioned Ensembl/ZFIN-linked transcript records.
- Homo sapiens: GRCh38, versioned Ensembl transcripts linked to RefSeq/GENCODE 50 records.

SELECTED ZEBRAFISH TRANSCRIPTS
- bin1a: ENSDART00000081606.5 (1530 nt)
- bin1b: ENSDART00000081761.7 (1878 nt)
- ryr1b: ENSDART00000036015.9 (15237 nt)
- myf6: ENSDART00000040266.4 (720 nt)
- spega: ENSDART00000167225.2 (9444 nt)
- spegb: ENSDART00000138473.3 (11941 nt)

SELECTED HUMAN TRANSCRIPTS
- BIN1: ENST00000316724.10 / NM_139343.3 (2487 nt)
- RYR1: ENST00000359596.8 / NM_000540.3 (15400 nt)
- MYF6: ENST00000228641.4 / NM_002469.3 (1329 nt)
- SPEG: ENST00000312358.12 / NM_005876.5 (10782 nt)

VALIDATION
All records were normalized to uppercase, wrapped at 60 nucleotides per line, checked for A/C/G/T/N-only content, and checked against the expected transcript lengths.
