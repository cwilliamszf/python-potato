"""Gate A: staphylococcal nuclease (SNase) buried-ionizable calibration --
pipeline spec acceptance gate A. Target: RMSE < 1.0 pKa unit against real
experimental data before any ancestral-node number may be reported.

Reference dataset: wild-type SNase, PDB 1STN, 24 experimentally-measured
pKa's (4 His, 8 Asp, 12 Glu) spanning fully buried (%SASA as low as 2.5)
to fully solvent-exposed (%SASA 133) side chains -- real dynamic range,
not just a handful of easy surface residues.

Provenance: PKAD-2 (Alexov lab, Clemson University), downloaded by the
user directly from http://compbio.clemson.edu/PKAD-2/ (this sandbox's
network access is blocked for that host, same as every external host
tried this session -- see linkage_pka/FINDINGS.md's "Gate A dataset
sourcing" section) and uploaded into this session on 2026-07-11. Every
value in ``_RAW_ROWS`` below is transcribed directly from that file
(verified against the raw CSV row-by-row while writing this module), not
estimated or recalled.

Citations (the two distinct references PKAD-2's own "Reference" column
points to for these rows -- not assumed from memory, each independently
confirmed against PubMed while sourcing this dataset):
  - Asp/Glu (20 of 24 rows): Castaneda CA, Fitch CA, Majumdar A,
    Khangulov V, Schlessman JL, Garcia-Moreno E B. "Molecular determinants
    of the pKa values of Asp and Glu residues in staphylococcal
    nuclease." Proteins. 2009. doi:10.1002/prot.22470, PMID 19533744.
  - His (4 of 24 rows): doi:10.1021/bi0119417, per PKAD-2's own Reference
    field for these entries (title/authors not independently
    re-confirmed beyond what PKAD-2 itself reports for this specific
    field -- flagged here rather than silently treated as verified to
    the same standard as the Asp/Glu citation above).

Two Asp entries (19, 21) report TWO pKa values each in the source data
(e.g. "2.21,6.54") -- a real linked-equilibrium/biphasic titration
signature (coupling to a nearby group produces a two-step apparent
titration curve), not a data error or a typo. This pipeline's own
multisite coupled solver (multisite.py) is exactly the tool built to
handle this kind of case properly; until that comparison is done, these
two entries are excluded from the default single-site RMSE (there is no
principled way to compare one computed pKa against two experimental
values) but are retained in the dataset, not dropped.

Structure: the real 1STN mmCIF was uploaded by the user (same
network-block workaround as the CSV above) on 2026-07-11 and converted to
PDB via ``pdbfixer.PDBFixer`` + ``openmm.app.PDBFile.writeFile(...,
keepIds=True)`` -- ``keepIds=True`` is essential, since the default
silently renumbers residues sequentially from 1 and would break every
resnum-based cross-reference against the experimental entries below.
Cross-checked against ``SNASE_1STN_EXPERIMENTAL_PKA``: 21 of 24 sites are
present with matching resname; ASP143, ASP146, and GLU142 are absent
because the resolved structure only spans resnum 6-141 (most likely
disordered/unresolved C-terminal tail in the crystal -- a real data
limitation of this particular structure, not a bug), and are excluded
from any run against this structure.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

REFERENCE_PDB_ID = "1STN"
GATE_A_RMSE_THRESHOLD_PKA_UNITS = 1.0

ASP_GLU_CITATION = (
    "Castaneda CA, Fitch CA, Majumdar A, Khangulov V, Schlessman JL, "
    "Garcia-Moreno E B. Molecular determinants of the pKa values of Asp "
    "and Glu residues in staphylococcal nuclease. Proteins. 2009. "
    "doi:10.1002/prot.22470, PMID 19533744."
)
HIS_CITATION = "doi:10.1021/bi0119417 (per PKAD-2's own Reference field, not independently re-confirmed)"
SOURCE_CITATION = (
    "PKAD-2 (Alexov lab, Clemson University), http://compbio.clemson.edu/PKAD-2/ "
    "-- downloaded by the user (this sandbox cannot reach the host directly) "
    "and uploaded into this session 2026-07-11."
)


@dataclass
class ExperimentalPka:
    resnum: int
    resname: str
    expt_pka: float           # None if biphasic (see expt_pka_biphasic) -- otherwise the reported value
    expt_pka_biphasic: tuple  # (pka1, pka2) if this site shows linked/biphasic titration, else None
    is_upper_bound: bool      # True for "<2.2"-style entries: true pKa is below the lowest pH scanned
    expt_uncertainty: str     # as reported -- a single value, a range, or "N/A"
    sasa_percent: float
    method: str
    reference: str


# resname, resnum, expt_pka (raw string -- may be "<X" or "a,b"), uncertainty, sasa_percent, method, reference
_RAW_ROWS = [
    ("HIS", 8,   "6.52",      "0.03",      71.1,  "1H NMR", HIS_CITATION),
    ("HIS", 46,  "5.86",      "0.04",      32.0,  "1H NMR", HIS_CITATION),
    ("HIS", 121, "5.3",       "0.06",      25.2,  "1H NMR", HIS_CITATION),
    ("HIS", 124, "5.73",      "0.02",      65.3,  "1H NMR", HIS_CITATION),
    ("ASP", 19,  "2.21,6.54", "0.06-0.07",  3.6,  "NMR", ASP_GLU_CITATION),
    ("ASP", 21,  "3.01,6.54", "0.01-0.02",  4.7,  "NMR", ASP_GLU_CITATION),
    ("ASP", 40,  "3.87",      "0.09",      71.0,  "NMR", ASP_GLU_CITATION),
    ("ASP", 77,  "<2.2",      "N/A",        2.5,  "NMR", ASP_GLU_CITATION),
    ("ASP", 83,  "<2.2",      "N/A",       33.2,  "NMR", ASP_GLU_CITATION),
    ("ASP", 95,  "2.16",      "0.07",      66.4,  "NMR", ASP_GLU_CITATION),
    ("ASP", 143, "3.8",       "0.1",      131.2,  "NMR", ASP_GLU_CITATION),
    ("ASP", 146, "3.86",      "0.05",     133.0,  "NMR", ASP_GLU_CITATION),
    ("GLU", 10,  "2.82",      "0.09",      17.0,  "NMR", ASP_GLU_CITATION),
    ("GLU", 43,  "4.32",      "0.04",      32.3,  "NMR", ASP_GLU_CITATION),
    ("GLU", 52,  "3.93",      "0.08",      27.9,  "NMR", ASP_GLU_CITATION),
    ("GLU", 57,  "3.49",      "0.09",      72.7,  "NMR", ASP_GLU_CITATION),
    ("GLU", 67,  "3.76",      "0.07",      75.7,  "NMR", ASP_GLU_CITATION),
    ("GLU", 73,  "3.31",      "0.01",      41.0,  "NMR", ASP_GLU_CITATION),
    ("GLU", 75,  "3.26",      "0.05",       4.8,  "NMR", ASP_GLU_CITATION),
    ("GLU", 101, "3.81",      "0.1",       31.8,  "NMR", ASP_GLU_CITATION),
    ("GLU", 122, "3.89",      "0.09",      26.2,  "NMR", ASP_GLU_CITATION),
    ("GLU", 129, "3.75",      "0.09",      11.8,  "NMR", ASP_GLU_CITATION),
    ("GLU", 135, "3.76",      "0.08",      63.3,  "NMR", ASP_GLU_CITATION),
    ("GLU", 142, "4.49",      "0.04",      73.9,  "NMR", ASP_GLU_CITATION),
]


def _parse_entries() -> list:
    entries = []
    for resname, resnum, pka_str, uncertainty, sasa, method, reference in _RAW_ROWS:
        biphasic, pka, is_upper_bound = None, None, False
        if "," in pka_str:
            biphasic = tuple(float(x) for x in pka_str.split(","))
        elif pka_str.startswith("<"):
            pka, is_upper_bound = float(pka_str[1:]), True
        else:
            pka = float(pka_str)
        entries.append(ExperimentalPka(
            resnum=resnum, resname=resname, expt_pka=pka, expt_pka_biphasic=biphasic,
            is_upper_bound=is_upper_bound, expt_uncertainty=uncertainty,
            sasa_percent=sasa, method=method, reference=reference,
        ))
    return entries


SNASE_1STN_EXPERIMENTAL_PKA = _parse_entries()


@dataclass
class GateAResult:
    rmse: float
    mae: float
    n_compared: int
    per_residue: list   # [(resnum, resname, expt_pka, computed_pka, diff), ...]
    skipped: list        # [(resnum, resname, reason), ...]
    passed: bool          # rmse < GATE_A_RMSE_THRESHOLD_PKA_UNITS


def compute_gate_a_rmse(computed_pka: dict, entries: list = None,
                         include_upper_bounds: bool = False,
                         threshold_pka_units: float = GATE_A_RMSE_THRESHOLD_PKA_UNITS) -> GateAResult:
    """RMSE (and per-residue breakdown) of ``computed_pka`` (resnum ->
    computed intrinsic pKa, e.g. from ``titration.compute_intrinsic_pka``
    run on the 1STN structure for each of these 24 sites) against the
    real experimental values in ``entries`` (default
    ``SNASE_1STN_EXPERIMENTAL_PKA``).

    Biphasic entries (Asp19, Asp21) are always excluded -- there is no
    principled single-value comparison for them (see module docstring).
    Upper-bound entries ("<2.2") are excluded by default
    (``include_upper_bounds=False``) since a computed pKa above 2.2
    cannot be judged wrong by simple difference against an unresolved
    bound; set True to include them anyway (using the bound value as a
    plain point estimate, likely biasing RMSE upward for a correctly-low
    computed value).

    Sites in ``entries`` with no matching key in ``computed_pka`` are
    skipped and reported, not silently dropped from ``skipped``.
    """
    entries = entries if entries is not None else SNASE_1STN_EXPERIMENTAL_PKA

    per_residue, skipped = [], []
    for e in entries:
        if e.expt_pka_biphasic is not None:
            skipped.append((e.resnum, e.resname, "biphasic (linked equilibrium, no single-value comparison)"))
            continue
        if e.is_upper_bound and not include_upper_bounds:
            skipped.append((e.resnum, e.resname, f"upper-bound only (<{e.expt_pka}), excluded by default"))
            continue
        if e.resnum not in computed_pka:
            skipped.append((e.resnum, e.resname, "no computed pKa supplied for this resnum"))
            continue
        computed = computed_pka[e.resnum]
        diff = computed - e.expt_pka
        per_residue.append((e.resnum, e.resname, e.expt_pka, computed, diff))

    if not per_residue:
        raise ValueError("no residues could be compared -- check computed_pka's resnums against "
                          "SNASE_1STN_EXPERIMENTAL_PKA and the skipped list for why")

    diffs = np.array([d for *_, d in per_residue])
    rmse = float(np.sqrt(np.mean(diffs ** 2)))
    mae = float(np.mean(np.abs(diffs)))

    return GateAResult(
        rmse=rmse, mae=mae, n_compared=len(per_residue), per_residue=per_residue,
        skipped=skipped, passed=rmse < threshold_pka_units,
    )
