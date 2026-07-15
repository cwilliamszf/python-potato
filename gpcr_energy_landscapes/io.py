"""
Loaders for the two inputs this tool needs from the rest of the pipeline:

1. An ensemble of conformer structures (tool 2's output: diverse
   conformations generated from an AlphaFold model, already protonated
   per-residue by tool 1 according to pKa/pH).
2. A table of per-structure Gibbs free energies (tool 3's output).

The interface contract is intentionally simple so it can be adapted to
whatever tools 1-3 actually emit:

* Structures: one PDB (or mmCIF) file per conformer, named
  ``<structure_id>.pdb`` inside a single directory. ``structure_id`` is the
  join key against the energy table.
* Energies: a CSV (or any pandas-readable table) with at minimum a
  structure-id column and a Gibbs free energy column (kcal/mol). An optional
  weight/population column is supported for ensembles where tool 3 also
  reports sampling weights.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import pandas as pd
from Bio.PDB import MMCIFParser, PDBParser
from Bio.PDB.Structure import Structure

_PDB_PARSER = PDBParser(QUIET=True)
_CIF_PARSER = MMCIFParser(QUIET=True)


def load_structure(path: str | Path, structure_id: Optional[str] = None) -> Structure:
    """Load a single conformer structure from a PDB or mmCIF file."""
    path = Path(path)
    sid = structure_id or path.stem
    if path.suffix.lower() in (".cif", ".mmcif"):
        return _CIF_PARSER.get_structure(sid, str(path))
    return _PDB_PARSER.get_structure(sid, str(path))


def load_ensemble(ensemble_dir: str | Path, pattern: str = "*.pdb") -> Dict[str, Structure]:
    """Load a directory of conformer files into ``{structure_id: Structure}``.

    ``structure_id`` is taken from each file's stem (filename without
    extension), which must match the id used in the energies table so the
    two can be joined in :func:`gpcr_energy_landscapes.pipeline.merge_with_energies`.
    """
    ensemble_dir = Path(ensemble_dir)
    files = sorted(ensemble_dir.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No structures matching '{pattern}' found in {ensemble_dir}")
    return {f.stem: load_structure(f) for f in files}


def load_energies(
    table: str | Path | pd.DataFrame,
    id_col: str = "structure_id",
    gibbs_col: str = "gibbs_kcal_mol",
) -> pd.DataFrame:
    """Load the per-structure Gibbs free energy table produced by tool 3.

    ``table`` may be a path to a CSV file or an already-loaded DataFrame.
    Returns a DataFrame indexed by ``structure_id``. Any additional columns
    (e.g. a sampling weight, ligand identity, pH) are passed through
    untouched and are available for downstream use (see ``weight_col`` in
    :mod:`gpcr_energy_landscapes.pipeline`).
    """
    df = table if isinstance(table, pd.DataFrame) else pd.read_csv(table)
    missing = {id_col, gibbs_col} - set(df.columns)
    if missing:
        raise ValueError(
            f"Energies table is missing required column(s) {sorted(missing)}; "
            f"found columns: {list(df.columns)}"
        )
    return df.set_index(id_col)
