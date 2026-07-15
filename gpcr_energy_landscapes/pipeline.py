"""
Orchestration: ensemble + energies -> CV table -> landscape / embedding.

Typical usage::

    from gpcr_energy_landscapes import io, pipeline
    from gpcr_energy_landscapes.collective_variables import BETA2AR_MICROSWITCHES

    ensemble = io.load_ensemble("conformers/")          # tool 2's output
    energies = io.load_energies("gibbs_energies.csv")   # tool 3's output
    refs = {"active": io.load_structure("active_ref.pdb"),
            "inactive": io.load_structure("inactive_ref.pdb")}

    cv_table = pipeline.compute_cv_table(ensemble, BETA2AR_MICROSWITCHES, refs=refs)
    merged = pipeline.merge_with_energies(cv_table, energies)

    landscape = pipeline.build_1d_landscape(merged, "tm5_bulge")
    landscape2d = pipeline.build_2d_landscape(merged, "tm5_bulge", "ionic_lock")
    embedding, model = pipeline.build_embedding_landscape(
        merged, feature_cols=["tm5_bulge", "ionic_lock"], method="pca"
    )
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from Bio.PDB.Structure import Structure

from . import collective_variables as cv
from . import dimensionality_reduction as dr
from . import energy_landscape as el


def compute_cv_table(
    ensemble: Dict[str, Structure],
    cv_defs: List[Dict],
    refs: Optional[Dict[str, Structure]] = None,
    model_index: int = 0,
) -> pd.DataFrame:
    """Evaluate a list of CV definitions against every structure in an
    ensemble. Returns a DataFrame indexed by structure_id, one column per CV."""
    rows = []
    for structure_id, structure in ensemble.items():
        row = {"structure_id": structure_id}
        row.update(cv.evaluate_cvs(structure, cv_defs, model_index=model_index, refs=refs))
        rows.append(row)
    return pd.DataFrame(rows).set_index("structure_id")


def merge_with_energies(cv_table: pd.DataFrame, energies: pd.DataFrame) -> pd.DataFrame:
    """Join a CV table (from :func:`compute_cv_table`) with an energies table
    (from :func:`gpcr_energy_landscapes.io.load_energies`) on structure_id."""
    merged = cv_table.join(energies, how="inner")
    if merged.empty:
        raise ValueError(
            "No overlapping structure_id between the CV table and the energies table -- "
            "check that tool 2's conformer filenames match tool 3's structure_id column."
        )
    dropped = len(cv_table) - len(merged)
    if dropped:
        import warnings

        warnings.warn(f"{dropped} structure(s) present in the ensemble had no matching energy and were dropped.")
    return merged


def build_1d_landscape(
    merged: pd.DataFrame,
    cv_name: str,
    gibbs_col: str = "gibbs_kcal_mol",
    weight_col: Optional[str] = None,
    temperature: float = 310.0,
    **kwargs,
) -> Dict[str, np.ndarray]:
    weights = merged[weight_col].to_numpy() if weight_col else None
    return el.landscape_1d(
        merged[cv_name].to_numpy(),
        gibbs=merged[gibbs_col].to_numpy(),
        weights=weights,
        temperature=temperature,
        **kwargs,
    )


def build_2d_landscape(
    merged: pd.DataFrame,
    cv_x: str,
    cv_y: str,
    gibbs_col: str = "gibbs_kcal_mol",
    weight_col: Optional[str] = None,
    temperature: float = 310.0,
    **kwargs,
) -> Dict[str, np.ndarray]:
    weights = merged[weight_col].to_numpy() if weight_col else None
    return el.landscape_2d(
        merged[cv_x].to_numpy(),
        merged[cv_y].to_numpy(),
        gibbs=merged[gibbs_col].to_numpy(),
        weights=weights,
        temperature=temperature,
        **kwargs,
    )


def build_embedding_landscape(
    merged: pd.DataFrame,
    feature_cols: List[str],
    method: str = "pca",
    inverse: bool = True,
    gibbs_col: str = "gibbs_kcal_mol",
    **kwargs,
):
    """Embed the ensemble in PC/MDS/t-SNE space (Figure 3 style) and attach
    each conformer's Gibbs free energy for downstream coloring. Returns
    ``(embedding_df, fitted_model)``."""
    _, matrix = dr.build_feature_matrix({name: merged[name].to_numpy() for name in feature_cols})
    if inverse:
        matrix = dr.inverse_distance_features(matrix)
    embedding, model = dr.embed(matrix, method=method, **kwargs)
    columns = [f"{method}_{i + 1}" for i in range(embedding.shape[1])]
    embedding_df = pd.DataFrame(embedding, columns=columns, index=merged.index)
    embedding_df[gibbs_col] = merged[gibbs_col]
    return embedding_df, model
