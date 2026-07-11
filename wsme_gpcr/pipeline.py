"""End-to-end orchestration: PDB -> structure -> secondary structure ->
contacts -> blocks -> WSME (+ optional DSC). Shared by the CLI and the
Streamlit GUI so both stay in sync, and by multi-pH batch runs.

pH matters beyond just electrostatic screening: it changes which atoms
carry a nonzero titratable charge (e.g. histidine's ND1/NE2 are neutral
at pH 7 but protonated below ~pH 6), which feeds back into the VdW
contact map itself (short-range contacts specifically exclude pairs
where *both* atoms are charged). So a pH sweep re-derives the structure,
contact map, and blocks at each pH, not just the electrostatic term.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np

from .blocking import BlockModel, build_blocks
from .contacts import ContactMap, compute_contact_map
from .coupling import CouplingResult, compute_coupling
from .dsc import DSCResult, compute_dsc
from .secondary_structure import assign_secondary_structure, secondary_structure_from_codes
from .structure import Structure, load_structure
from .wsme import WSMEParams, WSMEResult, run_wsme


@dataclass
class PipelineResult:
    ph: float
    structure: Structure
    ss_mask: np.ndarray
    contact_map: ContactMap
    block_model: BlockModel
    params: WSMEParams
    result: WSMEResult
    dsc_result: DSCResult = None
    coupling_result: CouplingResult = None
    warnings: list = field(default_factory=list)


def run_pipeline(
    pdb_path,
    chain: str = None,
    model: int = 0,
    ph: float = 7.0,
    ss_codes: str = None,
    block_size: int = 4,
    params: WSMEParams = None,
    with_dsc: bool = False,
    dsc_T_grid=None,
    with_coupling: bool = False,
) -> PipelineResult:
    """Run the full landscape (and optionally DSC / coupling) pipeline for one pH."""
    if params is None:
        params = WSMEParams()

    caught_warnings = []
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        structure = load_structure(pdb_path, chain=chain, model=model, ph=ph)
        caught_warnings = [str(w.message) for w in caught]

    if ss_codes is not None:
        ss_mask = secondary_structure_from_codes(ss_codes)
    else:
        ss_mask = assign_secondary_structure(structure)

    contact_map = compute_contact_map(structure)
    block_model = build_blocks(ss_mask, contact_map, block_size=block_size)
    result = run_wsme(structure, block_model, ss_mask, params)

    dsc_result = None
    if with_dsc:
        dsc_result = compute_dsc(structure, block_model, ss_mask, params, T_grid=dsc_T_grid)

    coupling_result = None
    if with_coupling:
        coupling_result = compute_coupling(structure, block_model, ss_mask, params)

    return PipelineResult(
        ph=ph,
        structure=structure,
        ss_mask=ss_mask,
        contact_map=contact_map,
        block_model=block_model,
        params=params,
        result=result,
        dsc_result=dsc_result,
        coupling_result=coupling_result,
        warnings=caught_warnings,
    )


DEFAULT_PH_VALUES = (7.0, 5.0, 3.5, 2.0)


def run_pipeline_multi_ph(
    pdb_path,
    ph_values=DEFAULT_PH_VALUES,
    chain: str = None,
    model: int = 0,
    ss_codes: str = None,
    block_size: int = 4,
    params: WSMEParams = None,
    with_dsc: bool = False,
    dsc_T_grid=None,
    with_coupling: bool = False,
    progress_callback=None,
) -> dict:
    """Run the pipeline independently at each pH. Returns {ph: PipelineResult}.

    ``progress_callback(ph, index, total)`` is called before each pH run,
    if given (useful for GUI progress bars on what can be a slow batch).
    """
    results = {}
    total = len(ph_values)
    for i, ph in enumerate(ph_values):
        if progress_callback:
            progress_callback(ph, i, total)
        results[ph] = run_pipeline(
            pdb_path,
            chain=chain,
            model=model,
            ph=ph,
            ss_codes=ss_codes,
            block_size=block_size,
            params=params,
            with_dsc=with_dsc,
            dsc_T_grid=dsc_T_grid,
            with_coupling=with_coupling,
        )
    return results
