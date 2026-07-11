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

from .alanine_scan import AlanineScanResult, run_alanine_scan
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
    pka_overrides: dict = None,
    ss_codes: str = None,
    block_size: int = 4,
    params: WSMEParams = None,
    with_dsc: bool = False,
    dsc_T_grid=None,
    with_coupling: bool = False,
) -> PipelineResult:
    """Run the full landscape (and optionally DSC / coupling) pipeline for one pH.

    ``pka_overrides`` maps an author residue number to a custom pKa (see
    ``structure.load_structure``), for probing a specific residue proposed
    to have an environment-shifted pKa (e.g. a candidate pH sensor).
    """
    if params is None:
        params = WSMEParams()

    caught_warnings = []
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        structure = load_structure(pdb_path, chain=chain, model=model, ph=ph, pka_overrides=pka_overrides)
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
    pka_overrides: dict = None,
    ss_codes: str = None,
    block_size: int = 4,
    params: WSMEParams = None,
    with_dsc: bool = False,
    dsc_T_grid=None,
    with_coupling: bool = False,
    progress_callback=None,
) -> dict:
    """Run the pipeline independently at each pH. Returns {ph: PipelineResult}.
    ``ph_values`` may be any iterable of floats -- a fine sweep (e.g.
    ``np.arange(6.0, 7.9, 0.2)``) works as well as a handful of values.

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
            pka_overrides=pka_overrides,
            ss_codes=ss_codes,
            block_size=block_size,
            params=params,
            with_dsc=with_dsc,
            dsc_T_grid=dsc_T_grid,
            with_coupling=with_coupling,
        )
    return results


@dataclass
class AlanineScanPipelineResult:
    ph: float
    structure: Structure
    ss_mask: np.ndarray
    block_model: BlockModel
    params: WSMEParams
    scan: AlanineScanResult
    warnings: list = field(default_factory=list)


def run_alanine_scan_pipeline(
    pdb_path,
    chain: str = None,
    model: int = 0,
    ph: float = 7.0,
    pka_overrides: dict = None,
    ss_codes: str = None,
    block_size: int = 4,
    params: WSMEParams = None,
    positions=None,
    max_positions: int = None,
    progress_callback=None,
) -> AlanineScanPipelineResult:
    """Load a structure and run a (by default, receptor-wide) alanine
    scan on it -- the general-purpose version of the workflow in
    alanine_scan.py, applicable to any PDB/mmCIF structure, not tied to
    a specific receptor.

    ``positions=None`` scans every eligible residue; pass an explicit
    list of author resnums to target specific sites, or ``max_positions``
    to evenly subsample the full site list for a faster run. See
    ``alanine_scan.estimate_scan_seconds`` for a time estimate before
    committing to a large scan.
    """
    if params is None:
        params = WSMEParams()

    caught_warnings = []
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        structure = load_structure(pdb_path, chain=chain, model=model, ph=ph, pka_overrides=pka_overrides)
        caught_warnings = [str(w.message) for w in caught]

    if ss_codes is not None:
        ss_mask = secondary_structure_from_codes(ss_codes)
    else:
        ss_mask = assign_secondary_structure(structure)

    contact_map = compute_contact_map(structure)
    block_model = build_blocks(ss_mask, contact_map, block_size=block_size)

    scan = run_alanine_scan(
        structure, ss_mask, params, positions=positions, max_positions=max_positions,
        block_size=block_size, wt_block_model=block_model,
        wt_chi_plus=compute_coupling(structure, block_model, ss_mask, params).chi_plus,
        progress_callback=progress_callback,
    )

    return AlanineScanPipelineResult(
        ph=ph, structure=structure, ss_mask=ss_mask, block_model=block_model,
        params=params, scan=scan, warnings=caught_warnings,
    )
