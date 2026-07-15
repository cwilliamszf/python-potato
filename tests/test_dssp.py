import shutil
from pathlib import Path

import numpy as np
import pytest

from wsme_gpcr.secondary_structure import (
    DsspNotAvailableError,
    _ensure_legacy_pdb_header,
    _find_dssp_binary,
    _parse_dssp_output,
    secondary_structure_from_dssp,
)
from wsme_gpcr.pipeline import run_pipeline
from wsme_gpcr.structure import load_structure

HAVE_DSSP = shutil.which("mkdssp") is not None or shutil.which("dssp") is not None
GPCR9I_PDB = "examples/data/gpcr_landscapes_reference/gpcr9i.pdb"

_SYNTHETIC_DSSP_TEXT = """\
  #  RESIDUE AA STRUCTURE BP1 BP2  ACC     N-H-->O    O-->H-N    N-H-->O    O-->H-N    TCO  KAPPA ALPHA  PHI   PSI    X-CA   Y-CA   Z-CA
    1    1 A M              0   0  180      0, 0.0     2,-0.3     0, 0.0     0, 0.0   0.000 360.0 360.0 360.0 155.4    0.0    0.0    0.0
    2    2 A K  H           0   0  150      0, 0.0     2,-0.3     0, 0.0     0, 0.0   0.000 360.0  60.0 -60.0 -40.0    0.0    0.0    0.0
    3    3 A A  H           0   0  120      0, 0.0     2,-0.3     0, 0.0     0, 0.0   0.000 360.0  60.0 -60.0 -40.0    0.0    0.0    0.0
    4    4 A L  E           0   0   90      0, 0.0     2,-0.3     0, 0.0     0, 0.0   0.000 360.0  60.0 -60.0 -40.0    0.0    0.0    0.0
    5    5 A G                 0   0   60      0, 0.0     2,-0.3     0, 0.0     0, 0.0   0.000 360.0  60.0 -60.0 -40.0    0.0    0.0    0.0
    6        !*             0   0    0      0, 0.0     0, 0.0     0, 0.0     0, 0.0   0.000 360.0 360.0 360.0 360.0    0.0    0.0    0.0
    7    7 B S              0   0   30      0, 0.0     2,-0.3     0, 0.0     0, 0.0   0.000 360.0  60.0 -60.0 -40.0    0.0    0.0    0.0
"""


def test_parse_dssp_output_basic():
    codes = _parse_dssp_output(_SYNTHETIC_DSSP_TEXT, chain_id="A")
    assert codes == {1: "-", 2: "H", 3: "H", 4: "E", 5: "-"}
    # chain break sentinel (row 6, "!") and the other chain's residue (row 7,
    # chain B) must both be excluded when filtering to chain A.
    assert 6 not in codes
    assert 7 not in codes


def test_parse_dssp_output_no_chain_filter_includes_all_chains():
    codes = _parse_dssp_output(_SYNTHETIC_DSSP_TEXT, chain_id=None)
    assert 7 in codes
    assert codes[7] == "-"


def test_parse_dssp_output_missing_header_raises():
    with pytest.raises(RuntimeError, match="RESIDUE"):
        _parse_dssp_output("not a real dssp file\n", chain_id="A")


def test_ensure_legacy_pdb_header_passthrough_for_cif(tmp_path):
    cif_path = tmp_path / "x.cif"
    cif_path.write_text("data_x\n")
    assert _ensure_legacy_pdb_header(str(cif_path), str(tmp_path)) == str(cif_path)


def test_ensure_legacy_pdb_header_passthrough_when_header_present(tmp_path):
    pdb_path = tmp_path / "x.pdb"
    pdb_path.write_text("HEADER    SOMETHING\nATOM      1  N   MET A   1\nEND\n")
    assert _ensure_legacy_pdb_header(str(pdb_path), str(tmp_path)) == str(pdb_path)


def test_ensure_legacy_pdb_header_adds_header_when_missing(tmp_path):
    pdb_path = tmp_path / "x.pdb"
    pdb_path.write_text("PFRMAT TS\nATOM      1  N   MET A   1\nEND\n")
    fixed = _ensure_legacy_pdb_header(str(pdb_path), str(tmp_path))
    assert fixed != str(pdb_path)
    fixed_text = Path(fixed).read_text()
    assert fixed_text.startswith("HEADER")
    assert "PFRMAT" not in fixed_text
    assert "ATOM      1  N   MET A   1" in fixed_text
    # original file must be untouched
    assert "PFRMAT" in pdb_path.read_text()


def test_find_dssp_binary_raises_dssp_not_available_error(monkeypatch):
    monkeypatch.setattr(shutil, "which", lambda name: None)
    with pytest.raises(DsspNotAvailableError, match="mkdssp"):
        _find_dssp_binary()


@pytest.mark.skipif(not HAVE_DSSP, reason="requires mkdssp")
def test_secondary_structure_from_dssp_real_gpcr9i_matches_known_result():
    """Regression test tied to FINDINGS.md's block-partition audit: real
    DSSP on gpcr9i (4DKL) was found to reproduce the paper's own real
    block count (76) exactly, unlike the geometric heuristic (75)."""
    structure = load_structure(GPCR9I_PDB, ph=7.0)
    mask = secondary_structure_from_dssp(GPCR9I_PDB, structure)
    assert mask.shape == (structure.nres,)
    assert mask.dtype == bool
    assert np.all(np.isin(mask, [True, False]))
    # matches the audit's reported structured-residue count for HGE-only
    assert mask.sum() == 242


@pytest.mark.skipif(not HAVE_DSSP, reason="requires mkdssp")
def test_run_pipeline_use_dssp_end_to_end_reproduces_paper_block_count():
    result = run_pipeline(GPCR9I_PDB, ph=7.0, use_dssp=True)
    assert result.block_model.nblocks == 76  # paper's real BlockDet_gpcr9i count
    assert result.ss_mask.sum() > 0


@pytest.mark.skipif(not HAVE_DSSP, reason="requires mkdssp")
def test_run_pipeline_use_dssp_differs_from_geometric_default():
    r_dssp = run_pipeline(GPCR9I_PDB, ph=7.0, use_dssp=True)
    r_geom = run_pipeline(GPCR9I_PDB, ph=7.0, use_dssp=False)
    assert r_dssp.block_model.nblocks != r_geom.block_model.nblocks


def test_run_pipeline_use_dssp_false_is_unaffected_by_dssp_availability():
    # default behavior (use_dssp=False) must not even attempt to find mkdssp.
    result = run_pipeline(GPCR9I_PDB, ph=7.0, use_dssp=False)
    assert result.block_model.nblocks > 0
