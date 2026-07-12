from pathlib import Path

import numpy as np
import pytest

from wsme_gpcr.asr import (
    NodePosteriors,
    ambiguous_core_resnums,
    evaluate_node_trustworthiness,
    parse_iqtree_state_file,
    run_asr_sensitivity_check,
    site_to_resnum,
)
from wsme_gpcr.blocking import BlockModel
from wsme_gpcr.pipeline import run_pipeline
from wsme_gpcr.wsme import WSMEParams

CI2 = Path(__file__).parent.parent / "examples" / "data" / "CI2.pdb"

_SYNTHETIC_STATE_TEXT = """\
# Ancestral state reconstruction for all nodes in tree.treefile
# comment line
Node\tSite\tState\tp_A\tp_R\tp_N\tp_D
NodeA\t1\tA\t0.90\t0.05\t0.03\t0.02
NodeA\t2\tR\t0.10\t0.85\t0.03\t0.02
NodeA\t3\t-\t0.25\t0.25\t0.25\t0.25
NodeA\t4\tN\t0.40\t0.10\t0.45\t0.05
NodeB\t1\tD\t0.02\t0.03\t0.05\t0.90
NodeB\t2\tA\t0.60\t0.20\t0.10\t0.10
"""


def test_parse_iqtree_state_file(tmp_path):
    p = tmp_path / "x.state"
    p.write_text(_SYNTHETIC_STATE_TEXT)
    parsed = parse_iqtree_state_file(str(p))

    assert set(parsed.keys()) == {"NodeA", "NodeB"}
    a = parsed["NodeA"]
    assert isinstance(a, NodePosteriors)
    assert a.node == "NodeA"
    assert a.site.tolist() == [1, 2, 3, 4]
    assert a.state == ["A", "R", "-", "N"]
    np.testing.assert_allclose(a.map_posterior, [0.90, 0.85, 0.25, 0.45], atol=1e-9)
    # second-best at site 4 (N is MAP=0.45): A=0.40 is the runner-up
    assert a.second_state[3] == "A"
    assert a.second_posterior[3] == pytest.approx(0.40)


def test_parse_iqtree_state_file_sorts_by_site(tmp_path):
    # deliberately out-of-order rows must come back sorted by Site
    text = "Node\tSite\tState\tp_A\tp_R\n" "X\t3\tR\t0.1\t0.9\n" "X\t1\tA\t0.9\t0.1\n" "X\t2\tA\t0.9\t0.1\n"
    p = tmp_path / "y.state"
    p.write_text(text)
    parsed = parse_iqtree_state_file(str(p))
    assert parsed["X"].site.tolist() == [1, 2, 3]


def test_parse_iqtree_state_file_missing_header_raises(tmp_path):
    p = tmp_path / "bad.state"
    p.write_text("# just a comment\n# another comment\n")
    with pytest.raises(ValueError, match="no header row"):
        parse_iqtree_state_file(str(p))


def test_site_to_resnum_cumulative_non_gap():
    node = NodePosteriors(
        node="X", site=np.array([1, 2, 3, 4, 5]), state=["M", "-", "G", "-", "K"],
        map_posterior=np.ones(5), second_state=["A"] * 5, second_posterior=np.zeros(5),
    )
    # non-gap sites (1-indexed positions 1,3,5) get resnum 1,2,3; gap
    # rows repeat the preceding cumulative count (not meant to be used).
    np.testing.assert_array_equal(site_to_resnum(node), [1, 1, 2, 2, 3])


def _synthetic_block_model(nres, block_of_residue, block_size=4):
    nblocks = int(block_of_residue.max()) + 1
    return BlockModel(
        nres=nres, nblocks=nblocks, block_size=block_size, block_of_residue=block_of_residue,
        block_residue_range=np.zeros((nblocks, 2), dtype=int),
        block_cmap=np.zeros((nblocks, nblocks)), block_elec=np.zeros((0, 5)),
    )


def test_ambiguous_core_resnums_real_ci2():
    result = run_pipeline(CI2, ph=7.0)
    st = result.structure
    # CI2's own numbering starts at resnum 19 (not 1), so site_to_resnum's
    # cumulative-non-gap-count needs 18 leading non-gap "filler" sites
    # before the real structure's residues start -- otherwise every site
    # would map to a resnum outside the structure entirely.
    n_filler = int(st.author_resnum.min()) - 1
    n_real = 6
    n_total = n_filler + n_real
    site = np.arange(1, n_total + 1)
    state = ["N"] * n_total
    map_posterior = np.full(n_total, 0.5)  # all ambiguous
    node = NodePosteriors(node="X", site=site, state=state, map_posterior=map_posterior,
                           second_state=["A"] * n_total, second_posterior=np.full(n_total, 0.3))

    ala_resnums = {int(rn) for rn, rname in zip(st.author_resnum, st.resname) if rname == "ALA"}
    ambiguous = ambiguous_core_resnums(node, st, posterior_threshold=0.8)
    # the 6 real sites map to resnums n_filler+1 .. n_filler+6, i.e. CI2's
    # own first 6 residues -- every returned resnum must be among those,
    # real, in-structure, and non-ALA/GLY/PRO.
    expected_range = set(range(n_filler + 1, n_filler + n_real + 1))
    assert set(ambiguous).issubset(expected_range)
    assert len(ambiguous) > 0, "expected at least one mutable ambiguous position in CI2's first 6 residues"
    for rn in ambiguous:
        assert rn not in ala_resnums


def test_ambiguous_core_resnums_respects_threshold():
    node = NodePosteriors(
        node="X", site=np.array([1, 2]), state=["N", "D"],
        map_posterior=np.array([0.95, 0.5]), second_state=["A", "A"], second_posterior=np.array([0.03, 0.3]),
    )

    class FakeStructure:
        author_resnum = np.array([1, 2])
        resname = ["ASN", "ASP"]

    st = FakeStructure()
    # site 1 (posterior 0.95) is confident, site 2 (0.5) is ambiguous
    assert ambiguous_core_resnums(node, st, posterior_threshold=0.8) == [2]
    # a stricter threshold makes both ambiguous
    assert ambiguous_core_resnums(node, st, posterior_threshold=0.99) == [1, 2]


def test_run_asr_sensitivity_check_no_ambiguous_positions_is_trivially_robust():
    result = run_pipeline(CI2, ph=7.0)
    check = run_asr_sensitivity_check(
        result.structure, result.block_model, result.ss_mask, result.params, ambiguous_resnums=[],
    )
    assert check.sensitivity_ok is True
    assert check.delta_frac == 0.0
    assert check.mutant_fold_frac == check.wt_fold_frac
    assert check.trustworthy == check.fold_ok


def test_run_asr_sensitivity_check_real_ci2_plumbing():
    result = run_pipeline(CI2, ph=7.0)
    st = result.structure
    # a real, non-ALA/GLY/PRO position that exists in CI2
    candidates = [int(rn) for rn, rname in zip(st.author_resnum, st.resname)
                  if rname not in ("ALA", "GLY", "PRO")]
    assert candidates, "CI2 should have at least one mutable position"

    check = run_asr_sensitivity_check(
        st, result.block_model, result.ss_mask, result.params, ambiguous_resnums=[candidates[0]],
        node="TestNode", posterior_threshold=0.8,
    )
    assert check.node == "TestNode"
    assert check.posterior_threshold == 0.8
    assert check.nblocks == result.block_model.nblocks
    assert 0.0 <= check.wt_fold_frac <= 1.0
    assert 0.0 <= check.mutant_fold_frac <= 1.0
    assert check.delta_frac == pytest.approx(check.mutant_fold_frac - check.wt_fold_frac)
    assert check.trustworthy == (check.fold_ok and check.sensitivity_ok)


def test_asr_sensitivity_result_reason_messages():
    from wsme_gpcr.asr import AsrSensitivityResult

    trustworthy = AsrSensitivityResult(node="A", posterior_threshold=0.8, nblocks=10,
                                        wt_fold_frac=0.9, fold_ok=True, sensitivity_ok=True, trustworthy=True)
    assert "trustworthy" in trustworthy.reason() and "not trustworthy" not in trustworthy.reason()

    unfolded = AsrSensitivityResult(node="B", posterior_threshold=0.8, nblocks=10,
                                     wt_fold_frac=0.2, fold_ok=False, sensitivity_ok=True, trustworthy=False)
    assert "does not fold" in unfolded.reason()

    artifact = AsrSensitivityResult(node="C", posterior_threshold=0.8, nblocks=10, ambiguous_resnums=[1, 2, 3],
                                     wt_fold_frac=0.9, delta_frac=-0.4, fold_ok=True, sensitivity_ok=False,
                                     trustworthy=False)
    assert "reconstruction-uncertainty artifact" in artifact.reason()


def test_evaluate_node_trustworthiness_end_to_end_ci2():
    node = NodePosteriors(
        node="TestNode",
        site=np.arange(1, 65),
        state=["N"] * 64,
        map_posterior=np.full(64, 0.5),  # everything "ambiguous" -- stress test, not realistic
        second_state=["A"] * 64,
        second_posterior=np.full(64, 0.3),
    )
    result = evaluate_node_trustworthiness(str(CI2), node, use_dssp=False)
    assert result.node == "TestNode"
    assert isinstance(result.trustworthy, bool) or isinstance(result.trustworthy, np.bool_)
    assert 0.0 <= result.wt_fold_frac <= 1.0
