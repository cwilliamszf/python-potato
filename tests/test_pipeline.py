from pathlib import Path

from wsme_gpcr.pipeline import DEFAULT_PH_VALUES, run_pipeline, run_pipeline_multi_ph
from wsme_gpcr.wsme import WSMEParams

CI2 = Path(__file__).parent.parent / "examples" / "data" / "CI2.pdb"


def test_run_pipeline_smoke():
    pr = run_pipeline(CI2, ph=7.0, params=WSMEParams.soluble_protein_defaults())
    assert pr.structure.nres == 65
    assert pr.block_model.nblocks > 0
    assert pr.result.zfin > 0
    assert pr.dsc_result is None


def test_run_pipeline_multi_ph_covers_all_default_values():
    results = run_pipeline_multi_ph(CI2, params=WSMEParams.soluble_protein_defaults())
    assert set(results.keys()) == set(DEFAULT_PH_VALUES)
    for ph, pr in results.items():
        assert pr.ph == ph
        assert pr.result.zfin > 0

    # Different pH values should generally give different partition functions
    # (charge assignment changes both the contact map and electrostatics).
    zfins = {ph: pr.result.zfin for ph, pr in results.items()}
    assert len(set(zfins.values())) > 1
