import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from linkage_pka.pipeline import (
    ActivationLinkageResult,
    StructureTitrationResult,
    compute_activation_linkage,
    find_coupled_pairs,
    identify_titratable_sites,
    residue_min_distance,
    run_structure_titration,
)
from linkage_pka.titration import GridParams, PqrAtom, read_pqr
from linkage_pka.multisite import MultiSiteTitrationResult

CI2 = Path(__file__).parent.parent / "examples" / "data" / "CI2.pdb"
PDB2PQR_AVAILABLE = shutil.which("pdb2pqr30") is not None
APBS_AVAILABLE = shutil.which("apbs") is not None and PDB2PQR_AVAILABLE


@pytest.fixture(scope="module")
def ci2_pqr(tmp_path_factory):
    if not PDB2PQR_AVAILABLE:
        pytest.skip("requires pdb2pqr30 on PATH")
    work = tmp_path_factory.mktemp("linkage_pipeline_pqr")
    pqr_path = work / "ci2.pqr"
    result = subprocess.run(
        ["pdb2pqr30", "--ff", "AMBER", "--with-ph", "7.0", "--titration-state-method", "propka",
         str(CI2), str(pqr_path)],
        capture_output=True, text=True, timeout=120,
    )
    assert result.returncode == 0, result.stderr
    return pqr_path


def _synthetic_atoms():
    # Two ASP residues close together (should couple), one LYS far away
    # (should not), and a non-titratable ALA (should be excluded entirely).
    atoms = []
    for resnum, resname, cx in [(1, "ASP", 0.0), (2, "ASH", 3.0), (3, "LYS", 100.0), (4, "ALA", 50.0)]:
        for name, dx in [("CA", 0.0), ("CB", 0.5), ("CG", 1.0)]:
            atoms.append(PqrAtom(
                serial=len(atoms) + 1, name=name, resname=resname, resnum=resnum,
                x=cx + dx, y=0.0, z=0.0, charge=0.0, radius=1.7,
            ))
    return atoms


def test_identify_titratable_sites_canonicalizes_and_excludes_non_titratable():
    atoms = _synthetic_atoms()
    sites = identify_titratable_sites(atoms)
    assert sites == [(1, "ASP"), (2, "ASP"), (3, "LYS")]  # ALA (resnum 4) excluded


def test_identify_titratable_sites_excludes_arg():
    atoms = [
        PqrAtom(serial=1, name="CA", resname="ARG", resnum=1, x=0, y=0, z=0, charge=0.0, radius=1.7),
    ]
    assert identify_titratable_sites(atoms) == []


def test_residue_min_distance_is_closest_approach_not_centroid():
    atoms = _synthetic_atoms()
    d = residue_min_distance(atoms, 1, 2)
    # closest atoms are CG@1.0 (resnum1) and CA@3.0 (resnum2) -> 2.0 Angstrom
    assert d == pytest.approx(2.0, abs=1e-6)


def test_find_coupled_pairs_respects_distance_cutoff():
    atoms = _synthetic_atoms()
    sites = identify_titratable_sites(atoms)
    close_pairs = find_coupled_pairs(atoms, sites, distance_cutoff_ang=10.0)
    assert close_pairs == [((1, "ASP"), (2, "ASP"))]  # only the close pair within 10 A

    all_pairs = find_coupled_pairs(atoms, sites, distance_cutoff_ang=1000.0)
    assert len(all_pairs) == 3  # all C(3,2) pairs among the 3 titratable sites


def test_find_coupled_pairs_empty_for_single_site():
    atoms = _synthetic_atoms()
    sites = [(1, "ASP")]
    assert find_coupled_pairs(atoms, sites) == []


def test_compute_activation_linkage_requires_matching_ph_grids():
    ph1 = np.array([6.0, 7.0])
    ph2 = np.array([6.0, 7.5])
    ms1 = MultiSiteTitrationResult(ph=ph1, theta={1: np.array([0.5, 0.4])}, clusters=[[1]],
                                    cluster_results=[], ln_z_total=np.array([0.1, 0.2]))
    ms2 = MultiSiteTitrationResult(ph=ph2, theta={1: np.array([0.6, 0.3])}, clusters=[[1]],
                                    cluster_results=[], ln_z_total=np.array([0.1, 0.2]))
    active = StructureTitrationResult(ph=ph1, sites=[(1, "ASP")], site_energies={}, pka_intrinsic={1: 7.0},
                                       coupling={}, coupled_pairs=[], multisite=ms1)
    inactive = StructureTitrationResult(ph=ph2, sites=[(1, "ASP")], site_energies={}, pka_intrinsic={1: 7.0},
                                         coupling={}, coupled_pairs=[], multisite=ms2)
    with pytest.raises(ValueError):
        compute_activation_linkage(active, inactive)


def test_compute_activation_linkage_from_synthetic_multisite_results():
    ph = np.array([6.0, 7.0, 8.0])
    theta_active = {1: np.array([0.9, 0.5, 0.1])}
    theta_inactive = {1: np.array([0.6, 0.5, 0.4])}
    ln_z_active = np.array([0.3, 0.2, 0.1])
    ln_z_inactive = np.array([0.1, 0.2, 0.3])

    ms_active = MultiSiteTitrationResult(ph=ph, theta=theta_active, clusters=[[1]],
                                          cluster_results=[], ln_z_total=ln_z_active)
    ms_inactive = MultiSiteTitrationResult(ph=ph, theta=theta_inactive, clusters=[[1]],
                                            cluster_results=[], ln_z_total=ln_z_inactive)
    active = StructureTitrationResult(ph=ph, sites=[(1, "ASP")], site_energies={}, pka_intrinsic={1: 7.0},
                                       coupling={}, coupled_pairs=[], multisite=ms_active)
    inactive = StructureTitrationResult(ph=ph, sites=[(1, "ASP")], site_energies={}, pka_intrinsic={1: 7.0},
                                         coupling={}, coupled_pairs=[], multisite=ms_inactive)

    result = compute_activation_linkage(active, inactive)
    assert isinstance(result, ActivationLinkageResult)
    assert np.allclose(result.delta_n_h, theta_active[1] - theta_inactive[1])
    assert list(result.resnums) == [1]
    top = result.top_contributors(7.0, n=1)
    assert top[0][0] == 1


@pytest.mark.skipif(not APBS_AVAILABLE, reason="requires apbs and pdb2pqr30 on PATH")
def test_run_structure_titration_end_to_end_on_two_close_sites(ci2_pqr, tmp_path):
    atoms = read_pqr(ci2_pqr)
    asp_resnums = sorted({a.resnum for a in atoms if a.resname == "ASP"})
    glu_resnums = sorted({a.resnum for a in atoms if a.resname == "GLU"})
    # Restrict to exactly 2 sites to keep this within the same runtime
    # budget as the single-site/single-pair tests in test_titration.py.
    sites = [(asp_resnums[0], "ASP"), (glu_resnums[0], "GLU")]

    grid = GridParams(dime=(33, 33, 33), glen=(50.0, 50.0, 50.0), gcent="mol 1",
                       pdie=2.0, sdie=78.54, ion_strength_m=0.150)
    model_grid = GridParams(dime=(33, 33, 33), glen=(25.0, 25.0, 25.0), gcent="mol 1",
                             pdie=2.0, sdie=78.54, ion_strength_m=0.150)

    result = run_structure_titration(
        atoms, frame=None, protein_grid_params=grid, model_grid_params=model_grid,
        ph_values=[6.0, 7.0, 8.0], work_dir=tmp_path / "struct",
        sites=sites, coupling_distance_cutoff_ang=1000.0,  # force the pair to be evaluated
    )

    assert set(result.pka_intrinsic) == {sites[0][0], sites[1][0]}
    assert all(np.isfinite(v) for v in result.pka_intrinsic.values())
    assert result.coupled_pairs == [(sites[0], sites[1])]
    assert (sites[0][0], sites[1][0]) in result.coupling
    assert np.isfinite(result.coupling[(sites[0][0], sites[1][0])])
    assert set(result.multisite.theta) == {sites[0][0], sites[1][0]}
    for arr in result.multisite.theta.values():
        assert np.all(arr >= 0.0) and np.all(arr <= 1.0)
