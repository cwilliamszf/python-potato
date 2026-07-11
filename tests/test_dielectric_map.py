import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from linkage_pka.dielectric_map import (
    DxMap,
    compute_dummy_maps,
    compute_energy_with_maps,
    read_dx,
    splice_membrane_slab,
    write_dx,
    write_maps,
)
from linkage_pka.membrane_frame import MembraneFrame

CI2 = Path(__file__).parent.parent / "examples" / "data" / "CI2.pdb"
APBS_AVAILABLE = shutil.which("apbs") is not None and shutil.which("pdb2pqr30") is not None


def _synthetic_dxmap(shape=(4, 5, 6), fill=78.54, origin=(0.0, 0.0, 0.0), delta=(1.0, 1.0, 1.0)) -> DxMap:
    data = np.full(shape, fill, dtype=float)
    return DxMap(data=data, origin=np.array(origin, dtype=float), delta=np.array(delta, dtype=float))


def test_dxmap_grid_positions_known_small_case():
    dxmap = _synthetic_dxmap(shape=(2, 2, 2), origin=(10.0, 20.0, 30.0), delta=(1.0, 2.0, 3.0))
    positions = dxmap.grid_positions()
    assert positions.shape == (8, 3)
    # First point (i=j=k=0) is exactly the origin.
    assert positions[0] == pytest.approx([10.0, 20.0, 30.0])
    # Flat order is z-fastest: index 1 is (i=0,j=0,k=1).
    assert positions[1] == pytest.approx([10.0, 20.0, 33.0])
    # Last point is (i=1,j=1,k=1).
    assert positions[-1] == pytest.approx([11.0, 22.0, 33.0])


def test_read_write_dx_roundtrip(tmp_path):
    rng = np.random.default_rng(0)
    dxmap = DxMap(
        data=rng.uniform(2.0, 78.54, size=(3, 4, 5)),
        origin=np.array([1.5, -2.5, 3.25]),
        delta=np.array([0.5, 0.5, 0.5]),
    )
    path = tmp_path / "test.dx"
    write_dx(path, dxmap, "dielx")
    reread = read_dx(path)

    assert reread.counts == dxmap.counts
    assert reread.origin == pytest.approx(dxmap.origin)
    assert reread.delta == pytest.approx(dxmap.delta)
    assert reread.data == pytest.approx(dxmap.data, rel=1e-6)


def _flat_membrane_frame(origin, axis, half_thickness):
    return MembraneFrame(
        origin=np.array(origin, dtype=float), axis=np.array(axis, dtype=float), half_thickness_ang=half_thickness,
        tm_mask_method="synthetic", tm_mask_resnums=[], plddt_threshold=float("nan"),
        r350_resnum=0, dry_motif="", y753_resnum=0, npxxy_motif="",
        half_thickness_fitted=False, explained_variance_ratio=1.0,
    )


def test_splice_membrane_slab_leaves_protein_interior_alone():
    shape = (6, 6, 6)
    dielx = _synthetic_dxmap(shape, fill=78.54)
    # Mark half the grid (x < 3) as "protein interior" (pdie).
    dielx.data[:3, :, :] = 2.0
    maps = {
        "dielx": dielx,
        "diely": _synthetic_dxmap(shape, fill=78.54),
        "dielz": _synthetic_dxmap(shape, fill=78.54),
        "kappa": _synthetic_dxmap(shape, fill=0.106),  # nonzero -> ion-accessible
        "charge": _synthetic_dxmap(shape, fill=0.0),
    }
    # Slab covers the whole grid along x (origin at grid center, huge half-thickness).
    frame = _flat_membrane_frame(origin=(2.5, 2.5, 2.5), axis=(1.0, 0.0, 0.0), half_thickness=100.0)

    spliced = splice_membrane_slab(maps, frame, membrane_dielectric=2.5, sdie=78.54)

    # Bulk-solvent points (x >= 3) get overridden to the membrane dielectric.
    assert np.all(spliced["dielx"].data[3:, :, :] == 2.5)
    # Protein-interior points (x < 3) are untouched, even though they're
    # geometrically "inside" the slab too.
    assert np.all(spliced["dielx"].data[:3, :, :] == 2.0)
    # Kappa is zeroed everywhere the slab covers (the whole grid here).
    assert np.all(spliced["kappa"].data == 0.0)
    # Charge is never touched by splicing.
    assert np.array_equal(spliced["charge"].data, maps["charge"].data)


def test_splice_membrane_slab_only_affects_points_inside_slab():
    shape = (6, 6, 6)
    maps = {
        "dielx": _synthetic_dxmap(shape, fill=78.54),
        "diely": _synthetic_dxmap(shape, fill=78.54),
        "dielz": _synthetic_dxmap(shape, fill=78.54),
        "kappa": _synthetic_dxmap(shape, fill=0.106),
        "charge": _synthetic_dxmap(shape, fill=0.0),
    }
    # Thin slab: with integer grid spacing, only x=2 and x=3 (distance 0.5
    # from the x=2.5 center) fall within a 0.6 half-thickness.
    frame = _flat_membrane_frame(origin=(2.5, 2.5, 2.5), axis=(1.0, 0.0, 0.0), half_thickness=0.6)
    spliced = splice_membrane_slab(maps, frame, membrane_dielectric=3.0, sdie=78.54)

    changed = spliced["dielx"].data != maps["dielx"].data
    assert changed.any()
    assert not changed.all()  # a thin slab must not cover the whole grid
    # Every changed point really is within the slab thickness of x=2.5.
    positions = maps["dielx"].grid_positions().reshape(shape + (3,))
    assert np.all(np.abs(positions[changed, 0] - 2.5) <= 0.6 + 1e-9)


@pytest.mark.skipif(not APBS_AVAILABLE, reason="requires apbs and pdb2pqr30 on PATH")
def test_compute_dummy_maps_and_membrane_splice_end_to_end(tmp_path):
    """Real integration test: PDB2PQR -> APBS mg-dummy maps -> splice a
    membrane slab -> re-solve via usemap, on a small grid for speed.
    Confirms the whole external-tool pipeline actually works together and
    that splicing a membrane measurably changes the physics."""
    pqr_path = tmp_path / "ci2.pqr"
    result = subprocess.run(
        ["pdb2pqr30", "--ff", "AMBER", "--with-ph", "7.0", "--titration-state-method", "propka",
         str(CI2), str(pqr_path)],
        capture_output=True, text=True, timeout=120,
    )
    assert result.returncode == 0, result.stderr

    dime = (33, 33, 33)
    glen = (50.0, 50.0, 50.0)
    gcent = "mol 1"
    pdie, sdie, ion = 2.0, 78.54, 0.150

    maps = compute_dummy_maps(pqr_path, dime, glen, gcent, pdie, sdie, ion, tmp_path / "dummy")
    assert set(maps.keys()) == {"dielx", "diely", "dielz", "kappa", "charge"}
    for m in maps.values():
        assert m.data.shape == dime
        assert np.isfinite(m.data).all()

    dielx = maps["dielx"]
    center = dielx.origin + 0.5 * np.array(dielx.counts) * dielx.delta
    frame = _flat_membrane_frame(origin=center, axis=(1.0, 0.0, 0.0), half_thickness=10.0)
    spliced = splice_membrane_slab(maps, frame, membrane_dielectric=2.0, sdie=sdie)

    plain_paths = write_maps(maps, tmp_path / "plain", stem_suffix="_plain")
    mem_paths = write_maps(spliced, tmp_path / "mem", stem_suffix="_mem")

    e_plain = compute_energy_with_maps(pqr_path, plain_paths, dime, glen, gcent, pdie, sdie, ion, tmp_path / "solve_plain")
    e_mem = compute_energy_with_maps(pqr_path, mem_paths, dime, glen, gcent, pdie, sdie, ion, tmp_path / "solve_mem")

    assert np.isfinite(e_plain) and np.isfinite(e_mem)
    assert e_mem != pytest.approx(e_plain, rel=1e-6)  # membrane slab must change the energy


@pytest.mark.skipif(not APBS_AVAILABLE, reason="requires apbs and pdb2pqr30 on PATH")
def test_compute_energy_with_maps_roundtrips_to_direct_solve(tmp_path):
    """Reading back APBS's own dummy-pass maps unmodified and re-solving via
    usemap must reproduce a direct (mol-based, non-usemap) solve to high
    precision -- the strongest possible check that read_dx/write_dx are
    faithful to APBS's own file format, not just self-consistent."""
    pqr_path = tmp_path / "ci2.pqr"
    result = subprocess.run(
        ["pdb2pqr30", "--ff", "AMBER", "--with-ph", "7.0", "--titration-state-method", "propka",
         str(CI2), str(pqr_path)],
        capture_output=True, text=True, timeout=120,
    )
    assert result.returncode == 0, result.stderr

    dime = (33, 33, 33)
    glen = (50.0, 50.0, 50.0)
    gcent = "mol 1"
    pdie, sdie, ion = 2.0, 78.54, 0.150

    direct_input = f"""\
read
    mol pqr {pqr_path.name}
end
elec name solv
    mg-manual
    dime {' '.join(str(d) for d in dime)}
    glen {' '.join(f'{g:.3f}' for g in glen)}
    gcent {gcent}
    mol 1
    lpbe
    bcfl mdh
    ion charge 1 conc {ion:.4f} radius 2.0
    ion charge -1 conc {ion:.4f} radius 2.0
    pdie {pdie}
    sdie {sdie}
    chgm spl2
    srfm mol
    srad 1.4
    swin 0.3
    sdens 10.0
    temp 298.15
    calcenergy total
    calcforce no
end
print elecEnergy solv end
quit
"""
    direct_dir = tmp_path / "direct"
    direct_dir.mkdir()
    shutil.copy(pqr_path, direct_dir / pqr_path.name)
    from linkage_pka.dielectric_map import _run_apbs, _ENERGY_RE
    stdout = _run_apbs(direct_input, direct_dir, input_name="direct.in")
    e_direct = float(_ENERGY_RE.findall(stdout)[-1])

    maps = compute_dummy_maps(pqr_path, dime, glen, gcent, pdie, sdie, ion, tmp_path / "dummy")
    paths = write_maps(maps, tmp_path / "roundtrip", stem_suffix="_rt")
    e_roundtrip = compute_energy_with_maps(pqr_path, paths, dime, glen, gcent, pdie, sdie, ion, tmp_path / "solve_rt")

    assert e_roundtrip == pytest.approx(e_direct, rel=1e-4)
