"""OpenDX dielectric/kappa map I/O and membrane-slab splicing for the
APBS-based Poisson-Boltzmann step of the linkage pipeline -- pipeline spec
step 2 (dielectric slab) x step 3 (PB solver).

Strategy: the real APBSmem approach (Callenberg, Choudhary, de Forest,
Gohara, Baker, Grabe, "APBSmem: A Graphical Interface for Electrostatic
Calculations at the Membrane," PLOS ONE 5(9):e12722, 2010 -- citation
verified via PubMed/PLOS listings, see chat history). Rather than
reimplementing molecular-surface geometry (a substantial undertaking on
its own -- Connolly/SES construction), let APBS compute its own
molecular-surface-based dielectric/kappa/charge maps for the protein (a
well-tested, correct algorithm already in APBS) on a cheap `mg-dummy`
(setup-only, no PB solve) pass, using `write dielx/diely/dielz/kappa/
charge dx <name>` -- confirmed against APBS 3.4.1's own bundled example
input files (/usr/share/apbs/examples/helix/Apbs_dummy.in, which is
itself a membrane/TM-protein example). Then splice in the membrane slab
in Python: wherever a grid point falls inside the membrane (per
``MembraneFrame.in_slab``) AND APBS decided it's *outside* the protein
(its dielectric value is the bulk-solvent one, not the low protein-
interior value), override it to the membrane's low dielectric, and zero
the kappa (ion-accessibility) value there too, since ions don't
penetrate the bilayer. Grid points already inside the protein are left
alone -- the protein interior isn't lipid, regardless of its height
along the membrane normal. The modified maps are written back out as
.dx files and fed to a second, real APBS solve via `usemap diel/kappa/
charge` (also confirmed against APBS's bundled examples, which use
exactly this read-back-and-override pattern for TM-protein calculations).

OpenDX format notes (verified empirically against APBS 3.4.1's own
output, not assumed from memory): the three dielectric maps (dielx/
diely/dielz) are each on their own axis-shifted staggered grid -- each
file's ``origin`` line already encodes that shift (e.g. dielx's origin is
offset by +delta_x/2 in x only relative to kappa/charge's shared
cell-center origin), so every map can be treated uniformly: compute each
grid point's real-space position from *that file's own* origin/delta/
counts, with no additional manual half-cell adjustment needed. Data is
row-major with the z index fastest-varying (shape (nx, ny, nz),
flat index = ix*ny*nz + iy*nz + iz), confirmed by round-tripping a
written-and-reread map and cross-checking dielectric value fractions
against the known solute/solvent split.
"""

from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

_DX_TYPE_COMMENT = {
    "dielx": "X-SHIFTED DIELECTRIC MAP",
    "diely": "Y-SHIFTED DIELECTRIC MAP",
    "dielz": "Z-SHIFTED DIELECTRIC MAP",
    "kappa": "KAPPA MAP",
    "charge": "CHARGE DISTRIBUTION",
}


@dataclass
class DxMap:
    data: np.ndarray     # (nx, ny, nz)
    origin: np.ndarray   # (3,) Angstrom
    delta: np.ndarray    # (3,) Angstrom, grid spacing along x/y/z (axis-aligned grids only)

    @property
    def counts(self) -> tuple:
        return self.data.shape

    def grid_positions(self) -> np.ndarray:
        """(nx*ny*nz, 3) real-space coordinates of every grid point, in the
        same flat (z-fastest) order as ``data.ravel()``."""
        nx, ny, nz = self.counts
        ix, iy, iz = np.meshgrid(np.arange(nx), np.arange(ny), np.arange(nz), indexing="ij")
        return np.stack([
            self.origin[0] + ix.ravel() * self.delta[0],
            self.origin[1] + iy.ravel() * self.delta[1],
            self.origin[2] + iz.ravel() * self.delta[2],
        ], axis=1)


def read_dx(path) -> DxMap:
    """Parse an OpenDX scalar grid file as written by APBS."""
    lines = Path(path).read_text().splitlines()
    counts = None
    deltas = []
    header_end = None
    n_items = None
    for i, line in enumerate(lines):
        if line.startswith("object 1 class gridpositions"):
            counts = tuple(int(x) for x in line.split()[-3:])
        elif line.startswith("origin"):
            origin = np.array([float(x) for x in line.split()[1:4]])
        elif line.startswith("delta"):
            deltas.append(np.array([float(x) for x in line.split()[1:4]]))
        elif line.startswith("object 3 class array"):
            n_items = int(line.split("items")[1].split()[0])
            header_end = i + 1
            break
    if counts is None or header_end is None:
        raise ValueError(f"{path}: not a recognizable APBS OpenDX file (missing header fields)")

    delta = np.array([deltas[0][0], deltas[1][1], deltas[2][2]])

    data_lines = []
    for line in lines[header_end:]:
        if line.startswith("attribute"):
            break
        data_lines.append(line)
    values = np.fromstring(" ".join(data_lines), sep=" ")
    if len(values) != n_items:
        raise ValueError(f"{path}: expected {n_items} data values, got {len(values)}")

    return DxMap(data=values.reshape(counts), origin=origin, delta=delta)


def write_dx(path, dxmap: DxMap, dx_type: str):
    """Write a DxMap back out in the same OpenDX layout APBS itself uses
    (verified by round-tripping through APBS's own `usemap` re-read --
    see tests)."""
    nx, ny, nz = dxmap.counts
    n_items = nx * ny * nz
    comment = _DX_TYPE_COMMENT.get(dx_type, dx_type.upper())
    lines = [
        "# Data from linkage_pka",
        "#",
        f"# {comment}",
        "#",
        f"object 1 class gridpositions counts {nx} {ny} {nz}",
        f"origin {dxmap.origin[0]:e} {dxmap.origin[1]:e} {dxmap.origin[2]:e}",
        f"delta {dxmap.delta[0]:e} 0.000000e+00 0.000000e+00",
        f"delta 0.000000e+00 {dxmap.delta[1]:e} 0.000000e+00",
        f"delta 0.000000e+00 0.000000e+00 {dxmap.delta[2]:e}",
        f"object 2 class gridconnections counts {nx} {ny} {nz}",
        f"object 3 class array type double rank 0 items {n_items} data follows",
    ]
    flat = dxmap.data.ravel()
    for i in range(0, n_items, 3):
        chunk = flat[i:i + 3]
        lines.append(" ".join(f"{v:e}" for v in chunk))
    lines += [
        'attribute "dep" string "positions"',
        'object "regular positions regular connections" class field',
        'component "positions" value 1',
        'component "connections" value 2',
        'component "data" value 3',
    ]
    Path(path).write_text("\n".join(lines) + "\n")


def _run_apbs(input_text: str, work_dir: Path, input_name: str = "apbs.in"):
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    input_path = work_dir / input_name
    input_path.write_text(input_text)
    result = subprocess.run(
        ["apbs", input_name], cwd=work_dir, capture_output=True, text=True, timeout=600,
    )
    if result.returncode != 0:
        raise RuntimeError(f"apbs failed (exit {result.returncode}) on {input_path}:\n{result.stdout}\n{result.stderr}")
    return result.stdout


_MAP_TYPES = ("dielx", "diely", "dielz", "kappa", "charge")


def compute_dummy_maps(pqr_path, dime, glen, gcent, pdie: float, sdie: float,
                        ion_strength_m: float, work_dir, srad: float = 1.4, swin: float = 0.3) -> dict:
    """Run APBS's `mg-dummy` (setup-only, no PB solve) with a sharp
    molecular ("mol") surface definition to get its own dielectric/kappa/
    charge maps for the bare protein, before any membrane slab is spliced
    in. Returns {"dielx": DxMap, "diely": DxMap, ..., "charge": DxMap}.
    """
    work_dir = Path(work_dir)
    dime_str = " ".join(str(d) for d in dime)
    glen_str = " ".join(f"{g:.3f}" for g in glen)
    input_text = f"""\
read
    mol pqr {Path(pqr_path).name}
end
elec name dummy
    mg-dummy
    dime {dime_str}
    glen {glen_str}
    gcent {gcent}
    mol 1
    lpbe
    bcfl mdh
    ion charge 1 conc {ion_strength_m:.4f} radius 2.0
    ion charge -1 conc {ion_strength_m:.4f} radius 2.0
    pdie {pdie}
    sdie {sdie}
    chgm spl2
    srfm mol
    srad {srad}
    swin {swin}
    sdens 10.0
    temp 298.15
    calcenergy no
    calcforce no
    write dielx dx dielx
    write diely dx diely
    write dielz dx dielz
    write kappa dx kappa
    write charge dx charge
end
quit
"""
    import shutil
    work_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(pqr_path, work_dir / Path(pqr_path).name)
    _run_apbs(input_text, work_dir, input_name="dummy_maps.in")

    maps = {}
    for map_type in _MAP_TYPES:
        candidates = list(work_dir.glob(f"{map_type}-PE*.dx")) or list(work_dir.glob(f"{map_type}.dx"))
        if not candidates:
            raise RuntimeError(f"expected APBS to write a {map_type} map in {work_dir}, found none")
        maps[map_type] = read_dx(candidates[0])
    return maps


def splice_membrane_slab(maps: dict, frame, membrane_dielectric: float = 2.0, sdie: float = 78.54,
                          sdie_atol: float = 1e-6) -> dict:
    """Return a new maps dict with the membrane slab (from ``frame``,
    a ``linkage_pka.membrane_frame.MembraneFrame``) spliced into the
    dielectric and kappa maps. A grid point is overridden only if it's
    inside the slab (``frame.in_slab``) AND APBS's own molecular-surface
    decision already placed it in bulk solvent (dielectric == sdie,
    kappa != 0) -- protein interior always wins over the membrane, since
    the protein occupies real volume the lipid can't also occupy. The
    charge map is untouched (surface charge is a property of the protein
    alone in this idealized model, not of the membrane).
    """
    out = dict(maps)
    for key in ("dielx", "diely", "dielz"):
        dxmap = maps[key]
        positions = dxmap.grid_positions()
        in_slab = frame.in_slab(positions).reshape(dxmap.counts)
        is_bulk = np.isclose(dxmap.data, sdie, atol=sdie_atol)
        override = in_slab & is_bulk
        new_data = np.where(override, membrane_dielectric, dxmap.data)
        out[key] = replace(dxmap, data=new_data)

    kappa = maps["kappa"]
    positions = kappa.grid_positions()
    in_slab = frame.in_slab(positions).reshape(kappa.counts)
    is_accessible = ~np.isclose(kappa.data, 0.0)
    override = in_slab & is_accessible
    out["kappa"] = replace(kappa, data=np.where(override, 0.0, kappa.data))

    return out


def write_maps(maps: dict, work_dir, stem_suffix: str = "_mem") -> dict:
    """Write each map in ``maps`` to ``<work_dir>/<type><stem_suffix>.dx``.
    Returns {"dielx": Path, ...}."""
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for map_type, dxmap in maps.items():
        path = work_dir / f"{map_type}{stem_suffix}.dx"
        write_dx(path, dxmap, map_type)
        paths[map_type] = path
    return paths


_ENERGY_RE = re.compile(r"Global net ELEC energy\s*=\s*([-+0-9.eE]+)\s*kJ/mol")


def compute_energy_with_maps(pqr_path, map_paths: dict, dime, glen, gcent,
                              pdie: float, sdie: float, ion_strength_m: float, work_dir,
                              srad: float = 1.4, swin: float = 0.3) -> float:
    """Real PB solve (`mg-manual`, `calcenergy total`) using the (possibly
    membrane-spliced) maps in ``map_paths`` instead of recomputing
    dielectric/kappa/charge from the molecule -- the standard APBS
    `usemap` pattern (confirmed against APBS's bundled TM-protein example,
    /usr/share/apbs/examples/helix/Apbs_solv-TEMPLATE.in). Returns the
    total electrostatic energy in kJ/mol.
    """
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    import shutil
    shutil.copy(pqr_path, work_dir / Path(pqr_path).name)
    for p in map_paths.values():
        if p.parent != work_dir:
            shutil.copy(p, work_dir / p.name)

    dime_str = " ".join(str(d) for d in dime)
    glen_str = " ".join(f"{g:.3f}" for g in glen)
    read_lines = [
        f"    mol pqr {Path(pqr_path).name}",
        f"    diel dx {map_paths['dielx'].name} {map_paths['diely'].name} {map_paths['dielz'].name}",
        f"    kappa dx {map_paths['kappa'].name}",
        f"    charge dx {map_paths['charge'].name}",
    ]
    input_text = f"""\
read
{chr(10).join(read_lines)}
end
elec name solv
    mg-manual
    dime {dime_str}
    glen {glen_str}
    gcent {gcent}
    mol 1
    lpbe
    bcfl mdh
    ion charge 1 conc {ion_strength_m:.4f} radius 2.0
    ion charge -1 conc {ion_strength_m:.4f} radius 2.0
    pdie {pdie}
    sdie {sdie}
    chgm spl2
    srfm mol
    srad {srad}
    swin {swin}
    sdens 10.0
    temp 298.15
    usemap diel 1
    usemap kappa 1
    usemap charge 1
    calcenergy total
    calcforce no
end
print elecEnergy solv end
quit
"""
    stdout = _run_apbs(input_text, work_dir, input_name="solve.in")
    matches = _ENERGY_RE.findall(stdout)
    if not matches:
        raise RuntimeError(f"could not find 'Global net ELEC energy' in APBS output:\n{stdout}")
    return float(matches[-1])
