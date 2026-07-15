"""Buried ionizable network detection, inspired by pHinder (Isom lab).

pHinder (no public source code was found -- this is an independent
reimplementation of the published two-step method, not a port) searches
a protein structure for spatially clustered, buried ionizable residues,
on the hypothesis that burying a titratable group is energetically
costly unless compensated by a specific protonation state -- making such
clusters natural candidates for pH-sensing or pH-dependent conformational
switches. See:

  Isom DG et al., "Buried ionizable networks are an ancient hallmark of
  G protein-coupled receptor activation." PNAS 2015.

Published methodology (two steps):
  1. Delaunay-triangulate the terminal side-chain atoms of all ionizable
     residues (Asp, Glu, His, Cys, Lys, Arg), then trim edges longer than
     10 A -- this is the "ionizable network."
  2. Classify each residue's burial by its depth relative to the
     molecular surface (buried / margin / exposed).

Step 1 is reproduced directly (scipy Delaunay + 10 A trim). Step 2's
true molecular-surface depth calculation isn't reimplemented here (that
needs a proper rolling-probe surface, e.g. via freesasa/MSMS, which
isn't a dependency of this package); instead burial is approximated by
local heavy-atom packing density (a standard, cheap desolvation/burial
proxy), with buried/margin/exposed assigned by percentile within the
structure rather than pHinder's fixed Angstrom thresholds -- documented
explicitly so it isn't mistaken for the original algorithm's output.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.spatial import Delaunay, cKDTree

from .structure import Structure

IONIZABLE_RESIDUES = {"ASP", "GLU", "HIS", "CYS", "LYS", "ARG"}

# One representative point per ionizable group: the centroid of its
# terminal (charge-bearing / titratable) side-chain atom(s).
REPRESENTATIVE_ATOMS = {
    "ASP": ("OD1", "OD2"),
    "GLU": ("OE1", "OE2"),
    "HIS": ("ND1", "NE2"),
    "CYS": ("SG",),
    "LYS": ("NZ",),
    "ARG": ("NE", "NH1", "NH2"),
}

EDGE_CUTOFF_ANGSTROM = 10.0


@dataclass
class IonizableNetworkResult:
    residue_index: np.ndarray  # (n,) 0-based residue index into the Structure
    author_resnum: np.ndarray  # (n,) author residue number
    resname: list  # (n,) 3-letter code
    position: np.ndarray  # (n, 3) representative coordinate
    burial_score: np.ndarray  # (n,) heavy-atom neighbor count within burial_radius
    burial_class: list  # (n,) "buried" / "margin" / "exposed"
    edges: list  # [(i, j)] indices into the arrays above, trimmed Delaunay edges
    networks: list = field(default_factory=list)  # connected components: list of index-lists

    def summary_table(self):
        """One row per ionizable residue, network id, and burial class."""
        network_id = np.full(len(self.residue_index), -1, dtype=int)
        for nid, members in enumerate(self.networks):
            for m in members:
                network_id[m] = nid
        rows = []
        for i in range(len(self.residue_index)):
            rows.append({
                "author_resnum": int(self.author_resnum[i]),
                "resname": self.resname[i],
                "burial_class": self.burial_class[i],
                "burial_score": int(self.burial_score[i]),
                "network_id": int(network_id[i]),
                "network_size": len(self.networks[network_id[i]]) if network_id[i] >= 0 else 0,
            })
        return rows

    def candidate_sensor_networks(self, min_size: int = 2, require_buried: bool = True):
        """Networks containing >=1 His alongside >=1 Asp/Glu (or another
        His), the classic proximity motif for a pKa-shifted, pH-sensing
        histidine -- optionally restricted to networks with at least one
        buried member. Returns a list of the (index-list) networks."""
        candidates = []
        for members in self.networks:
            if len(members) < min_size:
                continue
            has_his = any(self.resname[m] == "HIS" for m in members)
            has_acidic_or_2nd_his = sum(1 for m in members if self.resname[m] in ("ASP", "GLU", "HIS")) >= 2
            if not (has_his and has_acidic_or_2nd_his):
                continue
            if require_buried and not any(self.burial_class[m] in ("buried", "margin") for m in members):
                continue
            candidates.append(members)
        return candidates


def _connected_components(n: int, edges: list) -> list:
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for i, j in edges:
        union(i, j)

    groups = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    return [sorted(v) for v in groups.values() if len(v) > 0]


def compute_ionizable_network(
    structure: Structure,
    edge_cutoff: float = EDGE_CUTOFF_ANGSTROM,
    burial_radius: float = 10.0,
    buried_percentile: float = 66.7,
    exposed_percentile: float = 33.3,
) -> IonizableNetworkResult:
    """Detect buried ionizable networks (pHinder-style) in ``structure``.

    ``burial_radius``: radius (A) for the heavy-atom-neighbor-count burial
    proxy. ``buried_percentile``/``exposed_percentile``: residues at or
    above the buried percentile of neighbor count are "buried", at or
    below the exposed percentile are "exposed", the rest are "margin"
    (percentile-based so the classification adapts to the packing density
    of the structure at hand, rather than pHinder's fixed Angstrom-depth
    thresholds against a true molecular surface).
    """
    residue_index, author_resnum, resname, position = [], [], [], []
    for ridx in range(structure.nres):
        rname = structure.resname[ridx]
        if rname not in IONIZABLE_RESIDUES:
            continue
        atom_names_wanted = REPRESENTATIVE_ATOMS[rname]
        mask = (structure.atom_resindex == ridx) & np.isin(structure.atom_name, atom_names_wanted)
        if not np.any(mask):
            continue
        residue_index.append(ridx)
        author_resnum.append(structure.author_resnum[ridx])
        resname.append(rname)
        position.append(structure.coord[mask].mean(axis=0))

    residue_index = np.array(residue_index, dtype=int)
    author_resnum = np.array(author_resnum, dtype=int)
    position = np.array(position, dtype=float)
    n = len(residue_index)

    # ---- Step 1: Delaunay triangulation of ionizable representative points, trim to <=10 A ----
    edges = []
    if n >= 4:
        try:
            tri = Delaunay(position)
        except Exception:
            # Degenerate/near-coplanar point sets (rare for a real protein,
            # but possible for a small or unusually flat ionizable-residue
            # set) make qhull's default options fail; joggling the input
            # perturbs points enough to force a full-dimensional hull.
            tri = Delaunay(position, qhull_options="QJ")
        edge_set = set()
        for simplex in tri.simplices:
            for a in range(4):
                for b in range(a + 1, 4):
                    i, j = simplex[a], simplex[b]
                    edge_set.add((min(i, j), max(i, j)))
        for i, j in edge_set:
            d = np.linalg.norm(position[i] - position[j])
            if d <= edge_cutoff:
                edges.append((int(i), int(j)))

    # ---- Step 2 (approximation): burial via local heavy-atom packing density ----
    tree = cKDTree(structure.coord)
    burial_score = np.array([len(tree.query_ball_point(p, r=burial_radius)) for p in position])
    if n:
        lo = np.percentile(burial_score, exposed_percentile)
        hi = np.percentile(burial_score, buried_percentile)
        burial_class = [
            "buried" if s >= hi else ("exposed" if s <= lo else "margin") for s in burial_score
        ]
    else:
        burial_class = []

    networks = _connected_components(n, edges)

    return IonizableNetworkResult(
        residue_index=residue_index,
        author_resnum=author_resnum,
        resname=resname,
        position=position,
        burial_score=burial_score,
        burial_class=burial_class,
        edges=edges,
        networks=networks,
    )


def map_networks_to_blocks(result: IonizableNetworkResult, block_model) -> list:
    """For each network, return the set of WSME block indices its member
    residues fall into -- lets you cross-reference a candidate pH-sensor
    cluster with e.g. the coupling-free-energy matrix for those blocks."""
    out = []
    for members in result.networks:
        blocks = sorted({int(block_model.block_of_residue[result.residue_index[m]]) for m in members})
        out.append(blocks)
    return out
