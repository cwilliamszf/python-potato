"""End-to-end CLI: subsample an MSA for diversity, fold every variant with ColabFold,
filter junk, structurally cluster the survivors, and classify each representative as
active / inactive / intermediate.

Typical usage (after generating `receptor.a3m` with `colabfold_batch --msa-only` or the
ColabFold MMseqs2 server):

    python -m gpcr_ensemble.pipeline \\
        --a3m receptor.a3m --out results/ \\
        --tm3-resnum 131 --tm6-resnum 272 \\
        --inactive-ref inactive_template.pdb --active-ref active_template.pdb

Residue numbers for the TM3 (DRY-motif R3.50) and TM6 (~6.30-6.34) reference residues
should be looked up for your receptor via GPCRdb (https://gpcrdb.org) generic numbering,
or by aligning to a homolog of known structure.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

from . import activation_state as astate
from . import cluster as clust
from . import msa_subsample as sub
from . import run_colabfold as rc


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--a3m", required=True, help="Full MSA (.a3m) for the target GPCR")
    p.add_argument("--out", required=True, help="Output directory")
    p.add_argument("--tm3-resnum", type=int, required=True)
    p.add_argument("--tm6-resnum", type=int, required=True)
    p.add_argument("--inactive-ref", help="PDB of a known inactive-state reference structure")
    p.add_argument("--active-ref", help="PDB of a known active-state reference structure")
    p.add_argument("--ref-chain", default="A")
    p.add_argument("--num-seeds", type=int, default=8)
    p.add_argument("--num-recycle", type=int, default=3)
    p.add_argument("--plddt-cutoff", type=float, default=70.0)
    p.add_argument("--rmsd-cluster-cutoff", type=float, default=2.0, help="Angstrom")
    p.add_argument("--core-residue-range", nargs=2, type=int, default=None, metavar=("LO", "HI"))
    p.add_argument("--skip-fold", action="store_true", help="Assume ColabFold outputs already exist in <out>/models")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    out = Path(args.out)
    msa_dir = out / "msas"
    model_dir = out / "models"
    out.mkdir(parents=True, exist_ok=True)

    print(f"[1/5] Generating diverse subsampled/clustered MSAs from {args.a3m}")
    manifest = sub.generate_diverse_msas(args.a3m, msa_dir)
    print(f"      wrote {len(manifest)} MSA variants to {msa_dir}")

    if not args.skip_fold:
        print("[2/5] Running colabfold_batch over all MSA variants")
        rc.run(
            msa_dir,
            model_dir,
            num_seeds=args.num_seeds,
            num_recycle=args.num_recycle,
        )
    else:
        print("[2/5] --skip-fold set, expecting existing models in", model_dir)

    print("[3/5] Collecting models and filtering for stable folds")
    model_manifest = rc.collect_manifest(model_dir)
    kept = []
    for entry in model_manifest:
        if entry["pdb_path"] is None:
            continue
        coords = clust.load_ca_coords(entry["pdb_path"], chain="A")
        if astate.passes_fold_quality(
            coords, mean_plddt_value=entry["mean_plddt"], plddt_cutoff=args.plddt_cutoff
        ):
            entry["ca_coords"] = coords
            kept.append(entry)
    print(f"      kept {len(kept)}/{len(model_manifest)} models after pLDDT/geometry filtering")

    print("[4/5] Structurally clustering the ensemble")
    coord_sets = clust.common_core_coords(
        [e["ca_coords"] for e in kept],
        residue_range=tuple(args.core_residue_range) if args.core_residue_range else None,
    )
    rmsd_matrix = clust.pairwise_rmsd_matrix(coord_sets)
    labels = clust.cluster_by_rmsd(rmsd_matrix, args.rmsd_cluster_cutoff)
    reps = clust.select_representatives(labels, [e["mean_plddt"] or 0.0 for e in kept])
    print(f"      {len(reps)} structurally distinct clusters from {len(kept)} kept models")

    print("[5/5] Classifying activation state of each representative")
    thresholds = astate.DEFAULT_THRESHOLDS
    if args.inactive_ref and args.active_ref:
        inactive_coords = clust.load_ca_coords(args.inactive_ref, chain=args.ref_chain)
        active_coords = clust.load_ca_coords(args.active_ref, chain=args.ref_chain)
        thresholds = astate.calibrate_thresholds(
            inactive_coords, active_coords, args.tm3_resnum, args.tm6_resnum
        )
        print(f"      calibrated thresholds: inactive<={thresholds.inactive_max:.2f} A, "
              f"active>={thresholds.active_min:.2f} A")
    else:
        print("      no reference structures given; using generic class-A default thresholds "
              "(recommend calibrating with --inactive-ref/--active-ref)")

    rows = []
    for lab, idx in reps.items():
        entry = kept[idx]
        label, distance = astate.classify_model(
            entry["ca_coords"], args.tm3_resnum, args.tm6_resnum, thresholds
        )
        rows.append(
            {
                "cluster": lab,
                "tag": entry["tag"],
                "pdb_path": entry["pdb_path"],
                "mean_plddt": entry["mean_plddt"],
                "tm3_tm6_distance": round(distance, 2),
                "state": label,
            }
        )

    report_path = out / "ensemble_report.csv"
    with open(report_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()) if rows else
                                 ["cluster", "tag", "pdb_path", "mean_plddt", "tm3_tm6_distance", "state"])
        writer.writeheader()
        writer.writerows(rows)

    counts = {}
    for r in rows:
        counts[r["state"]] = counts.get(r["state"], 0) + 1
    print(f"\nWrote {report_path}")
    print(f"Ensemble state distribution: {counts}")
    return rows


if __name__ == "__main__":
    main()
