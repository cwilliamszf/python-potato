"""Thin wrapper around the `colabfold_batch` CLI to fold every subsampled/clustered MSA
produced by `msa_subsample.generate_diverse_msas`, with settings tuned to maximize
conformational diversity of the output ensemble rather than to get one "best" model:

- `--num-seeds`: multiple random seeds per MSA (structure module init + dropout draws differ).
- `--use-dropout`: keep training-time dropout active at inference (Del Alamo et al. 2022) --
  this alone, even on the *full* MSA, materially increases ensemble diversity.
- low `--num-recycle`: fewer recycles means the prediction has less chance to converge onto
  the single deepest energy minimum; combined with the above this trades some per-model
  confidence for ensemble breadth, which is the point here.
- `--model-type alphafold2_ptm` (5 monomer models) x multiple seeds x multiple MSAs gives a
  large, cheaply parallelizable grid.

This module does not require ColabFold/JAX to import; it only needs them at `run()` time,
so the rest of the package (subsampling, clustering, classification) stays independently
testable in this sandbox where colabfold/GPU are not available.
"""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import numpy as np


def build_command(
    msa_dir: str | Path,
    out_dir: str | Path,
    num_seeds: int = 8,
    num_recycle: int = 3,
    model_type: str = "alphafold2_ptm",
    num_models: int = 5,
    use_dropout: bool = True,
    custom_template_path: str | Path | None = None,
    extra_args: list[str] | None = None,
) -> list[str]:
    cmd = [
        "colabfold_batch",
        str(msa_dir),
        str(out_dir),
        "--num-seeds",
        str(num_seeds),
        "--num-recycle",
        str(num_recycle),
        "--model-type",
        model_type,
        "--num-models",
        str(num_models),
    ]
    if use_dropout:
        cmd.append("--use-dropout")
    if custom_template_path is not None:
        cmd += ["--templates", "--custom-template-path", str(custom_template_path)]
    if extra_args:
        cmd += extra_args
    return cmd


def run(
    msa_dir: str | Path,
    out_dir: str | Path,
    **kwargs,
) -> subprocess.CompletedProcess:
    """Invoke colabfold_batch over the whole directory of subsampled MSAs produced by
    `msa_subsample.generate_diverse_msas`. Requires colabfold_batch to be installed
    (e.g. via LocalColabFold, https://github.com/YoshitakaMo/localcolabfold) and a GPU."""
    if shutil.which("colabfold_batch") is None:
        raise RuntimeError(
            "colabfold_batch not found on PATH. Install ColabFold "
            "(https://github.com/YoshitakaMo/localcolabfold or `pip install colabfold[alphafold]`) "
            "and run on a machine with a GPU before calling run()."
        )
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    cmd = build_command(msa_dir, out_dir, **kwargs)
    return subprocess.run(cmd, check=True)


def collect_manifest(out_dir: str | Path) -> list[dict]:
    """After a colabfold_batch run, gather every predicted model's PDB path, rank, and
    mean pLDDT/PTM from the `*_scores_*.json` files ColabFold writes alongside the PDBs."""
    out_dir = Path(out_dir)
    manifest = []
    for score_path in sorted(out_dir.glob("*_scores_rank_*.json")):
        with open(score_path) as fh:
            scores = json.load(fh)
        stem = score_path.name.replace("_scores_", "_unrelaxed_").replace(".json", ".pdb")
        pdb_path = out_dir / stem
        if not pdb_path.exists():
            candidates = list(out_dir.glob(score_path.stem.replace("_scores_", "*") + "*.pdb"))
            pdb_path = candidates[0] if candidates else None
        manifest.append(
            {
                "tag": score_path.stem,
                "pdb_path": str(pdb_path) if pdb_path else None,
                "mean_plddt": float(np.mean(scores["plddt"])) if "plddt" in scores else None,
                "ptm": scores.get("ptm"),
            }
        )
    return manifest
