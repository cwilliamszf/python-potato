"""Minimal end-to-end example: compute and plot a bWSME landscape for the
bundled CI2 structure (the same example used in AthiNaganathan/WSMEmodel).

    python examples/run_ci2.py
"""

from pathlib import Path

from wsme_gpcr import (
    WSMEParams,
    assign_secondary_structure,
    build_blocks,
    compute_contact_map,
    load_structure,
    run_wsme,
)
from wsme_gpcr.plotting import plot_summary

HERE = Path(__file__).parent


def main():
    structure = load_structure(HERE / "data" / "CI2.pdb")
    ss_mask = assign_secondary_structure(structure)
    contacts = compute_contact_map(structure)
    blocks = build_blocks(ss_mask, contacts, block_size=4)

    params = WSMEParams.soluble_protein_defaults()
    result = run_wsme(structure, blocks, ss_mask, params)

    print(f"{structure.nres} residues -> {blocks.nblocks} blocks")
    print(f"Zfin = {result.zfin:.4e}")
    print(f"Free-energy minimum at n={result.n_values[result.fes.argmin()]} blocks")
    print(f"State counts: {result.stats}")

    out_path = HERE / "ci2_landscape.png"
    plot_summary(result, save_path=str(out_path))
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
