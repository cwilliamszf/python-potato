"""Streamlit GUI for wsme-gpcr, exposing every option available on the CLI.

Run with:
    streamlit run wsme_gpcr/app.py
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from wsme_gpcr.alanine_scan import estimate_scan_seconds, scannable_positions
from wsme_gpcr.pipeline import DEFAULT_PH_VALUES, run_alanine_scan_pipeline, run_pipeline, run_pipeline_multi_ph
from wsme_gpcr.plotting import (
    plot_1d_profile,
    plot_1d_profile_comparison,
    plot_2d_landscape,
    plot_2d_landscape_surface,
    plot_2d_landscape_surface_comparison,
    plot_comparison_grid,
    plot_coupling_matrix,
    plot_ddg_structure_map,
    plot_ddg_vs_distance,
    plot_dsc,
    plot_mutational_response,
    plot_residue_folding_probability,
)
from wsme_gpcr.wsme import WSMEParams

st.set_page_config(page_title="wsme-gpcr", layout="wide")
st.title("wsme-gpcr")
st.caption(
    "Python port of the blocked WSME (bWSME) conformational free-energy landscape model "
    "(AthiNaganathan/WSMEmodel + AthiNaganathan/GPCR-Landscapes)."
)

# ---------------------------------------------------------------- Sidebar --

with st.sidebar:
    st.header("Structure")
    pdb_file = st.file_uploader("PDB or mmCIF file", type=["pdb", "ent", "cif", "mmcif"])
    chain = st.text_input("Chain ID", value="", help="Leave blank to auto-select the first chain with standard residues")
    model_index = st.number_input("Model index", min_value=0, value=0, step=1, help="For multi-model files (e.g. NMR ensembles)")

    all_ph = st.checkbox(
        "Run for all pH values (7, 5, 3.5, 2)", value=False,
        help="Runs the full pipeline independently at each pH (pH changes which atoms are charged, "
        "so the contact map and electrostatics -- not just screening -- differ per pH). Roughly 4x slower.",
    )
    ph = st.selectbox("pH (charge assignment)", options=[7.0, 5.0, 3.5, 2.0], index=0, disabled=all_ph)
    show_comparison_grid = st.checkbox(
        "Show combined comparison grid (3D landscape + folding probability + coupling, one column per pH)",
        value=False, disabled=not all_ph,
        help="One big figure: a column per pH, rows for 3D landscape / residue folding probability / "
        "coupling (coupling row only if 'Compute residue-residue coupling free energy' is also checked). "
        "Requires 'Run for all pH values'.",
    )

    st.header("Secondary structure")
    ss_source = st.radio(
        "Source",
        options=["Auto (geometric, no STRIDE needed)", "Paste SS codes", "Upload SS codes file"],
        help="Auto uses a phi/psi Ramachandran classification. For exact fidelity to the original "
        "MATLAB tool, supply real STRIDE/DSSP per-residue codes (H/E/G/other) instead.",
    )
    ss_codes_text = None
    ss_codes_file = None
    if ss_source == "Paste SS codes":
        ss_codes_text = st.text_area("Per-residue SS codes (one char per residue, e.g. from STRIDE/DSSP)")
    elif ss_source == "Upload SS codes file":
        ss_codes_file = st.file_uploader("SS codes file (plain text)", type=["txt"], key="ss_file")

    st.header("Blocking")
    block_size = st.number_input("Block size (residues/block)", min_value=1, max_value=20, value=4, step=1)

    st.header("Model parameters")
    preset = st.selectbox(
        "Preset",
        options=["membrane (GPCR, dielectric=4)", "soluble protein (dielectric=29)"],
        index=0,
    )
    preset_key = "membrane" if preset.startswith("membrane") else "soluble"
    base_params = WSMEParams.soluble_protein_defaults() if preset_key == "soluble" else WSMEParams()

    with st.expander("Override individual parameters", expanded=False):
        temp = st.number_input("Temperature T (K)", value=float(base_params.T))
        ene = st.number_input("vdW energy per native contact, ene (kJ/mol)", value=float(base_params.ene), format="%.5f")
        ds = st.number_input("Entropic cost per residue, DS (kJ/mol/K)", value=float(base_params.DS), format="%.5f")
        dcp = st.number_input("Heat capacity change per contact, DCp (kJ/mol/K)", value=float(base_params.DCp), format="%.6f")
        ionic_strength = st.number_input("Ionic strength, IS (M)", value=float(base_params.IS), format="%.3f")
        dielectric = st.number_input("Medium dielectric constant", value=float(base_params.dielectric))

    st.header("DSC thermogram")
    run_dsc = st.checkbox("Compute DSC thermogram", value=False, help="Sweeps temperature; slower than the landscape alone (and ~4x slower again with all-pH)")
    dsc_tmin = st.number_input("DSC T min (K)", value=273.0, disabled=not run_dsc)
    dsc_tmax = st.number_input("DSC T max (K)", value=373.0, disabled=not run_dsc)
    dsc_tstep = st.number_input("DSC T step (K)", value=1.0, min_value=0.1, disabled=not run_dsc)

    st.header("Coupling analysis")
    run_coupling = st.checkbox(
        "Compute residue-residue coupling free energy", value=False,
        help="Thermodynamic coupling between every pair of blocks (do j and k tend to fold together?). "
        "Comparable cost to the landscape itself -- previously impractical at GPCR scale in the MATLAB tool.",
    )

    st.header("Alanine-scanning mutagenesis")
    run_ala_scan = st.checkbox(
        "Run in silico alanine scan", value=False,
        help="Mutates each scanned residue to alanine one at a time and compares its coupling "
        "free energy (chi_plus) to wild type -- Fig. 7 of Anantakrishnan & Naganathan, Nat Commun "
        "14, 128 (2023). Generalizable to any residue/structure, not tied to one receptor. "
        "Runs at the single 'pH (charge assignment)' value above, even if 'Run for all pH values' is checked "
        "(a mutational scan isn't repeated per pH).",
    )
    ala_scope = st.radio(
        "Positions to scan",
        options=["Evenly-spaced subsample (fast)", "Every eligible residue (slow, full receptor)", "Specific residues"],
        disabled=not run_ala_scan,
    )
    ala_max_n = st.number_input(
        "Max positions (subsample)", min_value=1, value=40, step=1,
        disabled=not (run_ala_scan and ala_scope == "Evenly-spaced subsample (fast)"),
        help="Evenly subsampled across the sequence so a capped scan still covers the whole structure.",
    )
    ala_positions_text = st.text_input(
        "Residue numbers (comma-separated, author numbering)", value="",
        disabled=not (run_ala_scan and ala_scope == "Specific residues"),
    )
    ala_top_n = st.number_input("Top hits to report", min_value=1, value=5, step=1, disabled=not run_ala_scan)
    if run_ala_scan:
        st.caption("~8s/position once the structure is loaded (varies with structure size) -- "
                   "a full receptor-wide scan (~250-300 residues) is a tens-of-minutes job.")

    run_button = st.button("Run", type="primary", use_container_width=True)

# ------------------------------------------------------------------ Run ---

if run_button:
    if pdb_file is None:
        st.error("Upload a PDB or mmCIF file first.")
        st.stop()

    suffix = Path(pdb_file.name).suffix or ".pdb"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(pdb_file.getvalue())
        tmp_path = tmp.name

    if ss_source == "Paste SS codes" and ss_codes_text:
        ss_codes = ss_codes_text.strip()
    elif ss_source == "Upload SS codes file" and ss_codes_file is not None:
        ss_codes = ss_codes_file.getvalue().decode().strip()
    else:
        ss_codes = None

    params = WSMEParams(
        T=temp, ene=ene, DS=ds, DCp=dcp, IS=ionic_strength, dielectric=dielectric,
        DDS=base_params.DDS, Tref=base_params.Tref,
    )
    dsc_T_grid = np.arange(dsc_tmin, dsc_tmax + dsc_tstep / 2, dsc_tstep) if run_dsc else None

    try:
        if all_ph:
            progress_bar = st.progress(0.0, text="Starting...")

            def progress(ph_val, i, total):
                progress_bar.progress((i) / total, text=f"Running pH {ph_val} ({i + 1}/{total})...")

            pipeline_results = run_pipeline_multi_ph(
                tmp_path, ph_values=DEFAULT_PH_VALUES, chain=chain or None, model=int(model_index),
                ss_codes=ss_codes, block_size=int(block_size), params=params,
                with_dsc=run_dsc, dsc_T_grid=dsc_T_grid, with_coupling=run_coupling, progress_callback=progress,
            )
            progress_bar.progress(1.0, text="Done")
        else:
            with st.spinner("Running full pipeline..."):
                pipeline_results = {ph: run_pipeline(
                    tmp_path, chain=chain or None, model=int(model_index), ph=ph,
                    ss_codes=ss_codes, block_size=int(block_size), params=params,
                    with_dsc=run_dsc, dsc_T_grid=dsc_T_grid, with_coupling=run_coupling,
                )}
    except Exception as e:
        st.error(f"Run failed: {e}")
        st.stop()

    for w in next(iter(pipeline_results.values())).warnings:
        st.warning(w)

    ala_scan_pr = None
    if run_ala_scan:
        base_structure = pipeline_results[ph].structure
        if ala_scope == "Specific residues":
            ala_positions = [int(v) for v in ala_positions_text.split(",") if v.strip()] or None
            ala_max_positions = None
        elif ala_scope == "Every eligible residue (slow, full receptor)":
            ala_positions = None
            ala_max_positions = None
        else:
            ala_positions = None
            ala_max_positions = int(ala_max_n)

        n_estimate = len(ala_positions) if ala_positions else (
            ala_max_positions if ala_max_positions else len(scannable_positions(base_structure))
        )
        est_min = estimate_scan_seconds(n_estimate) / 60
        ala_progress = st.progress(0.0, text=f"Alanine scan: {n_estimate} position(s), estimated ~{est_min:.1f} min...")

        def ala_progress_cb(resnum, i, total, elapsed):
            ala_progress.progress((i + 1) / total, text=f"Alanine scan: resnum {resnum} ({i + 1}/{total}, {elapsed:.0f}s elapsed)")

        try:
            ala_scan_pr = run_alanine_scan_pipeline(
                tmp_path, chain=chain or None, model=int(model_index), ph=ph,
                ss_codes=ss_codes, block_size=int(block_size), params=params,
                positions=ala_positions, max_positions=ala_max_positions,
                progress_callback=ala_progress_cb,
            )
            ala_progress.progress(1.0, text="Alanine scan done")
        except Exception as e:
            st.error(f"Alanine scan failed: {e}")
            ala_scan_pr = None

    if len(pipeline_results) > 1:
        st.subheader("Comparison across pH")
        fig = plot_1d_profile_comparison({ph_val: pr.result for ph_val, pr in pipeline_results.items()}).figure
        st.pyplot(fig)

        fig = plot_2d_landscape_surface_comparison({f"pH {ph_val}": pr.result for ph_val, pr in pipeline_results.items()})
        st.pyplot(fig)

        if show_comparison_grid:
            results_by_key = {f"pH {ph_val}": pr.result for ph_val, pr in pipeline_results.items()}
            coupling_by_key = None
            if all(pr.coupling_result is not None for pr in pipeline_results.values()):
                coupling_by_key = {f"pH {ph_val}": pr.coupling_result for ph_val, pr in pipeline_results.items()}
            with st.spinner("Building comparison grid..."):
                fig = plot_comparison_grid(results_by_key, coupling_by_key=coupling_by_key)
            st.pyplot(fig)

        import pandas as pd

        summary_rows = []
        for ph_val, pr in pipeline_results.items():
            r = pr.result
            summary_rows.append({
                "pH": ph_val,
                "residues": pr.structure.nres,
                "structured residues": int(pr.ss_mask.sum()),
                "VdW contacts": int(pr.contact_map.srcont.sum()),
                "electrostatic pairs": len(pr.contact_map.elec_pairs),
                "blocks": pr.block_model.nblocks,
                "Zfin": r.zfin,
                "argmin n (most stable RC)": int(r.n_values[r.fes.argmin()]),
                "% SSA": round(r.stats["pct_ssa"], 1),
                "% DSA": round(r.stats["pct_dsa"], 1),
                "% DSAw/L": round(r.stats["pct_dsawl"], 1),
            })
        st.dataframe(pd.DataFrame(summary_rows).set_index("pH"), use_container_width=True)

    ph_tabs = st.tabs([f"pH {ph_val}" for ph_val in pipeline_results])
    for ph_val, tab in zip(pipeline_results, ph_tabs):
        pr = pipeline_results[ph_val]
        result = pr.result
        with tab:
            col1, col2, col3 = st.columns(3)
            col1.metric("Zfin", f"{result.zfin:.3e}")
            col2.metric("SSA / DSA / DSAw-L states", f"{result.stats['n_states_ssa']} / {result.stats['n_states_dsa']} / {result.stats['n_states_dsawl']}")
            col3.metric("Partition fn % (SSA/DSA/DSAw-L)", f"{result.stats['pct_ssa']:.1f} / {result.stats['pct_dsa']:.1f} / {result.stats['pct_dsawl']:.1f}")

            tab_names = ["1D Profile", "2D Landscape", "3D Landscape", "Residue Folding Probability"]
            if pr.dsc_result:
                tab_names.append("DSC Thermogram")
            if pr.coupling_result:
                tab_names.append("Coupling Free Energy")
            inner_tabs = st.tabs(tab_names)

            with inner_tabs[0]:
                fig = plot_1d_profile(result).figure
                st.pyplot(fig)
                st.download_button(
                    "Download 1D_FreeEnergyProfile.txt",
                    "\n".join(f"{n} {fe:.3f}" for n, fe in zip(result.n_values, result.fes)),
                    file_name=f"1D_FreeEnergyProfile_pH{ph_val}.txt",
                    key=f"1d_{ph_val}",
                )

            with inner_tabs[1]:
                fig = plot_2d_landscape(result).figure
                st.pyplot(fig)
                lines = [f"{i} {j} {result.fes2D[i, j]:.3f}" for i in range(result.fes2D.shape[0]) for j in range(result.fes2D.shape[1])]
                st.download_button("Download 2D_FreeEnergySurface.txt", "\n".join(lines), file_name=f"2D_FreeEnergySurface_pH{ph_val}.txt", key=f"2d_{ph_val}")

            with inner_tabs[2]:
                fig = plot_2d_landscape_surface(result).figure
                st.pyplot(fig)

            with inner_tabs[3]:
                fig = plot_residue_folding_probability(result).figure
                st.pyplot(fig)

            next_tab = 4
            if pr.dsc_result:
                with inner_tabs[next_tab]:
                    fig = plot_dsc(pr.dsc_result).figure
                    st.pyplot(fig)
                    lines = [f"{T:.1f} {cp:.5f} {cpx:.5f}" for T, cp, cpx in zip(pr.dsc_result.T, pr.dsc_result.Cp, pr.dsc_result.Cp_excess)]
                    st.download_button("Download DSC_Thermogram.txt", "\n".join(lines), file_name=f"DSC_Thermogram_pH{ph_val}.txt", key=f"dsc_{ph_val}")
                next_tab += 1

            if pr.coupling_result:
                with inner_tabs[next_tab]:
                    fig = plot_coupling_matrix(pr.coupling_result).figure
                    st.pyplot(fig)
                    mat = pr.coupling_result.coupling_free_energy
                    lines = [f"{j} {k} {mat[j, k]:.3f} {pr.coupling_result.p_folded_folded[j, k]:.4f}"
                             for j in range(mat.shape[0]) for k in range(mat.shape[1])]
                    st.download_button("Download CouplingMatrix.txt", "\n".join(lines), file_name=f"CouplingMatrix_pH{ph_val}.txt", key=f"coupling_{ph_val}")
                next_tab += 1

    if ala_scan_pr is not None:
        scan = ala_scan_pr.scan
        st.subheader(f"Alanine scan results (pH {ala_scan_pr.ph}, {len(scan.positions)} position(s) scanned)")

        fig = plot_mutational_response(scan, highlight={r: str(r) for r, _ in scan.top_hits(int(ala_top_n))}).figure
        st.pyplot(fig)
        st.download_button(
            "Download MutationalResponse.txt",
            "\n".join(f"{b} {m:.3f} {s:.3f}" for b, (m, s) in enumerate(zip(scan.MR_mean, scan.MR_std))),
            file_name=f"MutationalResponse_pH{ala_scan_pr.ph}.txt",
        )

        import pandas as pd

        top_hits = scan.top_hits(int(ala_top_n))
        st.write("**Top hits** (largest total coupling perturbation, sum |ΔΔG+|):")
        st.dataframe(
            pd.DataFrame(top_hits, columns=["resnum", "sum |ΔΔG+| (kJ/mol)"]).set_index("resnum"),
            use_container_width=True,
        )

        records = scan.to_records()
        csv_lines = ["mutated_resnum,block,mean_ddG_plus"] + [
            f"{r['mutated_resnum']},{r['block']},{r['mean_ddG+']:.4f}" for r in records
        ]
        st.download_button("Download DeltaDeltaG.csv", "\n".join(csv_lines), file_name=f"DeltaDeltaG_pH{ala_scan_pr.ph}.csv")

        hit_tabs = st.tabs([f"resnum {r}" for r, _ in top_hits])
        for (resnum, score), hit_tab in zip(top_hits, hit_tabs):
            with hit_tab:
                col1, col2 = st.columns(2)
                with col1:
                    fig = plot_ddg_vs_distance(scan, resnum).figure
                    st.pyplot(fig)
                with col2:
                    fig = plt.figure(figsize=(6, 6))
                    ax = fig.add_subplot(projection="3d")
                    plot_ddg_structure_map(scan, resnum, ax=ax)
                    st.pyplot(fig)
else:
    st.write("Upload a structure and click **Run** in the sidebar to compute a landscape.")
    st.write(
        "Ship the bundled example first: `examples/data/CI2.pdb` "
        "(pH 7, block size 4, preset = soluble protein). "
        "Check **Run for all pH values** to get the analysis at pH 7, 5, 3.5, and 2 in one click."
    )
