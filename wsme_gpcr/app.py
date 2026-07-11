"""Streamlit GUI for wsme-gpcr, exposing every option available on the CLI.

Run with:
    streamlit run wsme_gpcr/app.py
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import streamlit as st

from wsme_gpcr.blocking import build_blocks
from wsme_gpcr.contacts import compute_contact_map
from wsme_gpcr.dsc import compute_dsc
from wsme_gpcr.plotting import plot_1d_profile, plot_2d_landscape, plot_dsc, plot_residue_folding_probability
from wsme_gpcr.secondary_structure import assign_secondary_structure, secondary_structure_from_codes
from wsme_gpcr.structure import load_structure
from wsme_gpcr.wsme import WSMEParams, run_wsme

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
    ph = st.selectbox("pH (charge assignment)", options=[7.0, 5.0, 3.5, 2.0], index=0)

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
    run_dsc = st.checkbox("Compute DSC thermogram", value=False, help="Sweeps temperature; slower than the landscape alone")
    dsc_tmin = st.number_input("DSC T min (K)", value=273.0, disabled=not run_dsc)
    dsc_tmax = st.number_input("DSC T max (K)", value=373.0, disabled=not run_dsc)
    dsc_tstep = st.number_input("DSC T step (K)", value=1.0, min_value=0.1, disabled=not run_dsc)

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

    with st.spinner("Loading structure..."):
        import warnings

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            structure = load_structure(tmp_path, chain=chain or None, model=int(model_index), ph=ph)
            for w in caught:
                st.warning(str(w.message))
    st.success(f"Loaded {structure.nres} residues, chain {structure.chain_id}")

    with st.spinner("Assigning secondary structure..."):
        if ss_source == "Paste SS codes" and ss_codes_text:
            codes = ss_codes_text.strip()
        elif ss_source == "Upload SS codes file" and ss_codes_file is not None:
            codes = ss_codes_file.getvalue().decode().strip()
        else:
            codes = None

        if codes is not None:
            if len(codes) != structure.nres:
                st.error(f"SS code string length ({len(codes)}) != number of residues ({structure.nres})")
                st.stop()
            ss_mask = secondary_structure_from_codes(codes)
        else:
            ss_mask = assign_secondary_structure(structure)
    st.info(f"{int(ss_mask.sum())}/{structure.nres} residues structured (helix/strand/3-10)")

    with st.spinner("Computing contact map..."):
        contact_map = compute_contact_map(structure)
    st.info(f"{int(contact_map.srcont.sum())} VdW contacts, {len(contact_map.elec_pairs)} electrostatic pairs")

    with st.spinner("Building blocks..."):
        block_model = build_blocks(ss_mask, contact_map, block_size=int(block_size))
    st.info(f"{block_model.nblocks} blocks")

    params = WSMEParams(
        T=temp, ene=ene, DS=ds, DCp=dcp, IS=ionic_strength, dielectric=dielectric,
        DDS=base_params.DDS, Tref=base_params.Tref,
    )

    with st.spinner("Running WSME (SSA/DSA/DSAw-L enumeration)..."):
        result = run_wsme(structure, block_model, ss_mask, params)

    col1, col2, col3 = st.columns(3)
    col1.metric("Zfin", f"{result.zfin:.3e}")
    col2.metric("SSA / DSA / DSAw-L states", f"{result.stats['n_states_ssa']} / {result.stats['n_states_dsa']} / {result.stats['n_states_dsawl']}")
    col3.metric("Partition fn % (SSA/DSA/DSAw-L)", f"{result.stats['pct_ssa']:.1f} / {result.stats['pct_dsa']:.1f} / {result.stats['pct_dsawl']:.1f}")

    dsc_result = None
    if run_dsc:
        with st.spinner(f"Running DSC sweep ({dsc_tmin}-{dsc_tmax} K, step {dsc_tstep})..."):
            T_grid = np.arange(dsc_tmin, dsc_tmax + dsc_tstep / 2, dsc_tstep)
            dsc_result = compute_dsc(structure, block_model, ss_mask, params, T_grid=T_grid)

    tabs = st.tabs(["1D Profile", "2D Landscape", "Residue Folding Probability"] + (["DSC Thermogram"] if dsc_result else []))

    with tabs[0]:
        fig = plot_1d_profile(result).figure
        st.pyplot(fig)
        st.download_button(
            "Download 1D_FreeEnergyProfile.txt",
            "\n".join(f"{n} {fe:.3f}" for n, fe in zip(result.n_values, result.fes)),
            file_name="1D_FreeEnergyProfile.txt",
        )

    with tabs[1]:
        fig = plot_2d_landscape(result).figure
        st.pyplot(fig)
        lines = [f"{i} {j} {result.fes2D[i, j]:.3f}" for i in range(result.fes2D.shape[0]) for j in range(result.fes2D.shape[1])]
        st.download_button("Download 2D_FreeEnergySurface.txt", "\n".join(lines), file_name="2D_FreeEnergySurface.txt")

    with tabs[2]:
        fig = plot_residue_folding_probability(result).figure
        st.pyplot(fig)

    if dsc_result:
        with tabs[3]:
            fig = plot_dsc(dsc_result).figure
            st.pyplot(fig)
            lines = [f"{T:.1f} {cp:.5f} {cpx:.5f}" for T, cp, cpx in zip(dsc_result.T, dsc_result.Cp, dsc_result.Cp_excess)]
            st.download_button("Download DSC_Thermogram.txt", "\n".join(lines), file_name="DSC_Thermogram.txt")
else:
    st.write("Upload a structure and click **Run** in the sidebar to compute a landscape.")
    st.write(
        "Ship the bundled example first: `examples/data/CI2.pdb` "
        "(pH 7, block size 4, preset = soluble protein)."
    )
