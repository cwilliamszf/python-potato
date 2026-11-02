# MCF7 pHluorin2-GPI spheroid drug screen — analysis pipeline

This folder holds the scripts that reproduce the analyses of experiment **261001_105300** (MCF7 P6 spheroids expressing pHluorin2-GPI, 31 conditions × 3 spheroids, imaged on a BioTek Cytation C10 at 24, 48 and 72 h). It starts from the raw Cytation image folders and ends with every table and figure from the project.

**Readout convention used everywhere:** **488/405 ratio, higher = more acidic.** The ratio is the 488-ex channel (Cytation label "GFP 469,525") divided by the 405-ex channel (Cytation label "Oxidized roGFP2 400,525"; the roGFP2 name is only the filter-set label). Fold-changes are relative to the DMSO wells on the same plate.

**Design caveat:** this is **one biological experiment**. The three spheroids per condition are technical replicates, and the same spheroids are re-imaged at each timepoint. Statistics treat the spheroid as the unit (well-clustered errors where timepoints are pooled). Independent biological replication is still outstanding.

---

## 1. Requirements

- Python 3.10 or later (tested with Python 3.11.16 on Windows 11)
- Packages: see `requirements.txt` (`pip install -r requirements.txt`). Tested versions: numpy 2.4.6, pandas 2.3.3, scipy 1.17.1, scikit-image 0.26.0, tifffile 2026.3.3, statsmodels 0.15.0, matplotlib 3.11.2.
- RAM: about 4 GB free is comfortable with the default 8 worker processes (each holds 0.2–0.4 GB). Lower `SPHEROID_WORKERS` if memory is short; the largest step holds about 1.5 GB in the main process.
- Disk: about 1 GB for `results/`.

## 2. Input data

`config.py` points at the Cytation export folder. Change `DATA_DIR` there, or set the environment variable `SPHEROID_DATA_DIR`. Expected layout:

```
<DATA_DIR>/
  261001_105300_Plate 1/      24 h   raw slices  A1_02_1_1Z0_GFP_001.tif ...  + #...ZProj/ subfolder
  261002_173508_Plate 3/      48 h
  261003_211651_Plate 1/      72 h
```

- **Raw slices** are named `<well>_<field>_<ch>_1Z<z>_<channel>_<n>.tif`. There are 28 z-slices at 10.1 µm steps, going **upward** from the autofocus plane. Channels: GFP / Oxidized roGFP2 / Bright Field.
- **Max projections** are the Cytation's own, in the `*ZProj*` subfolder: `<well>_-2_<ch>_1_ZProj[<channel>]_001.tif`.
- The 24 h plate has no ZProj image for 51 wells (D10–D12, rows E–H). The scripts build a max projection from the raw slices for those wells. The Cytation ZProj is a pixel-exact max projection, so the two are equivalent.

Outputs are written to `results/` next to the scripts (override with `SPHEROID_OUT_DIR`).

## 3. Running

```
python run_all.py              # main analysis: steps 3 4 5 6 7 1   (~1 h 45 min before parallelisation; see below)
python run_all.py 1b 2         # optional: session-1 QC figures and the superseded first spatial pass
python run_all.py 5 6          # re-run individual steps (later steps read the files earlier steps wrote)
```

Each step writes a log to `results/logs/step<N>.log`. You can also run any `stepN_*.py` directly.

| Step | Script | Runtime before* | What it does |
|---|---|---|---|
| 3 | `step3_extract_arrays.py` | 4 min | Segments every spheroid; stores binned 488/405 projections and masks |
| 4 | `step4_spatial_features.py` | 8 min | Background calibration, flat-fields, ~35 spatial features per spheroid, QC flags |
| 5 | `step5_statistics.py` | 3 min | Dose × time models, tracking, radial location, controls, matched-mean comparison, reproducibility |
| 6 | `step6_spatial_figures.py` | 0.5 min | Per-compound maps, dose × time grid, robustness figure |
| 7 | `step7_single_plane_validation.py` | 32 min | Single-plane (equatorial) validation against depth/optical artefacts; OGM 405 fluorescence |
| 1 | `step1_whole_spheroid.py` | 56 min | First analysis: whole-spheroid means, plate heatmaps, time course, dose–response, representative images |
| 1b | `step1b_session1_qc_figures.py` | 2 min | Segmentation QC montage; max-projection vs ratio-first projection pilot (6 wells) |
| 2 | `step2_legacy_spatial_v1.py` | 23 min | **Superseded** first spatial pass (raw ratio, no background calibration) |
| 17 | `step17_texture.py` | new: ~1 min on 4 cores | Bright-field texture (core darkness, granularity) and full-resolution surface puncta (needs step 4; `python run_all.py 17`, before 16) |
| 16 | `step16_primary_readouts.py` | new: < 1 min | The ten pre-specified primary readouts, sensitivity analyses, early-predicts-late, secondary readouts (needs steps 4 and 15; uses 17) |
| 15 | `step15_patch_dynamics.py` | new: ~4 min on 4 cores | Acid-patch dynamics across days: movement, growth relative to the spheroid, number / emergence / loss (needs step 4; `python run_all.py 15`) |

\*Original code, measured on the original machine (24 cores, data on a local disk). Current runtimes: next section.

Step 1 is independent of steps 3–7. Steps 4–7 need the output of step 3. Step 7 also needs `calibration.json` and `spatial_features_v2.csv` from step 4.

### Parallel processing and speed

Per-well work (segmentation, feature extraction, z-stack analysis, QC scan, permutation nulls) runs in worker processes. Set the number with the environment variable `SPHEROID_WORKERS` (default: CPU count − 1, at most 8). `SPHEROID_WORKERS=1` runs everything serially in one process, through the same code as before. Results are collected in plate/well order and every random draw is made in the main process, so the outputs do not depend on the number of workers. `qc_scan.py` uses the same setting.

Measured on a 4-core Linux machine with a synthetic dataset of the same size as experiment 1 (images in the file cache, so mostly CPU-bound). Seconds:

| Step | Original code | New code, `SPHEROID_WORKERS=1` | New code, 4 workers |
|---|---|---|---|
| 3 | 117 | – | 53 |
| 4 | 414 | 268 | 120 |
| 5 | 121 | 61 | 34 |
| 6 | 21 | – | 20 |
| 7 | 1523† | – | 318 |
| 1 | 2708† | 193 | 56 |
| **main run (3 4 5 6 7 1)** | **~4900 (82 min)** | – | **~600 (10 min)** |
| 1b | 100 | – | 78 |
| 2 (legacy) | 404 | – | 203 |
| T47D follow-up, step 3 / 4 / 5 (max projection) | 783† / 432 / 122 | – | 187 / 108 / 36 |
| `qc_scan.py`, T47D plates | 420 | – | 87 |

†The CPU was partly shared with other test runs during these original-code runs, so they are somewhat inflated. Profiled per well on an idle CPU, step 1 cost 9.6 s with the original code (6.7 s of it in one `disk(30)` dilation), and step 7 cost 4.4 s per spheroid (1.6 s reading the stack).

With 8 workers on the 24-core analysis PC, expect roughly 10 min or less for the main run. Steps 3 and 7 read every raw slice, so on that machine they will be limited by disk speed rather than core count.

What changed (`lib/parallel.py`, `lib/fastops.py`, `lib/whole_spheroid.py`; no change to any algorithm, parameter or output):
- **Exact fast morphology.** Step 1 spent ~7 s per well on one `binary_dilation` with `disk(30)` at 1960 × 1960 px. Dilations, closings and hole filling of single-object masks are now computed on the mask's bounding box (plus a margin wider than the footprint) with an exact Euclidean distance transform. This is ~300× faster for `disk(30)` and returns the same pixels. It also removes the dependence on scikit-image's `binary_*` functions, which are deprecated in 0.26 and removed in 0.28.
- **`np.nanmedian` along short axes** (flat-field binning, shared residual pattern) uses numpy's own algorithm without its masked-array overhead.
- **No repeated work.** The equatorial-plane search computes the edge ring once per spheroid instead of once per slice. Step 7 no longer reads the 28 bright-field slices of each stack, which it never used (one third of its I/O); a well whose bright-field files are missing or unreadable is still skipped, as before. Step 5's rotation search converts each patch map once per pair instead of 48 times. Radial binning selects bins from the masked-pixel vector instead of full images.
- **Verification.** `dev_tools/test_fastops.py` checks every fast function against the call it replaces (0 mismatches). The full pipeline (steps 1–7, 1b, 2, the T47D follow-up and `qc_scan.py`) was run with the original and the new code on a synthetic Cytation-format dataset (`dev_tools/make_synth.py`: 3 plates × 96 wells, 1960 × 1960 px, 28-slice stacks, 51 wells without ZProj). Every table, array, pickle and figure was identical (`dev_tools/compare_outputs.py`). It has not yet been re-run on the real images; running steps 3–5 once and comparing with the existing `results/` with `compare_outputs.py` is the recommended check.

## 4. What each step does (methods)

### Step 1 — whole-spheroid measurements (first analysis)

1. **Segmentation.** The bright-field projection is inverted, Gaussian-blurred (σ = 5 px) and Otsu-thresholded. Small objects (< 1000 px) are removed. The largest object is kept, preferring objects not touching the border, then holes are filled and the mask closed with disk(10). Masks outside 5,000–2,000,000 px are QC failures.
2. **Ratio.** Mean 488 and mean 405 intensity inside the mask, **without background subtraction** (`ratio_raw`). Subtracting an annulus background was rejected: the medium reads brighter than the spheroid in the 405 channel, which gave near-zero or negative denominators.
3. **Summaries and figures.**
   - Same-plate DMSO fold-change, then condition mean ± SEM (n = 3).
   - Plate heatmaps, condition × time heatmap, time course and bar charts.
   - 4-parameter logistic fits for OGM and Ogerin; the curve is drawn only where the EC50 falls inside the tested range. L/M/H compounds are plotted by dose rank.
   - Representative 48 h images: an RGB composite with a ratio overlay, and masked ratio maps on black.

Step 1 re-uses `per_well_measurements.csv` if it already exists in the output folder (only the figures are redrawn); set the environment variable `RECOMPUTE=1` to force re-measurement.

### Step 3 — extraction

Each spheroid is segmented as in step 1 on 2 × 2-binned bright field. The 488 and 405 max projections and the mask are then stored at 4 × 4 binning: 490 × 490 px, 1 px = 4 full-resolution px = 2.79 µm. Low-solidity masks (< 0.92) have attached debris trimmed by morphological opening.

### Step 4 — background models and spatial features

**Background.**
- **Medium contamination.** The medium itself reads 488/405 ≈ 2.0, which is *more acidic-looking* than the spheroids. The 405 signal of a spheroid is only 1.2–1.9× the medium. Any loss of spheroid signal therefore pushes the ratio toward "acidic".
- **Camera floor.** Estimated per plate from raw z-slices of 8 wells: the 0.5th percentile of each slice, minimum over slices, median over wells.
- **Flat-fields.** Built per plate from spheroid-free medium (median over wells, inpainted under the spheroids):
  - the raw ratio field;
  - per-channel medium fields;
  - the excitation-ratio field (floor-subtracted medium 488 / medium 405).
- **Four ratio models per pixel**, all divided by the matching field:

| Suffix | Model |
|---|---|
| (none) | raw ratio / ratio field |
| `_fs` | camera floor subtracted |
| `_ms` | full medium subtracted, only on pixels > 1.15× medium |
| `_cal` | **calibrated (primary readout):** subtract camera floor + α × (medium − floor) |

- **Choosing α.** For each plate, α is the value at which pixel ratio is uncorrelated with pixel brightness inside spheroids from near-null conditions: DMSO, Fluorobrite, Phloretin L/M/H, 0.5 µM OGM, 0.5 µM Ogerin. Original run: α = 0.32 / 0.32 / 0.22 for 24 / 48 / 72 h (`background_alpha_scan.csv`, `calibration.json`).

**Features per spheroid** (`spatial_features_v2.csv`). A 12 px edge margin is excluded.

| Group | Features |
|---|---|
| Distribution | median, CV, percentiles |
| Radial | rim (60 px band) vs core (inner 40 % of depth), rim − core, radial profile vs absolute and normalised depth |
| Heterogeneity | residual after removing the radial trend and the image-fixed shared residual pattern (median over all wells of a plate; two-pass); robust residual SD (`resid_sd_rob`, MAD-based); acidic / alkaline patch fraction (residual > 0.06); spatial correlation length (`corr_len_rob`, clipped residuals); Moran's I |
| Patch location | patch boundary enrichment (outer 20 %) |
| Brightness vs spectral | correlation of the brightness and spectral components of the residual |
| Morphology | area, circularity, solidity, edge roughness |

**QC flags.**
- `low_signal`: spheroid no brighter than the medium (488 or 405 signal / medium < 1.03). This removes Baf M/H at 48–72 h, among others.
- `qc_exclude`: `low_signal` plus visible debris (48 h H3, 72 h H3, 24 h B6, 72 h D9).
- Ace M wells H5 and H6 are excluded at every timepoint (fibre / debris shadow).
- 262 of 288 spheroid × timepoint images are used in the statistics.

### Step 5 — statistics

| Output | Method |
|---|---|
| `dose_time_models.csv` | Per compound and feature: OLS `feature ~ timepoint × dose_rank` (DMSO = rank 0), SEs clustered by well (= spheroid). Dose slope at each timepoint, Wald test of dose × time, Spearman ρ, Benjamini–Hochberg q. Uses non-Fluorobrite, non-`low_signal` spheroids. |
| `tracking_same_spheroid.csv`, `tracking_null_pairs.csv` | Each detrended patch map is resampled onto a normalised disc and correlated between days for the same spheroid vs 300 random different-spheroid pairs per day pair. Also reported: best match over rotation/mirroring, interior-only correlation, outline-shape correlation. |
| `radial_excess_profiles.csv` | Per-spheroid normalised-depth profile minus the same-plate DMSO mean profile. `edge_share` = outer 20 % − inner 40 %. |
| `background_model_sensitivity.csv` | Direction of each condition effect under all four background models. |
| `covariate_adjusted_effects.csv` | Condition coefficients with and without log spheroid brightness (SBR) and log area as covariates (OLS, HC3). |
| `matched_mean_<feature>.csv`, `matched_mean_family_tests.csv` | Each spheroid minus the mean of the 5 spheroids from *other* compounds (same timepoint) with the closest mean acidity; per-compound label-permutation test (2000 permutations, 48 + 72 h). |
| `feature_icc.csv` | ICC(1): fraction of spheroid-to-spheroid variance explained by condition. |
| `trajectory_classification.csv` | Leave-one-spheroid-out nearest-centroid classification of 24 → 48 → 72 h feature trajectories, with a 200× label-shuffle null. |

All random steps use fixed seeds, so reruns give identical numbers.

### Step 6 — spatial figures

- `maps_<compound>.svg`: rows = DMSO + doses; columns = 3 spheroids × 24/48/72 h (same well = same spheroid). Top block: calibrated 488/405 map; bottom block: detrended patch map. Yellow frame = low-signal spheroid.
- `dose_time_spatial_grid.svg`
- `robustness_tracking_trajectories.svg`

### Step 7 — single-plane validation

- **Equatorial plane:** the slice with the sharpest 488 outline (median 6 slices above z0).
- **Chromatic shift:** the 405 channel focuses ~2 slices (≈ 20 µm) higher than 488, found by high-pass cross-correlation between the stacks.
- **Max-projection source planes:** for each pixel, which slice supplied the max-projection value. 488 always comes from z0; 405 comes from planes up to ~190 µm higher.
- **Feature comparison:** all features are recomputed on the chromatic-matched equatorial plane (488 slice z, 405 slice z + 2) and compared with the max projection built from the same stack. Condition effects vs DMSO are expressed in pooled-SD units.
- **Depth series:** ratio, contrast vs DMSO and rim − core as a function of height above the equator.
- **Single-plane patch tracking**, plus the correlation between max-projection and single-plane patch maps.
- **Medium 405 signal vs compound dose**, after a position (row × column) correction. Result: **OGM fluoresces in the 405 channel**, so its apparent alkalinization cannot be separated from compound fluorescence with these data.

### Step 15 — acid-patch dynamics

Follows each spheroid's acid patches from one read to the next (same well = same spheroid; consecutive timepoints and first → last). Patches are the step-4 patches (log residual > log 1.06, ≥ 10 px), read from `spatial_maps_cal.pkl`, so it works on any results folder, including the equatorial-plane runs of the follow-up experiments. Method: `lib/patch_dynamics.py`.

- **Patch units.** Neighbouring patches fuse into one connected region at the threshold when their centres are closer than ~4 blob widths. By default each region is therefore split at its separate peaks: watershed of the log residual, seeded by maxima ≥ 0.04 log units (~4 %) more prominent than their surroundings. The patch area is unchanged. Set `SPHEROID_PATCH_SPLIT=components` for plain connected components and `SPHEROID_PATCH_PEAK_H` to change the prominence.
- **Common frame.** Each spheroid-timepoint is centred on its mask and scaled by its equivalent radius (R). Drift in the well and isotropic growth do not count as movement. Two registrations are reported (`register` column): `fixed` (no rotation) and `rotreg` (the rotation that best correlates the two log-residual maps is removed first; `rotation_deg`).
- **Linking.** One-to-one assignment on centroid distance (Hungarian), accepted ≤ 0.35 R → *persist*. Otherwise, by overlap in the common frame (≥ 30 % of the patch): later patch on an earlier patch → *split*, else *emerged*; earlier patch on a later patch → *merged*, else *vanished*.
- **1 Movement.** Centroid displacement of persisting patches (R per day and µm per day; radial and tangential parts). Patch-mask Jaccard distance (1 − |A∩B|/|A∪B| inside the region valid on both days). Centroid-set Jaccard distance (centroids within 0.15 R count as the same patch; 1 − m/(n₁ + n₂ − m)). The Jaccard distances measure how much the patch pattern changed, not how far patches moved; the displacement does that.
- **2 Growth.** log2 fold-change of patch area minus log2 fold-change of spheroid area, per persisting patch (`rel_growth_patch_*`) and for all patch area (`rel_growth_total_*`). 0 = the patch grows in proportion to the spheroid.
- **3 Number.** Patches per spheroid and per mm², emerged / vanished per day, splits, merges, turnover.
- **4 Patch acidity** (per patch in `patch_table.csv`; spheroid summaries in `patch_per_spheroid.csv`; figure `patch_acidity.svg`).
  - **Ratio:** calibrated 488/405 inside each patch (`ratio_mean`, `ratio_median`, `ratio_p95`). Patches are selected for being more acidic than their surroundings, so these values are high by construction: compare them between conditions, not with the rest of the spheroid.
  - **Contrast:** `log_contrast_depth` = ln(patch / non-patch tissue at the same normalised depth, ±0.05) is the preferred measure, being the least affected by the rim/core optical gradient. `log_contrast_ring` uses a 4-px ring around the patch instead.
  - **Over time:** for persisting patches, `contrast_change` and `ratio_log2fc` show whether a patch becomes more or less acidic.
  - **Shape:** `eq_diam_um`, `eccentricity`, `solidity`, `dist_edge_um`, `depth`.
  - **Spheroid level:** `patch_ratio_mean` (area-weighted) and `nonpatch_ratio_median`.
- **Signal-loss check.** Medium-subtracted brightness of each channel in the patch vs same-depth tissue (`sig488_log2_vs_depth`, `sig405_log2_vs_depth`).
  - **Why:** acidification moves the channels apart (the 405-excited signal of pHluorin2 falls; the 488-excited signal does not). Losing signal (fewer cells, necrosis, shadowing) dims both, and with residual medium in the calibrated model (background fraction < 1) a dim spot reads as acidic.
  - **Rule:** `signal_loss` = both channels ≥ 20 % dimmer than same-depth tissue. Such patches are flagged, not removed.
  - **Assumption:** the rule assumes the 488-excited signal does not fall with acidification. Check this against your nigericin calibration if you have one.
- **Chance level.** All metrics are also computed for 200 random pairs of *different* spheroids per transition (`patch_transitions_null.csv`; red dashed line in the figure). A same-spheroid value near the chance level means the pattern was not preserved.
- **Statistics.** Per-timepoint counts and densities: the step-5 dose × time model. Transition metrics: OLS slope on dose rank vs DMSO per compound and transition (HC3), Spearman ρ, BH q (`patch_dose_models.csv`).
- **Outputs.** `patch_table.csv` (every patch), `patch_per_spheroid.csv`, `patch_transitions.csv`, `patch_links.csv` (every link and event), `patch_transitions_null.csv`, `patch_condition_summary.csv`, `patch_dose_models.csv`, `patch_dynamics_tracks.svg`, `patch_dynamics_summary.svg`, `patch_acidity.svg`. `SPHEROID_PATCH_FIGURES_ONLY=1` redraws the figures from the saved tables.

**Validation** (`dev_tools/make_synth_patches.py` → `dev_tools/validate_patch_dynamics.py`). Synthetic plates with designed patch behaviour: movement 0.08 / 0.16 / 0.28 R per day, area growth ×1.3 / 1.7 / 2.2 per day relative to the spheroid, +1 / 2 / 4 new and −1 / 2 / 3 lost patches per day, one merge, one split, plus two nuisances (whole-spheroid rotation 20–90° per day; drift 150 px per day).
- Detected: 1199 of 1199 designed patches.
- Counts exact: 96 % of transitions in static spheroids.
- Movement: 0.084 / 0.166 / 0.296 R per day. Static noise floor 0.02 R per day.
- New / lost per transition: exact in every dose group.
- Merge and split: detected.
- Spheroid rotation: found to within 0.2°; `rotreg` removes the apparent movement (0.20–0.24 → 0.02 R).
- Drift: no movement detected.
- Patch acidity: designed acid patches measured 1.13× the ratio of same-depth tissue. Designed signal-loss spots (both channels ×0.35, no ratio change; TMZ wells; background fraction fixed at 0.3 as on the real plates via `dev_tools/synth_patch_experiment.json`) read as convincing acid patches (1.08×). The `signal_loss` flag caught 18 of 18 spots and 0 of 1,158 real acid patches.
- Relative growth: right direction and order but underestimated as patches grow larger (log2 0.38 / 0.77 / 1.14 per day measured as 0.35 / 0.61 / 0.97).

**Limitations.**
- **Growth is underestimated.** Patches are defined relative to the spheroid's own radial trend and median, so a patch that covers a large share of a radial ring raises that ring's median and is under-measured. The 1.5 px smoothing also inflates small patches. Compare relative growth between conditions rather than reading it as an absolute rate.
- **Fast movement is limited by the 0.35 R link limit.** Movement near that limit (0.28 R per day in the test) produces some false emerged/lost events.
- **Rotation search is biased.** It always finds some alignment, so its chance level is higher (patch-mask Jaccard 0.75 vs 0.92 for `fixed`); judge `rotreg` values against its own null.
- **Max projection merges depths.** Patches seen in a max projection come from different depths. Where single-plane data exist (follow-up `_equatorial` folders), run step 15 there as well.

### Primary readouts (pre-specified analysis plan) — step 16

The pipeline produces thousands of values per spheroid per projection from 3 spheroids per condition. To keep the confirmatory analysis small and fixed in advance, ten primary readouts are defined. Everything else is secondary (exploratory) or used for QC. Order: `python run_all.py 15 17 16`.

| # | Question | Readout | Table | Assessed at |
|---|---|---|---|---|
| 1 | Is the spheroid more acidic overall? | `ratio_median_cal`, as log2 fold-change vs same-plate DMSO | `spatial_features_v2.csv` | primary timepoint |
| 2 | How much of the spheroid is acidic patch? | `acid_patch_frac_cal` | `spatial_features_v2.csv` | primary timepoint |
| 3 | How acidic are the patches? | `patch_contrast_depth_mean` (ln, patch vs same-depth tissue) | `patch_per_spheroid.csv` | primary timepoint |
| 4 | How many patches? | `patch_density_per_mm2` | `patch_per_spheroid.csv` | primary timepoint |
| 5 | Do new patches appear? | `emerged_per_day` | `patch_transitions.csv` | primary transition |
| 6 | Do patches disappear? | `vanished_per_day` | `patch_transitions.csv` | primary transition |
| 7 | Do patches move? | `disp_per_day` (spheroid radii per day) | `patch_transitions.csv` | primary transition |
| 8 | Do patches grow faster than the spheroid? | `rel_growth_patch_per_day` (log2 per day; 0 = in proportion) | `patch_transitions.csv` | primary transition |
| 9 | Does the spheroid grow? | `growth_log2_per_day` (log2 area change per day) | `primary_readouts_per_spheroid.csv` | primary transition |
| 10 | Is cell mass kept? | `signal_log2_per_day` (log2 change per day of the integrated medium-subtracted 488 + 405 signal) | `primary_readouts_per_spheroid.csv` | primary transition |

**Analysis.**
- **Unit and reference:** the spheroid is the unit, and same-plate DMSO is the reference. Spheroids with `qc_exclude` and Fluorobrite are left out.
- **Primary timepoint and transition:** default to the last read and the last consecutive pair. Fix them before looking at the data, with `SPHEROID_PRIMARY_TP` and `SPHEROID_PRIMARY_PAIR`.
- **Registration for 5–8:** `fixed` by default. Set `SPHEROID_PRIMARY_REGISTER=rotreg` if the rotation check in `patch_dynamics_summary.svg` shows that spheroids turn between reads; otherwise rotation appears as movement and as new/lost patches.
- **Test:** for each compound, OLS of the readout on dose rank (DMSO = 0) with HC3 standard errors. Benjamini–Hochberg is applied across all 10 readouts × compounds as one family.
- **Single-dose conditions** (TMZ, Fluorobrite) have no dose trend and are not tested; they are shown in the figure only.

**Sensitivity analyses** (reported beside each test; a primary result should survive both):
- (a) adjusted for log spheroid area. Drugs that change spheroid size change patch counts and densities for reasons unrelated to pH.
- (b) readouts 2–8 recomputed without patches flagged `signal_loss`. A result that disappears here should be reported as unresolved rather than as a pH effect.
- (c) adjusted for the spheroid's own growth over the primary transition. This asks whether the effect survives accounting for growth inhibition. When growth itself changes with dose, the adjustment can absorb a real effect (on the synthetic plates it removed the Baf low-signal artefact), so read it as "is growth a sufficient explanation?", not as the corrected estimate.
- (d) readouts 1–4 adjusted for the same spheroid's value at the baseline read (`SPHEROID_BASELINE_TP`, default first read), which removes spheroid-to-spheroid variation. This is valid only if the drug has not yet acted at baseline, so the dose trend *at* the baseline read is reported beside it (`q_baseline_effect`). If that is significant, do not use (d).

**Total signal (readout 10).** The 488 + 405 sum changes much less with pH than their ratio does, so its trajectory tracks reporter-expressing cell mass. Falling total signal together with a rising ratio is the signature of a dying spheroid reading acidic. The sum is not perfectly pH-insensitive (the 405-excited signal of pHluorin2 falls with acidification), so check it against your calibration. It is not flat-field corrected: a spheroid that drifts toward the edge of the field reads a little dimmer (on the synthetic plates, the drifting Ace H spheroids lost 0.004 log2 per day).

**Early predicts late** (`early_predicts_late.csv`, `early_predicts_late.svg`).
- **Question:** within a condition, do features at an early read (`SPHEROID_EARLY_TP`; default the second read when the first is < 12 h, as for 3/24/48/72 h plates, otherwise the first) predict *later* changes in growth, total signal, mean ratio and patch fraction?
- **Predictors:** mean ratio, acid and alkaline patch fraction, patch contrast, patch density, signal-loss fraction, and the step-17 bright-field and puncta features.
- **Design:** the outcome window starts at the read *after* the early read (e.g. 24 h features → 48→72 h change). Early spheroid size is removed from both sides.
- **Statistic:** the partial Spearman correlation within condition, with a permutation p-value (the predictor shuffled among spheroids of the same condition, 2,000 times) and BH correction across all pairs.
- **Why this design:** on synthetic plates with no designed link, letting the early read start the outcome window produced correlations up to ρ = 0.86 (regression to the mean and shared measurement error). Model-based p-values with 2–3 spheroids per condition gave false positives. With the design above, 1 of 36 pairs reached nominal p < 0.05 and the smallest q was 0.50, as expected under no association. Power to find a real link has not been tested on designed data.
- **Interpretation:** these are associations between spheroids of the same condition, not causal effects.

**Secondary readouts** (`secondary_readouts_tests.csv`, their own BH family): alkaline patches (step 15 `*_alk`), and bright-field texture and surface puncta (step 17). The dose-trend test assumes a steady change with dose. It misses effects that are present at one dose only: on the synthetic plates the 5 µM-only acidic puncta and the 20 µM-only bright puncta were not picked up by the Ogerin trend, although `texture_qc.svg` shows them clearly.

**QC to read before interpreting.**
- **Whole-spheroid signal loss:** dim spheroids read as acidic (the effect behind `low_signal`). Experiment 1 uses a 1.03 cut-off and the follow-ups 1.15. On the synthetic plates, three dim spheroids at 1.04 passed the 1.03 cut-off and dominated readout 1 (+0.56 log2).
- **Rotation check:** the rotation panel of `patch_dynamics_summary.svg`.
- **Patch artefacts:** `frac_signal_loss` per condition (`patch_acidity.svg`).

**Secondary readouts.**
- `resid_sd_rob_cal`, `corr_len_rob_cal`, `patch_boundary_enrich_cal`.
- `rim_minus_core_cal`: optically confounded in max projections; use the equatorial plane.
- `jaccard_mask_dist`, `centroid_jaccard_dist`: pattern persistence, which mixes movement, emergence and growth.
- `contrast_change_mean`, `radial_shift_mean`, `mean_patch_area_um2`, `patch_ratio_mean`. The last is high by construction because patches are selected for acidity.

**Outputs.**
- `primary_readouts_per_spheroid.csv`: the eight readouts and their `__nosl` versions per spheroid.
- `primary_readouts_tests.csv`: slope, SE, p and q for the main test and both sensitivity analyses, with DMSO and top-dose means.
- `primary_readouts.svg`.

**Test on the synthetic patch plates.** 14 of 72 tests had q < 0.05:
- **Designed effects** (all detected): 3BP → new patches and density; Phlor → lost patches, density and patch area; Oz → movement; Rot → relative growth.
- **Artefacts** (both caught by the QC above): Olig movement is spheroid rotation (disappears with `rotreg`); Baf mean acidity is whole-spheroid signal loss.
- **Small spurious effect:** Oz also showed relative growth (0.21 log2 per day at the top dose against a designed 0). Patches moving to a different depth change their measured size slightly.

**Before first use on real plates.**
- Run steps 15–16 on the existing MCF7 results and check `patch_dynamics_tracks.svg` by eye.
- Confirm with the nigericin calibration that the 488-excited signal does not fall with acidification; the `signal_loss` flag assumes it does not.

### Step 15 — alkaline patches

Step 15 also tracks **alkaline** patches (log residual < −log 1.06, ≥ 10 px; step 4's definition, rebuilt from the stored map). Outputs are written with the suffix `_alk` (`patch_table_alk.csv`, `patch_transitions_alk.csv`, … `patch_dynamics_summary_alk.svg`). `SPHEROID_PATCH_POLARITIES=acid` skips them. The rebuilt patches match step 4's `alk_patch_frac_cal` in 96 % of spheroid-reads (max difference 0.0005; the stored map is float16).

**Caveat: mirror artefact.** Patches are defined relative to the spheroid's own radial median. When acid patches become large, they raise that median, and ordinary tissue then falls below it and reads as alkaline. On the synthetic plates, the growing acid patches of Rot produced a significant rise in alkaline patch density and emergence. Read an alkaline increase together with the acid patch fraction of the same spheroids.

### Step 17 — bright-field texture and surface puncta

Method and definitions: `lib/texture.py`. Outputs: `texture_features.csv` (per spheroid-read), `puncta_table.csv` (per punctum), `bf_maps.npz`, `texture_qc.svg`, `puncta_examples.svg`.

**Bright field** (2 × 2-binned, normalised to the well's medium level):
- `bf_core_vs_rim_log2`: core (normalised depth > 0.6) vs rim brightness, log2; < 0 means a darker core.
- `bf_texture_core`: granularity (SD of high-pass brightness).
- `bf_glcm_contrast`, `bf_glcm_homogeneity`, `bf_glcm_entropy`: grey-level co-occurrence texture of the core.
- `dark_core_frac`: fraction of core darker than 0.8 × the rim.
- `bf_patch_vs_depth_log2`: bright field inside acid patches vs same-depth tissue. Negative means patches sit in darker (denser or necrotic) tissue.
- Source: the Cytation ZProj bright field (a max projection, which lowers the contrast of a dark core) or, for follow-up plates, the bright-field slice at the equatorial plane (`bf_source`).
- Position in the field changes bright-field illumination. The drifting synthetic spheroids (Ace H) showed small but significant texture differences.

**Surface puncta** at full resolution (0.70 µm/px), in the outer 30 µm of the projected spheroid. Here the line of sight runs tangentially through the outer cell layers, with the least out-of-focus tissue stacked on top.
- **Detection:** Laplacian of Gaussian matched to a ~1 µm spot, threshold 5 × its robust noise SD, in either channel.
- **Per punctum:** `amp488` and `amp405` (brightening vs the local signal), and `contrast_log2` = 488/405 at the punctum vs a 3.5–6.3 µm ring around it. Kind: acidic (> log2 1.15), alkaline, or neutral.
- **Per spheroid:** `puncta_per_1000um2`, `acidic_puncta_per_1000um2`, `frac_acidic_puncta`, median contrast and amplitudes.
- **`rim_cellscale_sd`:** SD of the 0.6–2.8 µm band-pass log ratio in the band, i.e. cell-scale ratio heterogeneity that the binned maps average away. It includes camera noise and rises at low signal (the dim synthetic Baf spheroids read 5× higher), so apply the low-signal QC before reading it.
- **Optical limit:** with ~1 µm lateral resolution, puncta smaller than ~1 µm appear as ~1 µm spots with reduced amplitude. Out-of-focus light in a max projection lowers contrast further. On follow-up plates, the equatorial-plane run is the sharper view.

**Validation (synthetic plates).**
- **Acidic puncta** (488 × 1.1–2.0 at the centre, 40 per read, 5 µM Ogerin): 79–87 % detected across the amplitude range; 78 % of detected ones classed acidic.
- **Bright, non-acidic puncta** (both channels brighter, 20 µM Ogerin): 81–85 % detected, 0 % classed acidic.
- **False puncta** in spheroids without designed puncta: 0.05 per spheroid, none acidic.
- **Necrotic core** (50 µM Ogerin, core darkened 20 % / 35 % at 48 / 72 h): `bf_core_vs_rim_log2` −0.30 / −0.58 (designed −0.32 / −0.62); dark-core fraction 0.46 / 0.78; granularity and GLCM contrast up about 10-fold. All other spheroids stayed at 0.
- **Growth:** 0.328 log2/day measured against the designed 0.327 for every spheroid.

### Step 2 — legacy first spatial pass (superseded)

This is a verbatim replay of the interactive cells of the first spatial analysis: raw ratio, ratio flat-field, mask trimming, Ace M debris check, effect-size heatmap and overview. It is kept only so those figures can be regenerated. Its conclusions are superseded by steps 4–7; in particular, the DMSO "acidic rim" it reports is a max-projection artefact.

## 5. Output files → reports and figures

| Report (in the project) | Produced by |
|---|---|
| `analysis_report.md` (whole-spheroid) | step 1 |
| `spatial_analysis_report.md` (first spatial pass) | step 2 |
| `spatial_dose_time_report.md` | steps 4–6 |
| `optical_validation_report.md` | step 7 |

## 6. Verification of this package

The scripts were run end-to-end from the raw images and compared with the tables saved during the original analysis:

| Output | Agreement with the original analysis |
|---|---|
| `per_well_measurements.csv` (step 1) | identical (all 288 wells; max difference < 1e-15) |
| `stage1_status.csv`, masks (step 3) | identical (283 segmented, 5 failures, 20 debris-trimmed masks) |
| `background_alpha_scan.csv` → `calibration.json` (step 4) | identical scan; fitted α = 0.316 / 0.315 / 0.219 for 24 / 48 / 72 h |
| `spatial_features_v2.csv` (step 4) | identical (277 × 119; 15 `qc_exclude`) |
| `dose_time_models.csv`, `matched_mean_family_tests.csv`, `trajectory_classification.csv`, `tracking_*.csv`, `radial_excess_profiles.csv`, `background_model_sensitivity.csv`, `feature_icc.csv` (step 5) | identical (max difference < 1e-13, floating-point rounding) |
| `covariate_adjusted_effects.csv` (step 5) | coefficients identical; the stored p/q values at 24 h and 2 rows at 48 h differ from the copy saved during the original analysis (that copy has p = 1 for every 24 h row). The figures use only the coefficients. |
| all `zplane_*.csv`, `medium_background_by_well.csv` (step 7) | identical |
| `spatial_features_per_well.csv`, `spatial_effect_sizes_vs_DMSO.csv` (step 2, legacy) | identical |

Step 7 uses the background fractions rounded to 0.32 / 0.32 / 0.22, as in the original run (`lib/zstack_analysis.py`). Using the unrounded values from step 4 changes single-plane ratios by < 0.01.

## 7. Follow-up experiments (MCF7 repeat, HEK293)

The same code runs on other plates through a small JSON file in `experiments/` (data folder, ordered `{timepoint: plate folder}`, exclusions, segmentation method):

```
python run_experiments.py                     # all experiments/*.json: step 3 once, then steps 4-6 on max projection and on the equatorial plane
python step8_compare_experiments.py           # compound effects across datasets; set SPHEROID_EXP1_DIR to the main-run results folder
```

Differences from the first experiment, all switched on by the JSON (the first experiment is unaffected; re-running steps 3–6 with the generalised code reproduces every table above exactly):
- **No Cytation ZProj export.** Projections are built from the raw slices.
- **Coarse z (7 slices, 40 µm).** Step 3 reads each stack once and writes both the max projection (`<exp>_max`) and the equatorial plane (`<exp>_equatorial`; slice with the sharpest 488 outline; 488 and 405 from the same slice, since the ~20 µm chromatic shift is below the z-step). Step 7 (fine-z validation) does not apply.
- **Segmentation `v2`.** The smaller spheroids and strong phase halos defeat the original bright-field-max segmentation (it selected the whole field or a debris blob). `v2` segments the bright-field *min* projection after flattening and keeps the dark object closest to the image centre.
- **Background fraction.** If the brightness–ratio correlation in the near-null wells barely changes with α (spread < 0.15), α cannot be identified and the JSON value `alpha_fixed` (0.30 for the follow-ups) is used. `calibration.json` records which case applied.
- **Low-signal flag.** `low_signal_sbr` in the JSON (1.15 for the follow-ups; 1.03 in the original run). Step 8 applies 1.15 to every dataset, including experiment 1, so that all datasets are compared on the same footing.
- **Early timepoint.** A 2 h plate is treated as an early treatment timepoint, not a baseline. Every timepoint is compared with its own plate's DMSO wells, and no spheroid is normalised to its own 2 h value. Step 8 writes `dose_trend_by_timepoint.csv` and `fig_onset_2h_vs_late.svg` (`lib/fig_onset.py`).
- **Image QC.** `python qc_scan.py qc.csv` scans every image of the three experiments (in parallel; the optional `shard/n_shards` argument, e.g. `0/2`, still splits the work between separate runs; `SPHEROID_QC_BASES=<file.json>` replaces the hard-coded data folders): detector saturation (65504), medium background share and Dice of both bright-field segmentations against a fluorescence mask. Edit `BASES` to add plates. MCF7 repeat wells A8 and A10 are excluded (`exclude_wells_all_tp` in `experiments/MCF7_exp2_repeat.json`) because a 405-bright fibre saturates up to 34 % of the spheroid.
- **Timepoints.** Any set of labels (`2h`, `24h`, …); late-window analyses use the last two timepoints, tracking uses consecutive days plus first → last.

## 8. Known limitations

- **Layout and doses** are defined in code: plate layout in `lib/common.py` (`layout`), L/M/H doses in `lib/stage2_features.py` (`DOSE_UM`), exclusions in `config.py`. Doses are the "Cell" L/M/H columns of the standard dose sheet, read as final µM.
- **Filename patterns** assume Cytation C10 naming. On other systems, adjust `zproj_index` / `raw_index` in `lib/common.py` and the file names in the representative-image scripts.
- **Two thresholds are analysis choices:** the patch threshold (0.06) and rim width (60 px). They change absolute patch fractions but not the ranking of conditions.
- **Absolute pH** needs an in-situ calibration (nigericin/high-K⁺ buffers). Depth-dependent 405/488 attenuation cannot be fully excluded without a pH clamp.
- **Not in this public copy:** the collaborator plate configuration and its analysis scripts (steps 9–14), the cross-dataset results table (`reference/effects_all_datasets.csv`) and the grant-figure session log. Data folders in `config.py`, `qc_scan.py` and `experiments/*.json` are placeholders (`C:\path\to\data\...`): set `SPHEROID_DATA_DIR` or edit them.

## 9. Folder contents

```
README.md                      this file
requirements.txt
config.py                      data/output paths, plate folders, QC exclusions
run_all.py                     runs the steps in order
run_experiments.py, experiments/*.json, step8_compare_experiments.py   follow-up experiments
step1_whole_spheroid.py ... step7_single_plane_validation.py
lib/                           analysis modules, executed by the steps
  common.py                    layout, file indexing, loading, bright-field segmentation
  stage1_extract.py            binned extraction + debris trimming
  stage2_features.py           fields, background models, spatial features, dose table
  analysis_stats.py            dose-time models, matched mean, ICC, LOO classification
  tracking.py                  normalised-disc patch-map correlation
  zstack_extract.py, zstack_analysis.py, zfull.py, zcompare.py   single-plane validation
  make_figs.py, fig_*.py       figures
  figstyle.py                  matplotlib style helpers
  whole_spheroid.py            step 1 per-well segmentation + measurement
  parallel.py                  worker-process pool (SPHEROID_WORKERS)
  fastops.py                   exact fast morphology / nanmedian
  patch_dynamics.py, fig_patch_dynamics.py, fig_patch_acidity.py   step 15 acid / alkaline patch tracking and acidity
  texture.py                   step 17 bright-field texture and full-resolution surface puncta
dev_tools/                     exactness tests, synthetic test data (incl. designed patch dynamics), output comparison, step-15 validation
legacy/                        replay of the superseded first spatial pass
```
