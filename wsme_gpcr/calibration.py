"""xi (van der Waals interaction energy per native contact) calibration.

Per Anantakrishnan & Naganathan, Nat Commun 14:128 (2023), xi is the
model's single free parameter: it is NOT a fixed constant but must be
identified per structure so the predicted heat-capacity peak (melting
temperature Tm) falls at 333 K. Leaving xi at a generic default (as this
codebase did before this module existed) can leave the folded state
thermodynamically buried under conformational entropy -- exactly the
GPR68 inactive-structure failure this module was built to fix (see
FINDINGS.md and the "Prompt 1" task this module implements).

Units: this codebase's ``WSMEParams.ene`` (and everything computed here)
is in kJ/mol, matching the rest of ``wsme_gpcr``. The paper reports xi in
J/mol (effective mean -48.9 +/- 2.76 J/mol over 45 receptors, i.e.
-0.0489 +/- 0.00276 kJ/mol). Every value this module reports alongside a
paper-comparable number is labeled with its unit explicitly -- do not
assume kJ/mol vs J/mol from context alone when reading these results.

Two calibration modes, matching the paper exactly (do not conflate them):

  - Tm mode (``calibrate_xi_tm_mode``): single structure, one xi, solved
    so the model's own heat-capacity peak (or the trough between two
    peaks, if bimodal) lands at 333 K. This is a real calibration against
    an independent target (the melting temperature) -- the folded-state
    stability that falls out of it is a genuine model prediction.

  - Iso-stability mode (``calibrate_xi_isostability_mode``): given a
    REFERENCE structure whose xi has already been Tm-calibrated, solves
    for a companion structure's (e.g. the other conformational state's)
    xi so its folded-minus-unfolded free energy matches the reference's
    exactly. This is NOT a second independent calibration -- it imposes
    equal stability by construction. The relative stability of the two
    states under this mode must never be read as a result; only the
    common ΔG_fold value and pattern of which contacts/blocks differ are
    meaningful. This module's ``IsoStabilityResult`` carries an explicit
    warning string saying so, and callers must surface it every time this
    mode's output is reported (mirroring the guardrail in Prompt 1).

Reference temperature: 310 K throughout (the codebase's own
``WSMEParams`` default, and Prompt 1's own choice of "the 310 K 1D
profile") -- xi is calibrated once at pH 7.0 (fully neutral His; the
pipeline's default) and held fixed across any pH ramp built on top of it.
This module does not touch pH machinery at all.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import warnings
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

import numpy as np
from scipy.optimize import brentq

from .blocking import BlockModel
from .coupling import CouplingResult
from .dsc import DSCResult, compute_dsc
from .structure import Structure
from .wsme import WSMEParams, WSMEResult, run_wsme

# "Strongly coupled residue" Z-score threshold for fc (see compute_fc).
# This IS the paper's own verified definition (not a guessed proxy -- see
# the module's PDF, Methods/Results: "residue-averaged coupling free
# energies <DeltaGc> ... were Z-scored to account for intrinsic
# differences in the range of coupling free energies, and residues that
# exhibit a Z-score greater than one were labeled as strongly coupled").
# Confirmed directly against the paper's own bundled per-receptor
# DeltaGc_310 ground truth (53 receptors, GPCR-Landscapes reference
# repo): this Z-score procedure reproduces the paper's reported 13.0 +/-
# 4.5% almost exactly (13.29 +/- 4.27% measured), and DeltaGc_310 itself
# was confirmed to be the row-mean (not row-max) of CouplingMat_310 (r
# =0.9998 against a direct row-mean reconstruction) -- see FINDINGS.md's
# "fc definition corrected" entry for the full derivation. This superseded
# an earlier, incorrect implementation that thresholded raw |coupling
# free energy| against a fixed absolute value (1 RT ~= 2.58 kJ/mol, a
# defensible-sounding but wrong guess): that formulation saturated fc at
# 92-100% on every real receptor tested, because it wasn't scale-
# invariant per receptor the way the paper's own Z-score procedure is.
DEFAULT_FC_Z_THRESHOLD = 1.0

# Paper's reported effective mean +/- std, over 45 receptors, in J/mol.
PAPER_XI_MEAN_J_MOL = -48.9
PAPER_XI_STD_J_MOL = 2.76
PAPER_TARGET_TM_K = 333.0
PAPER_XI_BRACKET_KJ_MOL = (-0.080, -0.020)  # -80 to -20 J/mol, per Prompt 1
FOLD_WINDOW_FRAC = 0.15  # "top 15% of the reaction coordinate" post-condition


class CalibrationError(RuntimeError):
    """Raised when xi calibration cannot meet its target or post-condition.

    Always carries the Cp(T) curve (and, where relevant, the 310 K 1D
    profile) that led to the failure, per Prompt 1's explicit instruction
    not to return a number that fails its post-condition -- a caller
    catching this can inspect ``cp_result``/``fes_result`` to see why.
    """

    def __init__(self, message: str, cp_result: "TmResult" = None, fes_result: WSMEResult = None):
        super().__init__(message)
        self.cp_result = cp_result
        self.fes_result = fes_result


@dataclass
class TmResult:
    T: np.ndarray             # (n,) K -- the Cp(T) grid actually used
    Cp_excess: np.ndarray     # (n,) kJ/mol/K -- from the partition function alone, NO empirical baseline
    peak_temperatures: np.ndarray  # (1 or 2,) K, ascending
    peak_heights: np.ndarray       # (1 or 2,) kJ/mol/K, matching peak_temperatures
    is_bimodal: bool
    tm: float                 # the calibration target: the single peak's T, or the trough's T if bimodal
    trough_temperature: float = None  # only set if is_bimodal


def find_cp_peaks_and_tm(T: np.ndarray, Cp_excess: np.ndarray, prominence_frac: float = 0.02) -> TmResult:
    """Locate the heat-capacity peak(s) in an EXCESS Cp(T) curve (no
    empirical native-state baseline -- see module docstring) and derive
    Tm from them.

    Peak-finding uses ``scipy.signal.find_peaks`` with a prominence
    threshold relative to the curve's own dynamic range
    (``prominence_frac * (Cp.max() - Cp.min())``) rather than a fixed
    absolute value, so it adapts to protein size without a magic number
    -- default 2% is enough to reject spline-interpolation-scale wiggles
    (``compute_dsc`` applies two rounds of cubic-spline smoothing) while
    still catching a real secondary peak in a genuinely bimodal profile.

    Bimodal handling matches the paper exactly: when two peaks are
    present, Tm is defined as the temperature of the TROUGH between them,
    not either peak.
    """
    from scipy.signal import find_peaks

    T = np.asarray(T, dtype=float)
    Cp_excess = np.asarray(Cp_excess, dtype=float)
    if len(T) < 3:
        raise ValueError("need at least 3 temperature points to find peaks")

    span = Cp_excess.max() - Cp_excess.min()
    prominence = prominence_frac * span if span > 0 else None
    peak_idx, _ = find_peaks(Cp_excess, prominence=prominence)

    if len(peak_idx) == 0:
        raise CalibrationError(
            "no Cp(T) peak found at all -- the partition function's temperature "
            "dependence is too flat/monotonic to define a melting transition",
            cp_result=TmResult(T=T, Cp_excess=Cp_excess, peak_temperatures=np.zeros(0),
                                peak_heights=np.zeros(0), is_bimodal=False, tm=float("nan")),
        )

    if len(peak_idx) == 1:
        i = peak_idx[0]
        return TmResult(T=T, Cp_excess=Cp_excess, peak_temperatures=T[[i]], peak_heights=Cp_excess[[i]],
                         is_bimodal=False, tm=float(T[i]))

    # More than one peak: per the paper, use the two largest peaks and the
    # trough between them (a genuinely multi-peak profile beyond bimodal
    # would itself be a red flag -- surfaced via the returned peak count,
    # not silently collapsed to two).
    order = np.argsort(Cp_excess[peak_idx])[::-1][:2]
    two_idx = np.sort(peak_idx[order])
    i_lo, i_hi = two_idx
    trough_local = i_lo + int(np.argmin(Cp_excess[i_lo:i_hi + 1]))
    return TmResult(T=T, Cp_excess=Cp_excess, peak_temperatures=T[two_idx], peak_heights=Cp_excess[two_idx],
                     is_bimodal=True, tm=float(T[trough_local]), trough_temperature=float(T[trough_local]))


def compute_delta_g_fold(structure: Structure, block_model: BlockModel, ss_mask: np.ndarray,
                          params: WSMEParams, fold_window_frac: float = FOLD_WINDOW_FRAC) -> float:
    """Folded-well minus unfolded-well free energy (kJ/mol) at
    ``params.T``, from the 1D profile ``run_wsme`` already computes.

    "Folded well" / "unfolded well" are operationalized as the profile's
    minimum within the top/bottom ``fold_window_frac`` of the reaction
    coordinate (by block-count fraction) -- symmetric with the "folded
    minimum in the top 15%" post-condition Prompt 1 specifies for Tm-mode
    calibration. The paper's own exact windowing convention could not be
    independently verified (network access to the paper is blocked in
    this sandbox, same as every external host tried this session) -- this
    is a documented, defensible choice, not a verified transcription.
    """
    result = run_wsme(structure, block_model, ss_mask, params)
    nblocks = block_model.nblocks
    lo_cut = max(1, int(np.floor(fold_window_frac * nblocks)))
    hi_cut = max(1, int(np.ceil((1.0 - fold_window_frac) * nblocks)))
    unfolded_well = float(result.fes[result.n_values <= lo_cut].min())
    folded_well = float(result.fes[result.n_values >= hi_cut].min())
    return folded_well - unfolded_well


def _folded_minimum_ok(result: WSMEResult, nblocks: int, fold_window_frac: float = FOLD_WINDOW_FRAC) -> tuple:
    """Post-condition check: does the profile's GLOBAL minimum fall
    within the top ``fold_window_frac`` of the reaction coordinate?
    Returns (ok, n_at_minimum, fraction_at_minimum)."""
    global_min_idx = int(np.argmin(result.fes))
    n_at_min = int(result.n_values[global_min_idx])
    frac_at_min = n_at_min / nblocks
    ok = frac_at_min >= (1.0 - fold_window_frac)
    return ok, n_at_min, frac_at_min


DEFAULT_XI_SCAN_RANGE_J_MOL = (-65.0, -40.0)
DEFAULT_XI_SCAN_STEP_J_MOL = 0.25


@dataclass
class XiFoldScanResult:
    """Result of sweeping xi over a range and recording the WSME 1D
    landscape's fold fraction at each point -- see ``xi_fold_scan``."""

    xi_values_j_mol: np.ndarray
    fold_fracs: np.ndarray
    folds_anywhere: bool       # does ANY point in the scan clear the fold_window_frac bar?
    best_xi_j_mol: float       # the xi giving the highest fold fraction in the scan
    best_fold_frac: float
    n_transitions: int         # number of fold_ok True<->False flips across the scan -- a sharpness indicator


def xi_fold_scan(structure: Structure, block_model: BlockModel, ss_mask: np.ndarray,
                  params: WSMEParams = None,
                  xi_range_j_mol: tuple = DEFAULT_XI_SCAN_RANGE_J_MOL,
                  step_j_mol: float = DEFAULT_XI_SCAN_STEP_J_MOL,
                  fold_window_frac: float = FOLD_WINDOW_FRAC) -> XiFoldScanResult:
    """Does this structure fold ANYWHERE in a physically plausible xi
    range, rather than only at one fixed reference point?

    Necessary because a real paper reference receptor (4XNV/gpcr14i) was
    found to flip from 97.4% folded to 5.1% collapsed across a xi window
    under 0.7 J/mol (see FINDINGS.md's real-structure-control entry), and
    ``calibrate_xi_tm_mode`` -- the tool meant to locate a structure's own
    transition point via its Cp(T) peak -- fails broadly under real DSSP
    blocking, confirmed on that same real reference receptor, not just
    edge-case ancestral nodes. Single-point testing at one xi (e.g. the
    rhodopsin-derived default, -48.2 J/mol) can therefore misclassify a
    genuinely foldable structure as a failure purely by chance of which
    side of a sharp transition it happens to land on.

    ``xi_range_j_mol``/``step_j_mol`` default to a fairly fine scan
    (0.25 J/mol steps over a 25 J/mol window) since the transition found
    in the 4XNV case was this sharp; widen the range if a structure's own
    transition might plausibly sit outside the default window (e.g. an
    already-severely-collapsed structure at the default xi may need a
    more negative range to find where, if anywhere, it folds).
    """
    if params is None:
        params = WSMEParams()
    xi_values = np.arange(xi_range_j_mol[0], xi_range_j_mol[1] + step_j_mol / 2, step_j_mol)
    fold_fracs = np.zeros(len(xi_values))
    for i, xi in enumerate(xi_values):
        p = replace(params, ene=xi * 1e-3)
        res = run_wsme(structure, block_model, ss_mask, p)
        amin = int(np.argmin(res.fes))
        fold_fracs[i] = res.n_values[amin] / block_model.nblocks

    fold_ok_mask = fold_fracs >= (1.0 - fold_window_frac)
    folds_anywhere = bool(fold_ok_mask.any())
    best_idx = int(np.argmax(fold_fracs))
    n_transitions = int(np.sum(np.diff(fold_ok_mask.astype(int)) != 0))

    return XiFoldScanResult(
        xi_values_j_mol=xi_values, fold_fracs=fold_fracs, folds_anywhere=folds_anywhere,
        best_xi_j_mol=float(xi_values[best_idx]), best_fold_frac=float(fold_fracs[best_idx]),
        n_transitions=n_transitions,
    )


def _file_sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_commit_hash() -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                              cwd=Path(__file__).resolve().parent, timeout=10)
        return out.stdout.strip() if out.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


def _config_hash(**config) -> str:
    payload = json.dumps(config, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


@dataclass
class XiTmCalibrationResult:
    xi_kj_mol: float
    xi_j_mol: float
    z_score_vs_paper: float           # (xi_j_mol - PAPER_XI_MEAN_J_MOL) / PAPER_XI_STD_J_MOL
    tm_achieved_k: float
    target_tm_k: float
    is_bimodal: bool
    cp_result: TmResult
    fes_310: WSMEResult
    folded_minimum_ok: bool
    folded_minimum_n: int
    folded_minimum_frac: float
    provenance: dict = field(default_factory=dict)

    def summary_header(self) -> str:
        flag = "" if abs(self.z_score_vs_paper) <= 3 else "  ** OUTSIDE ~3 SIGMA OF THE PAPER'S DISTRIBUTION -- FLAGGED, NOT SILENTLY ACCEPTED **"
        return (
            f"xi calibrated (Tm mode): {self.xi_j_mol:.2f} J/mol "
            f"(z={self.z_score_vs_paper:+.2f} vs paper -48.9+/-2.76 J/mol){flag}\n"
            f"Tm achieved: {self.tm_achieved_k:.1f} K (target {self.target_tm_k:.1f} K"
            f"{', bimodal Cp -- trough used' if self.is_bimodal else ''})\n"
            f"Folded minimum at n/nblocks={self.folded_minimum_frac:.2%} "
            f"({'OK, in top 15%' if self.folded_minimum_ok else 'FAILED post-condition'})\n"
            "State stabilities and 1D profile shapes are only physically meaningful "
            "after this Tm-mode calibration -- see linkage_pka/FINDINGS.md and this "
            "module's docstring."
        )


def calibrate_xi_tm_mode(
    structure: Structure,
    block_model: BlockModel,
    ss_mask: np.ndarray,
    params: WSMEParams = None,
    target_tm_k: float = PAPER_TARGET_TM_K,
    xi_bracket_kj_mol: tuple = PAPER_XI_BRACKET_KJ_MOL,
    T_grid: np.ndarray = None,
    search_T_step_k: float = 1.0,
    xtol_kj_mol: float = 1e-4,
    structure_path=None,
) -> XiTmCalibrationResult:
    """Root-find xi (kJ/mol) such that Tm(xi) = ``target_tm_k`` (or the
    Cp trough, if the profile is bimodal at the solution), via Brent's
    method over ``xi_bracket_kj_mol`` (default -80 to -20 J/mol per
    Prompt 1). Raises ``CalibrationError`` (with the Cp(T) curve
    attached) if the bracket doesn't contain a root, or if the calibrated
    xi's 310 K profile fails the required folded-minimum post-condition
    -- this function never returns a result that fails that check.

    Performance note: on a GPCR-sized structure (~100 blocks), one
    ``compute_dsc`` sweep at the spec's default 0.5 K step (161 points)
    costs on the order of 2 minutes (measured: ~0.85 s per temperature
    point), and Brent's method typically needs 15-25 evaluations to
    converge -- a naive implementation would cost 30-60+ minutes.
    ``search_T_step_k`` (default 1.0 K) uses a coarser grid for every
    Brent *iteration*; only the FINAL, converged xi gets one full-
    resolution sweep at ``T_grid``'s resolution (default 0.5 K, per
    spec) for the returned ``TmResult`` and every reported Tm/peak value.
    This does not change any physics or approximate the model itself --
    it only trades search-time resolution for the (already
    spline-smoothed) Cp curve used purely to locate approximately where
    Tm(xi) crosses the target during the search. Real testing (see
    ``examples/calibration_regression.py``) found that too coarse a
    search grid (2.0 K) can misresolve closely-spaced bimodal Cp peaks
    for some receptors -- 1.0 K is a deliberate compromise, not
    guaranteed artifact-free for every structure; the function's own
    post-condition check on the FINAL full-resolution sweep is what
    actually guards against returning a bad answer, not the search
    grid's resolution.
    """
    if params is None:
        params = WSMEParams()
    if T_grid is None:
        T_grid = np.arange(280.0, 360.0 + 1e-9, 0.5)
    T_grid = np.asarray(T_grid, dtype=float)
    search_T_grid = np.arange(T_grid[0], T_grid[-1] + 1e-9, search_T_step_k)

    def tm_of_xi(xi_kj_mol: float, grid: np.ndarray) -> TmResult:
        p = replace(params, ene=xi_kj_mol)
        dsc = compute_dsc(structure, block_model, ss_mask, p, T_grid=grid)
        return find_cp_peaks_and_tm(dsc.T, dsc.Cp_excess)

    def safe_tm_of_xi(xi_kj_mol: float, grid: np.ndarray):
        """Never lets a "no Cp peak at this xi" failure propagate as an
        opaque crash -- returns (TmResult, None) or (None, CalibrationError)
        so callers can report exactly which xi/edge failed and why."""
        try:
            return tm_of_xi(xi_kj_mol, grid), None
        except CalibrationError as exc:
            return None, exc

    lo, hi = xi_bracket_kj_mol
    tm_lo, err_lo = safe_tm_of_xi(lo, search_T_grid)
    tm_hi, err_hi = safe_tm_of_xi(hi, search_T_grid)
    if err_lo is not None or err_hi is not None:
        lo_desc = f"Tm={tm_lo.tm:.1f} K" if err_lo is None else f"FAILED ({err_lo})"
        hi_desc = f"Tm={tm_hi.tm:.1f} K" if err_hi is None else f"FAILED ({err_hi})"
        raise CalibrationError(
            f"Cannot evaluate Tm(xi) at one or both bracket edges within the search grid "
            f"[{search_T_grid[0]:.1f}, {search_T_grid[-1]:.1f}] K: "
            f"xi={lo} kJ/mol -> {lo_desc}; xi={hi} kJ/mol -> {hi_desc}. "
            "This structure's melting transition is genuinely unresolvable somewhere in "
            "the requested bracket/T_grid for this structure -- widen T_grid or narrow "
            "xi_bracket_kj_mol explicitly; do not guess or silently widen it here.",
            cp_result=tm_lo or (err_lo.cp_result if err_lo else None) or tm_hi or (err_hi.cp_result if err_hi else None),
        )

    f_lo, f_hi = tm_lo.tm - target_tm_k, tm_hi.tm - target_tm_k
    if not np.isfinite(f_lo) or not np.isfinite(f_hi) or f_lo * f_hi > 0:
        raise CalibrationError(
            f"No sign change for Tm(xi)-{target_tm_k} across bracket [{lo}, {hi}] kJ/mol "
            f"(Tm(lo)={tm_lo.tm:.1f} K, Tm(hi)={tm_hi.tm:.1f} K) -- cannot bracket a root "
            "with Brent's method. This structure's melting transition may be genuinely "
            "outside the paper's expected xi range; do not widen the bracket silently.",
            cp_result=tm_hi,
        )

    def objective(xi_kj_mol: float) -> float:
        tm_result, err = safe_tm_of_xi(xi_kj_mol, search_T_grid)
        if err is not None:
            raise CalibrationError(
                f"Brent's method probed xi={xi_kj_mol:.6f} kJ/mol ({xi_kj_mol * 1000:.2f} J/mol) "
                f"inside the bracket and found no resolvable Cp(T) peak there: {err}. "
                "The bracket contains a region where the melting transition cannot be "
                "located -- narrow xi_bracket_kj_mol or widen T_grid; do not guess.",
                cp_result=err.cp_result,
            )
        return tm_result.tm - target_tm_k

    xi_calibrated = float(brentq(objective, lo, hi, xtol=xtol_kj_mol))
    cp_result = tm_of_xi(xi_calibrated, T_grid)  # final answer: full-resolution sweep

    p_310 = replace(params, ene=xi_calibrated, T=310.0)
    fes_310 = run_wsme(structure, block_model, ss_mask, p_310)
    ok, n_at_min, frac_at_min = _folded_minimum_ok(fes_310, block_model.nblocks)

    if not ok:
        raise CalibrationError(
            f"Calibrated xi={xi_calibrated * 1000:.2f} J/mol achieves Tm={cp_result.tm:.1f} K "
            f"(target {target_tm_k} K) but the 310 K profile's minimum is at "
            f"n={n_at_min}/{block_model.nblocks} ({frac_at_min:.1%} of the reaction coordinate), "
            "not in the required top 15% -- refusing to return a calibration that fails "
            "the folded-state post-condition.",
            cp_result=cp_result, fes_result=fes_310,
        )

    xi_j_mol = xi_calibrated * 1000.0
    z_score = (xi_j_mol - PAPER_XI_MEAN_J_MOL) / PAPER_XI_STD_J_MOL

    provenance = {
        "structure_sha256": _file_sha256(structure_path) if structure_path else None,
        "structure_path": str(structure_path) if structure_path else None,
        "commit": _git_commit_hash(),
        "config_hash": _config_hash(
            target_tm_k=target_tm_k, xi_bracket_kj_mol=xi_bracket_kj_mol,
            T_grid_lo=float(T_grid[0]), T_grid_hi=float(T_grid[-1]), T_grid_step=float(T_grid[1] - T_grid[0]),
            params={k: v for k, v in asdict(params).items() if k != "T"},
        ),
    }

    return XiTmCalibrationResult(
        xi_kj_mol=xi_calibrated, xi_j_mol=xi_j_mol, z_score_vs_paper=z_score,
        tm_achieved_k=cp_result.tm, target_tm_k=target_tm_k, is_bimodal=cp_result.is_bimodal,
        cp_result=cp_result, fes_310=fes_310, folded_minimum_ok=ok,
        folded_minimum_n=n_at_min, folded_minimum_frac=frac_at_min, provenance=provenance,
    )


@dataclass
class IsoStabilityResult:
    xi_reference_kj_mol: float
    xi_other_kj_mol: float
    delta_g_fold_common_kj_mol: float
    T_ref_k: float
    provenance: dict = field(default_factory=dict)
    warning: str = (
        "ISO-STABILITY MODE: xi_other was solved to force delta_g_fold(other) == "
        "delta_g_fold(reference) exactly. The relative stability of the two "
        "conformational states is IMPOSED by this calibration, not predicted -- "
        "it must never be read as a result."
    )

    def summary_header(self) -> str:
        return (
            f"xi (reference) = {self.xi_reference_kj_mol * 1000:.2f} J/mol, "
            f"xi (other) = {self.xi_other_kj_mol * 1000:.2f} J/mol\n"
            f"Common delta_G_fold = {self.delta_g_fold_common_kj_mol:.2f} kJ/mol at {self.T_ref_k:.1f} K\n"
            f"{self.warning}"
        )


def calibrate_xi_isostability_mode(
    reference_structure: Structure, reference_block_model: BlockModel, reference_ss_mask: np.ndarray,
    xi_reference_kj_mol: float,
    other_structure: Structure, other_block_model: BlockModel, other_ss_mask: np.ndarray,
    params: WSMEParams = None,
    T_ref_k: float = 310.0,
    xi_bracket_kj_mol: tuple = PAPER_XI_BRACKET_KJ_MOL,
    xtol_kj_mol: float = 1e-5,
    reference_structure_path=None,
    other_structure_path=None,
) -> IsoStabilityResult:
    """Solve for ``other``'s xi such that its folded-minus-unfolded free
    energy at ``T_ref_k`` matches ``reference``'s (whose xi has already
    been Tm-calibrated -- pass that value in as ``xi_reference_kj_mol``,
    do not re-derive it here). See module docstring: this imposes equal
    stability between the two states by construction, it does not predict
    it -- ``IsoStabilityResult.warning`` carries that caveat on every
    result and must be surfaced whenever this mode's output is reported.
    """
    if params is None:
        params = WSMEParams()

    p_ref = replace(params, ene=xi_reference_kj_mol, T=T_ref_k)
    target_delta_g = compute_delta_g_fold(reference_structure, reference_block_model, reference_ss_mask, p_ref)

    def objective(xi_kj_mol: float) -> float:
        p = replace(params, ene=xi_kj_mol, T=T_ref_k)
        return compute_delta_g_fold(other_structure, other_block_model, other_ss_mask, p) - target_delta_g

    lo, hi = xi_bracket_kj_mol
    f_lo, f_hi = objective(lo), objective(hi)
    if not np.isfinite(f_lo) or not np.isfinite(f_hi) or f_lo * f_hi > 0:
        raise CalibrationError(
            f"No sign change for delta_G_fold(other)-delta_G_fold(reference) across bracket "
            f"[{lo}, {hi}] kJ/mol (f(lo)={f_lo:.2f}, f(hi)={f_hi:.2f} kJ/mol) -- cannot bracket "
            "an iso-stability solution with Brent's method."
        )
    xi_other = float(brentq(objective, lo, hi, xtol=xtol_kj_mol))

    provenance = {
        "reference_structure_sha256": _file_sha256(reference_structure_path) if reference_structure_path else None,
        "reference_structure_path": str(reference_structure_path) if reference_structure_path else None,
        "other_structure_sha256": _file_sha256(other_structure_path) if other_structure_path else None,
        "other_structure_path": str(other_structure_path) if other_structure_path else None,
        "commit": _git_commit_hash(),
        "config_hash": _config_hash(
            T_ref_k=T_ref_k, xi_bracket_kj_mol=xi_bracket_kj_mol, xi_reference_kj_mol=xi_reference_kj_mol,
            params={k: v for k, v in asdict(params).items() if k != "T"},
        ),
    }

    return IsoStabilityResult(
        xi_reference_kj_mol=xi_reference_kj_mol, xi_other_kj_mol=xi_other,
        delta_g_fold_common_kj_mol=target_delta_g, T_ref_k=T_ref_k, provenance=provenance,
    )


def compute_fc(coupling_result: CouplingResult, block_model: BlockModel,
                z_threshold: float = DEFAULT_FC_Z_THRESHOLD) -> float:
    """Fraction of strongly coupled residues -- the paper's fc, reported
    as 13.0 +/- 4.5% over 45 receptors, reproduced here via the paper's
    own real procedure (not a guessed proxy; see
    ``DEFAULT_FC_Z_THRESHOLD``'s docstring for the verification against
    the paper's own bundled per-receptor ground truth):

    1. For each block, the row-mean of ``coupling_free_energy`` over all
       its partner blocks (NaN partners excluded) -- the paper's
       "residue-averaged coupling free energy" <DeltaGc>_i.
    2. Z-score that per-block vector using ITS OWN mean/std (i.e.
       relative to this one receptor's own coupling-magnitude scale, not
       an absolute kJ/mol cutoff) -- this is what makes fc comparable
       across receptors/structures of very different absolute coupling
       scale, and is the step the pipeline's earlier absolute-threshold
       implementation was missing entirely.
    3. A block counts as strongly coupled if its Z-score exceeds
       ``z_threshold`` (paper: > 1).

    fc is computed at RESIDUE granularity (matching the paper's "fraction
    of ... residues"): every residue in a strongly-coupled block counts,
    weighted by that block's real residue count, so blocks of different
    sizes contribute proportionally. A block whose row-mean is undefined
    (every partner NaN) is excluded from both the Z-scoring population
    and the coupled set -- it can neither pull the mean/std around nor
    itself be counted, rather than silently coercing to Z=0 or NaN>threshold
    (which is False either way, but excluding it from the population
    keeps the Z-scores of every OTHER block meaningful).
    """
    cfe = coupling_result.coupling_free_energy
    with warnings.catch_warnings():
        # a block whose every partner is NaN (fully masked-out coupling
        # row) legitimately produces "mean of empty slice" -- handled
        # explicitly below via `valid`, not a real problem to surface.
        warnings.simplefilter("ignore", category=RuntimeWarning)
        row_mean = np.nanmean(cfe, axis=1)
    valid = np.isfinite(row_mean)
    z = np.full(block_model.nblocks, np.nan)
    if valid.sum() >= 2 and np.nanstd(row_mean[valid]) > 0:
        z[valid] = (row_mean[valid] - row_mean[valid].mean()) / row_mean[valid].std()
    block_is_coupled = z > z_threshold  # NaN comparisons are False, correctly excluded

    residues_per_block = np.bincount(block_model.block_of_residue, minlength=block_model.nblocks)
    n_coupled_residues = int(residues_per_block[block_is_coupled].sum())
    return n_coupled_residues / block_model.nres
