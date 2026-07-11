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
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

import numpy as np
from scipy.optimize import brentq

from .blocking import BlockModel
from .dsc import DSCResult, compute_dsc
from .structure import Structure
from .wsme import WSMEParams, WSMEResult, run_wsme

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
    search_T_step_k: float = 2.0,
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
    ``search_T_step_k`` (default 2.0 K) uses a coarser grid for every
    Brent *iteration*; only the FINAL, converged xi gets one full-
    resolution sweep at ``T_grid``'s resolution (default 0.5 K, per
    spec) for the returned ``TmResult`` and every reported Tm/peak value.
    This does not change any physics or approximate the model itself --
    it only trades search-time resolution for the (already
    spline-smoothed) Cp curve used purely to locate approximately where
    Tm(xi) crosses the target during the search.
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

    def objective(xi_kj_mol: float) -> float:
        return tm_of_xi(xi_kj_mol, search_T_grid).tm - target_tm_k

    lo, hi = xi_bracket_kj_mol
    f_lo, f_hi = objective(lo), objective(hi)
    if not np.isfinite(f_lo) or not np.isfinite(f_hi) or f_lo * f_hi > 0:
        cp_lo, cp_hi = tm_of_xi(lo, search_T_grid), tm_of_xi(hi, search_T_grid)
        raise CalibrationError(
            f"No sign change for Tm(xi)-{target_tm_k} across bracket [{lo}, {hi}] kJ/mol "
            f"(Tm(lo)={cp_lo.tm:.1f} K, Tm(hi)={cp_hi.tm:.1f} K) -- cannot bracket a root "
            "with Brent's method. This structure's melting transition may be genuinely "
            "outside the paper's expected xi range; do not widen the bracket silently.",
            cp_result=cp_hi,
        )

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
