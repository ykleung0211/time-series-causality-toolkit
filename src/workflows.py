"""Non-interactive causal-analysis pipeline built on the reusable toolkit primitives."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .causal_analysis import (
    compute_ccm,
    compute_dtw_sequence,
    compute_te,
    granger_direction_score,
    print_parameter_sweep_report,
    reverse_warping_path,
    run_granger_causality_report,
    sweep_ccm_convergence_steps,
    sweep_ccm,
    sweep_transfer_entropy,
    warp_series_to_match,
)
from .plotting import plot_ccm_convergence, plot_dtw_alignment, plot_line_trend
from .preprocessing import PreprocessingConfig, lagged_cross_correlation, preprocess_series_pair
from .stationarity import adf_unit_root_test, make_series_stationary, print_adf_summary
from .surrogate import print_surrogate_summary, run_surrogate_test


ALIGNMENT_MODE_COMMON_INDEX = "common_index"
ALIGNMENT_MODE_DTW_WARPED = "dtw_warped"


@dataclass(frozen=True)
class AnalysisConfig:
    """Configuration for the main causal-analysis pipeline.

    Attributes:
        run_adf_check: Whether to perform ADF unit root tests.
        auto_difference_if_nonstationary: Whether to automatically difference non-stationary series.
        max_diff_order: Maximum order of differencing to apply.
        run_lagged_cross_correlation: Whether to compute lagged cross-correlation.
        lcc_max_lag: Maximum lag to check for lagged cross-correlation.
        plot_lagged_cross_correlation: Whether to plot the LCC frame.
        run_dtw: Whether to compute DTW.
        plot_dtw_alignment: Whether to plot the DTW alignment.
        alignment_mode: How to align series for Granger/TE/CCM. Either
            "common_index" (intersect on shared index) or "dtw_warped"
            (re-express series_two on series_one's timeline via DTW).
            Note: alignment only happens if run_dtw is True.
        run_granger: Whether to run Granger causality tests.
        granger_maxlag: Maximum lag for Granger tests.
        run_granger_surrogates: Whether to run surrogate tests for Granger.
        run_te: Whether to run transfer entropy sweeps.
        te_max_lag: Maximum lag for TE sweeps.
        te_max_embed_dim: Maximum embedding dimension for TE sweeps.
        run_te_surrogates: Whether to run surrogate tests for TE.
        run_ccm: Whether to run CCM sweeps.
        ccm_max_lag: Maximum lag for CCM sweeps.
        ccm_max_embed_dim: Maximum embedding dimension for CCM sweeps.
        run_ccm_convergence: Whether to run CCM convergence sweeps.
        ccm_library_step: Step size for CCM library fractions.
        run_ccm_surrogates: Whether to run surrogate tests for CCM.
        surrogate_method: Surrogate strategy to use: ``shuffle``, ``bootstrap``, or ``both``.
        n_surrogates: Number of surrogate samples.
        surrogate_seed: Seed for surrogate sampling.
    """

    run_adf_check: bool = True
    auto_difference_if_nonstationary: bool = True
    max_diff_order: int = 2

    run_lagged_cross_correlation: bool = False
    lcc_max_lag: int = 30
    plot_lagged_cross_correlation: bool = False

    run_dtw: bool = True
    plot_dtw_alignment: bool = True
    alignment_mode: str = ALIGNMENT_MODE_COMMON_INDEX

    run_granger: bool = True
    granger_maxlag: int = 5
    run_granger_surrogates: bool = True

    run_te: bool = True
    te_max_lag: int = 5
    te_max_embed_dim: int = 3
    run_te_surrogates: bool = True

    run_ccm: bool = True
    ccm_max_lag: int = 5
    ccm_max_embed_dim: int = 3
    run_ccm_convergence: bool = True
    ccm_library_step: float = 0.1
    run_ccm_surrogates: bool = True

    surrogate_method: str = "both"
    n_surrogates: int = 200
    surrogate_seed: int = 0


def _display_series_label(label: str) -> str:
    # example: if label is "AAPL (Apple Inc.)", return "AAPL"
    if " (" in label and label.endswith(")"):
        return label.rsplit(" (", 1)[0]
    return label


def _resolve_surrogate_methods(choice: str) -> list[str]:
    if choice == "shuffle":
        return ["shuffle"]
    if choice == "bootstrap":
        return ["bootstrap"]
    return ["shuffle", "bootstrap"]


def _align_by_common_index(series_one: pd.Series, series_two: pd.Series) -> tuple[pd.Series, pd.Series]:
    """
    Align two series on their common index for fixed-lag analysis.
    This is used in the "common index" alignment mode for Granger/TE/CCM.
    """
    common_index = series_one.index.intersection(series_two.index)
    if len(common_index) == 0:
        raise ValueError("The two series have no overlapping index values for common-index alignment.")
    aligned_one = series_one.loc[common_index].sort_index()
    aligned_two = series_two.loc[common_index].sort_index()
    return aligned_one, aligned_two


def _prepare_dtw_warped_series(
    series_one: pd.Series,
    series_two: pd.Series,
    dtw_alignment,
) -> tuple[pd.Series, pd.Series]:
    warped_one_to_two = warp_series_to_match(series_one, series_two, dtw_alignment)
    reverse_alignment = reverse_warping_path(dtw_alignment)
    warped_two_to_one = warp_series_to_match(series_two, series_one, reverse_alignment)
    return warped_one_to_two, warped_two_to_one


def _run_metric_surrogates(
    metric_name: str,
    series_one: pd.Series,
    series_two: pd.Series,
    label_one: str,
    label_two: str,
    summary: dict[str, object],
    config: AnalysisConfig,
) -> None:
    methods = _resolve_surrogate_methods(config.surrogate_method)

    def _run_metric(direction_label: str, embed_dim: int, lag: int, score_index: int) -> None:
        if metric_name == "TE":
            real_func = lambda x, y: compute_te(x, y, embed_dim=embed_dim, lag=lag)
            if direction_label == f"{label_one} -> {label_two}":
                base_x = series_one.values
                base_y = series_two.values
            else:
                base_x = series_two.values
                base_y = series_one.values
        else:
            real_func = lambda x, y: compute_ccm(x, y, embed_dim=embed_dim, lag=lag)[score_index]
            base_x = series_one.values
            base_y = series_two.values

        for method in methods:
            result = run_surrogate_test(real_func, base_x, base_y, n_surrogates=config.n_surrogates, method=method, seed=config.surrogate_seed)
            print_surrogate_summary(metric_name, f"{direction_label} (embed_dim={embed_dim}, lag={lag})", result, verbose=True)

    one_to_two = summary.get("one_to_two")
    two_to_one = summary.get("two_to_one")
    if isinstance(one_to_two, dict):
        _run_metric(f"{label_one} -> {label_two}", int(one_to_two["embed_dim"]), int(one_to_two["lag"]), score_index=0)
    if isinstance(two_to_one, dict):
        _run_metric(f"{label_two} -> {label_one}", int(two_to_one["embed_dim"]), int(two_to_one["lag"]), score_index=1)


def _run_granger_surrogates(
    series_one: pd.Series,
    series_two: pd.Series,
    label_one: str,
    label_two: str,
    maxlag: int,
    config: AnalysisConfig,
) -> None:
    if not config.run_granger_surrogates:
        return

    methods = _resolve_surrogate_methods(config.surrogate_method)

    def _run_direction(direction_label: str, target: pd.Series, source: pd.Series) -> None:
        for method in methods:
            result = run_surrogate_test(
                lambda shuffled_source, fixed_target: granger_direction_score(
                    pd.Series(fixed_target),
                    pd.Series(shuffled_source),
                    maxlag=maxlag,
                ),
                source.values,
                target.values,
                n_surrogates=config.n_surrogates,
                method=method,
                seed=config.surrogate_seed,
            )
            print_surrogate_summary("Granger", f"{direction_label} (maxlag={maxlag})", result, verbose=True)

    _run_direction(f"{label_one} -> {label_two}", series_two, series_one)
    _run_direction(f"{label_two} -> {label_one}", series_one, series_two)


def _best_ccm_params_from_summary(summary: dict[str, object], fallback_embed_dim: int, fallback_lag: int) -> tuple[int, int]:
    candidates: list[tuple[float, int, int]] = []

    one_to_two = summary.get("one_to_two")
    if isinstance(one_to_two, dict) and "one_two" in one_to_two:
        candidates.append(
            (
                float(one_to_two["one_two"]),
                int(one_to_two["embed_dim"]),
                int(one_to_two["lag"]),
            )
        )

    two_to_one = summary.get("two_to_one")
    if isinstance(two_to_one, dict) and "two_one" in two_to_one:
        candidates.append(
            (
                float(two_to_one["two_one"]),
                int(two_to_one["embed_dim"]),
                int(two_to_one["lag"]),
            )
        )

    if not candidates:
        return max(2, fallback_embed_dim), max(1, fallback_lag)

    _, best_embed_dim, best_lag = max(candidates, key=lambda item: item[0])
    return best_embed_dim, best_lag


def run_analysis_pipeline(
    series_one: pd.Series,
    series_two: pd.Series,
    label_one: str,
    label_two: str,
    preprocessing_config: PreprocessingConfig | None = None,
    analysis_config: AnalysisConfig | None = None,
) -> dict[str, object]:
    """Run the configured analysis pipeline and emit text reports without extra prompts."""
    config = analysis_config or AnalysisConfig()

    preprocessing_result = preprocess_series_pair(
        series_one,
        series_two,
        label_one,
        label_two,
        preprocessing_config or PreprocessingConfig(),
    )

    processed_one = preprocessing_result.left
    processed_two = preprocessing_result.right
    processed_label_one = _display_series_label(preprocessing_result.left_label)
    processed_label_two = _display_series_label(preprocessing_result.right_label)

    print(f"Downstream analyses will use {processed_label_one} and {processed_label_two}.")

    stationarity_one: dict[str, object] = {}
    stationarity_two: dict[str, object] = {}

    if config.run_adf_check:
        print("Doing ADF test...")
        if config.auto_difference_if_nonstationary:
            processed_one, stationarity_one = make_series_stationary(
                processed_one, processed_label_one,
                max_diff_order=config.max_diff_order, verbose=True,
            )
            processed_two, stationarity_two = make_series_stationary(
                processed_two, processed_label_two,
                max_diff_order=config.max_diff_order, verbose=True,
            )
            if stationarity_one.get("diff_order", 0) == 0 and stationarity_two.get("diff_order", 0) == 0:
                print("Both series are already stationary; no differencing was needed.")
        else:
            stationarity_one = adf_unit_root_test(processed_one, processed_label_one)
            stationarity_two = adf_unit_root_test(processed_two, processed_label_two)
            print_adf_summary(stationarity_one, alpha=0.05, verbose=True)
            print_adf_summary(stationarity_two, alpha=0.05, verbose=True)
            if not stationarity_one.get("stationary") or not stationarity_two.get("stationary"):
                print("Some series remain non-stationary. Set auto_difference_if_nonstationary=True to difference automatically.")

    lcc_frame = None
    if config.run_lagged_cross_correlation:
        lcc_frame = lagged_cross_correlation(processed_one, processed_two, max_lag=config.lcc_max_lag)
        valid = lcc_frame["correlation"].dropna()
        if valid.empty:
            print(f"No valid LCC values were produced for {processed_label_one} vs {processed_label_two}.")
        else:
            best_row = lcc_frame.loc[valid.abs().idxmax()]
            print(f"Peak absolute LCC at lag {int(best_row['lag'])}: {best_row['correlation']:.6f}")
            if config.plot_lagged_cross_correlation:
                plot_line_trend(
                    lcc_frame, "lag", ["correlation"],
                    f"Lagged cross-correlation: {processed_label_one} vs {processed_label_two}",
                    "Lag", "Pearson correlation",
                )

    dtw_alignment = None
    warped_one_to_two = None
    warped_two_to_one = None
    downstream_one = processed_one
    downstream_two = processed_two
    alignment_mode = ALIGNMENT_MODE_COMMON_INDEX

    if config.run_dtw:
        print("DTW is computed on the processed series shown above.")
        dtw_alignment = compute_dtw_sequence(processed_one, processed_two)
        print(f"DTW distance: {dtw_alignment.distance:.6f}")
        print(f"DTW normalized distance: {dtw_alignment.normalizedDistance:.6f}")

        alignment_mode = config.alignment_mode
        if alignment_mode == ALIGNMENT_MODE_DTW_WARPED:
            print("Using DTW-warped alignment for Granger/TE/CCM.")
            warped_one_to_two, warped_two_to_one = _prepare_dtw_warped_series(processed_one, processed_two, dtw_alignment)
            downstream_two = warped_two_to_one
        else:
            print("Using common-index alignment for Granger/TE/CCM.")
            try:
                downstream_one, downstream_two = _align_by_common_index(processed_one, processed_two)
            except ValueError:
                print("No overlapping index values were available; continuing without alignment.")
                downstream_one = processed_one
                downstream_two = processed_two

        if config.plot_dtw_alignment:
            plot_dtw_alignment(processed_one, processed_two, dtw_alignment, processed_label_one, processed_label_two)

    granger_report = None
    te_summary: dict[str, object] = {"one_to_two": None, "two_to_one": None}
    ccm_summary: dict[str, object] = {"one_to_two": None, "two_to_one": None}

    if config.run_granger:
        granger_report = run_granger_causality_report(
            downstream_one, downstream_two, processed_label_one, processed_label_two,
            maxlag=config.granger_maxlag, verbose=True,
        )
        if config.run_granger_surrogates:
            _run_granger_surrogates(
                downstream_one, downstream_two, processed_label_one, processed_label_two,
                config.granger_maxlag, config,
            )

    te_grid = None
    if config.run_te:
        te_grid = sweep_transfer_entropy(
            downstream_one.values, downstream_two.values,
            range(2, max(2, config.te_max_embed_dim) + 1),
            range(1, max(1, config.te_max_lag) + 1),
        )
        te_summary = print_parameter_sweep_report(
            te_grid, "transfer entropy", processed_label_one, processed_label_two, verbose=True,
        )
        if config.run_te_surrogates:
            _run_metric_surrogates(
                "TE", downstream_one, downstream_two, processed_label_one, processed_label_two,
                te_summary, config,
            )

    ccm_grid = None
    ccm_convergence = None
    if config.run_ccm:
        ccm_grid = sweep_ccm(
            downstream_one.values, downstream_two.values,
            range(2, max(2, config.ccm_max_embed_dim) + 1),
            range(1, max(1, config.ccm_max_lag) + 1),
        )
        ccm_summary = print_parameter_sweep_report(
            ccm_grid, "ccm", processed_label_one, processed_label_two, verbose=True,
        )

        if config.run_ccm_convergence:
            best_embed_dim, best_lag = _best_ccm_params_from_summary(
                ccm_summary,
                fallback_embed_dim=config.ccm_max_embed_dim,
                fallback_lag=config.ccm_max_lag,
            )
            ccm_convergence = sweep_ccm_convergence_steps(
                downstream_one.values, downstream_two.values,
                embed_dim=best_embed_dim, lag=best_lag,
                library_step=config.ccm_library_step,
            )
            plot_ccm_convergence(
                ccm_convergence,
                title=f"CCM convergence: {processed_label_one} vs {processed_label_two}",
                label_one_to_two=f"{processed_label_one} -> {processed_label_two}",
                label_two_to_one=f"{processed_label_two} -> {processed_label_one}",
            )

        if config.run_ccm_surrogates:
            _run_metric_surrogates(
                "CCM", downstream_one, downstream_two, processed_label_one, processed_label_two,
                ccm_summary, config,
            )

    print("All done.")
    return {
        "preprocessing": preprocessing_result,
        "stationarity": {"left": stationarity_one, "right": stationarity_two},
        "lagged_cross_correlation": lcc_frame,
        "dtw": dtw_alignment,
        "alignment_mode": alignment_mode,
        "warped_one_to_two": warped_one_to_two,
        "warped_two_to_one": warped_two_to_one,
        "granger": granger_report,
        "te_grid": te_grid,
        "te_summary": te_summary,
        "ccm_grid": ccm_grid,
        "ccm_summary": ccm_summary,
        "ccm_convergence": ccm_convergence,
    }
