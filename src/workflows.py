"""Interactive workflows built on the reusable toolkit primitives."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import yfinance as yf

from .causal_analysis import (
    compute_ccm,
    compute_dtw,
    compute_te,
    extract_warping_path,
    granger_direction_score,
    print_parameter_sweep_report,
    run_granger_causality_report,
    sweep_ccm_convergence_steps,
    sweep_ccm_grid,
    sweep_transfer_entropy,
    warp_series_to_match,
)
from .data_loader import LoadedSeriesPair, download_yfinance_series, get_ticker_name, load_two_series_from_csv
from .plotting import plot_ccm_convergence, plot_dtw_alignment, plot_line_trend, plot_preprocessing_results, plot_single_series
from .preprocessing import PreprocessingConfig, PreprocessingResult, lagged_cross_correlation, preprocess_series_pair
from .stationarity import adf_unit_root_test, make_series_stationary, print_adf_summary
from .surrogate import print_surrogate_summary, run_surrogate_test


ALIGNMENT_MODE_COMMON_INDEX = "common_index"
ALIGNMENT_MODE_DTW_WARPED = "dtw_warped"


@dataclass(frozen=True)
class AnalysisConfig:
    """Configuration for the main causal-analysis pipeline.

    Attributes:
        run_lagged_cross_correlation: Whether to compute lagged cross-correlation.
        plot_lagged_cross_correlation: Whether to plot the LCC frame.
        run_dtw: Whether to compute DTW.
        plot_dtw_alignment: Whether to plot the DTW alignment.
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

    run_lagged_cross_correlation: bool = False
    plot_lagged_cross_correlation: bool = False
    run_dtw: bool = True
    plot_dtw_alignment: bool = True
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


@dataclass(frozen=True)
class PipelineConfig:
    """Bundle preprocessing and analysis configuration for notebook or CLI demos."""

    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    analysis: AnalysisConfig = field(default_factory=AnalysisConfig)


def _prompt_text(question: str, default: str | None = None) -> str:
    suffix = f" [{default}]" if default is not None else ""
    answer = input(f"{question}{suffix}: ").strip()
    if answer:
        return answer
    if default is not None:
        return default
    return ""


def _prompt_bool(question: str, default: bool = False) -> bool:
    suffix = "[Y/n]" if default else "[y/N]"
    answer = input(f"{question} {suffix}: ").strip().lower()
    if not answer:
        return default
    return answer in {"y", "yes"}


def _prompt_choice(question: str, choices: list[str], default_index: int = 0) -> int:
    print(question)
    for index, choice in enumerate(choices, start=1):
        default_suffix = " (default)" if index - 1 == default_index else ""
        print(f"  {index}. {choice}{default_suffix}")
    answer = input(f"Select 1-{len(choices)} [{default_index + 1}]: ").strip()
    if not answer:
        return default_index
    try:
        selected = int(answer) - 1
    except ValueError:
        return default_index
    return selected if 0 <= selected < len(choices) else default_index


def _prompt_alignment_mode() -> str:
    choice = _prompt_choice(
        "Choose how to align the series for Granger, TE, and CCM after DTW",
        [
            "Use common index for Granger/TE/CCM (standard fixed-lag analysis).",
            "Use DTW warping path to warp one series to the other, then run Granger/TE/CCM on the warped data (variable-lag style).",
        ],
        default_index=0,
    )
    return ALIGNMENT_MODE_COMMON_INDEX if choice == 0 else ALIGNMENT_MODE_DTW_WARPED


def _prompt_surrogate_method() -> str:
    choice = _prompt_choice(
        "Choose the surrogate method",
        ["shuffle", "bootstrap", "both"],
        default_index=2,
    )
    return ["shuffle", "bootstrap", "both"][choice]


def _display_series_label(label: str) -> str:
    if " (" in label and label.endswith(")"):
        return label.rsplit(" (", 1)[0]
    return label


def _prompt_int(question: str, default: int) -> int:
    answer = input(f"{question} [{default}]: ").strip()
    if not answer:
        return default
    try:
        return int(answer)
    except ValueError:
        print(f"Invalid input. Using default {default}.")
        return default


def _is_valid_ticker(ticker: str) -> bool:
    try:
        frame = yf.download(ticker, period="5d", progress=False, auto_adjust=False)
    except Exception:
        return False
    return not frame.empty


def _prompt_nonempty(prompt: str, default: str | None = None) -> str:
    while True:
        suffix = f" [{default}]" if default else ""
        answer = input(f"{prompt}{suffix}: ").strip()
        if answer:
            return answer
        if default is not None:
            return default
        print("Input cannot be empty. Please try again.")


def _prompt_date(text: str, default: str) -> str:
    answer = input(f"{text} [{default}]: ").strip()
    return answer or default


def _prompt_date_range() -> tuple[str, str]:
    print("Specify analysis date range. Press Enter to use default (2015-01-01 to today).")
    while True:
        start_text = _prompt_date("Start date (YYYY-MM-DD)", "2015-01-01")
        end_text = _prompt_date("End date (YYYY-MM-DD)", pd.Timestamp.today().strftime("%Y-%m-%d"))
        start_ts = pd.to_datetime(start_text, errors="coerce")
        end_ts = pd.to_datetime(end_text, errors="coerce")
        if pd.isna(start_ts):
            print(f"Invalid start date '{start_text}'. Please try again.")
            continue
        if pd.isna(end_ts):
            print(f"Invalid end date '{end_text}'. Please try again.")
            continue
        if start_ts > end_ts:
            print("Start date is after end date. Please enter them again in chronological order.")
            continue
        return start_ts.strftime("%Y-%m-%d"), end_ts.strftime("%Y-%m-%d")


def _prompt_ticker_pair() -> tuple[str, str]:
    print("Enter two ticker symbols to analyze. Examples: ^IXIC, ^GSPC, AAPL, MSFT.")
    while True:
        ticker_one = _prompt_nonempty("First ticker")
        ticker_two = _prompt_nonempty("Second ticker")
        if ticker_one == ticker_two:
            print("The two tickers must be different. Please enter two distinct symbols.")
            continue
        if not _is_valid_ticker(ticker_one):
            print(f"Ticker '{ticker_one}' could not be validated from Yahoo Finance. Please try again.")
            continue
        if not _is_valid_ticker(ticker_two):
            print(f"Ticker '{ticker_two}' could not be validated from Yahoo Finance. Please try again.")
            continue
        return ticker_one, ticker_two


def _prompt_optional_series_names(default_one: str, default_two: str) -> tuple[str, str]:
    name_one = input(f"Optional display name for first series [{default_one}]: ").strip() or default_one
    name_two = input(f"Optional display name for second series [{default_two}]: ").strip() or default_two
    return name_one, name_two


def _prompt_data_source() -> int:
    return _prompt_choice(
        "Choose a data source",
        ["Yahoo Finance tickers", "CSV files with your own data"],
        default_index=0,
    )


def _prompt_preprocessing_config() -> PreprocessingConfig:
    base_choice = _prompt_choice(
        "Choose the base representation for both series",
        ["raw values", "percentage changes", "log changes"],
        default_index=2,
    )
    base_representation = ["raw", "returns", "log_returns"][base_choice]

    smoothing_window: int | None = None
    if _prompt_bool("Apply smoothing before the causal analysis?", default=False):
        smoothing_window = _prompt_int("Smoothing window size", 5)

    downsample_step: int | None = None
    downsample_freq: str | None = None
    if _prompt_bool("Apply downsampling before the causal analysis?", default=False):
        downsample_mode = _prompt_choice(
            "Choose a downsampling mode",
            ["every Nth observation", "calendar frequency alias (for datetime indexes)"],
            default_index=0,
        )
        if downsample_mode == 0:
            downsample_step = _prompt_int("Keep every Nth observation", 2)
        else:
            downsample_freq = _prompt_text("Enter pandas frequency alias (for example, W, M, or Q)", "W")

    standardize = _prompt_bool("Apply z-score standardization (subtract mean, divide by standard deviation)?", default=False)
    return PreprocessingConfig(
        base_representation=base_representation,
        smoothing_window=smoothing_window,
        downsample_step=downsample_step,
        downsample_freq=downsample_freq,
        standardize=standardize,
    )


def _prompt_analysis_config() -> AnalysisConfig:
    run_lcc = _prompt_bool("Run lagged cross-correlation (LCC) on the processed series?", default=False)
    plot_lcc = run_lcc and _prompt_bool("Plot the lagged cross-correlation chart?", default=True)

    run_dtw = _prompt_bool("Run DTW analysis?", default=True)
    plot_dtw = run_dtw and _prompt_bool("Plot the DTW alignment graph?", default=True)

    run_granger = _prompt_bool("Run Granger causality analysis?", default=True)
    granger_maxlag = _prompt_int("Maximum lag for Granger causality", 5) if run_granger else 5
    run_granger_surrogates = run_granger and _prompt_bool("Run surrogate tests for Granger causality?", default=True)

    run_te = _prompt_bool("Run TE sweep?", default=True)
    te_max_lag = _prompt_int("Maximum lag for TE", 5) if run_te else 5
    te_max_embed_dim = _prompt_int("Maximum embedding dimension for TE", 3) if run_te else 3
    run_te_surrogates = run_te and _prompt_bool("Run surrogate tests for TE now?", default=True)

    run_ccm = _prompt_bool("Run CCM sweep?", default=True)
    ccm_max_lag = _prompt_int("Maximum lag for CCM", 5) if run_ccm else 5
    ccm_max_embed_dim = _prompt_int("Maximum embedding dimension for CCM", 3) if run_ccm else 3
    run_ccm_convergence = run_ccm and _prompt_bool("Run CCM convergence analysis?", default=True)
    ccm_library_step = 0.1
    if run_ccm_convergence:
        ccm_library_step = float(_prompt_text("CCM library-size step", "0.1") or "0.1")
    run_ccm_surrogates = run_ccm and _prompt_bool("Run surrogate tests for CCM now?", default=True)

    surrogate_method = _prompt_surrogate_method()
    n_surrogates = _prompt_int("Number of surrogate samples", 200)
    seed_text = _prompt_text("Random seed for surrogate tests", "0")
    try:
        surrogate_seed = int(seed_text)
    except ValueError:
        surrogate_seed = 0

    return AnalysisConfig(
        run_lagged_cross_correlation=run_lcc,
        plot_lagged_cross_correlation=plot_lcc,
        run_dtw=run_dtw,
        plot_dtw_alignment=plot_dtw,
        run_granger=run_granger,
        granger_maxlag=granger_maxlag,
        run_granger_surrogates=run_granger_surrogates,
        run_te=run_te,
        te_max_lag=te_max_lag,
        te_max_embed_dim=te_max_embed_dim,
        run_te_surrogates=run_te_surrogates,
        run_ccm=run_ccm,
        ccm_max_lag=ccm_max_lag,
        ccm_max_embed_dim=ccm_max_embed_dim,
        run_ccm_convergence=run_ccm_convergence,
        ccm_library_step=ccm_library_step,
        run_ccm_surrogates=run_ccm_surrogates,
        surrogate_method=surrogate_method,
        n_surrogates=n_surrogates,
        surrogate_seed=surrogate_seed,
    )


def _load_yfinance_pair() -> LoadedSeriesPair:
    ticker_one, ticker_two = _prompt_ticker_pair()
    start, end = _prompt_date_range()
    default_one = get_ticker_name(ticker_one)
    default_two = get_ticker_name(ticker_two)
    name_one, name_two = _prompt_optional_series_names(default_one, default_two)
    loaded = download_yfinance_series(ticker_one, ticker_two, start, end, name_one=name_one, name_two=name_two)
    print(f"Loaded {len(loaded.left)} aligned observations for {loaded.left_name} and {loaded.right_name}.")
    return loaded


def _load_csv_pair() -> LoadedSeriesPair:
    left_path = Path(_prompt_text("Path to the first CSV file"))
    right_path = Path(_prompt_text("Path to the second CSV file"))
    left_value_column = _prompt_text("Value column for the first CSV")
    right_value_column = _prompt_text("Value column for the second CSV")
    left_index_column = _prompt_text("Optional time/index column for the first CSV (leave blank to use row order)", "") or None
    right_index_column = _prompt_text("Optional time/index column for the second CSV (leave blank to use row order)", "") or None
    default_left_name = left_value_column or "Series 1"
    default_right_name = right_value_column or "Series 2"
    left_name = _prompt_text("Display name for the first series", default_left_name)
    right_name = _prompt_text("Display name for the second series", default_right_name)
    loaded = load_two_series_from_csv(
        left_path,
        right_path,
        left_value_column,
        right_value_column,
        left_name=left_name,
        right_name=right_name,
        left_index_column=left_index_column,
        right_index_column=right_index_column,
    )
    print(f"Loaded {len(loaded.left)} observations for {loaded.left_name} and {loaded.right_name}.")
    return loaded


def _resolve_surrogate_methods(choice: str) -> list[str]:
    if choice == "shuffle":
        return ["shuffle"]
    if choice == "bootstrap":
        return ["bootstrap"]
    return ["shuffle", "bootstrap"]


def run_preprocessing_flow(
    series_one: pd.Series,
    series_two: pd.Series,
    label_one: str,
    label_two: str,
    config: PreprocessingConfig | None = None,
) -> tuple[pd.Series, pd.Series, dict[str, object], dict[str, object]]:
    """Interactively choose preprocessing steps for two series.

    The function returns the same four-item tuple as the historical API so the
    rest of the toolkit can keep working unchanged.
    """
    if _prompt_bool(f"Plot the raw series for {label_one} and {label_two} on separate charts?", default=True):
        plot_single_series(series_one, f"Raw series: {label_one}", label_one)
        plot_single_series(series_two, f"Raw series: {label_two}", label_two)

    effective_config = config or _prompt_preprocessing_config()
    result = preprocess_series_pair(series_one, series_two, label_one, label_two, effective_config)

    if _prompt_bool("Plot the preprocessed series comparison now?", default=True):
        base_config = PreprocessingConfig(base_representation=effective_config.base_representation)
        base_result = preprocess_series_pair(series_one, series_two, label_one, label_two, base_config)
        plot_preprocessing_results(
            base_result.left,
            result.left,
            f"Preprocessing comparison: {label_one}",
            base_result.left_label,
            result.left_label,
        )
        plot_preprocessing_results(
            base_result.right,
            result.right,
            f"Preprocessing comparison: {label_two}",
            base_result.right_label,
            result.right_label,
        )

    return result.left, result.right, result.summary_left, result.summary_right


def prompt_lagged_cross_correlation(
    series_one: pd.Series,
    series_two: pd.Series,
    label_one: str,
    label_two: str,
    max_lag: int = 30,
) -> pd.DataFrame | None:
    """Ask whether to plot lagged cross-correlation and return the computed frame."""
    answer = input(f"Do you want to plot lagged cross-correlation (LCC) for {label_one} vs {label_two}? [y/N]: ").strip().lower()
    if answer not in {"y", "yes"}:
        return None

    frame = lagged_cross_correlation(series_one, series_two, max_lag=max_lag)
    valid = frame["correlation"].dropna()
    if valid.empty:
        print(f"No valid LCC values were produced for {label_one} vs {label_two}.")
        return frame

    best_index = valid.abs().idxmax()
    best_row = frame.loc[best_index]
    print(f"Peak absolute LCC at lag {int(best_row['lag'])}: {best_row['correlation']:.6f}")
    plot_line_trend(
        frame,
        "lag",
        ["correlation"],
        f"Lagged cross-correlation: {label_one} vs {label_two}",
        "Lag",
        "Pearson correlation",
    )
    return frame


def _run_stationarity_check(
    series_one: pd.Series,
    series_two: pd.Series,
    label_one: str,
    label_two: str,
) -> tuple[pd.Series, pd.Series, str, str]:
    result_one = adf_unit_root_test(series_one, label_one)
    result_two = adf_unit_root_test(series_two, label_two)
    print_adf_summary(result_one, alpha=0.05, verbose=True)
    print_adf_summary(result_two, alpha=0.05, verbose=True)

    if not _prompt_bool("Difference non-stationary series until the ADF test accepts stationarity?", default=False):
        return series_one, series_two, label_one, label_two

    if bool(result_one.get("stationary")) and bool(result_two.get("stationary")):
        print("Both series are already stationary; no differencing was needed.")
        return series_one, series_two, label_one, label_two

    stationary_one = series_one
    stationary_two = series_two
    updated_label_one = label_one
    updated_label_two = label_two

    if not bool(result_one.get("stationary")):
        stationary_one, info_one = make_series_stationary(series_one, label_one, verbose=True)
        updated_label_one = f"{label_one} (stationary, diff order {info_one['diff_order']})"
    if not bool(result_two.get("stationary")):
        stationary_two, info_two = make_series_stationary(series_two, label_two, verbose=True)
        updated_label_two = f"{label_two} (stationary, diff order {info_two['diff_order']})"

    if bool(result_one.get("stationary")) or bool(result_two.get("stationary")):
        print("Only the non-stationary series were differenced.")

    common_index = stationary_one.index.intersection(stationary_two.index)
    return stationary_one.loc[common_index], stationary_two.loc[common_index], updated_label_one, updated_label_two


def _prepare_dtw_warped_series(
    series_one: pd.Series,
    series_two: pd.Series,
    dtw_alignment,
) -> tuple[pd.Series, pd.Series]:
    path_one, path_two = extract_warping_path(dtw_alignment)
    warped_one_to_two = warp_series_to_match(series_one, series_two, dtw_alignment)
    reverse_alignment = SimpleNamespace(index1=path_two, index2=path_one)
    warped_two_to_one = warp_series_to_match(series_two, series_one, reverse_alignment)
    return warped_one_to_two, warped_two_to_one


def _run_stepwise_interactive_analysis(
    series_one: pd.Series,
    series_two: pd.Series,
    label_one: str,
    label_two: str,
    preprocessing_config: PreprocessingConfig | None = None,
) -> dict[str, object]:
    preprocessing_result = run_preprocessing_flow(series_one, series_two, label_one, label_two, config=preprocessing_config)

    processed_one = preprocessing_result[0]
    processed_two = preprocessing_result[1]
    processed_label_one = _display_series_label(str(preprocessing_result[2].get("after_label", label_one)))
    processed_label_two = _display_series_label(str(preprocessing_result[3].get("after_label", label_two)))
    print(f"Downstream analyses will use {processed_label_one} and {processed_label_two}.")

    if not _prompt_bool("Do the ADF stationarity test now?", default=True):
        print("Skipping ADF test for now.")
        stationarity_one = adf_unit_root_test(processed_one, processed_label_one)
        stationarity_two = adf_unit_root_test(processed_two, processed_label_two)
    else:
        print("Doing ADF test...")
        processed_one, processed_two, processed_label_one, processed_label_two = _run_stationarity_check(
            processed_one,
            processed_two,
            processed_label_one,
            processed_label_two,
        )
        processed_label_one = _display_series_label(processed_label_one)
        processed_label_two = _display_series_label(processed_label_two)

        stationarity_one = adf_unit_root_test(processed_one, processed_label_one)
        stationarity_two = adf_unit_root_test(processed_two, processed_label_two)

    if not bool(stationarity_one.get("stationary")) or not bool(stationarity_two.get("stationary")):
        print("Some series remain non-stationary after preprocessing. The pipeline will continue with the transformed data.")

    if _prompt_bool("Run lagged cross-correlation (LCC) on the processed series?", default=False):
        lcc_frame = lagged_cross_correlation(processed_one, processed_two, max_lag=30)
        valid = lcc_frame["correlation"].dropna()
        if valid.empty:
            print(f"No valid LCC values were produced for {processed_label_one} vs {processed_label_two}.")
        else:
            best_index = valid.abs().idxmax()
            best_row = lcc_frame.loc[best_index]
            print(f"Peak absolute LCC at lag {int(best_row['lag'])}: {best_row['correlation']:.6f}")
            if _prompt_bool("Plot the lagged cross-correlation chart?", default=True):
                plot_line_trend(
                    lcc_frame,
                    "lag",
                    ["correlation"],
                    f"Lagged cross-correlation: {processed_label_one} vs {processed_label_two}",
                    "Lag",
                    "Pearson correlation",
                )
    else:
        lcc_frame = None

    dtw_alignment = None
    warped_one_to_two = None
    warped_two_to_one = None
    downstream_one = processed_one
    downstream_two = processed_two
    if _prompt_bool("Run DTW analysis?", default=True):
        print("DTW is computed on the processed series shown above.")
        dtw_alignment = compute_dtw(processed_one, processed_two)
        print(f"DTW distance: {dtw_alignment.distance:.6f}")
        print(f"DTW normalized distance: {dtw_alignment.normalizedDistance:.6f}")
        alignment_mode = _prompt_alignment_mode()
        if alignment_mode == ALIGNMENT_MODE_DTW_WARPED:
            print("Using DTW-warped alignment for Granger/TE/CCM.")
            warped_one_to_two, warped_two_to_one = _prepare_dtw_warped_series(processed_one, processed_two, dtw_alignment)
            downstream_two = warped_two_to_one
        else:
            print("Using common-index alignment for Granger/TE/CCM.")
        if _prompt_bool("Plot the DTW alignment graph?", default=True):
            plot_dtw_alignment(processed_one, processed_two, dtw_alignment, processed_label_one, processed_label_two)

    granger_report = None
    if _prompt_bool("Run Granger causality analysis?", default=True):
        granger_maxlag = _prompt_int("Maximum lag for Granger causality", 5)
        granger_report = run_granger_causality_report(
            downstream_one,
            downstream_two,
            processed_label_one,
            processed_label_two,
            maxlag=granger_maxlag,
            verbose=True,
        )
        if _prompt_bool("Run surrogate tests for Granger causality?", default=True):
            surrogate_method = _prompt_surrogate_method()
            _run_granger_surrogates(
                downstream_one,
                downstream_two,
                processed_label_one,
                processed_label_two,
                granger_maxlag,
                AnalysisConfig(surrogate_method=surrogate_method),
            )

    te_summary: dict[str, object] = {"one_to_two": None, "two_to_one": None}
    ccm_summary: dict[str, object] = {"one_to_two": None, "two_to_one": None}

    if _prompt_bool("Run TE sweep?", default=True):
        te_max_lag = _prompt_int("Maximum lag for TE", 5)
        te_max_embed_dim = _prompt_int("Maximum embedding dimension for TE", 3)
        te_grid = sweep_transfer_entropy(
            downstream_one.values,
            downstream_two.values,
            range(2, max(2, te_max_embed_dim) + 1),
            range(1, max(1, te_max_lag) + 1),
        )
        te_summary = print_parameter_sweep_report(te_grid, "transfer entropy", processed_label_one, processed_label_two, verbose=True)
        if _prompt_bool("Run surrogate tests for TE now?", default=True):
            surrogate_method = _prompt_surrogate_method()
            _run_metric_surrogates(
                "TE",
                downstream_one,
                downstream_two,
                processed_label_one,
                processed_label_two,
                te_summary,
                AnalysisConfig(surrogate_method=surrogate_method),
            )
    else:
        te_grid = None

    if _prompt_bool("Run CCM sweep?", default=True):
        ccm_max_lag = _prompt_int("Maximum lag for CCM", 5)
        ccm_max_embed_dim = _prompt_int("Maximum embedding dimension for CCM", 3)
        run_ccm_convergence = _prompt_bool("Run CCM convergence analysis?", default=True)
        ccm_library_step = 0.1
        if run_ccm_convergence:
            ccm_library_step = float(_prompt_text("CCM library-size step", "0.1") or "0.1")

        ccm_grid = sweep_ccm_grid(
            downstream_one.values,
            downstream_two.values,
            range(2, max(2, ccm_max_embed_dim) + 1),
            range(1, max(1, ccm_max_lag) + 1),
        )
        ccm_summary = print_parameter_sweep_report(ccm_grid, "ccm", processed_label_one, processed_label_two, verbose=True)
        if run_ccm_convergence:
            ccm_convergence = sweep_ccm_convergence_steps(
                downstream_one.values,
                downstream_two.values,
                embed_dim=max(2, ccm_max_embed_dim),
                lag=max(1, ccm_max_lag),
                library_step=ccm_library_step,
            )
            plot_ccm_convergence(
                ccm_convergence,
                title=f"CCM convergence: {processed_label_one} vs {processed_label_two}",
                label_one_to_two=f"{processed_label_one} -> {processed_label_two}",
                label_two_to_one=f"{processed_label_two} -> {processed_label_one}",
            )
        else:
            ccm_convergence = None
        if _prompt_bool("Run surrogate tests for CCM now?", default=True):
            surrogate_method = _prompt_surrogate_method()
            _run_metric_surrogates(
                "CCM",
                downstream_one,
                downstream_two,
                processed_label_one,
                processed_label_two,
                ccm_summary,
                AnalysisConfig(surrogate_method=surrogate_method),
            )
    else:
        ccm_grid = None
        ccm_convergence = None

    print("All done.")
    return {
        "preprocessing": preprocessing_result,
        "stationarity": {"left": stationarity_one, "right": stationarity_two},
        "lagged_cross_correlation": lcc_frame,
        "dtw": dtw_alignment,
        "alignment_mode": ALIGNMENT_MODE_DTW_WARPED if warped_one_to_two is not None else ALIGNMENT_MODE_COMMON_INDEX,
        "warped_one_to_two": warped_one_to_two,
        "warped_two_to_one": warped_two_to_one,
        "granger": granger_report,
        "te_grid": te_grid,
        "te_summary": te_summary,
        "ccm_grid": ccm_grid,
        "ccm_summary": ccm_summary,
        "ccm_convergence": ccm_convergence,
    }


def _run_noninteractive_analysis(
    series_one: pd.Series,
    series_two: pd.Series,
    label_one: str,
    label_two: str,
    preprocessing_config: PreprocessingConfig | None = None,
    analysis_config: AnalysisConfig | None = None,
    alignment_mode: str = ALIGNMENT_MODE_COMMON_INDEX,
) -> dict[str, object]:
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
    config = analysis_config or AnalysisConfig()

    stationarity_one = adf_unit_root_test(processed_one, processed_label_one)
    stationarity_two = adf_unit_root_test(processed_two, processed_label_two)

    stationarity_result = {
        "left": stationarity_one,
        "right": stationarity_two,
        "stationary_series": (processed_one, processed_two, processed_label_one, processed_label_two),
    }

    lcc_frame = None
    if config.run_lagged_cross_correlation:
        lcc_frame = lagged_cross_correlation(processed_one, processed_two, max_lag=30)

    dtw_alignment = compute_dtw(processed_one, processed_two) if config.run_dtw else None
    downstream_one = processed_one
    downstream_two = processed_two
    warped_one_to_two = None
    warped_two_to_one = None
    if dtw_alignment is not None and alignment_mode == ALIGNMENT_MODE_DTW_WARPED:
        warped_one_to_two, warped_two_to_one = _prepare_dtw_warped_series(processed_one, processed_two, dtw_alignment)
        downstream_two = warped_two_to_one

    granger_report = run_granger_causality_report(downstream_one, downstream_two, processed_label_one, processed_label_two, maxlag=config.granger_maxlag) if config.run_granger else None
    te_grid = sweep_transfer_entropy(downstream_one.values, downstream_two.values, range(2, max(2, config.te_max_embed_dim) + 1), range(1, max(1, config.te_max_lag) + 1)) if config.run_te else None
    ccm_grid = sweep_ccm_grid(downstream_one.values, downstream_two.values, range(2, max(2, config.ccm_max_embed_dim) + 1), range(1, max(1, config.ccm_max_lag) + 1)) if config.run_ccm else None
    ccm_convergence = sweep_ccm_convergence_steps(
        downstream_one.values,
        downstream_two.values,
        embed_dim=max(2, config.ccm_max_embed_dim),
        lag=max(1, config.ccm_max_lag),
        library_step=config.ccm_library_step,
    ) if config.run_ccm and config.run_ccm_convergence else None

    return {
        "preprocessing": preprocessing_result,
        "stationarity": stationarity_result,
        "lagged_cross_correlation": lcc_frame,
        "dtw": dtw_alignment,
        "alignment_mode": alignment_mode,
        "warped_one_to_two": warped_one_to_two,
        "warped_two_to_one": warped_two_to_one,
        "granger": granger_report,
        "te_grid": te_grid,
        "ccm_grid": ccm_grid,
        "ccm_convergence": ccm_convergence,
    }


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

    def _run_metric(direction_label: str, embed_dim: int, lag: int) -> None:
        if metric_name == "TE":
            if direction_label == f"{label_one} -> {label_two}":
                real_func = lambda x, y: compute_te(x, y, embed_dim=embed_dim, lag=lag)
                base_x = series_one.values
                base_y = series_two.values
            else:
                real_func = lambda x, y: compute_te(x, y, embed_dim=embed_dim, lag=lag)
                base_x = series_two.values
                base_y = series_one.values
        else:
            if direction_label == f"{label_one} -> {label_two}":
                real_func = lambda x, y: compute_ccm(x, y, embed_dim=embed_dim, lag=lag)[0]
                base_x = series_one.values
                base_y = series_two.values
            else:
                real_func = lambda x, y: compute_ccm(x, y, embed_dim=embed_dim, lag=lag)[1]
                base_x = series_two.values
                base_y = series_one.values

        for method in methods:
            result = run_surrogate_test(
                real_func,
                base_x,
                base_y,
                n_surrogates=config.n_surrogates,
                method=method,
                seed=config.surrogate_seed,
            )
            print_surrogate_summary(metric_name, f"{direction_label} (embed_dim={embed_dim}, lag={lag})", result, verbose=True)

    one_to_two = summary.get("one_to_two")
    two_to_one = summary.get("two_to_one")
    if isinstance(one_to_two, dict):
        _run_metric(f"{label_one} -> {label_two}", int(one_to_two["embed_dim"]), int(one_to_two["lag"]))
    if isinstance(two_to_one, dict):
        _run_metric(f"{label_two} -> {label_one}", int(two_to_one["embed_dim"]), int(two_to_one["lag"]))


def _run_granger_surrogates(series_one: pd.Series, series_two: pd.Series, label_one: str, label_two: str, maxlag: int, config: AnalysisConfig) -> None:
    if not config.run_granger_surrogates:
        return

    methods = _resolve_surrogate_methods(config.surrogate_method)

    def _run_direction(direction_label: str, target: pd.Series, source: pd.Series) -> None:
        for method in methods:
            result = run_surrogate_test(
                lambda x, y: granger_direction_score(pd.Series(x), pd.Series(y), maxlag=maxlag),
                target.values,
                source.values,
                n_surrogates=config.n_surrogates,
                method=method,
                seed=config.surrogate_seed,
            )
            print_surrogate_summary("Granger", f"{direction_label} (maxlag={maxlag})", result, verbose=True)

    _run_direction(f"{label_one} -> {label_two}", series_two, series_one)
    _run_direction(f"{label_two} -> {label_one}", series_one, series_two)


def execute_pipeline(
    series_one: pd.Series,
    series_two: pd.Series,
    label_one: str,
    label_two: str,
    preprocessing_config: PreprocessingConfig | None = None,
    analysis_config: AnalysisConfig | None = None,
    alignment_mode: str = ALIGNMENT_MODE_COMMON_INDEX,
) -> dict[str, object]:
    """Run the full pipeline without prompting for user input.

    This is the notebook-friendly entry point used by the CLI wrapper after the
    interactive config has been collected.
    """
    return _run_noninteractive_analysis(
        series_one,
        series_two,
        label_one,
        label_two,
        preprocessing_config=preprocessing_config,
        analysis_config=analysis_config,
        alignment_mode=alignment_mode,
    )


def run_analysis_pipeline(
    series_one: pd.Series,
    series_two: pd.Series,
    label_one: str,
    label_two: str,
    preprocessing_config: PreprocessingConfig | None = None,
    analysis_config: AnalysisConfig | None = None,
) -> dict[str, object]:
    """Run the configured analysis pipeline and emit the familiar text reports."""
    config = analysis_config or AnalysisConfig()
    preprocessing_result = preprocess_series_pair(
        series_one,
        series_two,
        label_one,
        label_two,
        preprocessing_config or PreprocessingConfig(),
    )
    processed_label_one = _display_series_label(preprocessing_result.left_label)
    processed_label_two = _display_series_label(preprocessing_result.right_label)
    print(f"Downstream analyses will use {processed_label_one} and {processed_label_two}.")

    if _prompt_bool("Do the ADF stationarity test now?", default=True):
        print("Doing ADF test...")
    else:
        print("Skipping ADF test for now.")
        return _run_noninteractive_analysis(
            series_one,
            series_two,
            label_one,
            label_two,
            preprocessing_config=preprocessing_config,
            analysis_config=analysis_config,
            alignment_mode=ALIGNMENT_MODE_COMMON_INDEX,
        )

    processed_one = preprocessing_result.left
    processed_two = preprocessing_result.right
    processed_one, processed_two, processed_label_one, processed_label_two = _run_stationarity_check(
        processed_one,
        processed_two,
        processed_label_one,
        processed_label_two,
    )
    processed_label_one = _display_series_label(processed_label_one)
    processed_label_two = _display_series_label(processed_label_two)

    stationarity_one = adf_unit_root_test(processed_one, processed_label_one)
    stationarity_two = adf_unit_root_test(processed_two, processed_label_two)

    if not bool(stationarity_one.get("stationary")) or not bool(stationarity_two.get("stationary")):
        print("Some series remain non-stationary after preprocessing. The pipeline will continue with the transformed data.")

    lcc_frame = None
    if config.run_lagged_cross_correlation:
        lcc_frame = lagged_cross_correlation(processed_one, processed_two, max_lag=30)
        valid = lcc_frame["correlation"].dropna()
        if valid.empty:
            print(f"No valid LCC values were produced for {processed_label_one} vs {processed_label_two}.")
        else:
            best_index = valid.abs().idxmax()
            best_row = lcc_frame.loc[best_index]
            print(f"Peak absolute LCC at lag {int(best_row['lag'])}: {best_row['correlation']:.6f}")
            if config.plot_lagged_cross_correlation:
                plot_line_trend(
                    lcc_frame,
                    "lag",
                    ["correlation"],
                    f"Lagged cross-correlation: {processed_label_one} vs {processed_label_two}",
                    "Lag",
                    "Pearson correlation",
                )

    dtw_alignment = None
    warped_one_to_two = None
    warped_two_to_one = None
    downstream_one = processed_one
    downstream_two = processed_two
    alignment_mode = ALIGNMENT_MODE_COMMON_INDEX
    if config.run_dtw:
        print("DTW is computed on the processed series shown above.")
        dtw_alignment = compute_dtw(processed_one, processed_two)
        print(f"DTW distance: {dtw_alignment.distance:.6f}")
        print(f"DTW normalized distance: {dtw_alignment.normalizedDistance:.6f}")
        alignment_mode = _prompt_alignment_mode()
        if alignment_mode == ALIGNMENT_MODE_DTW_WARPED:
            print("Using DTW-warped alignment for Granger/TE/CCM.")
            warped_one_to_two, warped_two_to_one = _prepare_dtw_warped_series(processed_one, processed_two, dtw_alignment)
            downstream_two = warped_two_to_one
        else:
            print("Using common-index alignment for Granger/TE/CCM.")
        if config.plot_dtw_alignment:
            plot_dtw_alignment(processed_one, processed_two, dtw_alignment, processed_label_one, processed_label_two)

    granger_report = None
    te_summary: dict[str, object] = {"one_to_two": None, "two_to_one": None}
    ccm_summary: dict[str, object] = {"one_to_two": None, "two_to_one": None}

    if config.run_granger:
        granger_report = run_granger_causality_report(downstream_one, downstream_two, processed_label_one, processed_label_two, maxlag=config.granger_maxlag, verbose=True)
        if config.run_granger_surrogates:
            surrogate_method = _prompt_surrogate_method()
            _run_granger_surrogates(
                downstream_one,
                downstream_two,
                processed_label_one,
                processed_label_two,
                config.granger_maxlag,
                AnalysisConfig(surrogate_method=surrogate_method, n_surrogates=config.n_surrogates, surrogate_seed=config.surrogate_seed),
            )

    if config.run_te:
        te_grid = sweep_transfer_entropy(downstream_one.values, downstream_two.values, range(2, max(2, config.te_max_embed_dim) + 1), range(1, max(1, config.te_max_lag) + 1))
        te_summary = print_parameter_sweep_report(te_grid, "transfer entropy", processed_label_one, processed_label_two, verbose=True)
        if config.run_te_surrogates:
            surrogate_method = _prompt_surrogate_method()
            _run_metric_surrogates("TE", downstream_one, downstream_two, processed_label_one, processed_label_two, te_summary, AnalysisConfig(surrogate_method=surrogate_method, n_surrogates=config.n_surrogates, surrogate_seed=config.surrogate_seed))
    else:
        te_grid = None

    if config.run_ccm:
        ccm_grid = sweep_ccm_grid(downstream_one.values, downstream_two.values, range(2, max(2, config.ccm_max_embed_dim) + 1), range(1, max(1, config.ccm_max_lag) + 1))
        ccm_summary = print_parameter_sweep_report(ccm_grid, "ccm", processed_label_one, processed_label_two, verbose=True)
        if config.run_ccm_convergence:
            ccm_convergence = sweep_ccm_convergence_steps(
                downstream_one.values,
                downstream_two.values,
                embed_dim=max(2, config.ccm_max_embed_dim),
                lag=max(1, config.ccm_max_lag),
                library_step=config.ccm_library_step,
            )
            plot_ccm_convergence(
                ccm_convergence,
                title=f"CCM convergence: {processed_label_one} vs {processed_label_two}",
                label_one_to_two=f"{processed_label_one} -> {processed_label_two}",
                label_two_to_one=f"{processed_label_two} -> {processed_label_one}",
            )
        else:
            ccm_convergence = None
        if config.run_ccm_surrogates:
            surrogate_method = _prompt_surrogate_method()
            _run_metric_surrogates("CCM", downstream_one, downstream_two, processed_label_one, processed_label_two, ccm_summary, AnalysisConfig(surrogate_method=surrogate_method, n_surrogates=config.n_surrogates, surrogate_seed=config.surrogate_seed))
    else:
        ccm_grid = None
        ccm_convergence = None

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


def run_full_analysis_for_pair(
    ticker_one: str,
    ticker_two: str,
    config: PipelineConfig | None = None,
    start: str = "2018-01-01",
    end: str = "2024-12-31",
    name_one: str | None = None,
    name_two: str | None = None,
) -> dict[str, object]:
    """Run the full pipeline for a pair of Yahoo Finance tickers.

    This is the notebook-friendly convenience wrapper for quick demos.
    """
    loaded = download_yfinance_series(ticker_one, ticker_two, start, end, name_one=name_one, name_two=name_two)
    print(f"Loaded {len(loaded.left)} aligned observations for {loaded.left_name} and {loaded.right_name}.")
    pipeline_config = config or PipelineConfig()
    return execute_pipeline(
        loaded.left,
        loaded.right,
        loaded.left_name,
        loaded.right_name,
        preprocessing_config=pipeline_config.preprocessing,
        analysis_config=pipeline_config.analysis,
    )


def run_general_workflow() -> dict[str, object]:
    """Start the full interactive workflow and let the user choose the data source."""
    source_choice = _prompt_data_source()
    if source_choice == 0:
        loaded = _load_yfinance_pair()
    else:
        loaded = _load_csv_pair()

    preprocessing_config = _prompt_preprocessing_config()
    return _run_stepwise_interactive_analysis(
        loaded.left,
        loaded.right,
        loaded.left_name,
        loaded.right_name,
        preprocessing_config=preprocessing_config,
    )


def run_yfinance_demo(
    ticker_one: str = "^IXIC",
    ticker_two: str = "^GSPC",
    start: str = "2018-01-01",
    end: str = "2024-12-31",
    name_one: str | None = None,
    name_two: str | None = None,
) -> dict[str, object]:
    """Convenience demo for notebooks or quick experiments with Yahoo Finance data."""
    return run_full_analysis_for_pair(
        ticker_one=ticker_one,
        ticker_two=ticker_two,
        start=start,
        end=end,
        name_one=name_one,
        name_two=name_two,
    )
