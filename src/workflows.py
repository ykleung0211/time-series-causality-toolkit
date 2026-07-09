"""Interactive workflows built on the reusable toolkit primitives."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .causal_analysis import (
    compute_ccm,
    compute_dtw,
    compute_te,
    granger_direction_score,
    print_parameter_sweep_report,
    run_granger_causality_report,
    sweep_ccm_grid,
    sweep_ccm_convergence_steps,
    sweep_transfer_entropy,
)
from .data_loader import download_yfinance_series, get_date_range, get_ticker_input, get_ticker_name, load_two_series_from_csv, prompt_optional_series_names
from .plotting import plot_dtw_alignment, plot_ccm_convergence
from .preprocessing import _prompt_bool as _preprocess_prompt_bool
from .preprocessing import _prompt_choice as _preprocess_prompt_choice
from .preprocessing import _prompt_int as _preprocess_prompt_int
from .preprocessing import prompt_lagged_cross_correlation, run_preprocessing_flow
from .stationarity import adf_unit_root_test, make_series_stationary, print_adf_summary
from .surrogate import print_surrogate_summary, run_surrogate_test


def _prompt_bool(question: str, default: bool = False) -> bool:
    return _preprocess_prompt_bool(question, default=default)


def _prompt_choice(question: str, choices: list[str], default_index: int = 0) -> int:
    return _preprocess_prompt_choice(question, choices, default_index=default_index)


def _prompt_int(question: str, default: int) -> int:
    return _preprocess_prompt_int(question, default)


def _prompt_text(question: str, default: str | None = None) -> str:
    suffix = f" [{default}]" if default else ""
    answer = input(f"{question}{suffix}: ").strip()
    if answer:
        return answer
    if default is not None:
        return default
    return ""


def _prompt_data_source() -> int:
    return _prompt_choice(
        "Choose a data source",
        ["Yahoo Finance tickers", "CSV files with your own data"],
        default_index=0,
    )


def _load_yfinance_pair() -> tuple[pd.Series, pd.Series, str, str]:
    ticker_one, ticker_two = get_ticker_input()
    start, end = get_date_range()
    default_one = get_ticker_name(ticker_one)
    default_two = get_ticker_name(ticker_two)
    name_one, name_two = prompt_optional_series_names(default_one, default_two)
    loaded = download_yfinance_series(ticker_one, ticker_two, start, end, name_one=name_one, name_two=name_two)
    print(f"Loaded {len(loaded.left)} aligned observations for {loaded.left_name} and {loaded.right_name}.")
    return loaded.left, loaded.right, loaded.left_name, loaded.right_name


def _load_csv_pair() -> tuple[pd.Series, pd.Series, str, str]:
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
    return loaded.left, loaded.right, loaded.left_name, loaded.right_name


def _run_stationarity_check(series_one: pd.Series, series_two: pd.Series, label_one: str, label_two: str) -> tuple[pd.Series, pd.Series, str, str]:
    result_one = adf_unit_root_test(series_one, label_one)
    result_two = adf_unit_root_test(series_two, label_two)
    print_adf_summary(result_one, alpha=0.05)
    print_adf_summary(result_two, alpha=0.05)

    if not _prompt_bool("Difference non-stationary series until the ADF test accepts stationarity?", default=False):
        return series_one, series_two, label_one, label_two

    stationary_one = series_one
    stationary_two = series_two
    updated_label_one = label_one
    updated_label_two = label_two

    if not bool(result_one.get("stationary")):
        stationary_one, info_one = make_series_stationary(series_one, label_one)
        updated_label_one = f"{label_one} (stationary, diff order {info_one['diff_order']})"
    if not bool(result_two.get("stationary")):
        stationary_two, info_two = make_series_stationary(series_two, label_two)
        updated_label_two = f"{label_two} (stationary, diff order {info_two['diff_order']})"

    common_index = stationary_one.index.intersection(stationary_two.index)
    return stationary_one.loc[common_index], stationary_two.loc[common_index], updated_label_one, updated_label_two


def _run_parameter_sweeps(series_one: pd.Series, series_two: pd.Series, label_one: str, label_two: str) -> tuple[dict[str, object], dict[str, object]]:
    te_summary: dict[str, object] = {"one_to_two": None, "two_to_one": None}
    ccm_summary: dict[str, object] = {"one_to_two": None, "two_to_one": None}

    if _prompt_bool("Run TE sweep?", default=True):
        max_lag = _prompt_int("Maximum lag for TE", 5)
        max_embed_dim = _prompt_int("Maximum embedding dimension for TE", 3)
        te_grid = sweep_transfer_entropy(series_one.values, series_two.values, range(2, max_embed_dim + 1), range(1, max_lag + 1))
        te_summary = print_parameter_sweep_report(te_grid, "transfer entropy", label_one, label_two)

        if _prompt_bool("Run surrogate tests for TE now?", default=True):
            _run_metric_surrogates(
                "TE",
                series_one,
                series_two,
                label_one,
                label_two,
                te_summary,
            )

    if _prompt_bool("Run CCM sweep?", default=True):
        max_lag = _prompt_int("Maximum lag for CCM", 5)
        max_embed_dim = _prompt_int("Maximum embedding dimension for CCM", 3)
        ccm_grid = sweep_ccm_grid(series_one.values, series_two.values, range(2, max(2, max_embed_dim) + 1), range(1, max_lag + 1))
        ccm_summary = print_parameter_sweep_report(ccm_grid, "ccm", label_one, label_two)

        if _prompt_bool("Run CCM convergence analysis?", default=True):
            library_step = float(_prompt_text("CCM library-size step", "0.1") or "0.1")
            ccm_convergence = sweep_ccm_convergence_steps(
                series_one.values,
                series_two.values,
                embed_dim=max(2, max_embed_dim),
                lag=max(1, max_lag),
                library_step=library_step,
            )
            plot_ccm_convergence(
                ccm_convergence,
                title=f"CCM convergence: {label_one} vs {label_two}",
                label_one_to_two=f"{label_one} -> {label_two}",
                label_two_to_one=f"{label_two} -> {label_one}",
            )

        if _prompt_bool("Run surrogate tests for CCM now?", default=True):
            _run_metric_surrogates(
                "CCM",
                series_one,
                series_two,
                label_one,
                label_two,
                ccm_summary,
            )

    return te_summary, ccm_summary


def _run_metric_surrogates(
    metric_name: str,
    series_one: pd.Series,
    series_two: pd.Series,
    label_one: str,
    label_two: str,
    summary: dict[str, object],
) -> None:
    method_choice = _prompt_choice(
        f"Choose the surrogate method for {metric_name}",
        ["shuffle", "bootstrap", "both"],
        default_index=2,
    )
    methods = ["shuffle"] if method_choice == 0 else ["bootstrap"] if method_choice == 1 else ["shuffle", "bootstrap"]
    n_surrogates = _prompt_int("Number of surrogate samples", 200)
    seed_text = _prompt_text("Random seed for surrogate tests", "0")
    try:
        seed = int(seed_text)
    except ValueError:
        seed = 0

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
            result = run_surrogate_test(real_func, base_x, base_y, n_surrogates=n_surrogates, method=method, seed=seed)
            print_surrogate_summary(metric_name, f"{direction_label} (embed_dim={embed_dim}, lag={lag})", result)

    one_to_two = summary.get("one_to_two")
    two_to_one = summary.get("two_to_one")
    if isinstance(one_to_two, dict):
        _run_metric(f"{label_one} -> {label_two}", int(one_to_two["embed_dim"]), int(one_to_two["lag"]))
    if isinstance(two_to_one, dict):
        _run_metric(f"{label_two} -> {label_one}", int(two_to_one["embed_dim"]), int(two_to_one["lag"]))


def _run_granger_surrogates(series_one: pd.Series, series_two: pd.Series, label_one: str, label_two: str, maxlag: int) -> None:
    if not _prompt_bool("Run surrogate tests for Granger causality?", default=True):
        return

    method_choice = _prompt_choice(
        "Choose the surrogate method for Granger",
        ["shuffle", "bootstrap", "both"],
        default_index=2,
    )
    methods = ["shuffle"] if method_choice == 0 else ["bootstrap"] if method_choice == 1 else ["shuffle", "bootstrap"]
    n_surrogates = _prompt_int("Number of surrogate samples", 200)
    seed_text = _prompt_text("Random seed for Granger surrogate tests", "0")
    try:
        seed = int(seed_text)
    except ValueError:
        seed = 0

    def _run_direction(direction_label: str, target: pd.Series, source: pd.Series) -> None:
        for method in methods:
            result = run_surrogate_test(
                lambda x, y: granger_direction_score(pd.Series(x), pd.Series(y), maxlag=maxlag),
                target.values,
                source.values,
                n_surrogates=n_surrogates,
                method=method,
                seed=seed,
            )
            print_surrogate_summary("Granger", f"{direction_label} (maxlag={maxlag})", result)

    _run_direction(f"{label_one} -> {label_two}", series_two, series_one)
    _run_direction(f"{label_two} -> {label_one}", series_one, series_two)


def run_analysis_pipeline(series_one: pd.Series, series_two: pd.Series, label_one: str, label_two: str) -> None:
    """Run the full analysis workflow on two aligned series."""
    processed_one, processed_two, summary_one, summary_two = run_preprocessing_flow(series_one, series_two, label_one, label_two)
    print(f"Downstream analyses will use {summary_one['after_label']} and {summary_two['after_label']}.")

    raw_label_one = label_one
    raw_label_two = label_two

    processed_one, processed_two, label_one, label_two = _run_stationarity_check(processed_one, processed_two, summary_one["after_label"], summary_two["after_label"])

    if _prompt_bool("Run lagged cross-correlation (LCC) on the processed series?", default=False):
        prompt_lagged_cross_correlation(processed_one, processed_two, label_one, label_two)

    print("DTW is computed on the processed series shown above.")
    dtw_alignment = compute_dtw(processed_one, processed_two)
    print(f"DTW distance: {dtw_alignment.distance:.6f}")
    print(f"DTW normalized distance: {dtw_alignment.normalizedDistance:.6f}")
    if _prompt_bool("Plot the DTW alignment graph?", default=True):
        plot_dtw_alignment(processed_one, processed_two, dtw_alignment, label_one, label_two)

    if _prompt_bool("Run Granger causality analysis?", default=True):
        maxlag = _prompt_int("Maximum lag for Granger causality", 5)
        run_granger_causality_report(processed_one, processed_two, raw_label_one, raw_label_two, maxlag=maxlag)
        _run_granger_surrogates(processed_one, processed_two, raw_label_one, raw_label_two, maxlag=maxlag)

    _run_parameter_sweeps(processed_one, processed_two, raw_label_one, raw_label_two)
    print("All done.")


def run_general_workflow() -> None:
    """Start the full interactive workflow and let the user choose the data source."""
    source_choice = _prompt_data_source()
    if source_choice == 0:
        series_one, series_two, label_one, label_two = _load_yfinance_pair()
    else:
        series_one, series_two, label_one, label_two = _load_csv_pair()
    run_analysis_pipeline(series_one, series_two, label_one, label_two)


def run_yfinance_demo(
    ticker_one: str = "^IXIC",
    ticker_two: str = "^GSPC",
    start: str = "2018-01-01",
    end: str = "2024-12-31",
    name_one: str | None = None,
    name_two: str | None = None,
) -> None:
    """Convenience demo for notebooks or quick experiments with Yahoo Finance data."""
    loaded = download_yfinance_series(ticker_one, ticker_two, start, end, name_one=name_one, name_two=name_two)
    print(f"Loaded {len(loaded.left)} aligned observations for {loaded.left_name} and {loaded.right_name}.")
    run_analysis_pipeline(loaded.left, loaded.right, loaded.left_name, loaded.right_name)