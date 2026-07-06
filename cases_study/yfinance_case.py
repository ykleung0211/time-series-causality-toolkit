"""Interactive finance case study using Yahoo Finance data."""

from __future__ import annotations

import pandas as pd

from src.download_data import download_data, get_date_range, get_ticker_input, get_ticker_name
from src.measures import (
    compute_ccm,
    compute_dtw,
    compute_te,
    granger_strength,
    sweep_ccm_convergence,
    sweep_ccm_grid,
    sweep_transfer_entropy,
)
from src.plotting import (
    plot_dtw_alignment,
    plot_line_trend,
    plot_series_comparison,
)
from src.preprocessing import downsample_series, smooth_series
from src.stationarity import make_series_stationary
from src.surrogate import print_surrogate_summary, run_surrogate_test


def _prompt_bool(question: str, default: bool = False) -> bool:
    """Prompt for a yes/no answer."""
    suffix = "[Y/n]" if default else "[y/N]"
    answer = input(f"{question} {suffix}: ").strip().lower()
    if not answer:
        return default
    return answer in {"y", "yes"}


def _prompt_int(question: str, default: int) -> int:
    """Prompt for an integer and fall back to a default."""
    answer = input(f"{question} (default {default}): ").strip()
    if not answer:
        return default
    try:
        return int(answer)
    except ValueError:
        print(f"Invalid input. Using default {default}.")
        return default


def _prompt_float(question: str, default: float) -> float:
    """Prompt for a float and fall back to a default."""
    answer = input(f"{question} (default {default}): ").strip()
    if not answer:
        return default
    try:
        return float(answer)
    except ValueError:
        print(f"Invalid input. Using default {default}.")
        return default


def _prompt_int_list(question: str, default: list[int]) -> list[int]:
    """Prompt for a comma-separated list of integers."""
    answer = input(f"{question} (comma-separated, default {default}): ").strip()
    if not answer:
        return default
    try:
        values = [int(item.strip()) for item in answer.split(",") if item.strip()]
        return values or default
    except ValueError:
        print(f"Invalid input. Using default {default}.")
        return default


def _prompt_float_list(question: str, default: list[float]) -> list[float]:
    """Prompt for a comma-separated list of floats."""
    answer = input(f"{question} (comma-separated, default {default}): ").strip()
    if not answer:
        return default
    try:
        values = [float(item.strip()) for item in answer.split(",") if item.strip()]
        return values or default
    except ValueError:
        print(f"Invalid input. Using default {default}.")
        return default


def _prompt_optional_seed() -> int | None:
    """Prompt for a random seed used by surrogate tests."""
    answer = input("Random seed for surrogate tests (blank for no seed): ").strip()
    if not answer:
        return None
    try:
        return int(answer)
    except ValueError:
        print("Invalid seed. Surrogate tests will use non-deterministic randomness.")
        return None


def _print_section(title: str) -> None:
    """Print a readable section header with blank lines around it."""
    print()
    print("=" * 80)
    print(title)
    print("=" * 80)


def _print_metric_grid_summary(frame: pd.DataFrame, metric_name: str, direction_label: str, value_column: str) -> None:
    """Print the best parameter combination for a grid-search metric."""
    if frame.empty or frame[value_column].dropna().empty:
        print(f"No valid {metric_name} results were produced for {direction_label}.")
        return
    best_row = frame.loc[frame[value_column].idxmax()]
    print(f"Best {metric_name} for {direction_label}: embed_dim={int(best_row['embed_dim'])}, lag={int(best_row['lag'])}, value={best_row[value_column]:.6f}")


def run_finance_case_study() -> None:
    """Run the finance-oriented example workflow using Yahoo Finance data."""
    ticker_one, ticker_two = get_ticker_input()
    name_one = get_ticker_name(ticker_one)
    name_two = get_ticker_name(ticker_two)
    start, end = get_date_range()

    print(f"\nDownloading {ticker_one} ({name_one}) and {ticker_two} ({name_two}) from {start} to {end}...")
    _, log_returns = download_data(ticker_one, ticker_two, start, end)
    print(f"Sequence length: {len(log_returns)} observations")

    downsample_enabled = _prompt_bool("Do you want to downsample?")
    downsample_step: int | None = None
    downsample_freq: str | None = None
    if downsample_enabled:
        print("Choose downsampling method: 1) every N-th sample  2) pandas frequency alias")
        method = input("Method [1/2]: ").strip() or "1"
        if method == "1":
            downsample_step = _prompt_int("Downsample step N", 2)
        else:
            downsample_freq = input("Pandas offset alias (e.g. W, M)").strip() or "W"
        plot_downsampled = _prompt_bool("Plot downsampled series?")
    else:
        plot_downsampled = False

    smooth_enabled = _prompt_bool("Do you want to smooth the series?")
    smooth_window = 5
    if smooth_enabled:
        smooth_window = _prompt_int("Smoothing window size", 5)

    series_one = log_returns[f"{ticker_one}_log_return"].copy()
    series_two = log_returns[f"{ticker_two}_log_return"].copy()

    if downsample_enabled:
        downsampled_one = downsample_series(series_one, step=downsample_step, freq=downsample_freq)
        downsampled_two = downsample_series(series_two, step=downsample_step, freq=downsample_freq)
        print(f"After downsampling: {len(downsampled_one)} observations")
        if plot_downsampled:
            plot_series_comparison(series_one, downsampled_one, f"Downsampled {name_one}", f"{name_one} original", f"{name_one} downsampled")
            plot_series_comparison(series_two, downsampled_two, f"Downsampled {name_two}", f"{name_two} original", f"{name_two} downsampled")
        series_one, series_two = downsampled_one, downsampled_two

    if smooth_enabled:
        pre_smooth_one = series_one.copy()
        pre_smooth_two = series_two.copy()
        series_one = smooth_series(series_one, window=smooth_window)
        series_two = smooth_series(series_two, window=smooth_window)
        print(f"After smoothing: {len(series_one)} observations")
        if _prompt_bool("Plot smoothing result?"):
            plot_series_comparison(pre_smooth_one, series_one, f"Smoothing comparison {name_one}", f"{name_one} pre-smooth", f"{name_one} smoothed")
            plot_series_comparison(pre_smooth_two, series_two, f"Smoothing comparison {name_two}", f"{name_two} pre-smooth", f"{name_two} smoothed")

    print("\n" + "=" * 80)
    print("STATIONARITY CHECK (ADF / UNIT-ROOT TEST)")
    print("=" * 80)
    adf_alpha = _prompt_float("ADF significance level alpha", 0.05)
    max_diff_order = _prompt_int("Max differencing order if non-stationary", 2)

    stationary_one, meta_one = make_series_stationary(series_one, f"{name_one} log return", alpha=adf_alpha, max_diff_order=max_diff_order)
    stationary_two, meta_two = make_series_stationary(series_two, f"{name_two} log return", alpha=adf_alpha, max_diff_order=max_diff_order)

    if not meta_one["valid"] or not meta_two["valid"]:
        print("\nStationarity test is not valid for at least one series.")
        print("Typical causes: too few samples after downsampling/smoothing, or near-constant values.")
        print("Suggested fix: reduce downsampling/smoothing, or use a longer date range.")
        if not _prompt_bool("Continue analysis anyway with the current transformed series?"):
            print("Stopped. Please adjust preprocessing and rerun.")
            raise SystemExit(1)

    if not meta_one["stationary"] or not meta_two["stationary"]:
        print("\nWarning: at least one series is still non-stationary after maximum differencing.")
        print("Granger, TE, and CCM results may be less reliable under unit roots.")
        if not _prompt_bool("Continue anyway?"):
            print("Stopped due to non-stationarity.")
            raise SystemExit(1)

    aligned_index = stationary_one.index.intersection(stationary_two.index)
    series_one = stationary_one.loc[aligned_index]
    series_two = stationary_two.loc[aligned_index]
    print(f"\nUsing differenced series for analysis: {name_one} d={meta_one['diff_order']}, {name_two} d={meta_two['diff_order']}")
    print(f"Final aligned length after stationarity processing: {len(series_one)}")
    if len(series_one) < 30:
        print("Warning: very short series after preprocessing may reduce reliability of DTW/Granger/TE/CCM results.")

    dtw_alignment = compute_dtw(series_one, series_two)
    print(f"\nDTW distance: {dtw_alignment.distance:.6f}")
    if _prompt_bool("Plot DTW alignment graph?"):
        plot_dtw_alignment(series_one, series_two, dtw_alignment, name_one, name_two)

    maxlag = _prompt_int("Choose maxlag for Granger causality", 5)
    granger_one_two, granger_two_one = granger_strength(series_one, series_two, maxlag=maxlag, verbose=False)
    _print_section("GRANGER CAUSALITY RESULTS (max F-stat across lags)")
    print(f"{name_one} -> {name_two}: {granger_one_two:.6f}")
    print()
    print(f"{name_two} -> {name_one}: {granger_two_one:.6f}")

    te_embed = _prompt_int("\nTE embedding dimension", 3)
    te_lag = _prompt_int("TE lag / tau", 1)
    ccm_embed = _prompt_int("CCM embedding dimension", 3)
    ccm_lag = _prompt_int("CCM lag / tau", 1)

    series_one_values = series_one.values
    series_two_values = series_two.values
    te_one_two = compute_te(series_one_values, series_two_values, te_embed, te_lag)
    te_two_one = compute_te(series_two_values, series_one_values, te_embed, te_lag)
    _print_section("TRANSFER ENTROPY RESULTS")
    print(f"{name_one} -> {name_two}: {te_one_two:.6f}")
    print()
    print(f"{name_two} -> {name_one}: {te_two_one:.6f}")

    ccm_one_two, ccm_two_one = compute_ccm(series_one_values, series_two_values, ccm_embed, ccm_lag)
    _print_section("CONVERGENT CROSS MAPPING RESULTS")
    print(f"{name_one} -> {name_two}: {ccm_one_two:.6f}")
    print()
    print(f"{name_two} -> {name_one}: {ccm_two_one:.6f}")

    if _prompt_bool("\nSearch TE embedding dimensions and lags to find the best combination?"):
        te_dims = _prompt_int_list("TE embedding dimensions to try", [2, 3, 4])
        te_lags = _prompt_int_list("TE lags to try", [1, 2, 3])
        te_grid = sweep_transfer_entropy(series_one_values, series_two_values, te_dims, te_lags)
        print("\nTE grid search completed.")
        _print_metric_grid_summary(te_grid, "TE", f"{name_one} -> {name_two}", "one_two")
        _print_metric_grid_summary(te_grid, "TE", f"{name_two} -> {name_one}", "two_one")

    ccm_grid = None
    if _prompt_bool("Search CCM embedding dimensions and lags to find the best combination?"):
        ccm_dims = _prompt_int_list("CCM embedding dimensions to try", [2, 3, 4])
        ccm_lags = _prompt_int_list("CCM lags to try", [1, 2, 3])
        ccm_grid = sweep_ccm_grid(series_one_values, series_two_values, ccm_dims, ccm_lags)
        print("\nCCM grid search completed.")
        _print_metric_grid_summary(ccm_grid, "CCM", f"{name_one} -> {name_two}", "one_two")
        _print_metric_grid_summary(ccm_grid, "CCM", f"{name_two} -> {name_one}", "two_one")

        if _prompt_bool("Plot CCM convergence over library size for the best direction?"):
            best_index = ccm_grid["one_two"].idxmax() if not ccm_grid["one_two"].dropna().empty else None
            if best_index is not None:
                best_row = ccm_grid.loc[best_index]
                library_fractions = _prompt_float_list("Library fractions to sweep", [0.2, 0.4, 0.6, 0.8, 1.0])
                convergence = sweep_ccm_convergence(
                    series_one_values,
                    series_two_values,
                    int(best_row["embed_dim"]),
                    int(best_row["lag"]),
                    library_fractions,
                )
                convergence = convergence.rename(
                    columns={
                        "one_two": f"{name_one} -> {name_two}",
                        "two_one": f"{name_two} -> {name_one}",
                    }
                )
                plot_line_trend(
                    convergence,
                    "fraction",
                    [f"{name_one} -> {name_two}", f"{name_two} -> {name_one}"],
                    f"CCM convergence: {name_one} vs {name_two}",
                    "Library fraction",
                    "CCM score",
                )

    seed = _prompt_optional_seed()
    do_shuffle = _prompt_bool("Do you want to run shuffle surrogate tests?")
    do_bootstrap = _prompt_bool("Do you want to run bootstrap surrogate tests?")
    n_surrogates = 200
    if do_shuffle or do_bootstrap:
        n_surrogates = _prompt_int("Number of surrogate samples", 200)

    granger_one_two_func = lambda x, y, maxlag=maxlag: granger_strength(pd.Series(x), pd.Series(y), maxlag)[0]
    granger_two_one_func = lambda x, y, maxlag=maxlag: granger_strength(pd.Series(x), pd.Series(y), maxlag)[1]
    te_one_two_func = lambda x, y, **kwargs: compute_te(x, y, te_embed, te_lag)
    te_two_one_func = lambda x, y, **kwargs: compute_te(y, x, te_embed, te_lag)
    ccm_one_two_func = lambda x, y, **kwargs: compute_ccm(x, y, ccm_embed, ccm_lag)[0]
    ccm_two_one_func = lambda x, y, **kwargs: compute_ccm(x, y, ccm_embed, ccm_lag)[1]

    if do_shuffle:
        print("\nRunning shuffle surrogate tests (randomization test for p-values)...")
        granger_shuffle_one_two = run_surrogate_test(granger_one_two_func, series_one_values, series_two_values, n_surrogates=n_surrogates, method="shuffle", seed=seed)
        granger_shuffle_two_one = run_surrogate_test(granger_two_one_func, series_one_values, series_two_values, n_surrogates=n_surrogates, method="shuffle", seed=seed)
        te_shuffle_one_two = run_surrogate_test(te_one_two_func, series_one_values, series_two_values, n_surrogates=n_surrogates, method="shuffle", seed=seed)
        te_shuffle_two_one = run_surrogate_test(te_two_one_func, series_one_values, series_two_values, n_surrogates=n_surrogates, method="shuffle", seed=seed)
        ccm_shuffle_one_two = run_surrogate_test(ccm_one_two_func, series_one_values, series_two_values, n_surrogates=n_surrogates, method="shuffle", seed=seed)
        ccm_shuffle_two_one = run_surrogate_test(ccm_two_one_func, series_one_values, series_two_values, n_surrogates=n_surrogates, method="shuffle", seed=seed)

        print_surrogate_summary("Granger (max F)", f"{name_one} -> {name_two}", granger_shuffle_one_two)
        print_surrogate_summary("Granger (max F)", f"{name_two} -> {name_one}", granger_shuffle_two_one)
        print_surrogate_summary("Transfer Entropy", f"{name_one} -> {name_two}", te_shuffle_one_two)
        print_surrogate_summary("Transfer Entropy", f"{name_two} -> {name_one}", te_shuffle_two_one)
        print_surrogate_summary("CCM", f"{name_one} -> {name_two}", ccm_shuffle_one_two)
        print_surrogate_summary("CCM", f"{name_two} -> {name_one}", ccm_shuffle_two_one)

    if do_bootstrap:
        print("\nRunning bootstrap surrogate tests (confidence intervals)...")
        granger_boot_one_two = run_surrogate_test(granger_one_two_func, series_one_values, series_two_values, n_surrogates=n_surrogates, method="bootstrap", seed=seed)
        granger_boot_two_one = run_surrogate_test(granger_two_one_func, series_one_values, series_two_values, n_surrogates=n_surrogates, method="bootstrap", seed=seed)
        te_boot_one_two = run_surrogate_test(te_one_two_func, series_one_values, series_two_values, n_surrogates=n_surrogates, method="bootstrap", seed=seed)
        te_boot_two_one = run_surrogate_test(te_two_one_func, series_one_values, series_two_values, n_surrogates=n_surrogates, method="bootstrap", seed=seed)
        ccm_boot_one_two = run_surrogate_test(ccm_one_two_func, series_one_values, series_two_values, n_surrogates=n_surrogates, method="bootstrap", seed=seed)
        ccm_boot_two_one = run_surrogate_test(ccm_two_one_func, series_one_values, series_two_values, n_surrogates=n_surrogates, method="bootstrap", seed=seed)

        print_surrogate_summary("Granger (max F)", f"{name_one} -> {name_two}", granger_boot_one_two)
        print_surrogate_summary("Granger (max F)", f"{name_two} -> {name_one}", granger_boot_two_one)
        print_surrogate_summary("Transfer Entropy", f"{name_one} -> {name_two}", te_boot_one_two)
        print_surrogate_summary("Transfer Entropy", f"{name_two} -> {name_one}", te_boot_two_one)
        print_surrogate_summary("CCM", f"{name_one} -> {name_two}", ccm_boot_one_two)
        print_surrogate_summary("CCM", f"{name_two} -> {name_one}", ccm_boot_two_one)

    print("\nAll done.")


def main() -> None:
    """Entry point for the finance case study."""
    run_finance_case_study()


if __name__ == "__main__":
    main()
