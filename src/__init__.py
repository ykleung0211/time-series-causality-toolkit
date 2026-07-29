"""Public API for the time-series causality toolkit.

This package exposes the reusable data-loading, preprocessing,
causal-analysis, surrogate testing, and plotting utilities used by
the non-interactive analysis pipeline.
"""

__version__ = "0.1.0"

from .data_loader import (
    LoadedSeriesPair,
    coerce_two_series,
    download_yfinance_series,
    get_ticker_name,
    load_two_series_from_csv,
)
from .preprocessing import (
    PreprocessingConfig,
    PreprocessingResult,
    compute_log_returns,
    compute_returns,
    downsample_series,
    lagged_cross_correlation,
    lagged_cross_correlation_report,
    preprocess_series_pair,
    preprocess_single_series,
    smooth_series,
    standardize_series,
    summarize_preprocessing,
)
from .stationarity import (
    adf_unit_root_test,
    make_series_stationary,
    print_adf_summary,
)
from .causal_analysis import (
    compute_ccm,
    compute_dtw_sequence,
    compute_te,
    extract_warping_path,
    run_granger_causality_report,
    print_parameter_sweep_report,
    sweep_transfer_entropy,
    sweep_ccm,
    sweep_ccm_convergence_steps,
    warp_series_to_match,
)
from .surrogate import (
    SurrogateResult,
    print_surrogate_summary,
    run_surrogate_test,
)
from .plotting import (
    plot_ccm_convergence,
    plot_ccm_heatmap,
    plot_dtw_alignment,
    plot_granger_results,
    plot_line_trend,
    plot_parameter_heatmap,
    plot_preprocessing_results,
    plot_single_series,
    plot_series_comparison,
    plot_te_heatmap,
)
from .workflows import (
    AnalysisConfig,
    run_analysis_pipeline,
)

__all__ = [
    "LoadedSeriesPair",
    "coerce_two_series",
    "download_yfinance_series",
    "get_ticker_name",
    "load_two_series_from_csv",
    "PreprocessingConfig",
    "PreprocessingResult",
    "compute_log_returns",
    "compute_returns",
    "downsample_series",
    "lagged_cross_correlation",
    "lagged_cross_correlation_report",
    "preprocess_series_pair",
    "preprocess_single_series",
    "smooth_series",
    "standardize_series",
    "summarize_preprocessing",
    "adf_unit_root_test",
    "make_series_stationary",
    "print_adf_summary",
    "compute_ccm",
    "compute_dtw_sequence",
    "compute_te",
    "extract_warping_path",
    "run_granger_causality_report",
    "print_parameter_sweep_report",
    "sweep_transfer_entropy",
    "sweep_ccm",
    "sweep_ccm_convergence_steps",
    "warp_series_to_match",
    "SurrogateResult",
    "run_surrogate_test",
    "print_surrogate_summary",
    "plot_ccm_convergence",
    "plot_ccm_heatmap",
    "plot_dtw_alignment",
    "plot_granger_results",
    "plot_line_trend",
    "plot_parameter_heatmap",
    "plot_preprocessing_results",
    "plot_single_series",
    "plot_series_comparison",
    "plot_te_heatmap",
    "AnalysisConfig",
    "run_analysis_pipeline",
]