"""Plotting helpers for the time-series causality toolkit."""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd


def plot_series_comparison(before: pd.Series, after: pd.Series, title: str, label_before: str, label_after: str) -> None:
    """Plot a before/after comparison for two aligned series."""
    plt.figure(figsize=(10, 4))
    plt.plot(before.index, before.values, alpha=0.5, label=label_before)
    plt.plot(after.index, after.values, "-r", label=label_after)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_single_series(series: pd.Series, title: str, label: str) -> None:
    """Plot a single series on a line chart."""
    plt.figure(figsize=(10, 4))
    plt.plot(series.index, series.values, label=label)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_dtw_alignment(series_one: pd.Series, series_two: pd.Series, alignment, label_one: str, label_two: str) -> None:
    """Plot a DTW warping path and cost matrix when available."""
    common_index = series_one.index.intersection(series_two.index)
    try:
        index_one = alignment.index1
        index_two = alignment.index2
    except Exception:
        try:
            path = alignment.path
            index_one = [point[0] for point in path]
            index_two = [point[1] for point in path]
        except Exception:
            index_one = None
            index_two = None

    if index_one is not None and index_two is not None:
        n_samples = min(200, len(index_one))
        sampled = list(range(0, len(index_one), max(1, len(index_one) // n_samples)))[:n_samples]
        plt.figure(figsize=(12, 5))
        plt.plot(common_index, series_one.loc[common_index].values, label=label_one)
        plt.plot(common_index, series_two.loc[common_index].values, label=label_two)
        for point in sampled:
            x_one = common_index[index_one[point]]
            x_two = common_index[index_two[point]]
            y_one = series_one.loc[x_one]
            y_two = series_two.loc[x_two]
            plt.plot([x_one, x_two], [y_one, y_two], color="gray", alpha=0.3)
        plt.title(f"DTW alignment lines ({label_one} vs {label_two})")
        plt.legend()
        plt.tight_layout()
        plt.show()

    try:
        cost_matrix = alignment.costMatrix
        plt.figure(figsize=(6, 6))
        plt.imshow(cost_matrix.T, origin="lower", aspect="auto", cmap="viridis")
        if index_one is not None and index_two is not None:
            plt.plot(index_one, index_two, "-r")
        plt.title("DTW cost matrix with warping path")
        plt.xlabel(label_one)
        plt.ylabel(label_two)
        plt.tight_layout()
        plt.show()
    except Exception:
        pass


def plot_parameter_heatmap(frame: pd.DataFrame, value_column: str, x_column: str, y_column: str, title: str) -> None:
    """Plot a heatmap for a parameter sweep frame."""
    pivot = frame.pivot(index=y_column, columns=x_column, values=value_column)
    plt.figure(figsize=(8, 6))
    plt.imshow(pivot.values, origin="lower", aspect="auto", cmap="magma")
    plt.xticks(range(len(pivot.columns)), pivot.columns)
    plt.yticks(range(len(pivot.index)), pivot.index)
    plt.colorbar(label=value_column)
    plt.title(title)
    plt.xlabel(x_column)
    plt.ylabel(y_column)
    plt.tight_layout()
    plt.show()


def plot_line_trend(
    frame: pd.DataFrame,
    x_column: str,
    y_columns: list[str],
    title: str,
    xlabel: str,
    ylabel: str,
    series_labels: dict[str, str] | None = None,
) -> None:
    """Plot one or more line series from a tabular frame."""
    plt.figure(figsize=(8, 5))
    for column in y_columns:
        label = series_labels.get(column, column) if series_labels else column
        plt.plot(frame[x_column], frame[column], marker="o", label=label)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_preprocessing_results(before: pd.Series, after: pd.Series, title: str, label_before: str, label_after: str) -> None:
    """Plot preprocessing before/after results."""
    plot_series_comparison(before, after, title, label_before, label_after)


def plot_granger_results(frame: pd.DataFrame, title: str = "Granger causality by lag") -> None:
    """Plot Granger F-statistics by lag for each direction."""
    plt.figure(figsize=(8, 5))
    for direction, direction_frame in frame.groupby("direction"):
        plt.plot(direction_frame["lag"], direction_frame["f_stat"], marker="o", label=direction)
    plt.title(title)
    plt.xlabel("Lag")
    plt.ylabel("F-statistic")
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_te_heatmap(frame: pd.DataFrame, value_column: str = "one_two", title: str = "Transfer entropy heatmap") -> None:
    """Plot a transfer-entropy heatmap."""
    plot_parameter_heatmap(frame, value_column=value_column, x_column="embed_dim", y_column="lag", title=title)


def plot_ccm_heatmap(frame: pd.DataFrame, value_column: str = "one_two", title: str = "CCM heatmap") -> None:
    """Plot a CCM heatmap."""
    plot_parameter_heatmap(frame, value_column=value_column, x_column="embed_dim", y_column="lag", title=title)


def plot_ccm_convergence(
    frame: pd.DataFrame,
    title: str = "CCM convergence",
    label_one_to_two: str = "one_two",
    label_two_to_one: str = "two_one",
) -> None:
    """Plot CCM convergence scores versus library fraction."""
    plot_line_trend(
        frame,
        "fraction",
        ["one_two", "two_one"],
        title,
        "Library fraction",
        "CCM score",
        series_labels={"one_two": label_one_to_two, "two_one": label_two_to_one},
    )
