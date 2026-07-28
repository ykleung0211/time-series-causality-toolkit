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
    plt.tight_layout() # Adjusts the padding between and around subplots to minimize overlap and ensure that labels, titles, and legends are not cut off.
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
        n_samples = min(200, len(index_one)) # Cap the number of alignment lines to plot for clarity
        sampled = list(range(0, len(index_one), max(1, len(index_one) // n_samples)))[:n_samples]
        plt.figure(figsize=(12, 5))
        plt.plot(series_one.index, series_one.values, label=label_one)
        plt.plot(series_two.index, series_two.values, label=label_two)

        # Draw lines connecting the aligned points between the two series
        for point in sampled:
            x_one = series_one.index[index_one[point]] # Get the x-coordinate (time index) for the aligned point in series_one
            x_two = series_two.index[index_two[point]] # Get the x-coordinate (time index) for the aligned point in series_two
            y_one = series_one.iloc[index_one[point]] # Get the y-coordinate (value) for the aligned point in series_one
            y_two = series_two.iloc[index_two[point]] # Get the y-coordinate (value) for the aligned point in series_two
            plt.plot([x_one, x_two], [y_one, y_two], color="gray", alpha=0.3)
        plt.title(f"DTW alignment lines ({label_one} vs {label_two})")
        plt.legend()
        plt.tight_layout()
        plt.show()

    try:
        cost_matrix = alignment.costMatrix
        plt.figure(figsize=(6, 6))

        # Plot the cost matrix, .T transposes the matrix so that the x-axis corresponds to series_one and the y-axis corresponds to series_two
        # The origin="lower" argument ensures that the (0,0) point is at the bottom-left corner of the plot
        # The aspect="auto" argument allows the aspect ratio of the plot to adjust automatically based on the data
        # The cmap="viridis" argument specifies the colormap to use for the cost matrix visualization
        plt.imshow(cost_matrix.T, origin="lower", aspect="auto", cmap="viridis")
        if index_one is not None and index_two is not None:
            plt.plot(index_one, index_two, "-r") # Plot the warping path on top of the cost matrix
        plt.title("DTW cost matrix with warping path")
        plt.xlabel(label_one)
        plt.ylabel(label_two)
        plt.tight_layout()
        plt.show()
    except Exception:
        pass


def plot_parameter_heatmap(frame: pd.DataFrame, value_column: str, x_column: str, y_column: str, title: str) -> None:
    """Plot a heatmap for a parameter sweep frame."""

    # Reshape the DataFrame (one row per embed_dim/lag/score) into a 2D matrix suitable for heatmap plotting, with embed_dim on the y-axis and lag on the x-axis
    pivot = frame.pivot(index=y_column, columns=x_column, values=value_column)
    plt.figure(figsize=(8, 6))
    plt.imshow(pivot.values, origin="lower", aspect="auto", cmap="magma")

    # xticks and yticks are set to the unique values of the x_column and y_column, respectively, to label the axes with the corresponding parameter values
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
    """
    Plot one or more line series from a tabular frame.
    
    Used for plotting CCM Convergence and lagged cross-correlation
    """
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
    label_one_to_two: str = "Series 1 → Series 2",
    label_two_to_one: str = "Series 2 → Series 1",
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
