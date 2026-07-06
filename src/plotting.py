# Reusable plotting helpers for the toolkit.

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd


def plot_series_comparison(before: pd.Series, after: pd.Series, title: str, label_before: str, label_after: str) -> None:
    # Plot a before/after comparison for preprocessing steps.
    common_index = before.index.intersection(after.index)
    plt.figure(figsize=(10, 4))
    plt.plot(before.loc[common_index].index, before.loc[common_index].values, alpha=0.5, label=label_before)
    plt.plot(after.loc[common_index].index, after.loc[common_index].values, "-r", label=label_after)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_dtw_alignment(series_one: pd.Series, series_two: pd.Series, alignment, label_one: str, label_two: str) -> None:
    # Plot a DTW alignment path and an optional cost matrix.
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
    # Plot a heatmap from a grid-search DataFrame.
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


def plot_line_trend(frame: pd.DataFrame, x_column: str, y_columns: list[str], title: str, xlabel: str, ylabel: str) -> None:
    # Plot one or more line trends from a DataFrame.
    plt.figure(figsize=(8, 5))
    for column in y_columns:
        plt.plot(frame[x_column], frame[column], marker="o", label=column)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.tight_layout()
    plt.show()
