"""Stationarity and unit-root diagnostics."""

from __future__ import annotations

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller


def adf_unit_root_test(series: pd.Series, series_name: str, alpha: float = 0.05) -> dict[str, object]:
    """Run an ADF test and return a structured summary.

    H0: a unit root exists, so the series is non-stationary.
    H1: the series is stationary.
    """
    cleaned = pd.Series(series).dropna()
    result: dict[str, object] = {
        "name": series_name,
        "n_obs": int(len(cleaned)),
        "valid": True,
        "stationary": False,
        "reason": None,
        "adf_stat": np.nan,
        "pvalue": np.nan,
        "usedlag": None,
        "crit": {},
    }

    if len(cleaned) < 20:
        result["valid"] = False
        result["reason"] = "Too few observations for a reliable ADF test (need roughly >= 20)."
        return result
    if float(cleaned.std()) == 0.0:
        result["valid"] = False
        result["reason"] = "Series variance is zero (constant series), so ADF is not meaningful."
        return result

    try:
        adf_stat, pvalue, usedlag, nobs, crit_vals, _ = adfuller(cleaned, autolag="AIC")
        result["adf_stat"] = float(adf_stat)
        result["pvalue"] = float(pvalue)
        result["usedlag"] = int(usedlag)
        result["n_obs"] = int(nobs)
        result["crit"] = {key: float(value) for key, value in crit_vals.items()}
        result["stationary"] = bool(pvalue < alpha)
    except Exception as exc:
        result["valid"] = False
        result["reason"] = f"ADF failed: {exc}"

    return result


def print_adf_summary(result: dict[str, object], alpha: float, verbose: bool = False) -> None:
    """Print a compact ADF report when verbose output is enabled."""
    if not verbose:
        return
    print("\n" + "-" * 80)
    print(f"ADF / unit-root test summary for {result['name']}")
    print("-" * 80)
    if not result["valid"]:
        print(f"Status: INVALID ({result['reason']})")
        print("-" * 80)
        return
    print(f"Observations:         {result['n_obs']}")
    print(f"ADF statistic:        {result['adf_stat']:.6f}")
    print(f"p-value:              {result['pvalue']:.6f}")
    print(f"Used lags:            {result['usedlag']}")
    print(f"Critical values:      {result['crit']}")
    if result["stationary"]:
        print(f"Conclusion:           Stationary (reject unit-root null at alpha={alpha})")
    else:
        print(f"Conclusion:           Non-stationary (fail to reject unit-root null at alpha={alpha})")
    print("-" * 80)


def make_series_stationary(
    series: pd.Series,
    series_name: str,
    alpha: float = 0.05,
    max_diff_order: int = 2,
    verbose: bool = False,
) -> tuple[pd.Series, dict[str, object]]:
    """Difference a series until the ADF test accepts stationarity or the limit is reached."""
    current = pd.Series(series).dropna()
    diff_order = 0

    while diff_order <= max_diff_order:
        stage_name = f"{series_name} (diff order {diff_order})"
        adf_result = adf_unit_root_test(current, stage_name, alpha=alpha)
        print_adf_summary(adf_result, alpha, verbose=verbose)

        if not adf_result["valid"]:
            return current, {"valid": False, "stationary": False, "diff_order": diff_order, "reason": adf_result["reason"]}
        if adf_result["stationary"]:
            return current, {"valid": True, "stationary": True, "diff_order": diff_order, "reason": None}

        diff_order += 1
        if diff_order <= max_diff_order:
            current = current.diff().dropna()

    return current, {
        "valid": True,
        "stationary": False,
        "diff_order": max_diff_order,
        "reason": f"Not stationary even after differencing up to order {max_diff_order}.",
    }
