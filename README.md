# Time Series Causality Toolkit

A Python toolkit for detecting and validating causal relationships between time series, 
supporting both linear (Granger causality) and nonlinear (Transfer Entropy, Convergent 
Cross Mapping) methods, with built-in surrogate significance testing.

## Why this exists

*"Does X actually drive Y, or are we just looking at a correlated coincidence?"* Standard tools (Granger causality, correlation, cointegration) answer this under a linear lens. Markets are not always linear, and even well-established techniques carry hidden failure modes — spurious detections, misidentified direction, or bias introduced by naive preprocessing.

This toolkit combines three complementary causality-detection methods, wraps each in surrogate/bootstrap significance testing, and — critically — includes a dedicated notebook whose sole purpose is to find where the toolkit itself breaks. Most causal-inference codebases show you when a method works. This one also shows you when it lies.

## What's inside

| Method | Captures | Weakness it covers |
|---|---|---|
| Granger causality | Linear, lagged predictive power | Fast, interpretable, but blind to nonlinear coupling |
| Transfer Entropy (TE) | Nonlinear, information-theoretic dependence | Sensitive to embedding/lag choice |
| Convergent Cross Mapping (CCM) | Nonlinear dynamical coupling (Sugihara framework) | Can misread synchronized systems as bidirectional |
| Lagged cross-correlation | Simple co-movement timing | No causal guarantee on its own |
| DTW realignment | Corrects for stale-price/async timing | Can *introduce* look-ahead bias if misapplied |

**Stationarity checks (ADF test)** and **automatic differencing** are appiled before any analysis. Every causal score is paired with **shuffle-surrogate and bootstrap significance testing** — a raw F-stat or CCM score is not a claim, a validated one is.

## Three notebooks, one narrative arc

### 1. Ground-truth validation — does the toolkit actually work?
Two synthetic systems with known causal direction (linear VAR, nonlinear coupled logistic maps) are used to benchmark all three methods against ground truth.

**Key finding:** No single method is universally reliable. Granger nails the linear system (F = 357.5, p = 0.005) but produces a *spurious significant result in the wrong direction* on the nonlinear system (F = 129.7, p < 0.001) — a real, cited failure mode of linear F-tests under nonlinear dynamics. CCM and TE correctly identify the true direction on the nonlinear system, but generate misleading near-symmetric scores when coupling is strong. Surrogate testing is what separates real signal from artifact in every case.

### 2. SPX–VIX volatility feedback — does it hold up on real markets?
The validated toolkit is applied to the well-documented SPX/VIX leverage relationship across three regimes (full sample, calm 2017, stress 2020).

**Key finding:** Lagged cross-correlation is stable and strongly negative across all regimes (-0.72 to -0.74 at lag 0). Granger causality (SPX→VIX) is significant and surrogate-validated in both calm and stress sub-regimes, but the direction flips in the naive full-sample regression — a concrete demonstration of regime-dependence and Simpson's-paradox-style aggregation bias. CCM confirms strong bidirectional coupling everywhere, consistent with the known feedback-loop structure between the index and its volatility index.

### 3. DTW alignment stress-test — where does the toolkit break?
A deliberate adversarial test: does Dynamic Time Warping, a natural-seeming fix for asynchronous/stale price data, introduce more bias than it removes?

**Key finding:** Yes, dramatically. On synthetic illiquid-vs-liquid asset data, naive Granger analysis shows a modest, correctly-directed signal (F = 55.48 vs 1.10). After DTW realignment, the causality direction **reverses and becomes far more "confident"** (F = 24.68 vs 82.82) — a textbook look-ahead bias artifact, since DTW's warping path uses future information to align the series. The diagnostic output (mean warp offset 0.99, range -5 to +9 bars) makes the mechanism concrete and auditable.

## Why this matters for research and trading

- **Methodological honesty:** every notebook distinguishes a raw statistic from a surrogate-validated claim.
- **Regime awareness:** the SPX/VIX study shows causal direction is not a fixed property of a market relationship — it can flip with the sample window.
- **Preprocessing risk:** the DTW notebook is a cautionary case study relevant to anyone aligning illiquid instruments, alternative data, or multi-venue tick data.
- **Reproducibility:** all findings are generated from deterministic seeds and included directly in the notebooks.


## Repository structure
```
├── case_study/
│   ├── 01_synthetic_validation.ipynb
│   ├── 02_spx_vix_volatility_feedback.ipynb
│   └── 03_dtw_alignment_and_preprocessing.ipynb
├── src/
│   ├── causal_analysis.py
│   ├── data_loader.py
│   ├── plotting.py
│   ├── preprocessing.py
│   ├── stationarity.py
│   ├── surrogate.py
│   └── workflows.py
├── LICENSE
└── requirements.txt
```


## Installation

```bash
git clone https://github.com/ykleung0211/time-series-causality-toolkit.git
cd time-series-causality-toolkit
pip install -r requirements.txt
```


## Getting started

```python
from src import PreprocessingConfig, AnalysisConfig, run_analysis_pipeline

results = run_analysis_pipeline(
    series_x, series_y,
    preprocessing_config=PreprocessingConfig(base_representation="raw", standardize=True),
    analysis_config=AnalysisConfig(
        run_granger=True, granger_max_lag=5, run_granger_surrogates=True,
        run_te=True, run_ccm=True, run_ccm_surrogates=True,
        n_surrogates=200, surrogate_seed=0,
    ),
)
```
