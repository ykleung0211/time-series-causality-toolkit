# Time-Series Causality Toolkit

A research-prototype Python toolkit for detecting and validating directional relationships between pairs of time series. It combines Granger causality, transfer entropy (TE) and convergent cross mapping (CCM), and attaches a null model to every score, so a number is only reported next to evidence that it is not noise.

Status: v0.2. Validated on simulated systems with known direction and on SPX/VIX. The next stage extends it to multivariate and high-dimensional data (see Roadmap).

## What is inside

| Component | Purpose | Known limitation |
|---|---|---|
| Granger causality | Linear, lagged predictability | Flags false reverse links under strong nonlinear coupling |
| Transfer entropy (ordinal) | Nonlinear information transfer | Biased at small samples; needs a matched null |
| CCM | Dynamical coupling | Reports both directions under synchronisation |
| Lagged cross-correlation | Lead-lag timing | Not causal on its own |
| DTW alignment (optional, banded) | Realign asynchronous series | Cuts power and gives unstable offsets |
| Surrogates | shuffle, circular shift, block bootstrap | p-values are floored at 1/(n_surrogates+1) |

Preprocessing: raw / returns / log returns, trailing smoothing, downsampling, z-scoring, ADF test with automatic differencing (both series are differenced to the same order). All series are matched on their date index, never by position.

## Findings so far

All runs are single simulated seeds with development-size surrogate counts (40 to 200), so p-values sit at their floors. Treat them as preliminary.

1. Synthetic validation (notebook 01). Linear system: Granger recovers X→Y (F=357.5 vs 1.6). TE points the right way weakly (0.087 vs 0.065). CCM favours the wrong direction (Y→X 0.480 vs X→Y 0.145, both significant). Nonlinear coupled maps: all three rank X→Y first, and all three also flag a false reverse link at strong coupling. At weak coupling the CCM reverse link disappears; the TE reverse link does not.
2. SPX/VIX (notebook 02). Lagged correlation peaks at lag 0 (-0.72 to -0.74). Granger SPX→VIX is significant in every regime. VIX→SPX is not significant in the calm and stress years under a circular-shift null, but has the larger F in the full sample. TE is at or below its null in the 250-observation regimes. CCM is high in both directions, consistent with VIX being derived from SPX options.
3. DTW stress test (notebook 03). With both series smoothed identically and matched on dates, lagged correlation recovers the true 3-day delay. Granger on the naive alignment gives F=122.9 vs 9.9. DTW-warped alignment keeps the direction (F=32.2 vs 3.0) but loses about 74% of the F-statistic and gives unstable offsets (std 2.8). A band constraint changes little.

## Lessons and corrections

An earlier version of notebook 03 reported that DTW reversed the causal direction. That result came from smoothing only one series and then pairing the series by position, which shifted them by four days. After the fix, the reversal disappeared. The same audit showed that a plain shuffle null made TE look wrong-direction on the linear system, which a circular-shift null corrected. Both corrections are covered by tests.

## Repository structure
├── case_study/
│ ├── 01_synthetic_validation.ipynb
│ ├── 02_spx_vix_volatility_feedback.ipynb
│ └── 03_dtw_alignment_and_preprocessing.ipynb
├── src/
│ ├── causal_analysis.py # Granger, TE, CCM, DTW, sweeps
│ ├── data_loader.py
│ ├── plotting.py
│ ├── preprocessing.py
│ ├── stationarity.py
│ ├── surrogate.py
│ └── workflows.py # AnalysisConfig, run_analysis_pipeline
├── requirements.txt
└── LICENSE


## Installation

```bash
git clone [https://github.com/ykleung0211/time-series-causality-toolkit.git](https://github.com/ykleung0211/time-series-causality-toolkit.git)
cd time-series-causality-toolkit
python -m venv .venv
pip install -r requirements.txt pytest
```

## Quick start

```python
from src import PreprocessingConfig, AnalysisConfig, run_analysis_pipeline

results = run_analysis_pipeline(
    series_x, series_y, "X", "Y",
    preprocessing_config=PreprocessingConfig(base_representation="raw", standardize=True),
    analysis_config=AnalysisConfig(
        run_dtw=False,
        run_granger=True, granger_maxlag=5, run_granger_surrogates=True,
        run_te=True, run_ccm=True, run_ccm_surrogates=True,
        n_surrogates=200, surrogate_seed=0, surrogate_method="circular_shift",
    ),
)
```

surrogate_method accepts shuffle, circular_shift, block_bootstrap, bootstrap (legacy), both, or all. TE and CCM take the maximum over the parameter grid inside every surrogate, so selection is accounted for.

## Roadmap

- Multivariate Granger via VAR with coefficient tables and false-discovery control
- Sparse (penalised) VAR and PCMCI for higher-dimensional systems
- Benchmarks on simulated systems with known causal graphs (sparse VAR, coupled maps, Lorenz-96), with TPR, FDR and SHD
- Latent-mixing benchmark: a known causal system embedded into 64 to 128 dimensions
- Real-data application after the benchmarks