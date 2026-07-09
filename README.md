# Time-Series Causality Toolkit

This project provides a reusable toolkit for pairwise time-series causality analysis plus a notebook demo.

## Structure

- `src/data_loader.py`: Yahoo Finance and CSV loading helpers
- `src/preprocessing.py`: returns, log returns, smoothing, downsampling, and z-score standardization
- `src/stationarity.py`: ADF / unit-root testing and optional differencing
- `src/causal_analysis.py`: DTW, Granger causality, transfer entropy, CCM, and parameter-sweep summaries
- `src/surrogate.py`: shuffle and bootstrap surrogate tests with reproducible seeds
- `src/plotting.py`: reusable visualization helpers
- `src/workflows.py`: interactive orchestration used by `main.py` and the notebook demo
- `case_study/yfinance_example.ipynb`: notebook demo that calls a single shared workflow function
- `main.py`: interactive entrypoint that lets you choose Yahoo Finance data or your own CSV data

## What the workflow does

1. Lets you choose the data source first: Yahoo Finance or your own CSV files.
	Yahoo Finance data are aligned on shared timestamps; CSV inputs can use row order or an optional time/index column.
2. Lets you choose raw prices, simple returns, or log returns, then optionally smooth, downsample, and z-score the series.
3. Runs ADF / unit-root checks and can difference non-stationary series if you want.
4. Computes DTW on the processed series, then optionally runs lagged cross-correlation.
5. Runs Granger causality without plotting by default.
6. Sweeps TE and CCM across separate lag and embedding-dimension ranges, then prints text reports instead of heatmaps.
7. Runs shuffle or bootstrap surrogate tests for the best TE and CCM parameter combinations.

## Install

```bash
python -m pip install -r requirement.txt
```

## Run

```bash
python main.py
```

For the notebook demo, open `case_study/yfinance_example.ipynb` and run the single demo cell after editing the example tickers if needed.