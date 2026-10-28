import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd

from src import AnalysisConfig, PreprocessingConfig, run_analysis_pipeline


def _linear_pair(n=250, seed=0):
    rng = np.random.default_rng(seed)
    x = np.zeros(n)
    y = np.zeros(n)
    for t in range(1, n):
        x[t] = 0.6 * x[t - 1] + rng.normal()
        y[t] = 0.5 * x[t - 1] + 0.3 * y[t - 1] + rng.normal()
    return pd.Series(x), pd.Series(y)


def test_pipeline_smoke_all_surrogate_methods():
    x, y = _linear_pair()
    config = AnalysisConfig(
        run_dtw=False, run_ccm_convergence=False,
        run_granger=True, granger_maxlag=3, run_granger_surrogates=True,
        run_te=True, te_max_lag=2, te_max_embed_dim=3, run_te_surrogates=True,
        run_ccm=True, ccm_max_lag=2, ccm_max_embed_dim=3, run_ccm_surrogates=True,
        surrogate_method="all", n_surrogates=5, surrogate_seed=0,
    )
    out = run_analysis_pipeline(
        x, y, "X", "Y",
        preprocessing_config=PreprocessingConfig(base_representation="raw", standardize=True),
        analysis_config=config,
    )
    assert out["granger"] is not None
    assert out["te_summary"]["one_to_two"] is not None
    assert out["ccm_summary"]["one_to_two"] is not None