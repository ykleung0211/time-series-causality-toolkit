import sys
import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd
from src import PreprocessingConfig, AnalysisConfig, run_analysis_pipeline

print("default surrogate_method ->", AnalysisConfig().surrogate_method)

np.random.seed(0)
n = 1000
x, y = np.zeros(n), np.zeros(n)
for t in range(2, n):
    x[t] = 0.6 * x[t-1] + np.random.normal(scale=1.0)
    y[t] = 0.5 * x[t-1] + 0.3 * y[t-1] + np.random.normal(scale=1.0)
sx, sy = pd.Series(x, name="X_linear_driver"), pd.Series(y, name="Y_linear_driven")

method = sys.argv[1] if len(sys.argv) > 1 else "default"
kw = dict(
    run_lagged_cross_correlation=False, run_dtw=False, plot_dtw_alignment=False,
    run_granger=False, run_te=True, te_max_lag=2, te_max_embed_dim=3, run_te_surrogates=True,
    run_ccm=False, run_ccm_convergence=False, n_surrogates=60, surrogate_seed=0,
)
if method != "default":
    kw["surrogate_method"] = method
run_analysis_pipeline(
    sx, sy, "X_linear_driver", "Y_linear_driven",
    preprocessing_config=PreprocessingConfig(base_representation="raw", standardize=True),
    analysis_config=AnalysisConfig(**kw),
)