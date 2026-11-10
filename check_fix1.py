import sys
import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd

from src.preprocessing import smooth_series, standardize_series
from src.surrogate import run_surrogate_test
from src.causal_analysis import granger_direction_score

# Same simulation as notebook 03 (order of random calls matters)
np.random.seed(0)
n, delay_days = 500, 3
true_signal = np.cumsum(np.random.normal(scale=1.0, size=n))
liquid = true_signal + np.random.normal(scale=0.3, size=n)
jitter = np.random.randint(-1, 2, size=n)
delayed_index = np.clip(np.arange(n) - delay_days + jitter, 0, n - 1)
illiquid = true_signal[delayed_index] + np.random.normal(scale=0.8, size=n)

liquid_s = pd.Series(liquid, name="LiquidAsset")
illiquid_s = pd.Series(illiquid, name="IlliquidAssetStale")
sm_l, sm_i = smooth_series(liquid_s, window=5), smooth_series(illiquid_s, window=5)

mode = sys.argv[1] if len(sys.argv) > 1 else "sweep"

if mode == "sweep":
    x = standardize_series(sm_l).diff().dropna()
    y = standardize_series(sm_i).diff().dropna()
    x, y = x.align(y, join="inner")
    f = lambda s, t: granger_direction_score(pd.Series(t), pd.Series(s), maxlag=10)
    print(f"n = {len(x)}")
    for bl in [8, 15, 30, 50]:
        r = run_surrogate_test(f, x.values, y.values, n_surrogates=200,
                               method="block_bootstrap", seed=0, block_len=bl)
        print(f"block_len={bl:>3}  observed F={r.real:.2f}  "
              f"95% interval=[{r.ci_low:.2f}, {r.ci_high:.2f}]")

elif mode == "pipeline":
    from src import PreprocessingConfig, AnalysisConfig, run_analysis_pipeline
    from src.workflows import _granger_block_len
    cfg = AnalysisConfig(
        run_lagged_cross_correlation=False, run_dtw=False, plot_dtw_alignment=False,
        run_granger=True, granger_maxlag=10, run_granger_surrogates=True,
        run_te=False, run_ccm=False, run_ccm_convergence=False,
        n_surrogates=200, surrogate_seed=0, surrogate_method="block_bootstrap",
    )
    print("helper block_len for n=476, maxlag=10 ->", _granger_block_len(476, 10, cfg))
    run_analysis_pipeline(
        sm_l, sm_i, "LiquidAsset", "IlliquidAssetStale",
        preprocessing_config=PreprocessingConfig(base_representation="raw", standardize=True),
        analysis_config=cfg,
    )