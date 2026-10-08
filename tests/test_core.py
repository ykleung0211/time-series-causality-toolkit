import numpy as np
import pandas as pd

from src.causal_analysis import granger_direction_score


def linear_xy(n=1500, seed=0):
    rng = np.random.default_rng(seed)
    x = np.zeros(n)
    y = np.zeros(n)
    for t in range(1, n):
        x[t] = 0.6 * x[t - 1] + rng.normal()
        y[t] = 0.5 * x[t - 1] + 0.3 * y[t - 1] + rng.normal()
    return pd.Series(x), pd.Series(y)


def test_granger_direction():
    x, y = linear_xy()
    forward = granger_direction_score(y, x, 3)   # target first, source second
    reverse = granger_direction_score(x, y, 3)
    assert forward > 5 * reverse

def test_lcc_uses_index_not_position():
    from src.preprocessing import lagged_cross_correlation
    rng = np.random.default_rng(0)
    base = pd.Series(rng.normal(size=400))
    delayed = base.shift(3).iloc[4:]          # right[t] = left[t-3]; index starts at 4
    out = lagged_cross_correlation(base, delayed, max_lag=8)
    peak = int(out.loc[out["correlation"].abs().idxmax(), "lag"])
    assert peak == 3

def test_granger_frame_joins_on_index():
    from src.causal_analysis import _prepare_granger_frame
    a = pd.Series(np.arange(10.0), index=range(10))
    b = pd.Series(np.arange(10.0), index=range(3, 13))
    frame = _prepare_granger_frame(a, b)
    assert len(frame) == 7
    assert (frame["target"].values == np.arange(3, 10)).all()
    assert (frame["source"].values == np.arange(0, 7)).all()