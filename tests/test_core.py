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

def test_harmonize_differencing():
    from src.workflows import _harmonize_differencing
    rng = np.random.default_rng(0)
    walk = pd.Series(np.cumsum(rng.normal(size=100)))
    flat = walk.diff().dropna()
    one, two, info_one, info_two = _harmonize_differencing(
        walk, flat, {"diff_order": 1}, {"diff_order": 0}, True)
    assert info_two["diff_order"] == 1
    assert len(two) == len(flat) - 1

def test_preprocessing_config_validation():
    import pytest
    from src.preprocessing import PreprocessingConfig
    with pytest.raises(ValueError):
        PreprocessingConfig(smoothing_method="gausian", smoothing_window=5)
    with pytest.raises(ValueError):
        PreprocessingConfig(smoothing_method="gaussian")
    with pytest.raises(ValueError):
        PreprocessingConfig(smoothing_method="moving_average", smoothing_window=5, smoothing_sigma=1.0)
    PreprocessingConfig(smoothing_method="gaussian", smoothing_window=5, smoothing_sigma=1.0)
    PreprocessingConfig(smoothing_window=5)

def test_circular_shift_surrogates_are_rotations():
    from src.surrogate import run_surrogate_test
    rng = np.random.default_rng(0)
    x = np.cumsum(rng.normal(size=300))
    y = rng.normal(size=300)
    seen = []

    def f(a, b):
        seen.append(np.array(a, copy=True))
        return float(np.corrcoef(a, b)[0, 1])

    run_surrogate_test(f, x, y, n_surrogates=5, method="circular_shift", seed=1)
    assert len(seen) == 6
    for s in seen[1:]:
        assert any(np.array_equal(np.roll(x, k), s) for k in range(1, 300))


def test_surrogate_p_value_floor_and_effective():
    from src.surrogate import run_surrogate_test
    rng = np.random.default_rng(0)
    x = rng.normal(size=200)
    y = x + 0.1 * rng.normal(size=200)
    res = run_surrogate_test(lambda a, b: abs(float(np.corrcoef(a, b)[0, 1])), x, y,
                             n_surrogates=99, method="shuffle", seed=0)
    assert abs(res.p_value - 0.01) < 1e-12
    assert res.effective is not None and res.effective > 0.5