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

def test_best_scores_are_grid_max():
    from src.causal_analysis import best_te_score, compute_te, best_ccm_score, compute_ccm
    rng = np.random.default_rng(0)
    x = rng.normal(size=400)
    y = np.roll(x, 1) + 0.5 * rng.normal(size=400)
    dims, lags = [2, 3], [1, 2]
    te_singles = [compute_te(x, y, d, l) for d in dims for l in lags]
    assert abs(best_te_score(x, y, dims, lags) - max(te_singles)) < 1e-12
    ccm_singles = [compute_ccm(x, y, d, l)[0] for d in dims for l in lags]
    assert abs(best_ccm_score(x, y, dims, lags, 0) - max(ccm_singles)) < 1e-12

def test_block_bootstrap_resamples_pairs_jointly():
    from src.surrogate import run_surrogate_test
    rng = np.random.default_rng(0)
    x = np.cumsum(rng.normal(size=200))
    y = 2.0 * x
    res = run_surrogate_test(lambda a, b: float(np.max(np.abs(b - 2.0 * a))), x, y,
                             n_surrogates=20, method="block_bootstrap", seed=0)
    assert res.real == 0.0
    assert np.all(res.surrogates == 0.0)
    assert res.ci_low is not None and res.ci_high is not None


def test_block_bootstrap_interval_covers_estimate():
    from src.surrogate import run_surrogate_test
    rng = np.random.default_rng(0)
    x = rng.normal(size=500)
    y = 0.5 * x + rng.normal(size=500)
    res = run_surrogate_test(lambda a, b: float(np.corrcoef(a, b)[0, 1]), x, y,
                             n_surrogates=200, method="block_bootstrap", seed=0)
    assert res.ci_low < res.real < res.ci_high

def test_ccm_convergence_uses_random_windows():
    from src.causal_analysis import sweep_ccm_convergence
    rng = np.random.default_rng(0)
    x = rng.normal(size=300)
    y = np.roll(x, 1) + 0.3 * rng.normal(size=300)
    frame = sweep_ccm_convergence(x, y, 2, 1, library_fractions=(0.5, 1.0), n_repeats=3, seed=0)
    assert list(frame["fraction"]) == [0.5, 1.0]
    assert frame.loc[0, "n_windows"] == 3
    assert frame.loc[1, "n_windows"] == 1


def test_dtw_window_constraint():
    from src.causal_analysis import compute_dtw_sequence, extract_warping_path
    rng = np.random.default_rng(0)
    base = np.cumsum(rng.normal(size=120))
    a = pd.Series(base)
    b = pd.Series(np.roll(base, 4) + 0.1 * rng.normal(size=120))
    compute_dtw_sequence(a, b)  # default stays unconstrained
    tight = compute_dtw_sequence(a, b, window_size=3)
    i, j = extract_warping_path(tight)
    assert np.max(np.abs(j - i)) <= 3