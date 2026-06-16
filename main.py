import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="statsmodels.tsa.stattools")

from statsmodels.tsa.stattools import grangercausalitytests
from dtw import dtw
from infomeasure import transfer_entropy
from crossmapy import ccm


def get_ticker_input():
    """
    Ask the user for two ticker symbols. Provide guidance on examples.
    """
    print("Enter two ticker symbols to analyze. Examples: ^IXIC (Nasdaq), ^GSPC (S&P 500), AAPL, MSFT.")
    t1 = input("First ticker: ").strip()
    t2 = input("Second ticker: ").strip()
    return t1, t2


def get_ticker_name(ticker):
    """
    Use yfinance to fetch a human friendly short name. Fall back to the symbol.
    """
    try:
        info = yf.Ticker(ticker).info
        name = info.get('shortName') or info.get('longName')
        return name if name else ticker
    except Exception:
        return ticker


def get_date_range():
    """
    Prompt user for start and end dates in YYYY-MM-DD format. Returns (start, end).
    """
    print("Specify analysis date range. Press Enter to use default (2015-01-01 to today).")
    s = input("Start date (YYYY-MM-DD): ").strip()
    e = input("End date (YYYY-MM-DD): ").strip()
    if not s:
        s = '2015-01-01'
    if not e:
        e = pd.Timestamp.today().strftime('%Y-%m-%d')
    # basic validation
    try:
        _s = pd.to_datetime(s)
        _e = pd.to_datetime(e)
        if _s > _e:
            print("Start date is after end date — swapping.")
            _s, _e = _e, _s
        return _s.strftime('%Y-%m-%d'), _e.strftime('%Y-%m-%d')
    except Exception:
        print("Invalid date(s). Using defaults.")
        return '2015-01-01', pd.Timestamp.today().strftime('%Y-%m-%d')


def download_data(t1, t2, start, end):
    """
    Download adjusted close prices for two tickers and compute log returns.
    Returns (prices_df, log_returns_df)
    """
    tickers = [t1, t2]
    data = yf.download(tickers, start=start, end=end, progress=False)['Close']
    data = data.dropna()
    if data.shape[1] != 2:
        # when single column returned, try to align
        raise RuntimeError('Failed to download two tickers. Check symbols.')
    data.columns = [t1, t2]
    lr = np.log(data).diff().dropna()
    lr.columns = [f"{t1}_log_return", f"{t2}_log_return"]
    return data, lr


def smooth_series(series, window=5):
    return series.rolling(window=window, center=True).mean().dropna()


def downsample_series(series, step=None, freq=None):
    """
    Downsample by integer step (take every N-th sample) or by pandas offset alias (freq='W').
    """
    if freq:
        return series.resample(freq).last().dropna()
    if step and step > 1:
        return series.iloc[::step]
    return series


def compute_dtw(series1, series2):
    common_idx = series1.index.intersection(series2.index)
    a = series1.loc[common_idx].values.reshape(-1, 1)
    b = series2.loc[common_idx].values.reshape(-1, 1)
    alignment = dtw(a, b, dist_method='euclidean', keep_internals=True)
    # return the full alignment object so we can plot alignment path/cost if desired
    return alignment


def prepare_granger_df(s1, s2, t1, t2):
    idx = s1.index.intersection(s2.index)
    df = pd.DataFrame({f"{t1}_log_return": s1.loc[idx], f"{t2}_log_return": s2.loc[idx]}).dropna()
    return df


def compute_granger_metric(s1, s2, maxlag):
    # compute Granger causality and return a numeric strength (max F-stat across lags)
    df = pd.concat([s1, s2], axis=1)
    # grangercausalitytests expects two-column array [Y, X]
    # We'll compute for both directions outside when needed.
    # Here just a helper placeholder
    return grangercausalitytests


def compute_te(s1_arr, s2_arr, embed_dim, tau):
    val = transfer_entropy(s1_arr, s2_arr, approach='ordinal', embedding_dim=embed_dim, step_size=tau)
    return float(val)


def compute_ccm(s1_arr, s2_arr, embed_dim, tau):
    data = np.column_stack([s1_arr, s2_arr])
    model = ccm.ConvergeCrossMapping(embed_dim=embed_dim, lag=tau)
    model.fit(data)
    scores = model.scores
    # return prediction skill for direction s1 -> s2 as scores[1,0]
    return float(scores[1, 0]), float(scores[0, 1])


def granger_strength(s1, s2, maxlag, verbose=False):
    # returns (s1->s2_strength, s2->s1_strength) using ssr_ftest F-stat
    df = pd.concat([s1, s2], axis=1).dropna()
    # s1 -> s2: Y = s2, X = s1
    res12 = grangercausalitytests(df[[df.columns[1], df.columns[0]]], maxlag=maxlag, verbose=verbose)
    res21 = grangercausalitytests(df[[df.columns[0], df.columns[1]]], maxlag=maxlag, verbose=verbose)
    # pick the max F-stat across lags
    def max_f(res):
        vals = []
        for lag, out in res.items():
            ft = out[0].get('ssr_ftest')
            if ft:
                vals.append(float(ft[0]))
        return max(vals) if vals else 0.0
    return max_f(res12), max_f(res21)


def surrogate_test_measure(real_func, X, Y, n_surr=200, method='shuffle', **kwargs):
    """
    real_func should be a callable that returns a numeric measure for X->Y using kwargs.
    method: 'shuffle' or 'bootstrap'
    Returns dict with real value and surrogate array.
    """
    real_val = real_func(X, Y, **kwargs)
    svals = []
    n = len(X)
    for _ in range(n_surr):
        if method == 'shuffle':
            Xs = X.copy()
            np.random.shuffle(Xs)
        else:
            idx = np.random.randint(0, n, size=n)
            Xs = X[idx]
        try:
            v = real_func(Xs, Y, **kwargs)
        except Exception:
            v = np.nan
        svals.append(float(v))
    svals = np.array([v for v in svals if not np.isnan(v)])
    return {'real': float(real_val), 'surr': svals}


def print_surrogate_summary(name, direction, res):
    real = res['real']
    s = res['surr']
    mean = s.mean()
    std = s.std()
    p_two_sided = (np.sum(np.abs(s) >= abs(real)) + 1) / (len(s) + 1)
    q025 = np.percentile(s, 2.5)
    q975 = np.percentile(s, 97.5)
    ci = (q025, q975)

    print('\n' + '-'*80)
    print(f"Surrogate test for {name} direction {direction}")
    print('-'*80)
    print(f"Real value:           {real:.6f}")
    print(f"Surrogate mean:       {mean:.6f}")
    print(f"Surrogate std dev:    {std:.6f}")
    print(f"p-value (two-sided):  {p_two_sided:.4f}")
    print(f"2.5% quantile:        {q025:.6f}")
    print(f"97.5% quantile:       {q975:.6f}")
    print(f"95% CI:               [{ci[0]:.6f}, {ci[1]:.6f}]")
    print('-'*80)


if __name__ == '__main__':
    # 1) User inputs
    t1, t2 = get_ticker_input()
    name1 = get_ticker_name(t1)
    name2 = get_ticker_name(t2)
    start, end = get_date_range()

    print(f"\nDownloading {t1} ({name1}) and {t2} ({name2}) from {start} to {end}...")
    prices, log_returns = download_data(t1, t2, start, end)
    length = len(log_returns)
    print(f"Sequence length: {length} observations")

    # ask downsampling and smoothing
    do_down = input("Do you want to downsample? [y/N]: ").strip().lower() == 'y'
    ds_step = None
    ds_freq = None
    if do_down:
        print("Choose downsampling method: 1) every N-th sample  2) pandas frequency alias (e.g. 'W' for weekly)")
        m = input("Method [1/2]: ").strip() or '1'
        if m == '1':
            ds_step = int(input("Downsample step N (e.g. 2): ").strip() or 2)
        else:
            ds_freq = input("Pandas offset alias (e.g. W, M): ").strip() or 'W'
        plot_ds = input("Plot downsampled series? [y/N]: ").strip().lower() == 'y'
    else:
        plot_ds = False

    do_smooth = input("Do you want to smooth the series? [y/N]: ").strip().lower() == 'y'
    smooth_window = 5
    if do_smooth:
        smooth_window = int(input("Smoothing window size (odd integer, default 5): ").strip() or 5)

    # apply downsampling and smoothing to log_returns
    s1 = log_returns[f"{t1}_log_return"].copy()
    s2 = log_returns[f"{t2}_log_return"].copy()

    if do_down:
        s1_ds = downsample_series(s1, step=ds_step, freq=ds_freq)
        s2_ds = downsample_series(s2, step=ds_step, freq=ds_freq)
        if plot_ds:
            plt.figure(figsize=(10,4))
            plt.plot(s1.index, s1, alpha=0.4, label=f"{name1} original")
            plt.plot(s1_ds.index, s1_ds, '-o', label=f"{name1} downsampled")
            plt.legend(); plt.title(f"Downsampled {name1}"); plt.show()
        s1, s2 = s1_ds, s2_ds
        print(f"After downsampling: {len(s1)} observations")

    plot_smooth = False
    if do_smooth:
        # capture pre-smoothing for plotting comparison
        s1_pre = s1.copy()
        s2_pre = s2.copy()
        s1 = smooth_series(s1, window=smooth_window)
        s2 = smooth_series(s2, window=smooth_window)
        print(f"After smoothing: {len(s1)} observations")
        plot_smooth = input('Plot smoothing result? [y/N]: ').strip().lower() == 'y'
        if plot_smooth:
            # align indices for plotting
            idx = s1_pre.index.intersection(s1.index)
            plt.figure(figsize=(10,4))
            plt.plot(s1_pre.loc[idx].index, s1_pre.loc[idx].values, alpha=0.5, label=f"{name1} pre-smooth")
            plt.plot(s1.loc[idx].index, s1.loc[idx].values, '-r', label=f"{name1} smoothed")
            plt.legend(); plt.title(f"Smoothing comparison {name1}"); plt.show()

            idx2 = s2_pre.index.intersection(s2.index)
            plt.figure(figsize=(10,4))
            plt.plot(s2_pre.loc[idx2].index, s2_pre.loc[idx2].values, alpha=0.5, label=f"{name2} pre-smooth")
            plt.plot(s2.loc[idx2].index, s2.loc[idx2].values, '-r', label=f"{name2} smoothed")
            plt.legend(); plt.title(f"Smoothing comparison {name2}"); plt.show()

    # DTW
    alignment = compute_dtw(s1, s2)
    dtw_dist = alignment.distance
    print(f"\nDTW distance: {dtw_dist:.6f}")
    plot_dtw = input('Plot DTW alignment graph? [y/N]: ').strip().lower() == 'y'
    if plot_dtw:
        try:
            common_idx = s1.index.intersection(s2.index)
            # try common path attributes
            try:
                i1 = np.array(alignment.index1)
                i2 = np.array(alignment.index2)
                path = np.vstack([i1, i2]).T
            except Exception:
                try:
                    path = np.array(alignment.path)
                    i1 = path[:,0].astype(int)
                    i2 = path[:,1].astype(int)
                except Exception:
                    path = None
            # plot series and sample matching lines (limit number to avoid clutter)
            if path is not None and len(path) > 0:
                nmax = min(200, len(path))
                samp_idx = np.linspace(0, len(path)-1, nmax).astype(int)
                plt.figure(figsize=(12,5))
                plt.plot(common_idx, s1.loc[common_idx].values, label=f"{name1}")
                plt.plot(common_idx, s2.loc[common_idx].values, label=f"{name2}")
                for j in samp_idx:
                    x1 = common_idx[i1[j]]
                    x2 = common_idx[i2[j]]
                    y1 = s1.loc[x1]
                    y2 = s2.loc[x2]
                    plt.plot([x1, x2], [y1, y2], color='gray', alpha=0.3)
                plt.legend(); plt.title(f"DTW alignment lines ({name1} ↔ {name2})"); plt.show()
            # try cost matrix
            try:
                C = alignment.costMatrix
                plt.figure(figsize=(6,6))
                plt.imshow(C.T, origin='lower', aspect='auto', cmap='viridis')
                if path is not None:
                    pts = np.array(path)
                    plt.plot(pts[:,0], pts[:,1], '-r')
                plt.title('DTW cost matrix with warping path'); plt.xlabel(name1); plt.ylabel(name2); plt.show()
            except Exception:
                pass
        except Exception as e:
            print('Unable to plot DTW alignment:', e)

    # Granger maxlag
    maxlag = int(input("\nChoose maxlag for Granger causality (default 5): ").strip() or 5)
    print('\nRunning Granger causality (verbose output will be shown)...')
    # prepare original (not-array) series for granger
    df_for_granger = prepare_granger_df(s1, s2, t1, t2)
    grangercausalitytests(df_for_granger[[f"{t2}_log_return", f"{t1}_log_return"]], maxlag=maxlag, verbose=True)
    grangercausalitytests(df_for_granger[[f"{t1}_log_return", f"{t2}_log_return"]], maxlag=maxlag, verbose=True)

    # TE and CCM params
    te_embed = int(input("\nTE embedding dimension (default 3): ").strip() or 3)
    te_tau = int(input("TE lag/tau (default 1): ").strip() or 1)
    ccm_embed = int(input("\nCCM embedding dimension (default 3): ").strip() or 3)
    ccm_tau = int(input("CCM lag/tau (default 1): ").strip() or 1)

    # Compute TE
    arr1 = s1.values
    arr2 = s2.values
    te_12 = compute_te(arr1, arr2, te_embed, te_tau)
    te_21 = compute_te(arr2, arr1, te_embed, te_tau)
    print('\n' + '='*80)
    print('TRANSFER ENTROPY RESULTS')
    print('='*80)
    print(f"{name1} → {name2}: {te_12:.6f}")
    print(f"{name2} → {name1}: {te_21:.6f}")

    # Compute CCM
    ccm_12, ccm_21 = compute_ccm(arr1, arr2, ccm_embed, ccm_tau)
    print('\n' + '='*80)
    print('CONVERGENT CROSS MAPPING RESULTS')
    print('='*80)
    print(f"{name1} → {name2}: {ccm_12:.6f}")
    print(f"{name2} → {name1}: {ccm_21:.6f}")

    # Surrogate testing for TE, CCM, and Granger
    do_shuffle = input('\nDo you want to run shuffle surrogate tests? [y/N]: ').strip().lower() == 'y'
    do_boot = input('Do you want to run bootstrap surrogate tests? [y/N]: ').strip().lower() == 'y'
    if do_shuffle or do_boot:
        n_surr = int(input('Number of surrogates (default 200): ').strip() or 200)

    # Run shuffle surrogates first (ordered: Granger, TE, CCM)
    if do_shuffle:
        print('\nRunning shuffle surrogate tests (this will randomize X series)')
        # Granger shuffle
        def granger_func(X, Y, maxlag=maxlag):
            sX = pd.Series(X)
            sY = pd.Series(Y)
            a, b = granger_strength(sX, sY, maxlag, verbose=False)
            return a
        gr12_res = surrogate_test_measure(granger_func, arr1, arr2, n_surr, method='shuffle')
        gr21_res = surrogate_test_measure(granger_func, arr2, arr1, n_surr, method='shuffle')
        print_surrogate_summary('Granger (max F)', f"{name1} → {name2}", gr12_res)
        print_surrogate_summary('Granger (max F)', f"{name2} → {name1}", gr21_res)

        # TE shuffle
        te12_res = surrogate_test_measure(lambda X, Y, **kw: compute_te(X, Y, te_embed, te_tau), arr1, arr2, n_surr, method='shuffle')
        te21_res = surrogate_test_measure(lambda X, Y, **kw: compute_te(X, Y, te_embed, te_tau), arr2, arr1, n_surr, method='shuffle')
        print_surrogate_summary('Transfer Entropy', f"{name1} → {name2}", te12_res)
        print_surrogate_summary('Transfer Entropy', f"{name2} → {name1}", te21_res)

        # CCM shuffle
        ccm12_res = surrogate_test_measure(lambda X, Y, **kw: compute_ccm(X, Y, ccm_embed, ccm_tau)[0], arr1, arr2, n_surr, method='shuffle')
        ccm21_res = surrogate_test_measure(lambda X, Y, **kw: compute_ccm(X, Y, ccm_embed, ccm_tau)[1], arr1, arr2, n_surr, method='shuffle')
        print_surrogate_summary('CCM', f"{name1} → {name2}", ccm12_res)
        print_surrogate_summary('CCM', f"{name2} → {name1}", ccm21_res)

    # Run bootstrap surrogates (ordered: Granger, TE, CCM)
    if do_boot:
        print('\nRunning bootstrap surrogate tests (resampling with replacement)')
        # Granger bootstrap
        def granger_func_b(X, Y, maxlag=maxlag):
            sX = pd.Series(X)
            sY = pd.Series(Y)
            a, b = granger_strength(sX, sY, maxlag, verbose=False)
            return a
        gr12_res_b = surrogate_test_measure(granger_func_b, arr1, arr2, n_surr, method='bootstrap')
        gr21_res_b = surrogate_test_measure(granger_func_b, arr2, arr1, n_surr, method='bootstrap')
        print_surrogate_summary('Granger (max F) (bootstrap)', f"{name1} → {name2}", gr12_res_b)
        print_surrogate_summary('Granger (max F) (bootstrap)', f"{name2} → {name1}", gr21_res_b)

        # TE bootstrap
        te12_res_b = surrogate_test_measure(lambda X, Y, **kw: compute_te(X, Y, te_embed, te_tau), arr1, arr2, n_surr, method='bootstrap')
        te21_res_b = surrogate_test_measure(lambda X, Y, **kw: compute_te(X, Y, te_embed, te_tau), arr2, arr1, n_surr, method='bootstrap')
        print_surrogate_summary('Transfer Entropy (bootstrap)', f"{name1} → {name2}", te12_res_b)
        print_surrogate_summary('Transfer Entropy (bootstrap)', f"{name2} → {name1}", te21_res_b)

        # CCM bootstrap
        ccm12_res_b = surrogate_test_measure(lambda X, Y, **kw: compute_ccm(X, Y, ccm_embed, ccm_tau)[0], arr1, arr2, n_surr, method='bootstrap')
        ccm21_res_b = surrogate_test_measure(lambda X, Y, **kw: compute_ccm(X, Y, ccm_embed, ccm_tau)[1], arr1, arr2, n_surr, method='bootstrap')
        print_surrogate_summary('CCM (bootstrap)', f"{name1} → {name2}", ccm12_res_b)
        print_surrogate_summary('CCM (bootstrap)', f"{name2} → {name1}", ccm21_res_b)

    print('\nAll done.')
