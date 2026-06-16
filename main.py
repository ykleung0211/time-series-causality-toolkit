import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from numpy.linalg import norm
import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="statsmodels.tsa.stattools")

from statsmodels.tsa.stattools import grangercausalitytests
from dtw import dtw
from dtw import *
from infomeasure import transfer_entropy
from crossmapy import ccm

def download_data():
    '''
    Download daily close prices for Nasdaq (^IXIC) and S&P 500 (^GSPC) 
    return a DataFrame with closing prices and log returns.
    '''
    tickers = ['^IXIC', '^GSPC']
    data = yf.download(tickers, start='2015-01-01', progress=False)['Close']

    data = data.dropna()
    data.columns = ['Nasdaq', 'S&P 500']

    # Compute log returns to remove strong trends and make the series more stationary 
    log_returns = np.log(data).diff().dropna() 
    log_returns.columns = ['Nasdaq_log_return', 'S&P 500_log_return']

    return data, log_returns



def smooth_series(series, window=5):
    # Simple moving average to smooth the series and reduce noise
    return series.rolling(window=window, center = True).mean().dropna()
    
def demo_smoothing(log_returns):
    nasdaq = log_returns['Nasdaq_log_return']
    sp500 = log_returns['S&P 500_log_return']

    smoothed_nasdaq = smooth_series(nasdaq)
    smoothed_sp500 = smooth_series(sp500)

    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    axes[0].plot(nasdaq.index, nasdaq, alpha=0.4, label = "Nasdaq Log Return (raw)")
    axes[0].plot(smoothed_nasdaq.index, smoothed_nasdaq, alpha=0.4, label = "Nasdaq Log Return (smoothed)")
    axes[0].legend()
       
    axes[1].plot(sp500.index, sp500, alpha=0.4, label = "S&P 500 Log Return (smoothed)")
    axes[1].plot(smoothed_sp500.index, smoothed_sp500, alpha=0.4, label = "S&P 500 Log Return (smoothed)")
    axes[1].legend()

    plt.tight_layout()
    plt.show()

    return smoothed_nasdaq, smoothed_sp500
    
   

def compute_dtw(series1, series2):
    ''' Compute the Dynamic Time Warping distance and alignment between two 1_D series.'''
        
    # DTW does not require the series to be of the same length, but we will use the smoothed versions of same length for better results.
    common_idx = series1.index.intersection(series2.index)
    series1_arr = series1.loc[common_idx].values.reshape(-1, 1)
    series2_arr = series2.loc[common_idx].values.reshape(-1, 1)

    alignment = dtw(series1_arr, series2_arr, dist_method = 'euclidean', keep_internals = True)
        
    dist = alignment.distance
    path = list(zip(alignment.index1, alignment.index2))
        
    print(f"DTW distance: {dist:.6f}")

    alignment.plot(type="twoway", offset = 1)
    plt.show()

    return dist, path, common_idx
    
def demo_dtw(smoothed_nasdaq, smoothed_sp500):
    dist, path, common_idx = compute_dtw(smoothed_nasdaq, smoothed_sp500)
        
    return dist
    


def prepare_granger_causality(nasdaq, sp500, max_lag=5):
    '''
    Build a two-column DataFrame for Granger causality test
    First column is Y, second column is X.
    We will test both directions by switching the order of columns.
    '''
    common_idx = nasdaq.index.intersection(sp500.index)
    df = pd.DataFrame({
            "Nasdaq_log_return": nasdaq.loc[common_idx],
            "SP500_log_return": sp500.loc[common_idx]
    }).dropna()

    return df
    
def compute_granger_causality(nasdaq, sp500, max_lag = 5):
    df = prepare_granger_causality(nasdaq, sp500, max_lag = max_lag)

    print("== Granger Causality Test: Does Nasdaq cause S&P 500? (Nasdaq -> S&P 500) ==")
    # Y = S&P 500, X = Nasdaq
    grangercausalitytests(df[["SP500_log_return", "Nasdaq_log_return"]], maxlag = max_lag, verbose = False)

    print("\n== Granger Causality Test: Does S&P 500 cause Nasdaq? (S&P 500 -> Nasdaq) ==")
    # Y = Nasdaq, X = S&P 500
    grangercausalitytests(df[["Nasdaq_log_return", "SP500_log_return"]], maxlag = max_lag, verbose = False)

   

def compute_te_ccm(nasdaq, sp500):
    common_idx = nasdaq.index.intersection(sp500.index)
    nasdaq_arr = nasdaq.loc[common_idx].values
    sp500_arr = sp500.loc[common_idx].values

    # Compute Transfer Entropy 
        
    ''' 
    For simplicity, we will use a small embedding dimension and lag. 
    In practice, you may want to optimize these parameters.
    '''
    embed_dim = 3
    tau = 1

    print("\n== Transfer Entropy (Nasdaq_log_return -> SP500_log_return) ==")
    te_xy = transfer_entropy(
        nasdaq_arr, sp500_arr,
        approach = "ordinal",
        embedding_dim = embed_dim,
        step_size = tau
    )
    print("TE Nasdaq -> S&P 500:", te_xy)

    print("\n== Transfer Entropy (SP500_log_return -> Nasdaq_log_return) ==")
    te_yx = transfer_entropy(
        sp500_arr, nasdaq_arr,
        approach = "ordinal",
        embedding_dim = embed_dim,
        step_size = tau
    )
    print("TE S&P 500 -> Nasdaq:", te_yx)

    # Compute Convergent Cross Mapping (CCM)
    data = np.column_stack([nasdaq_arr, sp500_arr])
    ccm_model = ccm.ConvergeCrossMapping(embed_dim=embed_dim, lag=tau)
    ccm_model.fit(data)
    scores = ccm_model.scores
    
    print("\n== CCM (Nasdaq_log_return -> SP500_log_return) ==")
    print("CCM Nasdaq -> S&P 500:", scores[1, 0])
    
    print("\n== CCM (SP500_log_return -> Nasdaq_log_return) ==")
    print("CCM S&P 500 -> Nasdaq:", scores[0, 1])

def shuffle_surrogate(series):
    ''' 
    Return a shuffled version of the series (destroys temoral structure but preserves distribution) for surrogate testing.
    '''
    shuffled = series.copy()
    np.random.shuffle(shuffled)
    return shuffled

def simple_te_surrogate_test(X, Y, num_surrogates=100):
    ''' 
    Compute TE(X->Y) for real data and for shuffled surrogates of X,
    then see where the real TE lies in that surrogate distribution.
    '''
    embed_dim = 3
    tau = 1

    real_te = transfer_entropy(
        X, Y, 
        approach="ordinal", 
        embedding_dim=embed_dim, 
        step_size=tau)
    surrogate_te = []
    for _ in range(num_surrogates):
        X_shuffled = shuffle_surrogate(X)
        te_surr = transfer_entropy(
            X_shuffled, Y, 
            approach="ordinal", 
            embedding_dim=embed_dim, 
            step_size=tau)
        surrogate_te.append(te_surr)

    surrogate_te = np.array(surrogate_te)
    p_value = (np.sum(surrogate_te >= real_te) + 1) / (num_surrogates + 1)  

    print("\nReal TE:", real_te)
    print("Surrogate TE mean:", surrogate_te.mean())
    print("Approximate p-value (one-sided):", p_value)

def simple_bootstrap_te(X, Y, num_bootstrap=100):
    ''' 
    Bootstrap TE by resampling time indexes with replacement
    '''
    embed_dim = 3
    tau = 1

    n = len(X)
    bootstrap_te = []
    for _ in range(num_bootstrap):
        idx = np.random.randint(0, n, size=n)
        X_boot = X[idx]
        Y_boot = Y[idx]
        te_boot = transfer_entropy(
            X_boot, Y_boot, 
            approach="ordinal", 
            embedding_dim=embed_dim, 
            step_size=tau)
        bootstrap_te.append(te_boot)

    bootstrap_te = np.array(bootstrap_te)
    print("\nBootstrap TE mean:", bootstrap_te.mean())
    print("Bootstrap TE 2.5% quantile:", np.percentile(bootstrap_te, 2.5))
    print("Bootstrap TE 97.5% quantile:", np.percentile(bootstrap_te, 97.5))

if __name__ == "__main__":
    _price, log_returns = download_data()
    smoothed_nasdaq, smoothed_sp500 = demo_smoothing(log_returns)
    dtw_distance = demo_dtw(smoothed_nasdaq, smoothed_sp500)            
    compute_granger_causality(smoothed_nasdaq, smoothed_sp500, max_lag = 5)
    compute_te_ccm(smoothed_nasdaq, smoothed_sp500)
    simple_te_surrogate_test(smoothed_nasdaq.values, smoothed_sp500.values, num_surrogates=100)
    simple_bootstrap_te(smoothed_nasdaq.values, smoothed_sp500.values, num_bootstrap=100)