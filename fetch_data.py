# env: python>=3.10  libs: requests, pandas, numpy, scikit-learn, matplotlib

import requests, pandas as pd, numpy as np
from datetime import datetime, timezone
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss, log_loss

GAMMA_BASE = "https://gamma-api.polymarket.com"   # Gamma REST
CLOB_BASE  = "https://clob.polymarket.com"    # CLOB REST (public market data)

# 3) Build per-second bars & microstructure features
def to_second_bars(trades_df):
    trades_df["sec"] = trades_df["ts"].dt.floor("S")
    bars = trades_df.groupby("sec").agg(
        last_yes=("mid_yes","last"),
        vol=("size","sum"),
        n_trades=("size","count")
    ).reset_index()
    bars["ret_1s"] = bars["last_yes"].pct_change()
    bars["abs_change_bp"] = (bars["last_yes"].diff().abs()*10000)
    # order-flow imbalance example (needs sides): OFI = buys - sells per second
    return bars

# 4) Accuracy & calibration (post-resolution)
def evaluate_accuracy(df_predictions, df_resolutions):
    """
    df_predictions: columns [market_id, ts, p_yes]
    df_resolutions: columns [market_id, resolved_at, outcome_yes (0/1)]
    Evaluate with Brier, log loss, calibration, and tail-slice metrics.
    """
    # last probability before close/resolution cutoff
    last_p = (df_predictions
              .sort_values(["market_id","ts"])
              .groupby("market_id")
              .tail(1))
    data = last_p.merge(df_resolutions[["market_id","outcome_yes"]], on="market_id", how="inner")
    p = np.clip(data["p_yes"].values, 1e-6, 1-1e-6)
    y = data["outcome_yes"].values

    metrics = {
        "brier": float(brier_score_loss(y, p)),
        "logloss": float(log_loss(y, p)),
        "ece_10": float(expected_calibration_error(p, y, n_bins=10)),
        "ece_20": float(expected_calibration_error(p, y, n_bins=20)),
        "brier_tail_low": float(brier_score_loss(y[p<=0.10], p[p<=0.10])) if (p<=0.10).any() else None,
        "brier_tail_high": float(brier_score_loss(y[p>=0.90], p[p>=0.90])) if (p>=0.90).any() else None,
    }
    # reliability curve
    frac_pos, mean_pred = calibration_curve(y, p, n_bins=20, strategy="uniform")
    calib = pd.DataFrame({"mean_pred": mean_pred, "emp_freq": frac_pos})
    return metrics, calib

def expected_calibration_error(p, y, n_bins=20):
    bins = np.linspace(0,1,n_bins+1)
    ece = 0.0
    for i in range(n_bins):
        mask = (p>bins[i]) & (p<=bins[i+1])
        if mask.any():
            conf = p[mask].mean()
            acc  = y[mask].mean()
            ece += (mask.mean()) * abs(acc - conf)
    return ece

# 5) Speed-of-move (in seconds)
def speed_metrics(second_bars):
    """
    Burstiness around news: count how often |Δp| > k bps within N seconds.
    """
    sb = second_bars.copy()
    # example thresholds
    spikes = (sb["abs_change_bp"]>50).rolling(10).sum()   # >50bp within 10s window
    return {
        "spikes_per_hour": float((sb["abs_change_bp"]>50).mean()*3600),
        "median_time_between_trades_s": float(1/ (sb["n_trades"].mean()/1.0) if sb["n_trades"].mean()>0 else np.nan)
    }
