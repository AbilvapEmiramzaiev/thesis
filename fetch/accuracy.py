# pip install requests pandas numpy python-dateutil matplotlib pyarrow
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from imports import *
from plot.plot_data import *
from plot.graphics import *

def brier_score(probs: Iterable[float], labels: Iterable[int]) -> float:
    p = np.asarray(list(probs), dtype=float)
    y = np.asarray(list(labels), dtype=float)
    if p.size == 0:
        return float("nan")
    return float(np.mean((p - y) ** 2))

def log_loss(probs: Iterable[float], labels: Iterable[int], eps: float = 1e-12) -> float:
    p = np.asarray(list(probs), dtype=float)
    y = np.asarray(list(labels), dtype=float)
    if p.size == 0:
        return float("nan")
    p = np.clip(p, eps, 1 - eps)
    ll = -(y * np.log(p) + (1 - y) * np.log(1 - p))
    return float(np.mean(ll))

def accuracy(probs: Iterable[float], labels: Iterable[int], threshold: float = 0.5) -> float:
    #simple check for correct prediction
    p = np.asarray(list(probs), dtype=float)
    y = np.asarray(list(labels), dtype=float)
    if p.size == 0:
        return float("nan")
    pred = (p >= threshold).astype(int)
    return float(np.mean(pred == y))


def last_prob_by_market(prices: pd.DataFrame, *, market_col: str = "market_id", p_col: str = "p", t_col: str = "t") -> pd.DataFrame:
    """Collapse price history to the last probability per market.
    Returns DataFrame with columns [market_id, p_last]. Assumes `t` is seconds since epoch
    or sortable ascending values.
    """
    df = prices[[market_col, t_col, p_col]].dropna().copy()
    df[market_col] = pd.to_numeric(df[market_col], errors="coerce").astype("Int64")
    # take the last per market by max timestamp
    idx = df.groupby(market_col)[t_col].idxmax()
    out = df.loc[idx, [market_col, p_col]].rename(columns={p_col: "p_last"}).reset_index(drop=True)
    return out

""" 
def derive_outcome_yes(markets: pd.DataFrame, *, id_col: str = "id") -> pd.DataFrame:
    cols = [id_col, "prob_yes", "prob_no"]
    for c in cols:
        if c not in markets.columns:
            return pd.DataFrame(columns=[id_col, "outcome_yes"]).astype({id_col: "Int64"})
    df = markets[[id_col, "prob_yes", "prob_no"]].copy()
    df[id_col] = pd.to_numeric(df[id_col], errors="coerce").astype("Int64")
    y = pd.Series(np.nan, index=df.index, dtype=float)
    y = np.where((df["prob_yes"]) >= 0.999, 1.0, y)
    y = np.where((df["prob_no"])  >= 0.999, 0.0, y)
    out = pd.DataFrame({id_col: df[id_col], "outcome_yes": pd.to_numeric(y, errors="coerce")}).dropna(subset=[id_col])
    return out """


def evaluate_markets(
    prices: pd.DataFrame,
    resolutions: pd.DataFrame,
    *,
    market_col_prices: str = "market_id",
    id_col_res: str = "id",
    outcome_col: str = "outcome_yes",
) -> Tuple[Dict[str, float], pd.DataFrame]:
    """Compute accuracy, Brier, and log-loss by merging last p with outcomes.
    Returns (metrics_dict, merged_frame) where `merged_frame` has columns
    [market_id, p_last, outcome_yes] for the joined markets.
    """
    lastp = last_prob_by_market(prices, market_col=market_col_prices)
    res = resolutions[[id_col_res, outcome_col]].dropna().copy()
    res[id_col_res] = pd.to_numeric(res[id_col_res], errors="coerce").astype("Int64")
    merged = lastp.merge(res, left_on=market_col_prices, right_on=id_col_res, how="inner")
    y = merged[outcome_col].astype(int).values
    p = merged["p_last"].values
    metrics = {
        "n": int(len(merged)),
        "accuracy@0.5": accuracy(p, y, threshold=0.5),
        "brier": brier_score(p, y),
        "log_loss": log_loss(p, y),
    }
    return metrics, merged[[market_col_prices, "p_last", outcome_col]]


def expected_calibration_error(p, y, n_bins=20):
    #ECE measures how far your predicted probabilities are from the actual frequency with which events occur.
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



if __name__ == "__main__":
    markets = read_markets_csv(f'{PROJECT_ROOT}/data/test_pipeline.csv')
    prices = pd.read_csv(f'{PROJECT_ROOT}/data/market_prices.csv')
    
    #markets = read_markets_csv(f'{PROJECT_ROOT}/data/categorical.csv')
    #prices = pd.read_csv(f'{PROJECT_ROOT}/data/market_prices_categorical.csv')