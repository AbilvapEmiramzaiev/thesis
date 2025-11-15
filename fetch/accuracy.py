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


def derive_outcome_yes(markets: pd.DataFrame) -> pd.DataFrame:
    # prices dont store clobtokenid. This function identifies outcome and then 
    # we can match prices to outcome (remember prices in different csv)
    df = markets[['id', "prob_yes", "prob_no"]].copy()
    df['id'] = pd.to_numeric(df['id'], errors="coerce").astype("Int64")

    outcome = pd.Series(np.nan, index=df.index, dtype=float)
    outcome = np.where(df["prob_yes"] >= 0.999, 1.0, outcome)
    outcome = np.where(df["prob_no"]  >= 0.999, 0.0, outcome)

    out = pd.DataFrame({
        'id': df['id'],
        'outcome_yes': pd.to_numeric(outcome, errors="coerce")
    })

    out = out.dropna(subset=['outcome_yes'])
    out = out.reset_index(drop=True)
    return out



def evaluate_markets(
    prices: pd.DataFrame,
    *,
    market_col_prices: str = "market_id",
) -> Tuple[Dict[str, float], pd.DataFrame]:
    """Compute accuracy, Brier, and log-loss by merging last p with outcomes.
    Returns (metrics_dict, merged_frame) where `merged_frame` has columns
    [market_id, p_last, outcome_yes] for the joined markets.
    """
    lastp = last_prob_by_market(prices, market_col=market_col_prices)
    y = prices['outcome'].astype(int).values
    p = (
        prices.sort_values("timestamp")               # sort so last is latest
            .groupby("market_id")
            .tail(1)                                # pick latest per group
            .reset_index(drop=True)
    )
    metrics = {
        "n": int(len(p)),
        "accuracy@0.5": accuracy(p, y, threshold=0.5),
        "brier": brier_score(p, y),
        "log_loss": log_loss(p, y),
    }
    return metrics, p[[market_col_prices, "p_last", 'outcome']]


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
    markets_lossers = read_markets_csv(f'{PROJECT_ROOT}/data/losser_binary_markets.csv')

    prices_lossers = pd.read_csv(f'{PROJECT_ROOT}/data/losser_binary_markets_prices.csv')
    prices_lossers['outcome'] = False
    prices = pd.read_csv(f'{PROJECT_ROOT}/data/market_prices.csv')
    prices['outcome'] = True
    
    all_markets = (
        pd.concat([
            markets_lossers,    
            markets[~markets['id'].isin(markets_lossers['id'])]
        ], ignore_index=True)
    )
    all_prices = (
        pd.concat([
            prices_lossers,    
            prices[~prices['market_id'].isin(prices_lossers['market_id'])]
        ], ignore_index=True)
    )
    #outcomes = derive_outcome_yes(all_markets) # find where YES/NO to understand what token prices represent

    metrics_all, eval_all = evaluate_markets(all_prices)
    # Evaluate losers subset
    loser_ids = pd.to_numeric(markets_lossers.get("id", pd.Series(dtype="Int64")), errors="coerce").astype("Int64")
    prices_losers = all_prices[ pd.to_numeric(all_prices["market_id"], errors="coerce").astype("Int64").isin(pd.unique(loser_ids.dropna())) ]
    metrics_losers, eval_losers = evaluate_markets(prices_losers, outcomes, market_col_prices="market_id", id_col_res="id", outcome_col="outcome_yes")
    # Evaluate non-losers subset
    nonloser_ids = set(pd.to_numeric(all_markets["id"], errors="coerce").astype("Int64")) - set(pd.unique(loser_ids.dropna()))
    prices_nonlosers = all_prices[ pd.to_numeric(all_prices["market_id"], errors="coerce").astype("Int64").isin(nonloser_ids) ]
    metrics_nonlosers, eval_nonlosers = evaluate_markets(prices_nonlosers, outcomes, market_col_prices="market_id", id_col_res="id", outcome_col="outcome_yes")
    # Compute ECE for overall set
    p_all = eval_all["p_last"].to_numpy(dtype=float)
    y_all = eval_all["outcome_yes"].to_numpy(dtype=int)
    ece_all = expected_calibration_error(p_all, y_all, n_bins=20)
    # Print summary
    def fmt(m):
        return (
            f"n={m['n']}  acc@0.5={m['accuracy@0.5']:.3f}  "
            f"brier={m['brier']:.4f}  log_loss={m['log_loss']:.4f}"
        )
    print("Overall:", fmt(metrics_all), f"  ece={ece_all:.4f}")
    print("Losers:", fmt(metrics_losers))
    print("Non-losers:", fmt(metrics_nonlosers))
    # Save detailed evaluation for inspection
    out_dir = Path(PROJECT_ROOT) / "data"
    (out_dir / "eval_all.csv").write_text(eval_all.to_csv(index=False))
    (out_dir / "eval_losers.csv").write_text(eval_losers.to_csv(index=False))
    (out_dir / "eval_nonlosers.csv").write_text(eval_nonlosers.to_csv(index=False))
    #markets = read_markets_csv(f'{PROJECT_ROOT}/data/categorical.csv')
    #prices = pd.read_csv(f'{PROJECT_ROOT}/data/market_prices_categorical.csv')