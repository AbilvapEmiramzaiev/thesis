# pip install requests pandas numpy python-dateutil matplotlib pyarrow
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from imports import *
from plot.plot_data import *
from plot.graphics import *
from fetch.tail_end_func import find_tailend_markets
from fetch.filtering import _market_resolution_time

DAYS_BEFORE = 7
MIN_DURATION = 30
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
    y = np.asarray(list(labels), dtype=bool)
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

def relative_brier(prices, markets, days, p_col="p", outcome_col="outcome"):
    #works correct tested 24.11
    prices = df_time_to_datetime(prices, 't')


    df = prices.merge(
        markets[["id", "prob_yes"]].rename(columns={"id": "market_id"}),
        on="market_id",
        how="inner"
    )
    if outcome_col not in df.columns:
        df[outcome_col] = (df["prob_yes"] == 1.0).astype(int)


    df = (
        df.sort_values("t")
          .groupby("market_id")
          .tail(days)  
          .reset_index(drop=True)
    )

    # Create day index (e.g., days before resolve)
    df["day"] = df.groupby("market_id")["t"].transform(
        lambda x: (x.max() - x).dt.days
    )

    # Compute daily Brier scores
    df["relative_brier"] = (df[p_col] - df[outcome_col])**2


    # Step 4: sum across all days each market existed
    summed_brier_score = df.groupby("market_id")["relative_brier"].sum()

    # Step 5: divide by total days for that market
    #total_days = df.groupby("market_id")["day"].nunique()

    result = (summed_brier_score / (days)).reset_index()
    result.columns = ["market_id", "relative_brier"]

    return result

def evaluate_markets(
    prices: pd.DataFrame,
    *,
    market_col_prices: str = "market_id",
) -> Tuple[Dict[str, float], pd.DataFrame]:
    #Tested, calculates correctly
    p = (
        prices.sort_values("t")               # sort so last is latest
            .groupby("market_id")
            .tail(1)                                # pick latest per group
            .reset_index(drop=True)
    )
    metrics = {
        "n": int(len(p)),
        "accuracy@0.5": accuracy(p["p"], p["outcome"], threshold=0.5),
        "brier": brier_score(p["p"], p["outcome"]),
        "log_loss": log_loss(p["p"], p["outcome"]),
    }
    return metrics

def pick_24h_before(
    group: pd.DataFrame,
    days_before: float = DAYS_BEFORE,
    resolve_col: str = "resolve_time",
) -> pd.Series:
    resolve_time = None
    if resolve_col in group.columns:
        resolve_time = pd.to_datetime(group[resolve_col].iloc[0], utc=True, errors="coerce")
    if pd.isna(resolve_time):
        resolve_time = group["t"].max()

    target = resolve_time - pd.Timedelta(days=days_before)
    group_sorted = group.sort_values("t")
    before_target = group_sorted[group_sorted["t"] <= target]

    if before_target.empty:
        fallback = group_sorted.iloc[0].copy()
        if "p" in fallback:
            fallback["p"] = np.nan
        return fallback

    return before_target.iloc[-1]

def evaluate_markets_bucketed(
    prices: pd.DataFrame,
    *,
    market_col_prices: str = "market_id",
    p_col: str = "p",
    outcome_col: str = "outcome",
    days_before: float = DAYS_BEFORE,
    # default buckets: (0.80, 0.85, 0.90) etc., expressed as probabilities
    bins: Iterable[float] = (0.0, 0.80, 0.85, 0.90, 0.95, 1.0),
) -> pd.DataFrame:
    p = (
        prices.sort_values("t")
        .groupby(market_col_prices)
        .apply(lambda group: pick_24h_before(group, days_before=days_before))
        #.tail(1)
        .reset_index(drop=True)
    )
    # ensure we work on numeric arrays
    probs = pd.to_numeric(p[p_col], errors="coerce")
    labels = pd.to_numeric(p[outcome_col], errors="coerce")
    mask_valid = probs.notna() & labels.notna()
    p = p.loc[mask_valid].reset_index(drop=True)
    probs = probs.loc[mask_valid].to_numpy(dtype=float)
    labels = labels.loc[mask_valid].to_numpy(dtype=int)
    
    bins = np.arange(0.05, 1.05, 0.05)
    bins_arr = np.asarray(list(bins), dtype=float)

    # assign bucket labels using pandas.cut
    p_bucketed = p.copy()
    p_bucketed["bucket"] = pd.cut(
        probs,
        bins=bins_arr,
        right=True,
        include_lowest=True,
    )

    rows = []
    for interval, group in p_bucketed.groupby("bucket", observed=True):
        if interval is None or group.empty:
            continue
        g_probs = group[p_col].to_numpy(dtype=float)
        g_labels = group[outcome_col].to_numpy(dtype=int)
        #group.to_csv(f'props-{interval}') #TODO: THIS IS NICE DEBUGGING
        rows.append(
            {
                "bucket": str(interval),
                "p_min": float(interval.left),
                "p_max": float(interval.right),
                "n": int(len(group)),
                "mean_pred": float(g_probs.mean()),
                "freq_pos": float(g_labels.mean()),
                "accuracy@0.5": accuracy(g_probs, g_labels, threshold=0.5),
                "brier": brier_score(g_probs, g_labels),
                "log_loss": log_loss(g_probs, g_labels),
            }
        )

    return pd.DataFrame(rows).sort_values("p_min").reset_index(drop=True)


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

def accuracy_all_markets(
    markets: pd.DataFrame,
    prices: pd.DataFrame,
    title: str = "Calibration plot (Reliability curve)",
    days_before: int = DAYS_BEFORE,
    #marketPath=f'{PROJECT_ROOT}/data/test_pipeline.csv',
    #pricesPath=f'{PROJECT_ROOT}/data/binary_yes_prices.csv'
):
    #markets = read_markets_csv(marketPath)
    #marketsForResolution = filter_by_duration(markets, DAYS_BEFORE)
    #prices = pd.read_csv(pricesPath)
    prices["t"] = pd.to_datetime(prices["t"],unit="s", utc=True)
    withOutcome = prices.merge(
        markets[['id', 'prob_yes','prob_no', 't_resolve']],
        left_on='market_id',
        right_on='id'
    )
    withOutcome = withOutcome.rename(columns={"t_resolve": "resolve_time"})
    withOutcome["resolve_time"] = pd.to_datetime(withOutcome["resolve_time"], utc=True, errors="coerce")
    withOutcome['outcome'] = (withOutcome['prob_yes'] >= 0.95).astype(int)
    grouped_outcomes = withOutcome.groupby('market_id')['outcome'].max()
    n_true = int(grouped_outcomes.sum())
    n_false = int(len(grouped_outcomes) - n_true)
  
    metrics_all = evaluate_markets_bucketed(withOutcome, days_before=days_before)
    
   # metrics_1w = evaluate_markets_bucketed(withOutcome, days_before=7)
   # metrics_1m = evaluate_markets_bucketed(withOutcome,days_before=30)
   # metrics_3m = evaluate_markets_bucketed(withOutcome,days_before=90)

    legend_extra = f"yes={n_true}, no={n_false}\n days before resolution = {DAYS_BEFORE}"
    fig, ax = plt.subplots()
    # TODO: One line case
    plot_calibration_line(ax, metrics_all, 'mean_pred', 'freq_pos', count_col='n', legend_extra=legend_extra)
    
    #TODO: different lines case
    #plot_calibration_line(ax, metrics_all, 'mean_pred', 'freq_pos', label=f'martkets 1day')
    #plot_calibration_line(ax, metrics_1w, 'mean_pred', 'freq_pos', label=f'martkets 1week')
    #plot_calibration_line(ax, metrics_1m, 'mean_pred', 'freq_pos', label=f'martkets 1month')
    #plot_calibration_line(ax, metrics_3m, 'mean_pred', 'freq_pos', label=f'martkets 3month')

    graphic_calibration(ax, title=title)
    print(metrics_all)


def accuracy_tailend_markets(
    tailend_markets: pd.DataFrame,
    tailend_prices: pd.DataFrame,
    title: str = "Tail-end calibration",
    days_before: int = DAYS_BEFORE,
) -> None:
    """Calibration plot where correctness is derived from tailend_label."""
    if tailend_markets.empty or tailend_prices.empty:
        print("No tail-end markets/prices provided.")
        return

    # Keep resolved markets only
    resolved = tailend_markets.copy()
  #  if "t_resolve" not in resolved.columns:
  #      resolved["t_resolve"] = resolved.apply(_market_resolution_time, axis=1)

    prices = tailend_prices.copy()
    prices = prices.drop(columns=["tailend_label"], errors="ignore")
    prices["t"] = pd.to_datetime(prices["t"], unit="s", utc=True, errors="coerce")
    merged = prices.merge(
        resolved[["id", "tailend_label", "t_resolve"]],
        left_on="market_id",
        right_on="id",
        how="inner",
    )
    merged = merged.rename(columns={"t_resolve": "resolve_time"})
    merged["resolve_time"] = pd.to_datetime(merged["resolve_time"], utc=True, errors="coerce")
    merged["outcome"] = (merged["tailend_label"] == "winner").astype(int)

    grouped_outcomes = merged.groupby("market_id")["outcome"].max()
    n_true = int(grouped_outcomes.sum())
    n_false = int(len(grouped_outcomes) - n_true)
    print('NTRUE, NFALSE', n_true, n_false)
    metrics = evaluate_markets_bucketed(merged, days_before=days_before)
    legend_extra = f"winner={n_true}, loser={n_false}\n days before resolution = {days_before}"
    high_prob_count = int(metrics.loc[metrics["p_min"] >= 0.8, "n"].sum())
    title_with_bucket = f"{title}\nBuckets 0.8-1.0 count={high_prob_count}"
    fig, ax = plt.subplots()
    plot_calibration_line(ax, metrics, "mean_pred", "freq_pos", count_col="n", legend_extra=legend_extra)
    graphic_calibration(ax, title=title_with_bucket)
    print(metrics)

def accuracy_relative_brier():
    markets = read_markets_csv(f'{PROJECT_ROOT}/data/test_pipeline.csv')
    prices = pd.read_csv(f'{PROJECT_ROOT}/data/test.csv')
    prices['outcome'] = 1
   
    markets_lossers = read_markets_csv(f'{PROJECT_ROOT}/data/losser_binary_markets.csv')
    prices_lossers = pd.read_csv(f'{PROJECT_ROOT}/data/losser_binary_markets_prices.csv')
    prices_lossers['outcome'] = 0
    #prices = prices[prices['market_id'] == 530873]
    #markets = markets[markets['id'] == 530684]
    #prices_lossers = prices_lossers[prices_lossers['market_id'] == 513494]
    #markets_lossers = markets_lossers[markets_lossers['id'] == 528366]
    
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
    tailended = find_tailend_markets(all_markets, all_prices)
    tailended.to_csv('d.csv')
    print(relative_brier(all_prices, tailended, 14)['relative_brier'].mean())


def accuracy_low_apy(title: str = "Low APY markets calibration plot"):
    markets_b = read_markets_csv(f'{PROJECT_ROOT}/data/binary_markets.csv')
    prices_b = read_prices_csv(f'{PROJECT_ROOT}/data/prices_binary_all.csv')
    markets_c = read_markets_csv(f'{PROJECT_ROOT}/data/categorical_markets_all.csv')
    prices_c = read_prices_csv((f'{PROJECT_ROOT}/data/prices_categorical_all.csv')) 
    markets = pd.concat([markets_b, markets_c], ignore_index=True)   
    prices = pd.concat([prices_b, prices_c], ignore_index=True)   
    prices= prices[prices['token'] == 'yes']
    low_api = 0.2
    finished_markets = markets[markets['closed'] == True]
    low_apy_markets = filter_by_avg_apy(
        finished_markets,
        prices,
        min_days_before_res=5.0,
        max_apy=low_api
    )
    low_apy_markets = filter_by_duration(
        low_apy_markets,
        min_days=MIN_DURATION
    )
    low_apy_markets.to_csv('text.csv')
    print(f"Low APY markets count: {len(low_apy_markets)}")
    low_apy_prices = prices[ prices['market_id'].isin( pd.to_numeric(low_apy_markets['id'], errors="coerce").astype("Int64").dropna().unique() ) ]
    accuracy_all_markets(
        low_apy_markets,
        low_apy_prices,
        title=f"Calibration plot for low APY {len(low_apy_markets)} markets, min duration {MIN_DURATION}d, days before res {DAYS_BEFORE}, low APY {low_api*100:.0f}%",
        days_before=3
    )
    #metrics_low_apy = evaluate_markets_bucketed(
    #    low_apy_prices,
    #    outcome_col='outcome'
    #)

    

if __name__ == "__main__":
   
    mode = 1
    markets_b = read_markets_csv(PROJECT_ROOT / "data/binary_markets.csv")
    markets_c = read_markets_csv(PROJECT_ROOT / "data/categorical_markets_all.csv")
    prices_b = read_prices_csv(PROJECT_ROOT / "data/prices_binary_all.csv")
    prices_c = read_prices_csv(PROJECT_ROOT / "data/prices_categorical_all.csv")
    MIN_DURATION = 30
    markets = pd.concat([markets_b, markets_c], ignore_index=True)
    prices = pd.concat([prices_b, prices_c], ignore_index=True)
    markets = markets[markets['closed'] == True]
    if mode == 1:
        accuracy_low_apy()
    if mode == 2:#tailend
        #get tailend prices if yes always around 90 and a winner then this is a correct prediction
        markets = filter_by_duration(markets, MIN_DURATION)
        markets = find_tailend_markets_by_merged_prices(markets, prices)
        prices = find_tailend_prices(markets, prices)
        markets.to_csv('accuracy.csv', index=False)
        prices.to_csv('accuracy_prices.csv')
        #print(f"Tail-end markets count: {markets['tailend_label'].notna().sum()}")
        print(f"Losers: {markets[markets['tailend_label'] == 'loser'].shape[0]}")
        accuracy_tailend_markets(
            markets, prices,
            title = f"Calibration plot (Reliability curve) for {len(markets)} tail-end markets. Looking on YES/NO tokens",
        ) 
    if mode == 3:
        markets = filter_by_duration(markets, 30)
        markets = find_tailend_markets_by_merged_prices(markets, prices)
        markets = markets[markets['tailend_token'] == 'yes']
        prices = prices[prices['token'] == 'yes']   #non tailend
        markets.to_csv('accuracy_no.csv', index=False)
        prices = prices.merge(
            markets[["id", "tailend_label"]],
            left_on="market_id",
            right_on="id",
            how="inner",
        )
        prices.to_csv('accuracy_prices_no.csv', index=False)
        accuracy_all_markets(
            markets,
            prices,
            title = f"Calibration plot (Reliability curve) for {len(markets)} tail-end markets. Looking on YES token",
        )
    #accuracy_relative_brier()
    
    sys.exit(0)

    # Evaluate losers subset
    loser_ids = pd.to_numeric(markets_lossers.get("id", pd.Series(dtype="Int64")), errors="coerce").astype("Int64")
    prices_losers = all_prices[ pd.to_numeric(all_prices["market_id"], errors="coerce").astype("Int64").isin(pd.unique(loser_ids.dropna())) ]
    metrics_losers, eval_losers = evaluate_markets(prices_losers, all_prices, market_col_prices="market_id", id_col_res="id", outcome_col="outcome_yes")
    # Evaluate non-losers subset
    nonloser_ids = set(pd.to_numeric(all_markets["id"], errors="coerce").astype("Int64")) - set(pd.unique(loser_ids.dropna()))
    prices_nonlosers = all_prices[ pd.to_numeric(all_prices["market_id"], errors="coerce").astype("Int64").isin(nonloser_ids) ]
    metrics_nonlosers, eval_nonlosers = evaluate_markets(prices_nonlosers, all_prices, market_col_prices="market_id", id_col_res="id", outcome_col="outcome_yes")
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
