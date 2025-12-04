import json
from typing import Optional, Tuple, List
import pandas as pd
from datetime import datetime
from config import TAILEND_LOSER_PRICE, TAILEND_RATE


def api_identify_market_outcome_winner_index(market: json) -> Optional[str]:
    """Identify the winning outcome of a market using the clobid field."""
    tol = 1e-3
    outcomes = market.get("outcomePrices", [])
    outcomes = json.loads(outcomes) if isinstance(outcomes, str) else outcomes
    if not outcomes:
        print(f"identify winner function: M-{market['id']}  with no clobTokenIds")
        return None

    # Find the outcome with the highest clobid
    for i, p_str in enumerate(outcomes):
            try:
                p = float(p_str)
            except ValueError:
                continue
            # Check if p is “close enough” to 1.0
            if abs(p - 1.0) <= tol:
                return i
            
def get_market_winner_clobTokenId(market: pd.Series) -> Optional[str]:
    winner = closer_to_one(float(market['prob_yes']), float(market['prob_no']))
    if winner == float(market['prob_yes']):
        return market['clobTokenIdYes']
    elif winner == float(market['prob_no']):
        return market['clobTokenIdNo']
    else:
        return None 

def closer_to_one(yes: float, no: float) -> float:
    """Return whichever of x or y is closer to 1."""
    if abs(yes - 1) < abs(no - 1):
        return yes
    elif abs(no - 1) < abs(yes - 1):
        return no
    else:
        return None

def filter_by_timeframe(markets: pd.DataFrame, start_ts: pd.Timestamp = None, end_ts: pd.Timestamp = None, spread: int = 5) -> pd.DataFrame:
    s = pd.to_datetime(markets["startDate"], utc=True, errors="coerce")
    e = pd.to_datetime(markets["endDate"],   utc=True, errors="coerce")
    mask = pd.Series(True, index=markets.index)
    if start_ts:
        mask &= s.between(start_ts - pd.Timedelta(days=spread), start_ts + pd.Timedelta(days=spread), inclusive="both")
    if end_ts:
        mask &= e.between(end_ts - pd.Timedelta(days=spread), end_ts + pd.Timedelta(days=spread), inclusive="both")
    return markets.loc[mask].copy()

    #mask = (s.between(lower, upper, inclusive="both")) & (e.between(lowerE, upperE, inclusive="both"))
    #return markets.loc[mask].copy()

def filter_by_duration(df: pd.DataFrame, min_days: int, max_days: int = None) -> pd.DataFrame:
    markets = df.copy()
    t_resolve =  pd.to_datetime(markets["closedTime"], utc=True, errors="coerce")\
        .fillna(pd.to_datetime(markets["endDate"], utc=True, errors="coerce"))
    
    duration = t_resolve - pd.to_datetime(markets["startDate"])
    threshold = pd.Timedelta(days=min_days)
    if max_days is not None:
        tmax = pd.Timedelta(days=max_days)
        mask = (duration >= threshold) & (duration <= tmax)
    else:
        mask = (duration >= threshold)
    markets["t_resolve"] = t_resolve
    markets["duration_days"] = duration.dt.days
    return markets[mask]

def _market_resolution_time(row: pd.Series) -> Optional[pd.Timestamp]:
    for col in ("t_resolve", "resolve_time", "closedTime", "endDate"):
        if col in row and not pd.isna(row[col]):
            return row[col]
    return None

def filter_by_avg_apy(
    markets: pd.DataFrame,
    prices: pd.DataFrame,
    *,
    min_apy: Optional[float] = None,
    max_apy: Optional[float] = None,
    min_days_before_res: float = 3.0,
) -> pd.DataFrame:
    """
    Keep only markets whose average APY (computed from price history) lies in [min_apy, max_apy].
    Thresholds are expressed as fractions (0.10 = 10%). The underlying APY formula
    returns percent values, so they are scaled down and stored in the `avg_apy` column.
    """
    from utils import compute_market_apy_series  # local import to avoid circular dependency
    required_cols = {"market_id", "p", "t"}
    if not required_cols.issubset(prices.columns):
        return markets.iloc[0:0].copy()

    prices_norm = prices.copy()
    prices_norm["market_id"] = pd.to_numeric(prices_norm["market_id"], errors="coerce").astype("Int64")
    prices_norm["p"] = pd.to_numeric(prices_norm["p"], errors="coerce")
    prices_norm["t"] = pd.to_numeric(prices_norm["t"], errors="coerce")
    prices_norm = prices_norm.dropna(subset=["market_id", "p", "t"])

    grouped = prices_norm.groupby("market_id", sort=False)
    filtered_rows: List[pd.Series] = []

    for _, market in markets.iterrows():
        market_id = pd.to_numeric(market.get("id"), errors="coerce")
        if pd.isna(market_id):
            continue
        market_id = int(market_id)
        if market_id not in grouped.groups:
            continue
        resolution_time = _market_resolution_time(market)
        if resolution_time is None:
            continue

        price_history = grouped.get_group(market_id)
        from utils import compute_market_apy_series  # local import to avoid circular dependency
        apy_series = compute_market_apy_series(
            price_history,
            resolution_time=resolution_time,
            min_days_before_res=min_days_before_res,
        )
        if apy_series.empty:
            continue
        avg_apy_percent = float(apy_series.mean())
        avg_apy = avg_apy_percent / 100.0  # store as fraction so 0.10 == 10%

        if min_apy is not None and avg_apy < min_apy:
            continue
        if max_apy is not None and avg_apy > max_apy:
            continue

        market_copy = market.copy()
        market_copy["avg_apy"] = avg_apy
        filtered_rows.append(market_copy)

    if not filtered_rows:
        return markets.iloc[0:0].copy()
    return pd.DataFrame(filtered_rows)

def filter_losser_tailend_markets(
    markets: pd.DataFrame,
    prices: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    tailend_market_ids = []

    for market_id in prices["market_id"].unique():
        market_prices = prices[prices["market_id"] == market_id]
        if filter_losser_prices(market_prices):
            tailend_market_ids.append(market_id)
    
    filtered_markets = markets[markets["id"].isin(tailend_market_ids)].copy()

    filtered_prices = prices[prices["market_id"].isin(tailend_market_ids)].copy()
    filtered_prices["p"] = 1 - filtered_prices["p"]

    return filtered_markets, filtered_prices

def filter_losser_prices(
        prices: pd.DataFrame,
    ) -> bool:
    return not is_prices_above_then(prices, threshold=TAILEND_LOSER_PRICE, required_pct=TAILEND_RATE)

 
  
def is_prices_above_then(
    prices: pd.DataFrame,
    threshold: float = TAILEND_LOSER_PRICE,
    required_pct: float = TAILEND_RATE,
) -> bool:
    cond = (prices["p"] >= threshold)
    frac = cond.mean()
    return frac >= required_pct
