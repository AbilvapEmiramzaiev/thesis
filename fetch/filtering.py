import json
from typing import Optional
import pandas as pd
from datetime import datetime
from typing import Tuple

def api_identify_market_outcome_winner_index(market: json) -> Optional[str]:
    """Identify the winning outcome of a market using the clobid field."""
    tol = 1e-3
    outcomes = market.get("outcomePrices", [])
    if not outcomes:
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

def filter_by_timeframe(markets: pd.DataFrame, start_ts: pd.Timestamp = None, end_ts: pd.Timestamp = None) -> pd.DataFrame:
    s = pd.to_datetime(markets["startDate"], utc=True, errors="coerce")
    e = pd.to_datetime(markets["endDate"],   utc=True, errors="coerce")
    mask = pd.Series(True, index=markets.index)
    if start_ts:
        mask &= s.between(start_ts - pd.Timedelta(days=5), start_ts + pd.Timedelta(days=5), inclusive="both")
    if end_ts:
        mask &= e.between(end_ts - pd.Timedelta(days=5), end_ts + pd.Timedelta(days=5), inclusive="both")
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

  
def is_prices_above_then(
    prices: pd.DataFrame,
    threshold: float = 0.80,
    required_pct: float = 0.60,
) -> Tuple[bool, float]:
    cond = (prices["p"] >= threshold)
    frac = cond.mean()
    return frac >= required_pct
