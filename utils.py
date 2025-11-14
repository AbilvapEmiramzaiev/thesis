from datetime import datetime
import pandas as pd
from pathlib import Path
import json
from config import *
def ts_to_utc(ts: int) -> None:
    return datetime.utcfromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')

def gamma_ts_to_utc(gammaTs: str) -> int:
    return int(datetime.fromisoformat(gammaTs.replace('Z', '+00:00')).timestamp())


def save_to_csv(data: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Markets {len(data)} saved to {output_path}")
    data.to_csv(output_path, index=False)

def parse_market(j: dict) -> pd.Series:
    if(j.get('clobTokenIds') is None):
        print(f'Market {j.get("id")} has no clobTokenIds, skipping')
        return pd.Series()
    for key in ["clobTokenIds", "outcomes", "outcomePrices"]:
        j[key] = json.loads(j[key])
     # Add derived numeric columns for clarity
    j["liquidityNum"] = to_float(j.get("liquidityNum") or j.get("liquidity"))
    j["volumeNum"] = to_float(j.get("volumeNum") or j.get("volume"))
    j["fee"] = to_float(j.get("fee")) if j.get("fee") else None
    j['outcomes'][YES_INDEX] = "Yes"
    j['outcomes'][NO_INDEX] = "No"
        # Add parsed probability fields if available
    if isinstance(j.get("outcomePrices"), list) and len(j["outcomePrices"]) >= 2:
        j["prob_yes"] = to_float(j["outcomePrices"][YES_INDEX])
        j["prob_no"] = to_float(j["outcomePrices"][NO_INDEX])
    j['eventId'] = j['events'][0]['id'] if j.get('events') and isinstance(j['events'], list) and len(j['events']) > 0 else None

    j["startDate"] = j.get("startDate") or j["events"][0].get("startDate") or j["events"][0].get("createAt")
    j['clobTokenIdYes'] = j['clobTokenIds'][YES_INDEX]
    j['clobTokenIdNo'] = j['clobTokenIds'][NO_INDEX]
    # Keep a subset of relevant research fields
    # "createdAt", "updatedAt", "closedAt", 'outcomes'
    cols = [
        "id", "question", "eventId", "slug", "conditionId",
        "startDate", "endDate","closedTime", 'category',
        "liquidityNum", "volumeNum", "prob_yes", "prob_no",
        "lastTradePrice", "bestBid", "bestAsk", 
        "active", "closed", "marketType", "fpmmLive",
        "resolutionSource", 'clobTokenIdYes', 'clobTokenIdNo' 
    ]
    data = {c: j.get(c) for c in cols}
    return pd.Series(data)

def to_float(x):
        try:
            return round(float(x), 2)
        except (TypeError, ValueError):
            return None
        
def normalize_time(markets: pd.DataFrame) -> pd.DataFrame:
    #make ISO-8601 YYYY-MM-DDTHH:MM:SSZ format
    
    def to_iso_z(s):
        if pd.isna(s):
            return ""
        # приводим к секундам (без миллисекунд). Убери .floor(...) если нужны мс
        s = s.floor("s")
        return s.strftime("%Y-%m-%dT%H:%M:%SZ")
    
    for c in TIME_COLS:
        if c in markets.columns:
            markets[c] = pd.to_datetime(markets[c], utc=True, errors="coerce")
            markets[c] = markets[c].apply(to_iso_z)
    return markets
    

def read_markets_csv(path: Path) -> pd.DataFrame:
    utc_conv = lambda s: pd.to_datetime(s, utc=True, errors="coerce")
    converters = {c: utc_conv for c in TIME_COLS}
    return pd.read_csv(path, converters=converters)


def compute_market_apy_series(
    df: pd.DataFrame,
    *,
    resolution_time,
    price_col: str = "p",
    ts_col: str = "t",
    min_days_before_res: float = 3.0,
) -> pd.Series:
    """
    Return APY time series (indexed by timestamp) for a single market.
    APY = ((1 - p) / p) * (365 / days_to_resolution)
    Excludes the last `min_days_before_res` days before resolution.
    """
    t = pd.to_datetime(df[ts_col], unit="s", utc=True)
    p = df[price_col].clip(1e-9, 1 - 1e-9)
    res_ts = pd.to_datetime(resolution_time, utc=True)
    dt_days = (res_ts - t).dt.total_seconds() / 86400.0
    m = dt_days > min_days_before_res #how much days before resolution to ignore
    t, p, dt_days = t[m], p[m], dt_days[m]
    apy = ((1.0 - p) / p) * (365.0 / dt_days)
    s = pd.Series(apy.values, index=t[m])
    return s.sort_index()