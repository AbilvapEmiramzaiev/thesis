from datetime import datetime
import pandas as pd
from pathlib import Path
import json
from config import *
from fetch.tail_end_func import YES_INDEX, NO_INDEX, find_tailend_markets_by_merged_prices,find_tailend_prices
from typing import Optional, List

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
    

_FLOAT32_MARKET_COLS = {
    "liquidityNum",
    "volumeNum",
    "prob_yes",
    "prob_no",
    "lastTradePrice",
    "bestBid",
    "bestAsk",
}


def read_markets_csv(path: Path) -> pd.DataFrame:
    """Load markets CSV quickly while keeping timestamp columns in UTC."""
    df = pd.read_csv(
        path,
        low_memory=False,
        memory_map=True,
    )

    for col in TIME_COLS:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], utc=True, errors="coerce")

    for col in _FLOAT32_MARKET_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("float32")

    return df

def read_prices_csv(path: Path) -> pd.DataFrame:
    prices = pd.read_csv(
        path,
        low_memory=False,                          
        dtype={"market_id": "string", "token": "string"} 
    )
    
    prices["market_id"] = pd.to_numeric(prices["market_id"], errors="coerce").astype("Int64")
    prices["p"] = pd.to_numeric(prices["p"], errors="coerce")
    prices = prices.dropna(subset=["market_id", "p", "token"])
    prices["token"] = prices["token"].str.lower().str.strip() 
    return prices

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


def df_time_to_datetime(df, column):
    df[column] = pd.to_datetime(df[column], unit='s', utc=True,  errors="coerce")
    return df

def get_ready_tailend_data(marketsPath, pricesPath):
    #boilerplate reader and parser
    markets = read_markets_csv(marketsPath)
    prices = read_prices_csv(pricesPath) 
    tailend = find_tailend_markets_by_merged_prices(markets, prices)
    tailend_prices = find_tailend_prices(tailend, prices)
    return tailend, tailend_prices
    

def write_short_duration_blacklist(
    binary_path: Path | str = PROJECT_ROOT / "data/binary_markets.csv",
    categorical_path: Path | str = PROJECT_ROOT / "data/categorical_markets_all.csv",
    output_path: Path | str = PROJECT_ROOT / "data/markets_blacklist.csv",
    max_hours: float = 24.0,
    *,
    price_paths: List[Path | str] | None,
    include_constant_price: bool = True,
    constant_price_value: float = 0.5,
) -> Path:
    """
    Build a blacklist of market IDs that should be ignored in downstream analysis.
        Price value to test for when `include_constant_price` is enabled.
    """
    market_sources = [Path(binary_path), Path(categorical_path)]
    cutoff_seconds = max_hours * 3600.0
    blacklist_ids: set[int] = set()

    for path in market_sources:
        if not path.exists():
            raise FileNotFoundError(f"Market CSV not found: {path}")

        df = read_markets_csv(path)
        start = pd.to_datetime(df["startDate"], utc=True, errors="coerce")
        end = pd.to_datetime(df["endDate"], utc=True, errors="coerce")
        duration = (end - start).dt.total_seconds()
        mask = duration < cutoff_seconds
        ids = pd.to_numeric(df.loc[mask, "id"], errors="coerce").dropna()
        if not ids.empty:
            blacklist_ids.update(int(v) for v in ids.tolist())

    if include_constant_price:
        price_sources = price_paths
        if not price_sources:
            price_sources = [
                PROJECT_ROOT / "data/prices_binary_all.csv",
                PROJECT_ROOT / "data/prices_categorical_all.csv",
            ]

        for price_path in price_sources:
            price_path = Path(price_path)
            if not price_path.exists():
                continue

            prices = read_prices_csv(price_path)
            if prices.empty:
                continue
            agg = (
                prices.groupby("market_id")["p"]
                .agg(["min", "max"])
                .reset_index()
            )
            matches = agg[
                (agg["min"] == constant_price_value)
                & (agg["max"] == constant_price_value)
            ]["market_id"]
            blacklist_ids.update(int(v) for v in matches.dropna().tolist())

    if not blacklist_ids:
        raise ValueError("No markets matched the blacklist criteria.")

    combined = (
        pd.Series(sorted(blacklist_ids), dtype="Int64", name="id")
    )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_frame().to_csv(output_path, index=False)
    return output_path
