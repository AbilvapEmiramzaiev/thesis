# pip install requests pandas numpy python-dateutil matplotlib pyarrow
from __future__ import annotations
import requests, pandas as pd, numpy as np
from dateutil import parser as dtp
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import time, json
import matplotlib.dates as mdates
from datetime import *
from config import *
from utils import *
from typing import Iterable, Dict, Tuple, List, Optional, Any


def _derive_event_keys(market: Any) -> Tuple[List[str], Optional[str]]:
    """Return list of event identifiers (slugs or titles) and the primary key."""
    getter = market.get if hasattr(market, "get") else lambda k, default=None: market[k] if k in market else default

    event_keys: List[str] = []
    raw_events = getter("events", None)
    if isinstance(raw_events, list):
        for event in raw_events:
            if isinstance(event, dict):
                for candidate in (event.get("slug"), event.get("id")):
                    if candidate:
                        event_keys.append(str(candidate))
                        break

    group_title = getter("groupItemTitle", None)
    if group_title:
        event_keys.append(str(group_title))

    # Remove duplicates while preserving order
    seen = set()
    normalized: List[str] = []
    for key in event_keys:
        if key not in seen:
            seen.add(key)
            normalized.append(key)

    primary_key = normalized[0] if normalized else None
    return normalized, primary_key


def fetch_markets(size: int = 0, offset: int = GAMMA_API_DEAD_MARKETS_OFFSET) -> pd.DataFrame:
    """Fetch multiple markets from Polymarket Gamma API, returns DataFrame.
    If size=0, fetch all markets. If size>0, fetch only the requested size."""
    url = f"{GAMMA_API}/markets"
    all_markets = []
    offset = offset
    page_size = 500
    while True:
        r = requests.get(url, params={"limit": page_size, "offset": offset})
        r.raise_for_status()
        data = r.json()
        if not data:
            break
        for market in data:
            if "clobTokenIds" in market:
                market["clobTokenIds"] = json.loads(market["clobTokenIds"])
            else:
                market["clobTokenIds"] = None

            event_keys, primary_key = _derive_event_keys(market)
            market["event_keys"] = event_keys
            market["primary_event_key"] = primary_key
            all_markets.append(market)
        if len(data) < page_size:
            break
        offset += page_size
        if size > 0 and len(all_markets) >= size:
            all_markets = all_markets[:size]
            break
    return pd.DataFrame(all_markets)


def fetch_market(market_id: str) -> pd.Series:
    r = requests.get(
                f"{GAMMA_API}/markets/{market_id}",
                params={"market": market_id, "limit": 1},
            )  
    r.raise_for_status()
    j = r.json()
    j["clobTokenIds"] = json.loads(j["clobTokenIds"])
    event_keys, primary_key = _derive_event_keys(j)
    j["event_keys"] = event_keys
    j["primary_event_key"] = primary_key
    return pd.Series(j)


def compute_event_market_counts(markets: pd.DataFrame) -> pd.Series:
    """Return number of distinct markets attached to each Gamma event key."""
    if "event_keys" not in markets.columns:
        raise ValueError("markets DataFrame must include an 'event_keys' column")

    exploded = (
        markets
        .explode("event_keys")
        .dropna(subset=["event_keys"])
    )
    if exploded.empty:
        return pd.Series(dtype=int)

    return exploded.groupby("event_keys")["id"].nunique()


def add_event_uniqueness_flags(markets: pd.DataFrame) -> pd.DataFrame:
    """Annotate markets with event-level counts and a single-market flag."""
    counts = compute_event_market_counts(markets)
    counts_dict = counts.to_dict()

    def _max_count(keys: Any) -> Optional[int]:
        if isinstance(keys, list) and keys:
            return max(counts_dict.get(key, 0) for key in keys)
        return None

    annotated = markets.copy()
    annotated["event_market_count"] = annotated["event_keys"].apply(_max_count)
    annotated["is_single_event_market"] = annotated["event_market_count"].apply(
        lambda cnt: bool(cnt == 1) if cnt is not None else False
    )
    return annotated


def is_single_market_event(
    market: pd.Series,
    markets: Optional[pd.DataFrame] = None
) -> bool:
    """True when no other Gamma market shares the same event key(s)."""
    event_keys = market.get("event_keys")
    if not event_keys:
        return False

    if markets is None:
        markets = fetch_markets()

    counts = compute_event_market_counts(markets)
    if counts.empty:
        return False

    return max(counts.get(key, 0) for key in event_keys) == 1

def fetch_trades(market_id: str, cicle: bool = False, end: int = -1) -> pd.DataFrame:
    """Public, no-auth trade history across both outcomes for a market."""
    url = f"{DATA_API}/trades"
    all_rows = []
    offset = 0
    page_size = 500
    while True:
        r = requests.get(
            url,
            params={"market": market_id, "limit": page_size, "offset": offset},
        )
        print(f"Fetching trades offset {offset}...")
        if r.status_code == 429:
            print("Rate limited, sleeping...")
            time.sleep(2)
            continue
        r.raise_for_status()
        chunk = r.json()
        print(len(chunk))

        if not chunk:
            break
        all_rows.extend(chunk)
        if not cicle or len(chunk) < page_size:

            break
        offset += page_size
        if end > 0 and offset >= end:
            break
    if not all_rows:
        return pd.DataFrame()

    df = pd.json_normalize(all_rows)
    df["ts"] = pd.to_datetime(df["timestamp"], unit='s', utc=True, errors="coerce")
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df = df.sort_values("ts").reset_index(drop=True)
    return df[["ts", "price", "outcome", "side", "size"]]

def fetch_market_prices_history(market_id: str, token_index: int, fidelity: int = 1440, startTs: int = False) -> pd.DataFrame:
    """Fetches historical market prices from Polymarket Data API. 1440 = daily"""
    market = fetch_market(market_id)
    token_id = market["clobTokenIds"][token_index]
    startTs = startTs if startTs else gamma_ts_to_utc(market['createdAt'])
    if market is None or market.empty or ("error" in market.index):
        raise RuntimeError(f"Market {market_id} not found or error: {market.get('error','unknown')}")
    url = f"{CLOB_API}/prices-history"
    all_rows = []
    r = requests.get(
        url,
        params={"market": token_id, 
                "startTs": startTs,
                "fidelity": fidelity,},
    )
    r.raise_for_status()
    all_rows = r.json()['history']
    print('Fetched', len(all_rows), 'price points for market', market_id, ' ', startTs)
    if not all_rows:
        return pd.DataFrame()
    df = pd.json_normalize(all_rows)
    df = df.sort_values("t").reset_index(drop=True)
    return df


def _coerce_timestamp(value: Any) -> Optional[pd.Timestamp]:
    if value is None:
        return None
    if isinstance(value, pd.Timestamp):
        return value.tz_convert("UTC") if value.tzinfo else value.tz_localize("UTC")
    if isinstance(value, (int, float, np.integer, np.floating)):
        return pd.to_datetime(value, unit='s', utc=True, errors="coerce")
    return pd.to_datetime(value, utc=True, errors="coerce")


def _prepare_price_series(
    prices: pd.DataFrame,
    price_col: str = "p",
    ts_col: str = "t"
) -> pd.DataFrame:
    """Return prices with UTC timestamps sorted ascending."""
    if ts_col not in prices.columns or price_col not in prices.columns:
        raise ValueError(f"prices must contain '{ts_col}' and '{price_col}' columns")

    df = prices[[ts_col, price_col]].dropna().copy()
    if df.empty:
        return df

    ts = df[ts_col]
    if np.issubdtype(ts.dtype, np.number):
        df["_ts"] = pd.to_datetime(ts, unit='s', utc=True, errors="coerce")
    else:
        df["_ts"] = pd.to_datetime(ts, utc=True, errors="coerce")

    df = df.dropna(subset=["_ts"]).sort_values("_ts").reset_index(drop=True)
    return df


def _build_time_segments(
    prices: pd.DataFrame,
    price_col: str = "p",
    ts_col: str = "t",
    window_start: Optional[Any] = None,
    window_end: Optional[Any] = None
) -> pd.DataFrame:
    """Piecewise-constant segments with durations (seconds) for each price spell."""
    prepared = _prepare_price_series(prices, price_col=price_col, ts_col=ts_col)
    if prepared.empty:
        return pd.DataFrame(columns=["start", "end", "duration_s", price_col])

    start_ts = _coerce_timestamp(window_start) or prepared["_ts"].iloc[0]
    end_ts = _coerce_timestamp(window_end) or prepared["_ts"].iloc[-1]

    if start_ts >= end_ts:
        return pd.DataFrame(columns=["start", "end", "duration_s", price_col])

    prior = prepared[prepared["_ts"] <= start_ts]
    if prior.empty:
        current_price = float(prepared.loc[0, price_col])
    else:
        current_price = float(prior.iloc[-1][price_col])

    time_points = [start_ts]
    segment_prices: List[float] = []

    for _, row in prepared.iterrows():
        ts = row["_ts"]
        price = float(row[price_col])
        if ts <= start_ts:
            current_price = price
            continue
        if ts >= end_ts:
            break
        time_points.append(ts)
        segment_prices.append(current_price)
        current_price = price

    time_points.append(end_ts)
    segment_prices.append(current_price)

    segments = pd.DataFrame({
        "start": time_points[:-1],
        "end": time_points[1:],
        "duration_s": [
            (t_end - t_start).total_seconds()
            for t_start, t_end in zip(time_points[:-1], time_points[1:])
        ],
        price_col: segment_prices,
    })

    segments = segments[segments["duration_s"] > 0].reset_index(drop=True)
    return segments


def is_tailend_market(market_prices: pd.DataFrame, low=0.10, high=0.90) -> bool:
    """A market is tail-end if either outcome is ever below `low` or above `high`."""
    return (market_prices['price'].le(low).any() or market_prices['price'].ge(high).any())


def _extract_market_timestamp(market: Any, *keys: str) -> Optional[pd.Timestamp]:
    getter = market.get if hasattr(market, "get") else lambda k, default=None: market[k] if k in market else default
    for key in keys:
        candidate = getter(key, None)
        ts = _coerce_timestamp(candidate)
        if ts is not None and not pd.isna(ts):
            return ts
    return None


def time_in_band(
    prices: pd.DataFrame,
    lower: float,
    upper: float,
    price_col: str = "p",
    ts_col: str = "t",
    window_start: Optional[Any] = None,
    window_end: Optional[Any] = None
) -> float:
    segments = _build_time_segments(prices, price_col=price_col, ts_col=ts_col, window_start=window_start, window_end=window_end)
    if segments.empty:
        return 0.0
    mask = (segments[price_col] >= lower) & (segments[price_col] <= upper)
    return float(segments.loc[mask, "duration_s"].sum())


def share_time_near_level(
    prices: pd.DataFrame,
    level: float = 0.90,
    tolerance: float = 0.02,
    price_col: str = "p",
    ts_col: str = "t",
    window_start: Optional[Any] = None,
    window_end: Optional[Any] = None
) -> float:
    lower = max(0.0, level - tolerance)
    upper = min(1.0, level + tolerance)
    segments = _build_time_segments(prices, price_col=price_col, ts_col=ts_col, window_start=window_start, window_end=window_end)
    total = segments["duration_s"].sum()
    if total <= 0:
        return 0.0
    in_band = (segments[price_col] >= lower) & (segments[price_col] <= upper)
    return float(segments.loc[in_band, "duration_s"].sum() / total)


def is_tailend_by_time_share(
    prices: pd.DataFrame,
    level: float = 0.90,
    tolerance: float = 0.02,
    min_share: float = 0.5,
    price_col: str = "p",
    ts_col: str = "t",
    window_start: Optional[Any] = None,
    window_end: Optional[Any] = None
) -> bool:
    share = share_time_near_level(
        prices,
        level=level,
        tolerance=tolerance,
        price_col=price_col,
        ts_col=ts_col,
        window_start=window_start,
        window_end=window_end,
    )
    return share >= min_share


def time_above_threshold(
    prices: pd.DataFrame,
    threshold: float = 0.90,
    price_col: str = "p",
    ts_col: str = "t",
    window_start: Optional[Any] = None,
    window_end: Optional[Any] = None
) -> float:
    segments = _build_time_segments(prices, price_col=price_col, ts_col=ts_col, window_start=window_start, window_end=window_end)
    if segments.empty:
        return 0.0
    high = segments[segments[price_col] >= threshold]
    return float(high["duration_s"].sum())


def share_above_threshold(
    prices: pd.DataFrame,
    threshold: float = 0.90,
    price_col: str = "p",
    ts_col: str = "t",
    window_start: Optional[Any] = None,
    window_end: Optional[Any] = None
) -> float:
    segments = _build_time_segments(prices, price_col=price_col, ts_col=ts_col, window_start=window_start, window_end=window_end)
    total = segments["duration_s"].sum()
    if total <= 0:
        return 0.0
    high_seconds = segments.loc[segments[price_col] >= threshold, "duration_s"].sum()
    return float(high_seconds / total)


def first_time_above_threshold(
    prices: pd.DataFrame,
    threshold: float = 0.90,
    price_col: str = "p",
    ts_col: str = "t",
    window_start: Optional[Any] = None,
    window_end: Optional[Any] = None
) -> Optional[pd.Timestamp]:
    segments = _build_time_segments(prices, price_col=price_col, ts_col=ts_col, window_start=window_start, window_end=window_end)
    high_segments = segments[segments[price_col] >= threshold]
    if high_segments.empty:
        return None
    return high_segments.iloc[0]["start"]


def is_tailend_far_from_resolution(
    market: Any,
    prices: pd.DataFrame,
    threshold: float = 0.90,
    lead_time_days: int = 90,
    min_duration_days: int = 30,
    tolerance: float = 0.02,
    price_col: str = "p",
    ts_col: str = "t"
) -> bool:
    market_end = _extract_market_timestamp(
        market,
        "endDate",
        "endDateIso",
        "closeTime",
        "closedTime",
        "resolveTime",
        "resolutionTime",
    )
    if market_end is None:
        return False

    lead_delta = pd.Timedelta(days=lead_time_days)
    pivot = market_end - lead_delta

    first_high = first_time_above_threshold(
        prices,
        threshold=threshold,
        price_col=price_col,
        ts_col=ts_col,
        window_end=market_end,
    )
    if first_high is None or (market_end - first_high) < lead_delta:
        return False

    high_seconds_before_pivot = time_above_threshold(
        prices,
        threshold=threshold,
        price_col=price_col,
        ts_col=ts_col,
        window_end=pivot,
    )
    if high_seconds_before_pivot <= 0:
        return False

    required_seconds = pd.Timedelta(days=min_duration_days).total_seconds()
    if high_seconds_before_pivot < required_seconds:
        return False

    share_near_level = share_time_near_level(
        prices,
        level=threshold,
        tolerance=tolerance,
        price_col=price_col,
        ts_col=ts_col,
        window_end=pivot,
    )
    return share_near_level > 0

def identify_market_bucket(
    prices: pd.Series,
    high_marks: Iterable[float] = (0.90, 0.92, 0.95, 0.97, 0.99)
) -> MarketBucket:
    s = prices.dropna()
    if s.empty:
        return MarketBucket.EMPTY

    highs = sorted(set(high_marks))                 # e.g. [0.90, 0.92, 0.95, 0.97, 0.99]
    lows  = sorted({round(1 - h, 2) for h in highs})  # e.g. [0.01, 0.03, 0.05, 0.08, 0.10]

    # check high-side: pick the tightest/highest satisfied
    for h in sorted(highs, reverse=True):
        if s.min() >= h:
            return HIGH_TO_ENUM[h]

    # check low-side: pick the tightest/smallest satisfied
    for l in lows:  # ascending; first satisfied is tightest (e.g., <=1%)
        if s.max() <= l:
            # l is like 0.01/0.03/... ensure it exists in mapping (round to 2 dp)
            l_key = round(l, 2)
            return LOW_TO_ENUM[l_key]

    return MarketBucket.NOT_TAILEND


def add_market_bucket(
    market: pd.Series,
    prices: pd.DataFrame,
    col_name: str = "p"
) -> pd.Series:
    """Add a constant 'market_bucket' column for the entire market."""
    label = identify_market_bucket(prices[col_name])
    df = market.copy()
    df["market_bucket"] = label
    return df

def plot_market(trades: pd.DataFrame):
    yes_line = trades[trades["outcome"] == "Yes"]
    no_line  = trades[trades["outcome"] == "No"]
    plt.figure(figsize=(11,5))
    plt.plot(yes_line.index, yes_line["price"], label="YES", color="blue")
    plt.plot(no_line.index, no_line["price"],  label="NO",  color="red")
    plt.ylabel("probability")
    plt.xlabel("Index")
    plt.ylim(0,1)
    plt.grid(True, axis="y", linestyle="--", alpha=0.6)
    plt.title(fetch_market(TEST_MARKET_ID)['title'])
    plt.yticks(np.arange(0, 1.05, 0.05)) 
    plt.gca().yaxis.set_major_formatter(PercentFormatter(xmax=1))
    plt.legend()
    plt.tight_layout()
    plt.show()   

def fetch_all_market_prices(market_id: str) -> pd.DataFrame:
    return fetch_market_prices_history(market_id, YES_INDEX)
    

def plot_market_history(prices: pd.DataFrame):
    x_utc = pd.to_datetime(prices['t'], unit='s', utc=True)

    plt.figure(figsize=(11,5))
    plt.plot(x_utc, prices['p'], label="price", color="blue")
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%M%d', tz=timezone.utc))
    plt.gcf().autofmt_xdate()
    plt.ylabel("probability")
    plt.xlabel("time")
    plt.ylim(0,1)
    plt.grid(True, axis="y", linestyle="--", alpha=0.6)
    plt.title(fetch_market(TEST_MARKET_ID)['question'])
    plt.yticks(np.arange(0, 1.05, 0.05)) 
    plt.gca().yaxis.set_major_formatter(PercentFormatter(xmax=1))
    plt.legend()
    plt.tight_layout()
    plt.show()
    
if __name__ == "__main__":
    market = fetch_market(TEST_TAILEND_MARKET_ID)
    prices = fetch_market_prices_history(market['id'], YES_INDEX)
    #print(prices.head())
    prices['market_id'] = market['id']
    print(prices.head())
    market = add_market_bucket(market, prices); 
    print(market, type(market)) 
    #plot_market_history(prices)
    #trades = fetch_trades(MARKET_ID, cicle=True, end=200000)
    #print(f"Fetched {len(trades)} trades")
    #line   = make_line(trades, freq="1min")      # this is the “blue line”
    #line_t = identify_tailend_market(line, low=0.10, high=0.90)
    #plot_market(trades)
