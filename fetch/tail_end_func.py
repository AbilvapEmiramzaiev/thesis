# pip install requests pandas numpy python-dateutil matplotlib pyarrow
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from imports import *


def fetch_markets(
    size: int = 0,
    page: int = 500,
    offset: int = GAMMA_API_DEAD_MARKETS_OFFSET,
    api_filters: Optional[Dict[str, Any]] = None,
    post_filters: Optional[List[Callable[[Dict[str, Any]], bool]]] = None,
) -> pd.DataFrame|None:
    """Fetch multiple markets from Polymarket Gamma API, returns DataFrame.
    - `api_filters`: dict of query params to add to the API call.
    - `post_filters`: list of functions `market -> bool` to filter the result after fetching."""

    url = f"{GAMMA_API}/markets"
    all_markets = []
    offset = offset
    page_size = page

    api_filters = api_filters or {}
    post_filters = post_filters or []

    while True:
        params: Dict[str, Any] = {"limit": page_size, "offset": offset}
        # merge in API-level filters
        params.update(api_filters)

        r = SESSION.get(url, params=params, timeout=DEFAULT_TIMEOUT)
        r.raise_for_status()
        data = r.json()
        if data is None or len(data) == 0:
            return None

        for market in data:
            if(market.get('clobTokenIds') is None):
                print(f"Skipping market {market.get('id')} with no clobTokenIds")
                continue
            parsed = parse_market(market)
            all_markets.append(parsed)

        if len(data) < page_size:
            break

        offset += page_size
        if size > 0 and len(all_markets) >= size:
            all_markets = all_markets[:size]
            break

    # now apply post-filters
    if post_filters:
        filtered = []
        for m in all_markets:
            keep = True
            for fn in post_filters:
                if isinstance(fn, str):
                    fn = globals().get(fn)
                if not fn(m):
                    keep = False
                    break
            if keep:
                filtered.append(m)
        all_markets = filtered
    print(f"Fetched {len(all_markets)} markets from offset {offset}")
    return pd.DataFrame(all_markets)


def fetch_market(market_id: str) -> pd.Series:
    r = requests.get(
                f"{GAMMA_API}/markets",
                params={"id": market_id, "limit": 1},
            )  
    r.raise_for_status()
    j = r.json()
    return parse_market(j[0])



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

def is_single_market_event(
    market: pd.Series,
) -> bool:
    """True when no other Gamma market shares the same event key(s)."""
    eventId = market['eventId']
    if not eventId:
        return False

    r = requests.get(
                f"{GAMMA_API}/events/{eventId}"
            )  
    r.raise_for_status()
    j = r.json()
    print(f'Marke {market['id']} is { "single-event" if len(j['markets']) == 1 else "multi-event"} market')
    return len(j['markets']) == 1

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

def fetch_market_prices_history(market: pd.Series, token_index: int, fidelity: int = 1440, startTs: int = False) -> pd.DataFrame:
    """Fetches historical market prices from Polymarket Data API. 1440 = daily"""
    token_id = market["clobTokenIds"][token_index]
    startTs = startTs if startTs else gamma_ts_to_utc(market['startDate'])
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

def fetch_all_market_prices(market_id: str) -> pd.DataFrame:
    return fetch_market_prices_history(market_id, YES_INDEX)
    

if __name__ == "__main__":
    markets = pd.read_csv(f'{PROJECT_ROOT}/data/test_pipeline.csv')
    normalized = normalize_time(markets)
    normalized.to_csv(Path('data/normalized.csv'), index=False) 
    #print(markets['question'])
    #markets.apply(lambda row: is_single_market_event(row), axis=1)
    #market = fetch_markets(3)
    #save_to_csv(market.to_frame().T, Path('data/market.csv'))
    #print(market.head(), type(market))
    #print(is_single_market_event(market[0]))
    """print(identify_market_outcome_winner_index(market))
    prices = fetch_market_prices_history(market['id'], YES_INDEX)
    print(prices.head()) """
    #prices['market_id'] = market['id']
    #print(prices.head())
    #market = add_market_bucket(market, prices); 
    #print(market, type(market)) 
    #plot_market_history(prices)
    #trades = fetch_trades(MARKET_ID, cicle=True, end=200000)
    #print(f"Fetched {len(trades)} trades")
    #line   = make_line(trades, freq="1min")      # this is the “blue line”
    #line_t = identify_tailend_market(line, low=0.10, high=0.90)
    #plot_market(trades)
