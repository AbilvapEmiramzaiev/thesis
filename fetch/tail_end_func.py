# pip install requests pandas numpy python-dateutil matplotlib pyarrow
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from imports import *
from plot.plot_data import *
from plot.graphics import *

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
            for fn, result in post_filters.items():
                if isinstance(fn, str):
                    fn = globals().get(fn)
                if fn(m) is not result:
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
    print(f'Market {market['id']} is { "single-event" if len(j['markets']) == 1 else "multi-event"} market')
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

def fetch_market_prices_history(startDate: str, clobTokenId: str, fidelity: int = 1440, startTs: int = False) -> pd.DataFrame:
    """Fetches historical market prices from Polymarket Data API. 1440 = daily"""
    startTs = startTs if startTs else gamma_ts_to_utc(startDate)
    url = f"{CLOB_API}/prices-history"
    all_rows = []
    r = SESSION.get(
        url,
        params={"market": clobTokenId, 
                "startTs": startTs,
                "fidelity": fidelity,},
    )
    r.raise_for_status()
    all_rows = r.json()['history']
    print('Fetched', len(all_rows), 'price points for market', clobTokenId[:3], '...',clobTokenId[-3:], ' ', startTs)
    if not all_rows:
        return pd.DataFrame()
    df = pd.json_normalize(all_rows)
    df = df.sort_values("t").reset_index(drop=True)
    return df




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


def is_strictly_tailend_market(market_prices: pd.DataFrame, low=0.10, high=0.90) -> bool:
    """A market is tail-end if either outcome is ever below `low` or above `high`."""
    return (market_prices['price'].le(low).any() or market_prices['price'].ge(high).any())

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
  
def find_tailend_markets(markets: pd.DataFrame,
                        prices: pd.DataFrame,
                        threshold: float = 0.90,
                        percent: float = 0.60) -> pd.DataFrame:
    tailend_markets = []
    markets['id'] = pd.to_numeric(markets['id'], errors='coerce').astype('Int64')

    for _, market in markets.iterrows():
        market_prices = prices[prices['market_id'] == int(market['id'])]
        if market_prices.empty:
            continue
        if is_prices_above_then(market_prices, threshold, percent):
            tailend_markets.append(market)
    if not tailend_markets:
        return pd.DataFrame(columns=markets.columns)
    return pd.DataFrame(tailend_markets)







if __name__ == "__main__":

    markets = read_markets_csv(f'{PROJECT_ROOT}/data/test_pipeline.csv')
    prices = pd.read_csv(f'{PROJECT_ROOT}/data/market_prices.csv')
    #tailended = filter_by_timeframe(markets, end_ts=pd.Timestamp('2024-12-31T12:59:59Z'))



    tailended = find_tailend_markets(markets, prices, TAILEND_PERCENT, TAILEND_RATE)
    
    prices['market_id'] = prices['market_id'].astype(int)
    tailended['id'] = tailended['id'].astype(int)
    tailenedPrices = prices[prices['market_id'].isin(markets['id'])]
    
    
    print(len(tailended))
    #graphic_apy_per_market(tailended, prices)
    #graphic_min_apy_line(tailended, prices)
    #graphic_apy_aggregated(tailended, tailenedPrices)
    graphic_apy_aggregated_many_years(tailended, tailenedPrices)


    tmp = markets.copy()
    tmp['end_dt'] = pd.to_datetime(tmp['endDate'], utc=True, errors='coerce').dt.normalize()
    tmp = tmp.dropna(subset=['end_dt'])
    tmp['day_of_month'] = tmp['end_dt'].dt.day
    tmp['ym'] = tmp['end_dt'].dt.to_period('M')

    # 1) Overall most common exact end *date* (YYYY-MM-DD)
    overall_exact_counts = tmp['end_dt'].value_counts()
    overall_top_exact_date = overall_exact_counts.index[0]
    overall_top_exact_count = overall_exact_counts.iloc[0]

    # 2) Overall most common *day-of-month* (e.g., 1st, 15th, …)
    dom_counts = tmp['day_of_month'].value_counts().sort_index()
    overall_top_dom = dom_counts.idxmax()
    overall_top_dom_count = dom_counts.max()

    print("Most common exact end date:", overall_top_exact_date.date(), "count:", overall_top_exact_count)
    print("Most common day-of-month:", overall_top_dom, "count:", overall_top_dom_count)

    # 3) For each month, the most popular *day-of-month* (your “each month → 1 date”)
    per_month_top = (
        tmp.groupby(['ym', 'day_of_month'])
        .size()
        .rename('n')
        .reset_index()
        .sort_values(['ym', 'n'], ascending=[True, False])
        .groupby('ym', as_index=False)
        .first()
        .rename(columns={'day_of_month': 'popular_day_of_month', 'n': 'count'})
    )
    print(per_month_top.head(10))

   #longest = filter_by_duration(modermarkets, 60, 70)
    
    #no_pricers = pd.read_csv(f'{PROJECT_ROOT}/data/no_price_markets.csv', names=['id'])
    #markets = markets[~markets['id'].isin(no_pricers['id'])]
    #find_tailend_markets(markets, prices, 0.90, 0.60)
    
    #start = pd.Timestamp('2025-03-01T00:00:00Z')
    #end = pd.Timestamp('2025-04-01T23:59:59Z')
    #df = markets.copy()
    #nostart = markets['startDate'].isna()
    #print(len(df[nostart]), 'markets with no startDate')
    #markets[nostart].to_csv(Path('data/nostart.csv'), index=False)
    #print(filter_by_timeframe(markets, start, end))
    #for min in range(0,100,5):
    #    print(f'Amount of markets with duration {min}-{min+5} days is {len(filter_by_duration(markets, min, min+5))}')
    #normalized = normalize_time(markets)
    #normalized.to_csv(Path('data/test_pipeline.csv'), index=False) 
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
