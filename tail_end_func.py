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

def fetch_market(market_id: str) -> pd.Series:
    r = requests.get(
                f"{GAMMA_API}/markets/{market_id}",
                params={"market": market_id, "limit": 1},
            )  
    r.raise_for_status()
    j = r.json()
    j["clobTokenIds"] = json.loads(j["clobTokenIds"])
    return j

def fetch_trades(market_id: str, cicle: bool = False, end: int = -1) -> pd.DataFrame:
    """Public, no-auth trade history across both outcomes for a market."""
    url = f"{DATA_API}/trades"
    all_rows = []
    offset = 0
    page_size = 500
    print(page_size)
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

def fetch_market_prices_history(market_id: str, token_index: int, interval: str, fidelity: int) -> pd.DataFrame:
    """Fetches historical market prices from Polymarket Data API."""
    market = fetch_market(market_id)
    token_id = market["clobTokenIds"][token_index]
    if not market or "error" in market:
        raise RuntimeError(f"Market {market_id} not found or error: {market.get('error','unknown')}")
    url = f"{CLOB_API}/prices-history"
    all_rows = []
    r = requests.get(
        url,
        params={"market": token_id, 
                "interval": interval,
                "fidelity": fidelity,},
    )
    r.raise_for_status()
    all_rows = r.json()['history']
    if not all_rows:
        return pd.DataFrame()
    df = pd.json_normalize(all_rows)
    df = df.sort_values("t").reset_index(drop=True)
    return df

def make_line(trades: pd.DataFrame, freq: str = "1min") -> pd.DataFrame:
    """
    Resample to a uniform timeline like the Polymarket chart.
    Strategy:
      - take the last trade per bar (line is step-like),
      - forward-fill gaps (so flat segments show when no trades),
      - clip to [0,1].
    """
    if trades.empty:
        raise RuntimeError("No trades returned for this market.")
    s = (trades.set_index("ts")["price"]
                 .resample(freq)
                 .last()
                 .ffill()
                 .clip(0, 1))
    line = s.to_frame("price").reset_index()
    return line

def identify_tailend_market(line: pd.DataFrame, low=0.10, high=0.90) -> pd.DataFrame:
    out = line.copy()
    out["is_tail_low"]  = out["price"] <= low
    out["is_tail_high"] = out["price"] >= high
    out["is_tail"]      = out["is_tail_low"] | out["is_tail_high"]
    return out

def tail_summary(line_tail: pd.DataFrame) -> dict:
    if line_tail.empty:
        return {"any_tail": False, "share_tail": 0.0, "longest_run_minutes": 0}
    share = float(line_tail["is_tail"].mean())
    # longest consecutive run of tail bars
    runs = (line_tail["is_tail"] != line_tail["is_tail"].shift()).cumsum()
    longest = int(line_tail.loc[line_tail["is_tail"]].groupby(runs).size().max() if line_tail["is_tail"].any() else 0)
    return {"any_tail": bool(line_tail["is_tail"].any()),
            "share_tail": round(share, 4),
            "longest_run_minutes": longest}

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
    market = fetch_market(TEST_MARKET_ID)
    prices = fetch_market_prices_history(TEST_MARKET_ID, YES_INDEX, "max", 30)
    plot_market_history(prices)
    #trades = fetch_trades(MARKET_ID, cicle=True, end=200000)
    #print(f"Fetched {len(trades)} trades")
    #line   = make_line(trades, freq="1min")      # this is the “blue line”
    #line_t = identify_tailend_market(line, low=0.10, high=0.90)
    #plot_market(trades)
