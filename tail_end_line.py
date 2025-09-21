# pip install requests pandas numpy python-dateutil matplotlib pyarrow
from __future__ import annotations
import requests, pandas as pd, numpy as np
from dateutil import parser as dtp
import matplotlib.pyplot as plt

DATA_API = "https://data-api.polymarket.com"
# Example: conditionId from your payload (replace with yours)
CONDITION_ID = "0xfa3a57f4d41a14e74abf91384bfc63db7b068d526d26bcdcd4bf49fd24bc4d25"

def fetch_trades(condition_id: str, limit: int = 200000) -> pd.DataFrame:
    """Public, no-auth trade history across both outcomes for a market."""
    url = f"{DATA_API}/trades"
    # Pull in chunks if you need more than the API's max page size
    r = requests.get(url, params={"market": condition_id, "limit": limit}, timeout=60)
    r.raise_for_status()
    df = pd.json_normalize(r.json())
    if df.empty:
        return df
    # Normalize fields you typically get from the Data API
    # expected: ['price','timestamp','outcome', ...]
    df["ts"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    # Convert to YES probability even when the trade was on the NO leg
    if "outcome" in df:
        out = df["outcome"].astype(str).str.upper()
        df["p_yes"] = np.where(out.str.contains("NO"), 1.0 - df["price"], df["price"])
    else:
        # If no outcome field, assume prices are YES
        df["p_yes"] = df["price"]
    df = df.sort_values("ts").reset_index(drop=True)
    return df[["ts", "p_yes"]]

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
    s = (trades.set_index("ts")["p_yes"]
                 .resample(freq)
                 .last()
                 .ffill()
                 .clip(0, 1))
    line = s.to_frame("p_yes").reset_index()
    return line

def tail_flags(line: pd.DataFrame, low=0.10, high=0.90) -> pd.DataFrame:
    out = line.copy()
    out["is_tail_low"]  = out["p_yes"] <= low
    out["is_tail_high"] = out["p_yes"] >= high
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

if __name__ == "__main__":
    trades = fetch_trades(CONDITION_ID, limit=200000)
    print(trades.head())
    line   = make_line(trades, freq="1min")      # this is the “blue line”
    line_t = tail_flags(line, low=0.10, high=0.90)
    print(tail_summary(line_t))

    # plot (similar look: step-ish with forward-filled flats)
    plt.figure(figsize=(9,3))
    plt.plot(line["ts"], line["p_yes"])
    plt.ylabel("YES probability")
    plt.xlabel("time")
    plt.ylim(0,1)
    plt.tight_layout()
    plt.show()

    # optional export
    line_t.to_parquet("polymarket_line.parquet", index=False)
