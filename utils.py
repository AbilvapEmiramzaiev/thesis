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
    print(f"Market {data['id']} saved to {output_path}")
    data.to_csv(output_path, index=False)

def parse_market(j: dict) -> pd.Series:
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

    j["startDate"] = j.get("startDate") or j["events"][0].get       ("startDate") or j["events"][0].get("createAt")
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