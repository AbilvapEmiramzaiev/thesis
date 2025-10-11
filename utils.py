from datetime import datetime
import pandas as pd
from pathlib import Path
import json
def ts_to_utc(ts: int) -> None:
    return datetime.utcfromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')

def gamma_ts_to_utc(gammaTs: str) -> int:
    return int(datetime.fromisoformat(gammaTs.replace('Z', '+00:00')).timestamp())


def save_to_csv(data: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(output_path, index=False)

def parse_market(j: dict) -> pd.Series:
    for key in ["clobTokenIds", "outcomes", "outcomePrices"]:
        j[key] = json.loads(j[key])

    def to_float(x):
        try:
            return float(x)
        except (TypeError, ValueError):
            return None

    # Add derived numeric columns for clarity
    j["liquidityNum"] = to_float(j.get("liquidityNum") or j.get("liquidity"))
    j["volumeNum"] = to_float(j.get("volumeNum") or j.get("volume"))
    j["fee"] = to_float(j.get("fee")) if j.get("fee") else None

    # Add parsed probability fields if available
    if isinstance(j.get("outcomePrices"), list) and len(j["outcomePrices"]) >= 2:
        j["prob_yes"] = to_float(j["outcomePrices"][0])
        j["prob_no"] = to_float(j["outcomePrices"][1])
    j['eventId'] = j['events'][0]['id'] if j.get('events') and isinstance(j['events'], list) and len(j['events']) > 0 else None
    # Keep a subset of relevant research fields
    cols = [
        "id", "question", "eventId", "slug", "conditionId", "endDate","startDate", "closedTime",
        "liquidityNum", "volumeNum", "prob_yes", "prob_no",
        "lastTradePrice", "bestBid", "bestAsk", "createdAt", "updatedAt", "closedAt",
        "active", "closed", "marketType", "fpmmLive",
        "resolutionSource", 'clobTokenIds', 'outcomes', 'outcomePrices'
    ]
    data = {c: j.get(c) for c in cols}
    return pd.Series(data)