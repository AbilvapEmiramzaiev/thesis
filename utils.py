from datetime import datetime


def ts_to_utc(ts: int) -> None:
    return datetime.utcfromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')

def gamma_ts_to_utc(gammaTs: str) -> int:
    return int(datetime.fromisoformat(gammaTs.replace('Z', '+00:00')).timestamp())