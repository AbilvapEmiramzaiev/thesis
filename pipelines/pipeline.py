"""Utilities for collecting Gamma market prices and plotting from cached CSVs."""
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from imports import *

from fetch.tail_end_func import (
    YES_INDEX,
    fetch_market_prices_history,
    fetch_markets,
)


def collect_market_prices(
    market_ids: Optional[Iterable[str] | Mapping[str, Any]] = None,
    *,
    limit: int = 100,
    offset: int = GAMMA_API_OLD_MARKETS_OFFSET,
    token_index: int = YES_INDEX,
    fidelity: int = 1440,
    output_path: Optional[Path] = None,
) -> pd.DataFrame:
    """Fetch price history for markets and return a long DataFrame.

    Args:
        market_ids: Iterable of ids or dict mapping id -> market payload to bypass discovery.
    """
    if market_ids is None:
        markets = fetch_markets(size=limit, offset=offset)
        if markets.empty:
            return pd.DataFrame()
        market_ids = {
            str(idx): record for idx, record in markets.set_index("id").to_dict(orient="index").items()
        }
    output_path = Path(output_path) if output_path else None
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
    header_written = bool(output_path and output_path.exists() and output_path.stat().st_size > 0)

    records: List[pd.DataFrame] = []
    for market_id in market_ids:
        try:
            index = identify_market_outcome_winner_index(market_ids[market_id])
            prices = fetch_market_prices_history(str(market_id), index, fidelity=fidelity)
        except Exception as exc:
            print(f"Failed to fetch prices for {market_id}: {exc}", file=sys.stderr)
            continue
        if prices.empty:
            continue
        df = prices[["t", "p"]].copy()
        df["market_id"] = str(market_id)
        df = df[["market_id", "t", "p"]]
        if output_path:
            df.to_csv(output_path, mode="a", header=not header_written, index=False)
            print(f"Appended {len(df)} rows for market {market_id} to {output_path}")
            header_written = True

        records.append(df)

    if not records:
        return pd.DataFrame(columns=["market_id", "t", "p"])

    return pd.concat(records, ignore_index=True)[["market_id", "t", "p"]]


def save_prices_to_csv(prices: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    prices.to_csv(output_path, index=False)


def _parse_market_ids(raw: Optional[List[str]]) -> Optional[List[str]]:
    if not raw:
        return None
    ids: List[str] = []
    for item in raw:
        ids.extend(part.strip() for part in item.split(",") if part.strip())
    return ids or None


def parse_args(argv: Iterable[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect Gamma market prices and cache them to CSV.")
    parser.add_argument("--output", type=Path, default=Path(CSV_OUTPUT_PATH), help="CSV path for cached prices")
    parser.add_argument("--markets", nargs="*", help="Explicit market ids (comma separated allowed)")
    parser.add_argument("--limit", type=int, default=100, help="Number of markets to fetch when ids not supplied")
    parser.add_argument("--offset", type=int, default=GAMMA_API_OLD_MARKETS_OFFSET, help="Initial offset for pagination")
    parser.add_argument("--fidelity", type=int, default=1440, help="Fidelity passed to prices-history endpoint")
    parser.add_argument("--token-index", type=int, default=YES_INDEX, help="Outcome index to download prices for")
    parser.add_argument("--plot-market", help="Optional market id to plot after collection")
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv if argv is not None else sys.argv[1:])
    market_ids = _parse_market_ids(args.markets)

    prices = collect_market_prices(
        market_ids,
        limit=args.limit,
        offset=args.offset,
        token_index=args.token_index,
        fidelity=args.fidelity,
        output_path=args.output,
    )

    if prices.empty:
        print("No price data collected.")
        return 0

    print(f"Appended {len(prices)} rows to {args.output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
