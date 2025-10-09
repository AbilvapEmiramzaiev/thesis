"""Utilities for collecting Gamma market prices and plotting from cached CSVs."""

from imports import *

from fetch.tail_end_func import (
    YES_INDEX,
    fetch_market_prices_history,
    fetch_markets,
)


def collect_market_prices(
    market_ids: Optional[List[str]] = None,
    *,
    limit: int = 100,
    offset: int = 0,
    token_index: int = YES_INDEX,
    fidelity: int = 1440,
) -> pd.DataFrame:
    """Fetch price history for markets and return a long DataFrame."""
    if market_ids is None or len(market_ids) == 0:
        markets = fetch_markets(size=limit, offset=offset)
        if markets.empty:
            return pd.DataFrame()
        market_ids = [str(market_id) for market_id in markets["id"].tolist()]

    records: List[pd.DataFrame] = []
    for market_id in market_ids:
        try:
            prices = fetch_market_prices_history(str(market_id), token_index, fidelity=fidelity)
        except Exception as exc:
            print(f"Failed to fetch prices for {market_id}: {exc}", file=sys.stderr)
            continue
        if prices.empty:
            continue
        df = prices[["t", "p"]].copy()
        df["market_id"] = str(market_id)
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
    parser.add_argument("--output", type=Path, default=Path("../" + CSV_OUTPUT_PATH), help="CSV path for cached prices")
    parser.add_argument("--markets", nargs="*", help="Explicit market ids (comma separated allowed)")
    parser.add_argument("--limit", type=int, default=100, help="Number of markets to fetch when ids not supplied")
    parser.add_argument("--offset", type=int, default=0, help="Initial offset for pagination")
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
    )

    if prices.empty:
        print("No price data collected.")
        return 0

    save_prices_to_csv(prices, args.output)
    print(f"Wrote {len(prices)} rows to {args.output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
