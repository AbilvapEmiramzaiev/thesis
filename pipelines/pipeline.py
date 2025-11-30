"""Utilities for collecting Gamma market prices and plotting from cached CSVs."""
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from imports import *

from fetch.tail_end_func import *


def collect_market_prices(
    markets: pd.DataFrame,
    out_path:Path,
    fidelity: int = 1440
) -> pd.DataFrame:
    if markets.empty:
        return pd.DataFrame()
    if out_path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
    header_written = bool(out_path and out_path.exists() and out_path.stat().st_size > 0)
    markets['id'] = pd.to_numeric(markets['id'], errors='coerce')
    markets = markets[markets['id'] > 606522] #TODO: delete
    #records: List[pd.DataFrame] = []
    for _, market in markets.iterrows():
        time.sleep(0.01)
        try:
            #index = get_market_winner_clobTokenId(market)
            #prices = fetch_market_prices_history(market['startDate'], index, fidelity=fidelity)

            prices_yes = fetch_market_prices_history(market['startDate'], market['clobTokenIdYes'], fidelity=fidelity)
            prices_yes['token'] = 'yes'
            prices_no = fetch_market_prices_history(market['startDate'], market['clobTokenIdNo'], fidelity=fidelity)
            prices_no['token'] = 'no'
        except Exception as exc:
            print(f"Failed to fetch prices for {market['id']}: {exc}", file=sys.stderr)
            continue
        prices = pd.concat([prices_yes, prices_no], ignore_index=True)
        if prices.empty:
            pd.DataFrame([[market["id"]]], columns=["id"]).to_csv(
                Path("data/no_price_markets.csv"),
                mode="a",
                header=False,
                index=False,
            )
            print(f"No price data for market {market['id']}")
            continue
        df = prices[["t", "p", "token"]].copy()
        df["market_id"] = str(market['id'])
        df = df[["market_id", "t", "p", "token"]]
        if out_path:
            df.to_csv(out_path, mode="a", header=not header_written, index=False)
            print(f"Appended\t{len(df)} rows for market {market['id']} to {out_path}")
            header_written = True

    #if not records:
    #    return pd.DataFrame(columns=["market_id", "t", "p"])

    #return pd.concat(records, ignore_index=True)[["market_id", "t", "p"]]


def save_prices_to_csv(prices: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    prices.to_csv(output_path, index=False)



def stream_markets_to_csv(
    limit: int|Literal['all'] = 'all',
    page_size: int = 500,
    offset: int = GAMMA_API_LAST_PIPELINE_OFFSET,
    out_path:Path = Path(f"data/binary_new_markets.csv")

) -> None:
    header_written = out_path.exists() and out_path.stat().st_size > 0
    saved = 0
    offset = offset
    while True:
        batch = fetch_markets(
            size=page_size,
            page=page_size,
            offset=offset,
            post_filters={"is_single_market_event": True}
        )

        if batch is None:
            print(f"API empty answer. Last offset = {offset}")
            break

        # if total is an int, cap the last batch
        if isinstance(limit, int):
            print('limit if')
            remaining = limit - saved
            if remaining <= 0:
                break
            if len(batch) > remaining:
                batch = batch.iloc[:remaining]

        # append
        if(len(batch)>0):
            batch.to_csv(
                out_path,
                mode="a",
                header=not header_written,
                index=False,
                quoting=csv.QUOTE_MINIMAL,
            )
            header_written = True

        saved += len(batch)
        offset += page_size                        
        print(f"Saved batch {len(batch)} (total saved={saved}), next offset={offset}")

        #if isinstance(limit, int) and saved >= limit:
        #    break

    print(f"Done. Wrote {saved} rows → {out_path}")

def stream_categorical_markets_to_csv(
    limit: int|Literal['all'] = 100,
    page_size: int = 500,
    offset: int = 0,
    out_path:Path = Path(f"data/losers_categorical.csv")

) -> None:
    header_written = out_path.exists() and out_path.stat().st_size > 0
    saved = 0
    offset = offset
    while True:
        batch = fetch_categorical_markets(
            size=page_size,
            page=page_size,
            offset=offset,
            shouldFetchWinners=False
        )

        if batch is None:
            print(f"API empty answer. Last offset = {offset}")
            break

        # if total is an int, cap the last batch
        if isinstance(limit, int):
            remaining = limit - saved
            if remaining <= 0:
                break
            if len(batch) > remaining:
                batch = batch.iloc[:remaining]

        # append
        if(len(batch)>0):
            batch.to_csv(
                out_path,
                mode="a",
                header=not header_written,
                index=False,
                quoting=csv.QUOTE_MINIMAL,
            )
            header_written = True

        saved += len(batch)
        offset += page_size                        
        print(f"Saved batch {len(batch)} (total saved={saved}), next offset={offset}")

        if isinstance(limit, int) and saved >= limit:
            break

    print(f"Done. Wrote {saved} rows → {out_path}")


def parse_limit(s: str) -> Union[int, Literal["all"]]:
    s = s.strip().lower()
    if s == "all":
        return "all"
    return int(s)

def parse_args(argv: Iterable[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect Gamma market prices and cache them to CSV.")
    parser.add_argument("--output", type=Path, default=Path(CSV_OUTPUT_PATH), help="CSV path for cached prices")
    parser.add_argument("--fetch-binary-markets", dest="fetch_binary_markets", action="store_true", help="Fetch markets to csv")
    parser.add_argument(
        "--limit",
        type=parse_limit,        
        default="all",             
        help='Number of markets to fetch (or "all")',
    )
    parser.add_argument("--page-size", dest="page_size", type=int, default=500, help="Page size for each markets fetch")
    parser.add_argument("--offset", type=int, default=GAMMA_API_LAST_PIPELINE_OFFSET, help="Initial offset for pagination")
    parser.add_argument("--fidelity", type=int, default=500, help="Fidelity passed to prices-history endpoint")
    parser.add_argument("--token-index", type=int, default=YES_INDEX, help="Outcome index to download prices for")
    parser.add_argument("--plot-market", help="Optional market id to plot after collection")
    parser.add_argument(
        "--fetch-prices",
        dest="fetch_prices",
        action="store_true",
        help="Fetch price history for the cached markets CSV and write to --output",
    )
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv if argv is not None else sys.argv[1:])
    #stream_categorical_markets_to_csv(limit=args.limit, page_size=args.page_size, offset=10000)
    #stream_markets_to_csv()
    
    if args.fetch_prices:
        print("Collecting prices for cached markets…")
        markets = pd.read_csv('data/binary_new_markets.csv')
        collect_market_prices(
            markets,
            fidelity=args.fidelity,
            out_path=args.output,
        )
        print(f"Finished writing prices to {args.output}")

    if(args.fetch_binary_markets):
        stream_markets_to_csv(limit=args.limit, page_size=args.page_size, offset=args.offset)
        return 0

   # if(args.fetch_categorical_markets):
   #     stream_categorical_markets_to_csv(limit=args.limit, page_size=args.page_size)
    #    return 0
    
    

    #markets = pd.read_csv('data/test_pipeline.csv')
    #prices = collect_market_prices(
    #    markets,
    #    fidelity=args.fidelity,
    #    out_path=args.output,
    #)



    return 0


if __name__ == "__main__":
    raise SystemExit(main())
